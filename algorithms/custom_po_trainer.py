# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
import inspect
import random
import warnings
import evaluate
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from transformers.utils import is_peft_available
from transformers import AutoModelForCausalLM, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
import datasets

from algorithms.custom_po_config import CustomPOConfig

if is_peft_available():
    from peft import (
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

EPS=1e-8

def cross_entropy_loss(logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False

class CustomPOTrainer(Trainer):
    r"""
    Initialize CustomPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss or `"hinge"` loss from SLiC paper.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        gen_dataset (`datasets.Dataset`):
            The dataset to use for generation
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        reference_free (`bool`, defaults to `False`):
            Whether to use a ref model in comparative loss.
        average_log_prob (`bool`, defaults to `False`):
            Whether to average log probability.
        lambda_sft (`float`, defaults to `0.0`):
            Weight on SFT loss.
        lambda_contrast (`float`, defaults to `1.0`):
            Weight on contrastive loss.
        sft_type (`str`, defaults to `token`):
            Type of normalization on SFT
        use_sft_weight (`bool`, defaults to `False`):
            Whether to use a reward weighted SFT loss -> RAFT/REST
        use_ccpo_loss (`bool`, defaults to `False`):
            Whether to use conservative cpo loss
        gamma (`float`, defaults to `0.5`):
            A target reward margin for the SimPO and DPO-gamma losses.
        label_smoothing (`float`, defaults to 0):
            The label smoothing factor. This argument is required if you want to use the default data collator.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "dpo-gamma", "simpo"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        gen_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        num_shared_layers: Optional[int] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        reference_free: bool = False,
        average_log_prob: bool = False,
        lambda_sft: float = 0.0,
        lambda_contrast: float = 1.0,
        sft_type: str = "token",
        use_sft_weight: bool = False,
        gamma: int = 0.5,
        label_smoothing: float = 0,
        use_reward_logit:bool = False,
        alpha_ropo: float = 14,
        group_by_scores: bool = False,
        scores_column_name: str = None,
        margin: float = 0.0,
        phi_star: float = -0.8,
    ):
        if model is None:
            raise ValueError("No model provided. Please provide a model to train.")

        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_kwargs to the CustomPOTrainer. But your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )
        
        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_kwargs to the CustomPOTrainer. But your model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the CustomPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False
         
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model = get_peft_model(model, peft_config)
        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        if not reference_free:
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model:
                # The `model` with adapters turned off will be used as the reference model
                self.ref_model = None
            else:
                self.ref_model = create_reference_model(model, num_shared_layers=num_shared_layers)
        else:
            self.ref_model = None

        if tokenizer is None:
            raise ValueError(
                "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
            )
        if max_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        self.max_prompt_length = max_prompt_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if not reference_free or self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.reference_free = reference_free
        self.generate_during_eval = generate_during_eval
        self.gen_dataset = gen_dataset
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.group_by_scores = group_by_scores
        self.scores_column_name = scores_column_name

        # introduced parameters for our setting
        self.sft_type = sft_type
        self.lambda_contrast = lambda_contrast
        self.lambda_sft = lambda_sft
        self.average_log_prob = average_log_prob
        self.use_sft_weight = use_sft_weight
        self.use_reward_logit = use_reward_logit
        self.alpha_ropo = alpha_ropo
        self.margin = margin

        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.phi_star = phi_star
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if self.loss_type == "simpo" or self.loss_type == "dpo-gamma":
            self.gamma = gamma
            # if self.lambda_contrast > 0:
            #     warnings.warn(
            #         "You are using CPO-SimPO method because you set a non-zero cpo_alpha. "
            #         "This will result in the CPO-SimPO method "
            #         "(https://github.com/fe1ixxu/CPO_SIMPO/tree/main). "
            #         "If you want to use a pure SimPO method, please set cpo_alpha to 0."
            #     )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        
        if not reference_free:
            if self.ref_model is None:
                if not hasattr(self.accelerator.unwrap_model(self.model), "disable_adapter"):
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                    )
            else:
                if self.is_deepspeed_enabled:
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
                else:
                    self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.group_by_scores:
            if isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.scores_column_name]
                    if self.scores_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(self.train_dataset)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    
    def _generate_batched(
        self,
        input_tensors: List[torch.Tensor],
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []
        # in case we have fewer examples than bs
        batch_size = min(len(input_tensors), batch_size)
        with torch.no_grad():
            for i in tqdm(range(0, len(input_tensors), batch_size)):
                # prevent overflow if query tensors are not even multiple of bs
                end_index = min(len(input_tensors), i + batch_size)
                batch = input_tensors[i:end_index]
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {"input_ids": batch, "attention_mask": batch_mask}
                padded_inputs = self.tokenizer.pad(
                    inputs,
                    padding="max_length",
                    max_length=self.max_prompt_length,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_tensors="pt",
                ).to(self.accelerator.device)
                
                generations = self.model.generate(**padded_inputs, **generation_kwargs)
                for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                    if not self.is_encoder_decoder:
                        output = generation[(1 - mask).sum() :]  # remove padding
                    else:
                        output = generation
                    if not return_prompt and not self.is_encoder_decoder:
                        output = output[(mask).sum() :]  # remove prompt
                    # this assumes that eos is the pad token - Sweta ?
                    if remove_padding and self.tokenizer.eos_token_id in output:
                        pad_mask = output == self.tokenizer.eos_token_id
                        pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                        output = output[: pad_start + 1]  # keep the eos token at the end
                    outputs.append(output)
        return outputs

    def prepare_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            if "best_response_input_ids" in batch:
                max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1], batch["best_response_input_ids"].shape[1])
            else:
                max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        for k in batch:
            if k.startswith("best_response") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("best_response", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k].to(self.accelerator.device), max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reward_chosen: torch.FloatTensor,
        reward_rejected: torch.FloatTensor,
        entropy: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors 
            contain the rewards for the chosen and rejected responses, respectively.
        """
        
        if self.loss_type == "mpo":
            if self.use_reward_logit:
                with torch.no_grad():
                    reward_prob = F.sigmoid(
                        torch.logit(torch.clamp(reward_chosen, min=EPS, max=1-EPS)) - 
                        torch.logit(torch.clamp(reward_rejected, min=EPS, max=1-EPS))
                        )
            else:
                reward_prob = F.sigmoid(reward_chosen - reward_rejected).to(self.accelerator.device)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        
        # hinge - ref_free = hing_norm
        if self.reference_free:
            reference_chosen_logps = 0
            reference_rejected_logps = 0

        ref_logratios = reference_chosen_logps - reference_rejected_logps
            
        logits = pi_logratios - ref_logratios - self.margin

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "simpo" or self.loss_type == "dpo-gamma":
            gamma_logratios = self.gamma / self.beta
            logits = logits - gamma_logratios
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "mallows-dpo":
            # https://arxiv.org/pdf/2405.14953 Eq 12
            losses = -F.logsigmoid(2 * self.phi_star * torch.log(entropy / torch.log(torch.tensor(self.tokenizer.vocab_size).to(self.accelerator.device))) * self.beta * logits)

        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )

        elif self.loss_type == "orpo":
            log_odds = (policy_chosen_logps - policy_rejected_logps) - (
                torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
            )
            losses = - F.logsigmoid(log_odds)

        elif self.loss_type == "fpo":
            losses = - (1-F.sigmoid(self.beta * logits))**self.gamma * F.logsigmoid(self.beta * logits)

        elif self.loss_type == "fcpo":
            losses = - (1-F.sigmoid(self.beta * logits))**self.gamma * F.logsigmoid(self.beta * pi_logratios)

        elif self.loss_type == "mpo":
            # use clamped logits
            losses = - reward_prob * F.logsigmoid(self.beta * logits) - (1-reward_prob) * F.logsigmoid(-self.beta * logits)

        elif self.loss_type == "ropo":
            # Eq 9 from https://arxiv.org/pdf/2404.04102.pdf
            losses = - (4 * self.alpha_ropo / (1 + self.alpha_ropo)**2 ) * F.logsigmoid(self.beta * logits) + (4 * self.alpha_ropo**2 / (1 + self.alpha_ropo)**2 ) * F.sigmoid(-self.beta * logits)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto', 'npo']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if self.average_log_prob:
            # normalized at the token level
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    
    def _get_batch_conditional_entropy(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id=-100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_ps = logits.softmax(-1)

        entropy = -torch.sum(per_token_ps * torch.log(per_token_ps + 1e-10), dim=-1)
        return (entropy * mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.prepare_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        all_entropy = self._get_batch_conditional_entropy(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:len_chosen*2]

        entropy = (all_entropy[:len_chosen] + all_entropy[len_chosen:len_chosen*2]) / 2


        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:len_chosen*2]
        
        best_logps = all_logps[2*len_chosen:]
        labels = concatenated_batch["concatenated_labels"]

        # batch normalized SFT loss on best response
        nll_loss = cross_entropy_loss(all_logits[2*len_chosen:], labels[2*len_chosen:])

        return (chosen_logps, rejected_logps, best_logps, chosen_logits, rejected_logits, nll_loss, entropy)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_best_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            nll_loss,
            entropy,
        ) = self.concatenated_forward(model, batch)

        if not self.reference_free:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)
        else:
            reference_chosen_logps, reference_rejected_logps = None, None

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            torch.tensor(batch['reward_chosen']).to(self.accelerator.device) if self.loss_type == "mpo" else None,
            torch.tensor(batch['reward_rejected']).to(self.accelerator.device) if self.loss_type == "mpo" else None,
            entropy if self.loss_type == "mallows-dpo" else None,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""

        if self.sft_type == "instance":
            if self.use_sft_weight and 'sft_weight' in batch:
                nll_loss = - torch.tensor(batch['sft_weight']).to(self.accelerator.device) * policy_best_logps
            else:
                nll_loss = - policy_best_logps

        losses_final =  torch.tensor(self.lambda_contrast) * losses.mean() +  torch.tensor(self.lambda_sft) * nll_loss.mean()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        metrics[f"{prefix}nll_loss"]  = nll_loss.detach().cpu().mean()
        metrics[f"{prefix}logps/best"] = policy_best_logps.detach().cpu().mean()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses_final, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if not self.reference_free:
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reference_output = self.model.generate(
                        batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                reference_output = self.ref_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = ""

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            metric = evaluate.load("bleu")

            # # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            # results_policy = metric.compute(predictions=policy_output_decoded, references=dataloader.dataset["best_response"])['bleu']
            # results_reference = metric.compute(predictions=ref_output_decoded, references=dataloader.dataset["best_response"])['bleu']

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol, ref]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    ),
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
