#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import os
import logging
import random
import sys

os.environ["WANDB_PROJECT"] = "custom-po"

import torch
import transformers
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_peft_config,
    get_tokenizer,
)
from custom_po_trainer import CustomPOTrainer
from custom_po_config import CustomPOConfig
from jinja2 import Template

from transformers import set_seed
set_seed(42)


logger = logging.getLogger(__name__)

SUMMARIZATION_TEMPLATE = "<|im_start|>user\nSUBREDDIT: r/{{ subreddit }}\nTITLE: {{ title }}\nPOST: {{ post }}\nTL;DR: <|im_end|> <|im_start|>assistant "
MT_TEMPLATE = "{{ prompt }}"


def apply_sum_template(example, algorithm):
    if algorithm in ["sft"]:
        template = Template(SUMMARIZATION_TEMPLATE)
        example["prompt"] = template.render(**example)
        example["chosen"] = example["summary"]
        example["rejected"] = ""

    elif algorithm in ["dpo", "cpo", "simpo", "dpo-gamma", "slic", "slic-dpo"]:
        template = Template(SUMMARIZATION_TEMPLATE)
        example["prompt"] = template.render(**example['info'])
        example["chosen"] = example["summaries"][example["choice"]]['text']
        example["rejected"] = example["summaries"][1 - example["choice"]]['text']

    return example

# same dataset for all algorithms
def apply_mt_template(example, algorithm):
    if algorithm in ["sft", "dpo", "cpo", "simpo", "dpo-gamma", "slic", "slic-dpo"]:
        example["prompt"] = example["prompt"]
        example["chosen"] = example["chosen"]
        example["rejected"] = example["rejected"]

    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, CustomPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["prompt", "chosen", "rejected"] if training_args.task == "mt"
                        else ["info", "summaries", "choice"] if training_args.task == "sum" 
                        else None,
        # seed=training_args.seed,
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args)

    #####################
    # Apply chat template
    #####################
    for split in data_args.dataset_splits:
        raw_datasets[split] = raw_datasets[split].map(
            apply_mt_template if training_args.task_name == "mt" else apply_sum_template if training_args.task_name == "sum" else None,
            fn_kwargs={
                "algorithm": training_args.algorithm
            },
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"] if training_args.task_name == "sum" and training_args.algorithm == "sft" \
        else raw_datasets["validation"].select(range(int(0.05 * len(raw_datasets["validation"]))))

    if training_args.shuffle:
        train_dataset = train_dataset.shuffle(seed=42)
        eval_dataset = eval_dataset.shuffle(seed=42)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{train_dataset[index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{train_dataset[index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{train_dataset[index]['rejected']}")

    if not training_args.reference_free:
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=model_args.load_in_8bit,
            use_cache=False if model_args.gradient_checkpointing else True,
        )
        model_ref.eval()
    else:
        model_ref = None


    #########################
    # Initialize training arguments:
    #########################
    training_args = TrainingArguments(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        learning_rate=training_args.learning_rate,
        evaluation_strategy="no" if training_args.no_eval else "steps",
        eval_steps=training_args.eval_steps,
        output_dir=training_args.output_dir,
        report_to=training_args.report_to,
        lr_scheduler_type=training_args.lr_scheduler_type,
        optim=training_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=training_args.run_name,
        warmup_ratio=training_args.warmup_ratio,
        save_strategy="steps",
        save_only_model=True,
        save_safetensors=True,
        adam_beta2=0.95,
    )

    model = training_args.model_name_or_path


    #########################
    # Instantiate DPO-gamma trainer
    #########################
    trainer = CustomPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=training_args.beta,
        gamma=training_args.gamma,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_type=training_args.contrast_loss_type, 
        max_prompt_length=training_args.max_prompt_length,
        max_length=training_args.max_length,
        generate_during_eval=training_args.generate_during_eval,
        average_log_prob=training_args.average_log_prob,
        reference_free=training_args.reference_free,
        lambda_sft=training_args.lambda_sft,
        lambda_contrast=training_args.lambda_contrast,
        sft_type=training_args.sft_type,
        margin=training_args.margin,
        peft_config=get_peft_config(training_args),
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["text-generation", "machine translation" if training_args.task == "mt" else "summarization"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
