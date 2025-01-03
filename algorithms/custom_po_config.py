from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class CustomPOConfig(TrainingArguments):

    # task parameters
    task_name: Optional[str] = field(default="mt", metadata={"help": "the task to run"}) # mt, summarization, etc.
    algorithm: Optional[str] = field(default="dpo", metadata={"help": "the algorithm to run"}) # dpo, sft, cpo, etc.

    # data parameters
    margin: Optional[float] = field(default=0.0, metadata={"help": "the margin for SimPO loss"}) # not used, use gamma instead
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    gamma: Optional[float] = field(default=2, metadata={"help": "the gamma parameter for focal loss"})
    phi_star: Optional[float] = field(default=2, metadata={"help": "the phi_star parameter for mallows-dpo loss"})
    alpha_ropo: Optional[float] = field(default=2, metadata={"help": "the alpha parameter for ropo loss"})
    
    # datasets
    dataset_name: Optional[str] = field(
        default="sardinelab/MT-pref",
        metadata={"help": "name of the dataset"},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of the dataset"},
    )
    eval_lps: Optional[str] = field(
        default="en-de,zh-en",
        metadata={"help": "language pairs used for evaluation separated by comma"},
    )
    shuffle: Optional[bool] = field(
        default=False, 
        metadata={"help": "shuffle training data"}
    )
    no_eval: Optional[bool] = field(
        default=False, 
        metadata={"help": "dont perform evaluation"}
    )
    preference_method: Optional[str] = field(default="b-w", metadata={"help": "preference method: b/w; all; strict"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="haoranxu/ALMA-7B",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-07, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "warmup ratio"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load model in 8 bit"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "use flash attention"})
    low_cpu_mem_usage: Optional[bool] = field(default=False, metadata={"help": "use low cpu memory"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "use peft configuration"})
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    lora_modules: Optional[str] = field(default="q_proj v_proj k_proj out_proj fc_in fc_out wte", metadata={"help": "the lora target modules"})

    # specific parameters
    contrast_loss_type: Optional[str] = field(
        default="sigmoid", metadata={"help": "the loss type - sigmoid/hinge/ipo/simpo"}
    )
    sft_type: Optional[str] = field(
        default="token", metadata={"help": "Whether to normalize sft at the batch or instance level"}
    )
    lambda_contrast: Optional[float] = field(
        default=1.0, metadata={"help": "weight on dpo loss"}
    )
    lambda_sft: Optional[float] = field(
        default=0.0, metadata={"help": "weight on supervised loss"}
    )
    reference_free: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use reference normalization"}
    )
    average_log_prob: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate during evaluation"}
    )
    generate_during_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate during evaluation"}
    )
    label_smoothing: Optional[float] = field(default=0.0, metadata={"help": "label smoothing"})

    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "max number of training steps"}) # 4 epochs
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})


    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    run_name: Optional[str] = field(default="dpo", metadata={"help": "name of the run"})
    
