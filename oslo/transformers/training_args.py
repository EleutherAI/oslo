import os
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union
from enum import Enum
import torch
from transformers.trainer_utils import SchedulerType, IntervalStrategy
from oslo.transformers.trainer_utils import OptimizerNames
from oslo.torch.distributed.parallel_mode import ParallelMode



@dataclass
class TrainingArguments:
    """
    output_dir (`str`):
        The output directory where the model predictions and checkpoints will be written.
    oslo_config_path_or_dict (`str` or `dict`):
        The value is either the location of oslo parallel json config file (e.g.,`ds_config.json`) or an already loaded json file as a `dict`"
    evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
        The evaluation strategy to adopt during training. Possible values are:
            - `"no"`: No evaluation is done during training.
            - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
            - `"epoch"`: Evaluation is done at the end of each epoch.
    per_device_train_batch_size (`int`, *optional*, defaults to 8):
        The batch size per GPU/TPU core/CPU for training.
    per_device_eval_batch_size (`int`, *optional*, defaults to 8):
        The batch size per GPU/TPU core/CPU for evaluation.
    gradient_accumulation_steps (`int`, *optional*, defaults to 1):
        Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

        <Tip warning={true}>

        When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
        evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

        </Tip>
    # TODO later
    # eval_accumulation_steps (`int`, *optional*):
    #     Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
    #     left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
    #     requires more memory).
    eval_delay (`float`, *optional*):
        Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
        evaluation_strategy.
    learning_rate (`float`, *optional*, defaults to 5e-5):
        The initial learning rate for [`AdamW`] optimizer.
    weight_decay (`float`, *optional*, defaults to 0):
        The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
        optimizer.
    max_grad_norm (`float`, *optional*, defaults to 1.0):
        Maximum gradient norm (for gradient clipping).
    num_train_epochs(`float`, *optional*, defaults to 3.0):
        Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
        the last epoch before stopping training).
    max_steps (`int`, *optional*, defaults to -1):
        If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
        In case of using a finite iterable dataset the training may stop before reaching the set number of steps
        when all data is exhausted
    lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
        The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
    log_level (`str`, *optional*, defaults to `passive`):
        Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug',
        'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the
    logging_first_step (`bool`, *optional*, defaults to `False`):
        Whether to log and evaluate the first `global_step` or not.
    logging_steps (`int`, *optional*, defaults to 500):
        Number of update steps between two logs if `logging_strategy="steps"`.
    #TODO SAVE 할 때 each node에서 save할지 합쳐서 할지
    seed (`int`, *optional*, defaults to 42):
        TODO seed np랑 python꺼는 해줘야하는지
    eval_steps (`int`, *optional*):
        Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same
        value as `logging_steps` if not set.
    dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
    past_index (`int`, *optional*, defaults to -1): # TODO for transformerxl/XLNet
        Some models like [TransformerXL](../model_doc/transformerxl) or [XLNet](../model_doc/xlnet) can make use of
        the past hidden states for their predictions. If this argument is set to a positive int, the `Trainer` will
        use the corresponding output (usually index 2) as the past state and feed it to the model at the next
        training step under the keyword argument `mems`.
    load_best_model_at_end (`bool`, *optional*, defaults to `False`):
        Whether or not to load the best model found during training at the end of training.

        <Tip>

        When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in the case
        it is "steps", `save_steps` must be a round multiple of `eval_steps`.

        </Tip>
    label_smoothing_factor (`float`, *optional*, defaults to 0.0):
        The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
        labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor +
        label_smoothing_factor/num_labels` respectively.
    optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
        The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor.
    gradient_checkpointing (`bool`, *optional*, defaults to `False`):
        If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    # TODO
    report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
        The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
        `"comet_ml"`, `"mlflow"`, `"tensorboard"` and `"wandb"`. Use `"all"` to report to all integrations
        installed, `"none"` for no integrations.
    """

    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    oslo_config_path_or_dict: Union[str, dict] = field(
        default=None,
        metadata={
            "help": "Enable oslo features and pass the path to json config file (e.g. ds_config.json) or an already loaded json file as a dict"
        },
    )
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy."
        },
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={
            "help": f"The scheduler type ({', '.join([e.value for e in SchedulerType])})to use "
        },
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    logging_first_step: bool = field(
        default=False, metadata={"help": "Log the first global_step"}
    )
    logging_steps: int = field(
        default=500, metadata={"help": "Log every X updates steps."}
    )
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps."}
    )

    # log_level: Optional[str] = field(
    #     default="passive",
    #     metadata={
    #         "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'.",
    #         "choices": trainer_log_levels.keys(),
    #     },
    # )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
        },
    )
    eval_steps: int = field(
        default=None, metadata={"help": "Run an evaluation every X steps."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )
    label_smoothing_factor: float = field(
        default=0.0,
        metadata={
            "help": "The label smoothing epsilon to apply (zero means no label smoothing)."
        },
    )
    optim: OptimizerNames = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    report_to: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )

    def __post_init__(self):
        # TODO set log level
        # TODO set log dir
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

        # TODO set eval strategy
        if isinstance(self.evaluation_strategy, IntervalStrategy):
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)

        if self.load_best_model_at_end:
            self.metric_for_best_model = "loss"

        self.oslo_config, self.parallel_context, self.model_wrappers = None, None, None

        if self.oslo_config_path_or_dict:
            from oslo.transformers.oslo_init import (
                OsloTrainerConfig,
                init_oslo_features,
            )

            # will be used later by the Trainer
            self.oslo_config = OsloTrainerConfig(self.oslo_config_path_or_dict)
            # logging.info(f"Oslo Config: {self.oslo_config}")
            self.parallel_context, self.model_wrappers = init_oslo_features(
                self.oslo_config
            )

    def __str__(self):
        self_as_dict = {
            k: f"<{k.upper()}>" if k.endswith("_token") else v
            for k, v in asdict(self).items()
        }

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def local_rank(self):
        """
        The local rank
        """
        if self.parallel_context:
            return self.parallel_context.get_local_rank(ParallelMode.DATA)
        else:
            return int(os.environ.get("LOCAL_RANK", -1))

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if self.parallel_context:
            return self.parallel_context.get_world_size(ParallelMode.DATA)
        elif self.local_rank != -1:
            return int(os.environ["WORLD_SIZE"])
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        if self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        if self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        train_batch_size = self.per_device_train_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        eval_batch_size = self.per_device_eval_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps > 0
            else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    # @cached_property
    @property
    def _setup_devices(self) -> torch.device:
        # Here, we'll use torch.distributed.
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", self.local_rank)
        # self._n_gpu = 1
        self._n_gpu = torch.cuda.device_count()
        # TODO ?
        # if device.type == "cuda":
        #     torch.cuda.set_device(device)
        return device

    @property
    def device(self) -> torch.device:
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


def get_batch_size(per_device_batch_size, n_gpu) -> int:
    """
    The actual batch size (may differ from `per_gpu_batch_size` in distributed training).
    """
    return per_device_batch_size * max(1, n_gpu)
