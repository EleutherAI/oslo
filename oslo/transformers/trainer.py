import math
import os
import random
import re
import shutil
import inspect
from packaging import version
import sys
import contextlib
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    Mapping,
)
import logging
from tqdm.auto import tqdm
import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PretrainedConfig,
    __version__,
)
from transformers.trainer_pt_utils import (
    LabelSmoother,
    IterableDatasetShard,
    ShardSampler,
    LengthGroupedSampler,
    distributed_broadcast_scalars,
    nested_numpify,
    find_batch_size,
    nested_concat,
    nested_truncate,
    nested_detach,
    distributed_concat,
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    get_parameter_names,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import oslo
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
)
from oslo.torch.nn.parallel.sequence_parallel import SequenceParallel
from oslo.torch.nn.parallel.data_parallel.data_parallel import DataParallel
from oslo.torch.nn.parallel.data_parallel._ddp.distributed_data_parallel import (
    DistributedDataParallel,
)
from oslo.transformers.data.data_collator import (
    DataCollator,
    default_data_collator,
)
from oslo.torch.utils.checkpoint.activation_checkpointing import ActivationCheckpointing
from oslo.torch.nn.parallel.data_parallel._fsdp.sharded_grad_scaler import (
    ShardedGradScaler,
)
from oslo.transformers.training_args import TrainingArguments
from oslo.transformers.trainer_utils import OptimizerNames, log_dist

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.yaml"

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):

        if args is None:
            # No Arguments passed
            output_dir = "tmp_trainer"
            log_dist(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)

        self.args = args

        default_collator = (
            default_data_collator
            if tokenizer is None
            else DataCollatorWithPadding(tokenizer)
        )
        self.data_collator = (
            data_collator if data_collator is not None else default_collator
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.optimizer = None
        self.lr_scheduler = None
        self.parallel_context = None
        self.model_wrappers = []

        self.label_smoother = None  # TODO

        if args.oslo_config:
            self.parallel_context, self.model_wrappers = (
                args.parallel_context,
                args.model_wrappers,
            )

        if (
            len(self.model_wrappers)
            > 0
            # or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
        ):
            self.place_model_on_device = False
        else:
            self.place_model_on_device = True

        if self.place_model_on_device:
            # log_dist(f"model device, args.device: {self.args.device}", rank=-1)
            kwargs = dict(device=self.args.device)
            model = model.to(**kwargs)

        self.model = model

        # Define and add callback
        # default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        default_callbacks = DEFAULT_CALLBACKS
        callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.callback_handler.add_callback(DEFAULT_PROGRESS_CALLBACK)

        if (
            train_dataset is not None
            and not hasattr(train_dataset, "__len__")
            and args.max_steps <= 0
        ):
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified"
            )

        self.do_grad_scaling = False
        if args.fp16 or args.bf16:
            self.do_grad_scaling = True
            self.scaler = ShardedGradScaler()
        # TODO Label Smoother

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ):
        resume_from_checkpoint = (
            None if not resume_from_checkpoint else resume_from_checkpoint
        )

        args = self.args
        # TODO Load potential model checkpoint
        # if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        #     resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        #     if resume_from_checkpoint is None:
        #         raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
        #
        # if resume_from_checkpoint is not None:
        #     self._load_from_checkpoint(resume_from_checkpoint)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )
        if len(train_dataloader) is not None:
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        self.state = TrainerState()
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model = ActivationCheckpointing(
                self.model,
                self.parallel_context,
                **self.args.oslo_config["activation_checkpointing"],
            )

        model = self._wrap_model(self.model_wrappers)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # # TODO Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        log_dist(f"  Num examples = {num_examples}")
        log_dist(f"  Num Epochs = {num_train_epochs}")
        log_dist(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        log_dist(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        log_dist(f"  Total number of train samples = {num_train_samples}")
        log_dist(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        log_dist(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        # start_time = time.time()
        epochs_trained = 0
        # steps_trained_in_current_epoch = 0
        # steps_trained_progress_bar = None

        # # TODO Check if continuing training from a checkpoint
        # if resume_from_checkpoint is not None and os.path.isfile(
        #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        # ):
        #     self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        #     epochs_trained = self.state.global_step // num_update_steps_per_epoch
        #     if not args.ignore_data_skip:
        #         steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
        #         steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        #     else:
        #         steps_trained_in_current_epoch = 0
        #
        #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        #     logger.info(f"  Continuing training from epoch {epochs_trained}")
        #     logger.info(f"  Continuing training from global step {self.state.global_step}")
        #     if not args.ignore_data_skip:
        #         logger.info(
        #             f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
        #             "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
        #             "flag to your launch command, but you will resume the training on data already seen by your model."
        #         )
        #         if self.is_local_process_zero() and not args.disable_tqdm:
        #             steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
        #             steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # log_dist(f"args.device: {args.device}", rank=-1)
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated every time .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.optimizer.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(
                train_dataloader.dataset, IterableDatasetShard
            ):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader
            # # TODO Reset the past mems state at the beginning of each epoch if necessary.
            # if args.past_index >= 0:
            #     self._past = None
            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # # TODO Skip past any already trained steps if resuming training
                # if steps_trained_in_current_epoch > 0:
                #     steps_trained_in_current_epoch -= 1
                #     if steps_trained_progress_bar is not None:
                #         steps_trained_progress_bar.update(1)
                #     if steps_trained_in_current_epoch == 0:
                #         self._load_rng_state(resume_from_checkpoint)
                #     continue
                # elif steps_trained_progress_bar is not None:
                #     steps_trained_progress_bar.close()
                #     steps_trained_progress_bar = None
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # TODO _no_sync_in_gradient_accumulation
                # if (
                #     ((step + 1) % args.gradient_accumulation_steps != 0)
                #     and args.local_rank != -1
                #     and args._no_sync_in_gradient_accumulation
                # ):
                #     # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                #     with model.no_sync():
                #         tr_loss_step = self.training_step(model, inputs)
                # else:
                tr_loss_step = self.training_step(model, inputs)

                if torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # TODO Gradient Clipping
                    # Optimizer step
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )
                if step < 0:
                    log_dist(
                        f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples.",
                        logging.WARNING,
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(
                    args, self.state, self.control
                )

                # TODO Continue ....

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # log_dist(f"Before self._prepare_inputs: \n{inputs}", rank=-1)
        inputs = self._prepare_inputs(inputs)  # TODO Check
        # log_dist(f"After self._prepare_inputs: \n{inputs}", rank=-1)
        if self.args.oslo_config.pipeline_parallelism:
            pp_loss = torch.tensor(0.0).to(self.args.device)
            num_micro_batches = (
                self.args.oslo_config.pipeline_parallelism["param"]["num_micro_batches"]
                if "num_micro_batches"
                in self.args.oslo_config.pipeline_parallelism["param"]
                else 1
            )
            for idx, out in enumerate(model(**inputs)):
                loss = out.loss
                loss = loss / num_micro_batches
                loss.backward()
                pp_loss += loss.detach().item()
            return pp_loss
        else:
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            return loss.detach()

    def _wrap_model(self, model_wrappers: List, training: bool = True):
        """
        Apply parallelism to the model desired by the user to the model based on the oslo_init setting
        """
        if not training:
            return self.model

        model = self.model
        # Distributed training (should be after apex fp16 initialization)
        if self.parallel_context is not None:
            for wrapper in model_wrappers:
                log_dist(f"Model wrapping with wrapper: {wrapper}")

                if wrapper == TensorParallel:
                    log_dist(self.args.oslo_config)
                    model = wrapper(
                        model,
                        parallel_context=self.parallel_context,
                        **self.args.oslo_config.tensor_parallelism["params"],
                    )
                elif wrapper == PipelineParallel:
                    model = wrapper(
                        model,
                        parallel_context=self.parallel_context,
                        **self.args.oslo_config.pipeline_parallelism["params"],
                    )
                elif wrapper == SequenceParallel:
                    model = wrapper(
                        model,
                        parallel_context=self.parallel_context,
                        **self.args.oslo_config.sequence_parallelism["params"],
                    )
                elif wrapper == DistributedDataParallel:
                    # model = model.to()
                    model = wrapper(
                        model,
                        parallel_context=self.parallel_context,
                        **self.args.oslo_config.data_parallelism["params"],
                    )
                elif wrapper == DataParallel:
                    self.create_optimizer()
                    model, self.optimizer = wrapper(
                        model,
                        self.optimizer,
                        parallel_context=self.parallel_context,
                        zero_stage=self.args.oslo_config["data_parallelism"][
                            "zero_stage"
                        ],
                    )
                log_dist(f"Model wrapping with {wrapper}")

            oslo.ready(model, self.parallel_context)
        return model

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        return self.args.process_index == 0

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            return len(dataloader.dataset)
        except (
            NameError,
            AttributeError,
            TypeError,
        ):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        # This if-case is for Oslo DP wrapper which pre-define optimizer before wrapping the model.
        if self.optimizer is None:
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        log_dist(f"Optimizer: {self.optimizer}")
        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments,
    ) -> (Type[torch.optim.Optimizer], Dict):
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """
        optimizer_kwargs = {"lr": args.learning_rate}

        if args.optim == OptimizerNames.ADAFACTOR:
            from transformers.optimization import Adafactor

            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW:
            from torch.optim import AdamW

            optimizer_cls = AdamW
        elif args.optim == OptimizerNames.ADAM:
            if args.oslo_config and args.oslo_config.cpu_offload:
                from oslo.torch.optim import CPUAdam

                optimizer_cls = CPUAdam
            else:
                from oslo.torch.optim import FusedAdam

                optimizer_cls = FusedAdam
        elif args.optim == OptimizerNames.ADAGRAD:
            if args.oslo_config and args.oslo_config.cpu_offload:
                from oslo.torch.optim import CPUAdagrad

                optimizer_cls = CPUAdagrad
            else:
                from oslo.torch.optim import FusedAdagrad

                optimizer_cls = FusedAdagrad
        elif args.optim == OptimizerNames.ADADELTA:
            from torch.optim import Adadelta

            optimizer_cls = Adadelta
        elif args.optim == OptimizerNames.ADAMW_BNB:
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer_cls = Adam8bit
            except ImportError:
                raise ValueError(
                    "Trainer tried to instantiate bnb Adam8bit but bnb is not installed!"
                )
        elif args.optim == OptimizerNames.SGD:
            from oslo.torch.optim import FusedSGD

            optimizer_cls = FusedSGD
        elif args.optim == OptimizerNames.NOVOGRAD:
            from oslo.torch.optim import FusedNovoGrad

            optimizer_cls = FusedNovoGrad
        elif args.optim == OptimizerNames.LAMB:
            from oslo.torch.optim import FusedLamb

            optimizer_cls = FusedLamb
        else:
            raise ValueError(
                f"Trainer cannot instantiate unsupported optimizer: {args.optim}. Support optimizers: {', '.join([e.value for e in OptimizerNames])}"
            )
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.parallel_context or self.lr_scheduler is None:
            from transformers import get_scheduler

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # log_dist(f"**inputs: {inputs}", rank=-1)

        outputs = model(**inputs)
        # # TODO: Save past state if it exists
        # # HF-TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            return data.to(**kwargs)
        return data

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                "training dataset contains keys expected by the model."
            )
        # TODO mems
        # if self.args.past_index >= 0 and self._past is not None:
        #     inputs["mems"] = self._past

        return inputs

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # generator = None
        # TODO
        # if self.args.world_size <= 1 and _is_torch_generator_available:
        #     generator = torch.Generator()
        #     # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
        #     # `args.seed`) if data_seed isn't provided.
        #     # Further on in this method, we default to `args.seed` instead.
        #     if self.args.data_seed is None:
        #         seed = int(torch.empty((), dtype=torch.int64).random_().item())
        #     else:
        #         seed = self.args.data_seed
        #     generator.manual_seed(seed)
        #
        # seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if (
            self.args.world_size > 1
            and self.args.oslo_config.data_parallelism is not None
        ):
            if not self.args.dataloader_drop_last:
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    # seed=seed, TODO oslo seed
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    # seed=seed, TODO oslo seed
                )
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # TODO later
        # if isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        log_dist(f"Collate_fn: {self.data_collator.__class__}")
        if self.args.dataloader_num_workers % self.args.world_size != 0:
            raise ValueError("dataloader_num_workers should be dividable by world_size")
        num_workers = self.args.dataloader_num_workers / self.args.world_size

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
                log_dist(
                    f"Dataset: {train_dataset.__class__} with\nbatch_size:{self.args.train_batch_size}\n world_size:{self.args.world_size}\n dataloader_drop_last: {self.args.dataloader_drop_last}"
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=num_workers,
            )
        train_sampler = self._get_train_sampler()
        log_dist(
            f"Sampler: {train_sampler.__class__} with\nbatch_size:{self.args.train_batch_size}\nworld_size:{self.args.world_size}, dataloader_drop_last: {self.args.dataloader_drop_last}"
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=num_workers,
        )

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        ctx_manager = (
            contextlib.nullcontext()
            if sys.version_info >= (3, 7)
            else contextlib.suppress()
        )
        return ctx_manager
