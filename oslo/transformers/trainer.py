import os
import time
import numpy as np
import contextlib
import logging
import math
import sys
import random
import warnings
from packaging import version
from typing import Any, Dict, List, Optional, Union, Type, Mapping, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import find_labels
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    DistributedSamplerWithLoop,
    ShardSampler,
    get_parameter_names,
    find_batch_size,
    distributed_concat,
    nested_numpify,
    nested_truncate,
    nested_concat,
    nested_detach,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    speed_metrics,
    denumpify_detensorize,
    has_length,
    EvalLoopOutput,
    TrainOutput,
    EvalPrediction,
    PREFIX_CHECKPOINT_DIR,
)

import oslo
from oslo.torch import ParallelMode
from oslo.torch.utils.extensions import save_pretrained as save_pretrained_oslo
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
)
from oslo.torch.utils.checkpoint.activation_checkpointing import ActivationCheckpointing
from oslo.transformers.data.data_collator import (
    DataCollator,
    default_data_collator,
)
from oslo.transformers.trainer_utils import OptimizerNames, log_dist
from oslo.transformers.training_args import TrainingArguments

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

        if args.save_on_each_node:
            self.should_save = self.is_local_process_zero()
        else:
            self.should_save = self.is_world_process_zero()

        self.model = model

        # Define and add callback
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
            # self.scaler = ShardedGradScaler()
        # TODO Label Smoother

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()

        default_label_names = find_labels(self.model.__class__)
        self.label_names = (
            default_label_names
            if self.args.label_names is None
            else self.args.label_names
        )

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
            args.train_batch_size
            * args.gradient_accumulation_steps
            * self.parallel_context.get_world_size(ParallelMode.DATA)
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

        self.model = self._wrap_model(self.model_wrappers)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # # TODO Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        log_dist("***** Running training *****")
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
        start_time = time.time()
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
                tr_loss_step = self.training_step(self.model, inputs)

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
                    self._maybe_log_save_evaluate(tr_loss, self.model)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
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
            self._maybe_log_save_evaluate(tr_loss, self.model)

            if self.control.should_training_stop:
                break

        log_dist("\n\nTraining completed.\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if self.parallel_context.get_local_rank(ParallelMode.GLOBAL) != -1:
                dist.barrier()
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        metrics["train_loss"] = train_loss
        self.is_in_train = False

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

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
        inputs = self._prepare_inputs(inputs)
        # log_dist(f"After self._prepare_inputs: \n{inputs}", rank=-1)
        if (
            self.args.oslo_config is not None
            and self.args.oslo_config.pipeline_parallelism
        ):
            with self.autocast_smart_context_manager():
                loss = self.compute_pp_loss(model, inputs)
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

    def _load_best_model(self):
        log_dist(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )

        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        if os.path.exists(best_model_path):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(best_model_path, map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)
        else:
            log_dist(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(
                load_result.missing_keys
            ) == set(self.model._keys_to_ignore_on_save):
                self.model.tie_weights()
            else:
                log_dist(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.",
                    level=logging.WARNING,
                )
        if len(load_result.unexpected_keys) != 0:
            log_dist(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.",
                level=logging.WARNING,
            )

    def _maybe_log_save_evaluate(self, tr_loss, model):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def _save_checkpoint(self, model, metrics=None):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        # Consolidate optimizer states (Zero 1,2 )
        if (
            self.args.oslo_config.data_parallelism is not None
            and self.args.oslo_config.data_parallelism["zero_stage"] < 2
        ):
            self.optimizer.consolidate_state_dict()

        # Save optimizer state on "should_save machine"
        if self.should_save:
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(
                    self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME)
                )

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.parallel_context.get_local_rank(ParallelMode.GLOBAL) == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = self.parallel_context.get_local_rank(ParallelMode.GLOBAL)
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth")
            )

        # # Maybe delete some older checkpoints. TODO
        # if self.should_save:
        #     self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
        state_dict=None,
    ):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        log_dist(f"Saving model checkpoint to {output_dir}")
        log_dist(type(self.model))
        log_dist(self.args.model_wrappers)

        if len(self.args.model_wrappers) > 0:
            save_pretrained_oslo(self.model, output_dir, state_dict=state_dict)
        else:
            if not isinstance(self.model, PreTrainedModel):
                if isinstance(unwrap_model(self.model), PreTrainedModel):
                    if state_dict is None:
                        state_dict = self.model.state_dict()
                    unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict
                    )
                else:
                    log_dist(
                        "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                    )
                    if state_dict is None:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(save_directory=str(output_dir))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # TODO error => TypeError: cannot pickle 'torch._C._distributed_c10d.ProcessGroupNCCL' object
        # # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        # ignore_keys: Optional[List[str]] = None,
        # metric_key_prefix: str = 'eval',
    ) -> Dict[str, float]:

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            # prediction_loss_only=True if self.compute_metrics is None else None,
            # ignore_keys=ignore_keys,
            # metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = (
            self.args.eval_batch_size
            * self.parallel_context.get_world_size(ParallelMode.DATA)
        )

        output.metrics.update(
            speed_metrics(
                "eval",
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        # prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args

        # prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else False

        model = self._wrap_model(self.model_wrappers, training=False)

        batch_size = self.args.eval_batch_size
        log_dist(f"***** Running {description} *****", rank=-1)
        if has_length(dataloader):
            log_dist(f"  Num examples = {self.num_examples(dataloader)}", rank=-1)
        else:
            log_dist("  Num examples: Unknown", rank=-1)

        log_dist(f"  Batch size = {batch_size}", rank=-1)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # if args.past_index >= 0:
        #     self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            # Prediction step
            loss, logits, labels = self.prediction_step(
                model,
                inputs,
                # prediction_loss_only,
                ignore_keys=ignore_keys,
            )
            # inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            # if inputs_decode is not None:
            #     inputs_decode = self._pad_across_processes(inputs_decode)
            #     inputs_decode = self._nested_gather(inputs_decode)
            #     inputs_host = (
            #         inputs_decode if inputs_host is None else nested_concat(
            #             inputs_host, inputs_decode, padding_index=-100))
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )
            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        # if args.past_index and hasattr(self, "_past"):
        #     # Clean the state at the end of the evaluation loop
        #     delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(
            eval_dataset, "num_examples"
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # # Metrics! TODO
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     if args.include_inputs_for_metrics:
        #         metrics = self.compute_metrics(
        #             EvalPrediction(predictions=all_preds,
        #                            label_ids=all_labels,
        #                            inputs=all_inputs))
        #     else:
        #         metrics = self.compute_metrics(
        #             EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # else:
        #     metrics = {}

        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tenasor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        elif self.parallel_context.get_local_rank(ParallelMode.DATA) != -1:
            tensors = distributed_concat(tensors)
        return tensors

    # Copied from Transformers.Trainer
    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(
                self._pad_across_processes(t, pad_index=pad_index) for t in tensor
            )
        elif isinstance(tensor, dict):
            return type(tensor)(
                {
                    k: self._pad_across_processes(v, pad_index=pad_index)
                    for k, v in tensor.items()
                }
            )
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        if tensor.shape[1] == max_size:
            return tensor

        # Then pad to the maximum size
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        # prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                    )
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # # TODO: this needs to be fixed and made cleaner later.
                # if self.args.past_index >= 0:
                #     self._past = outputs[self.args.past_index - 1]

        # if prediction_loss_only:
        #     return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

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
                # elif wrapper == SequenceParallel:
                #     model = wrapper(
                #         model,
                #         parallel_context=self.parallel_context,
                #         **self.args.oslo_config.sequence_parallelism["params"],
                #     )
                # elif wrapper == DistributedDataParallel:
                #     # model = model.to()
                #     model = wrapper(
                #         model,
                #         parallel_context=self.parallel_context,
                #         **self.args.oslo_config.data_parallelism["params"],
                #     )
                # elif wrapper == DataParallel:
                #     self.create_optimizer()
                #     model, self.optimizer = wrapper(
                #         model,
                #         self.optimizer,
                #         parallel_context=self.parallel_context,
                #         zero_stage=self.args.oslo_config["data_parallelism"][
                #             "zero_stage"
                #         ],
                #     )
                log_dist(f"Model wrapping with {wrapper}")

            oslo.ready(model, self.parallel_context)
        return model

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.parallel_context.get_local_rank(ParallelMode.GLOBAL) == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        return self.parallel_context.get_global_rank() == 0

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

    def compute_pp_loss(self, model, inputs, return_outputs=False):
        """ """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        pp_loss = torch.tensor(0.0).to(self.args.device)
        num_micro_batches = (
            self.args.oslo_config.pipeline_parallelism["param"]["num_micro_batches"]
            if "num_micro_batches"
            in self.args.oslo_config.pipeline_parallelism["param"]
            else 1
        )
        outputs = []
        for idx, out in enumerate(model(**inputs)):
            outputs.append(out)
            if labels is not None:
                loss = self.label_smoother(out, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = out["loss"] if isinstance(out, dict) else out[0]
            loss = loss / num_micro_batches
            loss.backward()
            pp_loss += loss.detach().item()
        return (pp_loss, outputs) if return_outputs else pp_loss

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
            self.parallel_context.get_local_rank(ParallelMode.DATA) > 1
            and self.args.oslo_config.data_parallelism is not None
        ):
            if not self.args.dataloader_drop_last:
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.parallel_context.get_local_rank(
                        ParallelMode.DATA
                    ),
                    rank=self.parallel_context.get_local_rank(ParallelMode.DATA),
                    # seed=seed, TODO oslo seed
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.parallel_context.get_local_rank(
                        ParallelMode.DATA
                    ),
                    rank=self.parallel_context.get_local_rank(ParallelMode.DATA),
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
        if (
            self.args.dataloader_num_workers
            % self.parallel_context.get_world_size(ParallelMode.DATA)
            != 0
        ):
            raise ValueError("dataloader_num_workers should be dividable by world_size")
        num_workers = (
            self.args.dataloader_num_workers
            / self.parallel_context.get_world_size(ParallelMode.DATA)
        )

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.parallel_context.get_local_rank(ParallelMode.DATA) > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.parallel_context.get_local_rank(
                        ParallelMode.DATA
                    ),
                    process_index=self.parallel_context.get_local_rank(
                        ParallelMode.DATA
                    ),
                )
                log_dist(
                    f"Dataset: {train_dataset.__class__} with\nbatch_size:{self.args.train_batch_size}\n world_size:{self.parallel_context.get_local_rank(ParallelMode.DATA)}\n dataloader_drop_last: {self.args.dataloader_drop_last}"
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=num_workers,
            )
        train_sampler = self._get_train_sampler()
        log_dist(
            f"Sampler: {train_sampler.__class__} with\nbatch_size:{self.args.train_batch_size}\nworld_size:{self.parallel_context.get_local_rank(ParallelMode.DATA)}, dataloader_drop_last: {self.args.dataloader_drop_last}"
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=num_workers,
        )

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:

        if self.parallel_context.get_world_size(ParallelMode.DATA) <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.parallel_context.get_world_size(ParallelMode.DATA),
                process_index=self.parallel_context.get_local_rank(ParallelMode.DATA),
            )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # if isinstance(eval_dataset, datasets.Dataset):
        #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        num_workers = (
            self.args.dataloader_num_workers
            / self.parallel_context.get_world_size(ParallelMode.DATA)
        )
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.parallel_context.get_local_rank(ParallelMode.DATA) > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.parallel_context.get_local_rank(
                        ParallelMode.DATA
                    ),
                    process_index=self.parallel_context.get_local_rank(
                        ParallelMode.DATA
                    ),
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=num_workers,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
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
