import os
from functools import partial
from logging import getLogger
from typing import Optional, Union, Callable

import torch
import torch.distributed as dist
from torch import nn

from transformers import PreTrainedModel

import oslo
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel.pipeline_parallel import (
    _PipelineParallel,
    PipelineParallel,
)
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.torch.nn.parallel.tensor_parallel.tensor_parallel import _TensorParallel
from oslo.torch.nn.parallel.utils import (
    allocate_params,
    get_parameter_dtype,
)


@torch.no_grad()
def save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    merge_checkpoints: bool = False,
    **kwargs,
):
    PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"

    if (
        self.parallel_context.get_world_size(ParallelMode.TENSOR) == 1
        and self.parallel_context.get_world_size(ParallelMode.PIPELINE) == 1
    ):
        if dist.get_rank() == 0:
            self.save_pretrained(
                save_directory=save_directory,
                save_config=save_config,
                state_dict=state_dict,
                save_function=save_function,
                **kwargs,
            )
        dist.barrier()
        return None

    if merge_checkpoints:
        model_to_save = self.__class__(self.config).eval()

        if state_dict is None:
            state_dict = self.state_dict()

        if hasattr(self, "oslo_wrappers"):
            for parallel_mode, wrapper in self.oslo_wrappers.items():
                if isinstance(wrapper, _TensorParallel):
                    model_to_save = TensorParallel(
                        model_to_save,
                        parallel_context=self.parallel_context,
                        memory_priority=wrapper.memory_priority,
                    )

                elif isinstance(wrapper, _PipelineParallel):
                    model_to_save = PipelineParallel(
                        model_to_save,
                        parallel_context=self.parallel_context,
                        num_micro_batches=wrapper.num_micro_batches,
                        memory_computation_balance=wrapper.memory_computation_balance,
                    )

        model_to_save.load_state_dict(state_dict)
        oslo.ready(model_to_save, parallel_context=self.parallel_context)

        if hasattr(model_to_save, "oslo_wrappers"):
            for parallel_mode, wrapper in model_to_save.oslo_wrappers.items():
                if hasattr(wrapper, "deparallelize"):
                    wrapper.deparallelize()

        if dist.get_rank() == 0:
            _save_pretrained_per_rank(
                self=model_to_save,
                save_directory=save_directory,
                save_config=save_config,
                save_function=save_function,
                deparallelized=True,
                **kwargs,
            )
            os.rename(
                os.path.join(save_directory, PARALLELIZED_WEIGHTS_NAME),
                os.path.join(save_directory, "pytorch_model.bin"),
            )
        dist.barrier()
        del model_to_save
        return None

    _save_pretrained_per_rank(
        self=self,
        save_directory=save_directory,
        save_config=save_config,
        state_dict=state_dict,
        save_function=save_function,
        **kwargs,
    )
    return None


# To avoid deadlock
@torch.no_grad()
def _save_pretrained_per_rank(
    self,
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    deparallelized: bool = False,
    **kwargs,
):
    logger = getLogger()
    PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"

    if os.path.isfile(save_directory):
        logger.error(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
        return

    os.makedirs(save_directory, exist_ok=True)

    # Only save the model itself if we are using distributed training
    model_to_save = self

    # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
    # we currently don't use this setting automatically, but may start to use with v5
    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.torch_dtype = str(dtype).split(".")[1]

    # Attach architecture to the config
    model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # Save the config
    if save_config:
        model_to_save.config.save_pretrained(save_directory)

    # Save the model
    if state_dict is None:
        state_dict = model_to_save.state_dict()

    # Handle the case where some state_dict keys shouldn't be saved
    if getattr(self, "_keys_to_ignore_on_save", None) is not None:
        state_dict = {
            k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save
        }

    # If we save using the predefined names, we can load using `from_pretrained`
    weights_name = PARALLELIZED_WEIGHTS_NAME
    weights_name = weights_name.replace(
        "tp_0", f"tp_{self.parallel_context.get_local_rank(ParallelMode.TENSOR)}"
    )
    weights_name = weights_name.replace(
        "pp_0", f"pp_{self.parallel_context.get_local_rank(ParallelMode.PIPELINE)}"
    )

    output_model_file = os.path.join(save_directory, weights_name)

    if self.parallel_context.get_world_size(ParallelMode.DATA) > 1:
        if self.parallel_context.get_local_rank(ParallelMode.DATA) == 0:
            save_function(state_dict, output_model_file)
    else:
        save_function(state_dict, output_model_file)

    if not deparallelized:
        dist.barrier()

    logger.info(f"Model weights saved in {output_model_file}")
    return None


def from_parallelized(self, path):
    """
    Example:
    >>> model = AnyModel()
    >>> model = TensorParallel(model, ...)
    >>> model.from_parallelized(path)
    """
    PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"
    parallelized_model_path = path

    file_names = {
        os.path.join(
            parallelized_model_path,
            PARALLELIZED_WEIGHTS_NAME.replace("tp_0", f"tp_{tp}").replace(
                "pp_0", f"pp_{pp}"
            ),
        )
        for tp in range(self.parallel_context.get_world_size(ParallelMode.TENSOR))
        for pp in range(self.parallel_context.get_world_size(ParallelMode.PIPELINE))
    }

    if os.path.isdir(parallelized_model_path):
        if all(os.path.isfile(file_name) for file_name in file_names):
            state_dict = torch.load(
                os.path.join(
                    parallelized_model_path,
                    PARALLELIZED_WEIGHTS_NAME.replace(
                        "tp_0",
                        f"tp_{self.parallel_context.get_local_rank(ParallelMode.TENSOR)}",
                    ).replace(
                        "pp_0",
                        f"pp_{self.parallel_context.get_local_rank(ParallelMode.PIPELINE)}",
                    ),
                )
            )

            if getattr(self, "_keys_to_ignore_on_save", None) is not None:
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k not in self._keys_to_ignore_on_save
                }

            self.load_state_dict(state_dict=state_dict, strict=False)

        else:
            raise FileNotFoundError(
                f"all the {file_names} are necessary. "
                f"but some of them do not exist. Please check your checkpoint files."
            )
    else:
        raise NotADirectoryError(
            f"directory named {parallelized_model_path} is not valid. "
        )


def restrict_embedding_resizing(model):
    def resize_token_embeddings(new_num_tokens: Optional[int] = None, **kwargs):
        raise RuntimeError(
            "you can't use ``model.resize_token_embeddings()`` if you initialized OSLO.\n"
            "please resize token embedding size before OSLO initialization."
        )

    setattr(
        model, "resize_token_embeddings", partial(resize_token_embeddings, self=model)
    )

    return model


def ready_torch(model: nn.Module, parallel_context: ParallelContext):
    allocate_params(model, parallel_context)
    restrict_embedding_resizing(model)

    setattr(model, "from_parallelized", partial(from_parallelized, self=model))
    setattr(model, "save_pretrained", partial(save_pretrained, self=model))
