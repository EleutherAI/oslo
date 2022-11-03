from collections import OrderedDict
from typing import Tuple, List

import torch
import torch.nn as nn
from oslo.torch.distributed import ParallelContext
from oslo.transformers.modeling_utils import OsloModel


def is_oslo_model(model: nn.Module):
    if isinstance(model, OsloModel):
        return True

    for module in model.modules():
        if isinstance(module, OsloModel):
            return True
    return False


def get_parameter_dtype(parameter: nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def _update_module_arguments(module: nn.Module, **kwargs):
    for k, v in kwargs.items():
        setattr(module, k, v)


def _remove_module_arguments(module: nn.Module, args: list):
    for k in args:
        delattr(module, k)


def allocate_params(model: nn.Module, parallel_context: ParallelContext):
    for parameter in model.parameters():
        if hasattr(parameter, "oslo_parallel"):
            # sorting parallel groups to fix parallelization order
            parameter.oslo_parallel = OrderedDict(
                sorted(parameter.oslo_parallel.items(), key=lambda item: str(item[0]))
            )
            device = parallel_context.ranks2device(parameter.oslo_parallel)
            if device is not None:
                parameter.data = parameter.to(
                    f"cuda:{device % parallel_context.local_world_size}"
                )
        else:
            parameter.data = parameter.to(torch.cuda.current_device())

    for buffer in model.buffers():
        if hasattr(buffer, "oslo_parallel"):
            # sorting parallel groups to fix parallelization order
            buffer.oslo_parallel = OrderedDict(
                sorted(buffer.oslo_parallel.items(), key=lambda item: str(item[0]))
            )
            device = parallel_context.ranks2device(buffer.oslo_parallel)
            if device is not None:
                buffer.data = buffer.to(
                    f"cuda:{device % parallel_context.local_world_size}"
                )
        else:
            buffer.data = buffer.to(torch.cuda.current_device())


def get_parallel_context(module: nn.Module, parallel_context: ParallelContext):
    if parallel_context is None:
        if hasattr(module, "parallel_context"):
            parallel_context = module.parallel_context
        else:
            raise ValueError(
                "Please input parallel context. \n"
                "There are two way to input parallel context: \n"
                "1. model.from_pretrained('model_name', parallel_context=parallel_context) \n"
                "2. model = XXXParallel(model, parallel_context=parallel_context)"
            )

    return parallel_context


def zero_rank_log(txt):
    import torch.distributed as dist

    if dist.get_rank() == 0:
        print(txt)

    dist.barrier()


def add_wrapper(module, mode, wrapper, parallel_context):
    if hasattr(module, "oslo_wrappers"):
        module.oslo_wrappers[mode] = wrapper
    else:
        setattr(module, "oslo_wrappers", {mode: wrapper})

    if not hasattr(module, "parallel_context"):
        setattr(module, "parallel_context", parallel_context)
