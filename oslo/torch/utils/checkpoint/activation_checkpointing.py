import importlib

import torch
from torch import nn

from oslo import ParallelContext
from oslo.torch.distributed import ParallelMode
from oslo.torch.utils.checkpoint._checkpoint_function import CheckpointFunction
from oslo.torch.utils.checkpoint._checkpoint_partitioner import CheckpointPartitioner
from oslo.torch.utils.checkpoint._rng_state_tracker import CudaRNGStatesTracker

RNG_TRACKER = None
PARTITIONER = None


def checkpoint(function, *args):
    """
    Activation checkpoint function.

    Args:
        function: user function object
        *args: arguments of user function

    Returns:
        Tuple: output of user function
    """
    all_outputs = []
    assert RNG_TRACKER is not None and PARTITIONER is not None, (
        "Activation Checkpointing is not initialized. "
        "Please initialize it using `model = ActivationCheckpointing(model, ...)`"
    )
    options = {"rng_tracker": RNG_TRACKER, "partitioner": PARTITIONER}
    CheckpointFunction.apply(function, options, all_outputs, *args)
    return tuple(all_outputs)


def ActivationCheckpointing(
    module: nn.Module,
    parallel_context: ParallelContext,
    partitioned_checkpointing: bool = False,
    contiguous_checkpointing: bool = False,
    cpu_checkpointing: bool = False,
):
    """
    Activation Checkpointing Engine

    Args:
        module (nn.Module): pytorch module
        parallel_context (ParallelContext): parallel context object
        partitioned_checkpointing (bool): partition activations to tensor parallel regions
        contiguous_checkpointing (bool): do defragmentation activation areas
        cpu_checkpointing (bool): offload activations to CPU

    Returns:
        nn.Module: module which will checkpoint activations
    """
    global RNG_TRACKER, PARTITIONER

    if partitioned_checkpointing:
        assert parallel_context.get_world_size(ParallelMode.TENSOR) > 1, (
            "If the param `partitioned_checkpointing` is True, "
            "the size of tensor parallelism must be greater than 1."
        )

    if contiguous_checkpointing:
        assert partitioned_checkpointing is True, (
            "`contiguous_checkpointing` can be used if `partitioned_checkpointing` is True. "
            "Please set `partitioned_checkpointing` to True."
        )

    if RNG_TRACKER is None:
        RNG_TRACKER = CudaRNGStatesTracker()

    if PARTITIONER is None:
        PARTITIONER = CheckpointPartitioner(
            process_group=parallel_context.get_group(ParallelMode.TENSOR),
            num_layers=module.config.num_hidden_layers,
            partitioned_checkpointing=partitioned_checkpointing,
            contiguous_checkpointing=contiguous_checkpointing,
            cpu_checkpointing=cpu_checkpointing,
        )

    imported = importlib.import_module(module.__module__)
    for name, attr in imported.__dict__.items():
        if attr is torch:
            getattr(imported, name).utils.checkpoint.checkpoint = checkpoint
        elif attr is torch.utils:
            getattr(imported, name).checkpoint.checkpoint = checkpoint
        elif attr is torch.utils.checkpoint:
            getattr(imported, name).checkpoint = checkpoint
        elif attr is torch.utils.checkpoint.checkpoint:
            setattr(imported, name, checkpoint)

    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()

    return module.train()
