from typing import List, Optional, Callable

from oslo.torch.nn.parallel.data_parallel._ddp.distributed_data_parallel import (
    DistributedDataParallel,
)
from oslo.torch.nn.parallel.data_parallel._fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
from oslo import ParallelContext


def DataParallel(
    module,
    optimizer,
    parallel_context: ParallelContext,
    zero_stage: int = 0,
    transformer_wrap_layers: Optional[List] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    cpu_offload: bool = False,
):
    if zero_stage == 0:
        return (
            DistributedDataParallel(module, parallel_context=parallel_context),
            optimizer,
        )

    elif zero_stage == 1:
        module = FullyShardedDataParallel(
            module=module,
            parallel_context=parallel_context,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            transformer_wrap_layers=transformer_wrap_layers,
            mixed_precision=mixed_precision,
            cpu_offload=CPUOffload(offload_params=cpu_offload),
        )
        optimizer = ZeroRedundancyOptimizer(
            module.parameters(),
            optimizer_class=optimizer.__class__,
            **optimizer.defaults,
        )
        return module, optimizer

    elif zero_stage == 2:
        module = FullyShardedDataParallel(
            module=module,
            parallel_context=parallel_context,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            transformer_wrap_layers=transformer_wrap_layers,
            mixed_precision=mixed_precision,
            cpu_offload=CPUOffload(offload_params=cpu_offload),
        )
        optimizer = ZeroRedundancyOptimizer(
            module.parameters(),
            optimizer_class=optimizer.__class__,
            **optimizer.defaults,
        )
        return module, optimizer

    elif zero_stage == 3:
        module = FullyShardedDataParallel(
            module=module,
            parallel_context=parallel_context,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            transformer_wrap_layers=transformer_wrap_layers,
            mixed_precision=mixed_precision,
            cpu_offload=CPUOffload(offload_params=cpu_offload),
        )
        optimizer = ZeroRedundancyOptimizer(
            module.parameters(),
            optimizer_class=optimizer.__class__,
            **optimizer.defaults,
        )
        return module, optimizer
    else:
        raise ValueError("param `zero_stage` must be one of the 0, 1, 2, 3.")
