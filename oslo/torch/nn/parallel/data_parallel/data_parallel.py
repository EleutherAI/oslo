from typing import Optional, Callable

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
    auto_wrap_policy: Optional[Callable] = None,
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
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            cpu_offload=CPUOffload(offload_params=cpu_offload),
        )
        optimizer = ZeroRedundancyOptimizer(
            module.parameters(),
            optimizer_class=optimizer.__class__,
            lr=optimizer.param_groups[0]["lr"],
        )
        return module, optimizer

    elif zero_stage == 2:
        return (
            FullyShardedDataParallel(
                module=module,
                parallel_context=parallel_context,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                cpu_offload=CPUOffload(offload_params=cpu_offload),
            ),
            optimizer,
        )

    elif zero_stage == 3:
        return (
            FullyShardedDataParallel(
                module=module,
                parallel_context=parallel_context,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                cpu_offload=CPUOffload(offload_params=cpu_offload),
            ),
            optimizer,
        )
    else:
        raise ValueError("param `zero_stage` must be one of the 0, 1, 2, 3.")
