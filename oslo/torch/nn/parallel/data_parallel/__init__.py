from oslo.torch.nn.parallel.data_parallel.distributed_data_parallel import (
    DistributedDataParallel,
)
from oslo.torch.nn.parallel.data_parallel.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
)
from oslo.torch.nn.parallel.data_parallel.sharded_data_parallel import (
    ShardedDataParallel,
)


def add_wrapper(module, mode, wrapper):
    if hasattr(module, "oslo_wrappers"):
        module.oslo_wrappers[mode] = wrapper
    else:
        setattr(module, "oslo_wrappers", {mode: wrapper})
