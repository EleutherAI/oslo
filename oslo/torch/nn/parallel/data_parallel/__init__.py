from oslo.torch.nn.parallel.data_parallel.data_parallel import (
    DistributedDataParallel,
)
from oslo.torch.nn.parallel.data_parallel.zero import *

__all__ = ["DistributedDataParallel", "ZeroRedundancyOptimizer"]
