from dataclasses import dataclass
from typing import Optional

from .distributed_spec import DistributedSpec, DistributedPlacementPattern
from oslo.torch.distributed.parallel_context import ParallelContext

from .compute_spec import ComputeSpec


@dataclass
class DistributedTensorSpec:
    """ DistributedTensorSpec
    
    A data class for specifications of the `DistributedTensor`.
    It contains attributes of `ProcessGroup`, `DistributedSpec`, `ComputeSpec`.
    The latter two attributes are optional. If not set, they are default value is `Replicate()` and `None`.
    """
    parallel_context: ParallelContext
    dist_attr: Optional[DistributedSpec] = DistributedSpec(DistributedPlacementPattern.REPLICATE)
    compute_attr: Optional[ComputeSpec] = None