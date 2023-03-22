from enum import Enum
from typing import List


__all__ = ["ReplicaSpec", "ShardSpec"]


class DistributedPlacementPattern(Enum):
    REPLICATE = "r"
    SHARD = "s"


class DistributedSpec:
    """DistributedSpec

    A class indicating Distributed Specification.
    The DistributedSpec only works for tensor parallel process groups.
    The dist spec of data parallel process groups can be automatically deduced.
    This is an internal data structure.
    The API for users should be `ShardSpec` and `ReplicaSpec`.

    Args:
        dist_placement_pattern (DistributedPlacementPattern):
            The pattern describing how tensors are distributed among processes.
            The dist_placement_pattern is picked from a limited set, now including two patterns: replicate and shard.
    """

    def __init__(
        self, dist_placement_pattern: DistributedPlacementPattern, **meta_info
    ):
        self.placement = dist_placement_pattern
        self.__dict__.update(meta_info)

    def __eq__(self, other: "DistributedSpec") -> bool:
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        attr_list = [f"{attr}={str(value)}" for attr, value in self.__dict__.items()]
        attr_str = ", ".join(attr_list)
        return f"DistributedSpec({attr_str})"


def ReplicaSpec() -> DistributedSpec:
    """ReplicaSpec

    A distributed specification represents the tensor is replicated among the tensor parallel process group.

    Returns:
        DistributedSpec: an replicated dist spec instance.
    """
    return DistributedSpec(DistributedPlacementPattern.REPLICATE)


def ShardSpec(dims: List[int], num_partitions: List[int]) -> DistributedSpec:
    """ShardSpec

    A distributed specification represents the tensor is sharded among the tensor parallel process group.

    Note:
        Currently, only shard on one dimension is valid. In another word, dims should be of size 1.

    Args:
        dims (List[int]): a list of dimensions
        num_partitions (List[int]): a list of partition number of each dimensions.

    Returns:
        DistributedSpec: an shard dist spec instance.
    """
    assert isinstance(dims, list) and isinstance(num_partitions, list)
    assert len(dims) == len(num_partitions)
    return DistributedSpec(
        DistributedPlacementPattern.SHARD,
        dims=tuple(dims),
        num_partitions=tuple(num_partitions),
    )
