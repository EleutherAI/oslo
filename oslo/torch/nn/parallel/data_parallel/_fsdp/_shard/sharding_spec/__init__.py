from oslo.torch.nn.parallel.data_parallel._fsdp._shard.sharding_spec.api import (
    DevicePlacementSpec,
    EnumerableShardingSpec,
    PlacementSpec,
    ShardingSpec,
    _infer_sharding_spec_from_shards_metadata,
)
from oslo.torch.nn.parallel.data_parallel._fsdp._shard.sharding_spec.chunk_sharding_spec import (
    ChunkShardingSpec,
)
