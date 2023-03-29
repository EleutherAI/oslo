import functools
from time import time
from typing import List, Optional, Tuple

import torch

from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.chunk import (
    Chunk,
    ChunkManager,
)
from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.memory_tracer import (
    MemStats,
)

from .memory_tracer import ChunkMemStatsCollector
from .placement_policy import PlacementPolicyFactory


class HeterogeneousMemoryManager:
    """
    Stateful Tensor Manager, inspired from PatrickStar

    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818

    Args:
        placement_policy (str): Which device to place *held* tensors. It can be 'cpu', 'cuda' and 'auto'.
            If it's 'cpu', parameters, gradients and optimizer states will be offloaded to CPU, which means min CUDA memory will be used.
            If it's 'cuda', they won't be offloaded, which means max CUDA memory will be used.
            If it's 'auto', they are moving dynamically based on CPU and CUDA memory usage. It will utilize Heterogeneousgeneous memory space evenly and well.
            Note that 'auto' policy can only work well when no other processes use CUDA during your training.
        chunk_manager (ChunkManager): A ``ChunkManager`` instance.
        memstats (MemStats, optional): a mem stats collected by a runtime mem tracer. if None then HeterogeneousMemoryManager will collect it during a warmup iteration.
    """

    def __init__(
        self,
        placement_policy: str,
        chunk_manager: ChunkManager,
        memstats: Optional[MemStats] = None,
    ) -> None:

        assert placement_policy in PlacementPolicyFactory.get_policy_names()
        self.policy_name = placement_policy
        policy_cls = PlacementPolicyFactory.create(placement_policy)
        self._chunk_manager = chunk_manager

        self._premade_memstats_ = memstats is not None
        self._memstats = memstats
        self._mem_stats_collector = (
            ChunkMemStatsCollector(chunk_manager, self._memstats)
            if policy_cls.need_mem_stats
            else None
        )
        self._placement_policy = policy_cls(chunk_manager, self._mem_stats_collector)
        self._compute_list: List[Tuple[Chunk, ...]] = []
        self._compute_idx: int = -1

        self._h2d_volume = 0
        self._d2h_volume = 0
        self._layout_time = 0
        self._evict_time = 0
        self._warmup = True
        self._comp_cuda_demand_time = 0

    def reset_attributes(self):
        """Reset the attributes of the manager.
        """
        self._compute_idx = -1
        self._h2d_volume = 0
        self._d2h_volume = 0
        self._layout_time = 0
        self._evict_time = 0
        self._comp_cuda_demand_time = 0

    @property
    def need_warmup(self) -> bool:
        """Check if the manager needs a warmup iteration.

        Returns:
            bool: True if the manager needs a warmup iteration.
        """
        return self.policy_name in ("auto", "const")

    @property
    def is_warmup(self):
        """Check if the manager is in warmup iteration."""
        return self._warmup

    def memstats(self):
        """Memstats

        get the memory statistics during training.
        The stats could be collected by a runtime memory tracer, or collected by the HeterogeneousMemoryManager.
        Note, for the latter, you can not access the memstats before warmup iteration finishes.
        """
        if self._premade_memstats_:
            return self._memstats
        else:
            assert (
                not self._warmup
            ), "HeterogeneousMemory has memstats after warm up! Now is during warmup."
            return self._mem_stats_collector._memstats

    def pre_iter(self, *args):
        """This function must be called when each iteration starts"""
        if self._mem_stats_collector and self._warmup:
            self._mem_stats_collector.start_collection()

    def post_iter(self):
        """This function must be called when each iteration finishes"""
        if self._mem_stats_collector and self._warmup:
            self._mem_stats_collector.finish_collection()
        self._warmup = False
        self.reset_attributes()

    def adjust_layout(self, chunks: Tuple[Chunk, ...]):
        """Adjust the layout of stateful tensors according to the information provided
        by mem_stats_collector, which should belongs to a Sharded Model.

        Args:
            chunks (Tuple[Chunk, ...]): A tuple of chunks.
        """
        # find stateful tensor in state COMPUTE
        start = time()
        self._record_chunks_order(chunks)
        cuda_demand, hold_cuda_tensor_list = self._get_layout_info(
            self._compute_idx, self._warmup, chunks
        )
        self._layout_time += time() - start

        vol, evict_time = self._placement_policy.evict_tensors(
            can_evict_chunks=hold_cuda_tensor_list,
            cuda_demand=cuda_demand,
            warmup=self._warmup,
            compute_list=self._compute_list,
            compute_idx=self._compute_idx,
        )

        self._d2h_volume += vol
        self._evict_time += evict_time
        # move COMPUTE tensors to CUDA
        self._h2d_volume += cuda_demand

    @functools.lru_cache(maxsize=None)
    def _get_layout_info(
        self, compute_idx: int, warmup: bool, chunks: Tuple[Chunk, ...]
    ) -> Tuple[int, Tuple[Chunk, ...]]:
        """Get the layout information of the chunks.

        Args:
            compute_idx (int): The index of the compute list.
            warmup (bool): Whether it is in the warmup iteration.
            chunks (Tuple[Chunk, ...]): A tuple of chunks.

        Returnes:
            Tuple[int, Tuple[Chunk, ...]]: The cuda demand and the chunks that can be evicted.
        """
        start = time()
        cuda_demand = 0
        for chunk in chunks:
            if chunk.device_type == "cuda":
                if chunk.is_gathered:
                    pass
                else:
                    cuda_demand += chunk.chunk_mem - chunk.shard_mem
            elif chunk.device_type == "cpu":
                cuda_demand += chunk.chunk_mem
            else:
                raise RuntimeError
        self._comp_cuda_demand_time += time() - start

        can_evict_chunks = self._chunk_manager.get_cuda_movable_chunks()
        return cuda_demand, can_evict_chunks

    def _record_chunks_order(self, chunks: Tuple[Chunk, ...]):
        """Record the chunks order."""
        self._compute_idx += 1
        if self._warmup and self._placement_policy.need_mem_stats:
            self._compute_list.append(chunks)

    @property
    def default_device(self) -> torch.device: 
        """Get the default device.

        Returns:
            torch.device: The default device.
        """
        return self._placement_policy.get_default_device()

    def sample_overall_data(self):
        """Sample the overall data of the model.
        """
        if self._mem_stats_collector:
            self._mem_stats_collector.sample_overall_data()

    def record_model_data_volume(self):
        """Record the model data volume.
        """
        if self._mem_stats_collector:
            self._mem_stats_collector.record_model_data_volume()

    @property
    def chunk_manager(self) -> ChunkManager:
        """Get the chunk manager.

        Returns:
            ChunkManager: The chunk manager.
        """
        return self._chunk_manager

    @property
    def cuda_margin_mem(self) -> Optional[float]:
        """Get the cuda margin memory.

        Returns:
            Optional[float]: The cuda margin memory.
        """
        if self._mem_stats_collector:
            return self._mem_stats_collector.cuda_margin_mem
        return None

    @property
    def is_cuda_margin_mem_avail(self) -> bool:
        """Check if the cuda margin memory is available."""
        return self._placement_policy.need_mem_stats

    @staticmethod
    def get_default_device(policy_name: str) -> torch.device:
        """Get the default device.

        Args:
            policy_name (str): The name of the placement policy.

        Returns:
            torch.device: The default device.
        """
        return PlacementPolicyFactory.get_default_device(policy_name)
