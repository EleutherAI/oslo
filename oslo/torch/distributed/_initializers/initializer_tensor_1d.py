import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)
from oslo.torch.distributed.parallel_mode import ParallelMode


class TensorParallel1DGroupInitializer(ProcessGroupInitializer):
    """Process group initializer for 1D tensor parallelism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size

    def init_dist_group(self):
        """
        Initialize 1D tensor parallel groups, and assign local_ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode, parallel_input_1d
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_1D

        for i in range(self.num_group):
            ranks = [
                i * self.tensor_parallel_size + j
                for j in range(self.tensor_parallel_size)
            ]
            group = dist.new_group(ranks)
            group_cpu = (
                dist.new_group(ranks, backend="gloo")
                if dist.get_backend() != "gloo"
                else group
            )

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }
