# from colossalai.context import ParallelMode
# from colossalai.core import global_context as gpc
from oslo.torch.distributed.parallel_mode import ParallelMode
rom oslo.torch.distributed import ParallelContext

class BaseStore:

    def __init__(self, dp_parallel_mode=ParallelMode.DATA):
        self._world_size = ParallelContext.get_world_size(dp_parallel_mode)
        self._local_rank = ParallelContext.get_local_rank(dp_parallel_mode)

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank
