import torch.distributed as dist
from torch.distributed import ProcessGroup


class BaseStore:
    """
    Base class for store classes.
    
    Args:
        torch_pg (ProcessGroup): The torch process group object used for distributed training.
    """

    def __init__(self, torch_pg: ProcessGroup):
        """
        Initialize the BaseStore class.
        """
        self._world_size = dist.get_world_size(group=torch_pg)
        self._local_rank = dist.get_rank(group=torch_pg)

    @property
    def world_size(self) -> int:
        """
        Returns:
            int: Total number of processes in the process group.
        """
        return self._world_size

    @property
    def local_rank(self) -> int:
        """
        Returns:
            int: Rank of the current process in the process group.
        """
        return self._local_rank