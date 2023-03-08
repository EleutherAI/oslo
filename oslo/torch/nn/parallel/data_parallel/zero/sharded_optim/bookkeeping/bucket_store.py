from torch import Tensor
from torch.distributed import ProcessGroup
from typing import List
from oslo.torch.nn.parallel.data_parallel.zero.sharded_optim.bookkeeping._base_store import (
    BaseStore,
)


class BucketStore(BaseStore):
    """
    Class for storing parameters and gradients during bucketing.

    Args:
        torch_pg (ProcessGroup): The torch process group object used for distributed training.
    """

    def __init__(self, torch_pg: ProcessGroup):
        """
        Initialize the BucketStore class.
        """
        super().__init__(torch_pg)
        self._params = dict()
        self._num_elements_in_bucket = dict()

        self.reset()

    def num_elements_in_bucket(self, reduce_rank: int = None):
        """
        Get the number of elements in a bucket.

        Args:
            reduce_rank (int, optional): The rank of the process to which the elements are being reduced.
                If None, reduction is being performed locally. Defaults to None.

        Returns:
            int: The number of elements in the bucket.
        """
        return self._num_elements_in_bucket[reduce_rank]

    def add_num_elements_in_bucket(self, num_elements: int, reduce_rank: int = None):
        """
        Add the number of elements to a bucket.

        Args:
            num_elements (int): The number of elements to add to the bucket.
            reduce_rank (int, optional): The rank of the process to which the elements are being reduced.
                If None, reduction is being performed locally. Defaults to None.
        """
        self._num_elements_in_bucket[reduce_rank] += num_elements

    def add_param(self, tensor: Tensor, reduce_rank: int = None):
        """
        Add a tensor to the bucket.

        Args:
            tensor (Tensor): The tensor to add to the bucket.
            reduce_rank (int, optional): The rank of the process to which the tensor is being reduced.
                If None, reduction is being performed locally. Defaults to None.
        """
        self._params[reduce_rank].append(tensor)

    def reset(self):
        """
        Reset the stored parameters and number of elements in the bucket for all reduce ranks.
        """
        keys = [None] + list(range(self._world_size))
        self._params = {rank: [] for rank in keys}
        self._num_elements_in_bucket = {rank: 0 for rank in keys}

    def reset_by_rank(self, reduce_rank=None):
        """
        Reset the stored parameters and number of elements in the bucket for a specific reduce rank.

        Args:
            reduce_rank (int, optional): The rank of the process to reset. If None, the local rank is used. Defaults to None.
        """
        self._params[reduce_rank] = []
        self._num_elements_in_bucket[reduce_rank] = 0

    def get_grad(self, reduce_rank: int = None) -> List[Tensor]:
        """
        Get the gradients of the parameters stored in the bucket for a specific reduce rank.

        Args:
            reduce_rank (int, optional): The rank of the process to get the gradients from.
                If None, the gradients are retrieved from the local process. Defaults to None.

        Returns:
            list of Tensor: The list of gradients for the parameters stored in the bucket.

        Raises:
            AssertionError: If any of the parameters stored in the bucket does not have a gradient.
        """
        param_list = self.get_param(reduce_rank)
        for param in param_list:
            # the param must have grad for reduction
            assert (
                param.grad is not None
            ), f"Parameter of size ({param.size()}) has None grad, cannot be reduced"
        return [param.grad for param in param_list]

    def get_param(self, reduce_rank: int = None) -> List[Tensor]:
        """
        Get the parameters stored in the bucket for a specific reduce rank.

        Args:
            reduce_rank (int, optional): The rank of the process to get the parameters from.
                If None, the parameters are retrieved from the local process. Defaults to None.

        Returns:
            list of Tensor: The list of parameters stored in the bucket.
        """
        return self._params[reduce_rank]
