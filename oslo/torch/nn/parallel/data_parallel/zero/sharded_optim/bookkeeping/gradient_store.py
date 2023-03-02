from typing import List

from torch import Tensor
from torch.distributed import ProcessGroup

from oslo.torch.nn.parallel.data_parallel.zero.sharded_optim.bookkeeping._base_store import (
    BaseStore,
)


class GradientStore(BaseStore):
    """
    Class for storing and managing the averaged gradients for a given group.
    The gradients can be added and retrieved by group id and
    the average gradients can be reset for a specific group id.

    Args:
        torch_pg (ProcessGroup): The torch process group object used for distributed training.
    """

    def __init__(self, torch_pg: ProcessGroup):
        """
        Initialize the GradientStore.
        """
        super().__init__(torch_pg)
        self._averaged_gradients = dict()
        self._grad_acc_objs = []

    def append_accumulate_grad_object(self, tensor: Tensor):
        """
        Append an object to accumulate gradients.

        Args:
            tensor: The object to accumulate gradients.
        """
        self._grad_acc_objs.append(tensor)

    def get_averaged_gradients_by_group(self, group_id: int) -> List[Tensor]:
        """
        Get the averaged gradients for a specific group.

        Args:
            group_id (int): The group ID.

        Returns:
            list of Tensor: The averaged gradients for the group.
        """
        if group_id not in self._averaged_gradients:
            self._averaged_gradients[group_id] = []
        return self._averaged_gradients.get(group_id)

    def append_average_gradient_by_group(self, group_id: int, tensor: Tensor):
        """
        Append an averaged gradient to a specific group.

        Args:
            group_id (int): The group ID.
            tensor (Tensor): The averaged gradient.
        """
        if group_id in self._averaged_gradients:
            self._averaged_gradients[group_id].append(tensor)
        else:
            self._averaged_gradients[group_id] = [tensor]

    def add_average_gradient_by_group(
        self, group_id: int, tensor_index: int, tensor: Tensor
    ):
        """
        Add an averaged gradient to a specific group.

        Args:
            group_id (int): The group ID.
            tensor_index (int): The index of the tensor in the group.
            tensor (Tensor): The averaged gradient.
        """
        self._averaged_gradients[group_id][tensor_index].add_(tensor)

    def reset_average_gradients_by_group(self, group_id: int):
        """
        Reset the averaged gradients for a specific group.

        Args:
            group_id (int): The group ID.
        """
        self._averaged_gradients[group_id] = []

    def reset_all_average_gradients(self):
        """
        Reset all the averaged gradients.
        """
        self._averaged_gradients = dict()
