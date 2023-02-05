from typing import List

from torch import Tensor
from torch.distributed import ProcessGroup

from ._base_store import BaseStore


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

    def add_accumulate_grad_object(self, tensor: Tensor):
        """
        Add an object to accumulate gradients.

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
        return self._averaged_gradients.get(group_id, [])

    def add_average_gradient_by_group(self, group_id: int, tensor: Tensor):
        """
        Add an averaged gradient to a specific group.

        Args:
            group_id (int): The group ID.
            tensor (Tensor): The averaged gradient.
        """
        if group_id in self._averaged_gradients:
            self._averaged_gradients[group_id].append(tensor)
        else:
            self._averaged_gradients[group_id] = [tensor]

    def reset_average_gradients_by_group(self, group_id: int):
        """
        Reset the averaged gradients for a specific group.

        Args:
            group_id (int): The group ID.
        """
        self._averaged_gradients[group_id] = []
