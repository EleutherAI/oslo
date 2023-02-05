from typing import List

from torch import Tensor
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class ParameterStore(BaseStore):
    """
    class that manages the mapping between parameters 
    and the ranks responsible for updating them, 
    as well as the state of reduction for each parameter.

    Args:
        torch_pg (ProcessGroup): The torch process group object used for distributed training.
    """

    def __init__(self, torch_pg: ProcessGroup):
        """
        Initialize the ParameterStore.
        """

        super().__init__(torch_pg)
        # param partitioning data structures
        self._fp16_param_to_rank = dit()
        self._rank_groupid_to_fp16_param_list = dict()
        self._rank_group_id_to_flat_fp16_param = dict()

        # param reduction data structures
        self._is_param_reduced = dict()
        self._reduced_param = []

    def set_param_to_rank(self, tensor: Tensor, rank: int) -> None:
        """
        Set the mapping between parameter to rank, each parameter should be owned by a rank.

        Args:
            tensor (Tensor): The fp16 parameters.
            ranks (int): The rank of which the process is responsible for updating the parameter
        """

        self._fp16_param_to_rank[tensor] = rank

    def get_param_rank(self, tensor: Tensor) -> int:
        """
        Gives the rank which the parameter belongs to

        Args:
            tensor (Tensor): The fp16 parameters.

        Returns:
            int: The rank of fp 16 params.

        """
        return self._fp16_param_to_rank[tensor]

    def belongs_to_current_rank(self, tensor: Tensor) -> bool:
        """
        Check whether a parameter is supposed to be updated by the process of the current rank

        Args:
            tensor (Tensor): The parameter.

        Returns:
            bool: True if the parameter should be updated by the current rank. Otherwise false.
        """

        tensor_rank = self._fp16_param_to_rank[tensor]
        return tensor_rank == self._local_rank

    def add_flat_fp16_param_by_rank_group(self, rank, group_id, tensor):
        """
        Add a flat FP16 parameter by rank and group.

        Args:
            rank (int): The rank.
            group_id (int): The group ID.
            tensor (Tensor): The flat FP16 parameter.
        """
        if rank not in self._rank_group_id_to_flat_fp16_param:
            self._rank_group_id_to_flat_fp16_param[rank] = dict()

        self._rank_group_id_to_flat_fp16_param[rank][group_id] = tensor

    def get_flat_fp16_param_by_rank_group(self, rank, group_id):
        """
        Get a flat FP16 parameter by rank and group.

        Args:
            rank (int): The rank.
            group_id (int): The group ID.

        Returns:
            Tensor: The flat FP16 parameter.
        """
        return self._rank_group_id_to_flat_fp16_param[rank][group_id]

    def is_param_reduced(self, tensor):
        """
        Check if a parameter has been reduced.

        Args:
            tensor (Tensor): The parameter to check.

        Returns:
            bool: True if the parameter has been reduced, False otherwise.
        """
        return self._is_param_reduced[tensor]

    def set_param_reduction_state(self, tensor, state):
        """
        Set the reduction state of a parameter.

        Args:
            tensor (Tensor): The parameter.
            state (bool): The reduction state.
        """
        self._is_param_reduced[tensor] = state

    def get_param_reduction_states(self):
        """
        Get the reduction states of all parameters.

        Returns:
            Dict[Tensor, bool]: The reduction states.
        """
        return self._is_param_reduced

    def reset_previous_reduced_params(self):
        """
        Reset the list of previously reduced parameters.
        """
        self._reduced_param = []

    def add_previous_reduced_param(self, tensor):
        """
        Add a reduced parameter to the list of previously reduced parameters.

        Args:
            tensor (Tensor): The reduced parameter.
        """
        self._reduced_param.append(tensor)

    def clear_grads_of_previous_reduced_params(self):
        """
        Clear the gradients of previously reduced parameters.
        """
        if len(self._reduced_param) > 0:


