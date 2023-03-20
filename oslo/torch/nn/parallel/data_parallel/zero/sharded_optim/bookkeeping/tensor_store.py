# Copyright 2021 HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by EleutherAI on 2023.

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from torch import Tensor
from typing import List


class TensorBucket:
    def __init__(self, max_size: int):
        """
        Initialize the TensorBucket with max_size.

        Args:
            max_size (int): Maximum size for the tensor bucket.
        """
        self._max_size = max_size
        self._current_size = 0
        self._bucket = []

    @property
    def max_size(self) -> int:
        """
        Get the maximum size for the tensor bucket.

        Returns:
            int: Maximum size for the tensor bucket.
        """
        return self._max_size

    @property
    def current_size(self) -> int:
        """
        Get the current size of the tensor bucket.

        Returns:
            int: Current size of the tensor bucket.
        """
        return self._current_size

    def is_full_or_oversized(self) -> bool:
        """
        Check if the tensor bucket is full or oversized.

        Returns:
            bool: True if the current size of the tensor bucket is greater than or equal to the maximum size,
                  False otherwise.
        """
        return self._current_size >= self._max_size

    def is_empty(self) -> bool:
        """
        Check if the tensor bucket is empty.

        Returns:
            bool: True if the tensor bucket is empty, False otherwise.
        """
        return len(self._bucket) == 0

    def add_to_bucket(self, tensor: Tensor, allow_oversize: bool = False):
        """
        Add a tensor to the tensor bucket.

        Args:
            tensor (torch.Tensor): Tensor to be added to the tensor bucket.
            allow_oversize (bool, optional): If True, allows tensor to be added even if it will exceed the max_size.
                                             Default: False.

        Raises:
            RuntimeError: If the tensor will exceed the max_size and allow_oversize is False.
        """
        tensor_size = tensor.numel()

        if not allow_oversize and self.will_exceed_max_size(tensor_size):
            msg = (
                f"The TensorBucket max size of {self._max_size} is exceeded by "
                f"the added tensor (size: {tensor_size})"
            )
            raise RuntimeError(msg)

        self._bucket.append(tensor)
        self._current_size += tensor_size

    def will_exceed_max_size(self, tensor_size: int) -> bool:
        """
        Check if the addition of a tensor to the bucket will exceed the max_size.

        Args:
            tensor_size (int): Size of the tensor to be added to the bucket.

        Returns:
            bool: True if adding the tensor to the bucket will exceed the max_size, False otherwise.
        """
        expected_size = self._current_size + tensor_size
        return expected_size > self._max_size

    def get_bucket(self) -> List[Tensor]:
        """
        Get the bucket of tensors.

        Returns:
            List[torch.Tensor]: List of tensors stored in the bucket.
        """
        return self._bucket

    def empty(self):
        """
        Empty the TensorBucket by resetting the bucket and current size to empty and zero.
        """
        self._bucket = []
        self._current_size = 0

    def flatten(self):
        """
        Flatten the dense tensors in the TensorBucket.

        Returns:
            The flattened tensors.
        """
        return _flatten_dense_tensors(self._bucket)

    def unflatten_and_copy(self, flat_tensor: Tensor):
        """
        Unflatten the flattened tensors and copy the result to the TensorBucket.

        Args:
            flat_tensor (Tensor): The flattened tensor.
        """
        unflattened_tensor_list = _unflatten_dense_tensors(flat_tensor, self._bucket)
        for old, new in zip(self._bucket, unflattened_tensor_list):
            old.copy_(new)
