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
