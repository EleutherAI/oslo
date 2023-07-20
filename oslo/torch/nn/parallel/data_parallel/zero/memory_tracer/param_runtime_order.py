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

from abc import ABC

import torch


class ParamGenerator(ABC):
    def append(self, param: torch.nn.Parameter):
        pass

    def generate(self):
        pass

    def clear(self):
        pass


class OrderedParamGenerator(ParamGenerator):
    """OrderedParamGenerator

    Contain the order of parameters visited during runtime.
    """

    def __init__(self) -> None:
        self.param_visited_order = []

    def append(self, param: torch.nn.Parameter):
        """Append a parameter to the param_visited_order.

        Args:
            param (torch.nn.Parameter): A torch parameter.
        """
        self.param_visited_order.append(param)

    def generate(self):
        """Generate the parameter order.

        Yields:
            torch.nn.Parameter: A torch parameter.
        """
        visited_set = set()
        for p in self.param_visited_order:
            if p not in visited_set:
                yield p
            visited_set.add(p)
        del visited_set

    def is_empty(self) -> bool:
        """Check if the param_visited_order is empty.

        Returns:
            bool: True if the param_visited_order is empty.
        """
        return len(self.param_visited_order) == 0

    def clear(self):
        """Clear the param_visited_order."""
        self.param_visited_order = []
