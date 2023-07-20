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

from typing import List, Optional

import torch

from .param_runtime_order import OrderedParamGenerator


class MemStats(object):
    """
    Store the non model data statistics used for heterogeneous memory manager and Zero optimizer.
    """

    def __init__(self) -> None:

        # (preop_step, List[param])
        self._step_param_dict = dict()
        # (param, List[preop_step])
        self._param_step_dict = dict()
        # (preop_step, non_model_data) non model data used during preop_step ~ (preop_step+1)
        self._step_nmd_dict = dict()
        self._param_runtime_order = OrderedParamGenerator()

        self._preop_step = 0

        self._prev_overall_cuda = -1
        self._max_overall_cuda = 0
        self._prev_md_cuda = -1

        # old version
        self._model_data_cuda_list = []
        self._model_data_cpu_list = []

        self._overall_cuda_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []

    def calc_max_cuda_non_model_data(self):
        """
        Calculate the maximum CUDA non-model data and store it in the step_nmd_dict.
        """
        if self._prev_overall_cuda != -1 and self._prev_md_cuda != -1:
            max_cuda_non_model_data = self._prev_overall_cuda - self._prev_md_cuda
            self._step_nmd_dict[self._preop_step - 1] = max_cuda_non_model_data
            # compatibility of the old version.
            self._non_model_data_cuda_list.append(max_cuda_non_model_data)

    def record_max_cuda_model_data(self, val: int):
        """
        Record the maximum CUDA model data.

        Args:
            val (int): The maximum CUDA model data value.
        """
        self._prev_md_cuda = val

    def record_max_cuda_overall_data(self, val: int):
        """
        Record the maximum CUDA overall data.

        Args:
            val (int): The maximum CUDA overall data value.
        """
        self._prev_overall_cuda = val
        self._max_overall_cuda = max(self._max_overall_cuda, val)

    @property
    def max_overall_cuda(self) -> int:
        """
        Get the maximum overall CUDA data.

        Returns:
            int: The maximum overall CUDA data value.
        """
        return self._max_overall_cuda

    def increase_preop_step(self, param_list: List[torch.nn.Parameter]) -> None:
        """
        Increase the pre-operation time step and store the list of parameters used between the current and next time steps.

        Args:
            param_list (List[torch.nn.Parameter]): A list of torch parameters.
        """
        for p in param_list:
            if p not in self._param_step_dict:
                self._param_step_dict[p] = [self._preop_step]
            else:
                self._param_step_dict[p].append(self._preop_step)
            self._param_runtime_order.append(p)
        self._step_param_dict[self._preop_step] = param_list
        self._preop_step += 1

    def param_used_step(self, param: torch.nn.Parameter) -> Optional[List[int]]:
        """param_used_step
        Get the timestep list using the param

        Args:
            param (torch.nn.Parameter): a torch param

        Returns:
            Optional[List[int]]: a list of int indicates the time step of preop hook.
        """
        if param not in self._param_step_dict:
            return None
        else:
            return self._param_step_dict[param]

    def param_order(self) -> OrderedParamGenerator:
        """
        Get the parameter order from the param_runtime_order object.

        Returns:
            OrderedParamGenerator: The OrderedParamGenerator object.

        Raises:
            RuntimeError: If the param_runtime_order is empty.
        """
        if self._param_runtime_order.is_empty():
            raise RuntimeError
        else:
            return self._param_runtime_order

    def non_model_data_list(self, device_type: str) -> List[int]:
        """
        Get the non-model data list for the specified device type.

        Args:
            device_type (str): The device type, either "cuda" or "cpu".

        Returns:
            List[int]: A list of non-model data.

        Raises:
            TypeError: If the device_type is not "cuda" or "cpu".
        """
        if device_type == "cuda":
            return self._non_model_data_cuda_list
        elif device_type == "cpu":
            return self._non_model_data_cpu_list
        else:
            raise TypeError

    def max_non_model_data(self, device_type: str) -> float:
        """
        Get the maximum non-model data for the specified device type.

        Args:
            device_type (str): The device type, either "cuda" or "cpu".

        Returns:
            float: The maximum non-model data.

        Raises:
            TypeError: If the device_type is not "cuda" or "cpu".
        """
        if device_type == "cuda":
            return max(self._non_model_data_cuda_list)
        elif device_type == "cpu":
            return max(self._non_model_data_cpu_list)
        else:
            raise TypeError

    def clear(self):
        """
        Clear all the data lists, dictionaries, and reset the pre-operation step value.
        """
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cpu_list = []
        self._non_model_data_cuda_list = []

        self._param_runtime_order.clear()
        self._step_param_dict.clear()
        self._param_step_dict.clear()
        self._step_nmd_dict.clear()
        self._preop_step = 0

        self._prev_overall_cuda = -1
        self._prev_md_cuda = -1
