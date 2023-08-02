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

from enum import Enum
from typing import List

import torch

from oslo.torch.nn.parallel.data_parallel.zero.hetero.chunk import (
    TensorState,
)
from oslo.torch.nn.parallel.data_parallel.zero.hetero.memory_manager import (
    HeteroMemoryManager,
)
from oslo.torch.nn.parallel.data_parallel._utils import is_ddp_ignored


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class HeterogeneousZeROHook:
    def __init__(self, hetero_memory_manager: HeteroMemoryManager) -> None:
        super().__init__()
        self._hetero_memory_manager = hetero_memory_manager
        self._chunk_manager = hetero_memory_manager.chunk_manager
        self._training_phase = TrainingPhase.FORWARD

    def pre_op(self, params):
        params = [p for p in params if not is_ddp_ignored(p)]
        chunks = self._chunk_manager.get_chunks(params)
        for p in params:
            self._chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        self._hetero_memory_manager.sample_overall_data()
        self._hetero_memory_manager.adjust_layout(chunks)
        for chunk in chunks:
            self._chunk_manager.access_chunk(chunk)

        # record cuda model data of the current OP
        self._hetero_memory_manager.record_model_data_volume()

    def post_op(self, params):
        params = [p for p in params if not is_ddp_ignored(p)]
        for p in params:
            tensor_state = (
                TensorState.HOLD
                if self._training_phase == TrainingPhase.FORWARD or not p.requires_grad
                else TensorState.HOLD_AFTER_BWD
            )
            self._chunk_manager.trans_tensor_state(p, tensor_state)

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_backward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def toggle_training_phase(self):
        if self._training_phase == TrainingPhase.FORWARD:
            self._training_phase = TrainingPhase.BACKWARD
        else:
            self._training_phase = TrainingPhase.FORWARD
