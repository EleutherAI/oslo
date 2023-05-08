from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from oslo.torch.nn.parallel.data_parallel.zero.chunk import (
    TensorState,
)
from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager import (
    HeterogeneousMemoryManager,
)
from oslo.torch.distributed.tensor.param_op_hook import (
    DistributedParamOpHook,
)
from oslo.torch.nn.parallel.data_parallel._utils import is_ddp_ignored


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class HeterogeneousZeROHook(DistributedParamOpHook):
    def __init__(self, heterogeneous_manager: HeterogeneousMemoryManager) -> None:
        super().__init__()
        self._heterogeneous_manager = heterogeneous_manager
        self._chunk_manager = heterogeneous_manager.chunk_manager
        self._training_phase = TrainingPhase.FORWARD

    def pre_op(self, params):
        params = [p for p in params if not is_ddp_ignored(p)]
        chunks = self._chunk_manager.get_chunks(params)
        for p in params:
            self._chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        self._heterogeneous_manager.sample_overall_data()
        self._heterogeneous_manager.adjust_layout(chunks)
        for chunk in chunks:
            self._chunk_manager.access_chunk(chunk)

        # record cuda model data of the current OP
        self._heterogeneous_manager.record_model_data_volume()

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

    @contextmanager
    def switch_training_phase(
        self, training_phase: TrainingPhase = TrainingPhase.BACKWARD
    ):
        old_training_phase = self._training_phase
        try:
            self._training_phase = training_phase
            yield
        finally:
            self._training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(
        switch_to_backward, training_phase=TrainingPhase.FORWARD
    )
