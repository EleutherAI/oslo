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

import time
from typing import Optional, List

from .memory_monitor import SyncCudaMemoryMonitor

from oslo.torch.nn.parallel.data_parallel.zero.chunk import (
    ChunkManager,
)

from .memory_stats import MemStats

from oslo.torch.nn.parallel.data_parallel.zero.utils import get_current_device
from .utils import get_device_memory_capacity


class ChunkMemStatsCollector:
    """
    Memory Statistic Collector for Chunks.

    Args:
        chunk_manager (ChunkManager): the chunk manager.
        memstats (Optional[MemStats], optional): memory statistics collected by RMT. Defaults to None.
    """

    def __init__(
        self, chunk_manager: ChunkManager, memstats: Optional[MemStats] = None
    ) -> None:
        self._mem_monitor = SyncCudaMemoryMonitor()
        self._sampling_time = []
        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0
        self.use_outside_memstats = memstats is not None
        self._memstats = memstats or MemStats()
        self._chunk_manager = chunk_manager

    def next_period_non_model_data_usage(self, device_type: str) -> int:
        """Maximum non model data memory usage during the next Op run

        Args:
            device_type (str): device type, can be 'cpu' or 'cuda'.

        Returns:
            int: max non model data memory usage of current sampling period
        """
        assert (
            not self._start_flag
        ), "Cannot get mem stats info during collection phase."
        assert (
            self._step_total > 0
        ), "Cannot get mem stats info before collection phase."
        assert len(self._memstats.non_model_data_list(device_type)) > self._step_idx, (
            f"{len(self._memstats.non_model_data_list(device_type))} should be > than step idx {self._step_idx}, "
            f"step total {self._step_total}"
        )
        next_non_model_data = self._memstats.non_model_data_list(device_type)[
            self._step_idx
        ]
        self._step_idx = (self._step_idx + 1) % self._step_total
        return next_non_model_data

    @property
    def sampling_time(self) -> List[float]:
        """Sampling time of each step.

        Returns:
            List[float]: sampling time of each step.
        """
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        """Start collection of memory statistics."""
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        """Finish collection of memory statistics."""
        self.sample_overall_data()
        # self._step_total = len(self._sampling_time)
        self._step_total = len(self._memstats.non_model_data_list("cuda"))
        self._start_flag = False

    def record_model_data_volume(self):
        """
        record model data volume on cuda and cpu.
        """
        if self._start_flag and not self.use_outside_memstats:
            cuda_mem = self._chunk_manager.total_mem["cuda"]
            self._memstats.record_max_cuda_model_data(cuda_mem)

    def sample_overall_data(self):
        """
        Sampling overall and non model data cuda memory statistics.
        """
        if self._start_flag and not self.use_outside_memstats:
            cuda_overall = self._mem_monitor.finish()
            self._memstats.record_max_cuda_overall_data(cuda_overall)
            self._memstats.calc_max_cuda_non_model_data()

            self._mem_monitor.start()

        if self._start_flag:
            self._sampling_time.append(time.time())

    def clear(self):
        """Clear memory statistics."""
        self._memstats.clear()
        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0

    @property
    def cuda_margin_mem(self) -> float:
        """Margin memory on cuda device."""
        return (
            get_device_memory_capacity(get_current_device())
            - self._memstats.max_overall_cuda
        )
