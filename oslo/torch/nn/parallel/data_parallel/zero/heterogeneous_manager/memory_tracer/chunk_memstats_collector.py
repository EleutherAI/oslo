import time
from typing import Optional

from .memory_monitor import SyncCudaMemoryMonitor

from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.chunk import (
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
        if memstats is not None:
            self.use_outside_memstats = True
            self._memstats = memstats
        else:
            self.use_outside_memstats = False
            self._memstats = MemStats()
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
    def sampling_time(self):
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self.sample_overall_data()
        # self._step_total = len(self._sampling_time)
        self._step_total = len(self._memstats.non_model_data_list("cuda"))
        self._start_flag = False
        print(f"finish_collection {self._step_total}")

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
        self._memstats.clear()
        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0

    @property
    def cuda_margin_mem(self) -> float:
        return (
            get_device_memory_capacity(get_current_device())
            - self._memstats.max_overall_cuda
        )
