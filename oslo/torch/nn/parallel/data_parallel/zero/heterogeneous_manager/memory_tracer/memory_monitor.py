
from abc import abstractmethod

import json
from time import time
import torch

class MemoryMonitor:
    """Base class for all types of memory monitor.
    All monitors should have a list called `time_stamps` and a list called `mem_stats`.
    """

    def __init__(self):
        self.time_stamps = []
        self.mem_stats = []

    def __len__(self):
        return len(self.mem_stats)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    def state_dict(self):
        return {
            "time_stamps": self.time_stamps,
            "mem_stats": self.mem_stats,
        }

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.state_dict(), f)

    def clear(self):
        self.mem_stats.clear()
        self.time_stamps.clear()


class SyncCudaMemoryMonitor(MemoryMonitor):
    """
    A synchronized cuda memory monitor.
    It only record the maximum allocated cuda memory from start point to finish point.
    """

    def __init__(self, power: int = 10):
        super().__init__()

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def finish(self) -> int:
        """
        return max gpu memory used since latest `start()`.
        Returns:
            int: max GPU memory
        """
        torch.cuda.synchronize()
        self.time_stamps.append(time())
        max_usage = torch.cuda.max_memory_allocated()
        self.mem_stats.append(max_usage)
        return max_usage