from abc import abstractmethod

import json
from time import time
import torch

from typing import Dict, List, Union


class MemoryMonitor:
    """
    Base class for all types of memory monitors.

    All monitors should have a list called `time_stamps` and a list called `mem_stats`.
    """

    def __init__(self):
        self.time_stamps = []
        self.mem_stats = []

    def __len__(self):
        return len(self.mem_stats)

    @abstractmethod
    def start(self):
        """Start monitoring memory usage."""
        pass

    @abstractmethod
    def finish(self):
        """Finish monitoring memory usage and record the results."""
        pass

    def state_dict(self) -> Dict[str, Union[List[float], List[int]]]:
        """
        Get the state dictionary containing time_stamps and mem_stats.

        Returns:
            Dict[str, Union[List[float], List[int]]]: A dictionary containing the time_stamps and mem_stats lists.
        """
        return {
            "time_stamps": self.time_stamps,
            "mem_stats": self.mem_stats,
        }

    def save(self, filename: str):
        """
        Save the state dictionary to a file.

        Args:
            filename (str): The name of the file to save the state dictionary to.
        """
        with open(filename, "w") as f:
            json.dump(self.state_dict(), f)

    def clear(self):
        """Clear the time_stamps and mem_stats lists."""
        self.mem_stats.clear()
        self.time_stamps.clear()


class SyncCudaMemoryMonitor(MemoryMonitor):
    """
    A synchronized CUDA memory monitor.

    It only records the maximum allocated CUDA memory from the start point to the finish point.
    """

    def __init__(self, power: int = 10):
        super().__init__()

    def start(self):
        """
        Start monitoring the CUDA memory usage.
        """
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def finish(self) -> int:
        """
        Finish monitoring the CUDA memory usage and record the maximum GPU memory used since the latest `start()`.

        Returns:
            int: The maximum GPU memory used.
        """
        torch.cuda.synchronize()
        self.time_stamps.append(time())
        max_usage = torch.cuda.max_memory_allocated()
        self.mem_stats.append(max_usage)
        return max_usage
