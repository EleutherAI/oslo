import torch
import psutil
import socket
from collections import Counter
from collections import namedtuple

from typing import Dict

from oslo.torch.nn.parallel.data_parallel.zero.utils import get_current_device
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode

import torch.distributed as dist


_GLOBAL_CUDA_MEM_FRACTION = 1.0
_GLOBAL_CPU_MEM_CAPACITY = -1


def _get_cpu_memory_info() -> Dict:
    """Get the memory info of the current process in bytes.

    Returns:
        Dict: a dict of memory info
    """
    ps_mem_info = namedtuple(
        "ps_mem_info", ["total", "free", "cached", "buffers", "used"]
    )
    try:
        # psutil reads the memory info from /proc/memory_info,
        # which results in returning the host memory instead of
        # that of container.
        # Here we try to read the container memory with method in:
        # https://stackoverflow.com/a/46213331/5163915
        mems = {}
        with open("/sys/fs/cgroup/memory/memory.meminfo", "rb") as f:
            for line in f:
                fields = line.split()
                mems[fields[0]] = int(fields[1]) * 1024
        total = mems[b"MemTotal:"]
        free = mems[b"MemFree:"]
        cached = mems[b"Cached:"]
        buffers = mems[b"Buffers:"]
        used = total - free - cached - buffers
        if used < 0:
            used = total - free
        mem_info = ps_mem_info(
            total=total, free=free, cached=cached, buffers=buffers, used=used
        )
    except FileNotFoundError:
        mems = psutil.virtual_memory()
        mem_info = ps_mem_info(
            total=mems.total,
            free=mems.free,
            cached=mems.cached,
            buffers=mems.buffers,
            used=mems.used,
        )
    return mem_info


def get_cpu_memory_capacity() -> int:
    """
    Get the cpu memory capacity. We may not use all of it.

    Returns:
        int: _description_
    """
    global _GLOBAL_CPU_MEM_CAPACITY
    if _GLOBAL_CPU_MEM_CAPACITY == -1:
        mem_info = _get_cpu_memory_info()
        return mem_info.total
    else:
        return _GLOBAL_CPU_MEM_CAPACITY


def detect_num_processes_on_current_node() -> int:
    """Detect the number of processes on the current node.

    Returns:
        int: the number of processes on the current node
    """
    hostname = socket.gethostname()
    ctx = ParallelContext.get_context()
    hostname_list = [None for _ in range(ctx.get_global_rank())]
    dist.all_gather_object(
        hostname_list, hostname, group=ctx.get_group(ParallelMode.GLOBAL)
    )
    counter = Counter(hostname_list)
    return counter[hostname]


def get_device_memory_capacity(device: torch.device) -> int:
    """
    Get the capacity of the memory of the device

    Args:
        device (torch.device): a device

    Returns:
        int: size in byte
    """
    assert isinstance(device, torch.device)
    if device.type == "cpu":
        # In the context of 1-CPU-N-GPU, the memory capacity of the current process is 1/N overall CPU memory.
        return get_cpu_memory_capacity() / detect_num_processes_on_current_node()
    if device.type == "cuda":
        return (
            torch.cuda.get_device_properties(get_current_device()).total_memory
            * _GLOBAL_CUDA_MEM_FRACTION
        )
