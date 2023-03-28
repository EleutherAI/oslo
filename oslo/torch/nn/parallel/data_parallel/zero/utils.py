import torch
import torch.distributed as dist

from heterogeneous_manager.heterogeneous_manager import Chunk


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")


def get_temp_total_chunk_on_cuda(chunk: Chunk):
    if chunk.is_gathered:
        return chunk.cuda_global_chunk

    if chunk.cuda_shard is not None:
        shard_temp = chunk.cuda_shard
    else:
        shard_temp = chunk.cpu_shard.to(get_current_device())

    total_temp = torch.zeros(chunk.chunk_size, dtype=chunk.dtype, device=get_current_device())
    gather_list = list(torch.chunk(input=total_temp, chunks=chunk.pg_size, dim=0))
    dist.all_gather(tensor_list=gather_list, tensor=shard_temp, group=chunk.torch_pg)

    return total_temp