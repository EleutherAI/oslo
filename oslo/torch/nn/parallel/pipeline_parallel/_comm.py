from typing import Any

from torch.distributed import rpc

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch.distributed.nn.functional import (
    send as _send,
    recv as _recv,
)
from ._sync import register_job
from ._job import Job, Metadata


KEY_NAME = "__KEY__"
VALUE_NAME = "__VALUE__"
META_NAME = "__META__"


def send_data(
    data: Any,
    src_rank: int,
    dst_rank: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode = ParallelMode.PIPELINE,
):
    # need to get global rank of dst device for rpc.
    # assumes that all ranks except `parallel_mode`
    # are same between src device and dst device
    ranks = parallel_context.get_local_ranks()
    ranks[parallel_mode] = dst_rank

    global_dst_rank = parallel_context.ranks2device(ranks)
    fut = rpc.rpc_async(
        to=f"RPC_WORKER_{global_dst_rank}",
        func=recv_data,
        args=(src_rank, dst_rank, parallel_context, parallel_mode),
    )

    _send(
        data,
        src_rank,
        dst_rank,
        parallel_context,
        parallel_mode,
    )

    fut.wait()


def recv_data(
    src_rank: int,
    dst_rank: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode = ParallelMode.PIPELINE,
):

    # TODO; acquire lock

    data = _recv(
        src_rank,
        dst_rank,
        parallel_context,
        parallel_mode,
    )

    yield

    unique_key = data[KEY_NAME]
    value = data[VALUE_NAME]
    metadata = data[META_NAME]

    # make a Job object
    meta = Metadata(
        is_request=metadata["is_request"],
        is_forward=metadata["is_forward"],
        is_training=metadata["is_training"],
        is_grad_enabled=metadata["is_grad_enabled"],
        is_fp16=metadata["is_fp16"],
        func_name=metadata["func_name"],
        src=metadata["src"],
        dst=metadata["dst"],
    )
    job = Job(
        unique_key=unique_key,
        tensors=value["tensors"],
        args_stub=value["args_stub"],
        kwargs_stub=value["kwargs_stub"],
        meta=meta,
    )

    # report to checker
    # TODO;

    # register job
    register_job(job)
