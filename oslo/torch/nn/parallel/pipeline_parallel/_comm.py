import time
from typing import Any
from queue import Queue

import torch
from torch.distributed import rpc

from oslo.torch.distributed import ParallelContext, ParallelMode

# from oslo.torch.distributed.nn.functional import (
#     send as _send,
#     recv as _recv,
# )
from ._sync import register_job, _RECV_QUEUES, _HANDSHAKE_QUEUES
from ._job import Job, Backward, FinalJob, HandshakeRequest, HandshakeResponse, Metadata


KEY_NAME = "__KEY__"
VALUE_NAME = "__VALUE__"
META_NAME = "__META__"


infos = dict()

_DEBUG = False


def enqueue_forward_ready_notice(rank):
    infos["FORWARD_READY_NOTICE"].put(
        rank
    )


def enqueue_forward_start_notice():
    infos["FORWARD_START_NOTICE"].put(
        "START"
    )


def notify_last_backward_done():
    infos["LAST_BACKWARD_NOTICE"].put(
        "FINISHED"
    )


def enqueue_batch_finished_notice(rank):
    print(f"RANK {torch.distributed.get_rank()} | NOTIFIED!!")

    infos["BATCH_FINISHED_NOTICE"].put(
        rank
    )


def enqueue_forward_finished_notice():
    infos["FORWARD_FINISHED_NOTICE"].put(
        "FINISHED"
    )


def enqueue_result(ind, data):
    infos["OUT_QUEUE"].put(
        (ind, data)
    )


def enqueue_backward_job(is_final, meta, unique_key, *grad_outputs):
    job = Backward(
        tensors=grad_outputs,
        unique_key=unique_key,
        stub=None,
        meta=meta,
    )

    # if True:
    #     print(f"RANK {torch.distributed.get_rank()} | enqueue backward {unique_key} from {meta.src} to {meta.dst}")

    register_job(job)


def enqueue_handshake_req(src, dst, queue_ind, recv_key):
    job = HandshakeRequest(
        src=src,
        dst=dst,
        qind=queue_ind,
        recv_key=recv_key,
    )

    register_job(job)


def enqueue_handshake_resp(src, dst, queue_ind, recv_key):
    job = HandshakeResponse(
        src=src,
        dst=dst,
        qind=queue_ind,
        recv_key=recv_key,
    )

    register_job(job)


def enqueue_data(data, recv_key):
    if _DEBUG:
        print(f"RANK {torch.distributed.get_rank()} | keys in _RECV_QUEUES: {_RECV_QUEUES.keys()}")

    _RECV_QUEUES[recv_key].put(data)


def _send(
        data,
        src_rank,
        dst_rank,
        recv_key,
):
    rpc.rpc_sync(
        to=f"RPC_WORKER_{dst_rank}",
        func=enqueue_data,
        args=(data, recv_key),
    )


def _recv(recv_key):
    while _RECV_QUEUES[recv_key].empty():
        time.sleep(0.05)

    data = _RECV_QUEUES[recv_key].get()
    return data


def send_data(
    data: Any,
    src_rank: int,
    dst_rank: int,
    parallel_mode: ParallelMode = ParallelMode.PIPELINE,
):
    parallel_context = infos["PC"]

    # need to get global rank of dst device for rpc.
    # assumes that all ranks except `parallel_mode`
    # are same between src device and dst device
    ranks = parallel_context.get_local_ranks()
    ranks[parallel_mode] = dst_rank

    global_dst_rank = parallel_context.ranks2device(ranks)

    recv_key = data[KEY_NAME]
    q = Queue()
    _HANDSHAKE_QUEUES[recv_key] = q

    rpc.rpc_sync(
        # to=f"RPC_WORKER_{global_dst_rank}",   # TODO; how to find global dst?
        to=f"RPC_WORKER_{dst_rank}",
        func=enqueue_handshake_req,
        args=(src_rank, dst_rank, recv_key, recv_key),
    )

    while q.empty():
        time.sleep(0.05)

    r = q.get()

    torch.cuda.set_device(
        torch.distributed.get_rank()
    )

    if _DEBUG:
        print(f"RANK {torch.distributed.get_rank()} | send? {r} {data[KEY_NAME]} ({src_rank} to {dst_rank})")

    _send(
        data,
        src_rank,
        dst_rank,
        recv_key=recv_key,
    )

    # print(f"{'_'.join(map(str, recv_key))}_send")
    # torch.save(data, f"tmp/{'_'.join(map(str, recv_key))}_send.pkl")

    del _HANDSHAKE_QUEUES[recv_key]
    del q


def recv_data(
        src_rank,
        dst_rank,
        recv_key,
):
    parallel_context = infos["PC"]

    if _DEBUG:
        print(f"RANK {torch.distributed.get_rank()} | prepare recv: from {src_rank} to {dst_rank}, {recv_key}")

    torch.cuda.set_device(
        torch.distributed.get_rank()
    )

    data = _recv(recv_key)

    # print(f"{'_'.join(map(str, recv_key))}_recv")
    # torch.save(data, f"tmp/{'_'.join(map(str, recv_key))}_recv.pkl")

    if _DEBUG:
        print(f"RANK {torch.distributed.get_rank()} | recv! {data[KEY_NAME]}")

    unique_key = data[KEY_NAME]
    value = data[VALUE_NAME]
    metadata = data[META_NAME]

    # make a Job object
    meta = Metadata(
        is_request=metadata["is_request"],
        is_first=metadata["is_first"],
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
        stub=value["stub"],
        meta=meta,
    )

    # report to checker
    # TODO;

    # register job
    register_job(job)
