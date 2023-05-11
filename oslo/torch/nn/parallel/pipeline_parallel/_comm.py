from typing import Any
from queue import Queue

import torch
from torch.distributed import rpc

from oslo.torch.distributed import ParallelMode

from ._types import CommunicationInformation
from ._sync import sleep, register_job, NOTIFICATIONS, QUEUES
from ._job import Job, Backward, HandshakeRequest, HandshakeResponse, Metadata


COMM_INFO = CommunicationInformation()

KEY_NAME = "__KEY__"
VALUE_NAME = "__VALUE__"
META_NAME = "__META__"


def enqueue_forward_ready_notice(rank):
    NOTIFICATIONS.FORWARD_READY_NOTICE.put(rank)


def enqueue_forward_start_notice():
    NOTIFICATIONS.FORWARD_START_NOTICE.put("START")


def notify_last_backward_done():
    NOTIFICATIONS.LAST_BACKWARD_NOTICE.put("FINISHED")


def enqueue_batch_finished_notice(rank):
    NOTIFICATIONS.BATCH_FINISHED_NOTICE.put(rank)


def enqueue_forward_finished_notice():
    NOTIFICATIONS.FORWARD_FINISHED_NOTICE.put("FINISHED")


def enqueue_result(ind, data):
    NOTIFICATIONS.OUTPUT.put(
        (ind, data)
    )


def enqueue_backward_job(meta, unique_key, *grad_outputs):
    job = Backward(
        tensors=grad_outputs,
        unique_key=unique_key,
        stub=None,
        meta=meta,
    )

    register_job(job)


def enqueue_handshake_req(src, dst, recv_key):
    job = HandshakeRequest(
        src=src,
        dst=dst,
        recv_key=recv_key,
    )

    register_job(job)


def enqueue_handshake_resp(src, dst, recv_key):
    job = HandshakeResponse(
        src=src,
        dst=dst,
        recv_key=recv_key,
    )

    register_job(job)


def enqueue_data(data, recv_key):
    QUEUES.RECV_QUEUES[recv_key].put(data)


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
    while QUEUES.RECV_QUEUES[recv_key].empty():
        sleep()

    data = QUEUES.RECV_QUEUES[recv_key].get()
    return data


def send_data(
    data: Any,
    src_rank: int,
    dst_rank: int,
):
    parallel_context = COMM_INFO.PARALLEL_CONTEXT

    # need to get global rank of dst device for rpc.
    # assumes that all ranks except `parallel_mode`
    # are same between src device and dst device
    ranks = parallel_context.get_local_ranks()
    ranks[ParallelMode.PIPELINE] = dst_rank

    global_dst_rank = parallel_context.ranks2device(ranks)

    recv_key = data[KEY_NAME]
    q = Queue()
    QUEUES.HANDSHAKE_QUEUES[recv_key] = q

    rpc.rpc_sync(
        # to=f"RPC_WORKER_{global_dst_rank}",   # TODO; how to find global dst?
        to=f"RPC_WORKER_{dst_rank}",
        func=enqueue_handshake_req,
        args=(src_rank, dst_rank, recv_key),
    )

    while q.empty():
        sleep()

    r = q.get()

    torch.cuda.set_device(
        torch.distributed.get_rank()
    )

    _send(
        data,
        src_rank,
        dst_rank,
        recv_key=recv_key,
    )

    del QUEUES.HANDSHAKE_QUEUES[recv_key]
    del q


def recv_data(
        src_rank,
        dst_rank,
        recv_key,
):
    parallel_context = COMM_INFO.PARALLEL_CONTEXT

    torch.cuda.set_device(
        torch.distributed.get_rank()
    )

    data = _recv(recv_key)

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
        stub=value["stub"],
        meta=meta,
    )

    # report to checker
    # TODO;

    # register job
    register_job(job)
