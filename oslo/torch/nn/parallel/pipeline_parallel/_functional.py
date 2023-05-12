from queue import Queue

import torch
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.distributed import rpc

from oslo.torch.nn.parallel.pipeline_parallel._buffers import (
    get_original_forward_function,
    save_activation,
    pop_activation,
)
from oslo.torch.nn.parallel.pipeline_parallel._messages import (
    pack_tensor_stub,
    unpack_tensor_stub,
)
from oslo.torch.nn.parallel.pipeline_parallel._sync import QUEUES, sleep
from oslo.torch.nn.parallel.pipeline_parallel._job import (
    Job, Backward, Input, HandshakeRequest, HandshakeResponse, Metadata
)
from oslo.torch.nn.parallel.pipeline_parallel._comm import (
    send_data, recv_data,
    enqueue_handshake_resp,
    enqueue_backward_job,
    notify_last_backward_done,
    KEY_NAME, VALUE_NAME, META_NAME,
    COMM_INFO,
)


_BACKWARD_KEYS = list()


def remote_request(
        src,
        dst,
        unique_key,
        queue,
        args_stub,
        kwargs_stub,
        is_forward,
        is_grad_enabled,
        is_training,
        is_fp16,
        func_name,
        tensors,
):
    # TODO; make the code below efficient
    data = dict()
    value = {
        "stub": [args_stub, kwargs_stub],
        "tensors": tensors,
    }

    meta = {
        "is_request": True,
        "is_forward": is_forward,
        "is_training": is_training,
        "is_grad_enabled": is_grad_enabled,
        "is_fp16": is_fp16,
        "func_name": func_name,
        "src": src,
        "dst": dst,
    }

    data[VALUE_NAME] = value
    data[KEY_NAME] = unique_key
    data[META_NAME] = meta

    # send
    send_data(
        data=data,
        src_rank=src,
        dst_rank=dst,
    )

    # register queue for recv
    QUEUES.RESPONSE_QUEUES[unique_key] = queue


def send_results(
        src,
        dst,
        data,
):
    # send
    send_data(
        data=data,
        src_rank=src,
        dst_rank=dst,
    )


def start_job(job):
    if isinstance(job, (Job, Backward)):
        tensors = job.tensors
        unique_key = job.unique_key
        stub = job.stub
        meta: Metadata = job.meta

        if meta.is_request:
            if meta.is_forward:
                result_stub, tensors = launch_forward(
                    meta.src,   # requested_from
                    meta.dst,   # current rank
                    meta.func_name,
                    unique_key,
                    meta.is_training,
                    meta.is_grad_enabled,
                    stub,
                    *tensors,
                )

                # reverse direction
                src, dst = meta.dst, meta.src

                # make data
                data = dict()
                value = {
                    "stub": result_stub,
                    "tensors": tensors,
                }
                new_meta = {
                    "is_request": False,
                    "is_first": False,
                    "is_forward": meta.is_forward,
                    "is_training": meta.is_training,
                    "is_grad_enabled": meta.is_grad_enabled,
                    "is_fp16": meta.is_fp16,
                    "func_name": meta.func_name,
                    "src": src,
                    "dst": dst,
                }

                data[VALUE_NAME] = value
                data[KEY_NAME] = unique_key
                data[META_NAME] = new_meta

                # TODO; need a handshake?
                # return; send to other device
                send_results(
                    data=data,
                    src=src,
                    dst=dst,
                )

            else:   # backward
                activation = pop_activation(unique_key)
                grad_outputs = tensors

                new_act = []
                new_grad = []

                # infos["LOCK"].acquire()

                # wait for copy stream
                # TODO; does rpc do sync?
                torch.cuda.synchronize(
                    torch.distributed.get_rank()
                )

                for act, grad in zip(activation, grad_outputs):
                    if act is not None and grad is not None and act.requires_grad:
                        new_act.append(act)
                        new_grad.append(grad)

                s = torch.cuda.Stream()

                if len(new_act) > 0 and len(new_grad) > 0:
                    with torch.cuda.stream(s):
                        torch.autograd.backward(tuple(new_act), tuple(new_grad))

                while not s.query():
                    sleep()

                _BACKWARD_KEYS.remove(unique_key)

                del activation, new_act, new_grad
                # TODO; empty cache properly
                torch.cuda.empty_cache()

                if len(_BACKWARD_KEYS) == 0:
                    notify_last_backward_done()

                # infos["LOCK"].release()

        # data receive
        else:
            queue = QUEUES.RESPONSE_QUEUES.pop(unique_key)

            queue.put(
                (stub, tensors)
            )

    elif isinstance(job, Input):
        # TODO; move or make as a function
        fn = job.fn
        is_grad_enabled = job.is_grad_enabled
        kwargs = job.kwargs
        unique_key = job.unique_key
        out_queue = job.out_queue

        # function to launch self.module. needs this
        # function because torch.set_grad_enabled() is
        # thread local.
        with torch.set_grad_enabled(is_grad_enabled):
            result = fn(**kwargs)   # *args is not used

        out_queue.put(
            (unique_key, result)
        )

    elif isinstance(job, HandshakeRequest):
        # notify source rank
        src = job.src
        dst = job.dst

        # create and register a queue for receive
        q = Queue()
        QUEUES.RECV_QUEUES[job.recv_key] = q

        parallel_context = COMM_INFO.PARALLEL_CONTEXT
        global_dst_rank = parallel_context.pp_rank_to_global_rank_for_rpc(
            pp_rank=src,
        )

        rpc.rpc_sync(
            to=f"RPC_WORKER_{global_dst_rank}",
            func=enqueue_handshake_resp,
            args=(src, dst, job.unique_key),  # reverse
        )

        recv_data(
            src,
            dst,
            job.recv_key,
        )

    elif isinstance(job, HandshakeResponse):
        # awake waiting thread
        q = QUEUES.HANDSHAKE_QUEUES[job.unique_key]
        q.put(
            f"okay - {job.unique_key}"
        )


def launch_forward(
        requested_from,
        current_rank,
        func_name,
        unique_key,
        is_training,
        is_grad_enabled,
        stub,
        *tensors,
):
    requires_grad_any = any([t.requires_grad for t in tensors])
    if is_training and is_grad_enabled and requires_grad_any:
        meta = Metadata(
            is_request=True,
            is_forward=False,
            is_training=is_training,
            is_grad_enabled=is_grad_enabled,
            is_fp16=False,  # TODO;
            func_name="",  # dummy
            src=current_rank,
            dst=requested_from,
        )

        tensors = apply_backward_job_enqueue(meta, unique_key, *tensors)

    (args, kwargs), _ = unpack_tensor_stub(stub, tensors)

    forward_fn = get_original_forward_function(func_name)
    with torch.set_grad_enabled(is_grad_enabled):
        result = forward_fn(*args, **kwargs)

    result_stub, tensors = pack_tensor_stub(result, [])

    need_activation_save = (
            any([t.requires_grad for t in tensors]) and is_training and is_grad_enabled
    )
    if need_activation_save:
        save_activation(unique_key, tensors)

    return result_stub, tensors


def add_backward_required_key(key):
    _BACKWARD_KEYS.append(key)


class _BackwardJobEnqueue(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, meta, unique_key, *args):
        ctx.meta = meta
        ctx.unique_key = unique_key
        ctx.num_nones = 2 + len(args)

        # mark; use async since this requires loose sync
        rpc.rpc_async(
            to=meta.dst,
            func=add_backward_required_key,
            args=(unique_key, ),
        )

        return args

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outputs):
        meta = ctx.meta
        unique_key = ctx.unique_key

        rpc.rpc_sync(
            to=meta.dst,
            func=enqueue_backward_job,
            args=(meta, unique_key, *grad_outputs),
        )

        return (None,) * ctx.num_nones


def apply_backward_job_enqueue(meta, unique_key, *args):
    return _BackwardJobEnqueue.apply(meta, unique_key, *args)
