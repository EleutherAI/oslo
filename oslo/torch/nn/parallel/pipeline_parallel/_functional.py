import torch
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.distributed import rpc

from oslo.torch.distributed import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._buffers import (
    get_original_forward_function,
    save_activation,
    pop_activation,
)
from oslo.torch.nn.parallel.pipeline_parallel._messages import (
    pack_tensor_stub,
    unpack_tensor_stub,
)
from oslo.torch.nn.parallel.pipeline_parallel._sync import (
    register_job_requires_backward,
    notify_backward_job_done,
    _RECV_QUEUES,
)
from oslo.torch.nn.parallel.pipeline_parallel._job import (
    Job, JobInitialization, Metadata
)
from oslo.torch.nn.parallel.pipeline_parallel._comm import (
    send_data, KEY_NAME, VALUE_NAME, META_NAME,
    infos
)


funcs = dict()


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
        parallel_context,
        parallel_mode=ParallelMode.PIPELINE,
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
        parallel_mode=parallel_mode
    )

    # register queue for recv
    _RECV_QUEUES[unique_key] = queue


def send_results(
        src,
        dst,
        data,
        parallel_context,
        parallel_mode=ParallelMode.PIPELINE,
):
    # send
    send_data(
        data=data,
        src_rank=src,
        dst_rank=dst,
        parallel_mode=parallel_mode
    )


def start_job(job):

    if isinstance(job, Job):
        tensors = job.tensors
        unique_key = job.unique_key
        stub = job.stub
        meta: Metadata = job.meta

        if meta.is_request:
            if meta.is_forward:
                result_stub, tensors = launch_forward(
                    meta.func_name,
                    unique_key,
                    meta.is_training,
                    meta.is_grad_enabled,
                    stub,
                    *tensors,
                )

                # reverse
                src, dst = meta.dst, meta.src

                # make data
                data = dict()
                value = {
                    "stub": result_stub,
                    "tensors": tensors,
                }
                new_meta = {
                    "is_request": False,
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

                # return; send to other device
                send_fn = funcs["send"]
                send_fn(
                    data=data,
                    src=src,
                    dst=dst,
                )

            else:   # backward
                pass

        # data receive
        else:

            # print("AHAHAHAHAH")

            queue = _RECV_QUEUES[unique_key]
            queue.put(
                (stub, tensors)
            )

    elif isinstance(job, JobInitialization):
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


def launch_forward(
        func_name,
        unique_key,
        is_training,
        is_grad_enabled,
        stub,
        *tensors,
):
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


def remote_module_forward(
    caller,
    location,
    unique_key,
    args_stub,
    kwargs_stub,
    requires_redirection,
    is_training,
    is_grad_enabled,
    *tensors
):
    if requires_redirection and is_training and is_grad_enabled:
        # prepare backward redirection to caller
        tensors = apply_backward_redirection(
            caller,
            unique_key,
            *tensors,
        )

    (args, kwargs), _ = unpack_tensor_stub([args_stub, kwargs_stub], tensors)

    forward_fn = get_original_forward_function(location)
    with torch.set_grad_enabled(is_grad_enabled):
        result = forward_fn(*args, **kwargs)

    result_stub, tensors = pack_tensor_stub(result, [])
    need_activation_save = (
        any([t.requires_grad for t in tensors]) and is_training and is_grad_enabled
    )
    if need_activation_save:
        save_activation(unique_key, tensors)

    return result_stub, tensors, need_activation_save


def launch_remote_backward(unique_key, *grad_outputs):
    activation = pop_activation(unique_key)

    new_act = []
    new_grad = []
    for act, grad in zip(activation, grad_outputs):
        if act is not None and grad is not None and act.requires_grad:
            new_act.append(act)
            new_grad.append(grad)

    if len(new_act) > 0 and len(new_grad) > 0:
        torch.autograd.backward(tuple(new_act), tuple(new_grad))
        notify_backward_job_done(unique_key)


# why
#  forward(ctx, req, *args, **kwargs)
#  ...
#  return args, kwargs
#  does not work???
#
#  because that is the design of Pytorch
#  see: github.com/pytorch/pytorch/issues/16940
#
# based on https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/pipe/rpc.py#L53
class _PipeBackwardRedirection(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, to, unique_key, *args):
        ctx.to = to
        ctx.unique_key = unique_key
        ctx.num_nones = 2 + len(args)

        # mark
        # TODO; can we do this before remote_forward
        #  without rpc call?
        rpc.rpc_sync(to=to, func=register_job_requires_backward, args=(unique_key,))

        return args

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outputs):
        to = ctx.to
        unique_key = ctx.unique_key

        rpc.rpc_async(
            to=to,
            func=launch_remote_backward,
            args=(unique_key, *grad_outputs),
        )

        return (None,) * ctx.num_nones


def apply_backward_redirection(to, unique_key, *args):
    return _PipeBackwardRedirection.apply(to, unique_key, *args)
