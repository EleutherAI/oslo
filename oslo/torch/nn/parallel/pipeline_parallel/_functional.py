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
from oslo.torch.nn.parallel.pipeline_parallel._sync import (
    register_job_requires_backward,
    notify_backward_job_done,
)


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
