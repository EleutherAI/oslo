import torch
from torch.distributed import rpc
from torch.cuda.amp import custom_fwd, custom_bwd

from ._buffers import _ACTIVATIONS


def launch_remote_backward(unique_key, *grad_outputs):
    activation = _ACTIVATIONS.pop(unique_key)

    # TODO; some output contains tuple of tuple..
    #   better way to deal with this?
    new_act = []
    new_grad = []
    for act, grad in zip(activation, grad_outputs):
        if act is not None and grad is not None and act.requires_grad:
            new_act.append(act)
            new_grad.append(grad)

    # s = torch.cuda.Stream()
    # with torch.cuda.stream(s):
    #     torch.autograd.backward(tuple(new_act), tuple(new_grad))
    # torch.cuda.current_stream().wait_stream(s)

    torch.autograd.backward(tuple(new_act), tuple(new_grad))

    print(f'backward done: {unique_key}')


# TODO; why
#  forward(ctx, req, *args, **kwargs)
#  ...
#  return args, kwargs
#  does not work???
#  ->
#  because that is the design of Pytorch
#  see: github.com/pytorch/pytorch/issues/16940
# based on https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/pipe/rpc.py#L53
class _PipeBackwardRedirection(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, to, unique_key, *args):
        ctx.to = to
        ctx.unique_key = unique_key
        ctx.num_nones = 2 + len(args)   # counting req

        # print(f'forward: {to=}, {unique_key=}')

        return args

    @staticmethod
    @custom_bwd
    @rpc.functions.async_execution
    def backward(ctx, *grad_outputs):
        to = ctx.to
        unique_key = ctx.unique_key

        # print(f'backward: {to=}, {unique_key=}')

        fut = rpc.rpc_async(
            to=to,
            func=launch_remote_backward,
            args=(unique_key, *grad_outputs),
        )
        # TODO; check whether this usage is correct
        #  need to deal nested rpc:
        #   https://pytorch.org/tutorials/intermediate/rpc_async_execution.html
        #   https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution
        # while not fut.done():
        #     time.sleep(0.)

        # fut.then(lambda f: f.wait())

        return (None, ) * ctx.num_nones


def apply_backward_redirection(to, unique_key, *args):
    return _PipeBackwardRedirection.apply(to, unique_key, *args)
