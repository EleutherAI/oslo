import torch
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.distributed import rpc

from oslo.torch.nn.parallel.pipeline_parallel._buffers import _ACTIVATIONS

_FORWARD_MARKER = set()

_LOCAL_BACKWARD_DONE = False

_NUM_BACKWARD_DONE = 0


def add_forward_marker(mark):
    _FORWARD_MARKER.add(mark)
    # print(f'ADD MARKER: {torch.distributed.get_rank()=}, {_FORWARD_MARKER=}')


def remove_forward_marker(mark):
    _FORWARD_MARKER.remove(mark)
    # print(f'REMOVE MARKER: {torch.distributed.get_rank()=}, {_FORWARD_MARKER=}')


def len_forward_marker():
    # print(f'{torch.distributed.get_rank()=}, {_FORWARD_MARKER=}')
    return len(_FORWARD_MARKER)


def increase_num_backward_done():
    global _NUM_BACKWARD_DONE
    _NUM_BACKWARD_DONE += 1


def get_num_backward_done():
    global _NUM_BACKWARD_DONE
    return _NUM_BACKWARD_DONE


def reset_num_backward_done():
    global _NUM_BACKWARD_DONE, _LOCAL_BACKWARD_DONE
    _NUM_BACKWARD_DONE = 0
    _LOCAL_BACKWARD_DONE = False


def launch_remote_backward(unique_key, *grad_outputs):
    activation = _ACTIVATIONS.pop(unique_key, [])   # TODO; is this safe?

    # TODO; HF output...
    if isinstance(activation, dict):
        activation = tuple(activation.values())

    # TODO; some activations are not tuple... WHY?
    if not isinstance(activation, tuple):
        activation = (activation, )

    # print(f'{unique_key=}, {type(activation)=}, {len(activation)=}, {len(grad_outputs)=}')

    # TODO; some output contains tuple of tuple..
    #   better way to deal with this?
    new_act = []
    new_grad = []
    for act, grad in zip(activation, grad_outputs):
        if act is not None and grad is not None and act.requires_grad:
            new_act.append(act)
            new_grad.append(grad)

    torch.autograd.backward(tuple(new_act), tuple(new_grad))
    remove_forward_marker(unique_key)


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
        ctx.num_nones = 2 + len(args)  # counting req

        # mark
        # TODO; can we do this before remote_forward
        #  without rpc call?
        rpc.rpc_sync(to=to, func=add_forward_marker, args=(unique_key,))

        return args

    @staticmethod
    @custom_bwd
    @rpc.functions.async_execution
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
