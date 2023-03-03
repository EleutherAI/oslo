import copy
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.utils import (
    get_parallel_context,
    add_wrapper,
    OsloParallelWrapper,
)
from oslo.torch.nn.parallel.data_parallel._reducer import Reducer
from oslo.torch.nn.parallel.data_parallel._utils import is_ddp_ignored


def free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


def _cast_float(args, dtype: torch.dtype):
    if isinstance(args, torch.Tensor) and torch.is_floating_point(args):
        args = args.to(dtype)
    elif isinstance(args, (list, tuple)):
        args = type(args)(_cast_float(t, dtype) for t in args)
    elif isinstance(args, dict):
        args = {k: _cast_float(v, dtype) for k, v in args.items()}
    return args


def DistributedDataParallel(
    module: nn.Module,
    parallel_context: ParallelContext,
    bucket_cap_mb: int = 25,
    rebuild_bucket: bool = True,
):
    ddp = _DistributedDataParallel(
        module=module,
        parallel_context=parallel_context,
        bucket_cap_mb=bucket_cap_mb,
        rebuild_bucket=rebuild_bucket,
    )

    add_wrapper(
        module,
        mode=ParallelMode.DATA,
        wrapper=ddp,
        parallel_context=parallel_context,
    )
    return module


class _DistributedDataParallel(OsloParallelWrapper):
    """Distributed data parallel wrapper for Oslo.
    Example:
        >>> from oslo.torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = torch.nn.Linear(20, 1)
        >>> model = DDP(model, parallel_context)
        >>> olso.ready(model, parallel_context)
        >>> logits = model(x)
        >>> loss = criterion(logits, labels)
        >>> loss.backward()
    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process group object
    """

    def __init__(
        self,
        module: torch.nn.Module,
        parallel_context: ParallelContext = None,
        bucket_cap_mb: int = 25,
        rebuild_bucket: bool = True,
    ) -> None:
        super().__init__(parallelism_priority=99)
        self.module = module
        self.module.zero_grad = self.zero_grad
        self.module_forward = module.forward

        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        assert parallel_context
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.dp_world_size = self.parallel_context.get_world_size(ParallelMode.DATA)

        self.reducer = Reducer(bucket_cap_mb)
        self.rebuild_bucket = rebuild_bucket
        last_backward = True
        for p in module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                p.register_hook(
                    partial(self.grad_handle, p, last_backward=last_backward)
                )
                last_backward = False

    def parallelize(self):
        self.forward = copy.copy(self.module.forward)

    def _backward(self):
        with torch.cuda.stream(self.comm_stream):
            self.reducer.flush()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if self.rebuild_bucket:
            self.reducer.free()
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.grad.device.type != "cpu":
                p.grad = p._saved_grad

    def grad_handle(self, p, grad, last_backward=False):
        if grad.device.type != "cpu":
            empty_grad = torch.empty_like(grad)
            free_storage(empty_grad)
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    self.reducer.all_reduce_async(
                        grad,
                        group=self.parallel_context.get_group(ParallelMode.DATA),
                        callback_fn=partial(self._save_grad, p),
                    )
                grad.record_stream(self.comm_stream)
            else:
                _DistributedDataParallel._save_grad(p, grad)

            if last_backward:
                Variable._execution_engine.queue_callback(self._backward)

            return empty_grad

        else:
            # TODO(jiaruifang) fixme
            # self.process_group.set_cpu_groups()  # TODO
            # dist.all_reduce(
            #     grad, group=self.process_group.cpu_dp_process_group()
            # )  # TODO
            # return grad
            raise NotImplementedError

    @staticmethod
    def _save_grad(p, grad):
        if hasattr(p, "_saved_grad"):
            p._saved_grad.add_(grad)
        else:
            p._saved_grad = grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        super().zero_grad(set_to_none=True)
        for p in self.module.parameters():
            if getattr(p, "_saved_grad", None) is not None:
                if set_to_none:
                    p._saved_grad = None
                else:
                    if p._saved_grad.grad_fn is not None:
                        p._saved_grad.detach_()
                    else:
                        p._saved_grad.requires_grad_(False)
                    p._saved_grad.zero_()
