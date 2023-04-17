import copy
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.utils import (
    get_parallel_context,
    add_wrapper,
    OsloParallelWrapper,
)
from oslo.torch.nn.parallel.data_parallel._reducer import Reducer
from oslo.torch.nn.parallel.data_parallel._utils import (
    free_storage,
    is_ddp_ignored,
    DistributedBackwardFunction,
)


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
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> oslo.ready(model, parallel_context)
        >>> model.zero_grad()
        >>> logits = model(x)
        >>> loss = criterion(logits, labels)
        >>> loss.backward()
        >>> optimizer.step()
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

        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        assert parallel_context
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.dp_world_size = self.parallel_context.get_world_size(ParallelMode.DATA)

        self.reducer = Reducer(bucket_cap_mb)
        self.rebuild_bucket = rebuild_bucket

        for p in module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle, p))

    def parallelize(self):
        self._forward = copy.copy(self.module.forward)
        self.module.zero_grad = self.zero_grad

    def forward(self, *args, **kwargs):
        # inputs must be `torch.Tensor` or collections that contain `torch.Tensor`
        inputs = self._forward(*args, **kwargs)
        if isinstance(inputs, dict):
            return type(inputs)(
                {
                    k: v
                    for k, v in zip(
                        outputs.keys(),
                        DistributedBackwardFunction.apply(self, *outputs.values()),
                    )
                }
            )

        single_output = isinstance(inputs, torch.Tensor)
        if single_output:
            inputs = (inputs,)

        outputs = _DistributedBackwardFunction.apply(self, *inputs)
        return outputs[0] if single_output else outputs

    def _pre_backward(self):
        pass

    def _post_backward(self):
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

    def grad_handle(self, p, grad):
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

            return empty_grad

        else:
            # You must model.to('cpu') after oslo.ready() to use cpu.
            dist.all_reduce(
                grad, group=self.parallel_context.get_cpu_group(ParallelMode.DATA)
            )
            return grad

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
