from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from oslo.torch.distributed import ParallelMode, ParallelContext
from oslo.torch.distributed.nn.functional import (
    all_gather,
    all_reduce,
    reduce_scatter,
    scatter,
)


class _BroadcastTensor1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, parallel_context: ParallelContext):
        if ctx:
            ctx.parallel_context = parallel_context
        return inputs

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        parallel_context = ctx.parallel_context
        return (
            all_reduce(
                grad,
                on_cpu=str(grad.device) == "cpu",
                async_op=False,
                parallel_context=parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
        )


class _ReduceTensor1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, parallel_context: ParallelContext):
        return all_reduce(
            inputs,
            on_cpu=str(inputs.device) == "cpu",
            async_op=False,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return grad, None


class _GatherTensor1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_context: ParallelContext):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
        return all_gather(
            inputs,
            dim=dim,
            on_cpu=str(inputs.device) == "cpu",
            async_op=False,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return (
            scatter(
                grad,
                dim=ctx.dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
            None,
        )


class _ScatterTensor1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_context: ParallelContext):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
        return scatter(
            inputs,
            dim=dim,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return (
            all_gather(
                grad,
                dim=ctx.dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
            None,
        )


class _ReduceScatterTensor1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_context: ParallelContext):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
        return reduce_scatter(
            inputs,
            dim,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return (
            all_gather(
                grad,
                dim=ctx.dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
            None,
        )


def broadcast_tensor_1d(inputs: Tensor, parallel_context: ParallelContext):
    return _BroadcastTensor1D.apply(inputs, parallel_context)


def reduce_tensor_1d(inputs: Tensor, parallel_context: ParallelContext):
    return _ReduceTensor1D.apply(inputs, parallel_context)


def gather_tensor_1d(inputs: Tensor, dim: int, parallel_context: ParallelContext):
    return _GatherTensor1D.apply(inputs, dim, parallel_context)


def scatter_tensor_1d(inputs: Tensor, dim: int, parallel_context: ParallelContext):
    return _ScatterTensor1D.apply(inputs, dim, parallel_context)


def reduce_scatter_tensor_1d(
    inputs: Tensor, dim: int, parallel_context: ParallelContext
):
    return _ReduceScatterTensor1D.apply(inputs, dim, parallel_context)


def split_1d(parallel_context, tensor, summa_dim, dim=-1):
    tensor = tensor.chunk(summa_dim, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
    ]
    return tensor


def gather_1d(parallel_context, tensor, summa_dim, dim=-1):
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_1D),
    )
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor
