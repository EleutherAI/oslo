from contextlib import contextmanager

import torch
import torch.distributed as dist
from numpy import prod

from .distributed_spec import DistributedSpec

from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed.parallel_context import ParallelContext

from typing import Callable


def floor_divide(numerator, denominator):
    """Only allow exact division.

    Args:
        numerator (int): Numerator of the division.
        denominator (int): Denominator of the division.

    Returns:
        int: the result of exact division.
    """
    assert denominator != 0, "denominator can not be zero"
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )
    return numerator // denominator


class TransformDistributedSpec(torch.autograd.Function):
    """
    TransformDistributedSpec is a custom autograd function for transforming a tensor's distributed specification
    during the forward and backward passes.

    This function works with the provided forward and backward transformation functions to ensure correct
    distributed specifications are used in both forward and backward passes.

    Args:
        ctx (torch.autograd.function.FunctionCtx): The context for the autograd function.
        tensor (torch.Tensor): The input tensor to be transformed.
        old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
        dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
        parallel_context (ParallelContext): The parallel context for the distributed training.
        forward_trans_func (Callable): The forward transformation function to apply during the forward pass.
        backward_trans_func (Callable): The backward transformation function to apply during the backward pass.

    Returns:
        torch.Tensor: The transformed tensor with the new distributed specification.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
        forward_trans_func: Callable,
        backward_trans_func: Callable,
    ) -> torch.Tensor:
        ctx.old_dist_spec = old_dist_spec
        ctx.dist_spec = dist_spec
        ctx.backward_trans_func = backward_trans_func
        ctx.parallel_context = parallel_context
        return forward_trans_func(tensor, old_dist_spec, dist_spec, parallel_context)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_outputs):
        return (
            ctx.backward_trans_func(
                grad_outputs, ctx.dist_spec, ctx.old_dist_spec, ctx.parallel_context
            ),
            None,
            None,
            None,
            None,
            None,
        )


class DistributedSpecManager:
    """
    DistributedSpecManager is a class that provides methods to handle the transformation
    of tensors with different distributed specifications. It enables the conversion of
    tensors from replicated to sharded form, and vice versa, or between different sharded
    forms, while handling the autograd function.

    This class provides static methods for various distributed specification transformation
    scenarios, and handles the forward and backward transformation functions for autograd
    compatibility. It also supports disabling the autograd function with a context manager.

    Attributes:
        _use_autograd_function (bool): Determines whether to use the autograd function or not.
    """

    _use_autograd_function: bool = True

    @staticmethod
    def _sanity_check(
        old_dist_spec: DistributedSpec, dist_spec: DistributedSpec
    ) -> None:
        pass

    @staticmethod
    def _shard_as(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """_shard_as: shard the tensor w.r.t a distributed specification.
        Assuming the tensor passed in is a global (replicated) tensor.
        Args:
            tensor (torch.Tensor): a global (replicated) tensor before shard
            dist_spec (DistributedSpec): the distributed spec. to be sharded as.
            parallel_context (ParallelContext): the parallel context object.
        Returns:
            torch.Tensor: a torch tensor after sharded.
        """
        assert (
            old_dist_spec.placement.value == "r"
        ), f"The old_dist_spec of DistributedSpecManager._shard_as must be REPLICATE!"
        DistributedSpecManager._sanity_check(old_dist_spec, dist_spec)

        chunk = tensor
        idx = parallel_context.get_ranks_in_group(ParallelMode.TENSOR)
        num_parts = prod(dist_spec.num_partitions)
        for i, dim in enumerate(dist_spec.dims):
            num_parts //= dist_spec.num_partitions[i]

            chunk_size = floor_divide(tensor.size(dim), dist_spec.num_partitions[i])
            chunk = chunk.narrow(dim, idx // num_parts * chunk_size, chunk_size)
            idx %= num_parts
        return chunk.clone().detach().contiguous()

    @staticmethod
    def _gather(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """_gather gather sharded tensors to a replicated one.
        Args:
            tensor (torch.Tensor): a shared torch tensor
            old_dist_spec (DistributedSpec): the distributed spec. of the tensor.
            parallel_context (ParallelContext): the parallel context object.
        Returns:
            torch.Tensor: a replicated tensor.
        """
        assert (
            old_dist_spec.placement.value == "s"
        ), f"The old_dist_spec of DistributedSpecManager._gather must be SHARD!"
        is_cpu_tensor = False
        if tensor.device.type == "cpu":
            # pytorch lower than 1.11 dose not support gather a cpu tensor.
            # Therefore, we transfer tensor to GPU before gather.
            saved_dev = tensor.device
            tensor.data = tensor.data.cuda()
            is_cpu_tensor = True

        buffer = [
            torch.empty_like(tensor)
            for _ in range(parallel_context.get_world_size(ParallelMode.TENSOR))
        ]
        assert tensor.device.type == "cuda"
        dist.all_gather(
            buffer, tensor, group=parallel_context.get_group(ParallelMode.TENSOR)
        )
        for i in range(len(old_dist_spec.dims) - 1, -1, -1):
            new_buffer = []
            dim = old_dist_spec.dims[i]
            num_parts = old_dist_spec.num_partitions[i]
            for start in range(0, len(buffer), num_parts):
                new_buffer.append(torch.cat(buffer[start : start + num_parts], dim))
            buffer = new_buffer
        assert len(buffer) == 1

        if is_cpu_tensor:
            buffer[0].data = buffer[0].data.to(saved_dev)
        return buffer[0]

    @staticmethod
    def _all_to_all(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """
        Performs an all-to-all communication among the processes to rearrange the tensor data.

        Args:
            tensor (torch.Tensor): The input tensor to be rearranged.
            old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
            dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
            parallel_context (ParallelContext): The parallel context for the distributed training.

        Returns:
            torch.Tensor: The rearranged tensor after all-to-all communication.
        """
        world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        if world_size == 1:
            return tensor

        assert tensor.device.type == "cuda", (
            "Currently, only CUDA Tensor with NCCL backend is supported for the requested AlltoAll "
            f"collective function, however, we got {tensor.device.type} device"
        )

        gather_dim = old_dist_spec.dims[0]
        scatter_dim = dist_spec.dims[0]
        shapes = list(tensor.shape)
        scattered_dim_size = shapes[scatter_dim] // world_size
        gathered_dim_size = shapes[gather_dim] * world_size
        shapes[scatter_dim] = scattered_dim_size

        scatter_list = [
            t.contiguous() for t in torch.tensor_split(tensor, world_size, scatter_dim)
        ]
        gather_list = [
            torch.empty(*shapes, dtype=tensor.dtype, device=tensor.device)
            for _ in range(world_size)
        ]
        dist.all_to_all(
            gather_list,
            scatter_list,
            group=parallel_context.get_group(ParallelMode.TENSOR),
        )

        output_ = torch.cat(gather_list, dim=gather_dim).contiguous()
        assert (
            output_.shape[scatter_dim] == scattered_dim_size
            and output_.shape[gather_dim] == gathered_dim_size
        )
        return output_

    @staticmethod
    def _r2r(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """
        Handles the transformation between two replicated distributed specifications.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.
            old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
            dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
            parallel_context (ParallelContext): The parallel context for the distributed training.

        Returns:
            torch.Tensor: The transformed tensor with the new distributed specification.
        """
        DistributedSpecManager._sanity_check(old_dist_spec, dist_spec)
        return tensor

    @staticmethod
    def _r2s(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """
        Handles the transformation from a replicated distributed specification to a sharded one.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.
            old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
            dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
            parallel_context (ParallelContext): The parallel context for the distributed training.

        Returns:
            torch.Tensor: The transformed tensor with the new distributed specification.
        """
        DistributedSpecManager._sanity_check(old_dist_spec, dist_spec)
        return DistributedSpecManager._shard_as(
            tensor, old_dist_spec, dist_spec, parallel_context
        )

    @staticmethod
    def _s2r(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """
        Handles the transformation from a sharded distributed specification to a replicated one.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.
            old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
            dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
            parallel_context (ParallelContext): The parallel context for the distributed training.

        Returns:
            torch.Tensor: The transformed tensor with the new distributed specification.
        """
        DistributedSpecManager._sanity_check(old_dist_spec, dist_spec)
        return DistributedSpecManager._gather(tensor, old_dist_spec, parallel_context)

    @staticmethod
    def _s2s(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """
        Handles the transformation between two sharded distributed specifications.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.
            old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
            dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
            parallel_context (ParallelContext): The parallel context for the distributed training.

        Returns:
            torch.Tensor: The transformed tensor with the new distributed specification.
        """
        DistributedSpecManager._sanity_check(old_dist_spec, dist_spec)
        if old_dist_spec == dist_spec:
            return tensor
        if len(old_dist_spec.dims) == 1 and len(dist_spec.dims) == 1:
            # use all-to-all to save memory
            return DistributedSpecManager._all_to_all(
                tensor, old_dist_spec, dist_spec, parallel_context
            )
        tensor = DistributedSpecManager._gather(tensor, old_dist_spec, parallel_context)
        return DistributedSpecManager._shard_as(
            tensor, old_dist_spec, dist_spec, parallel_context
        )

    @staticmethod
    def handle_trans_spec(
        tensor: torch.Tensor,
        old_dist_spec: DistributedSpec,
        dist_spec: DistributedSpec,
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
        """
        Handles the transformation of a tensor's distributed specification.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.
            old_dist_spec (DistributedSpec): The original distributed specification of the tensor.
            dist_spec (DistributedSpec): The target distributed specification to transform the tensor.
            parallel_context (ParallelContext): The parallel context for the distributed training.

        Returns:
            torch.Tensor: The transformed tensor with the new distributed specification.
        """
        assert isinstance(
            old_dist_spec, DistributedSpec
        ), f"{type(old_dist_spec)} should be DistributedSpec"
        assert isinstance(
            dist_spec, DistributedSpec
        ), f"{type(dist_spec)} should be DistributedSpec"
        forward_trans_handle = getattr(
            DistributedSpecManager,
            f"_{old_dist_spec.placement.value}2{dist_spec.placement.value}",
        )
        if not DistributedSpecManager._use_autograd_function:
            return forward_trans_handle(
                tensor, old_dist_spec, dist_spec, parallel_context
            )
        backward_trans_handle = getattr(
            DistributedSpecManager,
            f"_{dist_spec.placement.value}2{old_dist_spec.placement.value}",
        )
        return TransformDistributedSpec.apply(
            tensor,
            old_dist_spec,
            dist_spec,
            parallel_context,
            forward_trans_handle,
            backward_trans_handle,
        )

    @staticmethod
    @contextmanager
    def no_grad():
        """
        Context manager to temporarily disable the autograd function for the transformations
        performed by the DistributedSpecManager.

        Example usage:

        with DistributedSpecManager.no_grad():
            transformed_tensor = DistributedSpecManager.handle_trans_spec(...)
        """
        try:
            DistributedSpecManager._use_autograd_function = False
            yield
        finally:
            DistributedSpecManager._use_autograd_function = True
