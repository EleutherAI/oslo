# Copyright 2021 HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by EleutherAI on 2023.

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from math import sqrt, inf
import torch
from typing import Optional, Iterable, List, Union
from oslo.torch.distributed import ParallelMode


def is_model_parallel_parameter(p: torch.Tensor) -> bool:
    """
    Check if a parameter is parallel in either Pipeline or Tensor mode.

    Args:
        p (torch.Tensor): Parameter to check.

    Returns:
        bool: True if the parameter is parallel in either mode, False otherwise.
    """
    oslo_parallel = getattr(p, "oslo_parallel", {})
    parallel_modes = [
        ParallelMode.PIPELINE,
        ParallelMode.TENSOR_1D,
        ParallelMode.TENSOR_2D_ROW,
        ParallelMode.TENSOR_2D_COL,
        ParallelMode.TENSOR_2P5D_ROW,
        ParallelMode.TENSOR_2P5D_COL,
        ParallelMode.TENSOR_2P5D_DEP,
        ParallelMode.TENSOR_3D_INPUT,
        ParallelMode.TENSOR_3D_OUTPUT,
    ]
    return any(mode in oslo_parallel for mode in parallel_modes)


def flatten(input_: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Flatten the input tensor into a 1D tensor.

    Args:
        input_ (Iterable[torch.Tensor]): The input tensor to be flattened.

    Returns:
        torch.Tensor: The flattened tensor.
    """
    return _flatten_dense_tensors(input_)


def unflatten(flat: torch.Tensor, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Unflatten the flattened tensor back to its original shape.

    Args:
        flat (torch.Tensor): The flattened tensor.
        tensors (Iterable[torch.Tensor]): The original tensor shape.

    Returns:
        torch.Tensor: The unflattened tensor.
    """
    return _unflatten_dense_tensors(flat, tensors)


def calculate_global_norm_from_list(norm_list: List[float]) -> float:
    """
    Compute the total global norm from a list of norms.
    Args:
        norm_list (List[float]): List of norms to compute the total global norm from.
    Returns:
        float: Total global norm.
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    return sqrt(total_norm)


def reduce_tensor_dp_group(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    dst_local_rank: Optional[int] = None,
    dst_global_rank: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Reduce a tensor in the data parallel process group.

    Args:
        tensor (torch.Tensor): Tensor to reduce/all-reduce.
        dtype (Optional[torch.dtype]): Data type used in communication.
        dst_local_rank (Optional[int]): Local rank of the destination node.
        dst_global_rank (Optional[int]): Global rank of the destination node.
        group (Optional[dist.ProcessGroup]): Process group for the reduction.

    Returns:
        tensor (torch.Tensor): The reduced tensor.
    """

    # use the original dtype
    if dtype is None:
        dtype = tensor.dtype

    # cast the data to specified dtype for reduce/all-reduce
    if tensor.dtype != dtype:
        tensor_to_reduce = tensor.to(dtype)
    else:
        tensor_to_reduce = tensor

    world_size = dist.get_world_size(group=group)
    tensor_to_reduce.div_(world_size)

    # if rank is None, all reduce will be used
    # else, reduce is used
    use_all_reduce = dst_local_rank is None

    if use_all_reduce:
        dist.all_reduce(tensor_to_reduce, group=group)
    else:
        dist.reduce(tensor=tensor_to_reduce, dst=dst_global_rank, group=group)

    # recover the original dtype
    if tensor.dtype != dtype and tensor is not tensor_to_reduce:
        local_rank = dist.get_rank(group=group)
        if use_all_reduce or dst_local_rank == local_rank:
            tensor.copy_(tensor_to_reduce)

    return tensor


def has_inf_or_nan(tensor):
    """
    Check if a tensor has any NaN or Inf values.

    Args:
        tensor (torch.Tensor): Tensor to check.

    Returns:
        bool: True if the tensor has NaN or Inf values, False otherwise.
    """
    try:
        # if tensor is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as tensor
        # (which is true for some recent version of pytorch).
        tensor_sum = float(tensor.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # tensor_sum = float(tensor.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if (
            tensor_sum == float("inf")
            or tensor_sum == -float("inf")
            or tensor_sum != tensor_sum
        ):
            return True
        return False


def release_param_grad(tensor_list: List[torch.Tensor]):
    """
    Release the gradients of a list of tensors.

    Args:
        tensor_list (List[torch.Tensor]): List of tensors to release gradients for.
    """
    for tensor in tensor_list:
        tensor.grad = None


def compute_norm(
    gradients: Iterable[torch.Tensor],
    params: Iterable[torch.Tensor],
    dp_group: dist.ProcessGroup,
    mp_group: dist.ProcessGroup,
    norm_type: Union[int, float] = 2,
) -> float:
    """Clips gradient norm of an iterable of parameters.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Args:
        gradients (Iterable[Tensor]): an iterable of gradient tensors
        params (Iterable[Tensor]): an iterable of parameters that the gradients are associated with
        dp_group (torch.nn.parallel.ProcessGroup): data parallel process group
        mp_group (torch.nn.parallel.ProcessGroup): model parallel process group
         (float or int): max norm of the gradients
        norm_type (Union[int, float]): type of the used p-norm. Can be ``2`` for L2 norm or ``inf`` for infinity norm. (default: 2)

    Returns:
        float: Total norm of the gradients.
    """

    if mp_group is None:
        mp_rank = 0
    else:
        mp_rank = dist.get_rank(mp_group)

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        dist.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_group
        )

        # Take max across all GPUs.
        if mp_group is not None:
            dist.all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.MAX)
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        # if dist.get_rank() == 0:
        #    logger.info(f"Total Norm beginning {total_norm}")

        for g, p in zip(gradients, params):
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if is_model_parallel_parameter(p) or mp_rank == 0:
                param_norm = g.data.double().norm(2)
                total_norm += param_norm.item() ** 2

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=dp_group
        )

        if mp_group is not None:
            dist.all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.SUM)

        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)

    if (
        total_norm == float("inf")
        or total_norm == -float("inf")
        or total_norm != total_norm
    ):
        total_norm = -1

    return total_norm


def split_half_float_double(
    tensor_list: List[torch.Tensor],
) -> List[List[torch.Tensor]]:
    """
    Split the tensors in `tensor_list` into several lists according to their data type,
    which could be `torch.cuda.HalfTensor`, `torch.cuda.FloatTensor`,
    `torch.cuda.DoubleTensor`, or `torch.cuda.BFloat16Tensor`.

    Args:
        tensor_list (List[torch.Tensor]): List of PyTorch tensors.

    Returns:
        List[List[torch.Tensor]]: A list of lists, where each list contains tensors with the same data type.
    """
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor",
    ]
    buckets = []
    for _, dtype in enumerate(dtypes):
        bucket = [t for t in tensor_list if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def sync_param(flat_tensor: torch.Tensor, tensor_list: Iterable[torch.Tensor]):
    """
    Synchronize the flattened tensor and unflattened tensor list. When
    a list of tensor are flattened with `torch._utils._unflatten_dense_tensors`,
    a new tensor is created. Thus, the flat tensor and original tensor list do not
    share the same memory space. This function will update the tensor list so that
    they point to the same value.

    Args:
        flat_tensor (torch.Tensor): A flat tensor obtained by calling `torch._utils._unflatten_dense_tensors` on a tensor list
        tensor_list (Iterable[torch.Tensor]): A list of tensors corresponding to the flattened tensor
    """
    updated_params = unflatten(flat_tensor, tensor_list)

    # update the tensor data
    for p, q in zip(tensor_list, updated_params):
        p.data = q.data
