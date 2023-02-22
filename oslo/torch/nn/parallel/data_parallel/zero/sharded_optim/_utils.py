from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch._six import inf
import math
import torch
from typing import Optional, Iterable, List, Union
from oslo.torch.distributed import ParallelMode


# TODO
def is_model_parallel_parameter(p: torch.Tensor):
    """
    Check if a parameter is parallel in either Pipeline or Tensor mode.

    Args:
        p (torch.Tensor): Parameter to check.

    Returns:
        bool: True if the parameter is parallel in either mode, False otherwise.
    """
    parallel_mode = getattr(p, "oslo_parallel", dict())
    return (
        ParallelMode.PIPELINE in parallel_mode or ParallelMode.TENSOR in parallel_mode
    )


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


def has_inf_or_nan(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor has any NaN or Inf values.

    Args:
        tensor (torch.Tensor): Tensor to check.

    Returns:
        bool: True if the tensor has NaN or Inf values, False otherwise.
    """
    try:
        fp32_tensor = tensor.float()
    except RuntimeError as exception:
        if "value cannot be converted" not in exception.args[0]:
            raise exception
        return True
    else:
        if torch.isinf(fp32_tensor).any() or torch.isnan(fp32_tensor).any():
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
