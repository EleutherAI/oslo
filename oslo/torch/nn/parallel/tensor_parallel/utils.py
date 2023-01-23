import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelMode


def split_1d(tensor, world_size, dim, parallel_context):
    tensor = tensor.chunk(world_size, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
    ]
    return tensor


def gather_1d(tensor, world_size, dim, parallel_context):
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_1D),
    )
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def split_batch_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def split_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def split_embedding_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def split_layernorm_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def split_bias_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def gather_2d(tensor, summa_dim, parallel_context):
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list, tensor, parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
    )
    tensor = torch.cat(tensor_list, dim=-1)
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_2D_COL),
    )
    tensor = torch.cat(tensor_list, dim=0)
    return tensor


def split_batch_2p5d(tensor, tesseract_dim, parallel_context):
    tensor = tensor.chunk(tesseract_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
    ]
    return tensor


def split_2p5d(tensor, tesseract_dim, parallel_context):
    tensor = tensor.chunk(tesseract_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
    ]
    tensor = tensor.chunk(tesseract_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
    ]
    return tensor


def split_layernorm_2p5d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
    ]
    return tensor


def split_bias_2p5d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
    ]
    return tensor


def split_embedding_2p5d(tensor, summa_dim, dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
    ]
    return tensor


def gather_2p5d(tensor, tesseract_dim, parallel_context):
    tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
    dist.all_gather(
        tensor_list,
        tensor,
        parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
    )
    tensor = torch.cat(tensor_list, dim=-1)
    tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL),
    )
    tensor = torch.cat(tensor_list, dim=0)
    return tensor


def split_batch_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    return tensor


def split_input_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    return tensor


def split_weight_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
    ]
    return tensor


def split_layernorm_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    return tensor


def split_bias_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    return tensor


def split_embedding_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    return tensor


def gather_output_3d(tensor, cubic_dim, parallel_context):
    tensor_list = [torch.zeros_like(tensor) for _ in range(cubic_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_3D_OUTPUT),
    )
    tensor = torch.cat(tensor_list, dim=-1)
    tensor_list = [torch.zeros_like(tensor) for _ in range(cubic_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_3D_INPUT),
    )
    tensor = torch.cat(tensor_list, dim=0)
    tensor_list = [torch.zeros_like(tensor) for _ in range(cubic_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_3D_WEIGHT),
    )
    tensor = torch.cat(tensor_list, dim=0)
    return tensor
