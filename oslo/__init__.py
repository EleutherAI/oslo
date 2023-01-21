from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.extensions import ready_torch


def ready(model, parallel_context: ParallelContext):
    ready_torch(model, parallel_context)
