from oslo.torch.distributed import ParallelContext
from oslo.torch.utils.extensions import ready_torch


def ready(model, parallel_context: ParallelContext):
    ready_torch(model, parallel_context)
