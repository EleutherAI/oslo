import os
import torch
import torch.distributed as dist

from _utils import split_input_3d
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelCrossEntropyLoss3D


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1)


def print_rank0(msg: str, logger=None):
    """Print messages and save logs(optional). This is executed only if you are the rank-0 gpu.
    Args:
        msg (str): A string message to output.
        logger (:class:`colossalai.logging.DistributedLogger`, optional):
            The logger to record the message, defaults to None.
    """
    if parallel_context.get_global_rank() == 0:
        if logger is None:
            print(msg, flush=True)
        else:
            logger.info(msg)


tp_size = int(os.environ["WORLD_SIZE"])

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_3D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

criterion_master = torch.nn.CrossEntropyLoss()
criterion = VocabParallelCrossEntropyLoss3D(parallel_context=parallel_context)
rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR_2P5D)

batch_size = 4
seq_len = 6
num_classes = 8
cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
out_master = torch.randn(batch_size, seq_len, num_classes).cuda()
target = torch.randint(num_classes, size=(batch_size, seq_len), dtype=torch.long).cuda()
dist.broadcast(out_master, src=0)
dist.broadcast(target, src=0)

out = split_input_3d(out_master.clone(), cubic_dim, parallel_context=parallel_context)
out = out.clone()
out.requires_grad = True

loss = criterion(out, target)

out_master = out_master.clone()
out_master.requires_grad = True
loss_master = criterion_master(
    out_master.view(-1, out_master.size(-1)), target.view(-1)
)

check_equal(loss_master, loss)
print_rank0("vocab parallel loss forward: pass")

loss_master.backward()
loss.backward()

grad_master = out_master.grad
grad_master = split_input_3d(grad_master, cubic_dim, parallel_context=parallel_context)
grad = out.grad


check_equal(grad_master, grad)
print_rank0("vocab parallel loss backward: pass")
