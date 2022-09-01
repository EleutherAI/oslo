import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import add_wrapper
from oslo.torch.nn.parallel.data_parallel.distributed_data_parallel import (
    _DistributedDataParallel,
)


class _SequenceDataParallelState(object):
    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context


# based on `allreduce_hook` in
# torch.distributed.algorithm.ddp_comm_hooks.default_hooks
def _sequence_data_parallel_hook(
    state: _SequenceDataParallelState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    parallel_context = state.parallel_context
    group_to_use = parallel_context.get_group(ParallelMode.SEQUENCE_DP)
    div_factor = parallel_context.get_world_size(ParallelMode.DATA)

    # divide the tensor with DP size
    # tensor = bucket.get_tensor()
    tensor = bucket.buffer()
    tensor.div_(div_factor)

    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()

    return fut.then(lambda x: x.value()[0])


def SequenceDataParallel(
    module,
    parallel_context,
    dim=0,
    broadcast_buffers=True,
    bucket_cap_mb=25,
    find_unused_parameters=False,
    check_reduction=False,
    gradient_as_bucket_view=False,
    static_graph=False,
):
    sp = _DistributedDataParallel(
        module,
        device_ids=[torch.cuda.current_device()],
        output_device=torch.cuda.current_device(),
        dim=dim,
        broadcast_buffers=broadcast_buffers,
        process_group=parallel_context.get_group(ParallelMode.SEQUENCE_DP),
        bucket_cap_mb=bucket_cap_mb,
        find_unused_parameters=find_unused_parameters,
        check_reduction=check_reduction,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )
    sp.register_comm_hook(
        state=_SequenceDataParallelState(parallel_context),
        hook=_sequence_data_parallel_hook,
    )

    add_wrapper(module, ParallelMode.SEQUENCE_DP, sp)
    setattr(module, "forward", sp.forward)
    setattr(module, "train", sp.train)
    return module
