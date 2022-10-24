from functools import partial

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.data_parallel._ddp.distributed_data_parallel import (
    _DistributedDataParallel,
)
from oslo.torch.nn.parallel.utils import add_wrapper
from oslo.torch.utils.data import SequenceParallelCollator

SEQUENCE_PARALLEL_KEYS = [
    "input_ids",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "token_type_ids",
    "labels",
]


class _SequenceParallelState(object):
    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context


# based on `allreduce_hook` in
# torch.distributed.algorithm.ddp_comm_hooks.default_hooks
def _sequence_parallel_hook(
    state: _SequenceParallelState, bucket: dist.GradBucket
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


def SequenceParallel(
    module,
    parallel_context,
    pad_token_id=None,
    dim=1,
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
        state=_SequenceParallelState(parallel_context),
        hook=_sequence_parallel_hook,
    )

    add_wrapper(
        module,
        mode=ParallelMode.SEQUENCE_DP,
        wrapper=sp,
        parallel_context=parallel_context,
    )

    if (
        hasattr(module.config, "pad_token_id")
        and module.config.pad_token_id is not None
    ):
        pad_token_id = module.config.pad_token_id

    if pad_token_id is None:
        raise ValueError(
            "param `pad_token_id` must not be None. "
            "please define `pad_token_id` in your model config or input the pad token id."
        )

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError(
                "You can not use non-keyword arguments for sequence parallelism. "
                "For example, if you coded like `model(input_ids, attention_mask)`, "
                "please rewrite your code like "
                "`model(input_ids=input_ids, attention_mask=attention_mask)`."
            )

        collator = SequenceParallelCollator(
            parallel_keys=[key for key in kwargs if key in SEQUENCE_PARALLEL_KEYS],
            dim=dim,
            parallel_context=parallel_context,
            pad_token_id=pad_token_id,
        )

        return self.forward(*args, **collator(kwargs))

    setattr(module, "forward", partial(forward, self=sp))
    setattr(module, "train", sp.train)
    return module
