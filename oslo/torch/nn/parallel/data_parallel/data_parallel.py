from oslo import ParallelContext
from oslo.torch.nn.parallel.data_parallel.ddp.distributed_data_parallel import (
    DistributedDataParallel,
)


def DataParallel(
    module,
    parallel_context: ParallelContext,
    zero_stage: int = 0,
):
    # TODO: Mingu Kang

    if zero_stage == 0:
        return DistributedDataParallel(module, parallel_context=parallel_context)
    elif zero_stage == 1:
        pass
    elif zero_stage == 2:
        pass
    elif zero_stage == 3:
        pass
    else:
        raise ValueError("param `zero_stage` must be one of the 0, 1, 2, 3.")
