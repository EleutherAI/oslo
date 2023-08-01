from oslo.torch.nn.parallel.pipeline_parallel.pipeline_parallel import (
    PipelineParallel,
)

# initialize workers
# TODO; better way?
from oslo.torch.nn.parallel.pipeline_parallel._workers import *

__ALL__ = [PipelineParallel]
