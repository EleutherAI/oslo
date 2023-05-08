from oslo.torch.distributed.tensor import (
    DistributedTensor,
    DistributedParameter,
)
import torch
from oslo.torch.utils import get_free_port
import os
from oslo.torch.distributed.parallel_context import ParallelContext


def test_multiinheritance():
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["LOCAL_WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)
    ParallelContext.from_torch(data_parallel_size=1)
    dist_param = DistributedParameter(None, requires_grad=True)
    assert dist_param.dist_spec.placement.value == "r"
    assert isinstance(dist_param, DistributedTensor)
    assert isinstance(dist_param, torch.nn.Parameter)

    # __deepcopy__ overload
    import copy

    dist_param2 = copy.deepcopy(dist_param)
    assert isinstance(dist_param2, DistributedParameter)
    assert torch.equal(dist_param.data, dist_param2.data)
    assert dist_param.requires_grad == dist_param2.requires_grad

    # __torch_function__
    clone_param = torch.clone(dist_param)
    assert isinstance(clone_param, DistributedTensor)
