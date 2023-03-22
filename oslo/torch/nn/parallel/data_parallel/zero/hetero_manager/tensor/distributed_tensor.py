import operator
from copy import copy
from functools import lru_cache, reduce
from typing import Callable, Optional, Set

import torch

from .distributed_spec_manager import DistributedSpecManager
from .distributed_spec import DistributedSpec, DistributedPlacementPattern, ReplicaSpec
from .distributed_tensor_spec import DistributedTensorSpec

from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed.parallel_context import ParallelContext

from .const import TensorType
from .op_wrapper import _SHARDED_OPS


@lru_cache(None)
def _get_my_nowrap_functions() -> Set[Callable]:
    Tensor = torch.Tensor
    return {
        Tensor._base.__get__,
        Tensor.grad.__get__,
        Tensor._grad.__get__,
        Tensor.data.__get__,    # make .data returns torch.Tensor rather than DistributedTensor
    }


def _convert_output(output, dist_tensor_spec: DistributedTensorSpec):
    if type(output) == torch.Tensor:
        return DistributedTensor.from_torch_tensor(output, dist_tensor_spec)
    elif isinstance(output, (list, tuple)):
        return type(output)(_convert_output(o, dist_tensor_spec) for o in output)
    else:
        return output


def _get_spec_from_args(args, kwargs) -> DistributedTensorSpec:
    for elem in args:
        if isinstance(elem, DistributedTensor):
            parallel_context = elem.get_parallel_context()
            dp = elem.dist_spec
            return DistributedTensorSpec(parallel_context, dp)
        elif isinstance(elem, (list, tuple)):
            spec = _get_spec_from_args(elem, {})
            if spec is not None:
                return spec
    for _, v in kwargs.items():
        if isinstance(v, DistributedTensor):
            parallel_context = v.get_parallel_context()
            dp = v.dist_spec
            return DistributedTensorSpec(parallel_context, dp)
    return None


class DistributedTensor(torch.Tensor):
    """ Data Structure for Tensor in Oslo. It is a subclass of torch.Tensor.

    The DistributedTensor can be initialized with a PyTorch tensor in the following ways.

        >>> parallel_context = ParallelContext()
        >>> dist_tensor_a = DistributedTensor(torch.randn(2,3), spec = DistributedTensorSpec(parallel_context, ReplicaSpec()))
        >>> # The tensor passed in is a tensor after sharding but not a global tensor.
        >>> shard_spec = ShardSpec(parallel_context=ParallelContext(tensor_parallel_size=world_size),
        >>>                 dims=[0],
        >>>                 num_partitions=[world_size])
        >>> tensor_spec = DistributedTensorSpec(parallel_context, shard_spec)
        >>> dist_tensor_b = DistributedTensor.from_torch_tensor(t_ref.clone(), tensor_spec)

    Args:
        data (torch.Tensor): a torch tensor used as the payload the DistributedTensor.
        spec (DistributedTensorSpec, optional): the tensor spec of initialization. Defaults to DistributedTensorSpec(ReplicaSpec()).
    """
    torch_major = int(torch.__version__.split('.')[0])
    torch_minor = int(torch.__version__.split('.')[1])

    def __new__(cls, data: torch.Tensor, spec: DistributedTensorSpec) -> 'DistributedTensor':
        """
        The signature of the __new__ has to be consistent with the torch.Tensor.

        Args:
            data (torch.Tensor): a torch tensor used as the payload the DistributedTensor.
            spec (TensorSpec, optional): the tensor spec of initialization.

        Returns:
            DistributedTensor: a DistributedTensor wrappers the data.
        """
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, data: torch.Tensor, spec: Optional[DistributedTensorSpec] = None) -> None:
        # If not set spec, use a DP process group and replicate dist spec
        if spec is None:
            self.has_initialized = False
            self.dist_spec = ReplicaSpec()
            self.compute_spec = None
            self.parallel_context = ParallelContext.get_context()
        else:
            self.has_initialized = True
            self.dist_spec = spec.dist_attr
            self.compute_spec = spec.compute_attr
            if spec.parallel_context is None:
                self.parallel_context = ParallelContext.get_context()
            else:
                self.parallel_context = spec.parallel_context

        self._type = TensorType.NONMODEL

    def has_compute_spec(self) -> bool:
        return self.compute_spec is not None

    def is_model_data(self) -> bool:
        return self._type == TensorType.MODEL

    def get_parallel_context(self) -> 'ParallelContext':
        return self.parallel_context

    def set_parallel_context(self, parallel_context: ParallelContext):
        """set_parallel_context
        change the parallel_context of the DistributedTensor. Note that the valid use cases is limited.
        It works for the target parallel_context is DP and TP only and current dist spec of the Tensor is Replica.

        Args:
            parallel_context (ParallelContext): target parallel_context

        """
        assert isinstance(parallel_context, ParallelContext), f"parallel_context as type {type(parallel_context)} is invalid"
        # if the new parallel_context is the same as the old parallel_context, just returns
        if self.parallel_context == parallel_context:
            return
        assert self.parallel_context.get_world_size(ParallelMode.TENSOR) == 1 or self.parallel_context.get_world_size(ParallelMode.DATA) == 1, \
            "Can not set_parallel_context on a DistributedTensor whose parallel_context is both tp > 1 and world group > 1"
        assert self.dist_spec.placement.value == 'r', \
            "Can not set_parallel_context on a DistributedTensor whose dist spec is not Replica"

        self.parallel_context = parallel_context

    def get_dp_world_size(self) -> int:
        return self.parallel_context.get_world_size(ParallelMode.DATA)

    def get_tp_world_size(self) -> int:
        return self.parallel_context.get_world_size(ParallelMode.TENSOR)

    def set_dist_spec(self, dist_spec: DistributedSpec):
        """set_dist_spec
        set dist spec and change the payloads.

        Args:
            dist_spec (DistributedSpec): target dist spec.
        """
        assert isinstance(dist_spec, DistributedSpec)
        assert self.parallel_context is not None
        self._redistribute(dist_spec)

    def set_tensor_spec(self, dist_spec, compute_spec):
        if dist_spec is not None:
            assert isinstance(dist_spec, DistributedSpec), f"{type(dist_spec)}"
            self.set_dist_spec(dist_spec)
        if compute_spec is not None:
            self.compute_spec = compute_spec

    def has_compute_pattern(self, compute_pattern):
        return self.compute_spec.compute_pattern == compute_pattern

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        global _SHARDED_OPS
        if func in _SHARDED_OPS:
            func = _SHARDED_OPS[func]

        if cls.torch_major > 1 or (cls.torch_major == 1 and cls.torch_minor >= 12):
            # in order to trigger pre-op hook in the forward of checkpoint module
            # we have to capture the `backward` function
            # and make sure that it does not in `torch._C.DisableTorchFunction()` context
            if func is torch.Tensor.backward:
                assert len(args) == 1    # only has 1 paramter
                backward_tensor = torch.Tensor(args[0])
                tensor_kwargs = {k: torch.Tensor(v) if torch.is_tensor(v) else v for k, v in kwargs.items()}
                return backward_tensor.backward(**tensor_kwargs)

        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            if func in _get_my_nowrap_functions():
                return ret
            else:
                dist_tensor_spec = _get_spec_from_args(args, kwargs)
                return _convert_output(ret, dist_tensor_spec)

    def __repr__(self):
        output_list = [super(DistributedTensor, self).__repr__()]
        output_list.append(str(self.parallel_context))
        output_list.append(str(self.dist_spec))
        if self.compute_spec is not None:
            output_list.append(str(self.compute_spec))
        return "\n".join(output_list)

    def _redistribute(self, dist_spec: DistributedSpec) -> None:
        """_redistribute
        Note the function will not handle the logic of backward propagation!
        It is used during model tensor initializations as an internal function.

        Args:
            dist_spec (DistributedSpec): the target dist. spec.
        """
        assert self.grad_fn is None, "Current tensor has grad_fn and it can't get converted"
        with DistributedSpecManager.no_grad():
            self.data = DistributedSpecManager.handle_trans_spec(self.data, self.dist_spec, dist_spec, self.parallel_context)
        self.dist_spec = dist_spec

    def redistribute(self, dist_spec: DistributedSpec, parallel_context: Optional[ParallelContext] = None) -> 'DistributedTensor':
        """redistribute
        Redistribute the tensor among processes. The rule is like this:

        1. If the parallel_context is None, then redistribute the tensor payload among the TP process group. Keep the
        DP process group not changed.

        2. If the parallel_context is not not None and not equal to the current process group.
        First, convert the tensor as replicated among the TP process group.
        Second, reset the process group to the new parallel_context.
        Third, conver the tensor (new replicated both among the tp process group) to the new dist_spec.

        Args:
            dist_spec (DistributedSpec): the new dist spec.
            parallel_context (Optional[ParallelContext], optional): the new parallel context. Defaults to None.

        Returns:
            DistributedTensor: a redistributed DistributedTensor
        """
        if parallel_context is not None and parallel_context != self.get_parallel_context():
            # if the parallel_context is not equal, convert the current tensor to replicated
            handled = self.redistribute(ReplicaSpec())
        else:
            handled = self
            parallel_context = self.parallel_context

        ret = DistributedSpecManager.handle_trans_spec(handled, handled.dist_spec, dist_spec, parallel_context)
        return DistributedTensor.from_torch_tensor(ret, DistributedTensorSpec(parallel_context=parallel_context, dist_attr=dist_spec))

    def to_replicate_(self):
        """to_replicate_

        an inline member function, converting dist spec of the tensor to REPLICATE
        """
        self._redistribute(dist_spec=ReplicaSpec())

    def to_replicate(self) -> 'DistributedTensor':
        """to_replicate

        converting dist spec of the tensor to ReplicaSpec()
        """
        return self.redistribute(ReplicaSpec())

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor, spec: Optional[DistributedTensorSpec] = None) -> 'DistributedTensor':
        """from_torch_tensor

        A static method builds a `DistributedTensor` from a PyTorch Tensor.

        Args:
            tensor (torch.Tensor): the pytorch tensor, which is a local tensor for this rank not a global tensor.
            spec (Optional[DistributedTensorSpec], optional): tensor spec. Defaults to None.

        Returns:
            DistributedTensor: a DistributedTensor
        """
        tensor = tensor.as_subclass(DistributedTensor)
        tensor.__init__(tensor, spec=spec)
        return tensor

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = DistributedTensor(data, spec=copy(DistributedTensorSpec(self.parallel_context, self.dist_spec, self.compute_spec)))
            memo[id(self)] = tensor
            return tensor

    # override builtin functions which must use tensor in replicate placement #

    def size_local(self, *args) -> torch.Size:
        with torch._C.DisableTorchFunction():
            return super().size(*args)

    def size_global(self, *args) -> torch.Size:
        """size_global

        override the torch buildin size()
        the shape passed in must be in a replicate placement.

        Returns:
            torch.Size: the global tensor shape
        """
        if self.is_replicate():
            return self.size_local(*args)
        spec = self.dist_spec
        dims = spec.dims
        num_partitions = spec.num_partitions
        # import inspect
        # print(*['{:40}| {}:{}\n'.format(x.function, x.filename, x.lineno) for x in inspect.stack()])
        size_list = list(self.size_local())
        for dim, num_partition in zip(dims, num_partitions):
            size_list[dim] *= num_partition
        if args == ():
            return torch.Size(size_list)
        else:
            return size_list[args[0]]

    def numel_global(self):
        """Returns the number of elements in the tensor when it's replicated.
        """
        return reduce(operator.mul, self.size_global(), 1)

    # Some API for dist spec check

    def is_replicate(self):
        return self.dist_spec.placement == DistributedPlacementPattern.REPLICATE \
               or (len(self.dist_spec.num_partitions) == 1
                   and self.dist_spec.num_partitions[0] == 1) \
               or (self.parallel_context.get_world_size(ParallelMode.TENSOR) == 1)

    def is_shard_1dcol(self):
        return self.dist_spec.placement == DistributedPlacementPattern.SHARD \
               and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == -1

    def is_shard_1drow(self):
        return self.dist_spec.placement == DistributedPlacementPattern.SHARD \
               and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == 0

    def is_sharded(self):
        return self.dist_spec.placement == DistributedPlacementPattern.SHARD