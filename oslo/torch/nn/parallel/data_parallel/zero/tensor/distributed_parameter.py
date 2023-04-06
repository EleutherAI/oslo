from typing import Optional

import torch

from .distributed_tensor import DistributedTensor
from .const import TensorType
from .param_op_hook import DistributedParamOpHookManager
from .distributed_tensor_spec import DistributedTensorSpec


def filter_distributed_parameters(*args, **kwargs):
    param_list = []

    def get_distributed_parameters(element) -> None:
        if isinstance(element, list) or isinstance(element, tuple):
            for e in element:
                get_distributed_parameters(e)
        elif isinstance(element, dict):
            raise RuntimeError(
                "Found Dict: DistributedParameter can't deal with complicated arguments."
            )
        elif isinstance(element, DistributedParameter):
            param_list.append(element)
        return

    for a in args:
        get_distributed_parameters(a)
    for v in kwargs.values():
        get_distributed_parameters(v)

    return param_list


def replace_args(args, kwargs, new_args):
    args = new_args[: len(args)]
    for k, v in zip(kwargs.keys(), new_args[len(args) :]):
        kwargs[k] = v
    return tuple(args), kwargs


class DistributedParameter(DistributedTensor, torch.nn.Parameter):
    r"""A kind of DistributedTensor to be considered as a module parameter."""

    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        spec: DistributedTensorSpec = None,
    ) -> "DistributedParameter":
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        spec: DistributedTensorSpec = None,
    ) -> None:
        DistributedTensor.__init__(self, data, spec)
        self._type = TensorType.MODEL
        # a list contains modules sharing this DistributedParameter with others.
        self._shared_param_modules = []

    @property
    def shared_param_modules(self):
        return self._shared_param_modules

    @staticmethod
    def from_torch_tensor(
        tensor: torch.Tensor,
        requires_grad: bool = True,
        spec: DistributedTensorSpec = None,
    ) -> "DistributedParameter":
        tensor = tensor.as_subclass(DistributedParameter)
        tensor.__init__(tensor, requires_grad=requires_grad, spec=spec)
        return tensor

    def __repr__(self):
        return super(DistributedParameter, self).__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        if DistributedParamOpHookManager.has_hook():
            if not func.__name__.startswith("__"):
                if kwargs is None:
                    kwargs = {}
                params = filter_distributed_parameters(*args, **kwargs)
                if len(params) > 0:
                    with torch._C.DisableTorchFunction():
                        new_args = DistributedParamOpHookManager.pre_op(
                            params, *args, *kwargs.values()
                        )
                    args, kwargs = replace_args(args, kwargs, new_args)
                    ret = super().__torch_function__(func, types, args, kwargs)
                    with torch._C.DisableTorchFunction():
                        ret = DistributedParamOpHookManager.post_op(params, ret)
                    return ret
        return super().__torch_function__(func, types, args, kwargs)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = DistributedParameter(
                data,
                self.requires_grad,
                spec=DistributedTensorSpec(
                    self.get_parallel_context(), self.dist_spec, self.compute_spec
                ),
            )
            memo[id(self)] = tensor
            return tensor

    def __reduce_ex__(self, proto):
        # Adapted from torch._utils._rebuild_parameter
        # def _rebuild_distributed_parameter(data, requires_grad, backward_hooks):
        #     distributed_param = DistributedParameter(data, requires_grad)
        #     distributed_param._backward_hooks = backward_hooks
        #     return distributed_param

        # return (
        #     _rebuild_distributed_parameter,
        #     (self.data, self.requires_grad, OrderedDict())
        # )

        # TODO(jzy) we don't support object reflection now.
        # distspec cannot be pickled or rebuilt because it's tightly connected to runtime attribute `process_group`.
        raise NotImplementedError
