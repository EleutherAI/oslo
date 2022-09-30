from dataclasses import dataclass

import torch

from oslo.torch.nn.parallel.pipeline_parallel._utils import (
    _is_namedtuple,
    _is_private,
    _is_primitive,
)


@dataclass
class TensorStub(object):
    id: int


def pack_tensor_stub(obj, args_list):
    """
    Recursively replace Tensor member variables to TensorStub.
    Inspiration: https://github.com/pytorch/pytorch/blob/master/torch/distributed/utils.py#L48
    """
    if torch.is_tensor(obj):
        id_ = len(args_list)
        tensor_sub = TensorStub(id_)
        args_list.append(obj)
        obj = tensor_sub

        return obj, args_list

    elif _is_namedtuple(obj):
        obj_list = list(obj)
        for i in range(len(obj_list)):
            obj_list_i, args_list = pack_tensor_stub(obj_list[i], args_list)
            obj_list_i[i] = obj_list_i
        obj = obj.__class__._make(obj_list)  # use namedtuple's method

        return obj, args_list

    elif isinstance(obj, tuple):
        obj = list(obj)
        for i in range(len(obj)):
            obj_i, args_list = pack_tensor_stub(obj[i], args_list)
            obj[i] = obj_i

        obj = tuple(obj)
        return obj, args_list

    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj_i, args_list = pack_tensor_stub(obj[i], args_list)
            obj[i] = obj_i

        return obj, args_list

    elif isinstance(obj, dict):
        for k in obj.keys():
            obj_k, args_list = pack_tensor_stub(obj[k], args_list)
            obj[k] = obj_k

        return obj, args_list

    elif _is_primitive(obj):
        return obj, args_list

    else:  # other kinds of object
        members = [
            attr
            for attr in dir(obj)
            if not callable(getattr(obj, attr)) and not _is_private(attr)
        ]
        for m in members:
            obj_m = getattr(obj, m)
            obj_m, args_list = pack_tensor_stub(obj_m, args_list)
            setattr(obj, m, obj_m)

        return obj, args_list


def unpack_tensor_stub(obj, args_list):
    """
    Recursively replace TensorStub to original Tensor.
    Inspiration: https://github.com/pytorch/pytorch/blob/master/torch/distributed/utils.py#L48
    """
    if isinstance(obj, TensorStub):
        id_ = obj.id
        tensor = args_list[id_]
        return tensor, args_list

    elif _is_namedtuple(obj):
        obj_list = list(obj)
        for i in range(len(obj_list)):
            obj_list_i, args_list = unpack_tensor_stub(obj_list[i], args_list)
            obj_list_i[i] = obj_list_i
        obj = obj.__class__._make(obj_list)  # use namedtuple's method

        return obj, args_list

    elif isinstance(obj, tuple):
        obj = list(obj)
        for i in range(len(obj)):
            obj_i, args_list = unpack_tensor_stub(obj[i], args_list)
            obj[i] = obj_i

        obj = tuple(obj)
        return obj, args_list

    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj_i, args_list = unpack_tensor_stub(obj[i], args_list)
            obj[i] = obj_i

        return obj, args_list

    elif isinstance(obj, dict):
        for k in obj.keys():
            obj_k, args_list = unpack_tensor_stub(obj[k], args_list)
            obj[k] = obj_k

        return obj, args_list

    elif _is_primitive(obj):
        return obj, args_list

    else:  # other kinds of object
        members = [
            attr
            for attr in dir(obj)
            if not callable(getattr(obj, attr)) and not _is_private(attr)
        ]
        for m in members:
            obj_m = getattr(obj, m)
            obj_m, args_list = unpack_tensor_stub(obj_m, args_list)
            setattr(obj, m, obj_m)

        return obj, args_list
