import time

from oslo.torch.nn.parallel.pipeline_parallel._buffers import save_activation
from oslo.torch.nn.parallel.pipeline_parallel._functional import apply_backward_redirection
from oslo.torch.nn.parallel.pipeline_parallel._messages import pack_tensor_stub, unpack_tensor_stub

# original forward dictionary
_ORIGINAL_FORWARDS = dict()

# module device locations
_MODULE_DEVICE_LOCATIONS = dict()


_RESULT_DICT = dict()


_DONE_CHECKER = 0


_FORWARD_COUNTER = dict()


def get_result(ind):
    while ind not in _RESULT_DICT:
        time.sleep(0.0)
    return _RESULT_DICT[ind]


def reset_result():
    _RESULT_DICT.clear()


def get_forward_counter(loc):
    return _FORWARD_COUNTER[loc]


def increment_forward_counter(loc):
    _FORWARD_COUNTER[loc] += 1


def reset_forward_counter():
    for k in _FORWARD_COUNTER:
        _FORWARD_COUNTER[k] = 0


def increment_done():
    global _DONE_CHECKER
    _DONE_CHECKER += 1


def get_done():
    global _DONE_CHECKER
    return _DONE_CHECKER


def reset_done():
    global _DONE_CHECKER
    _DONE_CHECKER = 0


def remote_module_forward(caller, location, unique_key, args_stub, kwargs_stub, requires_redirection, *tensors):
    if requires_redirection:
        # prepare backward redirection to caller
        tensors = apply_backward_redirection(
            caller,
            unique_key,
            *tensors,
        )

    (args, kwargs), _ = unpack_tensor_stub([args_stub, kwargs_stub], tensors)

    forward_fn = _ORIGINAL_FORWARDS[location]
    result = forward_fn(*args, **kwargs)

    result_stub, tensors = pack_tensor_stub(result, [])
    need_activation_save = any([t.requires_grad for t in tensors])
    if need_activation_save:
        save_activation(unique_key, tensors)

    return result_stub, tensors, need_activation_save
