import time
from queue import PriorityQueue, Queue

import torch

from ._messages import disassemble_new_args
from ._buffers import save_activation, pop_activation
from ._functional import apply_backward_redirection


# original forward dictionary
_ORIGINAL_FORWARDS = dict()

# module device locations
_MODULE_DEVICE_LOCATIONS = dict()


# Job queue
_JOB_QUEUE = PriorityQueue()

# remote work result receiver
_RECEIVER = dict()


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


def reset_backward_notify():
    global _NOTIFY_BACKWARD_DONE
    _NOTIFY_BACKWARD_DONE = False


def backward_done_notify():
    global _NOTIFY_BACKWARD_DONE
    _NOTIFY_BACKWARD_DONE = True


def wait_backward_done():
    global _NOTIFY_BACKWARD_DONE
    while not _NOTIFY_BACKWARD_DONE:
        time.sleep(0.0)


def remote_module_forward(caller, location, unique_key, arg_keys, *args):
    # prepare backward redirection to caller
    args = apply_backward_redirection(
        caller,
        unique_key,
        *args,
    )

    args, kwargs = disassemble_new_args(args, arg_keys)
    forward_fn = _ORIGINAL_FORWARDS[location]
    result = forward_fn(*args, **kwargs)
    save_activation(unique_key, result)
    return result


def wait_remote_work_result(request_message):
    tag = request_message.tag
    assert tag in _RECEIVER, f"{tag=}"
    result = _RECEIVER[tag].get()
    torch.cuda.current_stream().synchronize()

    # delete a queue for communication
    _RECEIVER.pop(tag)
    return result


def response_with_result(req, tag, result, result_wrapped):
    result = (req, result, result_wrapped)
    _RECEIVER[tag].put(result)
    torch.cuda.current_stream().synchronize()


def run_remote_backward(req, *grad_outputs):
    # need to ensure that grad_outputs is fully received
    # TODO; no other way?
    torch.cuda.synchronize()

    tag = req.tag
    activation, req = pop_activation(tag)

    # TODO; some output contains tuple of tuple..
    #   better way to deal with this?
    new_act = []
    new_grad = []
    for act, grad in zip(activation, grad_outputs):
        if act is not None and grad is not None and act.requires_grad:
            new_act.append(act)
            new_grad.append(grad)

    torch.autograd.backward(tuple(new_act), tuple(new_grad))
