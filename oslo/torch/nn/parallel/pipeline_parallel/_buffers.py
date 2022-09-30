from oslo.torch.nn.parallel.pipeline_parallel._sync import (
    register_location_for_forward_counter,
)


# original forward dictionary
_ORIGINAL_FORWARDS = dict()

# module device locations
_MODULE_DEVICE_LOCATIONS = dict()


def register_original_forward_function(location, func, device):
    _ORIGINAL_FORWARDS[location] = func
    _MODULE_DEVICE_LOCATIONS[location] = device
    register_location_for_forward_counter(location)


def get_original_forward_function(location):
    return _ORIGINAL_FORWARDS[location]


def get_module_device_location(location):
    return _MODULE_DEVICE_LOCATIONS[location]


# Activations
_ACTIVATIONS = dict()


def save_activation(key, activation):
    _ACTIVATIONS[key] = activation


def pop_activation(key):
    return _ACTIVATIONS.pop(key, [])  # TODO; okay?
