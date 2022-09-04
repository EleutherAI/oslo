# Activations
_ACTIVATIONS = dict()


def save_activation(key, activation):
    _ACTIVATIONS[key] = activation


def pop_activation(key):
    return _ACTIVATIONS.pop(key)
