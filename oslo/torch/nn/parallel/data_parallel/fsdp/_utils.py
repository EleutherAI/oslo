from collections import OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import _get_stream
from torch.nn.utils.rnn import PackedSequence

"""Useful functions to deal with tensor types with other python container types."""


def _sync_params_and_buffers(
    process_group: dist.ProcessGroup,
    module_states: List[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
):
    """
    Synchronizes ``module_states`` (list of tensors) across all processes by
    broadcasting them from rank 0.
    """
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )


def _replace_by_prefix(
    state_dict: Dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::
        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _contains_batchnorm(module):
    return any(isinstance(mod, _BatchNorm) for mod in module.modules())


def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def _override_batchnorm_mixed_precision(module):
    for mod in module.modules():
        if isinstance(mod, _BatchNorm):
            mod._wrap_overrides = {"mixed_precision": None}  # type: ignore[assignment]


def _apply_to_tensors(
    fn: Callable,
    container: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict, PackedSequence],
) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(
        x: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict, PackedSequence]
    ) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)


def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    *args,
    **kwargs,
):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.
    """

    def f(module: torch.nn.Module, prefix: str, *args, **kwargs):
        # Call the module function before recursing over children (pre-order)
        module_fn(module, prefix, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is not None:
                new_prefix = prefix + submodule_name + "."
                f(submodule, new_prefix, *args, **kwargs)

    f(root_module, "", *args, **kwargs)
    return return_fn(*args, **kwargs)


def _recursive_to(inputs, target_gpu, use_side_stream_for_tensor_copies):
    r"""
    Recursively moves input to the target_gpu.
    """

    def to_map(obj):
        if isinstance(obj, torch.Tensor):
            if obj.device == torch.device("cuda", target_gpu):
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_gpu),)
            else:
                # Perform CPU -> GPU copies in a background stream. This code is
                # motivated from similar logic in torch/nn/parallel/_functions.py
                stream = _get_stream(target_gpu)
                with torch.cuda.stream(stream):
                    output = obj.to(target_gpu)
                # synchronize with the copy stream
                with torch.cuda.device(target_gpu):
                    current_stream = torch.cuda.current_stream()
                    # Sync the current stream with the copy stream
                    current_stream.wait_stream(stream)
                    # Ensure tensor memory is not reused until work on
                    # main stream is complete
                    output.record_stream(current_stream)  # type: ignore[arg-type]
                return (output,)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
        return [obj]

    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res


def _to_kwargs(inputs, kwargs, device_id, use_side_stream_for_tensor_copies):
    inputs = (
        _recursive_to(inputs, device_id, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    kwargs = (
        _recursive_to(kwargs, device_id, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
