from typing import Iterable

import torch


def is_ddp_ignored(p):
    return getattr(p, "_ddp_to_ignore", False)


def set_params_to_ignore(params_to_ignore: Iterable[torch.Tensor]) -> None:
    """Sets parameters to be ignored by DDP.
    This method must be called before initializing DistributedDataParallel.
    Example:
        >>> params_to_ignore = []
        >>> for p in module.parameters():
        >>>     if should_ignore(p):
        >>>         params_to_ignore.append(p)
        >>>         set_params_to_ignore(params_to_ignore)
        >>> module = DistributedDataParallel(module)
    Args:
        params_to_ignore (Iterable[torch.Tensor]): A list of parameters to be ignored.
    """
    for p in params_to_ignore:
        p._ddp_to_ignore = True
