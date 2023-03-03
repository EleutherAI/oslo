import random

import numpy as np
import torch


def set_seed(seed: int, cuda_deterministic: bool = False):
    """Set seed for random, numpy, torch.

    Args:
        seed (int): Random seed.
        cuda_deterministic (bool, optional): Deterministic for cuda. Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
