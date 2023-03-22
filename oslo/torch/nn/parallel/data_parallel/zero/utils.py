import torch


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")
