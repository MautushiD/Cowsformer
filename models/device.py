import torch


def get_device():
    """Get device (cuda, mps, or cpu)"""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device
