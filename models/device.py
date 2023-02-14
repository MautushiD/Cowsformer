import torch
import platform


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("GPU is available, CUDA used", flush=True)
        str_device = "cuda"
    elif platform.system() == "Darwin":
        print("M1 GPU is available, MPS used", flush=True)
        str_device = "mps"
    else:
        print("GPU is not available, CPU used", flush=True)
        str_device = "cpu"

    device = torch.device(str_device)
    return device
