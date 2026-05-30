import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seeds across all libraries to ensure reproducibility.
    Call this at the beginning of both train.py and inference.py.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # Disables auto-tuner that picks fastest convolution algorithm (varies run to run)
    torch.backends.cudnn.deterministic = True # Makes CUDA convolution deterministic at slight speed cost (needed for reproducibility)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Warning: could not enable full deterministic algorithms: {e}")
        
    # https://docs.pytorch.org/docs/2.12/notes/randomness.html

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add checkpoint loads/savess if needed (train and inference load checkpoints)
