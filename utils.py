import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    Set random seeds across all libraries to ensure reproducibility.
    Call this at the beginning of both train.py and inference.py.
    """
    
    # Controls randomness in PyTorch operations
    torch.manual_seed(seed)
    
    # Controls Python's built-in random 
    random.seed(seed)
    
    # Controls NumPy randomness 
    np.random.seed(seed)
    
    # Makes CUDA convolution deterministic at slight speed cost (needed for reproducibility) 
    torch.backends.cudnn.deterministic = True
    
    # Disables auto-tuner that picks fastest convolution algorithm (varies run to run)
    torch.backends.cudnn.benchmark = False
    
    # https://docs.pytorch.org/docs/2.12/notes/randomness.html
