import argparse
from typing import Tuple, Optional, Dict, Union, List, Type

import numpy as np
import torch as th

# Type aliases for better readability
Tensor = th.Tensor
Array = np.array


@th.jit.script
def normalize(x: Tensor) -> Tensor:
    """
    Normalize a tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    return (x - x.mean()) / (x.std() + 1e-8)


def str2bool(v):
    """
    Convert a string to a boolean value.

    Args:
        v (str): Input string.

    Returns:
        bool: Boolean value corresponding to the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid boolean value.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seeds(seed: int, deterministic: Optional[bool]):
    """
    Set the seed for random number generation to ensure reproducibility.

    Args:
        seed (int): The seed value.
        deterministic (Optional[bool]): Whether to make computations deterministic.
    """
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = deterministic


def set_torch(n_cpus: int, cuda: bool) -> th.device:
    """
    Configure PyTorch settings including number of CPU threads and CUDA usage.

    Args:
        n_cpus (int): Number of CPU threads to use.
        cuda (bool): Whether to use CUDA if available.

    Returns:
        th.device: The device to use for tensor computations.
    """
    th.set_num_threads(n_cpus)
    if th.cuda.is_available() and cuda:
        print("Using CUDA")
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")
