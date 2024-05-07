import argparse
from typing import List, Type, Dict, Optional

import numpy as np
import torch as th
import random

Tensor = th.Tensor
Array = np.array


@th.jit.script
def normalize(x: Tensor) -> Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seeds(seed: int, deterministic: Optional[bool]):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = deterministic


def set_torch(n_cpus: int, cuda: bool) -> th.device:
    th.set_num_threads(n_cpus)
    if th.cuda.is_available() and cuda:
        print("Using CUDA")
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")