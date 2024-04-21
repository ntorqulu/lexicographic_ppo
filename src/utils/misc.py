import argparse
from typing import List, Type, Dict

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


def set_seeds(seed: int, deterministic: bool = False):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # TODO: add deterministic?
    th.backends.cudnn.deterministic = deterministic


def set_torch(n_cpus: int, cuda: bool) -> th.device:
    th.set_num_threads(n_cpus)
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    # Check if the provided device is already the current device
    is_same_device = (device == th.cuda.current_device()) if device.type == 'cuda' else (device == th.device('cpu'))

    if is_same_device:
        return {k: th.as_tensor(d, dtype=astype) for k, d in zip(agents, data)}
    else:
        return {k: th.as_tensor(d, dtype=astype).to(device) for k, d in zip(agents, data)}
