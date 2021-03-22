import numpy as np
from typing import Callable, Union, Tuple


def generate_toy_data(f: Callable, sample_size: int, std: Union[int, float], domain: Tuple[int, float] = (0, 1)):
    x = np.linspace(domain[0], domain[1], sample_size)
    t = f(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def load_old_faithful():
    return np.genfromtxt("../datasets/old_faithful.csv", dtype=float, delimiter=',', skip_header=1)
