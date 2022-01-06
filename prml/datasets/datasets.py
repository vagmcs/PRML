# Types
from typing import Callable, Tuple

# Dependencies
import numpy as np

# Project
from prml import datasets_dir


def generate_toy_data(
    f: Callable[[np.ndarray], np.ndarray], sample_size: int, std: float, domain: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a toy dataset given a function, a domain and a sample size. Then
    adds Gaussian noise to the samples having zero mean and the given standard deviation.

    :param f: a function
    :param sample_size: the size of the sample
    :param std: the standard deviation of the Gaussian noise
    :param domain: the domain range
    :return: a tuple of the input, target arrays
    """
    x = np.linspace(domain[0], domain[1], sample_size)
    t = f(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def load_old_faithful() -> np.ndarray:
    """
    Loads the old faithful dataset. Old Faithful is a hydrothermal geyser
    in Yellowstone National Park in the state of Wyoming, USA. The data
    comprises 272 observations, each representing a single eruption, and
    contains two variables corresponding to the duration of the eruption,
    and the time until the next eruption in minutes.

    :return: an array of shape (272, 2)
    """
    return np.genfromtxt(datasets_dir / "old_faithful.csv", dtype=float, delimiter=",", skip_header=1)
