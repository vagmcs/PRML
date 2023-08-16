# Types
from typing import Callable, Optional, Tuple

# Standard Library
import struct

# Dependencies
import numpy as np
from sklearn.utils import resample

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


def load_planar_dataset(sample_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a binary class dataset having non-linear decision boundary.

    :param sample_size: the size of the sample
    :return: a tuple of the input, target arrays
    """
    # number of points per class
    n = int(sample_size / 2)

    # maximum ray of the flower
    a = 4

    x = np.zeros((sample_size, 2))
    y = np.zeros((sample_size, 1), dtype="uint8")

    for j in range(2):
        ix = range(n * j, n * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, n) + np.random.randn(n) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(n) * 0.2  # radius
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    y = y.T

    return x, y


def load_mnist_dataset(sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset. MNIST contains images of handwritten digits from 0 to 9.

    :param sample_size: samples the original dataset (the sample is stratified)
    :return: a tuple of digit images, target labels
    """

    labels_path = datasets_dir / "mnist/t10k-labels-idx1-ubyte"
    with open(labels_path, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
        labels = np.fromfile(file, np.uint8)

    images_path = datasets_dir / "mnist/t10k-images-idx3-ubyte"
    with open(images_path, "rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
        image_data = np.fromfile(file, np.uint8)

    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = image_data[i * rows * cols : (i + 1) * rows * cols]
        images[i] = img.reshape(28, 28)

    return (
        (np.array(images), labels)
        if sample_size is None
        else resample(np.array(images), labels, n_samples=sample_size, stratify=labels)
    )
