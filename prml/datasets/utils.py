# Standard Library
from collections.abc import Callable

# Dependencies
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_decision_boundary(
    model: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray, step: float = 0.01
) -> None:
    """
    Plots the decision boundary defined by a given classification function on a
    specified 2-dimensional dataset.

    :param model: a classification function
    :param x: (N, 2) array of data points
    :param y: (N, 1) array of labels
    :param step: data step granularity (default is 0.01)
    """
    if x.shape[1] != 2:
        raise ValueError(f"Dataset 'X' should have 2 dimensions but instead '{x.shape[1]}' found.")

    # Find boundaries
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # Create a grid of points having distance step between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    # Predict the function value for the whole grid
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap="coolwarm")
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.show()
