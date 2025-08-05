# Dependencies
import numpy as np

# Project
from prml.distribution.multivariate_gaussian import MultivariateGaussian

from .classifier import Classifier


class GenerativeClassifier(Classifier):
    def __init__(self) -> None:
        """
        Creates a generative classifier.
        """
        self._class_priors: list[float] = []
        self._class_densities: list[MultivariateGaussian] = []

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the classifier.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target classes
        """
        n_classes = np.max(t) + 1
        n = np.zeros(n_classes)
        mu, sigma = [], []
        for k in range(n_classes):
            x_k = x[t == k]
            n[k] = x_k.shape[0]
            self._class_priors.append(n[k] / x.shape[0])
            mu.append((1 / n[k]) * np.sum(x_k, axis=0))
            sigma.append((1 / n[k]) * (x_k - mu[k]).T @ (x_k - mu[k]))

        shared_sigma = sum([(n[k] / np.sum(n)) * sigma[k] for k in range(n_classes)])
        for k in range(n_classes):
            self._class_densities.append(MultivariateGaussian(mu[k][:, None], shared_sigma))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predicted classes
        """
        alpha_k = []
        for prior, density in zip(self._class_priors, self._class_densities):
            alpha_k.append(np.diag(density.pdf(x)) * prior)  # type: ignore

        alpha_k = np.array(alpha_k)  # type: ignore
        return np.argmax(alpha_k / np.sum(alpha_k, axis=0), axis=0)  # type: ignore
