# Standard Library
import abc

# Dependencies
import numpy as np

# Project
from .modules import Module

# Find a small float to avoid division by zero
_epsilon = np.finfo(float).eps


def clip(x: np.ndarray) -> np.ndarray:
    return x.clip(min=_epsilon)


class Loss(Module, metaclass=abc.ABCMeta):
    def derivative(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss function given an input and a target.

        :param _input: (N,) array of predictions
        :param _target: (N,) array of target values or classes
        :return: the derivative of the loss
        """
        return self._backwards(_input, _target)


class SSELoss(Loss):
    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the sum-of-squares error

        E = 1/2 Σ(y - t)^2

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target values
        :return: the sum-of-squares error
        """
        return 0.5 * np.sum((_input - _target) ** 2)

    def _backwards(self, _input: np.ndarray, _target: np.ndarray):
        """
        Computes the derivative of the loss over the predicted values.

        dE/dy = y - t

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target values
        :return: the derivative of the loss
        """
        return _input - _target


class BinaryCrossEntropyLoss(Loss):
    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the binary cross-entropy loss.

        E = -Σ tln(y) + (1-t)ln(1-y)

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target classes (0 or 1)
        :return: the binary cross-entropy loss
        """

        loss = np.log(clip(_input)) * _target + np.log(clip(1 - _input)) * (1 - _target)
        return -np.sum(loss) / _target.size

    def _backwards(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss over the predicted values.

        dE/dy = (1-t) / (1-y) - t / y

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target classes (0 or 1)
        :return: the derivative of the loss
        """
        return (1 - _target) / clip(1 - _input) - _target / clip(_input)


class CrossEntropyLoss(Loss):
    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the cross-entropy loss.

        E = -Σ Σ tln(y)

        :param _input: (N, K) array of predicted values
        :param _target: (N, K) array of target classes
        :return: the cross-entropy loss
        """
        n, _ = _input.shape
        loss = np.log(clip(_input)) * _target
        return -np.sum(loss) / n

    def _backwards(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss over the predicted values.

        dE/dy = -t / y

        :param _input: (N, K) array of predicted values
        :param _target: (N, K) array of target classes
        :return: the derivative of the loss
        """
        return -_target / clip(_input)


class GaussianNLLLoss(Loss):
    def __init__(self, n_components: int) -> None:
        self._n_components = n_components
        
    def _gaussian(self, x, mu, variance) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) ** 2 / variance)) / np.sqrt(2 * np.pi * variance)

    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the negative log-likelihood error of the Gaussian mixture.

        E = -Σ ln(Σ pi N(t|mu,sigma))

        :param _input: (N, (L + 2)K) array of concatenated parameters for the Gaussian mixture
        :param _target: (N, L) array of target values
        :return: the negative log-likelihood of the mixture
        """
        component_parameters = np.array_split(_input, self._n_components, axis=1)
        pi, mu, var = component_parameters[0], component_parameters[1], component_parameters[2]
        gaussian_pdf = self._gaussian(_target, mu, var)

        # log-sum-exp
        return -np.log(np.exp(np.log(pi) + np.log(gaussian_pdf)).sum(axis=1)).sum()

    def _backwards(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss with respect to the mixing coefficients, the component means
        and the component variances. Then returns the derivatives as a concatanted gradient vector.

        dE/dpi = pi - gamma

        dE/dmu = gamma ((mu - t) / sigma^2)

        dE/dsigma = gamma (D / sigma - (t - mu)^2 / sigma^3)

        :param _input:  (N, (L + 2)K) array of concatenated parameters for the Gaussian mixture
        :param _target: (N, L) array of target values
        :return: the derivative of the loss
        """

        component_parameters = np.array_split(_input, self._n_components, axis=1)
        pi, mu, var = component_parameters[0], component_parameters[1], component_parameters[2]

        probs = pi * self._gaussian(_target, mu, var)
        probs = probs - np.max(probs, axis=1, keepdims=True)
        exps = np.exp(probs)
        gamma = exps / np.sum(exps, axis=1, keepdims=True)

        dpi = pi - gamma
        dmu = gamma * (mu - _target) / var
        sigma = np.sqrt(var)
        dsigma = gamma * ((1 / sigma) - (_target - mu) ** 2 / sigma ** 3)

        return np.concatenate([dpi, dmu, dsigma], axis=1)
