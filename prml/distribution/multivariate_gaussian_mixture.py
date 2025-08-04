# Types
from typing import Optional, Union

# Dependencies
import numpy as np

from .generic_distribution import GenericDistribution
from .multivariate_gaussian import MultivariateGaussian


class MultivariateGaussianMixture(GenericDistribution):
    """
    The Multivariate **Gaussian** mixture distribution:

    p(x) = sum_{k=1}^K pi_k N(x|mu_k, cov_k)
    """

    def __init__(
        self,
        n_components: int,
        coefficients: Optional[np.ndarray] = None,
        mu: Optional[list[np.ndarray]] = None,
        cov: Optional[list[np.ndarray]] = None,
    ) -> None:
        self._n_components = n_components
        self._coefficients = coefficients
        if coefficients is not None and mu is not None and cov is not None:
            self._components = [MultivariateGaussian(mu, cov).formula for pi in coefficients]
            formula = sum([pi * gaussian.formula for (pi, gaussian) in zip(coefficients, self._components)])
            super().__init__(formula)
        else:
            self._components = None
            super().__init__(None)

    def ml(self, x: np.ndarray, k_means_init: bool = False) -> None:
        """
        Performs maximum likelihood estimation on the parameters using the given data by
        running the EM algorithm.

        :param x: an (N, D) array of data values
        :param k_means_init: initializes parameters using k-means
        """
        n, d = x.shape

        # initialize the coefficients
        _coefficients = np.ones(self._n_components) / self._n_components

        # initialize the means randomly
        _means = x[np.random.randint(x.shape[0], size=self._n_components), :]

        # initialize the covariances
        _covariances = [np.eye(d) for _ in range(self._n_components)]

        # initial evaluation of the log-likelihood
        prev_log_likelihood = 0
        log_likelihood = sum(
            _coefficients
            * np.array(
                [
                    np.diag(MultivariateGaussian(_means[i][:, None], _covariances[i]).pdf(x.T))
                    for i in range(self._n_components)
                ]
            ).T
        )[0]

        while log_likelihood != prev_log_likelihood:
            # E-step
            responsibilities = (
                _coefficients
                * np.array(
                    [
                        np.diag(MultivariateGaussian(_means[i][:, None], _covariances[i]).pdf(x.T))
                        for i in range(self._n_components)
                    ]
                ).T
            )
            responsibilities /= responsibilities.sum(axis=-1, keepdims=True)

            # M-step
            n_k = responsibilities.sum(axis=0)
            _coefficients = n_k / n
            _means = (x.T @ responsibilities / n_k).T
            _covariances = [
                ((responsibilities[:, i, None] * (x - _means[i])).T @ (x - _means[i])) / n_k
                for i in range(self._n_components)
            ]

            # re-evaluate the log-likelihood
            prev_log_likelihood = log_likelihood
            log_likelihood = sum(
                _coefficients
                * np.array(
                    [
                        np.diag(MultivariateGaussian(_means[i][:, None], _covariances[i]).pdf(x.T))
                        for i in range(self._n_components)
                    ]
                ).T
            )[0]

        self._coefficients = _coefficients
        self._components = [
            MultivariateGaussian(_means[i][:, None], _covariances[i]) for i in range(self._n_components)
        ]

    def pdf(self, x: np.ndarray) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function
        (PMF) of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self._coefficients is None:
            raise ValueError("The Gaussian mixture is undefined.")
        else:
            return sum([pi * gaussian.pdf(x) for (pi, gaussian) in zip(self._coefficients, self._components)])

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        pass
