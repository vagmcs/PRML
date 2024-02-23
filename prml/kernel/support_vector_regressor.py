# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.regression import Regression


class SupportVectorRegressor(Regression):
    """
    Support Vector Machine classifier.
    """

    def __init__(self, kernel: Kernel, epsilon: float, C: float = 1) -> None:
        """
        Creates a support vector classifier.
        """
        super().__init__()
        self.kernel = kernel
        self.c = C
        self._lambda = None
        self.b = 0
        self._support_vectors = None
        self._support_labels = None
        self._epsilon = epsilon

    @property
    def n_support_vectors(self) -> int:
        return 0 if self._support_vectors is None else len(self._support_vectors)

    @property
    def support_vectors(self) -> np.ndarray:
        return np.empty(0) if self._support_vectors is None else self._support_vectors

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 100, tol: float = 1e-3) -> None:
        """
        Performs sequential minimal optimization (SMO).

        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target values
        :param n_iter: number of iterations
        :param epsilon: tolerance threshold for KKT conditions
        """
        
        if x.ndim == 1:
            x = x[:, None]

        n_examples, _ = x.shape
        self._lambda = np.zeros(n_examples)
        self.b = 0

        # initially all data are assumed to be support vectors
        self._support_vectors = x
        self._support_labels = t

        # compute Gram matrix and error cache
        gram = self.kernel(x, x) if self.kernel else x @ x.T
        error_cache = self.predict(x) - t

        kkt_indices = np.arange(n_examples)

        passes = 0
        while passes < n_iter:
            i, kkt_indices = self.__first_alpha_heuristic(kkt_indices, tol)
            if i is None:
                break

            j = self.__second_choice_heuristic(error_cache, i)
            if i == j:
                continue

            # update boundaries
            L, H = self.__bounds(i, j)
            if L == H:
                continue
            
            # NOTE: assume i = u and j = v

            # compute eta
            eta = gram[i][i] + gram[j][j] - 2 * gram[i][j]
            if eta == 0:
                continue
            
            # compute delta
            delta = (2 * self._epsilon) / eta
                        
            # calculate predictions and errors
            error_i = self.predict(self._support_vectors[i, :]) - self._support_labels[i]
            error_j = self.predict(self._support_vectors[j, :]) - self._support_labels[j]

            # update alpha
            lambda_i_old = self._lambda[i]
            lambda_j_old = self._lambda[j]
            self._lambda[j] = lambda_j_old + (error_i - error_j) / eta
            self._lambda[i] = (lambda_i_old + lambda_j_old) - self._lambda[j]
            
            if self._lambda[i] * self._lambda[j] < 0: # if they have the same sign then the last term, involving epsilon, is zero
                if abs(self._lambda[i]) >= delta and abs(self._lambda[j]) >= delta:
                    self._lambda[j] = self._lambda[j] - np.sign(self._lambda[j]) * delta
                else:
                    self._lambda[j] = np.heaviside(abs(self._lambda[j]) - abs(self._lambda[i]), 0) * (lambda_i_old + lambda_j_old)

            self._lambda[j] = np.minimum(np.maximum(self._lambda[j], L), H)
            self._lambda[i] = (lambda_i_old + lambda_j_old) - self._lambda[j]
            
            # update beta
            self.__update_beta(i, j, gram, error_i, error_j, lambda_i_old, lambda_j_old)

            # update error cache
            error_cache[i] = self.predict(self._support_vectors[i, :]) - self._support_labels[i]
            error_cache[j] = self.predict(self._support_vectors[j, :]) - self._support_labels[j]

            # adapt errors in order to avoid getting stuck near zero values
            if -self.c < self._lambda[i] < self.c:
                error_cache[i] = 0
            if -self.c < self._lambda[j] < self.c:
                error_cache[j] = 0

            # next iteration
            passes += 1

        # store only support vectors
        support_vectors_idx = self._lambda != 0
        self._support_labels = self._support_labels[support_vectors_idx]
        self._support_vectors = self._support_vectors[support_vectors_idx, :]
        self._lambda = self._lambda[support_vectors_idx]

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output
        :return a tuple of (N,) arrays, one holding the target predictions, and one the y values
        """

        if self._lambda is None or self._support_vectors is None or self._support_labels is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        if x.ndim == 1:
            x = x[None, :]

        k = self.kernel(self._support_vectors, x) if self.kernel else self._support_vectors @ x.T
        return (self._lambda * self._support_labels) @ k + self.b

    def __kkt(self, i: int, tolerance: float):
        abs_error = abs(self.predict(self._support_vectors[i]) - self._support_labels[i])
        cond_a = self._lambda[i] == 0 and abs_error < self._epsilon + tolerance
        cond_b = abs(self._lambda[i]) == self.c and abs_error > self._epsilon + tolerance
        cond_c = -self.c < self._lambda[i] < self.c and self._lambda[i] != 0 and abs_error <= self._epsilon + tolerance and abs_error >= self._epsilon - tolerance
        
        #abs_error == self._epsilon + tolerance 
        return not(cond_a or cond_b or cond_c)

    def __first_alpha_heuristic(self, indices: np.ndarray, tolerance: float):
        for idx in indices:
            indices = np.delete(indices, np.argwhere(indices == idx))
            # check KKT
            if self.__kkt(idx, tolerance):
                return idx, indices

        indices = np.arange(len(self._lambda))
        indices = np.array([i for i in indices if self.__kkt(i, tolerance)])
        if len(indices) > 0:
            np.random.shuffle(indices)
            return indices[0], indices[1:]
        else:
            return None, indices

    def __second_choice_heuristic(self, error: np.ndarray, i: int):
        # find all alpha that are neither -C or C
        non_bounded_lambda_indices = np.argwhere((-self.c < self._lambda) & (self._lambda < self.c)).reshape(-1)

        # find the alpha index that maximizes the step size |Ea - Eb|
        if len(non_bounded_lambda_indices) > 0:
            if error[i] >= 0:
                return non_bounded_lambda_indices[np.argmin(error[non_bounded_lambda_indices])]
            else:
                return non_bounded_lambda_indices[np.argmax(error[non_bounded_lambda_indices])]
        else:
            return np.argmax(np.abs(error - error[i]))

    def __bounds(self, i: int, j: int) -> tuple[float, float]:
        return (
            max(self._lambda[i] + self._lambda[j] - self.c, -self.c),
            min(self.c, self.c + self._lambda[i] + self._lambda[j]),
        )

    def __update_beta(self, i: int, j: int, gram: np.ndarray, error_i, error_j, lambda_i_old, lambda_j_old) -> None:
        bi = (
            self.b
            - error_i
            - (self._lambda[i] - lambda_i_old) * gram[i][i]
            - (self._lambda[j] - lambda_j_old) * gram[i][j]
        )
        bj = (
            self.b
            - error_j
            - (self._lambda[i] - lambda_i_old) * gram[i][j]
            - (self._lambda[j] - lambda_j_old) * gram[j][j]
        )
        if -self.c < self._lambda[i] < self.c:
            self.b = bi
        elif -self.c < self._lambda[j] < self.c:
            self.b = bj
        else:
            self.b = (bi + bj) / 2
