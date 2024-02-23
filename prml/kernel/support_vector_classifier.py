# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.classifier import Classifier


class SupportVectorClassifier(Classifier):
    """
    Support Vector Machine classifier.
    """

    def __init__(self, kernel: Kernel, C: float = 1) -> None:
        """
        Creates a support vector classifier.
        """
        super().__init__()
        self.kernel = kernel
        self.c = C
        self.alpha = None
        self.b = 0
        self._support_vectors = None
        self._support_labels = None

    @property
    def n_support_vectors(self) -> int:
        return 0 if self._support_vectors is None else len(self._support_vectors)

    @property
    def support_vectors(self) -> np.ndarray:
        return np.empty(0) if self._support_vectors is None else self._support_vectors

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 100, epsilon: float = 1e-8) -> None:
        """
        Performs sequential minimal optimization (SMO).

        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target values
        :param n_iter: number of iterations
        :param epsilon: tolerance threshold for KKT conditions
        """

        n_examples, _ = x.shape
        self.alpha = np.zeros(n_examples)
        self.b = 0

        # initially all data are assumed to be support vectors
        self._support_vectors = x
        self._support_labels = t

        # compute Gram matrix and error cache
        gram = self.kernel(x, x) if self.kernel else x @ x.T
        error_cache = self.predict(x)[1] - t

        kkt_indices = np.arange(n_examples)

        passes = 0
        while passes < n_iter:
            i, kkt_indices = self.__first_alpha_heuristic(kkt_indices, epsilon)
            if i is None:
                break

            j = self.__second_choice_heuristic(error_cache, i)
            if i == j:
                continue

            # update boundaries
            L, H = self.__bounds(i, j)
            if L == H:
                continue

            # compute eta
            eta = gram[i][i] + gram[j][j] - 2 * gram[i][j]
            if eta == 0:
                continue

            # calculate predictions and errors
            error_i = self.predict(self._support_vectors[i, :])[1] - self._support_labels[i]
            error_j = self.predict(self._support_vectors[j, :])[1] - self._support_labels[j]

            # update alpha
            alpha_1_old = self.alpha[i]
            alpha_2_old = self.alpha[j]
            self.alpha[j] = alpha_2_old + self._support_labels[j] * (error_i - error_j) / eta
            self.alpha[j] = np.minimum(self.alpha[j], H)
            self.alpha[j] = np.maximum(self.alpha[j], L)
            self.alpha[i] = alpha_1_old + self._support_labels[i] * self._support_labels[j] * (
                alpha_2_old - self.alpha[j]
            )

            # update beta
            self.__update_beta(i, j, gram, error_i, error_j, alpha_1_old, alpha_2_old)

            # update error cache
            error_cache[i] = self.predict(self._support_vectors[i, :])[1] - self._support_labels[i]
            error_cache[j] = self.predict(self._support_vectors[j, :])[1] - self._support_labels[j]

            # adapt errors in order to avoid getting stuck near zero values
            if 0 < self.alpha[i] < self.c:
                error_cache[i] = 0
            if 0 < self.alpha[j] < self.c:
                error_cache[j] = 0

            # next iteration
            passes += 1

        # store only support vectors
        support_vectors_idx = self.alpha != 0
        self._support_labels = self._support_labels[support_vectors_idx]
        self._support_vectors = self._support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output
        :return a tuple of (N,) arrays, one holding the target predictions, and one the y values
        """

        if self.alpha is None or self._support_vectors is None or self._support_labels is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        if x.ndim == 1:
            x = x[None, :]

        k = self.kernel(self._support_vectors, x) if self.kernel else self._support_vectors @ x.T
        f = (self.alpha * self._support_labels) @ k + self.b
        return np.where(f >= 0, 1, -1), f

    def __kkt(self, i: int, epsilon: float):
        t_error = self._support_labels[i] * (self.predict(self._support_vectors[i])[1] - self._support_labels[i])
        cond_a = t_error < -epsilon and self.alpha[i] < self.c
        cond_b = t_error > epsilon and self.alpha[i] > 0
        return cond_a or cond_b

    def __first_alpha_heuristic(self, indices: np.ndarray, epsilon: float):
        for idx in indices:
            indices = np.delete(indices, np.argwhere(indices == idx))
            # check if they violate KKT
            if self.__kkt(idx, epsilon):
                return idx, indices

        indices = np.arange(len(self.alpha))
        indices = np.array([i for i in indices if self.__kkt(i, epsilon)])
        if len(indices) > 0:
            np.random.shuffle(indices)
            return indices[0], indices[1:]
        else:
            return None, indices

    def __second_choice_heuristic(self, error: np.ndarray, i: int):
        # find all alpha that are neither zero or C
        non_bounded_alpha_indices = np.argwhere((0 < self.alpha) & (self.alpha < self.c)).reshape(-1)

        # find the alpha index that maximizes the step size |Ea - Eb|
        if len(non_bounded_alpha_indices) > 0:
            if error[i] >= 0:
                return non_bounded_alpha_indices[np.argmin(error[non_bounded_alpha_indices])]
            else:
                return non_bounded_alpha_indices[np.argmax(error[non_bounded_alpha_indices])]
        else:
            return np.argmax(np.abs(error - error[i]))

    def __bounds(self, i: int, j: int) -> tuple[float, float]:
        if self._support_labels[i] == self._support_labels[j]:
            return (max(0, self.alpha[i] + self.alpha[j] - self.c), min(self.c, self.alpha[i] + self.alpha[j]))
        else:
            return (
                max(0, self.alpha[j] - self.alpha[i]),
                min(self.c, self.c + self.alpha[j] - self.alpha[i]),
            )

    def __update_beta(self, i: int, j: int, gram: np.ndarray, error_i, error_j, alpha_i_old, alpha_j_old) -> None:
        b1 = (
            self.b
            - error_i
            - self._support_labels[i] * (self.alpha[i] - alpha_i_old) * gram[i][i]
            - self._support_labels[j] * (self.alpha[j] - alpha_j_old) * gram[i][j]
        )
        b2 = (
            self.b
            - error_j
            - self._support_labels[i] * (self.alpha[i] - alpha_i_old) * gram[i][j]
            - self._support_labels[j] * (self.alpha[j] - alpha_j_old) * gram[j][j]
        )
        if 0 < self.alpha[i] < self.c:
            self.b = b1
        elif 0 < self.alpha[j] < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
