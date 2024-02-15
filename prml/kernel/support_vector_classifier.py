# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.classifier import Classifier


class SupportVectorClassifier(Classifier):
    """
    Support Vector Machine model.
    Sequential Minimal Optimization.
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
    """

    def __init__(self, kernel: Kernel, C: float = 1, max_iter=100) -> None:
        super().__init__()
        self.kernel = kernel
        self.alpha = None
        self.b = None
        self.support_labels = None
        self.support_vectors = None
        self.c = C
        self.max_iter = max_iter
        self.kkt_thr = 1e-8

        # self.b = np.array([])  # SVM's threshold
        # self.alpha = np.array([])  # Alpha parameters of the support vectors
        # self.support_vectors = np.array([])  # Matrix in which each row is a support vector
        # self.support_labels = np.array([])  # Vector with the ground truth labels of the support vectors

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim == 1:
            x = x[None, :]

        k = self.kernel(self.support_vectors, x) if self.kernel else self.support_vectors @ x.T
        # print(self.support_vectors.shape, x.shape, k.shape, (self.alpha * self.support_labels).shape)
        f = (self.alpha * self.support_labels) @ k + self.b
        return np.where(f >= 0, 1, -1), f

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train

        gram = self.kernel(x_train, x_train) if self.kernel else x_train @ x_train.T
        non_kkt_array = np.arange(N)  # Iterate all indices first
        error_cache = self.predict(x_train)[1] - y_train  # auxilary array for heuristics

        passes = 0
        while passes < self.max_iter:
            i_1, non_kkt_array = self.__first_alpha_heuristic(non_kkt_array, self.kkt_thr)
            if i_1 is None:
                break

            i_2 = self.__second_choice_heuristic(error_cache, i_1)
            if i_1 == i_2:
                continue

            # Update boundaries
            L, H = self.__bounds(i_1, i_2)
            if L == H:
                continue

            # Compute eta
            eta = gram[i_1][i_1] + gram[i_2][i_2] - 2 * gram[i_1][i_2]
            if eta == 0:
                continue

            # Calculate predictions and errors for x_1 and x_2
            E_1 = self.predict(self.support_vectors[i_1, :])[1] - self.support_labels[i_1]
            E_2 = self.predict(self.support_vectors[i_2, :])[1] - self.support_labels[i_2]

            # Update alpha
            alpha_1_old = self.alpha[i_1]
            alpha_2_old = self.alpha[i_2]
            self.alpha[i_2] = alpha_2_old + self.support_labels[i_2] * (E_1 - E_2) / eta
            self.alpha[i_2] = np.minimum(self.alpha[i_2], H)
            self.alpha[i_2] = np.maximum(self.alpha[i_2], L)
            self.alpha[i_1] = alpha_1_old + self.support_labels[i_1] * self.support_labels[i_2] * (
                alpha_2_old - self.alpha[i_2]
            )

            # Update beta
            self.__update_beta(i_1, i_2, gram, E_1, E_2, alpha_1_old, alpha_2_old)

            # Update error cache
            error_cache[i_1] = self.predict(self.support_vectors[i_1, :])[1] - self.support_labels[i_1]
            error_cache[i_2] = self.predict(self.support_vectors[i_2, :])[1] - self.support_labels[i_2]

            # fix the errors in order to avoid getting stuck
            if 0 < self.alpha[i_1] < self.c:
                error_cache[i_1] = 0
            if 0 < self.alpha[i_2] < self.c:
                error_cache[i_2] = 0

            passes += 1

        # Store only support vectors
        support_vectors_idx = self.alpha != 0
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        print(f"Training summary: {passes} iterations, {self.alpha.shape[0]} supprts vectors")

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

    def __first_alpha_heuristic(self, indices: np.ndarray, epsilon: float):
        for idx in indices:
            indices = np.delete(indices, np.argwhere(indices == idx))
            # check KKT
            if self.__kkt(idx, epsilon):
                return idx, indices

        indices = np.arange(len(self.alpha))
        indices = np.array([i for i in indices if self.__kkt(i, epsilon)])
        if len(indices) > 0:
            np.random.shuffle(indices)
            return indices[0], indices[1:]
        else:
            return None, indices

    def __kkt(self, i: int, epsilon: float):
        t_error = self.support_labels[i] * (self.predict(self.support_vectors[i])[1] - self.support_labels[i])
        cond_a = t_error < -epsilon and self.alpha[i] < self.c
        cond_b = t_error > epsilon and self.alpha[i] > 0
        return cond_a or cond_b

    def __bounds(self, i: int, j: int) -> tuple[float, float]:
        if self.support_labels[i] == self.support_labels[j]:
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
            - self.support_labels[i] * (self.alpha[i] - alpha_i_old) * gram[i][i]
            - self.support_labels[j] * (self.alpha[j] - alpha_j_old) * gram[i][j]
        )
        b2 = (
            self.b
            - error_j
            - self.support_labels[i] * (self.alpha[i] - alpha_i_old) * gram[i][j]
            - self.support_labels[j] * (self.alpha[j] - alpha_j_old) * gram[j][j]
        )
        if 0 < self.alpha[i] < self.c:
            self.b = b1
        elif 0 < self.alpha[j] < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
