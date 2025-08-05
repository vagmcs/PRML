# Dependencies
import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, n_clusters: int = 2) -> None:
        if n_clusters < 2:
            raise RuntimeError("The number of clusters cannot be less than 2.")

        self._n_clusters = n_clusters
        self._centers: np.ndarray | None = None
        self._history: list[tuple[np.ndarray, np.ndarray]] = list()

    @property
    def centers(self) -> np.ndarray | None:
        return self._centers

    @property
    def history(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._history

    def fit(self, x: np.ndarray, n_iter: int = 100) -> None:
        n, d = x.shape

        # initialize centers at random
        self._centers = x[np.random.choice(len(x), self._n_clusters, replace=False)]

        # optimize
        for _ in range(n_iter):
            prev_centers = np.copy(self._centers)

            # assign data points to clusters (E-step)
            r = np.zeros((n, self._n_clusters))
            d = cdist(x, self._centers, metric="euclidean")
            assignments = np.argmin(d, axis=-1)
            r = np.eye(self._n_clusters)[assignments]
            self._history.append((self._centers.copy(), assignments.copy()))

            # recompute the centers (M-step)
            self._centers = np.sum(x[:, None, :] * r[:, :, None], axis=0) / r.sum(axis=0)[:, None]

            # check for convergence
            if np.allclose(prev_centers, self._centers):
                break

    def update(self, x: np.ndarray) -> None:
        x = x[None, :] if x.ndim == 1 else x
        _, d = x.shape

        # initialize centers randomly
        if self._centers is None:
            self._centers = np.zeros((self._n_clusters, d))

        is_empty = self._centers.sum(axis=1) == 0
        if any(is_empty):
            i = is_empty.argmax()
            self._centers[i, :] = x

        # assign data points to clusters (E-step)
        d = cdist(x, self._centers, metric="euclidean")
        assignments = np.argmin(d, axis=-1)
        cluster_indicator = np.eye(self._n_clusters)[assignments]
        self._history.append((self._centers.copy(), assignments.copy()))

        # update cluster counts
        if not hasattr(self, "_r"):
            self._r = cluster_indicator
        else:
            self._r += cluster_indicator

        # recompute the centers (M-step)
        eta = 1 / self._r
        eta = np.where(eta == np.inf, 0, eta)
        self._centers += cluster_indicator.T * eta.T * (x - self._centers)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._centers is None:
            raise RuntimeError("Model centers are not initialized. Call 'fit' before 'predict'.")

        distances = cdist(x, self._centers, metric="euclidean")
        return np.asarray(np.argmin(distances, axis=-1))
