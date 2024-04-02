# Dependencies
import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, n_clusters: int = 2) -> None:
        if n_clusters < 2:
            raise RuntimeError("The number of clusters cannot be less than 2.")

        self._n_clusters = n_clusters
        self._centers: np.ndarray | None = None
        self._history: list[(np.ndarray, np.ndarray)] = list()
        
    @property
    def centers(self) -> np.ndarray:
        return self._centers
    
    @property
    def history(self) -> list[(np.ndarray, np.ndarray)]:
        return self._history

    def fit(self, x: np.ndarray, n_iter: int = 100) -> None:
        n, d = x.shape

        # initialize centers randomly
        self._centers = x[np.random.choice(len(x), self._n_clusters, replace=False)]

        # optimize
        for _ in range(n_iter):
            prev_centers = np.copy(self._centers)

            # assign data points to clusters (E-step)
            r = np.zeros((n, self._n_clusters))
            d = cdist(x, self._centers, metric="euclidean")
            assignments = np.argmin(d, axis=-1)
            r = np.eye(self._n_clusters)[assignments]
            self._history.append((self._centers, assignments))
            
            # recompute the centers (M-step)
            self._centers = np.sum(x[:, None, :] * r[:, :, None], axis=0) / r.sum(axis=0)[:, None]

            # check for convergence
            if np.allclose(prev_centers, self._centers):
                break

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = cdist(x, self._centers, metric="euclidean")
        return np.argmin(d, axis=-1)
