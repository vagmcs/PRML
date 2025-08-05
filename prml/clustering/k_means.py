# Standard Library
from dataclasses import dataclass

# Dependencies
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

# Project
from prml.helpers import array
from prml.helpers.array import Axis


@dataclass(frozen=True)
class Step:
    centers: NDArray[np.floating]
    assignments: NDArray[np.uint32]


class KMeans:
    def __init__(self, n_clusters: int = 2) -> None:
        if n_clusters < 2:
            raise RuntimeError("The number of clusters cannot be less than 2.")

        self._n_clusters = n_clusters
        self._centers: NDArray[np.floating] | None = None

        # keeps track of
        self._history: list[Step] = []

    @property
    def centers(self) -> NDArray[np.floating] | None:
        return self._centers

    @property
    def history(self) -> list[Step]:
        return self._history

    def fit(self, x: NDArray[np.floating], n_iter: int = 100) -> None:
        array.validate_2d(x)

        # initialize centers at random from the dataset
        self._centers = x[np.random.choice(len(x), self._n_clusters, replace=False)]

        # optimize
        for _ in range(n_iter):
            prev_centers = self._centers.copy()

            # assign data points to clusters (E-step)
            assignments = self.predict(x)
            r = np.eye(self._n_clusters)[assignments]

            # update history
            self._history.append(Step(self._centers.copy(), assignments.copy()))

            # recompute the centers (M-step)
            self._centers = array.cast_f(np.sum(x[:, None, :] * r[:, :, None], Axis.ROWS) / r.sum(Axis.ROWS)[:, None])

            # check for convergence
            if np.allclose(prev_centers, self._centers):
                break

    def update(self, x: NDArray[np.floating]) -> None:
        x = x[None, :] if x.ndim == 1 else x
        _, d = x.shape

        # initialize centers to zero
        if self._centers is None:
            self._centers = np.zeros((self._n_clusters, d))

        # if there are zero cluster centers, set the next data point as a center
        is_empty = array.cast_bool(self._centers.sum(Axis.COLS) == 0)
        if any(is_empty):
            self._centers[is_empty.argmax(), :] = x

        # assign data points to clusters (E-step)
        assignments = self.predict(x)
        cluster_indicator = np.eye(self._n_clusters)[assignments]

        # update history
        self._history.append(Step(self._centers.copy(), assignments.copy()))

        # update cluster counts
        if not hasattr(self, "_r"):
            self._r = cluster_indicator
        else:
            self._r += cluster_indicator

        # recompute the centers (M-step)
        eta = 1 / self._r
        eta = np.where(eta == np.inf, 0.0, eta)  # do not update cluster centers that have zero counts
        self._centers += cluster_indicator.T * eta.T * (x - self._centers)

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.unsignedinteger]:
        if self._centers is None:
            raise RuntimeError("Model centers are not initialized. Call 'fit' before 'predict'.")

        distances = cdist(x, self._centers, metric="euclidean")
        return array.cast_uint(np.argmin(distances, Axis.COLS))
