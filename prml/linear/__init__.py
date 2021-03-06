from .bayesian_regression import BayesianRegression
from .evidence_approximation import EvidenceApproximation
from .linear_regression import LinearRegression
from .ridge_regression import RidgeRegression

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "BayesianRegression",
    "EvidenceApproximation"
]
