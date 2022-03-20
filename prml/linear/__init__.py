# Project
from .bayesian_regression import BayesianRegression
from .evidence_approximation import EvidenceApproximation
from .fisher_linear_discriminant import FisherLinearDiscriminant
from .generative_classifier import GenerativeClassifier
from .least_square_classifier import LeastSquaresClassifier
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .logistic_regression import SoftmaxRegression
from .perceptron import Perceptron
from .ridge_regression import RidgeRegression

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "BayesianRegression",
    "EvidenceApproximation",
    "LeastSquaresClassifier",
    "FisherLinearDiscriminant",
    "Perceptron",
    "GenerativeClassifier",
    "LogisticRegression",
    "SoftmaxRegression",
]
