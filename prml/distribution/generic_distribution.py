import numpy as np
import sympy as sym
from typing import Union, Dict
from prml.distribution import Distribution


class GenericDistribution(Distribution):
    """
    **Generic** distribution.
    """

    def __init__(self, formula: sym.Expr):
        """
        Create a *Generic* distribution.

        :param formula: a symbolic formula
        """
        self._formula = formula

    @property
    def formula(self):
        """
        :return: the symbolic formula of the distribution
        """
        return self._formula

    def change_notation(self, theta_substitution: Dict[str, str]) -> 'GenericDistribution':
        """
        Change the notation of variables in the PDF function.

        :param theta_substitution: a dictionary mapping variable names into other names
        :return: a GenericDistribution object having changed variable names
        """
        return GenericDistribution(self._formula.subs(theta_substitution))

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        raise Exception("Cannot apply maximum likelihood estimation on a generic distribution.")

    def pdf(self, **kwargs) -> Union[np.ndarray, float]:
        """
        Compute the likelihood of the distribution on the given data, assuming
        that the data are independent and identically distributed.

        :param kwargs: a dictionary of variables to data values
        :return: the likelihood of the distribution
        """
        free_variables = list(self._formula.free_symbols)
        if not all([str(v) in kwargs for v in free_variables]) or len(free_variables) != len(kwargs):
            raise ValueError("Data not given for all variables.")

        return (sym.lambdify(free_variables, self._formula))(*[kwargs[str(v)] for v in free_variables])

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        raise Exception("Sampling from a generic distribution is not supported.")

    def __mul__(self, other: Distribution) -> 'GenericDistribution':
        """
        Symbolic multiplication of distributions.

        :param other: another distribution
        :return: a GenericDistribution object
        """
        return GenericDistribution(self._formula * other.formula)
