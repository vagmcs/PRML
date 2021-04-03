from typing import Dict, Union

import numpy as np
import sympy as sym
from distribution import Distribution


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
        # Map vector variables into matrix symbols
        theta = {
            v: sym.MatrixSymbol(theta_substitution[str(v)], v.shape[0], v.shape[1])
            if isinstance(v, sym.MatrixSymbol) else theta_substitution[str(v)]
            for v in self._formula.free_symbols
        }

        return GenericDistribution(self._formula.subs(theta))

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

        result = np.array([sym.lambdify(free_variables, self._formula)])
        for v in free_variables:
            if isinstance(v, sym.MatrixSymbol):
                result = np.array([f(sym.Matrix(x)) for x in kwargs[str(v)] for f in result])
            else:
                result = np.array([f(kwargs[str(v)]) for f in result][0])

        return result.astype(float)

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
