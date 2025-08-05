# Types
from typing import Any, cast

# Standard Library
from enum import IntEnum

# Dependencies
import numpy as np
from numpy.typing import NDArray


class Axis(IntEnum):
    ROWS = 0
    COLS = 1


def validate_2d(x: NDArray) -> tuple[int, int]:
    """
    Checks that the provided array has only 2 dimensions.

    Args:
        x (NDArray): an array to be validated

    Raises:
        RuntimeError: if the array is not 2-dimensional

    Returns:
        tuple[int, int]: the shape of the array in the form or a rows, columns tuple
    """
    if x.ndim != 2:
        raise RuntimeError(f"Provided array must be 2D, found: {x.shape}")
    return x.shape[0], x.shape[1]


def cast_f(x: Any) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], x)


def cast_uint(x: Any) -> NDArray[np.uint32]:
    return cast(NDArray[np.unsignedinteger], x)


def cast_int(x: Any) -> NDArray[np.int32]:
    return cast(NDArray[np.signedinteger], x)


def cast_bool(x: Any) -> NDArray[np.bool_]:
    return cast(NDArray[np.bool_], x)
