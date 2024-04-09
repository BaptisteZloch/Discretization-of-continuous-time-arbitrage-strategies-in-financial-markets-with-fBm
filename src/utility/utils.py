import numpy.typing as npt
import numpy as np
from typing import Union


def generate_t(
    n_steps: int = 100,
    T: Union[float, int] = 1,
) -> npt.NDArray[np.float32]:
    """Given a length in years T and a number of steps (n_steps) generate equally spaced t indices.

    Args:
        n_steps (int, optional): Number of steps. Defaults to 100.
        T (Union[float, int], optional): Horizon in years.. Defaults to 1.

    Returns:
        npt.NDArray[np.float32]: Equally spaced t indices
    """
    return np.linspace(0, T, num=n_steps)
