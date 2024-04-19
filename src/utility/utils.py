import numpy.typing as npt
import numpy as np
from typing import Union

def transaction_cost_L(volume_t: float, p_1: float, p_2: float) -> float:
    """Equation 2.17 : define the transaction costs

    Args:
        volume_t (float): Trading volume
        p_1 (float): proportionality factor p1 (in percent)
        p_2 (float): minimum fee p2 (in monetary units)

    Returns:
        float: The charged cost for the volume at t
    """
    return max(volume_t * p_1, p_2) * (volume_t > 0)

def generate_t(
    n_steps: int = 100,
    T: Union[float, int] = 1,
) -> npt.NDArray[np.float64]:
    """Given a length in years T and a number of steps (n_steps) generate equally spaced t indices.

    Args:
        n_steps (int, optional): Number of steps. Defaults to 100.
        T (Union[float, int], optional): Horizon in years.. Defaults to 1.

    Returns:
        npt.NDArray[np.float64]: Equally spaced t indices
    """
    return np.linspace(0, T, num=n_steps)


def a_order_power_mean(x: npt.NDArray[np.float64], a: int = 0) -> np.float64:
    """This function returns the a-order power mean over the vector x for a given a in relatives number.

    Args:
    ----
        x (npt.NDArray[np.float64]): The vector to compute the a-order power mean on.
        a (int, optional): The a-order power. Defaults to 0.

    Returns:
    ----
        np.float64: The power mean.
    """
    d = int(x.shape[0])
    if a == 0:
        return np.prod(x) ** (1 / d)
    elif a == np.inf:
        return np.max(x)
    elif a == -np.inf:
        return np.min(x)
    else:
        return (
            (1 / d) * np.sum(np.apply_along_axis(lambda x_i: x_i**a, axis=0, arr=x))
        ) ** (1 / a)
