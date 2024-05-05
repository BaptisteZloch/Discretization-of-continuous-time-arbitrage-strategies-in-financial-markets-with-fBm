import numpy.typing as npt
import numpy as np
from typing import Union


def convert_increment_to_indice(
    time_increment: float, time_vector: npt.NDArray[np.float64]
) -> int:
    """Given a time increment time_increment such as 0.1, this function convert it to the nearest index in the whole time vector

    Args:
        time_increment (float): Time increment
        time_vector (npt.NDArray[np.float64]): Time vector

    Returns:
        int: The integer index corresponding to i in t.
    """
    return int((np.abs(time_increment - time_vector)).argmin())


def compute_empirical_hurst_exponent(
    s_t: npt.NDArray[np.float64], t: npt.NDArray[np.float64]
) -> float:
    """Compute the empirical hurst exponent for a given time series signal s_t and the corresponding time vector t

    Args:
        s_t (npt.NDArray[np.float64]): The signal or time series
        t (npt.NDArray[np.float64]): The corresponding time vector

    Returns:
        float: The estimated hurst exponennt
    """
    assert (
        s_t.shape == t.shape
    ), "Error provide matching size time vector and signal vector"
    N = s_t.shape[0]
    NUMERATOR = np.sum(
        [
            (
                s_t[convert_increment_to_indice((i + 1) / N, t)]
                - s_t[convert_increment_to_indice((i) / N, t)]
            )
            ** 2
            for i in range(N - 1)
        ]
    )

    DENOMINATOR = np.sum(
        [
            (
                s_t[convert_increment_to_indice(2 * (i + 1) / N, t)]
                - s_t[convert_increment_to_indice((2 * i) / N, t)]
            )
            ** 2
            for i in range((N // 2) - 1)
        ]
    )

    return (-1 / (2 * np.log(2))) * np.log(0.5 * NUMERATOR / DENOMINATOR)


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
