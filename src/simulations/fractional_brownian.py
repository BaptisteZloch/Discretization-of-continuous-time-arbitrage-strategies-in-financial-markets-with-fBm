# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:33:44 2024

@author: jvilp
"""

from typing import Iterable, List, Literal, Tuple, Union
import pandas as pd
import numpy as np
import numpy.typing as npt
from math import pi

from utility.utils import generate_t


def generate_n_assets_portfolio(
    n_assets: int,
    n_steps: int = 100,
    T: Union[float, int] = 1,
    H: Union[float, List[float]] = 0.7,
    mu: float = 0.07,
    sigma: float = 0.2,
    s0: Union[float, int] = 100,
    add_risk_free_asset: bool = True,
    as_dataframe: bool = True,
    brownian_type: Literal["standard", "fractional"] = "fractional",
) -> Union[pd.DataFrame, npt.NDArray[np.float64]]:
    """Generate a portfolio of n assets each asset is independant and follows a geometrics brownian motion.

    Args:
    ----
        n_assets (int): The number of assets in the portfolio.
        n_steps (int, optional): Number of step of the simulation. Defaults to 100.
        T (Union[float, int], optional): Number of years. Defaults to 1.
        H (float, optional): The hurst exponent between 0 and 1. Defaults to 0.7.
        mu (float, optional): The drift term. Defaults to 0.07.
        sigma (float, optional): The sigma term. Defaults to 0.2.
        s0 (Union[float, int], optional): The initial price à time 0. Defaults to 100.
        add_risk_free_asset (bool, optional): Whether to add the risk free asset =1 as the first column of the dataframe. Defaults to True.
        as_dataframe (bool, optional): The assets price processes in Pandas DataFrame or Numpy array. Defaults to True.
        brownian_type (Literal[&quot;standard&quot;, &quot;fractional&quot;], optional): _description_. Defaults to "fractional".

    Returns:
    ----
        Union[pd.DataFrame, npt.NDArray[np.float64]]:  The assets price processes in Pandas DataFrame or Numpy array
    """
    if isinstance(H, List):
        assert len(H) == n_assets,"Error provide a valid number of parameter in the list"
    # if isinstance(mu, List):
    #     assert len(mu) == n_assets,"Error provide a valid number of parameter in the list"
    # if isinstance(sigma, List):
    #     assert len(sigma) == n_assets,"Error provide a valid number of parameter in the list"
        portfolio_paths = np.array(
            [
                generate_brownian_path(
                    n_steps=n_steps,
                    T=T,
                    H=H[i],
                    mu=mu,
                    sigma=sigma,
                    s0=s0,
                    brownian_type=brownian_type,
                )[-1]
                for i in range(n_assets)
            ]
        ).T
    else:
        portfolio_paths = np.array(
            [
                generate_brownian_path(
                    n_steps=n_steps,
                    T=T,
                    H=H,
                    mu=mu,
                    sigma=sigma,
                    s0=s0,
                    brownian_type=brownian_type,
                )[-1]
                for _ in range(n_assets)
            ]
        ).T
    if add_risk_free_asset:
        portfolio_paths = np.hstack((np.ones(shape=(n_steps, 1)) * s0, portfolio_paths))

    if as_dataframe:
        return pd.DataFrame(
            portfolio_paths,
            index=generate_t(n_steps=n_steps, T=T),
            columns=[
                f"S_{i if add_risk_free_asset else i+1}_t"
                for i in range(n_assets + 1 if add_risk_free_asset else n_assets)
            ],
        )
    return portfolio_paths


def generate_brownian_path(
    n_steps: int = 100,
    T: Union[float, int] = 1,
    H: float = 0.7,
    mu: float = 0.07,
    sigma: float = 0.2,
    s0: Union[float, int] = 100,
    brownian_type: Literal["standard", "fractional"] = "fractional",
    fbm_simulation_method: Literal["cholesky", "spectral_density"] = "cholesky",
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate a geometric brownian motion path

    Args:
    ----
        n_steps (int, optional): Number of step of the simulation. Defaults to 100.
        H (float, optional): The hurst exponent between 0 and 1. Defaults to 0.7.
        T (Union[float, int], optional): Number of years. Defaults to 1.
        mu (float, optional): The drift term. Defaults to 0.07.
        sigma (float, optional): The sigma term. Defaults to 0.2.
        s0 (Union[float, int], optional): The initial price à time 0. Defaults to 100.
        brownian_type (Literal[&quot;standard&quot;, &quot;fractional&quot;], optional): _description_. Defaults to "fractional".

    Returns:
    ----
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: A tuple of first the time steps and the simulated price
    """
    dt = T / n_steps
    t = np.linspace(0, T, num=n_steps)
    if brownian_type == "standard":
        db_t = np.random.normal(0, np.sqrt(dt), size=(n_steps))
        db_t[0] = 0  # B_0 = 0
        B_t = np.cumsum(db_t)  # Construct B_t
    else:
        B_t = fBM_simul(
            T=T, N=n_steps, H=H, fbm_simulation_method=fbm_simulation_method
        )
    S_t = s0 * np.exp(mu * t + sigma * B_t)  # Geometric transformation
    return t, S_t


def fbm_covariance_function(s: np.float64, t: np.float64, H: float) -> float:
    """Covariance function for the fractional brownian motion

    Args:
        s (int): The increment i-1
        t (int): The increment i
        H (float): The Hurst exponent

    Returns:
        float: The covariance between B_s and B_t.
    """
    return 0.5 * (
        np.power(s, 2.0 * H) + np.power(t, 2.0 * H) - np.power(np.abs(t - s), 2.0 * H)
    )


def fBM_simul(
    T: float,
    N: int,
    H: float,
    fbm_simulation_method: Literal["cholesky", "spectral_density"] = "cholesky",
) -> npt.NDArray[np.float32]:
    """Spectral simulation of fBM (Appendix B)

    Args:
        T (float): Time hoizon
        N (int): nb of time step
        H (float): Hurst exponent in [0, 1]

    Returns:
        npt.NDArray[np.float32]: _description_
    """
    if fbm_simulation_method == "spectral_density":
        if (
            N % 2 != 0
        ):  # if N is not pair we adjust it to ensure 0.5*N is include in integer and that delta stay the same
            T = (T / N) * (N + 1)
            N = N + 1
            shift = 1
        else:
            shift = 0

        dt = T / N
        phi = np.random.uniform(low=0, high=2 * pi, size=N)
        W_increment = [compute_Wk(k, N, H, phi) for k in range(0, N)]

        W = dt**H * np.cumsum(W_increment)
        W = np.array(W[: len(W) - shift])
        W[0] = 0
        return W

    elif fbm_simulation_method == "cholesky":
        time_grid = np.linspace(0.0, T, N, dtype=np.float64)[1:]
        fractional_covariance_matrix = np.matrix(
            [
                [
                    fbm_covariance_function(time_grid[i], time_grid[j], H)
                    for j in np.arange(N - 1)
                ]
                for i in np.arange(N - 1)
            ]
        )
        gaussian = np.matrix(np.random.normal(0.0, 1.0, N - 1))
        W = np.linalg.cholesky(fractional_covariance_matrix) * gaussian.T
        return np.insert(W.A1, 0, 0)
    else:
        raise ValueError("Error provide a valid method for generating the fBm.")


def compute_Wk(k: int, N: int, H: float, phi: npt.NDArray[np.float64]) -> float:
    """Compute fBM increment (B.2)

    Args:
        k (int): iteration over N
        N (int): nb of time step
        H (float): Hurst exponent range(0,1)

    Returns:
        float: The increment
    """
    return sum(
        map(
            lambda j: np.sqrt(2 / N)
            * (compute_Sf(j / N, N, H) ** 0.5)
            * (
                np.cos(2 * pi * j * k / N) * np.cos(phi[int(j + 0.5 * N)])
                - np.sin(2 * pi * j * k / N) * np.sin(phi[int(j + 0.5 * N)])
            ),
            range(int(-0.5 * N), int(0.5 * N)),
        )
    )  # Wk


def compute_Sf(f: float, N: int, H: float) -> float:
    """Power sprectral density aproximation (B.3)

    Args:
        f (float): frequency
        N (int): nb of time step
        H (float): Hurst exponent range(0,1)

    Returns:
        float: _description_
    """

    return 0.5 * sum(
        map(
            lambda m: (
                abs(m + 1) ** (2 * H) + abs(m - 1) ** (2 * H) - 2 * abs(m) ** (2 * H)
            )
            * np.cos(2 * pi * m * f),
            range(int(-0.5 * N), int(0.5 * N)),
        )
    )  # Sf


### Rendering
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 1  # time period
    N = 100  # nb of point
    H = 0.6  # hurst exponent
    fBM = fBM_simul(1, 100, 0.6)

    plt.plot(np.linspace(0, T, N), fBM, label="fBM")
    plt.title(f"fractional Brownian Motion with H={H}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
