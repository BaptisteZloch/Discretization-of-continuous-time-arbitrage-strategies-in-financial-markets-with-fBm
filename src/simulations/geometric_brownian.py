import numpy.typing as npt
import numpy as np
from typing import Tuple, Union

import pandas as pd
from tqdm import tqdm

from utility.utils import generate_t


def generate_brownian_path(
    n_steps: int = 100,
    T: Union[float, int] = 1,
    mu: Union[float, int] = 0.07,
    sigma: Union[float, int] = 0.2,
    s0: Union[float, int] = 100,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate a geometric brownian motion path

    Args:
    ----
        n_steps (int, optional): Number of step of the simulation. Defaults to 100.
        T (Union[float, int], optional): Number of years. Defaults to 1.
        mu (Union[float, int], optional): The drift term. Defaults to 0.07.
        sigma (Union[float, int], optional): The sigma term. Defaults to 0.2.
        s0 (Union[float, int], optional): The initial price à time 0. Defaults to 100.

    Returns:
    ----
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: A tuple of first the time steps and the simulated price
    """
    dt = T / n_steps
    t = np.linspace(0, T, num=n_steps)
    db_t = np.random.normal(0, np.sqrt(dt), size=(n_steps))
    db_t[0] = 0  # B_0 = 0
    B_t = np.cumsum(db_t)  # Construct B_t
    S_t = s0 * np.exp(mu * t + sigma * B_t)  # Geometric transformation
    return t, S_t


def generate_n_assets_portfolio(
    n_assets: int,
    n_steps: int = 100,
    T: Union[float, int] = 1,
    mu: Union[float, int] = 0.07,
    sigma: Union[float, int] = 0.2,
    s0: Union[float, int] = 100,
    add_risk_free_asset: bool = True,
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, npt.NDArray]:
    """_summary_

    Args:
    ----
        n_assets (int): The number of assets in the portfolio.
        n_steps (int, optional): Number of step of the simulation. Defaults to 100.
        T (Union[float, int], optional): Number of years. Defaults to 1.
        mu (Union[float, int], optional): The drift term. Defaults to 0.07.
        sigma (Union[float, int], optional): The sigma term. Defaults to 0.2.
        s0 (Union[float, int], optional): The initial price à time 0. Defaults to 100.
        add_risk_free_asset (bool, optional): _description_. Defaults to True.
        as_dataframe (bool, optional): _description_. Defaults to True.

    Returns:
    ----
        Union[pd.DataFrame, npt.NDArray]: _description_
    """
    portfolio_paths = np.array(
        [
            generate_brownian_path(n_steps=n_steps, T=T, mu=mu, sigma=sigma, s0=s0)[-1]
            for _ in tqdm(
                range(n_assets), desc="generate all paths of the portfolio", leave=False
            )
        ]
    ).T

    if add_risk_free_asset:
        portfolio_paths = np.hstack((np.ones(shape=(n_steps, 1)) * s0, portfolio_paths))

    if as_dataframe:
        return pd.DataFrame(
            portfolio_paths,
            index=generate_t(n_steps=n_steps, T=T),
            columns=[
                f"S_t_{i if add_risk_free_asset else i+1}"
                for i in range(n_assets + 1 if add_risk_free_asset else n_assets)
            ],
        )
    return portfolio_paths