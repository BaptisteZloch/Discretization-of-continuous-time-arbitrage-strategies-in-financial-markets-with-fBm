from abc import ABC, abstractmethod
from functools import reduce
from typing import List
import numpy as np
import numpy.typing as npt

from utility.utils import a_order_power_mean


class Strategy(ABC):
    @abstractmethod
    def get_allocation(self):
        pass


class SalopekStrategy(Strategy):
    def __init__(self, alpha: int, beta: int) -> None:
        self.__alpha = alpha
        self.__beta = beta

    def get_allocation(
        self, asset_i: float, s_t: npt.NDArray[np.float64], *args, **kwargs
    ) -> float:
        """Return allocation for the asset i given price of all asset à time t.

        Args:
        ----
            asset_i (float): The current price of the asset i
            s_t (npt.NDArray[np.float64]): The universe prices.

        Returns:
        ----
            float: The allocation for asset i.
        """
        return float(
            SalopekStrategy.__phi_i(a=self.__beta, s_i_t=asset_i, s_t=s_t)
            - SalopekStrategy.__phi_i(a=self.__alpha, s_i_t=asset_i, s_t=s_t)
        )

    @staticmethod
    def __phi_i(a: int, s_i_t: float, s_t: npt.NDArray[np.float64]) -> np.float64:
        """Equation 2.10

        Args:
        ----
            a (int): a-order power of the mean
            s_i_t (float): The price of asset i at time t
            s_t (npt.NDArray[np.float64]): The array of all assets at time t

        Returns:
        ----
            np.float64: Salopek strategy quantity allocation at time t for asset i.
        """
        d = int(s_t.shape[0])
        return (1 / d) * (((s_i_t / a_order_power_mean(x=s_t, a=a))) ** (a - 1))


class ShiryaevStrategy(Strategy):
    def __init__(self) -> None:
        pass

    def get_allocation(
        self,
        asset_i: float,
        s_t: npt.NDArray[np.float64],
        is_risk_free_asset: bool = False,
        *args,
        **kwargs
    ) -> float:
        assert s_t.shape[0] == 2, "Error strategy must have exactly 2 assets"
        if is_risk_free_asset is True:
            return float((1 / asset_i) * (reduce(lambda x1,x2: x1-x2, map(lambda x: x**2, s_t))))
        return  float((2 / s_t) * (reduce(lambda x1,x2: x1-x2, map(lambda x: x**2, s_t))))
