from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import numpy.typing as npt

from utility.utils import a_order_power_mean


class Strategy(ABC):
    def __init__(self, scaling_factor: int = 100) -> None:
        self._scaling_factor = scaling_factor

    @abstractmethod
    def get_allocation(self):
        pass


class SalopekStrategy(Strategy):
    def __init__(self, alpha: int, beta: int, scaling_factor: int = 100) -> None:
        super().__init__(scaling_factor=scaling_factor)
        self.__alpha = alpha
        self.__beta = beta

    def get_allocation(self, s_t: Dict[str, float], *args, **kwargs) -> List[float]:
        """Returns the asset allocation for the next iteration

        Args:
            s_t (Dict[str, float]): The dict of the current assets and prices

        Returns:
            List[float]: The new quantity to hold for the next iteration
        """
        prices = list(s_t.values())
        return [
            self._scaling_factor
            * float(
                SalopekStrategy.__phi_i(
                    a=self.__beta, s_i_t=price, s_t=np.array(prices)
                )
                - SalopekStrategy.__phi_i(
                    a=self.__alpha, s_i_t=price, s_t=np.array(prices)
                )
            )
            for price in prices
        ]

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
    def __init__(self, risk_free_asset_name: str, scaling_factor: int = 100) -> None:
        super().__init__(scaling_factor=scaling_factor)
        self.__risk_free_asset_name = risk_free_asset_name

    def get_allocation(self, s_t: Dict[str, float], *args, **kwargs):
        """Returns the asset allocation for the next iteration

        Args:
            s_t (Dict[str, float]): The dict of the current assets and prices

        Returns:
            List[float]: The new quantity to hold for the next iteration
        """
        assert len(s_t.keys()) == 2, "There must be only 2 assets"
        risky_asset_key = list(
            set(s_t.keys()).difference({self.__risk_free_asset_name})
        )[0]
        return [
            self._scaling_factor
            * (
                (1 / price) * (price**2 - (s_t[risky_asset_key] ** 2))
                if security == self.__risk_free_asset_name
                else (2 / s_t.get(self.__risk_free_asset_name, 1))
                * (price - s_t.get(self.__risk_free_asset_name, 1))
            )
            for security, price in s_t.items()
        ]
