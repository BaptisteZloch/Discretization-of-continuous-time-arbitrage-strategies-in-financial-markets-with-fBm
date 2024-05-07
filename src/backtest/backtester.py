from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import numpy.typing as npt
from utility.utils import transaction_cost_L


class Backtester:
    def __init__(
        self,
        universe_dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        self.__universe_dataframe = universe_dataframe

    def __handle_empty_dataframe(
        self,
        universe_dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Handle the optionality on the universe_dataframe argument.

        Args:
            universe_dataframe (Optional[pd.DataFrame], optional): The optional universe to use. Defaults to None.

        Raises:
            ValueError: No dataframe provided at all.

        Returns:
            pd.DataFrame: The dataframe to use.
        """
        if universe_dataframe is None and self.__universe_dataframe is not None:
            return self.__universe_dataframe
        if universe_dataframe is not None and isinstance(
            universe_dataframe, pd.DataFrame
        ):
            return universe_dataframe

        raise ValueError(
            "Error provide universe_dataframe either in the constructor or in the run_backtest method."
        )

    def run_backtest(
        self,
        allocation_function: Callable[
            [Dict[str, float]],
            List[float],
        ],
        p1: float = 0.1,
        p2: float = 0.5,
        universe_dataframe: Optional[pd.DataFrame] = None,
    ) -> Tuple[
        pd.DataFrame,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """_summary_

        Args:
            allocation_function (Callable[ [float,  npt.NDArray[np.float64],List[str]], float, ]): _description_
            p1 (float, optional): Proportionality factor p1 (in percent). Defaults to 0.1.
            p2 (float, optional): Minimum fee p2 (in monetary units). Defaults to 0.5.
            universe_dataframe (Optional[pd.DataFrame], optional): _description_. Defaults to None.
        """
        self.__universe_dataframe = self.__handle_empty_dataframe(
            universe_dataframe=universe_dataframe
        )
        quantities = []  # Phi_i_t
        volumes = []  # \Gamma_t

        transaction_costs = []  # L_t
        transaction_account = []  # D_t
        transaction_account_qty = []  # \Phi_{t}^{d+1}

        V_t_psi = []  # Using equation 2.14/2.22
        V_t_phi = []  # Using equation 2.21

        for (
            index,
            row,
        ) in (
            self.__universe_dataframe.iterrows()
        ):  ###################### Compute the new quantity (Equation 2.10) ###############################
            if index != self.__universe_dataframe.index[-1]:
                quantities.append(allocation_function(row.to_dict()))
            else:
                quantities.append([0] * len(row))
            ###################### Volume section (Equation 2.26) ###############################
            if index == self.__universe_dataframe.index[0]:  # Repurchasing
                volumes.append(
                    np.array(tuple(map(abs, quantities[-1]))) @ row.to_numpy()
                )
            elif index == self.__universe_dataframe.index[-1]:  # liquidating
                volumes.append(
                    np.array(tuple(map(abs, quantities[-2]))) @ row.to_numpy()
                )
            else:
                volumes.append(
                    np.array(
                        tuple(
                            map(
                                abs, np.array(quantities[-1]) - np.array(quantities[-2])
                            )
                        )
                    )
                    @ row.to_numpy()
                )

            ###################### Transaction cost section ###############################
            # Equation 2.17 : L_t^\Phi
            transaction_costs.append(transaction_cost_L(volumes[-1], p_1=p1, p_2=p2))
            ###################### Transaction account section ###############################
            # Equation 2.19 : D_t^\Phi
            if (
                index != self.__universe_dataframe.index[0]
                or index != self.__universe_dataframe.index[-1]
            ):  # Because 1.19 n between 1 and N-1
                try:
                    transaction_account.append(
                        (np.array(quantities[-1]) - np.array(quantities[-2]))
                        @ row.to_numpy()
                    )
                except:
                    transaction_account.append(
                        (np.array(quantities[-1]) - 0) @ row.to_numpy()
                    )

            else:
                transaction_account.append(0)

            ###################### Transaction account quantity section (Equation 2.20) ###############################
            # Equation 2.20 : \Phi_t^{d+1}
            if index == self.__universe_dataframe.index[0]:
                transaction_account_qty.append(-transaction_costs[-1])

            elif index == self.__universe_dataframe.index[-1]:
                net_revenue = (
                    np.array(quantities[-2]) @ row.to_numpy()
                ) - transaction_costs[
                    -1
                ]  # Equation 2.18 : R^\Gamma
                transaction_account_qty.append(
                    transaction_account_qty[-1] + net_revenue
                )

            else:
                transaction_account_qty.append(
                    transaction_account_qty[-1]
                    - transaction_account[-1]
                    - transaction_costs[-1]
                )
            ###################### Portfolio value ###############################
            # Using equation 2.21
            V_t_phi.append(
                np.array(quantities[-1]) @ row.to_numpy()
                + 1 * transaction_account_qty[-1]
            )  # Discrete
            # Using equation 2.22
            if index != self.__universe_dataframe.index[-1]:
                V_t_psi.append(V_t_phi[-1] - transaction_account_qty[-1])  # Continuous
            else:
                V_t_psi.append(
                    V_t_phi[-1] + (V_t_psi[-1] - V_t_phi[-2])
                )  # Valeur fictive pour faire beau

        return (
            pd.DataFrame(
                quantities,
                index=self.__universe_dataframe.index,
                columns=[f"phi_{i}" for i in range(1, len(quantities[0]) + 1)],
            ),
            np.array(V_t_psi),
            np.array(V_t_phi),
            np.array(transaction_account),
        )
