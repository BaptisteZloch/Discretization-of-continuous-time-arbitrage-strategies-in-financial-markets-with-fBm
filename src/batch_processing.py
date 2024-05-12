from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from typing import List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from backtest.backtester import Backtester
from simulations.fractional_brownian import generate_n_assets_portfolio
from strategy.strategy import SalopekStrategy

N_SIMULATION = 1000
N_WORKERS = 8

ALPHA = -30
BETA = 30

# fees (no fees now)
P1 = 0  # 0.1 proportionality factor p1 (in percent)
P2 = 0  # 0.5 minimum fee p2 (in monetary units)

SCALING_FACTOR = 100  # \gamma


def execute_a_batch(
    n_simulation: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    salopek_strat = SalopekStrategy(
        alpha=ALPHA, beta=BETA, scaling_factor=SCALING_FACTOR
    )
    bk_tester = Backtester()

    V_T_phi = []
    V_T_psi = []
    running_min = []
    V_T_psi_minus_V_T_phi = []

    for _ in tqdm(
        range(n_simulation),
        leave=False,
        desc="Computing MC simulation...",
        total=n_simulation,
    ):
        _, V_t_psi, V_t_phi, _ = bk_tester.run_backtest(
            universe_dataframe=generate_n_assets_portfolio(
                n_assets=2,
                n_steps=250,
                T=1,
                H=[0.9, 0.99],
                mu=0.05,
                sigma=0.1,
                s0=100,
                add_risk_free_asset=False,
                as_dataframe=True,
                brownian_type="fractional",
            ).drop(0),
            allocation_function=salopek_strat.get_allocation,
            p1=P1,
            p2=P2,
        )
        V_T_psi.append(V_t_psi[-1])
        V_T_phi.append(V_t_phi[-1])
        running_min.append(np.min(V_t_phi))
        V_T_psi_minus_V_T_phi.append(V_t_psi[-1] - V_t_phi[-1])

    return V_T_psi, V_T_phi, running_min, V_T_psi_minus_V_T_phi


if __name__ == "__main__":
    start_time = time.time()

    V_T_psi_all = []
    V_T_phi_all = []
    running_min_all = []
    V_T_psi_minus_V_T_phi_all = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        processes = [
            executor.submit(execute_a_batch, N_SIMULATION // N_WORKERS)
            for _ in range(N_WORKERS)
        ]

    for task in as_completed(processes):
        V_T_psi, V_T_phi, running_min, V_T_psi_minus_V_T_phi = task.result()
        V_T_psi_all += V_T_psi
        V_T_phi_all += V_T_phi
        running_min_all += running_min
        V_T_psi_minus_V_T_phi_all += V_T_psi_minus_V_T_phi
    print(
        f"--- Execution: {(time.time() - start_time):2f} seconds for {N_SIMULATION} iterations ---"
    )

    pd.DataFrame(
        {
            "V_T_psi_all": V_T_psi_all,
            "V_T_phi_all": V_T_phi_all,
            "running_min_all": running_min_all,
            "V_T_psi_minus_V_T_phi_all": V_T_psi_minus_V_T_phi_all,
        }
    ).to_csv(
        f".\\results\\salopek\\simulation_result_Salopek_no_fees_h_09_1.csv",
        index=False,
    )
    sys.exit(0)
