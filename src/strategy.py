import numpy.typing as npt
import numpy as np
from typing import Union
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from simulations.fractional_brownian import generate_brownian_path, generate_n_assets_portfolio
from utility.utils import generate_t, a_order_power_mean, transaction_cost_L


t, s_t = generate_brownian_path(n_steps=250, T=1, H=0.8, mu=0.15, sigma=0.2, s0=100, brownian_type="fractional")

df_ptf = generate_n_assets_portfolio(n_assets=2, n_steps=250, T=1, H=0.7, mu=0.05, sigma=0.1, s0=100, add_risk_free_asset=False, as_dataframe=True, brownian_type="fractional")


def phi_i(a: int, s_i_t: float, s_t: npt.NDArray[np.float64]) -> np.float64:
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


ALPHA = -30
BETA = 30

# fees (no fees now)
P1 = 0.1  # 0.1 proportionality factor p1 (in percent)
P2 = 0.5  # 0.5 minimum fee p2 (in monetary units)

SCALING_FACTOR = 100  # \gamma

# quantities = [[0.0 for _ in range(df_ptf.shape[-1])]]  # Phi_i_t
quantities = []  # Phi_i_t
volumes = []  # \Gamma_t

transaction_costs = []  # L_t
transaction_account = []  # D_t
transaction_account_qty = []  # \Phi_{t}^{d+1}

V_t_psi = []  # Using equation 2.14/2.22
V_t_phi = []  # Using equation 2.21

for index, row in tqdm(df_ptf.iterrows(), desc="Running the strategy...", leave=False):
###################### Compute the new quantity (Equation 2.10) ###############################
    new_quantity = [
        SCALING_FACTOR
        * float(
            phi_i(a=BETA, s_i_t=asset_i, s_t=row.to_numpy())
            - phi_i(a=ALPHA, s_i_t=asset_i, s_t=row.to_numpy())
        )
        for asset_i in row
    ]
    quantities.append(new_quantity)

N = len(quantities) - 1
for n, quant in enumerate(quantities):
    ###################### Volume section (Equation 2.26) ###############################
    if n == 0: # Repurchasing
        volumes.append(np.array(tuple(map(abs, quantities[0]))) @ df_ptf.iloc[0].to_numpy())
    elif n == N:  # liquidating
        volumes.append(np.array(tuple(map(abs, quantities[N]))) @ df_ptf.iloc[N].to_numpy())
    else:
        volumes.append(np.array(tuple(map(abs, np.array(quantities[n]) - np.array(quantities[n-1])))) @ df_ptf.iloc[n-1].to_numpy())

    ###################### Transaction cost section ###############################
    # Equation 2.17 : L_t^\Phi
    transaction_costs.append(transaction_cost_L(volumes[-1], p_1=P1, p_2=P2))

    ###################### Transaction account section ###############################
    # Equation 2.19 : D_t^\Phi
    if n != 0 or n != N:  # Because 1.19 n between 1 and N-1
        transaction_account.append((np.array(quantities[n]) - np.array(quantities[n-1])) @ df_ptf.iloc[n-1].to_numpy())
    else:
        transaction_account.append(0)

    ###################### Transaction account quantity section (Equation 2.20) ###############################
    # Equation 2.20 : \Phi_t^{d+1}
    if n == 0:
        transaction_account_qty.append(-transaction_costs[0])
    elif n == N:
        net_revenue = (np.array(quantities[N]) @ df_ptf.iloc[N].to_numpy()) - transaction_costs[N]  # Equation 2.18 : R^\Gamma
        transaction_account_qty.append(transaction_account_qty[-1] + net_revenue)
    else:
        transaction_account_qty.append(transaction_account_qty[n-1] - transaction_account[n-1] - transaction_costs[n-1])

    ###################### Portfolio value ###############################
    # Using equation 2.21
    if n != N:
        V_t_phi.append(np.array(quantities[n]) @ df_ptf.iloc[n].to_numpy() + transaction_account_qty[n]) # Discrete # +1*transaction_account_qty[-1]
    else:
        V_t_phi.append(transaction_account_qty[n])
    # Using equation 2.22
    V_t_psi.append(V_t_phi[-1] - transaction_account_qty[-1])  # Continuous

weights_df = pd.DataFrame(
    quantities,
    index=df_ptf.index,
    columns=[f"phi_{i}" for i in range(1, len(quantities[0]) + 1)],
)
plt.plot(transaction_account_qty) #\Phi_t^{d+1}

_, ax = plt.subplots(3, 1, figsize=(20, 20))
for col in df_ptf.columns:
    ax[0].plot(
        df_ptf[col],
        label=f"{col}",
    )

ax[0].set_xlabel("$t$ (in year)", fontsize=15)
ax[0].set_ylabel("Price of assets $S_t^i$", fontsize=15)
ax[0].set_title(f"Price processes", fontsize=20)
ax[0].grid()
ax[0].legend(fontsize=10)

for col in weights_df.columns:
    ax[1].plot(
        weights_df[col],
        label=r"Qty $\{}_t$".format(col.replace("_", "^")),
    )
ax_l = ax[1].twinx()
ax_l.fill_between(
    df_ptf.index,
    -np.array(transaction_account),
    color="red",
    step="pre",
    label=r"Transaction account $-D_t^\Phi$",
)

ax[1].set_xlabel("$t$ (in year)", fontsize=15)
ax[1].set_ylabel(r"Quantity of assets $\Phi_t^i$", fontsize=15)
ax_l.set_ylabel(r"Transaction account $-D_t^\Phi$", fontsize=15)
ax_l.set_ylim([-10,10])
ax[1].set_title(f"Strategies and rebalancing costs", fontsize=20)
ax[1].grid()
ax[1].legend(fontsize=10, loc="upper left")
ax_l.legend(fontsize=10, loc="upper right")
ax_2 = ax[2].twinx()
ax_2.plot(
    df_ptf.index,
    V_t_psi,
    label=r"$V_t^\Psi$ equation 2.22",
)
ax[2].plot(
    df_ptf.index,
    V_t_phi,
    linestyle="--",
    label=r"$V_t^\Phi$ equation 2.21",
)
ax_2.plot(
    df_ptf.index,
    np.array(V_t_phi) - np.array(V_t_psi),
    label=r"$V_t^\Phi-V_t^\Psi$",
)

ax[2].set_xlabel("$t$ (in year)", fontsize=15)
ax[2].set_ylabel(r"Portfolio value $V_t^\Phi$, $V_t^\Phi$", fontsize=15)
ax[2].set_title(f"Evolution of the portfolio value", fontsize=20)
ax[2].grid()
ax[2].legend(fontsize=15)
ax_2.grid()
ax_2.legend(fontsize=15)

plt.show()