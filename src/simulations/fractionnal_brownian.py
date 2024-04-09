# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:33:44 2024

@author: jvilp
"""

import numpy as np
from math import pi


def fBM_simul(
    T: float, N: int, H: float
):  # Spectral simulation of fBM (Appendix B) / Param ::: T: Time hoizon; N: nb of time step; H: Hurst exponent range(0,1)
    if (
        N % 2 != 0
    ):  # if N is not pair we adjust it to ensure 0.5*N is include in integer and that delta stay the same
        T = (T / N) * (N + 1)
        N = N + 1
        shift = 1
    else:
        shift = 0

    delta = T / N
    k_range = list(range(0, N))
    W_increment = [compute_Wk(k, N, H) for k in k_range]

    W = delta**H * np.cumsum(W_increment)
    W = W[: len(W) - shift]

    return W


def compute_Wk(
    k: int, N: int, H: float
):  # Compute fBM increment (B.2) / Param ::: k: iteration over N; N: nb of time step; H: Hurst exponent range(0,1)
    J = list(range(int(-0.5 * N), int(0.5 * N)))  # from -N/2 to N/2 -1
    phi = np.random.uniform(low=0, high=2 * pi, size=N)

    W_j = [
        compute_Sf(j / N, N, H) ** 0.5
        * (
            np.cos(2 * pi * j * k / N) * np.cos(phi[int(j + 0.5 * N)])
            - np.sin(2 * pi * j * k / N) * np.sin(phi[int(j + 0.5 * N)])
        )
        for j in J
    ]
    Wk = np.sqrt(2 / N) * sum(W_j)

    return Wk


def compute_Sf(
    f: float, N: int, H: float
):  # Power sprectral density aproximation (B.3) / Param ::: f: frequency; N: nb of time step; H: Hurst exponent range(0,1)
    M = list(range(int(-0.5 * N), int(0.5 * N)))  # from -N/2 to N/2 -1

    S_m = [
        (abs(m + 1) ** 2 * H + abs(m - 1) ** 2 * H - 2 * abs(m) ** 2 * H)
        * np.cos(2 * pi * m * f)
        for m in M
    ]
    Sf = 0.5 * sum(S_m)

    return Sf
