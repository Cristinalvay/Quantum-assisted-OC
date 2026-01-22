from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg as la

from .common import four_partition, inverse_check_cond


@dataclass(frozen=True)
class MMethodResult:
    lamb0: np.ndarray
    M: np.ndarray
    N: np.ndarray
    condM: float
    S: np.ndarray


def solve_m_method(C: np.ndarray, H: np.ndarray, e0: np.ndarray, T: float) -> MMethodResult:
    """
    Continuous Hamiltonian evolution (M method):
      S = expm(C*T) -> blocks U00,U01,U10,U11
      M = U11 - H U01
      N = H U00 - U10
      lamb0 = M^{-1} N e0
    """
    e0 = np.asarray(e0, dtype=float).reshape(-1) 
    C = np.asarray(C, dtype=float)


    S, U00, U01, U10, U11 = four_partition(C, T)

    n = C.shape[0] // 2
    H = np.asarray(H, dtype=float)
    if H.shape != (n, n):
        raise ValueError(f"H must have shape {(n, n)}, got {H.shape}")
    if e0.shape != (n,):
        raise ValueError(f"e0 must have shape {(n,)}, got {e0.shape}")

    M = U11 - H @ U01
    N = H @ U00 - U10

    condM = inverse_check_cond(M)
    lamb0 = la.solve(M, N @ e0)

    return MMethodResult(lamb0=lamb0, M=M, N=N, condM=condM, S=S)
