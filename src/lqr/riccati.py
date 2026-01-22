# src/lqr/riccati.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import linalg as la
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class AREOutput:
    P: np.ndarray          # (n, n)
    K: np.ndarray          # (m, n)
    Rinv_BT: np.ndarray    # (m, n)


@dataclass(frozen=True)
class DREOutput:
    t: np.ndarray          # (N,)
    P: np.ndarray          # (N, n, n)
    K: np.ndarray          # (N, m, n)
    Rinv_BT: np.ndarray    # (m, n)


# -------------------------------------------------
# 1) Gains
# -------------------------------------------------

def lqr_gain_are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> AREOutput:
    """
    Continuous-time infinite-horizon LQR (ARE):
        A^T P + P A - P B R^{-1} B^T P + Q = 0
        K = R^{-1} B^T P
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P = la.solve_continuous_are(A, B, Q, R)
    Rinv_BT = la.solve(R, B.T)         # (m, n)
    K = Rinv_BT @ P                    # (m, n)
    return AREOutput(P=P, K=K, Rinv_BT=Rinv_BT)


def lqr_gain_dre(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    T: float,
    H_T: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
    num_points: int = 1001,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-8,
    enforce_symmetry: bool = True,
) -> DREOutput:
    """
    Continuous-time finite-horizon LQR via DRE:
        -dP/dt = A^T P + P A - P B R^{-1} B^T P + Q
        P(T) = H_T
        K(t) = R^{-1} B^T P(t)

    Returns P(t), K(t) on a forward time grid t in [0,T].
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    H_T = np.asarray(H_T, dtype=float)

    n = A.shape[0]
    m = B.shape[1]

    if H_T.shape != (n, n):
        raise ValueError(f"H_T must be {(n,n)}, got {H_T.shape}")

    Rinv_BT = la.solve(R, B.T)  # (m, n)

    def riccati_ode(_t: float, P_flat: np.ndarray) -> np.ndarray:
        P = P_flat.reshape(n, n)
        term1 = A.T @ P + P @ A
        term2 = P @ B @ (Rinv_BT @ P)   # P B R^{-1} B^T P
        dPdt = -(term1 - term2 + Q)
        return dPdt.reshape(n * n)

    # build a backward evaluation grid (T -> 0)
    if t_eval is None:
        t_desc = np.linspace(T, 0.0, int(num_points))
    else:
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        if t_eval[0] < -1e-12 or t_eval[-1] < T - 1e-9:
            raise ValueError("t_eval must cover [0,T] and be ascending.")
        t_desc = t_eval[::-1].copy()  # descending

    sol = solve_ivp(
        riccati_ode,
        t_span=(float(T), 0.0),
        y0=H_T.reshape(n * n),
        t_eval=t_desc,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"DRE integration failed: {sol.message}")

    # sol.t is descending (T -> 0). Turn into forward (0 -> T).
    P_desc = sol.y.T.reshape(-1, n, n)     # aligned with sol.t (descending)
    t_fwd = sol.t[::-1]
    P_fwd = P_desc[::-1]

    if enforce_symmetry:
        P_fwd = 0.5 * (P_fwd + np.transpose(P_fwd, (0, 2, 1)))

    # K(t) = R^{-1} B^T P(t)  -> (N, m, n)
    K_fwd = np.matmul(Rinv_BT[None, :, :], P_fwd)

    return DREOutput(t=t_fwd, P=P_fwd, K=K_fwd, Rinv_BT=Rinv_BT)


# -------------------------------------------------
# 2) Closed-loop simulations in error coordinates
# -------------------------------------------------

def simulate_constant_gain(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    *,
    e0: np.ndarray,
    t_eval: np.ndarray,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    de/dt = (A - B K) e,  e(0)=e0
    Returns (t, E) where E is (N, n).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.asarray(K, dtype=float)
    e0 = np.asarray(e0, dtype=float).reshape(-1)
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

    A_cl = A - B @ K

    def ode(_t, e):
        return A_cl @ e

    sol = solve_ivp(
        ode,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=e0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Constant-gain simulation failed: {sol.message}")

    return sol.t, sol.y.T


def simulate_timevarying_gain_piecewise(
    A: np.ndarray,
    B: np.ndarray,
    t_grid: np.ndarray,
    K_grid: np.ndarray,
    *,
    e0: np.ndarray,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    de/dt = (A - B K(t)) e using piecewise-constant K on [t_i, t_{i+1}].

    Inputs
    ------
    t_grid: (N,) ascending
    K_grid: (N, m, n) or (N, n) for single-input squeezed
    e0: (n,)

    Returns (t_grid, E) with E shape (N, n).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    e = np.asarray(e0, dtype=float).reshape(-1)

    K_grid = np.asarray(K_grid, dtype=float)
    if K_grid.ndim == 2:
        # allow (N, n) for single-input; make it (N, 1, n)
        K_grid = K_grid[:, None, :]

    if K_grid.shape[0] != t_grid.size:
        raise ValueError("K_grid and t_grid must have same length")

    E_all = [e.copy()]

    for i in range(len(t_grid) - 1):
        t0, t1 = float(t_grid[i]), float(t_grid[i + 1])
        K_i = K_grid[i]              # (m, n)
        A_cl = A - B @ K_i

        def ode(_t, ee):
            return A_cl @ ee

        sol = solve_ivp(
            ode,
            t_span=(t0, t1),
            y0=e,
            t_eval=[t0, t1],
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Segment integration failed: {sol.message}")

        e = sol.y[:, -1]
        E_all.append(e.copy())

    E = np.vstack(E_all)  # (N, n)
    return t_grid, E


__all__ = [
    "AREOutput",
    "DREOutput",
    "lqr_gain_are",
    "lqr_gain_dre",
    "simulate_constant_gain",
    "simulate_timevarying_gain_piecewise",
]