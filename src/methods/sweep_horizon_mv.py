# src/experiments/sweep_horizon_mv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from methods.method_m import solve_m_method
from methods.method_v import solve_v_method
from lqr.riccati import lqr_gain_dre


# -------------------------
# Helpers
# -------------------------

def rel_norm(test: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    """||test-ref|| / max(||ref||, eps)"""
    test = np.asarray(test, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(ref)), eps)
    return float(np.linalg.norm(test - ref) / denom)


def _safe_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.linalg.norm(x))


def _safe_maxabs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x)))


# -------------------------
# Output container
# -------------------------

@dataclass
class SweepMVOutput:
    T_vals: np.ndarray
    m_grids: List[int]
    n_state: int

    # condition numbers
    condM: np.ndarray                 # (nT,)
    condV: Dict[int, np.ndarray]      # m -> (nT,)

    # lambda0 vectors
    lamb0_DRE: np.ndarray             # (nT, n_state)
    lamb0_M: np.ndarray               # (nT, n_state)
    lamb0_V: Dict[int, np.ndarray]    # m -> (nT, n_state)

    # comparisons (relative errors)
    rel_M_vs_DRE: np.ndarray          # (nT,)
    rel_V_vs_DRE: Dict[int, np.ndarray]  # m -> (nT,)

    # linear system residuals
    resM_2: np.ndarray                # (nT,)
    resV_2: Dict[int, np.ndarray]     # m -> (nT,)
    


# -------------------------
# Main sweep
# -------------------------

def sweep_MV_over_T(
    make_model_for_T: Callable[[float], object],
    T_vals: Sequence[float],
    *,
    m_grids: Sequence[int] = (2, 4, 10, 50, 100),
    dre_method: str = "Radau",
) -> SweepMVOutput:
    """
    Sweep over horizons T to compare M and V methods, and compare λ0 against DRE:
      λ0_DRE = P(0) e0  (finite-horizon Riccati)

    The model returned by make_model_for_T(T) must provide:
      - A, B, Q, R, H
      - C, e0, T
    (integrate_leader=False is recommended for speed)
    """
    T_vals = np.asarray(T_vals, dtype=float).reshape(-1)
    nT = int(T_vals.size)
    if nT == 0:
        raise ValueError("T_vals must be non-empty")

    m_grids_list = [int(m) for m in m_grids]
    if any(m <= 1 for m in m_grids_list):
        raise ValueError("m_grids must all be >= 2")

    # build first model to know n_state
    model0 = make_model_for_T(float(T_vals[0]))
    n_state = int(np.asarray(model0.e0).size)

    # allocate
    condM = np.zeros(nT, dtype=float)
    condV: Dict[int, np.ndarray] = {m: np.zeros(nT, dtype=float) for m in m_grids_list}

    lamb0_DRE = np.zeros((nT, n_state), dtype=float)
    lamb0_M = np.zeros((nT, n_state), dtype=float)
    lamb0_V: Dict[int, np.ndarray] = {m: np.zeros((nT, n_state), dtype=float) for m in m_grids_list}

    rel_M_vs_DRE = np.zeros(nT, dtype=float)
    rel_V_vs_DRE: Dict[int, np.ndarray] = {m: np.zeros(nT, dtype=float) for m in m_grids_list}

    resM_2 = np.zeros(nT, dtype=float)
    resV_2: Dict[int, np.ndarray] = {m: np.zeros(nT, dtype=float) for m in m_grids_list}

    for k, T in enumerate(T_vals):
        model = make_model_for_T(float(T))
        e0 = np.asarray(model.e0, dtype=float).reshape(-1)
        if e0.size != n_state:
            raise ValueError(f"Model at T={T} has e0 size {e0.size} != n_state={n_state}")

        # ---- DRE λ0 = P(0)e0 ----
        DRE = lqr_gain_dre(
            model.A, model.B, model.Q, model.R,
            T=float(model.T),
            H_T=model.H,
            t_eval=np.array([0.0, float(model.T)]),  # only [0, T]
            method=dre_method,
        )
        lamb0_DRE[k, :] = (DRE.P[0] @ e0).reshape(-1)


        # ---- M method ----
        m_res = solve_m_method(model.C, model.H, e0, float(model.T))
        condM[k] = float(m_res.condM)
        lamb0_M[k, :] = np.asarray(m_res.lamb0, dtype=float).reshape(-1)

        # residual M: ||M λ0 - N e0||
        if hasattr(m_res, "M") and hasattr(m_res, "N"):
            rhsM = m_res.N @ e0
            rM = (m_res.M @ m_res.lamb0) - rhsM
            resM_2[k] = _safe_norm(rM)
        else:
            resM_2[k] = np.nan
        # compare M vs DRE
        rel_M_vs_DRE[k] = rel_norm(lamb0_M[k, :], lamb0_DRE[k, :])


        # ---- V method(s) ----
        for m in m_grids_list:
            v_res = solve_v_method(model.C, model.H, e0, float(model.T), int(m))
            condV[m][k] = float(v_res.condV)
            lamb0_V[m][k, :] = np.asarray(v_res.lamb0, dtype=float).reshape(-1)

            # residual V: ||V z - rhs||
            rV = v_res.system.V @ v_res.z - v_res.system.rhs
            resV_2[m][k] = _safe_norm(rV)
            # compare V vs DRE
            rel_V_vs_DRE[m][k] = rel_norm(lamb0_V[m][k, :], lamb0_DRE[k, :])
    
        print(f"Done T={T:g}")

    return SweepMVOutput(
        T_vals=T_vals,
        m_grids=m_grids_list,
        n_state=n_state,

        condM=condM,
        condV=condV,

        lamb0_DRE=lamb0_DRE,
        lamb0_M=lamb0_M,
        lamb0_V=lamb0_V,

        rel_M_vs_DRE=rel_M_vs_DRE,
        rel_V_vs_DRE=rel_V_vs_DRE,
    
        resM_2=resM_2,
        resV_2=resV_2,
    )














# -------------------------
# Plotting (no lines + B/N-friendly markers)
# -------------------------

def _resolve_colors(
    colors: Union[str, Sequence[str]],
    n: int,
    *,
    cmap_range: Tuple[float, float] = (0.30, 0.90),  
    as_hex: bool = True,
) -> List[str]:
    """
    If colors is a colormap name (str), sample n colors from it.
    If colors is a list/tuple of colors, cycle as needed.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors)
        lo, hi = cmap_range

        if n == 1:
            xs = np.array([(lo + hi) / 2.0])
        else:
            xs = np.linspace(lo, hi, n)

        cols = [cmap(float(x)) for x in xs]
        if as_hex:
            cols = [mcolors.to_hex(c) for c in cols]
        return cols

    cols = list(colors)
    if len(cols) == 0:
        raise ValueError("colors sequence cannot be empty")
    return [cols[i % len(cols)] for i in range(n)]






def plot_kappa_vs_T(
    out: SweepMVOutput,
    *,
    title: str = r"Condition number $\kappa$ vs Time horizon $T$",
    colors: Union[str, Sequence[str]] = "Purples",
    markers: Optional[Sequence[str]] = None,
    hollow: bool = False,  
    edge_black: bool = False, 
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots κ(M) and κ(V) vs T using markers only (no lines)."""
    fig, ax = plt.subplots(figsize=(6.5,4.5))

    ms = sorted(out.condV.keys())
    n_curves = 1 + len(ms)

    if markers is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x"]

    cols = _resolve_colors(colors, n_curves)

    # M
    mfc0 = "none" if hollow else cols[0]
    mec0 = "k" if edge_black else cols[0]
    mew0 = 0.7

    ax.semilogy(
        out.T_vals, out.condM,
        linestyle="None",
        marker=markers[0 % len(markers)],
        color=cols[0],
        markerfacecolor=mfc0,
        markeredgecolor=mec0,
        markeredgewidth=mew0,
        markersize=8,
        label=r"$\kappa(M)$",
    )

    # V curves
    for j, m in enumerate(ms, start=1):
        mfc = "none" if hollow else cols[j]
        mec = "k" if edge_black else cols[j]
        mew = 0.7

        ax.semilogy(
            out.T_vals, out.condV[m],
            linestyle="None",
            marker=markers[j % len(markers)],
            color=cols[j],
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markeredgewidth=mew,
            markersize=8,
            label=fr"$\kappa(V)$  m={m}",
        )

    ax.set_xlabel("Time Horizon (s)", fontsize=16)
    ax.set_ylabel(r"Condition number ($\kappa$)", fontsize=16)
    #ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    #ax.set_title(title)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()
    return fig, ax








def plot_rel_lambda0_vs_T_DRE(
    out: SweepMVOutput,
    *,
    title: str = r"Relative error method $M$/$V$ vs DRE for $\lambda_0$",
    colors: Union[str, Sequence[str]] = "Purples",
    markers: Optional[Sequence[str]] = None,
    hollow: bool = False,
    edge_black: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots ||λ0(method) - λ0_DRE|| / ||λ0_DRE|| for M and each V(m)."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    ms = sorted(out.rel_V_vs_DRE.keys())
    n_curves = 1 + len(ms)

    if markers is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x"]

    cols = _resolve_colors(colors, n_curves)

    # M vs DRE
    ax.semilogy(
        out.T_vals, out.rel_M_vs_DRE,
        linestyle="None",
        marker=markers[0 % len(markers)],
        color=cols[0],
        markerfacecolor=("none" if hollow else cols[0]),
        markeredgecolor=("k" if edge_black else cols[0]),
        markeredgewidth=0.7,
        markersize=8,
        label="M vs DRE",
    )

    # V(m) vs DRE
    for j, m in enumerate(ms, start=1):
        ax.semilogy(
            out.T_vals, out.rel_V_vs_DRE[m],
            linestyle="None",
            marker=markers[j % len(markers)],
            color=cols[j],
            markerfacecolor=("none" if hollow else cols[j]),
            markeredgecolor=("k" if edge_black else cols[j]),
            markeredgewidth=0.7,
            markersize=8,
            label=f"V(m={m}) vs DRE",
        )

    ax.set_xlabel("Time horizon T(s)")
    ax.set_ylabel(r"$\|\lambda_0-\lambda_0^{DRE}\|/\|\lambda_0^{DRE}\|$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax







from typing import Optional, Sequence, Tuple, Union, List
import matplotlib.pyplot as plt

def plot_residuals2_vs_T(
    out: SweepMVOutput,
    *,
    title: str = r"Linear residuals vs Time horizon $T$",
    colors: Union[str, Sequence[str]] = "Purples",
    markers: Optional[Sequence[str]] = None,
    hollow: bool = False,
    edge_black: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots linear system residuals (2-norm) vs T:
      - M: rM = M λ0 - N e0  -> out.resM_2
      - V: rV = V z - rhs   -> out.resV_2[m]

    Markers only (no lines).
    Returns (fig, ax).
    """
    ms = sorted(out.resV_2.keys())
    n_curves = 1 + len(ms)

    if markers is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x"]

    cols = _resolve_colors(colors, n_curves)

    def _mfc(j):  # marker face color
        return "none" if hollow else cols[j]

    def _mec(j):  # marker edge color
        return "k" if edge_black else cols[j]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # M residual 2-norm
    ax.semilogy(
        out.T_vals, out.resM_2,
        linestyle="None",
        marker=markers[0 % len(markers)],
        color=cols[0],
        markerfacecolor=_mfc(0),
        markeredgecolor=_mec(0),
        markeredgewidth=0.7,
        markersize=8,
        label=r"$\|M\lambda_0 - N e_0\|_2$",
    )

    # V residual 2-norm for each m
    for j, m in enumerate(ms, start=1):
        ax.semilogy(
            out.T_vals, out.resV_2[m],
            linestyle="None",
            marker=markers[j % len(markers)],
            color=cols[j],
            markerfacecolor=_mfc(j),
            markeredgecolor=_mec(j),
            markeredgewidth=0.7,
            markersize=8,
            label=fr"$\|V z - rhs\|_2$ (m={m})",
        )

    ax.set_xlabel("Time horizon T(s)")
    ax.set_ylabel("Residual (2-norm)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig, ax



__all__ = [
    "sweep_MV_over_T",
    "plot_kappa_vs_T",
    "plot_rel_lambda0_vs_T_DRE",
    "plot_residuals2_vs_T",
]