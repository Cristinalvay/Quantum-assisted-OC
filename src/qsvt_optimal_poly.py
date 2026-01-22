"""qsvt_optimal_poly

Utilities for building the optimal Chebyshev polynomial used in QSVT-style
approximations of 1/x on the symmetric interval set

    S(a) = [-1, -a] U [a, 1],   with 0 < a ≤ 1.

This is commonly used for matrix inversion via singular value transformation:
if A has singular values in [σ_min, σ_max], scaling by σ_max maps them into
[a, 1] where a = σ_min/σ_max = 1/κ.

The implementation here follows the closed-form construction used in the
QSVT matrix inversion literature (see "Matrix inversion polynomials for the quantum singular value transformation" arXiv:2507.15537 
and "Two exact quantum signal processing results" arXiv:2505.10710)

"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Sequence, Union, List
import numpy as np

from pathlib import Path
import json



# -----------------------------
# Optimal 1/x polynomial on S(a)
# -----------------------------

def helper_Lfrac(n: int, x: float, a: float) -> float:
    """Compute mathcal{L}_n(x; a)"""
    alpha = (1 + a) / (2 * (1 - a))

    l1 = (x + (1 - a) / (1 + a)) / alpha
    l2 = (x**2 + (1 - a) / (1 + a) * x / 2 - 1/2) / alpha**2

    if n == 1:
        return l1

    for _ in range(3, n + 1):
        l1, l2 = l2, x * l2 / alpha - l1 / (4 * alpha**2)

    return l2

def helper_P(x: float, n: int, a: float) -> float:
    """Compute values of the polynomial P_(2n-1)(x; a)"""
    return (1 - (-1)**n * (1+a)**2 / (4*a) * helper_Lfrac(n, (2 * x**2 - (1 + a**2))/(1-a**2), a))/x

def poly(d: int, a: float) -> np.polynomial.Chebyshev:
    """Returns Chebyshev polynomial for optimal polynomial
    Args:
    d (int): odd degree
    a (float): 1/kappa for range [a,1]"""
    if d % 2 == 0:
        raise ValueError("d must be odd")
    coef = np.polynomial.chebyshev.chebinterpolate(
    helper_P, d, args=((d+1)//2, a))
    coef[0::2] = 0 # force even coefficients exactly zero
    return np.polynomial.Chebyshev(coef)

def error_for_degree(d: int, a: float) -> float:
    """Returns the poly error for degree d and a=1/kappa"""
    if d % 2 == 0:
        raise ValueError("d must be odd")
    n = (d+1)//2
    return (1-a)**n / (a * (1+a)**(n-1))

def mindegree_for_error(epsilon: float, a: float) -> int:
    """Returns the minimum degree d for a poly with error epsilon, a=1/kappa"""
    n = math.ceil((np.log(1/epsilon) + np.log(1/a) + np.log(1+a))
                  / np.log((1+a) / (1-a)))
    return 2*n-1

# -----------------------------










# -----------------------------
# Optimal 1/x polynomial for a given kappa and target epsilon
# -----------------------------
def _grids_for_a(a: float, n_full: int, n_branch: int):
    xs_full  = np.linspace(-1.0, 1.0, int(n_full))
    xs_left  = np.linspace(-1.0, -a, int(n_branch))
    xs_right = np.linspace(a,  1.0, int(n_branch))
    return xs_full, xs_left, xs_right

def _plot_panel_for_degree(
    ax,
    *,
    d: int,
    a: float,
    xs_full: np.ndarray,
    xs_left: np.ndarray,
    xs_right: np.ndarray,
    label_prefix: str = "p(x)",
    show_legend: bool = False,
    color: str 
):
    """
    Draw one panel: p_d(x) on [-1,1], 1/x on S(a), and ±ε(d,a) band.
    Returns (polynomial, eps_d).
    """
    p = poly(int(d), a)
    eps_d = error_for_degree(int(d), a)

    # p(x) everywhere
    ax.plot(xs_full, p(xs_full),color=color, label=f"{label_prefix}", linewidth=2.5)
    # 1/x branches separately 
    ax.plot(xs_left,  1.0 / xs_left,  color='k', label="1/x", linewidth=2)
    ax.plot(xs_right, 1.0 / xs_right, color='k', linewidth=2)
    # error band only on target region S(a)
    ax.fill_between(xs_left,  1.0 / xs_left  - eps_d, 1.0 / xs_left  + eps_d, color="lightgray")
    ax.fill_between(xs_right, 1.0 / xs_right - eps_d, 1.0 / xs_right + eps_d, color="lightgray")

    ax.axvline(-a, linestyle="--", color="gray")
    ax.axvline(a,  linestyle="--", color="gray")


    if show_legend:
        ax.legend(fontsize=14)

    return p, eps_d

def optimal_poly_for_kappa(kappa: float, target_epsilon: float, *, verbose: bool = True):
    """
    Given condition number κ and desired uniform error ε on [1/κ, 1],
    returns (d, polynomial, actual_epsilon, a) where a = 1/κ.

    Uses:
      d = mindegree_for_error(target_epsilon, a)
      polynomial = poly(d, a)
      actual_epsilon = error_for_degree(d, a)
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if not (0 < target_epsilon < 1):
        raise ValueError("target_epsilon should typically be in (0, 1)")

    a = 1.0 / float(kappa)
    d = mindegree_for_error(target_epsilon, a)
    polynomial = poly(d, a)

    # actual error bound (could be <= target_epsilon because d is integer)
    actual_epsilon = error_for_degree(d, a)

    if verbose:
        print(f"Degree {d} polynomial achieving error {actual_epsilon} in [{a}, 1]:")
        print(polynomial)

    return d, polynomial, actual_epsilon, a
 
# Example for given kappa and epsilon
# d, polynomial, actual_epsilon, a = optimal_poly_for_kappa(kappa=6, target_epsilon=0.47)

def plot_optimal_poly_for_kappa(
    kappa: float,
    target_epsilon: float,
    *,
    n_full: int = 4000,
    n_branch: int = 2000,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (9, 4),
    ax=None,
    show: bool = True,
    color: str 
):
    """
    Plots the optimal polynomial p(x) on [-1,1] and compares it with 1/x on S(a)=[-1,-a]∪[a,1],
    including the uniform error band ±epsilon.

    Returns: (fig, ax, d, polynomial, actual_epsilon, a)
    """
    import matplotlib.pyplot as plt  # local import

    d, polynomial, actual_epsilon, a = optimal_poly_for_kappa(
        kappa=kappa, target_epsilon=target_epsilon, verbose=False
    )

    # Grids
    # polynomial on the full interval
    # 1/x only on S(a) = [-1,-a] U [a,1]
    xs_full, xs_left, xs_right = _grids_for_a(a, n_full, n_branch)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    _plot_panel_for_degree(
        ax,
        d=d,
        a=a,
        xs_full=xs_full,
        xs_left=xs_left,
        xs_right=xs_right,
        label_prefix="Optimal p(x)",
        show_legend=True,
        color=color

    )     
    ax.grid(False)

    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("value")
    ax.set_title(f"Optimal polynomial, d={d}, κ={kappa}, ε={actual_epsilon:.2f}")
    if ylim is not None:
        ax.set_ylim(*ylim)

    if created_fig:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax, d, polynomial, actual_epsilon, a

# Example for given kappa and epsilon
# plot_optimal_poly_for_kappa(kappa=6, target_epsilon=0.47)

def plot_poly_degree_comparison(
    kappa: float,
    ds: Sequence[int],
    *,
    n_full: int = 4000,
    n_branch: int = 2000,
    ylim: Optional[Tuple[float, float]] = None,
    show: bool = True,
    figsize_per_ax: Tuple[float, float] = (4.0, 4.0),
    color: str
):
    """
    Compare several degrees in subplots. Returns (fig, axs, a).
    """
    import matplotlib.pyplot as plt

    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if len(ds) == 0:
        raise ValueError("ds must be a non-empty sequence of degrees")

    a = 1.0 / float(kappa)
    xs_full, xs_left, xs_right = _grids_for_a(a, n_full, n_branch)

    fig_w = figsize_per_ax[0] * len(ds)
    fig_h = figsize_per_ax[1]
    fig, axs = plt.subplots(1, len(ds), figsize=(fig_w, fig_h), constrained_layout=True)
    if len(ds) == 1:
        axs = [axs]

    for i, d in enumerate(ds):
        ax = axs[i]
        _, eps_d = _plot_panel_for_degree(
            ax,
            d=int(d),
            a=a,
            xs_full=xs_full,
            xs_left=xs_left,
            xs_right=xs_right,
            label_prefix="p(x)",
            show_legend=(i == 0),
            color=color

        )
        ax.set_title(f"d={int(d)}, ε≈{eps_d:.2g}", fontsize=16)
        ax.set_xlabel("x", fontsize=14)
        if i == 0:
           pass
        if ylim is not None:
            ax.set_ylim(*ylim)

    if show:
        plt.show()

    return fig, axs, a

# Example for given kappa and epsilons
# plot_poly_degree_comparison(kappa=6, ds=[5, 11, 15, 25, 35])
# -----------------------------



















# -----------------------------
# Optimal uniform error for a given kappa 
# -----------------------------


def optimal_uniform_error(d: Union[int, Sequence[int], np.ndarray], kappa: float):
    """
    Optimal uniform error ε(d) for the optimal 1/x polynomial on S(a), a=1/κ.

    - If d is an int -> returns float
    - If d is array-like -> returns np.ndarray

    Uses a numerically-stable log-domain computation and clips to avoid zeros.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    ds = np.asarray(d, dtype=int)

    # validate odd degrees
    if np.any(ds % 2 == 0):
        raise ValueError("All degrees d must be odd.")

    a = 1.0 / float(kappa)
    n = (ds + 1) // 2
    log_eps = n * np.log1p(-a) - np.log(a) - (n - 1) * np.log1p(a)
    eps = np.exp(log_eps)

    eps = np.maximum(eps, np.finfo(float).tiny)

    # return scalar if input was scalar
    if np.isscalar(d):
        return float(eps.item())
    return eps


def plot_error_vs_degree(
    kappa: float,
    d_max: int,
    *,
    logy: bool = True,
    ax=None,
    show: bool = True,
    figsize: Tuple[float, float] = (7.2, 4.4),
    color: str
):
    """Plot ε(d) vs odd degree d up to d_max for a given κ. Returns (fig, ax, ds, eps)."""
    import matplotlib.pyplot as plt

    if d_max < 1:
        raise ValueError("d_max must be >= 1")
    if d_max % 2 == 0:
        d_max += 1

    ds = np.arange(1, d_max + 1, 2)
    
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True
    else:
        fig = ax.figure

    eps = optimal_uniform_error(ds, kappa)

    ax.plot(ds, eps, color=color)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree d (odd)")
    ax.set_ylabel("Optimal uniform error ε(d)")
    ax.set_title(f"Optimal 1/x approximation error vs degree (κ={kappa:g})")
    ax.grid(True, which="both", linewidth=0.5)

    if created:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax, ds, eps

def plot_error_vs_degree_subplots(
    kappas: Sequence[float],
    d_max: Union[int, Sequence[int]],
    *,
    logy: bool = True,
    show: bool = True,
    figsize_per_ax: Tuple[float, float] = (4.0, 3.6),
    color:str
):
    """
    Plot ε(d) vs degree for each κ in its own subplot.

    Parameters
    ----------
    kappas:
        Sequence of κ values.
    d_max:
        Either:
          - int: same max degree for all κ
          - sequence of ints: per-κ max degree (must match len(kappas))
    Returns
    -------
    fig, axs, ds_list, eps_by_kappa
        ds_list is a list of degree arrays (one per κ).
        eps_by_kappa maps κ -> eps array (matching ds_list entry).
    """
    import matplotlib.pyplot as plt

    if len(kappas) == 0:
        raise ValueError("kappas must be a non-empty sequence")

    # normalize d_max into a list matching kappas
    if isinstance(d_max, (int, np.integer)):
        d_maxs = [int(d_max)] * len(kappas)
    else:
        d_maxs = list(d_max)
        if len(d_maxs) != len(kappas):
            raise ValueError("If d_max is a sequence, it must have the same length as kappas")

    # setup figure
    fig_w = figsize_per_ax[0] * len(kappas)
    fig_h = figsize_per_ax[1]
    fig, axs = plt.subplots(1, len(kappas), figsize=(fig_w, fig_h), constrained_layout=True)
    if len(kappas) == 1:
        axs = [axs]

    ds_list: List[np.ndarray] = []
    eps_by_kappa: Dict[float, np.ndarray] = {}

    for i, (kappa, dm) in enumerate(zip(kappas, d_maxs)):
        kappa = float(kappa)
        if kappa <= 0:
            raise ValueError("All kappa values must be > 0")
        if dm < 1:
            raise ValueError("Each d_max must be >= 1")

        # ensure odd max degree
        if dm % 2 == 0:
            dm += 1

        ds = np.arange(1, dm + 1, 2)
        eps = optimal_uniform_error(ds, kappa)  # uses your canonical function

        ds_list.append(ds)
        eps_by_kappa[kappa] = eps

        ax = axs[i]
        ax.plot(ds, eps, color=color)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"κ={kappa:g}")
        ax.set_xlabel("Polynomial degree d (odd)")
        if i == 0:
            ax.set_ylabel("Optimal uniform error ε(d)")
        ax.grid(True, which="both", linewidth=0.5)

    if show:
        plt.show()

    return fig, axs, ds_list, eps_by_kappa


def plot_error_vs_degree_comparison(
    kappas: Sequence[float],
    d_max: int,
    *,
    logy: bool = True,
    ax=None,
    show: bool = True,
    figsize: Tuple[float, float] = (7.2, 4.4),
    colors: Optional[Sequence[str]] = None,
):
    """Plot ε(d) vs degree for multiple κ on one plot. Returns (fig, ax, ds, eps_by_kappa)."""
    import matplotlib.pyplot as plt

    if len(kappas) == 0:
        raise ValueError("kappas must be a non-empty sequence")
    if d_max < 1:
        raise ValueError("d_max must be >= 1")
    if d_max % 2 == 0:
        d_max += 1

    ds = np.arange(1, d_max + 1, 2)

    
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True
    else:
        fig = ax.figure

    if colors is not None and len(colors) == 0:
        raise ValueError("colors must be None or a non-empty sequence")

    eps_by_kappa = {}
    for i, kappa in enumerate(kappas):
        eps = optimal_uniform_error(ds, float(kappa))
        eps_by_kappa[float(kappa)] = eps

        c = None
        if colors is not None:
            c = colors[i % len(colors)]   # cycle

        ax.plot(ds, eps, label=f"κ={kappa:g}", color=c)


    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree d (odd)")
    ax.set_ylabel("Optimal uniform error ε(d)")
    ax.set_title("Optimal 1/x approximation error vs degree")
    ax.grid(True, which="both", linewidth=0.5)
    ax.legend()

    if created:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax, ds, eps_by_kappa
# Example for given kappas
# plot_error_vs_degree_comparison([6, 50, 500], d_max=201, logy=True)


































def _repo_root_from_this_file() -> Path:
    # .../repo/src/qsvt/qsvt_optimal_poly.py -> parents[2] = .../repo
    return Path(__file__).resolve().parents[2]


def save_qsvt_figures_and_results(
    kappa: float,
    target_epsilon: float,
    *,
    ds=(5, 11, 15, 25, 35),
    fig_dir: str | Path = "figures",
    results_dir: str | Path = "results",
    prefix: str = "qsvt",
    dpi: int = 200,
    save_formats=("png",),
    also_save_pdf: bool = False,
):
    """
    Compute optimal degree/polynomial for (kappa, target_epsilon), generate plots, and save:
      - figures/<prefix>_optimal_kappa{...}_eps{...}_d{...}.png
      - figures/<prefix>_compare_kappa{...}_ds-...png
      - results/<prefix>_optimal_kappa{...}_eps{...}_d{...}.json (+ .txt)

    Returns a dict with the saved paths + computed objects.
    """
    import matplotlib.pyplot as plt  # local import

    root = _repo_root_from_this_file()
    fig_path = (root / fig_dir)
    res_path = (root / results_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)

    a = 1.0 / float(kappa)
    d = mindegree_for_error(float(target_epsilon), a)
    p = poly(d, a)
    eps = error_for_degree(d, a)

    # ---------- single figure ----------
    fig1, ax1, *_ = plot_optimal_poly_for_kappa(
        kappa=kappa,
        target_epsilon=target_epsilon,
        show=False,
    )

    base1 = f"{prefix}_optimal_kappa{kappa:g}_eps{target_epsilon:g}_d{d}"
    saved_fig1 = []
    fmts = list(save_formats) + (["pdf"] if also_save_pdf else [])
    for fmt in fmts:
        out = fig_path / f"{base1}.{fmt}"
        fig1.savefig(out, dpi=dpi, bbox_inches="tight")
        saved_fig1.append(str(out))
    plt.close(fig1)

    # ---------- comparison figure ----------
    fig2, axs2, _ = plot_poly_degree_comparison(
        kappa=kappa,
        ds=list(ds),
        show=False,
    )

    ds_tag = "-".join(str(int(x)) for x in ds)
    base2 = f"{prefix}_compare_kappa{kappa:g}_ds-{ds_tag}"
    saved_fig2 = []
    for fmt in fmts:
        out = fig_path / f"{base2}.{fmt}"
        fig2.savefig(out, dpi=dpi, bbox_inches="tight")
        saved_fig2.append(str(out))
    plt.close(fig2)

    # ---------- results files ----------
    # guarda coeficientes Chebyshev + resumen
    payload = {
        "kappa": float(kappa),
        "target_epsilon": float(target_epsilon),
        "a": float(a),
        "degree": int(d),
        "eps_bound": float(eps),
        "chebyshev_coef": [float(c) for c in np.asarray(p.coef)],
        "polynomial_repr": repr(p),
        "comparison_degrees": [int(x) for x in ds],
        "figures": {"single": saved_fig1, "comparison": saved_fig2},
    }

    res_base = res_path / f"{base1}"
    json_file = str(res_base.with_suffix(".json"))
    txt_file = str(res_base.with_suffix(".txt"))

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"kappa = {kappa}\n")
        f.write(f"target_epsilon = {target_epsilon}\n")
        f.write(f"a = {a}\n")
        f.write(f"degree d = {d}\n")
        f.write(f"eps_bound = {eps}\n\n")
        f.write("polynomial:\n")
        f.write(str(p))
        f.write("\n")

    return {
        "degree": d,
        "polynomial": p,
        "eps_bound": eps,
        "a": a,
        "saved_fig_single": saved_fig1,
        "saved_fig_comparison": saved_fig2,
        "saved_json": json_file,
        "saved_txt": txt_file,
    }






















__all__ = [
    "helper_Lfrac",
    "helper_P",
    "poly",
    "error_for_degree",
    "mindegree_for_error",

    "optimal_poly_for_kappa",
    "plot_optimal_poly_for_kappa",
    "plot_poly_degree_comparison",
    
    "optimal_uniform_error",
    "plot_error_vs_degree",
    "plot_error_vs_degree_subplots",
    "plot_error_vs_degree_comparison",
]
