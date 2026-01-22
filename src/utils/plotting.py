from __future__ import annotations

import re
from typing import Callable, Literal, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from utils.trajectory import TrajectoryResult

Kind = Literal["cart", "vehicle"]
Panel = Literal["X", "LAM", "E", "U"]
VehicleUMode = Literal["tilde", "absolute"]  # ũ or u=u0+ũ


def _infer_kind_from_model_name(model_name: str) -> Kind:
    if model_name == "cart_pendulum":
        return "cart"
    if model_name == "vehicle_platoon":
        return "vehicle"
    raise ValueError(
        f"Unknown model_name={model_name!r}. Expected 'cart_pendulum' or 'vehicle_platoon'."
    )


def _parse_cart_plot_only(
    plot_only: Optional[Sequence[str]],
    *,
    n: int,
    labels: Sequence[str],
) -> Optional[list[int]]:
    """
    plot_only examples:
      ("x1","x3") -> [0,2]
      ("s", r"\\theta") -> indices by matching labels
    """
    if plot_only is None:
        return None

    idxs: list[int] = []
    for tok in plot_only:
        tok = str(tok).strip()

        m = re.fullmatch(r"x(\d+)", tok)
        if m:
            k = int(m.group(1)) - 1
        else:
            if tok not in labels:
                raise ValueError(
                    f"Unknown plot_only token {tok!r}. "
                    f"Use 'x1'..'x{n}' or one of labels={tuple(labels)}"
                )
            k = labels.index(tok)

        if not (0 <= k < n):
            raise ValueError(f"plot_only index out of range: {tok!r} -> {k}, n={n}")

        if k not in idxs:
            idxs.append(k)

    return idxs


def plot_result(
    tr: TrajectoryResult,
    *,
    panels: Sequence[Panel] = ("X", "LAM", "E", "U"),
    title: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    # -----------------
    # cart options
    # -----------------
    cart_labels: Sequence[str] = ("s", "v", r"\theta", r"\omega"),
    cart_plot_only: Optional[Sequence[str]] = None,
    cart_colors: Sequence[str] | str = "Purples", # cart_colors=("#B175D8", "#7D77D8", "#6F0E8D", "#350E7E")
    u_color: str = "#282829",
    # -----------------
    # vehicle options
    # -----------------
    all_followers: bool = True,
    follower_idx: int = 0,
    vehicle_component: int = 0,  # 0=s,1=v,2=a when overlay=False
    vehicle_cmap: str = "Purples",
    # overlay mode (like your old plot): same panel draws s/v/a by linestyle
    vehicle_overlay_components: bool = False,
    vehicle_components: Sequence[int] = (0, 1, 2),
    vehicle_linestyles: Sequence[str] = ("-", "--", ":"),
    vehicle_comp_labels: Sequence[str] = (r"Position ($m$)", r"Velocity ($m/s$)", r"Acceleration ($m/s^2$)"),
    vehicle_show_legend: bool = True,
    # U options (vehicle): include u0 and/or absolute inputs
    include_u0: bool = False,
    u0_fun: Optional[Callable[[float], float]] = None,
    u0_color: str = "#444444",
    vehicle_u_mode: VehicleUMode = "tilde",
) -> Tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot TrajectoryResult for either cart or vehicle models.

    Vehicle:
      - overlay=False: plot ONE component (s or v or a) for all followers in a panel.
      - overlay=True : plot s,v,a together in the same panel for all followers:
          follower = color, component = linestyle.
    """
    kind = _infer_kind_from_model_name(tr.model_name)

    t = np.asarray(tr.t, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("t must have length >= 2")

    X = np.asarray(tr.X, dtype=float)
    LAM = np.asarray(tr.LAM, dtype=float)
    E = np.asarray(tr.E, dtype=float)
    U = np.asarray(tr.U, dtype=float)

    panels = tuple(panels)
    if len(panels) == 0:
        raise ValueError("panels must contain at least one of ('X','LAM','E','U').")

    # figure size
    if figsize is None:
        figsize = (8.0, 3.1*len(panels))

    fig, axs = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    axs_list = [axs] if len(panels) == 1 else list(axs)

    # -------------------------
    # helpers: cart
    # -------------------------
    def _plot_cart_matrix(ax, A: np.ndarray, prefix: str, *, use_prefix_in_label: bool = False):
        n = A.shape[1]
        if len(cart_labels) != n:
            raise ValueError(f"cart_labels must have length {n}. Got {len(cart_labels)}")

        # --- build colors ---
        if isinstance(cart_colors, str):
            cmap = plt.get_cmap(cart_colors)
            colors = cmap(np.linspace(0.35, 0.90, n))  # N colors from colormap
        else:
            if len(cart_colors) < n:
                raise ValueError(f"cart_colors must have at least {n} colors. Got {len(cart_colors)}")
            colors = list(cart_colors)

        sel = _parse_cart_plot_only(cart_plot_only, n=n, labels=cart_labels)
        idxs = sel if sel is not None else list(range(n))

        for i in idxs:
            lab = cart_labels[i]
            color = colors[i]
            legend = fr"${prefix}_{{{lab}}}$" if use_prefix_in_label else fr"${lab}$"
            ax.plot(t, A[:, i], label=legend, color=color)

        ax.legend(ncol=min(4, len(idxs)), fontsize=8)


    # -------------------------
    # helpers: vehicle
    # -------------------------
    def _vehicle_reshape(A: np.ndarray) -> tuple[int, np.ndarray]:
        if A.ndim != 2:
            raise ValueError(f"vehicle expects 2D arrays, got {A.shape}")
        if A.shape[1] % 3 != 0:
            raise ValueError(f"vehicle expects dim multiple of 3. Got {A.shape[1]}")
        nF = A.shape[1] // 3
        A3 = A.reshape(t.size, nF, 3)
        return nF, A3

    def _get_vehicle_cols(nF: int):
        cmap = plt.get_cmap(vehicle_cmap)
        xs = np.linspace(0.3, 1.0, max(nF, 2))
        return cmap(xs)[:nF]

    def _plot_vehicle_panel(ax, A: np.ndarray, name: str, *, overlay: bool):
        nF, A3 = _vehicle_reshape(A)
        cols = _get_vehicle_cols(nF)

        if all_followers and overlay:
            # all followers, all components on same axis
            for i in range(nF):
                for c in vehicle_components:
                    if c not in (0, 1, 2):
                        raise ValueError("vehicle_components must contain only 0,1,2")
                    ax.plot(
                        t, A3[:, i, c],
                        color=cols[i],
                        ls=vehicle_linestyles[c],
                        lw=1.5,
                    )

            #ax.set_title(name, fontsize=14)

            if vehicle_show_legend:
                #handles_f = [Line2D([0], [0], color=cols[i], lw=2, label=f"i={i+1}") for i in range(nF)]
                handles_c = [
                    Line2D([0], [0], color="k", lw=2, ls=vehicle_linestyles[c], label=vehicle_comp_labels[c])
                    for c in vehicle_components
                ]
                #leg1 = ax.legend(handles=handles_f, title="Follower", ncol=min(nF, 6), loc="upper right", fontsize=8)
                #ax.add_artist(leg1)
                ax.legend(
                    handles=handles_c,
                    title="Component",
                    loc="lower right",
                    fontsize=14,          # tamaño de los items: s/v/a
                    title_fontsize=16,    # tamaño del título: "Component"
                )


            return

        # fallback behavior
        if all_followers:
            c = int(vehicle_component)
            if c not in (0, 1, 2):
                raise ValueError("vehicle_component must be 0(s), 1(v), or 2(a)")
            for i in range(nF):
                ax.plot(t, A3[:, i, c], color=cols[i], lw=1.5)
            comp = ["s", "v", "a"][c]
            ax.set_title(f"{name} (all followers), component={comp}")
        else:
            j = int(follower_idx)
            if not (0 <= j < nF):
                raise ValueError(f"follower_idx must be in [0, {nF-1}]")
            ax.plot(t, A3[:, j, 0], label="Position (m)", lw=1.5)
            ax.plot(t, A3[:, j, 1], label="Velocity (m/s)", lw=1.5)
            ax.plot(t, A3[:, j, 2], label="Acceleration (m/s²)", lw=1.5)
            ax.legend(ncol=3, fontsize=12)
            ax.set_title(f"{name} (follower {j+1})")

    def _plot_U(ax):
        # -------- cart --------
        if kind == "cart":
            U1 = U
            if U1.ndim == 2 and U1.shape[1] == 1:
                U1 = U1[:, 0]
            if U1.ndim != 1:
                raise ValueError("cart expects U as (N,) or (N,1)")
            ax.plot(t, U1, color=u_color)
            ax.set_title("u(t)")
            return

        # -------- vehicle --------
        # U is expected (N,n) for ũ_i, or (N,) for single input
        u0_vec = None
        if include_u0 or vehicle_u_mode == "absolute":
            if u0_fun is None:
                raise ValueError("Pass u0_fun=... when include_u0=True or vehicle_u_mode='absolute'")
            u0_vec = np.fromiter((float(u0_fun(tt)) for tt in t), dtype=float)

        if U.ndim == 1:
            # single input
            Uplot = U
            if vehicle_u_mode == "absolute":
                Uplot = U + u0_vec
                ax.set_title("u(t) = u0(t) + ũ(t)")
            else:
                ax.set_title("ũ(t)")

            ax.plot(t, Uplot, color=u_color, lw=1.5)
            if include_u0:
                ax.plot(t, u0_vec, color=u0_color, lw=2.0, label="u0")
                #ax.legend(fontsize=12)
            return

        if U.ndim != 2:
            raise ValueError(f"vehicle expects U as 1D or 2D, got {U.shape}")

        m_inputs = U.shape[1]
        cols = _get_vehicle_cols(m_inputs)

        if vehicle_u_mode == "absolute":
            Uplot = U + u0_vec[:, None]
            #ax.set_title(r"Contol inputs: $u_i$ = $u_0$ + $\tilde{u}_i$", fontsize=14)
        else:
            Uplot = U
            #ax.set_title(r"Differential inputs $\tilde{u}_i(t)$", fontsize=14)

        if include_u0:
            ax.plot(t, u0_vec, color=u0_color, lw=2, label=r"$u_0$")

        for j in range(m_inputs):
            lab = rf"$u_{{{j+1}}}$" if vehicle_u_mode == "absolute" else rf"$\tilde{{u}}_{{{j+1}}}$"
            ax.plot(t, Uplot[:, j], color=cols[j], lw=1.5, label=lab)

        ax.legend(ncol=min(4, m_inputs + (1 if include_u0 else 0)), fontsize=14)

    # -------------------------
    # plot panels
    # -------------------------
    for ax, p in zip(axs_list, panels):
        if p == "X":
            if kind == "cart":
                _plot_cart_matrix(ax, X, "x")
                ax.set_title("X(t)")
            else:
                _plot_vehicle_panel(ax, X, "X(t)", overlay=False)
            ax.set_ylabel("X")

        elif p == "LAM":
            if kind == "cart":
                _plot_cart_matrix(ax, LAM, r"\lambda")
                ax.set_title(r"$\lambda(t)$")
            else:
                _plot_vehicle_panel(ax, LAM, "Costates λ", overlay=vehicle_overlay_components)
            ax.set_ylabel("λ")

        elif p == "E":
            if kind == "cart":
                _plot_cart_matrix(ax, E, "e")
                ax.set_title("E(t)")
            else:
                _plot_vehicle_panel(
                    ax, E,
                    r"Relative errors: $e_i - e_0$",
                    overlay=vehicle_overlay_components,
                )
                ax.axhline(0.0, ls="--", lw=1.0, alpha=0.6, color="gray")
            ax.set_ylabel("Relative deviations (e)", fontsize=16)

        elif p == "U":
            _plot_U(ax)
            ax.set_ylabel("Control (u)", fontsize=16)

        else:
            raise ValueError(f"Unknown panel '{p}'. Use 'X','LAM','E','U'.")

        #ax.grid(True, color="#DDDDDD")

    axs_list[-1].set_xlabel("Time (s)", fontsize=16)
    fig.suptitle(title or None, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.998])

    if show:
        plt.show()



    # for ax in axs_list:
    #     ax.set_xscale("log")
    #     tmin= 1e-1   
    #     ax.set_xlim(left=tmin) 

    # axs_list[-1].set_xlabel("Time (s)", fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.998])

    # if show:
    #     plt.show()


    return fig, axs_list



























def _resolve_colors(
    colors,
    n,
    *,
    cmap_range=(0.30, 0.90),
    as_hex=True,
):
    if n <= 0:
        raise ValueError("n must be >= 1")

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors)
        lo, hi = cmap_range
        xs = np.array([(lo + hi) / 2.0]) if n == 1 else np.linspace(lo, hi, n)
        cols = [cmap(float(x)) for x in xs]
        if as_hex:
            cols = [mcolors.to_hex(c) for c in cols]
        return cols

    cols = list(colors)
    if len(cols) == 0:
        raise ValueError("colors sequence cannot be empty")
    return [cols[i % len(cols)] for i in range(n)]


def plot_u_errors_overlay(
    trajectories,
    tr_ref,
    *,
    error_type="relative",     # "relative" or "absolute"
    reduce="elementwise",      # "elementwise" or "norm"
    magnitude=True,            # abs(error) if True
    eps=1e-12,
    components=None,           # only for elementwise
    legend="trajectory",       # "trajectory" | "method" | "all" | "none"
    alpha_element=0.9,
    alpha_norm=1.0,
    title=None,                # None/"" => no title
    yscale="log",              # "linear" | "log" | "symlog"
    symlog_linthresh=1e-10,
    symlog_linscale=1.0,
    # trajectory palette (your helper)
    traj_colors="Purples",
    traj_cmap_range=(0.30, 0.90),
    # labels/styles per trajectory
    traj_labels=None,          # list/tuple aligned with trajectories OR dict {id(tr): "..."} OR None (uses tr.label or tr.method)
    traj_linestyles=None,      # list like ["-","--",":","-."] or None
    # time window
    t_min=None,
    t_max=None,
    # style
    fontsize=16,
    linewidth=3.0,
):
    # ---- helpers ----
    def _u_to_2d(U):
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            return U.reshape(-1, 1)
        if U.ndim == 2:
            return U
        raise ValueError(f"U must be 1D or 2D, got {U.ndim}D")

    def _interp_u(t_ref, U_ref, t_new):
        t_ref = np.asarray(t_ref, float).reshape(-1)
        t_new = np.asarray(t_new, float).reshape(-1)
        U_ref = _u_to_2d(U_ref)
        U_new = np.empty((t_new.size, U_ref.shape[1]), float)
        for j in range(U_ref.shape[1]):
            U_new[:, j] = np.interp(t_new, t_ref, U_ref[:, j])
        return U_new

    def _align_method_to_ref(tr, tr_ref):
        t = np.asarray(tr.t, float).reshape(-1)
        U = _u_to_2d(tr.U)

        t0, t1 = tr_ref.t[0], tr_ref.t[-1]
        mask = (t >= t0) & (t <= t1)
        t_use = t[mask]
        U_use = U[mask, :]

        U_ref = _interp_u(tr_ref.t, tr_ref.U, t_use)
        if U_use.shape[1] != U_ref.shape[1]:
            raise ValueError(
                f"Control dim mismatch: {getattr(tr,'method','?')} has m={U_use.shape[1]}, "
                f"ref has m={U_ref.shape[1]}"
            )
        return t_use, U_use, U_ref

    def compute_u_error(tr, tr_ref):
        t_use, U, Uref = _align_method_to_ref(tr, tr_ref)
        diff = U - Uref

        if error_type == "absolute":
            err = diff if reduce == "elementwise" else np.linalg.norm(diff, axis=1)
        elif error_type == "relative":
            if reduce == "elementwise":
                denom = np.maximum(np.abs(Uref), eps)
                err = diff / denom
            else:
                num = np.linalg.norm(diff, axis=1)
                den = np.maximum(np.linalg.norm(Uref, axis=1), eps)
                err = num / den
        else:
            raise ValueError("error_type must be 'absolute' or 'relative'")
        return t_use, err

    def _get_traj_label(tr, i):
        if traj_labels is None:
            return getattr(tr, "label", getattr(tr, "method", "traj"))
        if isinstance(traj_labels, (list, tuple)):
            return traj_labels[i]
        if isinstance(traj_labels, dict):
            return traj_labels.get(id(tr), getattr(tr, "method", "traj"))
        raise ValueError("traj_labels must be None, list/tuple, or dict")


    # ---- plotting ----
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # components
    m_ref = _u_to_2d(tr_ref.U).shape[1]
    comp_idx = list(range(m_ref)) if components is None else list(components)

    # linestyles
    ls_cycle = ["-", "--", ":", "-."]

    # trajectories actually plotted (exclude ref-method)
    plot_trajs = [tr for tr in trajectories if getattr(tr, "method", None) != getattr(tr_ref, "method", None)]
    if len(plot_trajs) == 0:
        raise ValueError("No trajectories to plot (all have same method as tr_ref).")

    # base color per trajectory (your helper)
    base_cols = _resolve_colors(traj_colors, len(plot_trajs), cmap_range=traj_cmap_range, as_hex=True)
    base_color_by_id = {id(tr): c for tr, c in zip(plot_trajs, base_cols)}

    # store representative style for legend entries
    legend_style = {}  # key -> (color, linestyle)

    for i, tr in enumerate(trajectories):
        if getattr(tr, "method", None) == getattr(tr_ref, "method", None):
            continue

        tr_label = _get_traj_label(tr, i)

        ls = (ls_cycle[i % len(ls_cycle)] if traj_linestyles is None
              else traj_linestyles[i % len(traj_linestyles)])

        t_use, err = compute_u_error(tr, tr_ref)

        # time window
        if t_min is not None or t_max is not None:
            lo = -np.inf if t_min is None else t_min
            hi =  np.inf if t_max is None else t_max
            msk = (t_use >= lo) & (t_use <= hi)
            t_use = t_use[msk]
            err = err[msk]

        if magnitude:
            err = np.abs(err)

        # yscale handling
        if yscale == "log":
            if not magnitude:
                raise ValueError("yscale='log' incompatible with magnitude=False (signed error). Use 'symlog' or 'linear'.")
            err = np.maximum(err, eps)

        base_hex = base_color_by_id[id(tr)]

        # ---- plot ----
        if reduce == "norm":
            if legend == "trajectory":
                lab = tr_label
                key = ("trajectory", tr_label)
            elif legend == "method":
                lab = getattr(tr, "method", "traj")
                key = ("method", getattr(tr, "method", "traj"))
            elif legend == "all":
                lab = tr_label
                key = ("trajectory", tr_label)
            else:
                lab = "_nolegend_"
                key = None

            ax.plot(
                t_use, err,
                color=base_hex, alpha=alpha_norm, linestyle=ls, linewidth=linewidth,
                label=lab
            )

            if key is not None and key not in legend_style:
                legend_style[key] = (base_hex, ls)

        else:
        
            for k, j in enumerate(comp_idx):
                col = base_hex

                if legend == "all":
                    lab = f"{tr_label} u[{j}]"
                    key = ("all", tr_label, j)
                elif legend == "trajectory":
                    lab = tr_label if k == 0 else "_nolegend_"
                    key = ("trajectory", tr_label)
                elif legend == "method":
                    lab = getattr(tr, "method", "traj") if k == 0 else "_nolegend_"
                    key = ("method", getattr(tr, "method", "traj"))
                else:
                    lab = "_nolegend_"
                    key = None

                ax.plot(
                    t_use, err[:, j],
                    color=col, alpha=alpha_element, linestyle=ls, linewidth=linewidth,
                    label=lab
                )

                if key is not None and key not in legend_style:
                    legend_style[key] = (base_hex, ls)

    # ---- axes styling ----
    ax.grid(False)

    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, linscale=symlog_linscale, base=10)
    else:
        ax.set_yscale(yscale)

    ax.set_xlabel("Time (s)", fontsize=fontsize)

    # ylabel with norm formula if needed
    if reduce == "norm" and error_type == "absolute":
        ylab = r"$\|u_{\mathrm{DRE}}(t)-u_{\mathrm{method}}(t)\|_2$"
    elif reduce == "norm" and error_type == "relative":
        ylab = r"$\dfrac{\|u_{\mathrm{DRE}}(t)-u_{\mathrm{method}}(t)\|_2}{\max(\|u_{\mathrm{DRE}}(t)\|_2,\varepsilon)}$"
    else:
        ylab = r"$\mathrm{error}(t)$"

    ax.set_ylabel(ylab, fontsize=fontsize)

    # no title unless explicitly provided
    if title is not None and title != "":
        ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis="both", which="both", labelsize=fontsize)

    if t_min is not None or t_max is not None:
        ax.set_xlim(t_min if t_min is not None else None,
                    t_max if t_max is not None else None)

    # ---- legend ----
    if legend != "none":
        if legend in ("trajectory", "method"):
            handles, labels = [], []
            for key, (c, ls) in legend_style.items():
                if legend == "trajectory" and key[0] != "trajectory":
                    continue
                if legend == "method" and key[0] != "method":
                    continue
                name = key[1]
                handles.append(Line2D([0], [0], color=c, linewidth=2.5, linestyle=ls, alpha=1.0))
                labels.append(name)
            ax.legend(handles, labels, loc="upper left", fontsize=12)
        else:
            ax.legend(loc="best", fontsize=14)

    plt.tight_layout()
    return fig, ax
















