from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

from models.lqr_model import LQRModel 
import numpy as np

# -------------------------
# Reconstruct X from error
# -------------------------

# Cart-pendulum
def x_of_e_cart(E: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """
    Reconstruct state trajectory X from error trajectory E for cart-pendulum:
        X(t) = E(t) + x_ref

    Parameters
    ----------
    E : (N, n) array
        Error trajectory.
    x_ref : (n,) array
        Reference state.

    Returns
    -------
    X : (N, n) array
        State trajectory.
    """
    E = np.asarray(E, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    if E.ndim != 2:
        raise ValueError(f"E must be 2D (N,n), got shape {E.shape}")
    n = E.shape[1]
    if x_ref.shape != (n,):
        raise ValueError(f"x_ref must have shape {(n,)}, got {x_ref.shape}")

    return E + x_ref[None, :]
# from utils.trajectory import x_of_e_cart
# X = x_of_e_cart(E, model.x_ref)




# Vehicle platoon
def x_of_e_vehicle(E: np.ndarray, X0_traj: np.ndarray, d: float, n_followers: int) -> np.ndarray:
    """
    Reconstruct absolute follower states X_f(t) from error trajectory E(t) and leader state X0(t).

    Convention used in your notebook:
      e_i = [ s_i - s0 + i*d ,  v_i - v0 ,  a_i - a0 ]   for i=1..n

    Therefore:
      s_i = e_s_i + s0 - i*d
      v_i = e_v_i + v0
      a_i = e_a_i + a0

    Parameters
    ----------
    E : (N, 3n) array
        Error trajectory stacked follower-by-follower.
    X0_traj : (N, 3) array
        Leader state trajectory [s0, v0, a0] at the same N times.
    d : float
        Desired spacing.
    n_followers : int
        Number of followers (n).

    Returns
    -------
    Xf : (N, 3n) array
        Absolute follower states stacked [X1; X2; ...; Xn] with Xi=[si,vi,ai].
    """
    E = np.asarray(E, dtype=float)
    X0_traj = np.asarray(X0_traj, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must be 2D (N,3n), got shape {E.shape}")
    if X0_traj.ndim != 2 or X0_traj.shape[1] != 3:
        raise ValueError(f"X0_traj must be (N,3), got shape {X0_traj.shape}")
    if E.shape[0] != X0_traj.shape[0]:
        raise ValueError(f"E and X0_traj must have same N. Got {E.shape[0]} vs {X0_traj.shape[0]}")

    N, dim = E.shape
    if dim != 3 * n_followers:
        raise ValueError(f"E must have 3*n columns. Got {dim}, expected {3*n_followers}")

    # r_vec = [1*d,0,0, 2*d,0,0, ..., n*d,0,0]  shape (3n,)
    r_vec = np.zeros(3 * n_followers, dtype=float)
    for i in range(n_followers):
        r_vec[3*i + 0] = (i + 1) * d  # only spacing component

    # tile leader state across followers: (N,3) -> (N,3n)
    X0_tiled = np.tile(X0_traj, (1, n_followers))

    # Xf = E - r_vec + X0_tiled
    return E - r_vec[None, :] + X0_tiled
# from utils.trajectory import x_of_e_vehicle
# X = x_of_e_vehicle(E, model.leader_traj, d=model.params["d"], n_followers=model.params["n"])


def compute_control_from_lam(LAM: np.ndarray, Rinv_BT: np.ndarray) -> np.ndarray:
    """
    Unified control computation:
      U = - R^{-1} B^T λ

    LAM: (N, n_state)
    Rinv_BT: (m_inputs, n_state)
    Returns:
      (N, m_inputs) or (N,) if m_inputs==1
    """
    LAM = np.asarray(LAM, dtype=float)
    Rinv_BT = np.asarray(Rinv_BT, dtype=float)

    if Rinv_BT.ndim != 2:
        raise ValueError("Rinv_BT must be 2D (m_inputs x n_state)")
    if LAM.ndim != 2:
        raise ValueError("LAM must be 2D (N x n_state)")
    if Rinv_BT.shape[1] != LAM.shape[1]:
        raise ValueError(f"Rinv_BT shape {Rinv_BT.shape} incompatible with LAM shape {LAM.shape}")

    U = (-(Rinv_BT @ LAM.T)).T  # (N, m_inputs)
    if U.shape[1] == 1:
        return U[:, 0]
    return U




# -------------------------
# Compact trajectory info for plotting/comparisons
# -------------------------
Method = Literal["M", "V", "ARE", "DRE"]  

@dataclass(frozen=True)
class TrajectoryResult:
    """
    Standard container for a trajectory produced by any method (M, V, ARE, DRE, ...).
    """
    method: Method  
    model_name: str           # "cart_pendulum" or "vehicle_platoon"
    t: np.ndarray
    E: np.ndarray
    LAM: np.ndarray
    X: np.ndarray
    U: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model_name not in ("cart_pendulum", "vehicle_platoon"):
            raise ValueError(f"Unknown model_name={self.model_name!r}")

        t = np.asarray(self.t, dtype=float).reshape(-1)
        E = np.asarray(self.E, dtype=float)
        LAM = np.asarray(self.LAM, dtype=float)
        X = np.asarray(self.X, dtype=float)
        U = np.asarray(self.U, dtype=float)

        N = t.size
        if t.ndim != 1:
            raise ValueError("t must be 1D")

        if E.ndim != 2 or LAM.ndim != 2:
            raise ValueError("E and LAM must be 2D arrays (N, dim)")
        if E.shape != LAM.shape:
            raise ValueError(f"E and LAM must have same shape. Got {E.shape} vs {LAM.shape}")
        if E.shape[0] != N:
            raise ValueError(f"t and E must have same N. Got {N} vs {E.shape[0]}")

        if X.ndim != 2 or X.shape[0] != N:
            raise ValueError(f"X must be 2D with shape (N, dim). Got {X.shape}, N={N}")
        if X.shape[1] != E.shape[1]:
            raise ValueError(f"X and E must have same dim. Got X {X.shape[1]} vs E {E.shape[1]}")

        if U.ndim == 1:
            if U.shape[0] != N:
                raise ValueError(f"U must have length N. Got {U.shape}, N={N}")
        elif U.ndim == 2:
            if U.shape[0] != N:
                raise ValueError(f"U must have shape (N, m). Got {U.shape}, N={N}")
        else:
            raise ValueError("U must be 1D or 2D")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "LAM", LAM)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "U", U)



def make_trajectory_result(
    *,
    method: Method,       
    model: LQRModel,
    t: np.ndarray,
    E: np.ndarray,
    LAM: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    meta: Optional[Mapping[str, Any]] = None,
) -> TrajectoryResult:
    
    return TrajectoryResult(
        method=method,
        model_name=model.name,
        t=t,
        E=E,
        LAM=LAM,
        X=X,
        U=U,
        meta=dict(meta or {}),
    )


__all__ = [
    "x_of_e_cart",
    "x_of_e_vehicle",
    "compute_control_from_lam",
    "make_trajectory_result",
]