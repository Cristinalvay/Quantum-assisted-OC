from __future__ import annotations
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from utils.trajectory import x_of_e_cart, x_of_e_vehicle




def integrate_with_lamb0_yielded(
    *,
    C: np.ndarray,
    e0: np.ndarray,
    lamb0: np.ndarray,
    T: float,
    t_eval: np.ndarray,
    # reconstruction controls
    model_name: str | None = None,
    x_ref: Optional[np.ndarray] = None,          # for cart
    leader_traj: Optional[np.ndarray] = None,    # for vehicle
    d: Optional[float] = None,                   # for vehicle
    n_followers: Optional[int] = None,           # for vehicle
    # control
    Rinv_BT: Optional[np.ndarray] = None,
    method: str = "Radau",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[OdeResult, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Integrate the linear Hamiltonian ODE:
        z' = C z,  where z = [e; λ] ∈ R^{2n}

    Parameters
    ----------
    C:
        (2n x 2n) Hamiltonian matrix.
    e0:
        (n,) initial error/state part.
    lamb0:
        (n,) initial costate part.
    T:
        final time.
    t_eval:
        times at which to sample the solution (1D array).
    x_ref:
        optional (n,) reference state to reconstruct X = E + x_ref.
        If None, X is returned as None.
    Rinv_BT:
        optional (m x n) matrix R^{-1} B^T, to compute control:
            U = - R^{-1} B^T λ
        If None, U is returned as None.

    Returns
    -------
    sol:
        solve_ivp solution object
    E:
        (len(t_eval), n) error trajectory
    LAM:
        (len(t_eval), n) costate trajectory
    X:
        (len(t_eval), n) state trajectory (optional, depends on model_name + params)
    U:
        (len(t_eval), m) control trajectory  (optional if Rinv_BT provided)
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or (C.shape[0] % 2) != 0:
        raise ValueError("C must be square (2n×2n).")

    n2 = C.shape[0]
    n = n2 // 2

    e0 = np.asarray(e0, dtype=float).reshape(-1)
    lamb0 = np.asarray(lamb0, dtype=float).reshape(-1)

    if e0.shape != (n,):
        raise ValueError(f"e0 must have shape {(n,)}, got {e0.shape}.")
    if lamb0.shape != (n,):
        raise ValueError(f"lamb0 must have shape {(n,)}, got {lamb0.shape}.")
    if T <= 0:
        raise ValueError("T must be > 0.")

    # ODE: z' = C z
    def ode(t, z):
        return C @ z

    z0 = np.concatenate([e0, lamb0])  # (2n,)

    # For Radau/BDF, providing constant Jacobian helps
    jac = (lambda t, z: C) if method.lower() in {"radau", "bdf"} else None

    sol = solve_ivp(
        ode,
        (0.0, float(T)),
        z0,
        t_eval=np.asarray(t_eval, dtype=float),
        method=method,
        jac=jac,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    Z = sol.y.T  # (N, 2n)
    E = Z[:, :n]
    LAM = Z[:, n:]

    # -------- reconstruct X depending on model --------
    X: Optional[np.ndarray] = None
    if model_name is not None:
        if model_name == "cart_pendulum":
            if x_ref is None:
                raise ValueError("For model_name='cart_pendulum' you must provide x_ref.")
            X = x_of_e_cart(E, x_ref)

        elif model_name == "vehicle_platoon":
            if leader_traj is None or d is None or n_followers is None:
                raise ValueError(
                    "For model_name='vehicle_platoon' you must provide leader_traj, d, and n_followers."
                )
            X = x_of_e_vehicle(E, leader_traj, float(d), int(n_followers))

        else:
            # unknown model -> leave X as None
            X = None

    # -------- compute control U --------
    U: Optional[np.ndarray] = None
    if Rinv_BT is not None:
        Rinv_BT = np.asarray(Rinv_BT, dtype=float)
        if Rinv_BT.ndim != 2 or Rinv_BT.shape[1] != n:
            raise ValueError(f"Rinv_BT must have shape (m, {n}), got {Rinv_BT.shape}.")
        U = (-(Rinv_BT @ LAM.T)).T  # (N, m)
        if U.shape[1] == 1:
            U = U[:, 0]  # squeeze to (N,) for scalar control

    return sol, E, LAM, X, U
  




