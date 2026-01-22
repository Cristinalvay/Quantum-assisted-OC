from __future__ import annotations

from typing import Optional, Sequence, Callable
import numpy as np
from scipy.integrate import solve_ivp

from methods.common import build_hamiltonian_C
from .lqr_model import LQRModel


def make_vehicle_platoon(
    *,
    n: int = 5,
    d: float = 5.0,
    tau: float = 0.5,
    T: float = 10.0,
    dt: float = 0.001,
    # initial absolute trajectories (leader + n followers) length n+1
    s_abs: Optional[np.ndarray] = None,
    v_abs: Optional[np.ndarray] = None,
    a_abs: Optional[np.ndarray] = None,
    # weights (editable)
    Q0: Optional[np.ndarray] = None,
    R0: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    # leader input (editable)
    u0: Optional[Callable[[float], float]] = None,
    integrate_leader: bool = True,
) -> LQRModel:
    """
    Vehicle platoon (n followers) model in error coordinates e ∈ R^{3n}.

    Editable:
      - initial abs states s_abs/v_abs/a_abs
      - per-vehicle weights Q0, R0
      - terminal H (full matrix)
      - leader input u0(t)
      - integrate_leader toggle
    """
    # ------------------- time grid -------------------
    N = int(T / dt) + 1
    t_eval = np.linspace(0.0, T, N)

    # ----------------- Single vehicle model ----------------
    # (d/dt(X) = A0*X + b0*U)
    A0 = np.array(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 0.0, -1.0 / tau]],
        dtype=float,
    )
    b0 = np.array([[0.0], [0.0], [1.0 / tau]], dtype=float)

    # weights defaults
    if Q0 is None:   # on [spacing error, rel speed, rel accel]
        Q0 = np.diag([1.0, 1.0, 1.0]).astype(float)
    else:
        Q0 = np.asarray(Q0, dtype=float)
    if R0 is None:   # scalar input weight (control effort)
        R0 = np.array([[0.5]], dtype=float)
    else:
        R0 = np.asarray(R0, dtype=float)

    if Q0.shape != (3, 3):
        raise ValueError(f"Q0 must be (3,3), got {Q0.shape}")
    if R0.shape != (1, 1):
        raise ValueError(f"R0 must be (1,1), got {R0.shape}")

    # ----------------- Stacked n-followers model --------------
    # d/dt(X) = Af*X + bf*U
    # d/dt(e) = Af*e + bf*\tilde{U}             \tilde{U} = U - (1n \otimes u0)
    A = np.kron(np.eye(n), A0)  # (3n x 3n)
    B = np.kron(np.eye(n), b0)  # (3n x n)

    Q = np.kron(np.eye(n), Q0)  # (3n x 3n)
    R = np.kron(np.eye(n), R0)  # (n x n)

    if H is None:
        H = np.eye(3 * n, dtype=float)
    else:
        H = np.asarray(H, dtype=float)
    if H.shape != (3 * n, 3 * n):
        raise ValueError(f"H must be {(3*n,3*n)}, got {H.shape}")

    # ----------------- initial absolute data ----------------
    # x = [x1; x2; ...; xn] where xi = [si; vi; ai] for vehicle i
    # u = [u1; u2; ...; un] where ui is the input for vehicle i
    if s_abs is None:
        s0 = 300.0
        d0 = 60.0
        s_abs = s0 - np.arange(0, n+1) * d0   # length n+1, spacing ideal
    if v_abs is None:
        v_abs = np.full(n+1, 7.0, dtype=float)
    if a_abs is None:
        a_abs = np.zeros(n+1, dtype=float)

    s_abs = np.asarray(s_abs, dtype=float).reshape(-1)
    v_abs = np.asarray(v_abs, dtype=float).reshape(-1)
    a_abs = np.asarray(a_abs, dtype=float).reshape(-1)

    if len(s_abs) != n + 1 or len(v_abs) != n + 1 or len(a_abs) != n + 1:
        raise ValueError("s_abs, v_abs, a_abs must have length n+1 (leader + n followers).")

    x0_lead = np.array([s_abs[0], v_abs[0], a_abs[0]], dtype=float)

    # initial errors e0 ∈ R^{3n}  (relative to leader)
    # e_i = Xi - X0 + [id; 0; 0] = [s_i - s_0 - i*d;  v_i - v_0;  a_i - a_0]
    ef_list = []
    # Initial followers state (absolute) 
    for i in range(n):
        s_i, v_i, a_i = s_abs[i + 1], v_abs[i + 1], a_abs[i + 1]
        ef_list += [(s_i - s_abs[0]) + (i+1)*d,    # spacing: follower − leader + id
                (v_i - v_abs[0]),              # relative speed
                (a_i - a_abs[0])]              # relative accel
    e0 = np.array(ef_list, dtype=float)        # e(0) ∈ R^{3n}

    x_ref = np.zeros_like(e0)
    x0 = e0.copy()

    # ----------------- leader input & integration ----------------
    if u0 is None:
        def u0(t: float) -> float:
            return 20.0 * np.sin(2.0 * t)  # leader input can be time dependent!

    def leader_ode(t, x):                  # x = [s0, v0, a0]
        return (A0 @ x.reshape(3, 1) + b0 * u0(t)).ravel()

    leader_traj = None
    leader_t = None
    leader_at = None

    # Leader can be integrated before any LQR calculations as it is exogenous
    if integrate_leader:
        sol = solve_ivp(
            leader_ode, (0.0, T), x0_lead, t_eval=t_eval,
            method="Radau", rtol=1e-10, atol=1e-12
        )
        leader_traj = sol.y.T # (N x 3), columns [s0,v0,a0]
        leader_t = sol.t      # (N x 1)

        def leader_at(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t, dtype=float).reshape(-1)
            out = np.zeros((t.size, 3), dtype=float)
            for j in range(3):
                out[:, j] = np.interp(t, leader_t, leader_traj[:, j])
            return out

    # ----------------- Hamiltonian ----------------
    C, Rinv_BT = build_hamiltonian_C(A, B, Q, R)

    return LQRModel(
        name="vehicle_platoon",
        A=A, B=B, Q=Q, R=R, H=H,
        x_ref=x_ref, x0=x0, e0=e0,
        T=T, dt=dt, t_eval=t_eval,
        C=C, Rinv_BT=Rinv_BT,
        leader_at=leader_at,
        leader_traj=leader_traj,
        leader_t=leader_t,
        params={
            "n": n, "d": d, "tau": tau,
            "A0": A0, "b0": b0,
            "Q0": Q0, "R0": R0,
            "u0": u0,
        },
    )
