from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from methods.common import build_hamiltonian_C
from .lqr_model import LQRModel


def make_cart_pendulum(
    *,
    MODEL_NAME: str = "cart_pendulum",
    T: float = 10.0,
    dt: float = 0.001,
    m: float = 1.0,
    M: float = 5.0,
    L: float = 2.0,
    g: float = -10.0,
    delta: float = 1.0,
    # weights (optional overrides)
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    R_scalar: Optional[float] = None,
    H_diag: Optional[Sequence[float]] = None,
    # reference / initial (optional overrides)
    x_ref: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
) -> LQRModel:
    """
    Build Cart-Pendulum LQRModel.

    Overrides:
      - Q, R, H can be passed directly (arrays)
      - R_scalar sets R = [[R_scalar]]
      - H_diag sets H = diag(H_diag)
      - x_ref and x0 can be passed; e0 is computed as e0 = x0 - x_ref
    """
    # ------------------ System matrices ------------------
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -delta / M, m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -delta / (M * L), -(M + m) * g / (M * L), 0.0],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.0],
            [1.0 / M],
            [0.0],
            [1.0 / (M * L)],
        ],
        dtype=float,
    )

    n = A.shape[0]  # 4

    # ------------------ Defaults / overrides for Q,R,H ------------------
    if Q is None:
        Q = np.eye(n, dtype=float)
    else:
        Q = np.asarray(Q, dtype=float)

    if R_scalar is not None:
        R = np.array([[float(R_scalar)]], dtype=float)
    elif R is None:
        R = np.array([[1e-2]], dtype=float)
    else:
        R = np.asarray(R, dtype=float)

    if H_diag is not None:
        H = np.diag(np.asarray(H_diag, dtype=float))
    elif H is None:
        H = np.diag([0.0, 10.0, 50.0, 10.0]).astype(float)
    else:
        H = np.asarray(H, dtype=float)

    # ------------------ Validate shapes ------------------
    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape {(n,n)}, got {Q.shape}")
    if R.shape != (1, 1):
        raise ValueError(f"R must have shape (1,1) for cart pendulum, got {R.shape}")
    if H.shape != (n, n):
        raise ValueError(f"H must have shape {(n,n)}, got {H.shape}")

    # ------------------ Reference / initial ------------------
    if x_ref is None:
        x_ref = np.array([1.0, 0.0, np.pi, 0.0], dtype=float)
    else:
        x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    if x0 is None:
        x0 = np.array([-1.0, 0.0, np.pi + 0.1, 0.0], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(-1)

    if x_ref.shape != (n,):
        raise ValueError(f"x_ref must have shape {(n,)}, got {x_ref.shape}")
    if x0.shape != (n,):
        raise ValueError(f"x0 must have shape {(n,)}, got {x0.shape}")

    e0 = x0 - x_ref

    # ------------------ Time grid ------------------
    N = int(T / dt) + 1
    t_eval = np.linspace(0.0, T, N)

    # ------------------ Precomputations ------------------
    C, Rinv_BT = build_hamiltonian_C(A, B, Q, R)

    return LQRModel(
        name=MODEL_NAME,
        A=A,
        B=B,
        Q=Q,
        R=R,
        H=H,
        x_ref=x_ref,
        x0=x0,
        e0=e0,
        T=T,
        dt=dt,
        t_eval=t_eval,
        C=C,
        Rinv_BT=Rinv_BT,
        params={"m": m, "M": M, "L": L, "g": g, "delta": delta},
    )

# --- Usage examples ---
# Changing only x0: model = make_cart_pendulum(x0=[-2, 0, np.pi+0.2, 0])
# Changing reference: model = make_cart_pendulum(x_ref=[0, 0, np.pi, 0])
# Changing Q, R, H: model = make_cart_pendulum(Q=np.diag([10, 1, 10, 1]), R_scalar=1e-3, H_diag=[0, 20, 80, 20])
