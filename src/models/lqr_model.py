from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LQRModel:
    name: str

    # continuous-time linear system: x' = A x + B u
    A: np.ndarray
    B: np.ndarray

    # LQR weights
    Q: np.ndarray
    R: np.ndarray
    H: np.ndarray

    # reference + initial
    x_ref: np.ndarray
    x0: np.ndarray
    e0: np.ndarray

    # time grid
    T: float
    dt: float
    t_eval: np.ndarray  # fine time grid (for ODE integration)

    # derived (precomputed)
    C: np.ndarray              # Hamiltonian (2n x 2n)
    Rinv_BT: np.ndarray        # R^{-1} B^T  (m_inputs x n)

    # optional extras (params, leader info, etc.)
    params: Dict[str, Any] = None
    leader_at: Optional[Callable[[np.ndarray], np.ndarray]] = None  # for vehicle plots
    leader_traj: Optional[np.ndarray] = None                        # (N x 3) if you want
    leader_t: Optional[np.ndarray] = None                           # (N,) if you want

    @property
    def N(self) -> int:
        return int(len(self.t_eval))

    @property
    def t_span(self) -> Tuple[float, float]:
        return (0.0, float(self.T))