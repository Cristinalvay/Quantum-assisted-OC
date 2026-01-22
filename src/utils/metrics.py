from __future__ import annotations
import numpy as np

def rel_norm(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    """
    Relative L2 error: ||a-b|| / max(||a||, ||b||, eps)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / max(na, nb, eps))
