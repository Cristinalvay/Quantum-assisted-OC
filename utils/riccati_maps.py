from __future__ import annotations
import numpy as np

# -------------------------------------------------
# 1) Costates from errors
# -------------------------------------------------

def lam_of_e_are(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    ARE (constant P): λ(t_k) = P e(t_k)
    E: (N,n), P: (n,n) -> LAM: (N,n)
    """
    E = np.asarray(E, dtype=float)
    P = np.asarray(P, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must be (N,n), got {E.shape}")
    N, n = E.shape
    if P.shape != (n, n):
        raise ValueError(f"P must be {(n, n)}, got {P.shape}")

    # (N,n): (P @ E.T).T  ==  E @ P.T
    return E @ P.T


def lam_of_e_dre(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    DRE (time-varying P): λ(t_k) = P(t_k) e(t_k)
    E: (N,n), P: (N,n,n) -> LAM: (N,n)
    """
    E = np.asarray(E, dtype=float)
    P = np.asarray(P, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must be (N,n), got {E.shape}")
    N, n = E.shape
    if P.shape != (N, n, n):
        raise ValueError(f"P must be (N,n,n)={(N, n, n)}, got {P.shape}")

    return np.einsum("nij,nj->ni", P, E)


# -------------------------------------------------
# 2) Controls from errors
# -------------------------------------------------

def u_of_e_are(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    ARE (constant K): u(t_k) = -K e(t_k)
    E: (N,n), K: (m,n) -> U: (N,m) or (N,) if m==1
    """
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must be (N,n), got {E.shape}")
    N, n = E.shape
    if K.ndim != 2 or K.shape[1] != n:
        raise ValueError(f"K must be (m,{n}), got {K.shape}")

    U = -(E @ K.T)  # (N,m)
    return U[:, 0] if U.shape[1] == 1 else U


def u_of_e_dre(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    DRE (time-varying K): u(t_k) = -K(t_k) e(t_k)
    E: (N,n), K: (N,m,n) -> U: (N,m) or (N,) if m==1
    """
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must be (N,n), got {E.shape}")
    N, n = E.shape
    if K.ndim != 3 or K.shape[0] != N or K.shape[2] != n:
        raise ValueError(f"K must be (N,m,n) with N={N}, n={n}. Got {K.shape}")

    U = -np.einsum("nij,nj->ni", K, E)  # (N,m)
    return U[:, 0] if U.shape[1] == 1 else U


__all__ = [
    "lam_of_e_are",
    "lam_of_e_dre",
    "u_of_e_are",
    "u_of_e_dre",
] 
