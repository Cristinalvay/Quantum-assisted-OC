from __future__ import annotations
from utils.trajectory import x_of_e_cart, x_of_e_vehicle

from typing import Optional, Tuple
import numpy as np
from scipy.linalg import expm, solve
from scipy.sparse import issparse
from scipy.sparse.linalg import svds, splu, LinearOperator
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence

def build_hamiltonian_C(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the Hamiltonian matrix:
        C = [[ A,   -B R^{-1} B^T ],
             [ -Q,  -A^T        ]]
    Works for B (nxm), R (mxm), Q (nxn).

    Also computes R^{-1} B^T for later use.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if Q.shape != (n, n):
        raise ValueError("Q must have same shape as A")
    if B.shape[0] != n:
        raise ValueError("B must have same number of rows as A")
    if R.shape[0] != R.shape[1] or R.shape[0] != B.shape[1]:
        raise ValueError("R must be square with size equal to number of inputs (B.shape[1])")

    # R^{-1} B^T, stable solve
    Rinv_BT = solve(R, B.T)          # (m×n)
    BRB = B @ Rinv_BT                   # (n×n)

    C = np.block([[A,    -BRB],
                  [-Q,   -A.T]])
    return C, Rinv_BT


def four_partition(C: np.ndarray, T: float):
    """
    Partition expm(C*T) into 4 blocks:
      S = expm(C*T) = [[U00, U01],
                       [U10, U11]]
    where each Uij is (nxn) with n = C.shape[0]//2.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or (C.shape[0] % 2) != 0:
        raise ValueError("C must be square (2nx2n)")
    if T <= 0:
        raise ValueError("T must be > 0")

    n2 = C.shape[0]
    n = n2 // 2

    S = expm(C * T)
    U00 = S[:n, :n]
    U01 = S[:n, n:]
    U10 = S[n:, :n]
    U11 = S[n:, n:]
    return S, U00, U01, U10, U11

# For dense and not too large matrices
def inverse_check_cond(M: np.ndarray, warn_thresh: float = 1e16) -> float:
    """ 
    Check if M is invertible by looking at its condition number.
    The condition number is the ratio of the largest to smallest singular value (Singular value: non-negative square root of eigenvalue of M^T M).
    If cond(M) is very large, M is close to singular and numerical errors may be amplified (M can stretch or compress space too much in some directions).
    """
    M = np.asarray(M, dtype=float)
    c = float(np.linalg.cond(M))
    if c > warn_thresh:
        print(f"Warning: matrix ill-conditioned (cond ~ {c:.2e}).")
    return c

# For sparse and/or large matrices
def estimate_kappa(
    A,
    *,
    dense_threshold: int = 2500,
    tol: float = 1e-6,
    maxiter: int = 20000,
):
    """
    Returns (smin, smax, kappa, used_dense).

    Robust for sparse matrices:
      - smax from svds(A, which="LM")
      - smin tries svds(A, which="SM"), and if ARPACK fails:
          smin ≈ 1 / smax(A^{-1}) using LU + LinearOperator
    """
    # --- Dense path ---
    if not issparse(A):
        Ad = np.asarray(A, dtype=float)
        s = np.linalg.svd(Ad, compute_uv=False)
        smax = float(s[0])
        smin = float(s[-1])
        smin = max(smin, np.finfo(float).tiny)
        return smin, smax, smax / smin, True

    # --- Sparse path ---
    A = A.tocsr()
    n, m = A.shape
    if n != m:
        raise ValueError(f"estimate_kappa expects square matrix, got {A.shape}")

    # if small enough, convert to dense (exact, but O(n^3))
    if n <= dense_threshold:
        Ad = A.toarray()
        s = np.linalg.svd(Ad, compute_uv=False)
        smax = float(s[0])
        smin = float(s[-1])
        smin = max(smin, np.finfo(float).tiny)
        return smin, smax, smax / smin, True

    # 1) smax is usually easy
    smax = float(
        svds(A, k=1, which="LM", return_singular_vectors=False, tol=tol, maxiter=maxiter)[0]
    )

    # 2) try smin directly
    try:
        smin = float(
            svds(A, k=1, which="SM", return_singular_vectors=False, tol=tol, maxiter=maxiter)[0]
        )
        smin = max(smin, np.finfo(float).tiny)
        return smin, smax, smax / smin, False

    except (ArpackNoConvergence, RuntimeError):
        # 3) fallback: smin ≈ 1 / smax(A^{-1})
        try:
            lu = splu(A.tocsc())

            def matvec(v):
                return lu.solve(v)

            Ainv = LinearOperator(A.shape, matvec=matvec, rmatvec=matvec, dtype=float)

            smax_inv = float(
                svds(Ainv, k=1, which="LM", return_singular_vectors=False, tol=tol, maxiter=maxiter)[0]
            )
            smin = 1.0 / max(smax_inv, np.finfo(float).tiny)
            return smin, smax, smax / max(smin, np.finfo(float).tiny), False

        except Exception:
            # if LU fails (singular / extremely ill-conditioned), just return inf kappa
            return 0.0, smax, float("inf"), False

