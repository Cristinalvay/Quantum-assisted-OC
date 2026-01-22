from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .common import four_partition, estimate_kappa


@dataclass(frozen=True)
class VSystem:
    V: sparse.csr_matrix
    rhs: np.ndarray
    S_step: np.ndarray
    dt: float


@dataclass(frozen=True)
class VMethodResult:
    system: VSystem
    z: np.ndarray          # stacked solution (m*2n,)
    Z: np.ndarray          # reshaped (m,2n)
    E: np.ndarray          # (m,n)
    LAM: np.ndarray        # (m,n)
    lamb0: np.ndarray      # (n,)
    lambT: np.ndarray      # (n,)
    condV: float


def assemble_stack(C: np.ndarray, H: np.ndarray, b: np.ndarray, T: float, m: int) -> VSystem:
    """
    Discrete Hamiltonian evolution (V method):
    builds sparse V such that V * z = rhs, where z stacks y_k = [e_k; λ_k], k=0..m-1.

    Constraints:
      - initial condition: e_0 = b
      - recurrence: y_k - S y_{k-1} = 0 (S = expm(C*dt))
      - terminal: [-H  I] y_{m-1} = 0
    """
    C = np.asarray(C, dtype=float)
    n2 = C.shape[0]
    if C.ndim != 2 or C.shape[0] != C.shape[1] or (n2 % 2) != 0:
        raise ValueError("C must be square (2n×2n)")
    n = n2 // 2

    H = np.asarray(H, dtype=float)
    if H.shape != (n, n):
        raise ValueError(f"H must have shape {(n, n)}, got {H.shape}")

    b = np.asarray(b, dtype=float).reshape(-1)
    if b.shape != (n,):
        raise ValueError(f"b must have shape {(n,)}, got {b.shape}")

    if m < 2:
        raise ValueError("m must be >= 2")
    if T <= 0:
        raise ValueError("T must be > 0")

    dt = T / (m - 1)
    S, U00, U01, U10, U11 = four_partition(C, dt)

    rows, cols, data = [], [], []
    rhs = []
    row = 0

    # Initial condition: [I 0] y_0 = b  --> e0 = b  (top half of y0)
    for i in range(n):
        rows.append(row) # row index 
        cols.append(i)   # column index
        # This two create A[0,0], A[1,1], A[2,2], A[3,3] (till n-1 which is 3)
        data.append(1.0) # value to create I_n
        rhs.append(b[i]) # right hand side vector 
        row += 1

   # The non-specified values are 0   
    # Keep from row = n (in this case 4th row)  

    # Recurrences: y_k - S y_{k-1} = 0, k=1..m-1

    # Need to do for k = 1, rows n ... n+(2n-1)
    # Need to do for k = 2, rows n+2n ... n+(2*2n-1)
    for k in range(1, m):
        for r in range(n2): 
            # +I on y_k[r]
            rows.append(row) 
            cols.append(k*n2 + r)  # column for y_k[r]
            data.append(1.0)
            # y_{k-1} coefficient (-S)
            for c in range(n2):
                rows.append(row)
                cols.append((k-1)*n2 + c)
                data.append(-S[r, c])
            rhs.append(0.0) 
            row += 1

    # Terminal: [-H I] y_{m-1} = 0
    off = (m-1)*n2  # starting column for block y_{m-1}
    for i in range(n):
        for c in range(n):
            rows.append(row)
            cols.append(off + c)
            data.append(-H[i, c])
        rows.append(row)
        cols.append(off + n + i)
        data.append(1.0)
        rhs.append(0.0)
        row += 1

    V = sparse.coo_matrix((data, (rows, cols)), shape=(row, m * n2)).tocsr()
    rhs = np.array(rhs, dtype=float)

    if V.shape[0] != V.shape[1]:
        raise ValueError(f"V must be square; got {V.shape}")

    return VSystem(V=V, rhs=rhs, S_step=S, dt=dt)


def solve_v_method(C: np.ndarray, H: np.ndarray, e0: np.ndarray, T: float, m: int) -> VMethodResult:
    sys = assemble_stack(C, H, e0, T, m)

    z = spsolve(sys.V, sys.rhs)   # (m*2n,)
    z = np.asarray(z, dtype=float).reshape(-1)

    n2 = C.shape[0]
    n = n2 // 2

    Z = z.reshape(m, n2)
    E = Z[:, :n]
    LAM = Z[:, n:]

    lamb0 = LAM[0].copy()
    lambT = LAM[-1].copy()

    _, _, condV, _ = estimate_kappa(sys.V)

    return VMethodResult(system=sys, z=z, Z=Z, E=E, LAM=LAM, lamb0=lamb0, lambT=lambT,condV=float(condV))



