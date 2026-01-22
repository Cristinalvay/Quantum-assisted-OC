import numpy as np
from scipy.linalg import expm, norm


def gram_schmidt(a):
    """
    Create an n*n unitary matrix where the first column is replaced by the unit complex vector a,
    and the rest are the columns of the identity matrix.

    Parameters:
    a (numpy.ndarray): An n*1 unit complex vector.

    Returns:
    numpy.ndarray: An n*n unitary matrix.
    """
    if isinstance(a, list):
        a = np.array(a)

    # Create an n*n identity matrix with complex entries
    n = len(a) #n (int): The dimension of the matrix.
    U = np.eye(n, dtype=np.complex128)

    # Replace the first column with the vector a
    U[:, 0] = a.flatten()

    # Perform the Gram-Schmidt orthogonalization process
    for j in range(n):
        # Take the j-th column
        v = U[:, j]
        # Subtract its projection onto the already orthogonalized basis vectors
        for i in range(j):
            v -= np.dot(U[:, i].conj().T, v) * U[:, i]
        # Normalize the column
        U[:, j] = v / np.linalg.norm(v)

    return U

def initial_schro_fp(p, order=1):
    """
    The initial function for the Schrodingerisation scheme.
    """

    fp = np.exp(-np.abs(p))
    if order == 2:
        indices = np.where((p>-1) & (p<0))[0]
        fp[indices] = (3/np.e-3) * p[indices]**3 + (4/np.e-5) * p[indices]**2 - p[indices] + 1
    return fp


def qft_matrix(n_qubits: int, bit_reversal: bool = False, dtype=np.complex128) -> np.ndarray:
    """
    Construct the QFT matrix for n_qubits qubits (size N = 2**n_qubits).

    Parameters
    ----------
    n_qubits : int
        Number of qubits, n >= 1.
    bit_reversal : bool
        Whether to multiply the matrix on the right by the bit-reversal permutation
        (i.e., the pairwise SWAPs at the end of the circuit) to restore the “natural
        binary order”.
        - False: Return the pure DFT form F_N (the mathematical definition), equivalent
        to having performed the final SWAPs in the circuit.
        - True : Return the “circuit-native” output ordering (omit the final SWAPs),
        which is equivalent to applying bit-reversal to the columns.
    dtype : numpy dtype
        Complex precision, default is np.complex128.

    Returns
    -------
    U : np.ndarray, shape (N, N), dtype=dtype
        Matrix representation of the QFT (a unitary matrix).

    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    N = 1 << n_qubits  # 2**n_qubits

    # 生成指数表 j*k
    j = np.arange(N, dtype=np.int64).reshape(N, 1)
    k = np.arange(N, dtype=np.int64).reshape(1, N)
    exponent = j * k  # (N, N)

    # 复指数根
    omega = np.exp(2j * np.pi / N).astype(dtype)

    # DFT/QFT 矩阵
    U = (omega ** exponent) / np.sqrt(N)
    U = U.astype(dtype, copy=False)

    if bit_reversal:
        # 对列执行 bit-reversal（等价于省略电路末端 SWAP）
        # 生成 [0..N-1] 的反转位序索引
        idx = np.arange(N)
        # 把 idx 的二进制位反转
        rev = np.zeros_like(idx)
        for b in range(n_qubits):
            rev = (rev << 1) | ((idx >> b) & 1)
        U = U[:, rev]
    return U

def iqft_matrix(n_qubits: int, bit_reversal: bool = False, dtype=np.complex128) -> np.ndarray:
    """IQFT（QFT†）矩阵，直接取共轭转置即可。"""
    U = qft_matrix(n_qubits, bit_reversal=bit_reversal, dtype=dtype)
    return U.conj().T






__all__ = ['gram_schmidt', 'initial_schro_fp', 'qft_matrix', 'iqft_matrix']