"""
Linear Algebra Operations for NEGF Quantum Transport

This module provides optimized linear algebra operations for Non-Equilibrium
Green's Function (NEGF) calculations with automatic GPU acceleration using
CuPy when available.

Author: William Livernois
"""

try:
    import cupy as cp
    isCuda = cp.cuda.is_available()
except:
    isCuda = False

import numpy as np


def times(A, B, C=None, use_gpu=None):
    """
    Optimized matrix multiplication with optional GPU acceleration.
    Performs A @ B or A @ B @ C with automatic GPU/CPU selection and precision optimization.

    Parameters
    ----------
    A, B : ndarray
        Input matrices for multiplication
    C : ndarray, optional
        Third matrix for triple product A @ B @ C
    use_gpu : bool, optional
        Force GPU usage (True) or CPU usage (False). If None, auto-detect.

    Returns
    -------
    ndarray
        Result of matrix multiplication A @ B or A @ B @ C
    """
    if use_gpu is None:
        use_gpu = isCuda

    solver = cp if use_gpu else np

    # Convert inputs to device arrays
    A_device = solver.asarray(A)
    B_device = solver.asarray(B)

    if C is None:
        # Simple multiplication A @ B
        result = solver.matmul(A_device, B_device)
    else:
        # Triple product A @ B @ C (common in NEGF: Gr @ Gamma @ Ga)
        C_device = solver.asarray(C)
        if use_gpu:
            # Use intermediate result to minimize GPU memory allocation
            temp = solver.matmul(A_device, B_device)
            result = solver.matmul(temp, C_device)
        else:
            # CPU can handle chained operations efficiently
            result = solver.matmul(solver.matmul(A_device, B_device), C_device)

    # Return CPU array
    return result.get() if use_gpu else result


def Gr(F, S, g, E):
    """
    Calculate retarded Green's function at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    ndarray
        Retarded Green's function G(E) = [ES - F - Σ(E)]^(-1)
    """
    solver = cp if isCuda else np
    mat = solver.array(E*S - F - g.sigmaTot(E))
    I = solver.eye(mat.shape[0])
    result = solver.linalg.solve(mat, I)
    return result.get() if isCuda else result


def DOSg(F, S, g, E):
    """
    Calculate density of states at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    float
        Density of states at energy E
    """
    return -np.trace(np.imag(Gr(F, S, g, E)))/np.pi


def eig(A, use_gpu=None):
    """
    Parallel eigenvalue decomposition for dense, non-Hermitian matrices.

    Parameters
    ----------
    A : ndarray
        Input matrix (can be non-Hermitian)
    use_gpu : bool, optional
        Force GPU usage (True) or CPU usage (False). If None, auto-detect.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues of A
    eigenvectors : ndarray
        Right eigenvectors of A

    Notes
    -----
    For large dense matrices, this function provides better performance than
    NumPy's eig() by utilizing GPU acceleration when available. For very large
    matrices (>5000x5000), consider using iterative methods instead.
    """
    if use_gpu is None:
        use_gpu = isCuda

    solver = cp if use_gpu else np
    A_device = solver.asarray(A)

    eigenvalues, eigenvectors = solver.linalg.eig(A_device)

    if use_gpu:
        # Return CPU arrays
        return eigenvalues.get(), eigenvectors.get()
    else:
        return eigenvalues, eigenvectors


def eigh(A, use_gpu=None):
    """
    Parallel eigenvalue decomposition for Hermitian matrices.

    Parameters
    ----------
    A : ndarray
        Input Hermitian matrix
    use_gpu : bool, optional
        Force GPU usage (True) or CPU usage (False). If None, auto-detect.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues of A (real, sorted in ascending order)
    eigenvectors : ndarray
        Eigenvectors of A

    Notes
    -----
    This function is optimized for Hermitian matrices and is generally faster
    and more numerically stable than eig() for such matrices.
    """
    if use_gpu is None:
        use_gpu = isCuda

    solver = cp if use_gpu else np
    A_device = solver.asarray(A)

    eigenvalues, eigenvectors = solver.linalg.eigh(A_device)

    if use_gpu:
        # Return CPU arrays
        return eigenvalues.get(), eigenvectors.get()
    else:
        return eigenvalues, eigenvectors

def matrix_power(S, power, use_gpu=None):
    """
    Calculate matrix power S^p using eigendecomposition with GPU acceleration.

    Parameters
    ----------
    S : ndarray
        Input matrix (should be Hermitian for numerical stability)
    power : float
        Power to raise matrix to (e.g., 0.5 for sqrt, -0.5 for inverse sqrt)
    use_gpu : bool, optional
        Force GPU usage (True) or CPU usage (False). If None, auto-detect.

    Returns
    -------
    ndarray
        Matrix power S^p

    Notes
    -----
    This function is optimized for Hermitian matrices (like overlap matrices)
    and uses eigendecomposition: S^p = V @ D^p @ V^H where S = V @ D @ V^H.
    For large matrices, GPU acceleration provides significant speedup.
    """
    if use_gpu is None:
        use_gpu = isCuda

    solver = cp if use_gpu else np
    S_device = solver.asarray(S)

    # Use eigh for Hermitian matrices (more stable and faster than eig)
    eigenvalues, eigenvectors = solver.linalg.eigh(S_device)

    # Handle numerical precision for near-zero eigenvalues
    eigenvalues = solver.maximum(eigenvalues, 1e-16)
    powered_eigenvalues = solver.power(eigenvalues, power)

    # Reconstruct matrix: S^p = V @ D^p @ V^H
    result = times(eigenvectors,solver.diag(powered_eigenvalues),eigenvectors.conj().T)

    return result.get() if use_gpu else result
