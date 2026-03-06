"""
Utility functions compiled with JIT for gauNEGF.

Contains commonly used pure mathematical functions that are reused
across multiple modules in the gauNEGF package.
"""

import jax.numpy as jnp
from jax import jit


@jit
def fractional_matrix_power(S, power):
    """
    Calculate matrix power S^p using eigendecomposition.
    Supports fractional powers including negative values like -0.5.

    Parameters
    ----------
    S : jax array
        Input matrix (should be Hermitian for numerical stability)
    power : float
        Power to raise matrix to (e.g., 0.5 for sqrt, -0.5 for inverse sqrt)

    Returns
    -------
    jax array
        Matrix power S^p

    Notes
    -----
    This function is optimized for Hermitian matrices (like overlap matrices)
    and uses eigendecomposition: S^p = V @ D^p @ V^H where S = V @ D @ V^H.

    Unlike JAX's matrix_power, this function properly handles fractional
    powers including negative values.
    """
    # Use eigh for Hermitian matrices (more stable and faster than eig)
    eigenvalues, eigenvectors = eigh(S)

    # Handle numerical precision for near-zero eigenvalues
    eigenvalues = jnp.maximum(eigenvalues, 1e-16)
    powered_eigenvalues = jnp.power(eigenvalues, power)

    # Reconstruct matrix: S^p = V @ D^p @ V^H
    result = eigenvectors @ jnp.diag(powered_eigenvalues) @ eigenvectors.conj().T

    return result

def regularizeOverlap(S_raw, tol_ratio=1e-10):
    """
    Make overlap positive definite via diagonal shift (Tikhonov regularization).

    Computes S_reg = S + eps * I where eps is just large enough to push
    the smallest eigenvalue above ``tol_ratio * max(abs(eigenvalues))``.
    If already PSD, the original matrix is returned unchanged.

    This preserves all off-diagonal elements exactly, keeping zeros as
    zeros and orbital-to-orbital overlaps intact.  Only the diagonal
    (orbital self-overlap) is modified.

    Parameters
    ----------
    S_raw : jax array, shape (N, N)
        Raw overlap matrix (Hermitian, possibly indefinite)
    tol_ratio : float
        Relative tolerance (fraction of largest absolute eigenvalue)

    Returns
    -------
    jax array, shape (N, N)
        Regularized overlap matrix (Hermitian, positive definite)

    References
    ----------
    Tikhonov, Dokl. Akad. Nauk SSSR 151, 501 (1963).
    Saunders & Hillier, Int. J. Quantum Chem. 7, 699 (1973).
    Gaussian IOp(3/18): overlap eigenvalue threshold for linear dependency.
    """
    eigvals = jnp.linalg.eigvalsh(S_raw)
    lambda_min = jnp.min(eigvals)
    max_abs_eig = jnp.max(jnp.abs(eigvals))
    tol = tol_ratio * max_abs_eig
    eps = jnp.maximum(tol - lambda_min, 0.0)
    return S_raw + eps * jnp.eye(S_raw.shape[0], dtype=S_raw.dtype)


# Simple numpy operations

@jit
def inv(A):
    return jnp.linalg.solve(A, jnp.eye(A.shape[0]))

@jit
def eig(A):
    return jnp.linalg.eig(A)

@jit
def eigh(A):
    return jnp.linalg.eigh(A)

