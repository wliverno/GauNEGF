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

# Simple numpy operations

@jit
def inv(A):
    """
    Compute matrix inverse using JAX linalg solve.

    Solves the linear system A @ X = I for X, which is equivalent to
    computing the inverse A^(-1). This method is more numerically stable
    than direct inversion for ill-conditioned matrices.

    Parameters
    ----------
    A : jax array
        Square invertible matrix (NxN).

    Returns
    -------
    jax array
        Inverse matrix A^(-1) (NxN).
    """
    return jnp.linalg.solve(A, jnp.eye(A.shape[0]))

@jit
def eig(A):
    """
    Compute eigenvalues and eigenvectors of a square matrix.

    Wrapper around JAX's general eigenvalue decomposition for non-Hermitian
    matrices. For Hermitian matrices, prefer eigh() which uses a more stable
    algorithm.

    Parameters
    ----------
    A : jax array
        Square matrix (NxN), may be complex or non-Hermitian.

    Returns
    -------
    eigenvalues : jax array of shape (N,)
        Eigenvalues of A (may be complex).
    eigenvectors : jax array of shape (NxN)
        Eigenvectors as columns, with A @ eigenvectors = eigenvectors @ diag(eigenvalues).
    """
    return jnp.linalg.eig(A)

@jit
def eigh(A):
    """
    Compute eigenvalues and eigenvectors of a Hermitian matrix.

    Wrapper around JAX's Hermitian eigenvalue decomposition. Assumes the
    input matrix is Hermitian (A = A^H) and uses a more stable algorithm
    than eig().

    Parameters
    ----------
    A : jax array
        Hermitian matrix (NxN).

    Returns
    -------
    eigenvalues : jax array of shape (N,)
        Real-valued eigenvalues of A in ascending order.
    eigenvectors : jax array of shape (NxN)
        Eigenvectors as columns, with A @ eigenvectors = eigenvectors @ diag(eigenvalues).
    """
    return jnp.linalg.eigh(A)

