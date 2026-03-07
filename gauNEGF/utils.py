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
    return jnp.linalg.solve(A, jnp.eye(A.shape[0]))

@jit
def eig(A):
    return jnp.linalg.eig(A)

@jit
def eigh(A):
    return jnp.linalg.eigh(A)

