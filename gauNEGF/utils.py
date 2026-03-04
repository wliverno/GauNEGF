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

@jit
def correct_hs(H_raw, S_raw, tol_ratio=1e-10):
    """
    Project H_raw and S_raw onto the subspace where S_raw is positive definite.
    
    Parameters:
    -----------
    H_raw : np.ndarray, shape (N, N)
        Raw Hamiltonian matrix at a k-point (Hermitian)
    S_raw : np.ndarray, shape (N, N)
        Raw overlap matrix at a k-point (Hermitian, possibly indefinite)
    tol_ratio : float
        Relative tolerance for eigenvalue selection (fraction of max eigenvalue)
    
    Returns:
    --------
    H_red : np.ndarray, shape (r, r)
        Projected Hamiltonian in the subspace (non-orthogonal basis)
    S_red : np.ndarray, shape (r, r)
        Projected overlap = diagonal matrix of kept eigenvalues
    U_plus : np.ndarray, shape (N, r)
        Basis vectors of the subspace (orthonormal)
    """
    # Step 2: diagonalize S_raw
    eigvals, eigvecs = jnp.linalg.eigh(S_raw)
    
    # Step 3: select positive eigenvalues above tolerance
    max_eig = jnp.max(eigvals)
    tol = tol_ratio * max_eig
    keep_mask = eigvals > tol
    any_kept = jnp.any(keep_mask)
    eigvals_keep = jnp.where(keep_mask, eigvals, 0)
    U_plus = eigvecs * keep_mask[jnp.newaxis, :]
    
    # Step 4: project H and S
    H_red = U_plus.conj().T @ H_raw @ U_plus
    S_red = jnp.diag(eigvals_keep)  # because U_plus^T S_raw U_plus = diag(eigvals_keep)
   
    # If nothing was kept, return original matrices (silent failure)
    H_out = jnp.where(any_kept, H_red, H_raw)
    S_out = jnp.where(any_kept, S_red, S_raw)
    return H_out, S_out

def fixHSList(Flist, Slist=None, default='none'):
    """
    Convert F,S lists to jnp arrays with appropriate None handling.

    Parameters
    ----------
    Flist : list of arrays
        Hamiltonian/Fock matrices
    Slist : list of arrays/None, or None
        Overlap matrices
    default : str
        How to handle None entries in Slist:
        'none' = preserve None (for stau sentinel in sigma)
        'identity' = replace with identity matrix
        'zeros' = replace with zeros_like
    """
    Flist = [jnp.array(f) for f in Flist]
    if Slist is None:
        Slist = [None] * len(Flist)
    Sout = []
    for f, s in zip(Flist, Slist):
        if s is None:
            if default == 'identity':
                Sout.append(jnp.eye(len(f)))
            elif default == 'zeros':
                Sout.append(jnp.zeros_like(f))
            else:
                Sout.append(None)
        else:
            Sout.append(jnp.array(s))
    return Flist, Sout

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

