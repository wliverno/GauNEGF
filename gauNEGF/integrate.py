"""
JAX-powered Green's Functions Integration

Clean, modular integration functions using the generic parallelize.py framework.
Supports both retarded (Gr) and lesser (G<) Green's functions.

Author: William Livernois
"""

import jax
import jax.numpy as jnp
import numpy as np
import threading
from jax import jit
from gauNEGF.parallelize import parallelize_energy_calculation

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# =============================================================================
# INTEGRATION-SPECIFIC CONSTANTS
# =============================================================================

# Performance thresholds for vmap vs worker decision
SMALL_MATRIX_THRESHOLD = 500          # Use vmap for matrices smaller than this
MAX_VMAP_MEMORY_GB = 4.0              # Use vmap if estimated memory < this (GB)

# Memory calculation constants
MEMORY_PER_MATRIX_FACTOR = 16         # Bytes per complex128 element
BYTES_TO_GB = 1e9                     # Conversion factor


class Integrator:
    """
    Thread-safe accumulator for weighted matrix integration.

    Provides memory-efficient integration by accumulating results
    as they are computed, avoiding storage of intermediate matrices.
    """

    def __init__(self, shape):
        """
        Initialize integrator with zero matrix.

        Parameters
        ----------
        shape : tuple
            Shape of matrices to accumulate (typically (N, N))
        """
        self.result = np.zeros(shape, dtype=complex)
        self.lock = threading.Lock()

    def accumulate(self, weight, matrix):
        """
        Thread-safe accumulation of weighted matrix.

        Parameters
        ----------
        weight : float or complex
            Integration weight for this matrix
        matrix : ndarray
            Matrix to add to accumulated result

        Returns
        -------
        int
            Success indicator (1) - keeps memory footprint minimal
        """
        with self.lock:
            self.result += weight * matrix
        return 1





# =============================================================================
# MODULE-LEVEL JIT FUNCTIONS (clean, no nesting)
# =============================================================================

def _gr_matrix_ops(E, F, S, sigma_total, weight):
    """Retarded Green's function matrix operations (used by both vmap and workers)."""
    mat = E * S - F - sigma_total
    return weight * jnp.linalg.inv(mat)


def _gless_matrix_ops(E, F, S, sigma_total, sigma_contact, weight):
    """Lesser Green's function matrix operations (used by both vmap and workers)."""
    mat = E * S - F - sigma_total
    Gr_E = jnp.linalg.inv(mat)
    Ga_E = jnp.conj(Gr_E).T
    gamma_E = 1j * (sigma_contact - jnp.conj(sigma_contact).T)
    gless = Gr_E @ gamma_E @ Ga_E
    return weight * gless


def _compute_gr_vmap(energies, F, S, sigma_list, weights):
    """Vectorized retarded Green's function computation."""
    # vmap over energies, sigma_list, weights (axis 0), but not F, S (broadcast)
    weighted_grs = jax.vmap(_gr_matrix_ops, in_axes=(0, None, None, 0, 0))(energies, F, S, sigma_list, weights)
    return jnp.sum(weighted_grs, axis=0)


def _compute_gless_vmap(energies, F, S, sigma_total_list, sigma_contact_list, weights):
    """Vectorized lesser Green's function computation."""
    # vmap over energies, sigma lists, weights (axis 0), but not F, S (broadcast)
    weighted_gless = jax.vmap(_gless_matrix_ops, in_axes=(0, None, None, 0, 0, 0))(energies, F, S, sigma_total_list, sigma_contact_list, weights)
    return jnp.sum(weighted_gless, axis=0)







def GrInt(F, S, g, Elist, weights):
    """
    Integrate retarded Green's function over energy using JAX parallelization.

    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator with sigmaTot(E) method
    Elist : ndarray
        Array of energies in eV (Mx1)
    weights : ndarray
        Array of weights for each energy (Mx1)

    Returns
    -------
    ndarray
        Integrated retarded Green's function (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    # Convert to JAX arrays
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Decision logic: vmap for small matrices, workers for large matrices
    matrix_size = F.shape[0]
    num_energies = len(Elist)
    estimated_memory_gb = (num_energies * matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB
    use_vmap = (matrix_size < SMALL_MATRIX_THRESHOLD and estimated_memory_gb < MAX_VMAP_MEMORY_GB)

    if use_vmap:
        print(f"Using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Pre-compute all sigma values for vmap
        sigma_list = []
        for E in Elist:
            sigma_total = g.sigmaTot(E)
            sigma_list.append(sigma_total)

        sigma_jax = jnp.array(sigma_list)

        # Apply JIT compilation with known matrix shapes, then use vmap
        jit_vmap_func = jit(_compute_gr_vmap)
        result = jit_vmap_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
        return np.array(result)

    else:
        print(f"Using workers: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Use worker threads with on-demand sigma computation
        integrator = Integrator(F.shape)

        # Apply JIT compilation to matrix operations with known shapes
        jit_matrix_ops = jit(_gr_matrix_ops)

        def worker_function(E):
            """Simple worker function - no nesting."""
            # Find energy index and weight
            distances = np.abs(Elist - E)
            idx = np.argmin(distances)
            weight = weights[idx]

            # Get sigma values
            sigma_total = jnp.array(g.sigmaTot(E))

            weighted_result = jit_matrix_ops(E, F_jax, S_jax, sigma_total, weight)
            return integrator.accumulate(1.0, weighted_result)

        # Use parallelize framework
        results = parallelize_energy_calculation(Elist, worker_function, matrix_size=matrix_size)
        return integrator.result




def GrLessInt(F, S, g, Elist, weights, ind=None):
    """
    Integrate lesser Green's function over energy using JAX parallelization.

    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator
    Elist : ndarray
        Array of energies in eV (Mx1)
    weights : ndarray
        Array of weights for each energy (Mx1)
    ind : int, optional
        Contact index for partial density calculation (default: None)

    Returns
    -------
    ndarray
        Integrated lesser Green's function (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    # Convert to JAX arrays
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Decision logic: vmap for small matrices, workers for large matrices
    matrix_size = F.shape[0]
    num_energies = len(Elist)
    estimated_memory_gb = (num_energies * matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB
    use_vmap = (matrix_size < SMALL_MATRIX_THRESHOLD and estimated_memory_gb < MAX_VMAP_MEMORY_GB)

    if use_vmap:
        print(f"Using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Pre-compute all sigma values for vmap
        sigma_total_list = []
        sigma_contact_list = []

        for E in Elist:
            sigma_total = g.sigmaTot(E)
            sigma_total_list.append(sigma_total)

            if ind is None:
                sigma_contact = sigma_total  # Use total self-energy
            else:
                sigma_contact = g.sigma(E, ind)  # Use contact-specific self-energy
            sigma_contact_list.append(sigma_contact)

        sigma_total_jax = jnp.array(sigma_total_list)
        sigma_contact_jax = jnp.array(sigma_contact_list)

        # Apply JIT compilation with known matrix shapes, then use vmap
        jit_vmap_func = jit(_compute_gless_vmap)
        result = jit_vmap_func(Elist_jax, F_jax, S_jax, sigma_total_jax, sigma_contact_jax, weights_jax)
        return np.array(result)

    else:
        print(f"Using workers: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Use worker threads with on-demand sigma computation
        integrator = Integrator(F.shape)

        # Apply JIT compilation to matrix operations with known shapes
        jit_matrix_ops = jit(_gless_matrix_ops)

        def worker_function(E):
            """Simple worker function - no nesting."""
            # Find energy index and weight
            distances = np.abs(Elist - E)
            idx = np.argmin(distances)
            weight = weights[idx]

            # Get sigma values
            sigma_total = jnp.array(g.sigmaTot(E))
            if ind is None:
                sigma_contact = sigma_total
            else:
                sigma_contact = jnp.array(g.sigma(E, ind))

            weighted_result = jit_matrix_ops(E, F_jax, S_jax, sigma_total, sigma_contact, weight)
            return integrator.accumulate(1.0, weighted_result)

        # Use parallelize framework
        results = parallelize_energy_calculation(Elist, worker_function, matrix_size=matrix_size)
        return integrator.result



