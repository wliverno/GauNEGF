"""
JAX-powered Green's Functions Integration

Supports both retarded (Gr) and lesser (G<) Green's functions.

Author: William Livernois
"""

import jax
import jax.numpy as jnp
import numpy as np
import threading
import os
import time
from jax import jit
from gauNEGF.parallelize import parallelize_energy_calculation, parallel_logger

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

# Jit G^R function: (g is static)
def _gr_matrix_ops(g, E, F, S):
    """Retarded Green's function matrix operations (used by both vmap and workers)."""
    mat = E * S - F - g.sigmaTot(E)
    return jnp.linalg.inv(mat)

# Jit G< function: (g, ind are static)
def _gless_matrix_ops(g, ind, E, F, S):
    """Lesser Green's function matrix operations (used by both vmap and workers)."""
    mat = E * S - F - g.sigmaTot(E)
    Gr_E = jnp.linalg.inv(mat)
    Ga_E = jnp.conj(Gr_E).T
    gamma_E = 1j * (g.sigma(E, ind) - jnp.conj(g.sigma(E, ind)).T)
    gless = Gr_E @ gamma_E @ Ga_E
    return gless



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
    
    point_func = jax.jit(_gr_matrix_ops, static_argnums=(0,))
    point = lambda E: point_func(g, E, F_jax, S_jax)

    if use_vmap:
        parallel_logger.debug(f"GrInt using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")
        start_time = time.time()
        weighted_func = lambda E, weight: weight * point(E)
        result = jax.vmap(weighted_func)(Elist_jax, weights_jax)
        integrated = jnp.sum(result, axis=0)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrInt vmap completed in {elapsed:.3f}s")
        return integrated

    else:
        parallel_logger.debug(f"GrInt using workers: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")
        start_time = time.time()

        # Use worker threads with on-demand sigma computation
        integrator = Integrator(F.shape)

        # Map energies to weights for worker function
        energy_weight_map = dict(zip(Elist, weights))
        worker_function = lambda E: integrator.accumulate(energy_weight_map[E], point(E))

        results = parallelize_energy_calculation(Elist, worker_function, matrix_size=matrix_size)

        assert np.sum(results) == len(Elist), "Integrator result shape does not match Elist shape"

        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrInt workers completed in {elapsed:.3f}s")
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

    point_func = jax.jit(_gless_matrix_ops, static_argnums=(0, 1,))
    point = lambda E: point_func(g, ind, E, F_jax, S_jax)

    if use_vmap:
        parallel_logger.debug(f"GrLessInt using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")
        start_time = time.time()
        weighted_func = lambda E, weight: weight * point(E)
        result = jax.vmap(weighted_func)(Elist_jax, weights_jax)
        integrated = jnp.sum(result, axis=0)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrLessInt vmap completed in {elapsed:.3f}s")
        return integrated

    else:
        parallel_logger.debug(f"GrLessInt using workers: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")
        start_time = time.time()

        # Use worker threads with on-demand sigma computation
        integrator = Integrator(F.shape)

        # Map energies to weights for worker function
        energy_weight_map = dict(zip(Elist, weights))
        worker_function = lambda E: integrator.accumulate(energy_weight_map[E], point(E))

        results = parallelize_energy_calculation(Elist, worker_function, matrix_size=matrix_size)

        assert np.sum(results) == len(Elist), "Integrator result shape does not match Elist shape"

        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrLessInt workers completed in {elapsed:.3f}s")
        return integrator.result




