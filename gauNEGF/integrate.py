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
FORCE_SYNCHRONOUS = True              # Force synchronous operation (for accurate
                                      # timing, reduced speed)

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

    def accumulate(self, matrix):
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
            self.result += matrix
        return 1

# =============================================================================
# MODULE-LEVEL JIT FUNCTIONS (clean, no nesting)
# =============================================================================

# Jit G^R function: (g is static)
@jit
def _gr_matrix_ops(sigTot, E, F, S):
    """Retarded Green's function matrix operations (used by both vmap and workers)."""
    mat = E * S - F - sigTot
    return jnp.linalg.inv(mat)

# Jit G< function: (g, ind are static)
@jit
def _gless_matrix_ops(sig, sigTot, E, F, S):
    """Lesser Green's function matrix operations (used by both vmap and workers)."""
    mat = E * S - F - sigTot
    Gr_E = jnp.linalg.inv(mat)
    Ga_E = jnp.conj(Gr_E).T
    gamma_E = 1j * (sig - jnp.conj(sig).T)
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
        
    start_time = time.time()

    # Convert to JAX arrays
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Decision logic: vmap for small matrices, workers for large matrices
    matrix_size = F.shape[0]
    num_energies = len(Elist)
    matrix_size_gb = (matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB
    
    def weighted_func(E, weight):
        sigTot = g.sigmaTot(E)
        Gr = _gr_matrix_ops(sigTot, E, F_jax, S_jax)
        return weight*Gr

    if num_energies*matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.debug(f"GrInt using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {num_energies*matrix_size_gb:.2f}GB")
        result = jax.vmap(weighted_func)(Elist_jax, weights_jax).block_until_ready()
        integrated = jnp.sum(result, axis=0)
        del result # Memory leak
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(integrated)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrInt vmap completed in {elapsed:.3f}s")
        return integrated

    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB//matrix_size_gb))
        parallel_logger.debug(f"GrInt using batched mapping: {matrix_size}x{matrix_size} matrix, {num_energies} energies, Batch size: {batch_size} ({MAX_VMAP_MEMORY_GB:.2f}GB/batch)")
        start_time = time.time()
        def scan_fn(carry, inputs):
            E_batch, w_batch = inputs
            result = jax.vmap(weighted_func)(E_batch, w_batch)
            carry += jnp.sum(result, axis=0)
            count = jnp.ones(result.shape[0])
            del result # Memory leak
            return carry, count
    
        # Reshape into fixed-size batches
        n_batches = len(Elist) // batch_size
        Elist_batched = Elist_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        weights_batched = weights_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        Elist_tail = Elist_jax[n_batches*batch_size:]
        weights_tail = weights_jax[n_batches*batch_size:]
    
        # scan over batches (sequential), then vmap within each batch (parallel)
        result= jnp.zeros_like(F_jax, dtype=complex)
        result, count = jax.lax.scan(scan_fn, result, (Elist_batched, weights_batched))
        total = np.sum(count)
        if len(Elist_tail)>0:
            result, count2 = scan_fn(result, (Elist_tail, weights_tail))
            total += np.sum(count2)
        assert total == num_energies, "Integration only used {total} points, expected {num_energies} points"
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(result)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrInt map completed in {elapsed:.3f}s")
        return result



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

    start_time = time.time()
    
    # Convert to JAX arrays
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Decision logic: vmap for small matrices, workers for large matrices
    matrix_size = F.shape[0]
    num_energies = len(Elist)
    matrix_size_gb = (matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB 
    def weighted_func(E, weight):
        sigTot = g.sigmaTot(E)
        sigma = jax.lax.cond(ind==None, lambda E: sigTot, lambda E: g.sigma(E, ind), E)
        Gless = _gless_matrix_ops(sigma, sigTot, E, F_jax, S_jax)
        return weight*Gless

    if num_energies*matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.debug(f"GrLessInt using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {num_energies*matrix_size_gb:.2f}GB")
        start_time = time.time()
        result = jax.vmap(weighted_func)(Elist_jax, weights_jax)
        integrated = jnp.sum(result, axis=0)
        del result # Memory leak
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(integrated)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrLessInt vmap completed in {elapsed:.3f}s")
        return integrated
    
    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB//matrix_size_gb))
        parallel_logger.debug(f"GrLessInt using batched mapping: {matrix_size}x{matrix_size} matrix, {num_energies} energies, Batch size: {batch_size} ({MAX_VMAP_MEMORY_GB:.2f}GB/batch)")
        start_time = time.time()
        def scan_fn(carry, inputs):
            E_batch, w_batch = inputs
            result = jax.vmap(weighted_func)(E_batch, w_batch)
            carry += jnp.sum(result, axis=0)
            count = jnp.ones(result.shape[0])
            del result # Memory leak
            return carry, count
    
        # Reshape into fixed-size batches
        n_batches = len(Elist) // batch_size
        Elist_batched = Elist_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        weights_batched = weights_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        Elist_tail = Elist_jax[n_batches*batch_size:]
        weights_tail = weights_jax[n_batches*batch_size:]
    
        # scan over batches (sequential), then vmap within each batch (parallel)
        result= jnp.zeros_like(F_jax, dtype=complex)
        result, count = jax.lax.scan(scan_fn, result, (Elist_batched, weights_batched))
        total = np.sum(count)
        if len(Elist_tail)>0:
            result, count2 = scan_fn(result, (Elist_tail, weights_tail))
            total += np.sum(count2)
        assert total == num_energies, "Integration only used {total} points, expected {num_energies} points"
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(result)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GrLess map completed in {elapsed:.3f}s")
        return result


