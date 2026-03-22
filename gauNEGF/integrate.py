"""
JAX-powered Green's Functions Integration

Supports both retarded (Gr) and lesser (G<) Green's functions.

Author: William Livernois
"""

import numpy as np
import os
import time
import socket
import tempfile
import logging

# IMPORTANT: Import config BEFORE jax to set up JAX environment
from gauNEGF.config import LOG_LEVEL, LOG_PERFORMANCE, ETA, shard_array

import jax
import jax.numpy as jnp
from jax import jit

# Setup node-specific logging for integration operations
hostname = socket.gethostname()
pid = os.getpid()

if LOG_PERFORMANCE:
    log_file = f'integrate_performance_{hostname}_{pid}.log'
else:
    temp_dir = tempfile.gettempdir()
    log_file = os.path.join(temp_dir, f'integrate_performance_{hostname}_{pid}.log')

log_level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)

parallel_logger = logging.getLogger('gauNEGF.integrate')
parallel_logger.setLevel(log_level)

# Create file handler that appends (avoid duplicate handlers on reload)
if not parallel_logger.handlers:
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    parallel_logger.addHandler(handler)

parallel_logger.info("JAX integration framework initialized")
parallel_logger.debug(f"Number of devices: {len(jax.devices())}")
parallel_logger.debug(f"Device List: {jax.devices()}")

# =============================================================================
# INTEGRATION-SPECIFIC CONSTANTS
# =============================================================================

MAX_VMAP_MEMORY_GB = 1.0              # Use vmap if estimated memory < this (GB)
FORCE_SYNCHRONOUS = False             # Force synchronous operation (for accurate timing)

# Memory calculation constants
MEMORY_PER_MATRIX_FACTOR = 16         # Bytes per complex128 element
BYTES_TO_GB = 1e9                     # Conversion factor

# =============================================================================
# MODULE-LEVEL JIT FUNCTIONS (clean, no nesting)
# =============================================================================

# Jit G^R function: (g is static)
@jit
def _gr_matrix_ops(sigTot, E, F, S):
    """Retarded Green's function matrix operations (used by both vmap and workers)."""
    mat = (E + 1j*ETA) * S - F - sigTot
    return jnp.linalg.solve(mat, jnp.eye(F.shape[0]))

# Jit G< function: (g, ind are static)
@jit
def _gless_matrix_ops(sig, sigTot, E, F, S):
    """Lesser Green's function matrix operations (used by both vmap and workers)."""
    I = jnp.eye(F.shape[0])
    mat_r = (E + 1j*ETA) * S - F - sigTot
    mat_a = (E - 1j*ETA) * S - F - jnp.conj(sigTot).T
    Gr_E = jnp.linalg.solve(mat_r, I)
    Ga_E = jnp.linalg.solve(mat_a, I)
    gamma_E = 1j * (sig - jnp.conj(sig).T)
    gless = Gr_E @ gamma_E @ Ga_E
    return gless

def _GIntSeq(weighted_func, F, S, g, Elist, weights, ind=None):
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    start_time = time.time()

    # Convert to JAX arrays
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    matrix_size = F.shape[0]
    num_energies = len(Elist_jax)
    parallel_logger.info(f"GInt using sequencial (lax.scan): {matrix_size}x{matrix_size} matrix, {num_energies} energies")
    start_time = time.time()
    
    # Function for lax.scan
    def scan_fn(carry, inputs):
        E, w = inputs
        carry += weighted_func(E, w, F_jax, S_jax, g)
        return carry, 1

    # Integrate
    result = jnp.zeros_like(F_jax, dtype=complex)
    result, count = jax.lax.scan(scan_fn, result, (Elist_jax, weights_jax))
    total = np.sum(count)
    assert total == num_energies, f"Integration only used {total} points, expected {num_energies} points"

    # Time and return
    if FORCE_SYNCHRONOUS:
        jax.block_until_ready(result)
    elapsed = time.time() - start_time
    parallel_logger.debug(f"GInt seq completed in {elapsed:.3f}s")
    return result



def _GInt(weighted_func, F, S, g, Elist, weights, ind=None):
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

    if num_energies * matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.info(f"GInt using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {num_energies*matrix_size_gb:.2f}GB")
        # Shard energy points across devices for parallel computation
        Elist_sharded = shard_array(Elist_jax, axis=0)
        weights_sharded = shard_array(weights_jax, axis=0)
        result = jax.vmap(weighted_func, in_axes=(0, 0, None, None, None))(Elist_sharded, weights_sharded, F_jax, S_jax, g)
        integrated = jnp.sum(result, axis=0)
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(integrated)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GInt vmap completed in {elapsed:.3f}s")
        return integrated

    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB//matrix_size_gb))
        parallel_logger.info(f"GInt using batched mapping: {matrix_size}x{matrix_size} matrix, {num_energies} energies, Batch size: {batch_size} ({MAX_VMAP_MEMORY_GB:.2f}GB/batch)")
        start_time = time.time()
        def scan_fn(carry, inputs):
            E_batch, w_batch = inputs
            # Shard batch across devices for parallel computation
            E_batch_sharded = shard_array(E_batch, axis=0)
            w_batch_sharded = shard_array(w_batch, axis=0)
            result = jax.vmap(weighted_func, in_axes=(0, 0, None, None, None))(E_batch_sharded, w_batch_sharded, F_jax, S_jax, g)
            carry += jnp.sum(result, axis=0)
            count = jnp.ones(result.shape[0])
            return carry, count

        # Reshape into fixed-size batches
        n_batches = len(Elist) // batch_size
        Elist_batched = Elist_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        weights_batched = weights_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        Elist_tail = Elist_jax[n_batches*batch_size:]
        weights_tail = weights_jax[n_batches*batch_size:]

        # scan over batches (sequential), then vmap within each batch (parallel)
        result = jnp.zeros_like(F_jax, dtype=complex)
        result, count = jax.lax.scan(scan_fn, result, (Elist_batched, weights_batched))
        total = np.sum(count)
        if len(Elist_tail)>0:
            result, count2 = scan_fn(result, (Elist_tail, weights_tail))
            total += np.sum(count2)
        assert total == num_energies, f"Integration only used {total} points, expected {num_energies} points"
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(result)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GInt map completed in {elapsed:.3f}s")
        return result



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
    def weighted_func_Gr(E, weight, F_jax, S_jax, g):
        sigTot = g.sigmaTot(E)
        Gr = _gr_matrix_ops(sigTot, E, F_jax, S_jax)
        return weight * Gr
    parallel_logger.info(f"Calculating G^R with GInt...")
    return _GInt(weighted_func_Gr, F, S, g, Elist, weights)


def _GIntCross(F, S, g, Elist, weights):
    """Single-pass vmap integration of G^R matrix and cross-term scalar.

    Computes sigmaTot, G^R, and crossTermQ once per energy point, accumulating:
    - matrix: sum_k w_k * G^R(z_k)
    - scalar: sum_k w_k * Tr(G^R(z_k) @ Q_tot(z_k))

    Uses vmap over energy points. The cross-term accumulation is inlined with
    a zero-initialized Q_tot to avoid the None type transition in crossTermQTot
    that would break JAX tracing.
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"

    start_time = time.time()
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)
    matrix_size = F.shape[0]
    num_energies = len(Elist)
    num_contacts = g.num_contacts

    def weighted_combined(E, w, F_jax, S_jax, g):
        sigTot = g.sigmaTot(E)
        Gr = _gr_matrix_ops(sigTot, E, F_jax, S_jax)
        # Inline crossTermQTot with zero-init (vmappable, no None type change)
        Q_tot = jnp.zeros_like(F_jax, dtype=complex)
        for i in range(num_contacts):
            Q_i = g.crossTermQ(E, i)
            if Q_i is not None:  # static at trace time (stau is None check)
                Q_tot = Q_tot + Q_i
        return w * Gr, w * jnp.trace(Gr @ Q_tot)

    matrix_size_gb = (matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB

    if num_energies * matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.info(
            f"GIntCross using vmap: {matrix_size}x{matrix_size} matrix, "
            f"{num_energies} energies (single-pass)")
        Elist_sharded = shard_array(Elist_jax, axis=0)
        weights_sharded = shard_array(weights_jax, axis=0)
        matrices, scalars = jax.vmap(
            weighted_combined, in_axes=(0, 0, None, None, None)
        )(Elist_sharded, weights_sharded, F_jax, S_jax, g)
        matrix_sum = jnp.sum(matrices, axis=0)
        scalar_sum = jnp.sum(scalars)
    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB // matrix_size_gb))
        parallel_logger.info(
            f"GIntCross using batched: {matrix_size}x{matrix_size} matrix, "
            f"{num_energies} energies, batch={batch_size} (single-pass)")

        def scan_fn(carry, inputs):
            mat_acc, scl_acc = carry
            E_batch, w_batch = inputs
            E_batch_sharded = shard_array(E_batch, axis=0)
            w_batch_sharded = shard_array(w_batch, axis=0)
            mats, scls = jax.vmap(
                weighted_combined, in_axes=(0, 0, None, None, None)
            )(E_batch_sharded, w_batch_sharded, F_jax, S_jax, g)
            mat_acc = mat_acc + jnp.sum(mats, axis=0)
            scl_acc = scl_acc + jnp.sum(scls)
            return (mat_acc, scl_acc), None

        n_batches = num_energies // batch_size
        Elist_batched = Elist_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        weights_batched = weights_jax[:n_batches * batch_size].reshape(n_batches, batch_size)
        Elist_tail = Elist_jax[n_batches * batch_size:]
        weights_tail = weights_jax[n_batches * batch_size:]

        init = (jnp.zeros_like(F_jax, dtype=complex), 0.0 + 0j)
        (matrix_sum, scalar_sum), _ = jax.lax.scan(
            scan_fn, init, (Elist_batched, weights_batched))

        if len(Elist_tail) > 0:
            (matrix_sum, scalar_sum), _ = scan_fn(
                (matrix_sum, scalar_sum), (Elist_tail, weights_tail))

    if FORCE_SYNCHRONOUS:
        jax.block_until_ready(matrix_sum)
    elapsed = time.time() - start_time
    parallel_logger.debug(f"GIntCross completed in {elapsed:.3f}s")
    return matrix_sum, scalar_sum


def GrIntCross(F, S, g, Elist, weights):
    """Integrate G^R with co-accumulation of cross-term scalar.

    Returns (lineInt, cross_scalar) where:
    - lineInt = sum_k w_k * G^R(z_k)  (NxN matrix, same as GrInt)
    - cross_scalar = sum_k w_k * Tr(G^R(z_k) @ Q_tot(z_k))  (complex scalar)

    The cross-term delta_N = -(1/pi) * Im(cross_scalar).

    For orthogonal systems (crossTermQTot returns None), uses the fast
    vmap/scan path via GrInt with no cross-term computation. For non-orthogonal
    systems, uses a single-pass vmap that computes sigmaTot, G^R, and Q_tot
    once per energy point.
    """
    # Fast path: orthogonal system -- use vmap/scan GrInt, no cross-term
    Q_check = g.crossTermQTot(Elist[0]) if len(Elist) > 0 else None
    if Q_check is None:
        lineInt = GrInt(F, S, g, Elist, weights)
        return lineInt, 0.0 + 0j

    # Non-orthogonal: single-pass vmap (sigmaTot + crossTermQ once per point)
    parallel_logger.info("Calculating G^R + cross-term with single-pass GrIntCross...")
    return _GIntCross(F, S, g, Elist, weights)


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
    def weighted_func_GrLess(E, weight, F_jax, S_jax, g):
        useTot = (ind is None)
        sigTot = g.sigmaTot(E)
        sigma = sigTot if useTot else g.sigma(E, ind)
        Gless = _gless_matrix_ops(sigma, sigTot, E, F_jax, S_jax)
        return weight * Gless
    parallel_logger.info(f"Calculating G< with GInt...")
    return _GInt(weighted_func_GrLess, F, S, g, Elist, weights, ind)
