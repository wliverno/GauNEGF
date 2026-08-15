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
# Effective bytes per element per energy point. 16 is the size of ONE
# complex128 element; the kernels hold several live matrices per point
# (_gless_matrix_ops materializes I, mat_r, mat_a, Gr_E, Ga_E, gamma_E,
# gless, and jnp.linalg.solve takes its own workspace), so 16 undercounts
# the batch cost ~3x. MEASURED 2026-08-04 on GPU as the slope of peak
# device bytes vs n_points (cancels fixed pool/executable overhead):
# 2.93x across 4 configs, 1.4% spread -> 16*2.93 ~= 47.
MEMORY_PER_MATRIX_FACTOR = 47         # Effective bytes/element/point (measured)
BYTES_TO_GB = 1e9                     # Conversion factor

# =============================================================================
# PERSISTENT COMPILED-KERNEL CACHE (2026-07-07)
# =============================================================================
# The integrators previously rebuilt their vmap/scan closures on EVERY call,
# defeating jax's function-identity cache: each SCF cycle re-traced and
# re-compiled everything (leaking ~1.1k memory mappings/cycle until
# vm.max_map_count killed the job at ~cycle 45 on big-basis systems).
# Kernels close over the surfG object g, whose state can mutate (setF /
# updateFermi), so persistence needs explicit invalidation: mutating methods
# bump g._gauNEGF_version, and kernels are cached per
# (kind, id(g), version). Within one version, jax's own jit cache then
# handles shape specialization (the adaptive ladder's few lengths compile
# once each and are reused for the rest of the run).
_KERNEL_CACHE = {}

def _cached_kernel(kind, g, builder):
    """Return (and memoize) a compiled kernel for this g's current version."""
    key = (kind, id(g), getattr(g, '_gauNEGF_version', 0))
    fn = _KERNEL_CACHE.get(key)
    if fn is None:
        # evict kernels for stale versions of this same object
        for k in [k for k in _KERNEL_CACHE
                  if k[0] == kind and k[1] == id(g) and k[2] != key[2]]:
            del _KERNEL_CACHE[k]
        fn = builder()
        _KERNEL_CACHE[key] = fn
        parallel_logger.info(f"Kernel compiled+cached: {kind} v{key[2]}")
    return fn

def clear_kernel_cache():
    """Drop all cached kernels (emergency valve; see config
    CLEAR_JAX_CACHES_PER_CYCLE)."""
    _KERNEL_CACHE.clear()

def _current_dfermis(g):
    """Per-contact fermi shifts as a jnp array (traced kernel argument).
    Bethe-protocol objects expose gList[i].dFermi; others get a dummy
    zero vector (their sigmaTot ignores the argument via the *_dfs-less
    call path chosen at kernel build time)."""
    gl = getattr(g, 'gList', None)
    if gl is None:
        return jnp.zeros(1)
    return jnp.array([getattr(c, 'dFermi', 0.0) for c in gl], dtype=float)

def _sigma_tot_call(g):
    """Build-time choice: Bethe-protocol g takes traced dFermis; other
    protocols keep their (E,)-only signature (version-bump invalidation
    covers their mutations - see surfG1D.setF)."""
    if getattr(g, 'gList', None) is not None:
        return lambda E, dfs: g.sigmaTot(E, dFermis=dfs)
    return lambda E, dfs: g.sigmaTot(E)

# =============================================================================
# MODULE-LEVEL JIT FUNCTIONS (clean, no nesting)
# =============================================================================

@jit
def _gr_matrix_ops(sigTot, E, F, S, eta):
    """Retarded Green's function matrix operations (used by both vmap and workers)."""
    mat = (E + 1j*eta) * S - F - sigTot
    return jnp.linalg.solve(mat, jnp.eye(F.shape[0]))

@jit
def _gless_matrix_ops(sig, sigTot, E, F, S, eta):
    """Lesser Green's function matrix operations (used by both vmap and workers)."""
    I = jnp.eye(F.shape[0])
    mat_r = (E + 1j*eta) * S - F - sigTot
    mat_a = (E - 1j*eta) * S - F - jnp.conj(sigTot).T
    Gr_E = jnp.linalg.solve(mat_r, I)
    Ga_E = jnp.linalg.solve(mat_a, I)
    gamma_E = 1j * (sig - jnp.conj(sig).T)
    gless = Gr_E @ gamma_E @ Ga_E
    return gless


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

    # Cached jitted kernels: g is closed over (its version keys the cache);
    # E/w/F/S are runtime arguments so jax's jit cache specializes per shape
    # ONCE and reuses across every subsequent call/cycle.
    kind = getattr(weighted_func, '_kernel_kind', weighted_func.__name__)

    if num_energies * matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.info(f"GInt using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {num_energies*matrix_size_gb:.2f}GB")
        def _build_vmap():
            def kern(E, w, F_, S_, dfs):
                res = jax.vmap(weighted_func, in_axes=(0, 0, None, None, None, None))(E, w, F_, S_, dfs, g)
                return jnp.sum(res, axis=0)
            return jax.jit(kern)
        kernel = _cached_kernel(('vmap', kind), g, _build_vmap)
        Elist_sharded = shard_array(Elist_jax, axis=0)
        weights_sharded = shard_array(weights_jax, axis=0)
        integrated = kernel(Elist_sharded, weights_sharded, F_jax, S_jax,
                            _current_dfermis(g))
        if FORCE_SYNCHRONOUS:
            jax.block_until_ready(integrated)
        elapsed = time.time() - start_time
        parallel_logger.debug(f"GInt vmap completed in {elapsed:.3f}s")
        return integrated

    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB//matrix_size_gb))
        parallel_logger.info(f"GInt using batched mapping: {matrix_size}x{matrix_size} matrix, {num_energies} energies, Batch size: {batch_size} ({MAX_VMAP_MEMORY_GB:.2f}GB/batch)")
        start_time = time.time()


        # Pad to a whole number of batches (2026-07-07): a variable-length
        # tail call compiles a fresh XLA executable per distinct remainder,
        # feeding the vm.max_map_count leak. Padding with weight=0 points
        # (E = last energy, a known-safe evaluation point) is exact and
        # keeps ONE compiled shape family per batch geometry.
        n_batches = (len(Elist) + batch_size - 1) // batch_size
        n_pad = n_batches * batch_size - len(Elist)
        if n_pad > 0:
            Elist_jax = jnp.concatenate([Elist_jax, jnp.full(n_pad, Elist_jax[-1])])
            weights_jax = jnp.concatenate([weights_jax, jnp.zeros(n_pad, dtype=weights_jax.dtype)])
        Elist_batched = Elist_jax.reshape(n_batches, batch_size)
        weights_batched = weights_jax.reshape(n_batches, batch_size)

        # scan over batches (sequential), vmap within each batch; the whole
        # scan is one cached jitted kernel (specializes per (n_batches,
        # batch_size) shape once, then reused every call/cycle)
        def _build_scan():
            def kern(Eb, wb, F_, S_, dfs):
                def scan_fn(carry, inputs):
                    E_batch, w_batch = inputs
                    res_b = jax.vmap(weighted_func,
                                     in_axes=(0, 0, None, None, None, None))(
                        E_batch, w_batch, F_, S_, dfs, g)
                    carry += jnp.sum(res_b, axis=0)
                    return carry, jnp.ones(res_b.shape[0])
                init = jnp.zeros_like(F_, dtype=complex)
                res, count = jax.lax.scan(scan_fn, init, (Eb, wb))
                return res, jnp.sum(count)
            return jax.jit(kern)
        kernel = _cached_kernel(('scan', kind), g, _build_scan)
        result, total = kernel(Elist_batched, weights_batched, F_jax, S_jax,
                               _current_dfermis(g))
        total = float(total)
        assert total == num_energies + n_pad, f"Integration used {total} points, expected {num_energies}+{n_pad} padded"
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
    _stot = _sigma_tot_call(g)
    def weighted_func_Gr(E, weight, F_jax, S_jax, dfs, g):
        sigTot = _stot(E, dfs)
        eta = max(g.eta, ETA)
        Gr = _gr_matrix_ops(sigTot, E, F_jax, S_jax, eta)
        return weight * Gr
    weighted_func_Gr._kernel_kind = 'gr'
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

    _stot = _sigma_tot_call(g)
    threads_shifts = getattr(g, 'gList', None) is not None
    def weighted_combined(E, w, F_jax, S_jax, dfs, g):
        sigTot = _stot(E, dfs)
        eta = max(g.eta, ETA)
        Gr = _gr_matrix_ops(sigTot, E, F_jax, S_jax, eta)
        # Inline crossTermQTot with zero-init (vmappable, no None type change)
        Q_tot = jnp.zeros_like(F_jax, dtype=complex)
        for i in range(num_contacts):
            Q_i = g.crossTermQ(E, i, dFermi=dfs[i]) if threads_shifts else g.crossTermQ(E, i)
            if Q_i is not None:  # static at trace time (stau is None check)
                Q_tot = Q_tot + Q_i[2]
        return w * Gr, w * jnp.trace(Gr @ Q_tot)

    matrix_size_gb = (matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB

    if num_energies * matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.info(
            f"GIntCross using vmap: {matrix_size}x{matrix_size} matrix, "
            f"{num_energies} energies (single-pass)")
        def _build_cross_vmap():
            def kern(E, w, F_, S_, dfs):
                mats, scls = jax.vmap(
                    weighted_combined, in_axes=(0, 0, None, None, None, None)
                )(E, w, F_, S_, dfs, g)
                return jnp.sum(mats, axis=0), jnp.sum(scls)
            return jax.jit(kern)
        kernel = _cached_kernel(('cross_vmap', 'gr_cross'), g, _build_cross_vmap)
        Elist_sharded = shard_array(Elist_jax, axis=0)
        weights_sharded = shard_array(weights_jax, axis=0)
        matrix_sum, scalar_sum = kernel(Elist_sharded, weights_sharded,
                                        F_jax, S_jax, _current_dfermis(g))
    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB // matrix_size_gb))
        parallel_logger.info(
            f"GIntCross using batched: {matrix_size}x{matrix_size} matrix, "
            f"{num_energies} energies, batch={batch_size} (single-pass)")


        # Pad to whole batches (see _GInt note - one compiled shape family)
        n_batches = (num_energies + batch_size - 1) // batch_size
        n_pad = n_batches * batch_size - num_energies
        if n_pad > 0:
            Elist_jax = jnp.concatenate([Elist_jax, jnp.full(n_pad, Elist_jax[-1])])
            weights_jax = jnp.concatenate([weights_jax, jnp.zeros(n_pad, dtype=weights_jax.dtype)])
        Elist_batched = Elist_jax.reshape(n_batches, batch_size)
        weights_batched = weights_jax.reshape(n_batches, batch_size)

        def _build_cross_scan():
            def kern(Eb, wb, F_, S_, dfs):
                def scan_fn(carry, inputs):
                    mat_acc, scl_acc = carry
                    E_batch, w_batch = inputs
                    mats, scls = jax.vmap(
                        weighted_combined, in_axes=(0, 0, None, None, None, None)
                    )(E_batch, w_batch, F_, S_, dfs, g)
                    return (mat_acc + jnp.sum(mats, axis=0),
                            scl_acc + jnp.sum(scls)), None
                init = (jnp.zeros_like(F_, dtype=complex), 0.0 + 0j)
                (m, sc), _ = jax.lax.scan(scan_fn, init, (Eb, wb))
                return m, sc
            return jax.jit(kern)
        kernel = _cached_kernel(('cross_scan', 'gr_cross'), g, _build_cross_scan)
        matrix_sum, scalar_sum = kernel(Elist_batched, weights_batched,
                                        F_jax, S_jax, _current_dfermis(g))

    if FORCE_SYNCHRONOUS:
        jax.block_until_ready(matrix_sum)
    elapsed = time.time() - start_time
    parallel_logger.debug(f"GIntCross completed in {elapsed:.3f}s")
    return matrix_sum, scalar_sum


@jit
def _gless_cross_matrix_ops(sigma, sigTot, E, F_jax, S_jax, eta):
    """Gless = Gr Gamma_b Ga, computed alongside Gr from one solve."""
    I = jnp.eye(F_jax.shape[0], dtype=complex)
    mat_r = (E + 1j * eta) * S_jax - F_jax - sigTot
    Gr = jnp.linalg.solve(mat_r, I)
    gamma = 1j * (sigma - sigma.conj().T)
    Gless = Gr @ gamma @ Gr.conj().T
    return Gless, Gr


def _GLessIntCross(weighted_func, F, S, g, Elist, weights):
    """Structural copy of _GIntCross's vmap/scan/padding logic, taking the
    weighted (matrix, scalar) kernel as an argument so per-contact window
    kernels can share the same batching machinery."""
    assert Elist.size == weights.size, "Elist and weights must have the same length"

    start_time = time.time()
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)
    matrix_size = F.shape[0]
    num_energies = len(Elist)
    kind = getattr(weighted_func, '_kernel_kind', weighted_func.__name__)

    matrix_size_gb = (matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB

    if num_energies * matrix_size_gb < MAX_VMAP_MEMORY_GB:
        parallel_logger.info(
            f"GLessIntCross using vmap: {matrix_size}x{matrix_size} matrix, "
            f"{num_energies} energies (single-pass)")
        def _build_cross_vmap():
            def kern(E, w, F_, S_, dfs):
                mats, scls = jax.vmap(
                    weighted_func, in_axes=(0, 0, None, None, None, None)
                )(E, w, F_, S_, dfs, g)
                return jnp.sum(mats, axis=0), jnp.sum(scls)
            return jax.jit(kern)
        kernel = _cached_kernel(('cross_vmap', kind), g, _build_cross_vmap)
        Elist_sharded = shard_array(Elist_jax, axis=0)
        weights_sharded = shard_array(weights_jax, axis=0)
        matrix_sum, scalar_sum = kernel(Elist_sharded, weights_sharded,
                                        F_jax, S_jax, _current_dfermis(g))
    else:
        batch_size = max(1, int(MAX_VMAP_MEMORY_GB // matrix_size_gb))
        parallel_logger.info(
            f"GLessIntCross using batched: {matrix_size}x{matrix_size} matrix, "
            f"{num_energies} energies, batch={batch_size} (single-pass)")

        # Pad to whole batches (see _GInt note - one compiled shape family)
        n_batches = (num_energies + batch_size - 1) // batch_size
        n_pad = n_batches * batch_size - num_energies
        if n_pad > 0:
            Elist_jax = jnp.concatenate([Elist_jax, jnp.full(n_pad, Elist_jax[-1])])
            weights_jax = jnp.concatenate([weights_jax, jnp.zeros(n_pad, dtype=weights_jax.dtype)])
        Elist_batched = Elist_jax.reshape(n_batches, batch_size)
        weights_batched = weights_jax.reshape(n_batches, batch_size)

        def _build_cross_scan():
            def kern(Eb, wb, F_, S_, dfs):
                def scan_fn(carry, inputs):
                    mat_acc, scl_acc = carry
                    E_batch, w_batch = inputs
                    mats, scls = jax.vmap(
                        weighted_func, in_axes=(0, 0, None, None, None, None)
                    )(E_batch, w_batch, F_, S_, dfs, g)
                    return (mat_acc + jnp.sum(mats, axis=0),
                            scl_acc + jnp.sum(scls)), None
                init = (jnp.zeros_like(F_, dtype=complex), 0.0 + 0j)
                (m, sc), _ = jax.lax.scan(scan_fn, init, (Eb, wb))
                return m, sc
            return jax.jit(kern)
        kernel = _cached_kernel(('cross_scan', kind), g, _build_cross_scan)
        matrix_sum, scalar_sum = kernel(Elist_batched, weights_batched,
                                        F_jax, S_jax, _current_dfermis(g))

    if FORCE_SYNCHRONOUS:
        jax.block_until_ready(matrix_sum)
    elapsed = time.time() - start_time
    parallel_logger.debug(f"GLessIntCross completed in {elapsed:.3f}s")
    return matrix_sum, scalar_sum


def GrLessIntCross(F, S, g, Elist, weights, ind):
    """Window integrator: (sum w*Gless, sum w*(own_tail + device_tail)).

    Cross pieces of the per-contact kernel W_ind:
      own:  +Im Tr[Gr Q_fwd_ind] + Im Tr[Ga Q_rev_ind]
      tail: -(1/2) sum_a Tr[Gless (Q_rev_a + Q_rev_a^dag)]
    Occupation weights (f2-f1) arrive INSIDE `weights`; no fermi factors
    here. Per-contact only: ind=None is the Gamma_L+Gamma_R mispairing.
    """
    if ind is None:
        raise ValueError(
            "GrLessIntCross requires a specific contact index: the window "
            "kernel is per-contact (pairing rule); ind=None would build it "
            "from Gamma_tot.")
    ind = int(ind)
    if not (-g.num_contacts <= ind < g.num_contacts):
        raise ValueError(
            f"GrLessIntCross: ind={ind} out of range for "
            f"{g.num_contacts} contacts.")
    ind = ind % g.num_contacts   # normalize -1 -> last
    _stot = _sigma_tot_call(g)
    threads_shifts = getattr(g, 'gList', None) is not None
    num_contacts = g.num_contacts

    def weighted_gless_cross(E, w, F_jax, S_jax, dfs, g):
        sigTot = _stot(E, dfs)
        sig_b = g.sigma(E, ind, dFermi=dfs[ind]) if threads_shifts else g.sigma(E, ind)
        eta = max(g.eta, ETA)
        Gless, Gr = _gless_cross_matrix_ops(sig_b, sigTot, E, F_jax,
                                            S_jax, eta)
        Ga = Gr.conj().T
        own = jnp.asarray(0.0 + 0j)
        Q_b = g.crossTermQ(E, ind, dFermi=dfs[ind]) if threads_shifts else g.crossTermQ(E, ind)
        if Q_b is not None:               # static at trace time
            own = (jnp.trace(Gr @ Q_b[0]).imag
                   + jnp.trace(Ga @ Q_b[1]).imag) + 0j
        tail = jnp.asarray(0.0 + 0j)
        for a in range(num_contacts):
            Q_a = g.crossTermQ(E, a, dFermi=dfs[a]) if threads_shifts else g.crossTermQ(E, a)
            if Q_a is not None:           # static at trace time
                Qr = Q_a[1]
                tail = tail - 0.5 * jnp.trace(Gless @ (Qr + Qr.conj().T))
        return w * Gless, w * (own + tail)
    weighted_gless_cross._kernel_kind = f'gless_cross_{ind}'
    parallel_logger.info("Calculating G< + cross with GLessIntCross...")
    return _GLessIntCross(weighted_gless_cross, F, S, g, Elist, weights)


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
    _stot = _sigma_tot_call(g)
    # gList classes (Bethe) thread the shared dfs shift array; others keep
    # their own stored per-contact shifts, so dfs would just be zeros for them.
    threads_shifts = getattr(g, 'gList', None) is not None
    def weighted_func_GrLess(E, weight, F_jax, S_jax, dfs, g):
        useTot = (ind is None)
        sigTot = _stot(E, dfs)
        if useTot:
            sigma = sigTot
        elif threads_shifts:
            sigma = g.sigma(E, ind, dFermi=dfs[ind])
        else:
            sigma = g.sigma(E, ind)
        eta = max(g.eta, ETA)
        Gless = _gless_matrix_ops(sigma, sigTot, E, F_jax, S_jax, eta)
        return weight * Gless
    weighted_func_GrLess._kernel_kind = f'gless_{ind}'
    parallel_logger.info(f"Calculating G< with GInt...")
    return _GInt(weighted_func_GrLess, F, S, g, Elist, weights, ind)
