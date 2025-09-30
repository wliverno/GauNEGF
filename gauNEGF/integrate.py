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
import os
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from jax import jit
from gauNEGF.parallelize import parallelize_energy_calculation, parallel_logger

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# =============================================================================
# ADAPTIVE SIGMA PARALLELIZATION
# =============================================================================

# Minimum number of energies to justify multiprocessing overhead
MIN_ENERGIES_FOR_PARALLEL_SIGMA = 4

def check_blas_configured():
    """
    Check if BLAS threading is configured (any value, just needs to be set).

    Returns
    -------
    bool
        True if any BLAS threading variable is set
    """
    return (
        'OMP_NUM_THREADS' in os.environ or
        'MKL_NUM_THREADS' in os.environ or
        'OPENBLAS_NUM_THREADS' in os.environ
    )


def is_jit_compatible(surf_g_obj):
    """
    Detect if a surfG object is JAX/JIT compatible.

    Args:
        surf_g_obj: A surface Green's function object

    Returns:
        bool: True if JAX/JIT compatible, False if NumPy-based
    """
    # Check for explicit JIT flag
    if hasattr(surf_g_obj, 'isJIT') and surf_g_obj.isJIT:
        return True

    # Check if sigmaTot returns JAX arrays
    try:
        test_result = surf_g_obj.sigmaTot(0.0)
        if hasattr(test_result, '__array_namespace__') and 'jax' in str(type(test_result)):
            return True
        if str(type(test_result)).startswith("<class 'jax"):
            return True
    except:
        pass

    # Check if methods are JIT-compiled functions
    if hasattr(surf_g_obj, 'sigmaTot'):
        if hasattr(surf_g_obj.sigmaTot, '__wrapped__'):  # JIT wrapper signature
            return True

    return False


def _compute_sigma_worker(args):
    """
    Worker function for multiprocessing.Pool (must be module-level for pickling).

    Args:
        args: Tuple of (surf_g_obj, E)

    Returns:
        Sigma matrix at energy E
    """
    surf_g_obj, E = args
    return surf_g_obj.sigmaTot(E)


def parallelize_sigma_adaptive(surf_g_obj, energies, max_workers=None):
    """
    Three-tier adaptive parallelization strategy for sigma computation.

    Tier 1: JAX pmap - Multi-GPU JIT-compiled (fastest for GPU clusters)
    Tier 2: multiprocessing.Pool - NumPy parallel (21.91x speedup on HPC when BLAS configured)
    Tier 3: Sequential - Fallback when parallelization unavailable

    Args:
        surf_g_obj: Surface Green's function object (JAX or NumPy)
        energies: List/array of energy values
        max_workers: Max workers for multiprocessing.Pool (default: None = auto-detect)

    Returns:
        list: List of sigma matrices
    """
    num_energies = len(energies)

    # =========================================================================
    # TIER 1: JAX pmap (JIT-compiled + multi-GPU)
    # =========================================================================
    if is_jit_compatible(surf_g_obj):
        try:
            # JAX shard_map parallelization
            from jax.experimental.shard_map import shard_map
            from jax.sharding import PartitionSpec as P, Mesh

            num_devices = len(jax.devices())

            if num_devices > 1 and num_energies % num_devices == 0:
                parallel_logger.debug(f"Using JAX pmap: {num_devices} devices for {num_energies} energies")

                mesh = Mesh(jax.devices(), axis_names=('devices',))
                energies_per_device = num_energies // num_devices
                energies_array = jnp.array(energies).reshape(num_devices, energies_per_device)

                def compute_sigma_batch(energy_batch):
                    results = []
                    for i in range(energy_batch.shape[0]):
                        E = energy_batch[i]
                        sigma = surf_g_obj.sigmaTot(E)
                        results.append(sigma)
                    return jnp.stack(results)

                with mesh:
                    sigma_results = shard_map(
                        compute_sigma_batch,
                        mesh=mesh,
                        in_specs=P('devices'),
                        out_specs=P('devices', None, None)
                    )(energies_array)

                # Flatten results
                sigma_list = []
                for device_idx in range(num_devices):
                    for energy_idx in range(energies_per_device):
                        sigma_list.append(sigma_results[device_idx, energy_idx])

                return sigma_list
        except Exception as e:
            parallel_logger.debug(f"JAX pmap unavailable: {e}")
            # Fall through to Tier 2

    # =========================================================================
    # TIER 2: NumPy + multiprocessing.Pool (production-ready parallelization)
    # =========================================================================
    blas_configured = check_blas_configured()

    if blas_configured and num_energies >= MIN_ENERGIES_FOR_PARALLEL_SIGMA:
        # Determine optimal worker count
        if max_workers is None:
            max_workers = min(num_energies, multiprocessing.cpu_count())

        parallel_logger.debug(f"Using multiprocessing.Pool: {max_workers} workers for {num_energies} energies (BLAS configured)")

        # Prepare arguments for worker function (must be picklable)
        args_list = [(surf_g_obj, E) for E in energies]

        # Use multiprocessing.Pool (BLAS must be configured before Python starts)
        with Pool(processes=max_workers) as pool:
            sigma_list = pool.map(_compute_sigma_worker, args_list)

        return sigma_list

    # =========================================================================
    # TIER 3: Sequential fallback
    # =========================================================================
    parallel_logger.warning(
        f"Using sequential computation: BLAS threading not configured "
        f"(set OMP_NUM_THREADS or MKL_NUM_THREADS), {num_energies} energies"
    )

    sigma_list = [surf_g_obj.sigmaTot(E) for E in energies]
    return sigma_list

# =============================================================================
# INTEGRATION-SPECIFIC CONSTANTS
# =============================================================================

# Performance thresholds for vmap vs worker decision
SMALL_MATRIX_THRESHOLD = 500          # Use vmap for matrices smaller than this
MAX_VMAP_MEMORY_GB = 4.0              # Use vmap if estimated memory < this (GB)

# Memory calculation constants
MEMORY_PER_MATRIX_FACTOR = 16         # Bytes per complex128 element
BYTES_TO_GB = 1e9                     # Conversion factor

# =============================================================================
# JIT COMPILATION CACHES
# =============================================================================

# Module-level caches for compiled functions by matrix size
_compiled_gr_vmap = {}
_compiled_gless_vmap = {}
_compiled_gr_matrix_ops = {}
_compiled_gless_matrix_ops = {}


def _get_compiled_gr_vmap(matrix_size):
    """Get cached or compile new GR vmap function for given matrix size."""
    if matrix_size not in _compiled_gr_vmap:
        _compiled_gr_vmap[matrix_size] = jit(_compute_gr_vmap)
    return _compiled_gr_vmap[matrix_size]


def _get_compiled_gless_vmap(matrix_size):
    """Get cached or compile new G< vmap function for given matrix size."""
    if matrix_size not in _compiled_gless_vmap:
        _compiled_gless_vmap[matrix_size] = jit(_compute_gless_vmap)
    return _compiled_gless_vmap[matrix_size]


def _get_compiled_gr_matrix_ops(matrix_size):
    """Get cached or compile new GR matrix ops function for given matrix size."""
    if matrix_size not in _compiled_gr_matrix_ops:
        _compiled_gr_matrix_ops[matrix_size] = jit(_gr_matrix_ops)
    return _compiled_gr_matrix_ops[matrix_size]


def _get_compiled_gless_matrix_ops(matrix_size):
    """Get cached or compile new G< matrix ops function for given matrix size."""
    if matrix_size not in _compiled_gless_matrix_ops:
        _compiled_gless_matrix_ops[matrix_size] = jit(_gless_matrix_ops)
    return _compiled_gless_matrix_ops[matrix_size]


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
        parallel_logger.debug(f"Using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Pre-compute all sigma values using pmap parallelization
        num_devices = len(jax.devices())
        if num_devices > 1 and num_energies % num_devices == 0:
            parallel_logger.debug(f"Computing sigma with pmap using {num_devices} devices")
            # Reshape energies for pmap distribution
            energies_per_device = num_energies // num_devices
            energies_reshaped = np.array(Elist).reshape(num_devices, energies_per_device)

            # pmap function for sigma computation
            def compute_sigma_batch(energy_batch):
                return jnp.array([g.sigmaTot(float(E)) for E in energy_batch])

            pmapped_sigma_compute = jax.pmap(compute_sigma_batch)
            sigma_batches = pmapped_sigma_compute(energies_reshaped)

            # Flatten back to list
            sigma_list = [sigma_batches[i, j] for i in range(num_devices) for j in range(energies_per_device)]
        else:
            # Use adaptive parallelization for sigma computation
            sigma_list = parallelize_sigma_adaptive(g, Elist)

        sigma_jax = jnp.array(sigma_list)

        # Use regular vmap for matrix operations (already optimal)
        jit_vmap_func = _get_compiled_gr_vmap(matrix_size)
        result = jit_vmap_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
        return np.array(result)

    else:
        parallel_logger.debug(f"Using workers: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Use worker threads with on-demand sigma computation
        integrator = Integrator(F.shape)

        # Get cached or compile matrix ops function for this matrix size
        jit_matrix_ops = _get_compiled_gr_matrix_ops(matrix_size)

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
        parallel_logger.debug(f"Using vmap: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Pre-compute all sigma values using pmap parallelization
        num_devices = len(jax.devices())
        if num_devices > 1 and num_energies % num_devices == 0:
            parallel_logger.debug(f"Computing sigma with pmap using {num_devices} devices")
            # Reshape energies for pmap distribution
            energies_per_device = num_energies // num_devices
            energies_reshaped = np.array(Elist).reshape(num_devices, energies_per_device)

            # pmap function for sigma total computation
            def compute_sigma_total_batch(energy_batch):
                return jnp.array([g.sigmaTot(float(E)) for E in energy_batch])

            pmapped_sigma_compute = jax.pmap(compute_sigma_total_batch)
            sigma_batches = pmapped_sigma_compute(energies_reshaped)
            sigma_total_list = [sigma_batches[i, j] for i in range(num_devices) for j in range(energies_per_device)]

            if ind is None:
                sigma_contact_list = sigma_total_list  # Use total self-energy
            else:
                # pmap function for contact sigma computation
                def compute_sigma_contact_batch(energy_batch):
                    return jnp.array([g.sigma(float(E), ind) for E in energy_batch])

                pmapped_contact_compute = jax.pmap(compute_sigma_contact_batch)
                contact_batches = pmapped_contact_compute(energies_reshaped)
                sigma_contact_list = [contact_batches[i, j] for i in range(num_devices) for j in range(energies_per_device)]
        else:
            # Use adaptive parallelization for sigma computation
            sigma_total_list = parallelize_sigma_adaptive(g, Elist)
            if ind is None:
                sigma_contact_list = sigma_total_list  # Use total self-energy
            else:
                # Individual contact sigma still sequential (could be optimized later)
                sigma_contact_list = [g.sigma(E, ind) for E in Elist]

        sigma_total_jax = jnp.array(sigma_total_list)
        sigma_contact_jax = jnp.array(sigma_contact_list)

        # Use regular vmap for matrix operations (already optimal)
        jit_vmap_func = _get_compiled_gless_vmap(matrix_size)
        result = jit_vmap_func(Elist_jax, F_jax, S_jax, sigma_total_jax, sigma_contact_jax, weights_jax)
        return np.array(result)

    else:
        parallel_logger.debug(f"Using workers: {matrix_size}x{matrix_size} matrix, {num_energies} energies, {estimated_memory_gb:.2f}GB")

        # Use worker threads with on-demand sigma computation
        integrator = Integrator(F.shape)

        # Get cached or compile matrix ops function for this matrix size
        jit_matrix_ops = _get_compiled_gless_matrix_ops(matrix_size)

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



