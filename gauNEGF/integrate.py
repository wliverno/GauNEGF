"""
Memory-Efficient Vectorized Green's Functions

A standalone package for computing retarded and lesser Green's functions 
with automatic GPU memory management using CuPy when available.

Author: William Livernois
"""

try:
    import cupy as cp
    isCuda = cp.cuda.is_available()
    device = cp.cuda.Device()
    free_memory, total_memory = device.mem_info
    print(f"GPU Memory configured: {free_memory/1e9:.1f} GB free of {total_memory/1e9:.1f} GB total")
except:
    isCuda = False

import numpy as np
import logging
import socket
import os
import time

# Setup node-specific logging for GPU/parallel operations
hostname = socket.gethostname()
pid = os.getpid()
log_file = f'gpu_performance_{hostname}_{pid}.log'

gpu_logger = logging.getLogger('gauNEGF.gpu')
gpu_logger.setLevel(logging.DEBUG)

# Create file handler that appends
if not gpu_logger.handlers:  # Avoid duplicate handlers on reload
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    gpu_logger.addHandler(handler)

def get_optimal_chunk_size(N, M, calculation_type='Gr'):
    """
    Calculate optimal chunk size based on available memory using CuPy built-in APIs.

    Parameters
    ----------
    N : int
        Matrix dimension (NxN)
    M : int
        Number of energy points
    calculation_type : str
        Type of calculation: 'Gr' (peak 3 arrays) or 'GrLess' (peak 5 arrays)

    Returns
    -------
    int
        Optimal chunk size for energy points
    """

    if not isCuda:
        # For CPU, use conservative chunking to avoid memory issues
        chunk_size = max(1, M // 4)
        gpu_logger.debug(f"CPU mode: chunk size {chunk_size} for N={N}, M={M}")
        return chunk_size

    try:
        # Use CuPy memory pool to get accurate memory info
        mempool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        free_mem, total_mem = device.mem_info

        # Account for memory already in use by the pool
        used_by_pool = mempool.used_bytes()
        available_mem = free_mem - used_by_pool

        # Use actual peak memory counts from memory tracking analysis:
        # GrIntVectorized: peak 3 NxNxM arrays
        # GrLessVectorized: peak 5 NxNxM arrays
        peak_arrays = 3 if calculation_type == 'Gr' else 5
        bytes_per_array = N * N * 16  # complex128 = 16 bytes per element
        bytes_per_energy = peak_arrays * bytes_per_array

        # Use 90% of available memory for safety
        safe_mem = int(0.9 * available_mem)
        chunk_size = max(1, min(M, safe_mem // bytes_per_energy))

        gpu_logger.debug(f"Memory calc: N={N}, {calculation_type}, peak_arrays={peak_arrays}, "
                        f"available={available_mem/1e9:.2f}GB, chunk_size={chunk_size}")

        return chunk_size

    except Exception as e:
        gpu_logger.warning(f"Memory calculation failed: {e}, using conservative chunk size")
        return max(1, M // 4)

def Gr(F, S, g, E):
    """
    Calculate retarded Green's function at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    ndarray
        Retarded Green's function G(E) = [ES - F - Σ(E)]^(-1)
    """
    solver = cp if isCuda else np
    mat = solver.array(E*S - F - g.sigmaTot(E))
    result = solver.linalg.inv(mat)
    return result.get() if isCuda else result

def DOSg(F, S, g, E):
    """
    Calculate density of states at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    float
        Density of states at energy E
    """
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

def GrInt(F, S, g, Elist, weights):
    """
    Integrate retarded Green's function for a list of energies using vectorization.
    Uses proactive memory management with performance logging.

    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator with sigmaTot(E) method
    Elist : ndarray
        Array of energies in eV (Mx1) - kept as numpy for g.sigmaTot()
    weights : ndarray
        Array of weights for each energy (Mx1)

    Returns
    -------
    ndarray
        Retarded Green's function G(E) integrated over the energy grid (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    start_time = time.perf_counter()
    M = Elist.size
    N = F.shape[0]
    solver = cp if isCuda else np
    device_name = 'GPU' if isCuda else 'CPU'

    # Get optimal chunk size proactively
    chunk_size = get_optimal_chunk_size(N, M, 'Gr')

    # Log calculation start
    memory_mb = N * N * M * 16 / 1e6
    gpu_logger.info(f"Starting GrInt: Matrix {N}x{N}x{M} ({memory_mb:.1f}MB) | Device: {device_name} | Chunk: {chunk_size}")

    if isCuda:
        mempool = cp.get_default_memory_pool()
        initial_memory = mempool.used_bytes()
        peak_memory = initial_memory

    # Use chunking approach similar to existing GrIntChunked but with proactive sizing
    Gint = np.zeros((N, N), dtype=solver.complex128)

    i = 0
    chunk_count = 0
    while i < M:
        end_idx = min(i + chunk_size, M)
        actual_chunk_size = end_idx - i
        chunk_count += 1

        chunk_start = time.perf_counter()

        try:
            Elist_chunk = Elist[i:end_idx]
            weights_chunk = weights[i:end_idx]

            # Use existing GrIntVectorized function - no math changes
            chunk_result = GrIntVectorized(F, S, g, Elist_chunk, weights_chunk, solver)
            Gint += chunk_result

            chunk_time = time.perf_counter() - chunk_start
            throughput = actual_chunk_size / chunk_time
            current_memory = mempool.used_bytes()
            peak_memory = max(peak_memory, current_memory)

            gpu_logger.debug(f"Chunk {chunk_count}: {actual_chunk_size} energies in {chunk_time:.3f}s ({throughput:.2E} energies/s)")

            i = end_idx  # Success - move to next chunk

        except (MemoryError, (cp.cuda.memory.OutOfMemoryError if isCuda else Exception)) as e:
            gpu_logger.warning(f"Memory error in chunk {chunk_count}: {e}")

            if isCuda:
                cp.get_default_memory_pool().free_all_blocks()

            old_chunk_size = chunk_size
            chunk_size = max(1, chunk_size // 2)
            gpu_logger.info(f"Reducing chunk size: {old_chunk_size} -> {chunk_size}")
            # Don't increment i - retry this chunk

    total_time = time.perf_counter() - start_time

    # Final logging
    if isCuda:
        actual_peak = peak_memory - initial_memory
        gpu_logger.info(f"Completed GrInt: Total {total_time:.3f}s | Chunks: {chunk_count} | Peak memory: {actual_peak/1e6:.1f}MB")
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()  # Ensure cleanup completes
    else:
        gpu_logger.info(f"Completed GrInt: Total {total_time:.3f}s | Chunks: {chunk_count}")

    return Gint


def GrIntVectorized(F, S, g, Elist, weights, solver):
    """
    Original vectorized implementation - preserved exactly.

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
    solver : module
        Either numpy or cupy module for array operations

    Returns
    -------
    ndarray
        Retarded Green's function integrated over the energy grid (NxN)
    """
    M = Elist.size
    N = F.shape[0]

    # Convert array types to match solver:
    Elist_ = solver.array(Elist, dtype=solver.complex128)
    weights = solver.array(weights)
    S = solver.array(S, dtype=solver.complex128)
    F = solver.array(F, dtype=solver.complex128)

    # Memory tracking: 0 N×N×M arrays allocated

    #Generate vectorized matrices conserving memory
    ES_minus_F_minus_Sig = Elist_[:, None, None] * solver.tile(solver.array(S), (M, 1, 1))
    # Memory tracking: 1 N×N×M array (ES_minus_F_minus_Sig)

    ES_minus_F_minus_Sig -= solver.tile(solver.array(F), (M, 1, 1))
    ES_minus_F_minus_Sig -= solver.array([g.sigmaTot(E) for E in Elist])
    # Memory tracking: Still 1 N×N×M array (ES_minus_F_minus_Sig, temp tiles destroyed)

    Gr_vec = solver.linalg.solve(ES_minus_F_minus_Sig, solver.tile(solver.eye(N), (M, 1, 1)))
    # Memory tracking: PEAK 3 N×N×M arrays (ES_minus_F_minus_Sig + Gr_vec + temp eye tile)

    del ES_minus_F_minus_Sig
    # Memory tracking: 1-2 N×N×M arrays (Gr_vec + potential linalg temp)

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gr_vec*weights[:, None, None], axis=0)
    # Memory tracking: 1-2 N×N×M arrays (temp from multiplication)

    del Gr_vec
    # Memory tracking: 0 N×N×M arrays

    return Gint.get() if isCuda else Gint


def GrLessInt(F, S, g, Elist, weights, ind=None):
    """
    Integrate nonequilibrium Green's function for a list of energies using vectorization.
    Uses proactive memory management with performance logging.

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
        Nonequilibrium Green's function G<(E) integrated over the energy grid (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    start_time = time.perf_counter()
    M = Elist.size
    N = F.shape[0]
    solver = cp if isCuda else np
    device_name = 'GPU' if isCuda else 'CPU'

    # Get optimal chunk size proactively
    chunk_size = get_optimal_chunk_size(N, M, 'GrLess')

    # Log calculation start
    memory_mb = N * N * M * 16 / 1e6
    gpu_logger.info(f"Starting GrLessInt: Matrix {N}x{N}x{M} ({memory_mb:.1f}MB) | Device: {device_name} | Chunk: {chunk_size}")

    if isCuda:
        mempool = cp.get_default_memory_pool()
        initial_memory = mempool.used_bytes()
        peak_memory=initial_memory

    # Use chunking approach similar to existing GrLessChunked but with proactive sizing
    Gint = np.zeros((N, N), dtype=solver.complex128)

    i = 0
    chunk_count = 0
    while i < M:
        end_idx = min(i + chunk_size, M)
        actual_chunk_size = end_idx - i
        chunk_count += 1

        chunk_start = time.perf_counter()

        try:
            Elist_chunk = Elist[i:end_idx]
            weights_chunk = weights[i:end_idx]

            # Use existing GrLessVectorized function - no math changes
            chunk_result = GrLessVectorized(F, S, g, Elist_chunk, weights_chunk, solver, ind)
            Gint += chunk_result

            chunk_time = time.perf_counter() - chunk_start
            throughput = actual_chunk_size / chunk_time
            current_memory = mempool.used_bytes()
            peak_memory = max(peak_memory, current_memory)

            gpu_logger.debug(f"Chunk {chunk_count}: {actual_chunk_size} energies in {chunk_time:.3f}s ({throughput:.2e} energies/s)")

            i = end_idx  # Success - move to next chunk

        except (MemoryError, (cp.cuda.memory.OutOfMemoryError if isCuda else Exception)) as e:
            gpu_logger.warning(f"Memory error in chunk {chunk_count}: {e}")

            if isCuda:
                cp.get_default_memory_pool().free_all_blocks()

            old_chunk_size = chunk_size
            chunk_size = max(1, chunk_size // 2)
            gpu_logger.info(f"Reducing chunk size: {old_chunk_size} -> {chunk_size}")
            # Don't increment i - retry this chunk

    total_time = time.perf_counter() - start_time

    # Final logging
    if isCuda:
        actual_peak = peak_memory - initial_memory
        gpu_logger.info(f"Completed GlessInt: Total {total_time:.3f}s | Chunks: {chunk_count} | Peak memory: {actual_peak/1e6:.1f}MB")
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()  # Ensure cleanup completes
    else:
        gpu_logger.info(f"Completed GrLessInt: Total {total_time:.3f}s | Chunks: {chunk_count}")

    return Gint



def GrLessVectorized(F, S, g, Elist, weights, solver, ind):
    """
    Full Vectorized G< implementation.

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
    solver : module
        Either numpy or cupy module for array operations
    ind : int or None
        Contact index for partial density calculation

    Returns
    -------
    ndarray
        Nonequilibrium Green's function G<(E) integrated over the energy grid (NxN)
    """
    M = Elist.size
    N = F.shape[0]

    # Convert array types to match solver:
    Elist_ = solver.array(Elist, dtype=solver.complex128)
    weights = solver.array(weights)
    S = solver.array(S, dtype=solver.complex128)
    F = solver.array(F, dtype=solver.complex128)

    # Memory tracking: 0 N×N×M arrays allocated

    #Generate Gr and Ga vectorized matrices conserving memory
    ES_minus_F_minus_Sig = Elist_[:, None, None] * solver.tile(S, (M, 1, 1))
    # Memory tracking: 1 N×N×M array (ES_minus_F_minus_Sig)

    ES_minus_F_minus_Sig -= solver.tile(F, (M, 1, 1))
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])
    # Memory tracking: 2 N×N×M arrays (ES_minus_F_minus_Sig + SigmaTot)

    ES_minus_F_minus_Sig -= SigmaTot
    # Memory tracking: Still 2 N×N×M arrays

    Gr_vec = solver.linalg.solve(ES_minus_F_minus_Sig, solver.tile(solver.eye(N), (M, 1, 1)))
    # Memory tracking: PEAK 4 N×N×M arrays (ES_minus_F_minus_Sig + SigmaTot + Gr_vec + temp eye)

    del ES_minus_F_minus_Sig
    # Memory tracking: 2-3 N×N×M arrays (SigmaTot + Gr_vec + potential linalg temp)

    Ga_vec = solver.conj(Gr_vec).transpose(0, 2, 1)
    # Memory tracking: 3-4 N×N×M arrays (SigmaTot + Gr_vec + Ga_vec + potential temp)

    # Calculate Gamma:
    if ind is None:
        SigList = SigmaTot  # Memory tracking: No new array, just reference
    else:
        del SigmaTot
        # Memory tracking: Reduces by 1 N×N×M array
        SigList = solver.array([g.sigma(E, ind) for E in Elist])
        # Memory tracking: Back to same count with new SigList

    GammaList = 1j * (SigList - solver.conj(SigList).transpose(0, 2, 1))
    # Memory tracking: PEAK 5 N×N×M arrays (SigList + Gr_vec + Ga_vec + GammaList + temp from subtraction)

    del SigList
    # Memory tracking: 3-4 N×N×M arrays (Gr_vec + Ga_vec + GammaList + potential temp)

    # Calculate Gless:
    Gless_vec = solver.matmul(solver.matmul(Gr_vec, GammaList), Ga_vec)
    # Memory tracking: 4-5 N×N×M arrays during matmul chain

    del Gr_vec, Ga_vec, GammaList
    # Memory tracking: 1 N×N×M array (Gless_vec)

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gless_vec*weights[:, None, None], axis=0)
    del Gless_vec
    # Memory tracking: 0 N×N×M arrays

    return Gint.get() if isCuda else Gint



