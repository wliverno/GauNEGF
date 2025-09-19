"""
Simple Loop-Based Green's Functions

A standalone package for computing retarded and lesser Green's functions
with automatic GPU acceleration using CuPy when available.

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
from gauNEGF.config import LOG_LEVEL, LOG_PERFORMANCE

# Setup node-specific logging for GPU/parallel operations
hostname = socket.gethostname()
pid = os.getpid()

if LOG_PERFORMANCE:
    log_file = f'integrate_performance_{hostname}_{pid}.log'
else:
    log_file = f'/tmp/integrate_performance_{hostname}_{pid}.log'

gpu_logger = logging.getLogger('gauNEGF.gpu')
log_level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)
gpu_logger.setLevel(log_level)

# Create file handler that appends
if not gpu_logger.handlers:  # Avoid duplicate handlers on reload
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    gpu_logger.addHandler(handler)


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
    Integrate retarded Green's function for a list of energies using simple loops.
    Relies on CuPy/NumPy built-in parallelization for matrix operations.

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

    # Log calculation start
    memory_gb = N * N * 16 / 1e9  # Memory per matrix in GB
    gpu_logger.info(f"Starting GrInt: {N}x{N} matrices, {M} energies ({memory_gb:.2f}GB per matrix) | Device: {device_name}")

    # Convert to solver arrays once
    F_solver = solver.array(F, dtype=solver.complex128)
    S_solver = solver.array(S, dtype=solver.complex128)
    Gint = solver.zeros((N, N), dtype=solver.complex128)

    # Simple loop over energy points
    for i, (E, weight) in enumerate(zip(Elist, weights)):
        sigma_E = solver.array(g.sigmaTot(E), dtype=solver.complex128)
        mat = E * S_solver - F_solver - sigma_E
        try:
            Gr_E = solver.linalg.inv(mat)
        except (np.linalg.LinAlgError, (cp.linalg.LinAlgError if isCuda else Exception)):
            gpu_logger.warning(f"Singular matrix at energy {E:.6f} eV, using pseudoinverse")
            Gr_E = solver.linalg.pinv(mat)
        Gint += weight * Gr_E

        if i % max(1, M // 10) == 0:  # Log progress at 10% intervals
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            progress = 100 * (i + 1) / M
            gpu_logger.debug(f"Progress: {progress:.1f}% ({i+1}/{M}) ({rate:.1f} energies/s)")

    total_time = time.perf_counter() - start_time
    throughput = total_time / M
    gpu_logger.info(f"Completed GrInt: {total_time:.3f}s total ({throughput:.2e} sec/energy)")

    if isCuda:
        cp.get_default_memory_pool().free_all_blocks()

    return Gint.get() if isCuda else Gint


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
    Integrate nonequilibrium Green's function for a list of energies using simple loops.
    Relies on CuPy/NumPy built-in parallelization for matrix operations.

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

    # Log calculation start
    memory_gb = N * N * 16 / 1e9  # Memory per matrix in GB
    gpu_logger.info(f"Starting GrLessInt: {N}x{N} matrices, {M} energies ({memory_gb:.2f}GB per matrix) | Device: {device_name}")

    # Convert to solver arrays once
    F_solver = solver.array(F, dtype=solver.complex128)
    S_solver = solver.array(S, dtype=solver.complex128)
    Gint = solver.zeros((N, N), dtype=solver.complex128)

    # Simple loop over energy points
    for i, (E, weight) in enumerate(zip(Elist, weights)):
        # Calculate Gr
        sigma_tot = solver.array(g.sigmaTot(E), dtype=solver.complex128)
        mat = E * S_solver - F_solver - sigma_tot
        try:
            Gr_E = solver.linalg.inv(mat)
        except (np.linalg.LinAlgError, (cp.linalg.LinAlgError if isCuda else Exception)):
            gpu_logger.warning(f"Singular matrix at energy {E:.6f} eV, using pseudoinverse")
            Gr_E = solver.linalg.pinv(mat)

        # Calculate Ga = Gr†
        Ga_E = solver.conj(Gr_E).T

        # Calculate Gamma
        if ind is None:
            Sigma_E = sigma_tot  # Reuse already computed sigmaTot
        else:
            Sigma_E = solver.array(g.sigma(E, ind), dtype=solver.complex128)

        Gamma_E = 1j * (Sigma_E - solver.conj(Sigma_E).T)

        # Calculate G< = Gr * Gamma * Ga
        Gless_E = solver.matmul(solver.matmul(Gr_E, Gamma_E), Ga_E)
        Gint += weight * Gless_E

        if i % max(1, M // 10) == 0:  # Log progress at 10% intervals
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            progress = 100 * (i + 1) / M
            gpu_logger.debug(f"Progress: {progress:.1f}% ({i+1}/{M}) ({rate:.1f} energies/s)")

    total_time = time.perf_counter() - start_time
    throughput = total_time/M
    gpu_logger.info(f"Completed GrLessInt: {total_time:.3f}s total ({throughput:.2e} sec/energy)")

    if isCuda:
        cp.get_default_memory_pool().free_all_blocks()

    return Gint.get() if isCuda else Gint



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



