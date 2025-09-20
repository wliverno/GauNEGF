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
import threading
import queue
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


# Device detection for multi-device computing
def detect_devices():
    """
    Detect available compute devices for parallel processing.

    Returns
    -------
    dict
        Dictionary with device information including GPU count and total workers
    """
    device_info = {
        'cpu_workers': 1,
        'gpu_workers': 0,
        'total_workers': 1,
        'gpu_devices': []
    }

    if isCuda:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            device_info['gpu_workers'] = gpu_count
            device_info['total_workers'] = gpu_count + 1  # +1 for CPU

            # Get individual GPU info
            for i in range(gpu_count):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    free_mem, total_mem = cp.cuda.Device(i).mem_info
                    device_info['gpu_devices'].append({
                        'id': i,
                        'name': props['name'].decode('utf-8'),
                        'free_memory_gb': free_mem / 1e9,
                        'total_memory_gb': total_mem / 1e9
                    })

            gpu_logger.info(f"Detected {gpu_count} GPU(s) + 1 CPU = {device_info['total_workers']} total workers")
            for gpu in device_info['gpu_devices']:
                gpu_logger.debug(f"GPU {gpu['id']}: {gpu['name']} ({gpu['free_memory_gb']:.1f}GB free)")

        except Exception as e:
            gpu_logger.warning(f"Error detecting GPUs: {e}, falling back to CPU-only")
            device_info['gpu_workers'] = 0
            device_info['total_workers'] = 1
    else:
        gpu_logger.info("CUDA not available, using CPU-only computation")

    return device_info


def worker_gr(work_queue, result_lock, shared_result, F, S, g, device_id, worker_id):
    """
    Memory-optimized unified worker function for processing Gr energy points on CPU or GPU.

    Parameters
    ----------
    work_queue : queue.Queue
        Queue containing (energy, weight, index) tuples
    result_lock : threading.Lock
        Lock for thread-safe result accumulation
    shared_result : dict
        Shared dictionary containing the accumulated result matrix
    F, S : ndarray
        Fock and overlap matrices
    g : surfG object
        Surface Green's function calculator
    device_id : int or None
        GPU device ID (None for CPU)
    worker_id : str
        Worker identification for logging
    """
    # Determine device and solver
    is_gpu = device_id is not None
    solver = cp if is_gpu else np
    device_name = f"GPU-{device_id}" if is_gpu else "CPU"

    try:
        if is_gpu:
            cp.cuda.Device(device_id).use()

        gpu_logger.debug(f"{device_name} worker {worker_id} started")
        local_count = 0
        N = F.shape[0]

        # Convert to device arrays once
        F_device = solver.array(F, dtype=solver.complex128)
        S_device = solver.array(S, dtype=solver.complex128)

        # Local accumulator to minimize GPU→CPU transfers
        local_result = solver.zeros((N, N), dtype=solver.complex128)

        # Pre-allocate workspace arrays for GPU memory efficiency
        if is_gpu:
            workspace_sigma = solver.zeros((N, N), dtype=solver.complex128)
            workspace_mat = solver.zeros((N, N), dtype=solver.complex128)

        while True:
            try:
                # Get work from queue with timeout
                E, weight, idx = work_queue.get(timeout=1.0)

                # Compute Green's function for this energy point
                if is_gpu:
                    # Get sigma from surface calculator (CPU) and transfer to GPU workspace
                    sigma_cpu = g.sigmaTot(E)
                    workspace_sigma[:] = cp.asarray(sigma_cpu, dtype=cp.complex128)
                    workspace_mat[:] = E * S_device - F_device - workspace_sigma
                    sigma_E = workspace_sigma
                    mat = workspace_mat
                else:
                    # For CPU, ensure we get proper numpy array
                    sigma_cpu = g.sigmaTot(E)
                    sigma_E = np.asarray(sigma_cpu, dtype=np.complex128)
                    mat = E * S_device - F_device - sigma_E

                try:
                    Gr_E = solver.linalg.inv(mat)
                except (np.linalg.LinAlgError, (cp.linalg.LinAlgError if is_gpu else type(None))):
                    gpu_logger.warning(f"{device_name} worker {worker_id}: Singular matrix at E={E:.6f} eV, using pseudoinverse")
                    Gr_E = solver.linalg.pinv(mat)

                # Accumulate locally (on device) to minimize transfers
                local_result += weight * Gr_E

                local_count += 1
                work_queue.task_done()

                # Clean up temporary arrays if not reused
                if not is_gpu:
                    del sigma_E, mat, Gr_E

            except queue.Empty:
                # No more work available
                break
            except Exception as e:
                gpu_logger.error(f"{device_name} worker {worker_id} error: {e}")
                work_queue.task_done()
                break

        # Single GPU→CPU transfer at the end (much more efficient)
        if local_count > 0:
            local_result_cpu = local_result.get() if is_gpu else local_result
            with result_lock:
                shared_result['matrix'] += local_result_cpu

        # Clean up memory
        if is_gpu:
            del F_device, S_device, local_result, workspace_sigma, workspace_mat
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU cleanup is handled by Python's garbage collector
            del F_device, S_device, local_result

        gpu_logger.debug(f"{device_name} worker {worker_id} completed {local_count} energy points")

    except Exception as e:
        gpu_logger.error(f"{device_name} worker {worker_id} failed to initialize: {e}")


def worker_grless(work_queue, result_lock, shared_result, F, S, g, ind, device_id, worker_id):
    """
    Memory-optimized unified worker function for processing GrLess energy points on CPU or GPU.

    Parameters
    ----------
    work_queue : queue.Queue
        Queue containing (energy, weight, index) tuples
    result_lock : threading.Lock
        Lock for thread-safe result accumulation
    shared_result : dict
        Shared dictionary containing the accumulated result matrix
    F, S : ndarray
        Fock and overlap matrices
    g : surfG object
        Surface Green's function calculator
    ind : int or None
        Contact index for partial density calculation
    device_id : int or None
        GPU device ID (None for CPU)
    worker_id : str
        Worker identification for logging
    """
    # Determine device and solver
    is_gpu = device_id is not None
    solver = cp if is_gpu else np
    device_name = f"GPU-{device_id}" if is_gpu else "CPU"

    try:
        if is_gpu:
            cp.cuda.Device(device_id).use()

        gpu_logger.debug(f"{device_name} worker {worker_id} started for GrLess")
        local_count = 0
        N = F.shape[0]

        # Convert to device arrays once
        F_device = solver.array(F, dtype=solver.complex128)
        S_device = solver.array(S, dtype=solver.complex128)

        # Local accumulator to minimize GPU→CPU transfers
        local_result = solver.zeros((N, N), dtype=solver.complex128)

        # Pre-allocate workspace arrays for GPU memory efficiency
        if is_gpu:
            workspace_sigma_tot = solver.zeros((N, N), dtype=solver.complex128)
            workspace_mat = solver.zeros((N, N), dtype=solver.complex128)
            workspace_sigma_E = solver.zeros((N, N), dtype=solver.complex128)
            workspace_gamma = solver.zeros((N, N), dtype=solver.complex128)
            workspace_temp = solver.zeros((N, N), dtype=solver.complex128)

        while True:
            try:
                # Get work from queue with timeout
                E, weight, idx = work_queue.get(timeout=1.0)

                # Calculate Gr
                if is_gpu:
                    # Get sigma from surface calculator (CPU) and transfer to GPU workspace
                    sigma_cpu = g.sigmaTot(E)
                    workspace_sigma_tot[:] = cp.asarray(sigma_cpu, dtype=cp.complex128)
                    workspace_mat[:] = E * S_device - F_device - workspace_sigma_tot
                    sigma_tot = workspace_sigma_tot
                    mat = workspace_mat
                else:
                    # For CPU, ensure we get proper numpy array
                    sigma_cpu = g.sigmaTot(E)
                    sigma_tot = np.asarray(sigma_cpu, dtype=np.complex128)
                    mat = E * S_device - F_device - sigma_tot

                try:
                    Gr_E = solver.linalg.inv(mat)
                except (np.linalg.LinAlgError, (cp.linalg.LinAlgError if is_gpu else type(None))):
                    gpu_logger.warning(f"{device_name} worker {worker_id}: Singular matrix at E={E:.6f} eV, using pseudoinverse")
                    Gr_E = solver.linalg.pinv(mat)

                # Calculate Ga = Gr†
                Ga_E = solver.conj(Gr_E).T

                # Calculate Gamma
                if ind is None:
                    Sigma_E = sigma_tot  # Reuse already computed sigmaTot
                else:
                    if is_gpu:
                        sigma_ind_cpu = g.sigma(E, ind)
                        workspace_sigma_E[:] = cp.asarray(sigma_ind_cpu, dtype=cp.complex128)
                        Sigma_E = workspace_sigma_E
                    else:
                        sigma_ind_cpu = g.sigma(E, ind)
                        Sigma_E = np.asarray(sigma_ind_cpu, dtype=np.complex128)

                if is_gpu:
                    workspace_gamma[:] = 1j * (Sigma_E - solver.conj(Sigma_E).T)
                    Gamma_E = workspace_gamma
                else:
                    Gamma_E = 1j * (Sigma_E - solver.conj(Sigma_E).T)

                # Calculate G< = Gr * Gamma * Ga
                if is_gpu:
                    # Use workspace for intermediate result
                    workspace_temp[:] = solver.matmul(Gr_E, Gamma_E)
                    Gless_E = solver.matmul(workspace_temp, Ga_E)
                else:
                    Gless_E = solver.matmul(solver.matmul(Gr_E, Gamma_E), Ga_E)

                # Accumulate locally (on device) to minimize transfers
                local_result += weight * Gless_E

                local_count += 1
                work_queue.task_done()

                # Clean up temporary arrays if not reused
                if not is_gpu:
                    del sigma_tot, mat, Gr_E, Ga_E, Sigma_E, Gamma_E, Gless_E

            except queue.Empty:
                # No more work available
                break
            except Exception as e:
                gpu_logger.error(f"{device_name} worker {worker_id} error: {e}")
                work_queue.task_done()
                break

        # Single GPU→CPU transfer at the end (much more efficient)
        if local_count > 0:
            local_result_cpu = local_result.get() if is_gpu else local_result
            with result_lock:
                shared_result['matrix'] += local_result_cpu

        # Clean up memory
        if is_gpu:
            del F_device, S_device, local_result
            del workspace_sigma_tot, workspace_mat, workspace_sigma_E, workspace_gamma, workspace_temp
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU cleanup is handled by Python's garbage collector
            del F_device, S_device, local_result

        gpu_logger.debug(f"{device_name} worker {worker_id} completed {local_count} energy points")

    except Exception as e:
        gpu_logger.error(f"{device_name} worker {worker_id} failed to initialize: {e}")


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
    Integrate retarded Green's function for a list of energies using multi-device parallelization.
    Distributes energy points across available CPUs and GPUs for parallel processing.

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

    # Detect available devices
    device_info = detect_devices()
    total_workers = device_info['total_workers']

    # Log calculation start
    memory_gb = N * N * 16 / 1e9
    gpu_logger.info(f"Starting GrInt: {N}x{N} matrices, {M} energies ({memory_gb:.2f}GB per matrix)")
    gpu_logger.info(f"Using {total_workers} workers: {device_info['gpu_workers']} GPU(s) + {device_info['cpu_workers']} CPU")

    # Create work queue with all energy points
    work_queue = queue.Queue()
    for i, (E, weight) in enumerate(zip(Elist, weights)):
        work_queue.put((E, weight, i))

    # Initialize shared result
    result_lock = threading.Lock()
    shared_result = {'matrix': np.zeros((N, N), dtype=np.complex128)}

    # Create and start worker threads
    threads = []

    # Start CPU worker
    cpu_thread = threading.Thread(
        target=worker_gr,
        args=(work_queue, result_lock, shared_result, F, S, g, None, "CPU-0")
    )
    threads.append(cpu_thread)
    cpu_thread.start()

    # Start GPU workers (if available)
    for gpu_id in range(device_info['gpu_workers']):
        gpu_thread = threading.Thread(
            target=worker_gr,
            args=(work_queue, result_lock, shared_result, F, S, g, gpu_id, f"GPU-{gpu_id}")
        )
        threads.append(gpu_thread)
        gpu_thread.start()

    # Progress monitoring
    def monitor_progress():
        last_remaining = M
        while any(t.is_alive() for t in threads):
            try:
                remaining = work_queue.qsize()
                if remaining != last_remaining:
                    completed = M - remaining
                    progress = 100.0 * completed / M
                    elapsed = time.perf_counter() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    gpu_logger.debug(f"Progress: {progress:.1f}% ({completed}/{M}) ({rate:.1f} energies/s)")
                    last_remaining = remaining
            except:
                pass
            time.sleep(1.0)

    # Start progress monitor
    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Wait for all workers to complete
    for thread in threads:
        thread.join()

    # Ensure all tasks are done
    work_queue.join()

    total_time = time.perf_counter() - start_time
    throughput = total_time / M
    gpu_logger.info(f"Completed GrInt: {total_time:.3f}s total ({throughput:.2e} sec/energy)")

    return shared_result['matrix']


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
    Integrate nonequilibrium Green's function for a list of energies using multi-device parallelization.
    Distributes energy points across available CPUs and GPUs for parallel processing.

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

    # Detect available devices
    device_info = detect_devices()
    total_workers = device_info['total_workers']

    # Log calculation start
    memory_gb = N * N * 16 / 1e9
    gpu_logger.info(f"Starting GrLessInt: {N}x{N} matrices, {M} energies ({memory_gb:.2f}GB per matrix)")
    gpu_logger.info(f"Using {total_workers} workers: {device_info['gpu_workers']} GPU(s) + {device_info['cpu_workers']} CPU")

    # Create work queue with all energy points
    work_queue = queue.Queue()
    for i, (E, weight) in enumerate(zip(Elist, weights)):
        work_queue.put((E, weight, i))

    # Initialize shared result
    result_lock = threading.Lock()
    shared_result = {'matrix': np.zeros((N, N), dtype=np.complex128)}

    # Create and start worker threads
    threads = []

    # Start CPU worker
    cpu_thread = threading.Thread(
        target=worker_grless,
        args=(work_queue, result_lock, shared_result, F, S, g, ind, None, "CPU-0")
    )
    threads.append(cpu_thread)
    cpu_thread.start()

    # Start GPU workers (if available)
    for gpu_id in range(device_info['gpu_workers']):
        gpu_thread = threading.Thread(
            target=worker_grless,
            args=(work_queue, result_lock, shared_result, F, S, g, ind, gpu_id, f"GPU-{gpu_id}")
        )
        threads.append(gpu_thread)
        gpu_thread.start()

    # Progress monitoring
    def monitor_progress():
        last_remaining = M
        while any(t.is_alive() for t in threads):
            try:
                remaining = work_queue.qsize()
                if remaining != last_remaining:
                    completed = M - remaining
                    progress = 100.0 * completed / M
                    elapsed = time.perf_counter() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    gpu_logger.debug(f"Progress: {progress:.1f}% ({completed}/{M}) ({rate:.1f} energies/s)")
                    last_remaining = remaining
            except:
                pass
            time.sleep(1.0)

    # Start progress monitor
    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Wait for all workers to complete
    for thread in threads:
        thread.join()

    # Ensure all tasks are done
    work_queue.join()

    total_time = time.perf_counter() - start_time
    throughput = total_time / M
    gpu_logger.info(f"Completed GrLessInt: {total_time:.3f}s total ({throughput:.2e} sec/energy)")

    return shared_result['matrix']



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

