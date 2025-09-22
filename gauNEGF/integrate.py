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
except ImportError:
    isCuda = False

import numpy as np
import logging
import socket
import os
import time
import threading
import queue
import multiprocessing
from gauNEGF.config import LOG_LEVEL, LOG_PERFORMANCE, USE_FLOAT32

# Precision configuration based on benchmark results
if USE_FLOAT32:
    COMPUTE_DTYPE = np.complex64
else:
    COMPUTE_DTYPE = np.complex128

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

# Log precision configuration after logger is set up
if USE_FLOAT32:
    gpu_logger.info("Using float32 precision for 19-47x GPU speedup")
else:
    gpu_logger.warning("Using float64 precision - expect 19-47x GPU performance penalty")


# Device detection for multi-device computing
def detect_devices():
    """
    Detect available compute devices for parallel processing.

    Returns
    -------
    dict
        Dictionary with device information including GPU count and device details
    """
    device_info = {
        'gpu_workers': 0,
        'gpu_devices': []
    }

    if isCuda:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            device_info['gpu_workers'] = gpu_count

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

            gpu_logger.info(f"Detected {gpu_count} GPU(s)")
            for gpu in device_info['gpu_devices']:
                gpu_logger.debug(f"GPU {gpu['id']}: {gpu['name']} ({gpu['free_memory_gb']:.1f}GB free)")

        except Exception as e:
            gpu_logger.warning(f"Error detecting GPUs: {e}, falling back to CPU-only")
            device_info['gpu_workers'] = 0
    else:
        gpu_logger.info("CUDA not available, using CPU-only computation")

    return device_info


def worker_gr(work_queue, result_lock, shared_result, F, S, g, device_id, worker_id, progress_counter=None):
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

        # Convert to device arrays once with configurable precision
        if is_gpu and isCuda:
            device_dtype = cp.complex64 if USE_FLOAT32 else cp.complex128
            gpu_logger.debug(f"{device_name} worker {worker_id} using dtype: {device_dtype} (CuPy), is_gpu={is_gpu}, isCuda={isCuda}")
        else:
            device_dtype = COMPUTE_DTYPE
            gpu_logger.debug(f"{device_name} worker {worker_id} using dtype: {device_dtype} (NumPy), is_gpu={is_gpu}, isCuda={isCuda}")
        F_device = solver.array(F, dtype=device_dtype)
        S_device = solver.array(S, dtype=device_dtype)

        # Local accumulator to minimize GPU-to-CPU transfers
        local_result = solver.zeros((N, N), dtype=device_dtype)

        # Pre-allocate workspace arrays for GPU memory efficiency
        if is_gpu:
            workspace_sigma = solver.zeros((N, N), dtype=device_dtype)
            workspace_mat = solver.zeros((N, N), dtype=device_dtype)

        while True:
            try:
                # Get work from queue with timeout
                E, weight, idx = work_queue.get(timeout=1.0)

                # Compute Green's function for this energy point
                if is_gpu:
                    # Get sigma from surface calculator (CPU) and transfer to GPU workspace
                    sigma_cpu = g.sigmaTot(E)
                    workspace_sigma[:] = cp.asarray(sigma_cpu, dtype=device_dtype)
                    workspace_mat[:] = E * S_device - F_device - workspace_sigma
                    sigma_E = workspace_sigma
                    mat = workspace_mat
                else:
                    # For CPU, ensure we get proper numpy array
                    sigma_cpu = g.sigmaTot(E)
                    sigma_E = np.asarray(sigma_cpu, dtype=COMPUTE_DTYPE)
                    mat = E * S_device - F_device - sigma_E

                try:
                    # Use solve() for better performance than inv()
                    solve_start = time.perf_counter()
                    I = solver.eye(mat.shape[0], dtype=mat.dtype)
                    Gr_E = solver.linalg.solve(mat, I)
                    solve_time = time.perf_counter() - solve_start
                    gpu_logger.debug(f"{device_name} worker {worker_id} matrix solve: {solve_time:.4f}s for E={E:.6f}")
                except (solver.linalg.LinAlgError, np.linalg.LinAlgError):
                    gpu_logger.warning(f"{device_name} worker {worker_id}: Singular matrix at E={E:.6f} eV, using pseudoinverse")
                    Gr_E = solver.linalg.pinv(mat)

                # Accumulate locally (on device) to minimize transfers
                local_result += weight * Gr_E

                local_count += 1
                work_queue.task_done()

                # Update progress counter
                if progress_counter is not None:
                    with result_lock:
                        progress_counter['completed'] += 1

                # Clean up temporary arrays if not reused
                if not is_gpu:
                    del sigma_cpu, sigma_E, mat, I, Gr_E

            except queue.Empty:
                # No more work available
                break
            except Exception as e:
                gpu_logger.error(f"{device_name} worker {worker_id} error: {e}")
                work_queue.task_done()
                break

        # Single GPU-to-CPU transfer at the end (much more efficient)
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



def worker_grless(work_queue, result_lock, shared_result, F, S, g, ind, device_id, worker_id, progress_counter=None):
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

        # Convert to device arrays once with configurable precision
        if is_gpu and isCuda:
            device_dtype = cp.complex64 if USE_FLOAT32 else cp.complex128
            gpu_logger.debug(f"{device_name} worker {worker_id} using dtype: {device_dtype} (CuPy), is_gpu={is_gpu}, isCuda={isCuda}")
        else:
            device_dtype = COMPUTE_DTYPE
            gpu_logger.debug(f"{device_name} worker {worker_id} using dtype: {device_dtype} (NumPy), is_gpu={is_gpu}, isCuda={isCuda}")
        F_device = solver.array(F, dtype=device_dtype)
        S_device = solver.array(S, dtype=device_dtype)

        # Local accumulator to minimize GPU-to-CPU transfers
        local_result = solver.zeros((N, N), dtype=device_dtype)

        # Pre-allocate workspace arrays for GPU memory efficiency
        if is_gpu:
            workspace_sigma_tot = solver.zeros((N, N), dtype=device_dtype)
            workspace_mat = solver.zeros((N, N), dtype=device_dtype)
            workspace_sigma_E = solver.zeros((N, N), dtype=device_dtype)
            workspace_gamma = solver.zeros((N, N), dtype=device_dtype)
            workspace_temp = solver.zeros((N, N), dtype=device_dtype)

        while True:
            try:
                # Get work from queue with timeout
                E, weight, idx = work_queue.get(timeout=1.0)

                # Calculate Gr
                if is_gpu:
                    # Get sigma from surface calculator (CPU) and transfer to GPU workspace
                    sigma_cpu = g.sigmaTot(E)
                    workspace_sigma_tot[:] = cp.asarray(sigma_cpu, dtype=device_dtype)
                    workspace_mat[:] = E * S_device - F_device - workspace_sigma_tot
                    sigma_tot = workspace_sigma_tot
                    mat = workspace_mat
                else:
                    # For CPU, ensure we get proper numpy array
                    sigma_cpu = g.sigmaTot(E)
                    sigma_tot = np.asarray(sigma_cpu, dtype=COMPUTE_DTYPE)
                    mat = E * S_device - F_device - sigma_tot

                try:
                    # Use solve() for better performance than inv()
                    solve_start = time.perf_counter()
                    I = solver.eye(mat.shape[0], dtype=mat.dtype)
                    Gr_E = solver.linalg.solve(mat, I)
                    solve_time = time.perf_counter() - solve_start
                    gpu_logger.debug(f"{device_name} worker {worker_id} matrix solve: {solve_time:.4f}s for E={E:.6f}")
                except (solver.linalg.LinAlgError, np.linalg.LinAlgError):
                    gpu_logger.warning(f"{device_name} worker {worker_id}: Singular matrix at E={E:.6f} eV, using pseudoinverse")
                    Gr_E = solver.linalg.pinv(mat)

                # Calculate Ga = Gr†
                Ga_E = solver.conj(Gr_E).T

                # Calculate Gamma
                if ind is None:
                    sigma_E = sigma_tot  # Reuse already computed sigmaTot
                else:
                    if is_gpu:
                        sigma_ind_cpu = g.sigma(E, ind)
                        workspace_sigma_E[:] = cp.asarray(sigma_ind_cpu, dtype=device_dtype)
                        sigma_E = workspace_sigma_E
                    else:
                        sigma_ind_cpu = g.sigma(E, ind)
                        sigma_E = np.asarray(sigma_ind_cpu, dtype=COMPUTE_DTYPE)

                if is_gpu:
                    workspace_gamma[:] = 1j * (sigma_E - solver.conj(sigma_E).T)
                    gamma_E = workspace_gamma
                else:
                    gamma_E = 1j * (sigma_E - solver.conj(sigma_E).T)

                # Calculate G< = Gr * Gamma * Ga
                if is_gpu:
                    # Use workspace for intermediate result
                    workspace_temp[:] = solver.matmul(Gr_E, gamma_E)
                    Gless_E = solver.matmul(workspace_temp, Ga_E)
                else:
                    Gless_E = solver.matmul(solver.matmul(Gr_E, gamma_E), Ga_E)

                # Accumulate locally (on device) to minimize transfers
                local_result += weight * Gless_E

                local_count += 1
                work_queue.task_done()

                # Update progress counter
                if progress_counter is not None:
                    with result_lock:
                        progress_counter['completed'] += 1

                # Clean up temporary arrays if not reused
                if not is_gpu:
                    del sigma_tot, mat, Gr_E, Ga_E, sigma_E, gamma_E, Gless_E

            except queue.Empty:
                # No more work available
                break
            except Exception as e:
                gpu_logger.error(f"{device_name} worker {worker_id} error: {e}")
                work_queue.task_done()
                break

        # Single GPU-to-CPU transfer at the end (much more efficient)
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
    if isCuda:
        dtype = cp.complex64 if USE_FLOAT32 else cp.complex128
    else:
        dtype = COMPUTE_DTYPE
    mat = solver.array(E*S - F - g.sigmaTot(E), dtype=dtype)
    I = solver.eye(mat.shape[0], dtype=dtype)
    result = solver.linalg.solve(mat, I)
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

def times(A, B, C=None, use_gpu=None):
    """
    Optimized matrix multiplication with optional GPU acceleration.
    Performs A @ B or A @ B @ C with automatic GPU/CPU selection and precision optimization.

    Parameters
    ----------
    A, B : ndarray
        Input matrices for multiplication
    C : ndarray, optional
        Third matrix for triple product A @ B @ C
    use_gpu : bool, optional
        Force GPU usage (True) or CPU usage (False). If None, auto-detect.

    Returns
    -------
    ndarray
        Result of matrix multiplication A @ B or A @ B @ C
    """
    if use_gpu is None:
        use_gpu = isCuda

    solver = cp if use_gpu else np
    if use_gpu and isCuda:
        dtype = cp.complex64 if USE_FLOAT32 else cp.complex128
    else:
        dtype = COMPUTE_DTYPE

    # Convert inputs to device arrays with optimized precision
    A_device = solver.asarray(A, dtype=dtype)
    B_device = solver.asarray(B, dtype=dtype)

    if C is None:
        # Simple multiplication A @ B
        result = solver.matmul(A_device, B_device)
    else:
        # Triple product A @ B @ C (common in NEGF: Gr @ Gamma @ Ga)
        C_device = solver.asarray(C, dtype=dtype)
        if use_gpu:
            # Use intermediate result to minimize GPU memory allocation
            temp = solver.matmul(A_device, B_device)
            result = solver.matmul(temp, C_device)
        else:
            # CPU can handle chained operations efficiently
            result = solver.matmul(solver.matmul(A_device, B_device), C_device)

    # Return CPU array
    return result.get() if use_gpu else result

def configure_workers_for_system(device_info, M, matrix_size=None):
    """
    Smart worker configuration that respects energy points and matrix size.

    Parameters
    ----------
    device_info : dict
        Device information from detect_devices()
    M : int
        Number of energy points to process
    matrix_size : int, optional
        Matrix dimension N for size-aware optimization

    Returns
    -------
    tuple
        (num_cpu_workers, original_env_vars) where original_env_vars is for cleanup
    """
    # Check for SLURM allocated cores first, then fall back to all available cores
    cpu_cores = None
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        gpu_logger.info(f"SLURM detected: Using {cpu_cores} allocated CPU cores (SLURM_CPUS_PER_TASK)")
    elif 'SLURM_CPUS_ON_NODE' in os.environ:
        cpu_cores = int(os.environ['SLURM_CPUS_ON_NODE'])
        gpu_logger.info(f"SLURM detected: Using {cpu_cores} allocated CPU cores (SLURM_CPUS_ON_NODE)")
    elif 'SLURM_NPROCS' in os.environ:
        cpu_cores = int(os.environ['SLURM_NPROCS'])
        gpu_logger.info(f"SLURM detected: Using {cpu_cores} allocated CPU cores (SLURM_NPROCS)")

    if cpu_cores is None:
        cpu_cores = multiprocessing.cpu_count()
        gpu_logger.info(f"No SLURM detected: Using all {cpu_cores} available CPU cores")

    original_env_vars = {}

    # Store original environment variables for cleanup
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        original_env_vars[var] = os.environ.get(var)

    # Matrix-size-aware threading configuration for parallel approach
    if matrix_size is None or matrix_size < 500:
        # Small matrices: Fewer workers with more threads (better BLAS utilization)
        threads_per_worker = max(8, cpu_cores // 2)
        optimal_cpu_workers = max(1, cpu_cores // threads_per_worker)
    elif matrix_size < 1500:
        # Medium matrices: Balanced threading
        threads_per_worker = 4
        optimal_cpu_workers = max(1, cpu_cores // threads_per_worker)
    else:
        # Large matrices: More workers with moderate threading
        threads_per_worker = 8
        optimal_cpu_workers = max(1, cpu_cores // threads_per_worker)

    # CRITICAL: Never create more workers than energy points!
    if M <= 1:
        # Single energy point: use all cores in one worker
        num_cpu_workers = 1
        threads_to_use = cpu_cores
    elif M < 10:
        # Few energy points: limit workers but ensure good threading
        if device_info['gpu_workers'] > 0:
            # With GPU: Ensure GPU gets guaranteed work by using fewer CPU workers
            num_cpu_workers = max(1, min(M - 1, optimal_cpu_workers // 2, 2))  # Leave work for GPU
            threads_to_use = max(4, cpu_cores // num_cpu_workers)
            gpu_logger.info(f"Few energy points with GPU: Using {num_cpu_workers} CPU workers to ensure GPU gets work")
        else:
            # CPU-only: can use more workers
            num_cpu_workers = min(M, optimal_cpu_workers, 4)  # Cap at 4 workers for small workloads
            threads_to_use = max(4, cpu_cores // num_cpu_workers)
    elif device_info['gpu_workers'] == 0:
        # CPU-only: CREATE AS MANY WORKERS AS POSSIBLE (respect energy point limit)
        num_cpu_workers = min(M, optimal_cpu_workers)
        threads_to_use = threads_per_worker
        gpu_logger.info(f"CPU-only: Maximizing workers - {num_cpu_workers} workers for {M} energies")
    else:
        # With GPU: Use fewer CPU workers (GPU is 19-47x faster, will get most work naturally)
        cores_per_gpu = 4  # Reserve 4 cores per GPU for thread management
        reserved_cores = device_info['gpu_workers'] * cores_per_gpu
        available_cores = max(8, cpu_cores - reserved_cores)  # Ensure decent minimum
        max_cpu_workers = max(1, available_cores // 4)

        # Use fewer CPU workers when GPU present for natural load balancing
        num_cpu_workers = min(M, max_cpu_workers // 2)  # Use half the CPU workers when GPU present
        threads_to_use = max(4, available_cores // num_cpu_workers)
        gpu_logger.info(f"With GPU: Using {num_cpu_workers} CPU workers (reduced for GPU priority), {reserved_cores} cores reserved for {device_info['gpu_workers']} GPUs")

    # Set threading environment
    os.environ['OMP_NUM_THREADS'] = str(threads_to_use)
    os.environ['MKL_NUM_THREADS'] = str(threads_to_use)

    # Log configuration
    matrix_info = f"matrix={matrix_size}x{matrix_size}, " if matrix_size else ""
    gpu_logger.info(f"Worker config: {matrix_info}M={M} energies, {num_cpu_workers} CPU workers x {threads_to_use} threads")
    if device_info['gpu_workers'] > 0:
        gpu_logger.info(f"GPU config: {device_info['gpu_workers']} GPU workers (natural priority through speed)")
    if num_cpu_workers > M and M > 1:
        gpu_logger.warning(f"Note: {num_cpu_workers} workers for {M} energy points - some workers will be idle")

    return num_cpu_workers, original_env_vars


def cleanup_environment_variables(original_env_vars):
    """
    Restore original environment variables after computation.

    Parameters
    ----------
    original_env_vars : dict
        Original environment variable values to restore
    """
    for var, original_value in original_env_vars.items():
        if original_value is not None:
            os.environ[var] = original_value
        else:
            os.environ.pop(var, None)


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

    # Check if we should use optimized small matrix path
    if N < 500 and M < 20:
        gpu_logger.info(f"Small matrix optimization: {N}x{N} with {M} energies - using vectorized approach")
        solver = cp if isCuda else np
        try:
            result = GrIntVectorized(F, S, g, Elist, weights, solver)
            total_time = time.perf_counter() - start_time
            gpu_logger.info(f"Completed GrInt (small matrix): {total_time:.3f}s total")
            return result
        finally:
            pass  # No environment cleanup needed for vectorized approach

    # Detect available devices and configure workers for parallel approach
    device_info = detect_devices()
    num_cpu_workers, original_env_vars = configure_workers_for_system(device_info, M, N)
    total_workers = num_cpu_workers + device_info['gpu_workers']

    # Log calculation start
    memory_gb = N * N * 16 / 1e9
    gpu_logger.info(f"Starting GrInt: {N}x{N} matrices, {M} energies ({memory_gb:.2f}GB per matrix)")
    gpu_logger.info(f"Using {total_workers} workers: {device_info['gpu_workers']} GPU(s) + {num_cpu_workers} CPU")

    try:
        # Simple shared queue approach (like original) - GPU gets priority through fewer CPU workers
        work_queue = queue.Queue()
        for i, (E, weight) in enumerate(zip(Elist, weights)):
            work_queue.put((E, weight, i))

        # Initialize shared result and progress counter
        result_lock = threading.Lock()
        shared_result = {'matrix': np.zeros((N, N), dtype=COMPUTE_DTYPE)}
        progress_counter = {'completed': 0}

        # Create and start worker threads
        threads = []

        # Start GPU workers FIRST (they're faster, so they'll get most work naturally)
        for gpu_id in range(device_info['gpu_workers']):
            gpu_thread = threading.Thread(
                target=worker_gr,
                args=(work_queue, result_lock, shared_result, F, S, g, gpu_id, f"GPU-{gpu_id}", progress_counter)
            )
            threads.append(gpu_thread)
            gpu_thread.start()

        # Start CPU workers
        for cpu_id in range(num_cpu_workers):
            cpu_thread = threading.Thread(
                target=worker_gr,
                args=(work_queue, result_lock, shared_result, F, S, g, None, f"CPU-{cpu_id}", progress_counter)
            )
            threads.append(cpu_thread)
            cpu_thread.start()

        # Progress monitoring with corrected timing
        def monitor_progress():
            last_completed = 0
            work_start_time = None

            while any(t.is_alive() for t in threads):
                try:
                    with result_lock:
                        completed = progress_counter['completed']

                    if completed > 0 and work_start_time is None:
                        # Start timing when first energy point completes (after GPU initialization)
                        work_start_time = time.perf_counter()
                        gpu_logger.info("First energy point completed, starting performance timing")

                    if completed != last_completed and work_start_time is not None:
                        progress = 100.0 * completed / M
                        work_elapsed = time.perf_counter() - work_start_time
                        rate = completed / work_elapsed if work_elapsed > 0.01 else 0
                        remaining = M - completed
                        eta = remaining / rate if rate > 0 else 0

                        if completed < M:
                            gpu_logger.debug(f"Progress: {progress:.1f}% ({completed}/{M}) ({rate:.1f} energies/s, ETA: {eta:.1f}s)")

                        last_completed = completed
                except Exception:
                    pass
                time.sleep(0.5)  # More frequent updates for fast GPU work

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

        # Performance monitoring
        precision_str = "float32" if USE_FLOAT32 else "float64"
        gpu_workers = device_info['gpu_workers']
        speedup_potential = "19-47x" if USE_FLOAT32 and gpu_workers > 0 else "2.5x"

        gpu_logger.info(f"Completed GrInt: {total_time:.3f}s total ({throughput:.2e} sec/energy)")
        gpu_logger.info(f"Performance: {precision_str} precision, {gpu_workers} GPU workers (natural priority), potential speedup: {speedup_potential}")

        if gpu_workers > 0 and not USE_FLOAT32:
            gpu_logger.warning(f"GPU performance penalty: Using float64 instead of float32 (expected 19-47x slower)")

        return shared_result['matrix']

    finally:
        # Always restore original environment variables, even if an exception occurs
        cleanup_environment_variables(original_env_vars)


def GrIntVectorized(F, S, g, Elist, weights, solver):
    """
    Reference implementation for math verification - preserved exactly from original code.

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

    # Convert array types to match solver with configurable precision:
    if isCuda and solver == cp:
        dtype = solver.complex64 if USE_FLOAT32 else solver.complex128
    else:
        dtype = COMPUTE_DTYPE
    Elist_ = solver.array(Elist, dtype=dtype)
    weights = solver.array(weights)
    S = solver.array(S, dtype=dtype)
    F = solver.array(F, dtype=dtype)

    # Memory tracking: 0 NxNxM arrays allocated

    #Generate vectorized matrices conserving memory
    ES_minus_F_minus_Sig = Elist_[:, None, None] * solver.tile(solver.array(S), (M, 1, 1))
    # Memory tracking: 1 NxNxM array (ES_minus_F_minus_Sig)

    ES_minus_F_minus_Sig -= solver.tile(solver.array(F), (M, 1, 1))
    ES_minus_F_minus_Sig -= solver.array([g.sigmaTot(E) for E in Elist])
    # Memory tracking: Still 1 NxNxM array (ES_minus_F_minus_Sig, temp tiles destroyed)

    Gr_vec = solver.linalg.solve(ES_minus_F_minus_Sig, solver.tile(solver.eye(N), (M, 1, 1)))
    # Memory tracking: PEAK 3 NxNxM arrays (ES_minus_F_minus_Sig + Gr_vec + temp eye tile)

    del ES_minus_F_minus_Sig
    # Memory tracking: 1-2 NxNxM arrays (Gr_vec + potential linalg temp)

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gr_vec*weights[:, None, None], axis=0)
    # Memory tracking: 1-2 NxNxM arrays (temp from multiplication)

    del Gr_vec
    # Memory tracking: 0 NxNxM arrays

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

    # Check if we should use optimized small matrix path
    if N < 500 and M < 20:
        gpu_logger.info(f"Small matrix optimization: {N}x{N} with {M} energies - using vectorized approach")
        solver = cp if isCuda else np
        try:
            result = GrLessVectorized(F, S, g, Elist, weights, solver, ind)
            total_time = time.perf_counter() - start_time
            gpu_logger.info(f"Completed GrLessInt (small matrix): {total_time:.3f}s total")
            return result
        finally:
            pass  # No environment cleanup needed for vectorized approach

    # Detect available devices and configure workers for parallel approach
    device_info = detect_devices()
    num_cpu_workers, original_env_vars = configure_workers_for_system(device_info, M, N)
    total_workers = num_cpu_workers + device_info['gpu_workers']

    # Log calculation start
    memory_gb = N * N * 16 / 1e9
    gpu_logger.info(f"Starting GrLessInt: {N}x{N} matrices, {M} energies ({memory_gb:.2f}GB per matrix)")
    gpu_logger.info(f"Using {total_workers} workers: {device_info['gpu_workers']} GPU(s) + {num_cpu_workers} CPU")

    try:
        # Simple shared queue approach (like original) - GPU gets priority through fewer CPU workers
        work_queue = queue.Queue()
        for i, (E, weight) in enumerate(zip(Elist, weights)):
            work_queue.put((E, weight, i))

        # Initialize shared result and progress counter
        result_lock = threading.Lock()
        shared_result = {'matrix': np.zeros((N, N), dtype=COMPUTE_DTYPE)}
        progress_counter = {'completed': 0}

        # Create and start worker threads
        threads = []

        # Start GPU workers FIRST (they're faster, so they'll get most work naturally)
        for gpu_id in range(device_info['gpu_workers']):
            gpu_thread = threading.Thread(
                target=worker_grless,
                args=(work_queue, result_lock, shared_result, F, S, g, ind, gpu_id, f"GPU-{gpu_id}", progress_counter)
            )
            threads.append(gpu_thread)
            gpu_thread.start()

        # Start CPU workers
        for cpu_id in range(num_cpu_workers):
            cpu_thread = threading.Thread(
                target=worker_grless,
                args=(work_queue, result_lock, shared_result, F, S, g, ind, None, f"CPU-{cpu_id}", progress_counter)
            )
            threads.append(cpu_thread)
            cpu_thread.start()

        # Progress monitoring with corrected timing
        def monitor_progress():
            last_completed = 0
            work_start_time = None

            while any(t.is_alive() for t in threads):
                try:
                    with result_lock:
                        completed = progress_counter['completed']

                    if completed > 0 and work_start_time is None:
                        # Start timing when first energy point completes (after GPU initialization)
                        work_start_time = time.perf_counter()
                        gpu_logger.info("First energy point completed, starting performance timing")

                    if completed != last_completed and work_start_time is not None:
                        progress = 100.0 * completed / M
                        work_elapsed = time.perf_counter() - work_start_time
                        rate = completed / work_elapsed if work_elapsed > 0.01 else 0
                        remaining = M - completed
                        eta = remaining / rate if rate > 0 else 0

                        if completed < M:
                            gpu_logger.debug(f"Progress: {progress:.1f}% ({completed}/{M}) ({rate:.1f} energies/s, ETA: {eta:.1f}s)")

                        last_completed = completed
                except Exception:
                    pass
                time.sleep(0.5)  # More frequent updates for fast GPU work

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

        # Performance monitoring
        precision_str = "float32" if USE_FLOAT32 else "float64"
        gpu_workers = device_info['gpu_workers']
        speedup_potential = "31-57x" if USE_FLOAT32 and gpu_workers > 0 else "2.6x"

        gpu_logger.info(f"Completed GrLessInt: {total_time:.3f}s total ({throughput:.2e} sec/energy)")
        gpu_logger.info(f"Performance: {precision_str} precision, {gpu_workers} GPU workers (natural priority), potential speedup: {speedup_potential}")

        if gpu_workers > 0 and not USE_FLOAT32:
            gpu_logger.warning(f"GPU performance penalty: Using float64 instead of float32 (expected 31-57x slower)")

        return shared_result['matrix']

    finally:
        # Always restore original environment variables, even if an exception occurs
        cleanup_environment_variables(original_env_vars)



def GrLessVectorized(F, S, g, Elist, weights, solver, ind):
    """
    Reference implementation for math verification - full vectorized G< implementation.

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

    # Convert array types to match solver with configurable precision:
    if isCuda and solver == cp:
        dtype = solver.complex64 if USE_FLOAT32 else solver.complex128
    else:
        dtype = COMPUTE_DTYPE
    Elist_ = solver.array(Elist, dtype=dtype)
    weights = solver.array(weights)
    S = solver.array(S, dtype=dtype)
    F = solver.array(F, dtype=dtype)

    # Memory tracking: 0 NxNxM arrays allocated

    #Generate Gr and Ga vectorized matrices conserving memory
    ES_minus_F_minus_Sig = Elist_[:, None, None] * solver.tile(S, (M, 1, 1))
    # Memory tracking: 1 NxNxM array (ES_minus_F_minus_Sig)

    ES_minus_F_minus_Sig -= solver.tile(F, (M, 1, 1))
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])
    # Memory tracking: 2 NxNxM arrays (ES_minus_F_minus_Sig + SigmaTot)

    ES_minus_F_minus_Sig -= SigmaTot
    # Memory tracking: Still 2 NxNxM arrays

    Gr_vec = solver.linalg.solve(ES_minus_F_minus_Sig, solver.tile(solver.eye(N), (M, 1, 1)))
    # Memory tracking: PEAK 4 NxNxM arrays (ES_minus_F_minus_Sig + SigmaTot + Gr_vec + temp eye)

    del ES_minus_F_minus_Sig
    # Memory tracking: 2-3 NxNxM arrays (SigmaTot + Gr_vec + potential linalg temp)

    Ga_vec = solver.conj(Gr_vec).transpose(0, 2, 1)
    # Memory tracking: 3-4 NxNxM arrays (SigmaTot + Gr_vec + Ga_vec + potential temp)

    # Calculate Gamma:
    if ind is None:
        SigList = SigmaTot  # Memory tracking: No new array, just reference
    else:
        del SigmaTot
        # Memory tracking: Reduces by 1 NxNxM array
        SigList = solver.array([g.sigma(E, ind) for E in Elist])
        # Memory tracking: Back to same count with new SigList

    GammaList = 1j * (SigList - solver.conj(SigList).transpose(0, 2, 1))
    # Memory tracking: PEAK 5 NxNxM arrays (SigList + Gr_vec + Ga_vec + GammaList + temp from subtraction)

    del SigList
    # Memory tracking: 3-4 NxNxM arrays (Gr_vec + Ga_vec + GammaList + potential temp)

    # Calculate Gless:
    Gless_vec = solver.matmul(solver.matmul(Gr_vec, GammaList), Ga_vec)
    # Memory tracking: 4-5 NxNxM arrays during matmul chain

    del Gr_vec, Ga_vec, GammaList
    # Memory tracking: 1 NxNxM array (Gless_vec)

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gless_vec*weights[:, None, None], axis=0)
    del Gless_vec
    # Memory tracking: 0 NxNxM arrays

    return Gint.get() if isCuda else Gint

