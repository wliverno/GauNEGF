"""
Generic JAX-based parallelization framework for energy-dependent calculations.

A flexible system that adapts the proven integrate.py queue architecture to work with
any energy-dependent function using JAX for optimal CPU/GPU performance.

Author: Based on integrate.py architecture
"""

import os
import time
import threading
import queue
import multiprocessing
import socket
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import logging
from jax import jit

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# =============================================================================
# CONFIGURABLE CONSTANTS - Tune these based on testing
# =============================================================================

# Performance thresholds (based on JAX benchmarking)
SMALL_MATRIX_THRESHOLD = 500          # Use vmap for matrices smaller than this
MAX_VMAP_MEMORY_GB = 4.0              # Use vmap if estimated memory < this (GB)
MAX_EFFICIENT_CPU_CORES = 8           # Performance drops after this many cores (from testing)

# Worker configuration
FEW_ENERGY_THRESHOLD = 10             # Switch to limited worker mode below this
MAX_WORKERS_SMALL_WORKLOAD = 4        # Cap workers for small workloads

# Queue and monitoring
WORKER_TIMEOUT_SECONDS = 1.0          # Queue timeout for workers
PROGRESS_UPDATE_INTERVAL = 1.0        # Progress monitoring interval

# Memory optimization (from integrate.py)
MEMORY_PER_MATRIX_FACTOR = 16         # Bytes per complex128 element
BYTES_TO_GB = 1e9                     # Conversion factor

# =============================================================================

# Setup node-specific logging for parallel operations (copied from integrate.py)
hostname = socket.gethostname()
pid = os.getpid()

from gauNEGF.config import LOG_LEVEL, LOG_PERFORMANCE

if LOG_PERFORMANCE:
    log_file = f'parallelize_performance_{hostname}_{pid}.log'
else:
    temp_dir = tempfile.gettempdir()
    log_file = os.path.join(temp_dir, f'parallelize_performance_{hostname}_{pid}.log')

log_level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)

parallel_logger = logging.getLogger('gauNEGF.parallel')
parallel_logger.setLevel(log_level)

# Create file handler that appends (avoid duplicate handlers on reload)
if not parallel_logger.handlers:
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    parallel_logger.addHandler(handler)

parallel_logger.info("JAX parallel framework initialized")


def detect_hardware():
    """
    Detect available JAX devices and CPU cores (SLURM-aware).

    Returns
    -------
    dict
        Hardware configuration with JAX devices and CPU core count
    """
    # JAX device detection
    jax_devices = jax.devices()
    gpu_available = any('gpu' in str(device).lower() for device in jax_devices)

    # SLURM-aware CPU detection (copied from integrate.py logic)
    cpu_cores = None
    slurm_source = None

    # Priority order: most specific to least specific
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        slurm_source = "SLURM_CPUS_PER_TASK"
    elif 'SRUN_CPUS_PER_TASK' in os.environ:
        cpu_cores = int(os.environ['SRUN_CPUS_PER_TASK'])
        slurm_source = "SRUN_CPUS_PER_TASK"
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        cpu_cores = int(os.environ['SLURM_NTASKS_PER_NODE'])
        slurm_source = "SLURM_NTASKS_PER_NODE"
    elif 'SLURM_CPUS_ON_NODE' in os.environ:
        cpu_cores = int(os.environ['SLURM_CPUS_ON_NODE'])
        slurm_source = "SLURM_CPUS_ON_NODE"
    elif 'SLURM_NPROCS' in os.environ:
        cpu_cores = int(os.environ['SLURM_NPROCS'])
        slurm_source = "SLURM_NPROCS"

    if cpu_cores is None:
        cpu_cores = multiprocessing.cpu_count()
        parallel_logger.info(f"No SLURM detected: Using all {cpu_cores} available CPU cores")
    else:
        parallel_logger.info(f"SLURM detected: Using {cpu_cores} allocated CPU cores ({slurm_source})")

    return {
        'jax_devices': jax_devices,
        'gpu_available': gpu_available,
        'cpu_cores': cpu_cores,
        'slurm_source': slurm_source
    }


def configure_workers(hardware_info, num_energy_points, matrix_size=None):
    """
    Configure optimal number of workers based on hardware and problem size.
    Copied logic from integrate.py configure_workers_for_system().

    Parameters
    ----------
    hardware_info : dict
        Hardware information from detect_hardware()
    num_energy_points : int
        Number of energy points to process
    matrix_size : int, optional
        Matrix dimension for memory optimization

    Returns
    -------
    int
        Optimal number of CPU workers
    """
    cpu_cores = hardware_info['cpu_cores']
    gpu_available = hardware_info['gpu_available']
    M = num_energy_points

    # Each worker gets MAX_EFFICIENT_CPU_CORES for optimal performance
    optimal_cpu_workers = max(1, cpu_cores // MAX_EFFICIENT_CPU_CORES)

    # CRITICAL: Never create more workers than energy points!
    if M <= 1:
        num_workers = 1
    elif M < FEW_ENERGY_THRESHOLD:
        # Few energy points: limit workers
        if gpu_available:
            num_workers = max(1, min(M - 1, 2))  # Leave work for GPU
            parallel_logger.info(f"Few energy points with GPU: Using {num_workers} workers")
        else:
            num_workers = min(M, optimal_cpu_workers, MAX_WORKERS_SMALL_WORKLOAD)
    elif not gpu_available:
        # CPU-only: maximize workers (respect energy point limit and efficiency)
        num_workers = min(M, optimal_cpu_workers)
        parallel_logger.info(f"CPU-only: Using {num_workers} workers for {M} energies")
    else:
        # With GPU: fewer CPU workers (GPU gets priority through speed)
        max_cpu_workers = max(1, optimal_cpu_workers // 2)
        num_workers = min(M, max_cpu_workers)
        parallel_logger.info(f"With GPU: Using {num_workers} CPU workers (GPU priority)")

    # Log configuration
    matrix_info = f"matrix={matrix_size}x{matrix_size}, " if matrix_size else ""
    parallel_logger.info(f"Worker config: {matrix_info}M={M} energies, {num_workers} workers")
    parallel_logger.info(f"CPU cores: {cpu_cores} total, {num_workers} workers × {MAX_EFFICIENT_CPU_CORES} cores each")
    if gpu_available:
        parallel_logger.info(f"JAX GPU available: {hardware_info['jax_devices']}")

    return num_workers


def generic_worker(work_queue, result_lock, shared_result, user_function,
                  use_gpu, worker_id, **function_kwargs):
    """
    Generic worker that executes user function for each energy point.
    Returns results as dictionary indexed by energy point index.

    Parameters
    ----------
    work_queue : queue.Queue
        Queue containing (energy, index) tuples
    result_lock : threading.Lock
        Lock for thread-safe result accumulation
    shared_result : dict
        Shared dictionary for accumulating results
    user_function : callable
        JIT-compiled JAX function that takes (energy, **kwargs) and returns result
    use_gpu : bool
        Whether this worker should use GPU (JAX handles device automatically)
    worker_id : str
        Worker identification for logging
    **function_kwargs
        Additional arguments to pass to user_function
    """
    device_name = "GPU" if use_gpu else "CPU"

    try:
        parallel_logger.debug(f"{device_name} worker {worker_id} started")
        local_count = 0
        local_results = {}  # Store results by index

        while True:
            try:
                # Get work from queue with timeout
                E, idx = work_queue.get(timeout=WORKER_TIMEOUT_SECONDS)

                # Execute user's function (JIT-compiled JAX)
                start_time = time.perf_counter()
                function_result = user_function(E, **function_kwargs)
                calc_time = time.perf_counter() - start_time

                # Store result with its energy and index
                local_results[idx] = {
                    'energy': E,
                    'result': np.array(function_result)  # Convert JAX -> NumPy
                }

                local_count += 1
                work_queue.task_done()

                parallel_logger.debug(f"{device_name} worker {worker_id}: E={E:.6f}, calc_time={calc_time:.4f}s")

            except queue.Empty:
                # No more work available
                break
            except Exception as e:
                # Worker failure is critical - propagate exception
                parallel_logger.error(f"{device_name} worker {worker_id} calculation error: {e}")
                work_queue.task_done()
                # Store the exception in shared result to be raised later
                with result_lock:
                    shared_result['worker_error'] = f"{device_name} worker {worker_id} failed: {e}"
                raise RuntimeError(f"Worker {worker_id} failed during calculation") from e

        # Transfer results to shared storage
        if local_count > 0:
            with result_lock:
                if 'results' not in shared_result:
                    shared_result['results'] = {}
                shared_result['results'].update(local_results)

        parallel_logger.debug(f"{device_name} worker {worker_id} completed {local_count} energy points")

    except Exception as e:
        # Worker initialization/critical failure - this is a big deal!
        parallel_logger.error(f"{device_name} worker {worker_id} critical failure: {e}")
        with result_lock:
            shared_result['worker_error'] = f"{device_name} worker {worker_id} critical failure: {e}"
        raise RuntimeError(f"Critical worker failure in {worker_id}") from e


def parallelize_energy_calculation(energy_list, user_function, matrix_size=None,
                                 **function_kwargs):
    """
    Generic parallelization for any energy-dependent function.

    Parameters
    ----------
    energy_list : array_like
        List of energy values to process
    user_function : callable
        JAX function that takes (energy, **kwargs) -> result
        Should be JIT-compiled for best performance
    matrix_size : int, optional
        Matrix dimension (for memory optimization decisions)
    **function_kwargs
        Additional arguments passed to user_function

    Returns
    -------
    dict
        Dictionary with results indexed by energy point:
        {0: {'energy': E0, 'result': result0}, 1: {'energy': E1, 'result': result1}, ...}

    Examples
    --------
    # For transmission calculation:
    @jit
    def transmission_at_energy(E, F, S, sigma_calc):
        mat = E * S - F - sigma_calc.get_sigma_total(E)
        Gr = jnp.linalg.inv(mat)
        # ... transmission calculation
        return transmission

    results = parallelize_energy_calculation(
        energy_list, transmission_at_energy,
        matrix_size=F.shape[0], F=F, S=S, sigma_calc=sigma_calc
    )
    """
    start_time = time.perf_counter()
    energy_array = np.array(energy_list)
    M = len(energy_array)

    # Detect hardware and configure workers
    hardware_info = detect_hardware()

    # Memory-based vmap decision: use JAX vmap if estimated memory usage is reasonable
    use_vmap = False
    if matrix_size is not None:
        # Estimate memory: M energy points × matrix_size² × 16 bytes (complex128)
        estimated_memory_gb = (M * matrix_size * matrix_size * MEMORY_PER_MATRIX_FACTOR) / BYTES_TO_GB
        use_vmap = (matrix_size < SMALL_MATRIX_THRESHOLD and estimated_memory_gb < MAX_VMAP_MEMORY_GB)

    if use_vmap:
        parallel_logger.info(f"Memory-based vmap optimization: {matrix_size}x{matrix_size} × {M} energies = {estimated_memory_gb:.1f}GB - using JAX vmap")

        # Vectorized approach using JAX vmap
        @jit
        def vectorized_calculation(energies):
            return jax.vmap(lambda E: user_function(E, **function_kwargs))(energies)

        results_array = vectorized_calculation(energy_array)

        # Convert to same format as worker results
        results_dict = {}
        for i, (E, result) in enumerate(zip(energy_array, results_array)):
            results_dict[i] = {
                'energy': E,
                'result': np.array(result)
            }

        total_time = time.perf_counter() - start_time
        parallel_logger.info(f"Completed (vectorized): {total_time:.3f}s total")
        return results_dict

    # Large problem: use worker queue system
    num_workers = configure_workers(hardware_info, M, matrix_size)

    parallel_logger.info(f"Starting parallel calculation: {M} energies, {num_workers} workers")

    # Create work queue - simplified (energy, index) tuples
    work_queue = queue.Queue()
    for i, E in enumerate(energy_array):
        work_queue.put((E, i))

    # Initialize shared result
    result_lock = threading.Lock()
    shared_result = {}

    # Create and start worker threads
    threads = []

    # GPU worker (if available) - starts first to get priority
    if hardware_info['gpu_available']:
        gpu_thread = threading.Thread(
            target=generic_worker,
            args=(work_queue, result_lock, shared_result, user_function, True, "GPU-0"),
            kwargs=function_kwargs
        )
        threads.append(gpu_thread)
        gpu_thread.start()

    # CPU workers
    for cpu_id in range(num_workers):
        cpu_thread = threading.Thread(
            target=generic_worker,
            args=(work_queue, result_lock, shared_result, user_function, False, f"CPU-{cpu_id}"),
            kwargs=function_kwargs
        )
        threads.append(cpu_thread)
        cpu_thread.start()

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
                    parallel_logger.debug(f"Progress: {progress:.1f}% ({completed}/{M}) ({rate:.1f} energies/s)")
                    last_remaining = remaining
            except:
                pass
            time.sleep(PROGRESS_UPDATE_INTERVAL)

    # Start progress monitor
    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Wait for all workers to complete
    for thread in threads:
        thread.join()

    work_queue.join()

    # Check for worker errors
    if 'worker_error' in shared_result:
        raise RuntimeError(shared_result['worker_error'])

    total_time = time.perf_counter() - start_time
    throughput = total_time / M

    parallel_logger.info(f"Completed parallel calculation: {total_time:.3f}s total ({throughput:.2e} sec/energy)")

    return shared_result.get('results', {})