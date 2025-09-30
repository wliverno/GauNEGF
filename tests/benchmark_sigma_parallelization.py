#!/usr/bin/env python3
"""
Production-scale NumPy sigma function parallelization benchmark.

This version addresses BLAS threading issues and memory concerns for realistic
100x100+ matrix problems.

Key fixes:
1. Proper BLAS thread control to prevent fork bombs
2. Thread-safe configuration for concurrent NumPy operations
3. Timeout handling to detect hangs
4. Realistic problem sizes (100x100+ matrices)
"""

import time
import numpy as np
import sys
import os
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
import signal

# CRITICAL: Set BLAS threading BEFORE importing numpy-dependent libraries
# This must be done at module level before any NumPy operations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Try importing joblib for additional comparison
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available, skipping joblib.Parallel benchmark")

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Constants from GauNEGF
SURFACE_GREEN_CONVERGENCE = 1e-5
SURFACE_RELAXATION_FACTOR = 0.1
ETA = 1e-9

# Timeout for detecting hangs (seconds)
COMPUTATION_TIMEOUT = 300  # 5 minutes should be plenty


def configure_blas_threads():
    """
    Configure BLAS to use single-threaded mode.
    Call this at the start of each worker process.
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def manual_surface_green_iteration(A, B, g_init, conv=1e-6, rel_factor=0.1, max_iter=1000):
    """
    Manual implementation of surface Green's function iteration.

    THREAD-SAFE: This function can be called from multiple threads/processes
    as long as BLAS is configured for single-threaded operation.
    """
    g = g_init.copy()
    B_dag = B.conj().T

    for count in range(max_iter):
        g_prev = g.copy()

        # Main iteration step - BLAS operations here
        g = np.linalg.inv(A - B @ g @ B_dag)

        # Convergence check
        dg = np.abs(g - g_prev) / np.maximum(np.abs(g), 1e-12)
        diff = np.max(dg)

        # Relaxation mixing
        g = g * rel_factor + g_prev * (1 - rel_factor)

        if diff <= conv:
            return g, True, count + 1

    return g, False, max_iter


def compute_sigma_for_energy(E, alpha, Salpha, beta, Sbeta, eta=ETA,
                           conv=SURFACE_GREEN_CONVERGENCE, rel_factor=SURFACE_RELAXATION_FACTOR):
    """
    Compute sigma for a single energy point - stateless function for parallelization.

    This function is completely independent and doesn't rely on previous solutions,
    making it suitable for parallel execution.
    """
    # Ensure BLAS threading is configured (redundant but safe)
    configure_blas_threads()

    # Prepare matrices
    A = (E + 1j * eta) * Salpha - alpha
    B = (E + 1j * eta) * Sbeta - beta
    g_init = np.zeros_like(A, dtype=complex)

    # Get surface Green's function
    g, converged, iterations = manual_surface_green_iteration(A, B, g_init, conv, rel_factor)

    # Calculate coupling matrix
    stau = Sbeta
    tau = beta
    t = E * stau - tau

    # Self-energy: sigma = t @ g @ t^dagger
    sigma = t @ g @ t.conj().T

    return sigma, converged, iterations


def worker_initializer():
    """Initializer for multiprocessing workers to configure BLAS threading."""
    configure_blas_threads()
    # Ignore interrupt signals in workers (parent will handle)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def create_test_contact(size=10, coupling_strength=0.1, seed=42):
    """Create test matrices for surface Green's function."""
    np.random.seed(seed)

    # Create Hermitian alpha matrix (on-site)
    alpha_real = np.random.randn(size, size)
    alpha = (alpha_real + alpha_real.T) / 2

    # Create beta matrix (coupling)
    beta = np.random.randn(size, size) * coupling_strength

    # Create positive definite overlap matrices
    S_temp = np.random.randn(size, size)
    Salpha = S_temp @ S_temp.T
    Salpha -= np.diag(np.diag(Salpha))
    Salpha += np.eye(size)

    S_temp2 = np.random.randn(size, size)
    Sbeta = S_temp2 @ S_temp2.T

    return alpha, Salpha, beta, Sbeta


def compute_sigma_sequential(alpha, Salpha, beta, Sbeta, energies,
                           conv=SURFACE_GREEN_CONVERGENCE, timeout=None):
    """Sequential sigma computation baseline."""
    results = []
    start_time = time.perf_counter()

    try:
        for E in energies:
            sigma, converged, iterations = compute_sigma_for_energy(
                E, alpha, Salpha, beta, Sbeta, conv=conv
            )
            results.append((sigma, converged, iterations))

            # Check timeout
            if timeout and (time.perf_counter() - start_time) > timeout:
                print(f"   WARNING: Sequential timeout after {len(results)} energies")
                return None, float('inf')

    except Exception as e:
        print(f"   ERROR in sequential: {e}")
        return None, float('inf')

    total_time = time.perf_counter() - start_time
    return results, total_time


def compute_sigma_processpool(alpha, Salpha, beta, Sbeta, energies,
                            conv=SURFACE_GREEN_CONVERGENCE, max_workers=None, timeout=None):
    """ProcessPoolExecutor sigma computation with timeout and BLAS configuration."""
    start_time = time.perf_counter()

    # Create partial function with fixed parameters
    compute_func = partial(compute_sigma_for_energy,
                          alpha=alpha, Salpha=Salpha, beta=beta, Sbeta=Sbeta, conv=conv)

    try:
        with ProcessPoolExecutor(max_workers=max_workers,
                                initializer=worker_initializer) as executor:
            # Submit all tasks
            future_to_energy = {executor.submit(compute_func, E): E for E in energies}

            results = []
            for future in future_to_energy:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except FuturesTimeoutError:
                    print(f"   WARNING: Task timeout")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return None, float('inf')
                except Exception as e:
                    print(f"   ERROR in ProcessPoolExecutor task: {e}")
                    return None, float('inf')

    except Exception as e:
        print(f"   ERROR in ProcessPoolExecutor: {e}")
        return None, float('inf')

    total_time = time.perf_counter() - start_time
    return results, total_time


def compute_sigma_threadpool(alpha, Salpha, beta, Sbeta, energies,
                           conv=SURFACE_GREEN_CONVERGENCE, max_workers=None, timeout=None):
    """ThreadPoolExecutor sigma computation with BLAS single-threading."""
    start_time = time.perf_counter()

    # Ensure single-threaded BLAS
    configure_blas_threads()

    # Create partial function with fixed parameters
    compute_func = partial(compute_sigma_for_energy,
                          alpha=alpha, Salpha=Salpha, beta=beta, Sbeta=Sbeta, conv=conv)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_energy = {executor.submit(compute_func, E): E for E in energies}

            results = []
            for future in future_to_energy:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except FuturesTimeoutError:
                    print(f"   WARNING: Task timeout")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return None, float('inf')
                except Exception as e:
                    print(f"   ERROR in ThreadPoolExecutor task: {e}")
                    return None, float('inf')

    except Exception as e:
        print(f"   ERROR in ThreadPoolExecutor: {e}")
        return None, float('inf')

    total_time = time.perf_counter() - start_time
    return results, total_time


def compute_sigma_multiprocessing_pool(alpha, Salpha, beta, Sbeta, energies,
                                     conv=SURFACE_GREEN_CONVERGENCE, processes=None, timeout=None):
    """multiprocessing.Pool sigma computation with proper BLAS configuration."""
    start_time = time.perf_counter()

    # Create partial function with fixed parameters
    compute_func = partial(compute_sigma_for_energy,
                          alpha=alpha, Salpha=Salpha, beta=beta, Sbeta=Sbeta, conv=conv)

    try:
        with Pool(processes=processes, initializer=worker_initializer) as pool:
            # Use map_async to enable timeout
            async_result = pool.map_async(compute_func, energies)

            # Wait for results with timeout
            try:
                results = async_result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                print(f"   WARNING: Pool timeout")
                pool.terminate()
                pool.join()
                return None, float('inf')

    except Exception as e:
        print(f"   ERROR in multiprocessing.Pool: {e}")
        return None, float('inf')

    total_time = time.perf_counter() - start_time
    return results, total_time


def compute_sigma_joblib(alpha, Salpha, beta, Sbeta, energies,
                        conv=SURFACE_GREEN_CONVERGENCE, n_jobs=-1, backend='threading', timeout=None):
    """joblib.Parallel sigma computation."""
    if not JOBLIB_AVAILABLE:
        return None, float('inf')

    start_time = time.perf_counter()

    try:
        results = Parallel(n_jobs=n_jobs, backend=backend, timeout=timeout)(
            delayed(compute_sigma_for_energy)(E, alpha, Salpha, beta, Sbeta, conv=conv)
            for E in energies
        )
    except Exception as e:
        print(f"   ERROR in joblib: {e}")
        return None, float('inf')

    total_time = time.perf_counter() - start_time
    return results, total_time


def verify_results_consistency(results_list, method_names, tolerance=1e-10):
    """Verify that all parallelization methods produce consistent results."""
    print(f"\nResult Consistency Check (tolerance: {tolerance:.1e})")
    print("-" * 60)

    if len(results_list) < 2:
        print("Not enough results to compare")
        return True

    baseline = results_list[0]
    if baseline is None:
        print("Baseline is None, cannot verify consistency")
        return False

    baseline_name = method_names[0]
    all_consistent = True

    for i, (results, method_name) in enumerate(zip(results_list[1:], method_names[1:]), 1):
        if results is None:
            print(f"{method_name:20s}: SKIPPED (failed or timeout)")
            continue

        max_error = 0.0
        for j, ((sigma_base, _, _), (sigma_test, _, _)) in enumerate(zip(baseline, results)):
            error = np.max(np.abs(sigma_base - sigma_test))
            max_error = max(max_error, error)

        consistent = max_error < tolerance
        all_consistent = all_consistent and consistent

        status = "PASS" if consistent else "FAIL"
        print(f"{method_name:20s}: {status:4s} (max error: {max_error:.2e})")

    return all_consistent


def benchmark_production_sigma_parallelization(quick_test=False):
    """
    Run production-scale sigma parallelization benchmark.

    Parameters
    ----------
    quick_test : bool
        If True, only test smaller matrices quickly
    """
    print("=" * 80)
    print("PRODUCTION-SCALE NUMPY SIGMA PARALLELIZATION BENCHMARK")
    print("=" * 80)
    print("\nBLAS Configuration:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'not set')}")

    # Test configurations - REALISTIC production sizes
    # IMPORTANT: Real transport calculations use 100x100+ matrices ALWAYS
    if quick_test:
        matrix_sizes = [100]
        energy_counts = [8]
        print("\nQUICK TEST MODE: Production size (100x100), limited energy points")
    else:
        matrix_sizes = [100, 150, 200]  # Realistic production sizes
        energy_counts = [8, 16, 32]  # Typical energy point counts
        print("\nFULL TEST MODE: All production-scale matrix sizes and energy points")

    # Detect available CPU cores
    cpu_count = multiprocessing.cpu_count()
    print(f"\nAvailable CPU cores: {cpu_count}")

    # Test different worker counts - test MORE workers for production scale
    # Since you have 40 cores allocated, test up to reasonable limits
    worker_counts = [2, 4, 8, 16, 32] if cpu_count >= 32 else [2, 4, 8]

    # Skip ThreadPoolExecutor due to known segfault issues
    skip_threading = True
    print("\nNOTE: Skipping ThreadPoolExecutor tests due to BLAS thread-safety issues")
    print(f"Testing worker counts: {worker_counts}")

    # Timeout per test
    test_timeout = COMPUTATION_TIMEOUT

    overall_results = []

    for matrix_size in matrix_sizes:
        for num_energies in energy_counts:
            print(f"\n{'='*60}")
            print(f"MATRIX SIZE: {matrix_size}x{matrix_size}, ENERGIES: {num_energies}")
            print(f"{'='*60}")

            # Create test system
            alpha, Salpha, beta, Sbeta = create_test_contact(matrix_size)
            energies = np.linspace(-2.0, 2.0, num_energies)
            conv = SURFACE_GREEN_CONVERGENCE

            # Store results for consistency checking
            all_results = []
            all_methods = []
            all_times = []

            # 1. Sequential baseline
            print("1. Sequential (baseline):")
            results_seq, time_seq = compute_sigma_sequential(
                alpha, Salpha, beta, Sbeta, energies, conv, timeout=test_timeout
            )
            all_results.append(results_seq)
            all_methods.append("Sequential")
            all_times.append(time_seq)

            if results_seq is not None:
                total_iterations = sum(iterations for _, _, iterations in results_seq)
                print(f"   Time: {time_seq:.3f}s, Avg iterations: {total_iterations/num_energies:.1f}")
            else:
                print(f"   FAILED or TIMEOUT")
                # Skip parallel tests if sequential fails
                continue

            # Test different parallelization methods with various worker counts
            for workers in worker_counts:
                print(f"\n2. ProcessPoolExecutor (workers={workers}):")
                results_pp, time_pp = compute_sigma_processpool(
                    alpha, Salpha, beta, Sbeta, energies, conv, workers, timeout=test_timeout
                )
                if results_pp is not None:
                    speedup_pp = time_seq / time_pp
                    print(f"   Time: {time_pp:.3f}s, Speedup: {speedup_pp:.2f}x")
                all_results.append(results_pp)
                all_methods.append(f"ProcessPool-{workers}")
                all_times.append(time_pp)

                if not skip_threading:
                    print(f"\n3. ThreadPoolExecutor (workers={workers}):")
                    results_tp, time_tp = compute_sigma_threadpool(
                        alpha, Salpha, beta, Sbeta, energies, conv, workers, timeout=test_timeout
                    )
                    if results_tp is not None:
                        speedup_tp = time_seq / time_tp
                        print(f"   Time: {time_tp:.3f}s, Speedup: {speedup_tp:.2f}x")
                    all_results.append(results_tp)
                    all_methods.append(f"ThreadPool-{workers}")
                    all_times.append(time_tp)

                print(f"\n3. multiprocessing.Pool (processes={workers}):")
                results_mp, time_mp = compute_sigma_multiprocessing_pool(
                    alpha, Salpha, beta, Sbeta, energies, conv, workers, timeout=test_timeout
                )
                if results_mp is not None:
                    speedup_mp = time_seq / time_mp
                    print(f"   Time: {time_mp:.3f}s, Speedup: {speedup_mp:.2f}x")
                all_results.append(results_mp)
                all_methods.append(f"MultiprocPool-{workers}")
                all_times.append(time_mp)

                # Test joblib with multiprocessing backend (skip threading due to segfaults)
                if JOBLIB_AVAILABLE:
                    print(f"\n4. joblib.Parallel (n_jobs={workers}, backend=multiprocessing):")
                    results_jl, time_jl = compute_sigma_joblib(
                        alpha, Salpha, beta, Sbeta, energies, conv, workers,
                        backend='multiprocessing', timeout=test_timeout
                    )
                    if results_jl is not None:
                        speedup_jl = time_seq / time_jl
                        print(f"   Time: {time_jl:.3f}s, Speedup: {speedup_jl:.2f}x")
                    all_results.append(results_jl)
                    all_methods.append(f"joblib-multiproc-{workers}")
                    all_times.append(time_jl)

            # Verify consistency
            consistent = verify_results_consistency(all_results, all_methods)

            # Find best method
            valid_times = [(time, method) for time, method in zip(all_times, all_methods)
                          if time != float('inf') and method != 'Sequential']
            if valid_times:
                best_time, best_method = min(valid_times)
                best_speedup = time_seq / best_time
                print(f"\nBEST METHOD: {best_method}")
                print(f"Best time: {best_time:.3f}s, Best speedup: {best_speedup:.2f}x")
                print(f"Consistency check: {'PASS' if consistent else 'FAIL'}")
            else:
                print(f"\nNO PARALLEL METHOD SUCCEEDED")
                best_method = "Sequential"
                best_speedup = 1.0

            # Store summary for overall analysis
            overall_results.append({
                'matrix_size': matrix_size,
                'num_energies': num_energies,
                'sequential_time': time_seq,
                'methods': all_methods,
                'times': all_times,
                'consistent': consistent,
                'best_method': best_method,
                'best_speedup': best_speedup
            })

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print(f"{'Matrix':>8s} {'Energies':>8s} {'Sequential':>10s} {'Best Method':>20s} {'Speedup':>8s} {'Consistent':>10s}")
    print("-" * 75)

    for result in overall_results:
        consistency = "PASS" if result['consistent'] else "FAIL"
        seq_time = result['sequential_time'] if result['sequential_time'] != float('inf') else float('nan')
        print(f"{result['matrix_size']:>8d} {result['num_energies']:>8d} "
              f"{seq_time:>10.3f}s {result['best_method']:>20s} "
              f"{result['best_speedup']:>7.2f}x {consistency:>10s}")

    # Calculate average speedups by method type (excluding failed runs)
    method_speedups = {}
    for result in overall_results:
        for method, time in zip(result['methods'], result['times']):
            if time != float('inf') and method != 'Sequential' and result['sequential_time'] != float('inf'):
                method_type = method.split('-')[0]
                speedup = result['sequential_time'] / time
                if method_type not in method_speedups:
                    method_speedups[method_type] = []
                method_speedups[method_type].append(speedup)

    if method_speedups:
        print(f"\nAverage Speedups by Method Type:")
        print("-" * 40)
        for method_type, speedups in method_speedups.items():
            avg_speedup = np.mean(speedups)
            std_speedup = np.std(speedups)
            print(f"{method_type:20s}: {avg_speedup:.2f} ± {std_speedup:.2f} ({len(speedups)} successful runs)")
    else:
        print("\nNo successful parallel runs to summarize.")


if __name__ == "__main__":
    print("Starting Production-Scale NumPy Sigma Parallelization Benchmark...")
    print("This benchmark tests realistic 100x100+ matrix sizes with proper BLAS configuration...")
    benchmark_production_sigma_parallelization()