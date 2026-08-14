#!/usr/bin/env python3
"""
Test BLAS Threading Within Multiprocessing Workers

Verifies that NumPy BLAS threading works correctly within separate processes
and measures performance scaling.
"""

import os
import sys
import time
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def blas_worker(args):
    """Worker that tests BLAS threading within a single process."""
    blas_threads, matrix_size, iterations = args

    # Set BLAS threading for THIS process
    os.environ['OMP_NUM_THREADS'] = str(blas_threads)
    os.environ['MKL_NUM_THREADS'] = str(blas_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
    os.environ['BLIS_NUM_THREADS'] = str(blas_threads)

    # Force reload of BLAS settings (import numpy after setting env vars)
    import numpy as np

    # Create test matrices
    np.random.seed(42)  # Consistent results
    A = np.random.random((matrix_size, matrix_size)).astype(np.float64)
    B = np.random.random((matrix_size, matrix_size)).astype(np.float64)

    # Make A positive definite
    A = A @ A.T + np.eye(matrix_size) * 0.01

    results = []

    for i in range(iterations):
        # Test different BLAS operations
        start = time.perf_counter()

        # Matrix multiplication (Level 3 BLAS)
        C = A @ B

        # Cholesky decomposition (should benefit from threading)
        L = np.linalg.cholesky(A)

        # Eigenvalue decomposition (computationally intensive)
        eigs = np.linalg.eigvals(A)

        end = time.perf_counter()
        results.append(end - start)

    avg_time = np.mean(results)
    std_time = np.std(results)

    return {
        'blas_threads': blas_threads,
        'process_id': os.getpid(),
        'avg_time': avg_time,
        'std_time': std_time,
        'env_omp': os.environ.get('OMP_NUM_THREADS'),
        'iterations': len(results)
    }

def test_single_process_scaling():
    """Test BLAS threading scaling within a single process."""
    print("Testing BLAS scaling within single process...")
    print("-" * 50)

    matrix_size = 1500
    iterations = 3
    thread_counts = [1, 2, 4, 8, 16]

    single_process_results = []

    for threads in thread_counts:
        print(f"Testing {threads} BLAS threads...")

        # Run in current process
        result = blas_worker((threads, matrix_size, iterations))
        single_process_results.append(result)

        print(f"  Time: {result['avg_time']:.3f}±{result['std_time']:.3f}s")

    return single_process_results

def test_multiprocess_blas():
    """Test BLAS threading across multiple processes."""
    print("\nTesting BLAS threading across multiple processes...")
    print("-" * 50)

    matrix_size = 1000
    iterations = 2

    # Test configurations: (num_processes, blas_threads_per_process)
    configs = [
        (1, 8),  # Single process, 8 BLAS threads
        (2, 4),  # Two processes, 4 BLAS threads each
        (4, 2),  # Four processes, 2 BLAS threads each
        (8, 1),  # Eight processes, 1 BLAS thread each
    ]

    multiprocess_results = []

    for num_processes, blas_threads in configs:
        print(f"Testing {num_processes} processes x {blas_threads} BLAS threads...")

        # Create work for each process
        work_args = [(blas_threads, matrix_size, iterations) for _ in range(num_processes)]

        start_total = time.perf_counter()

        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                process_results = list(executor.map(blas_worker, work_args))

            end_total = time.perf_counter()

            # Calculate statistics
            total_time = end_total - start_total
            avg_worker_time = np.mean([r['avg_time'] for r in process_results])
            max_worker_time = max([r['avg_time'] for r in process_results])

            result = {
                'num_processes': num_processes,
                'blas_threads': blas_threads,
                'total_time': total_time,
                'avg_worker_time': avg_worker_time,
                'max_worker_time': max_worker_time,
                'success': True
            }

            multiprocess_results.append(result)

            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg worker time: {avg_worker_time:.3f}s")
            print(f"  Max worker time: {max_worker_time:.3f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            multiprocess_results.append({
                'num_processes': num_processes,
                'blas_threads': blas_threads,
                'success': False,
                'error': str(e)
            })

    return multiprocess_results

def main():
    print("=" * 60)
    print("BLAS THREADING IN MULTIPROCESSING TEST")
    print("=" * 60)
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print(f"NumPy version: {np.__version__}")

    # Test 1: Single process BLAS scaling
    single_results = test_single_process_scaling()

    # Test 2: Multiprocess BLAS
    multi_results = test_multiprocess_blas()

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Single process scaling analysis
    print("Single Process BLAS Scaling:")
    baseline = single_results[0]['avg_time'] if single_results else 0
    for result in single_results:
        speedup = baseline / result['avg_time'] if result['avg_time'] > 0 else 0
        efficiency = speedup / result['blas_threads'] if result['blas_threads'] > 0 else 0
        print(f"  {result['blas_threads']} threads: {speedup:.2f}x speedup, {efficiency:.2f} efficiency")

    print()

    # Multiprocess scaling analysis
    print("Multiprocess Configurations:")
    for result in multi_results:
        if result['success']:
            total_threads = result['num_processes'] * result['blas_threads']
            print(f"  {result['num_processes']}p x {result['blas_threads']}t = {total_threads} total: {result['total_time']:.3f}s")

    # Find optimal configuration
    successful_multi = [r for r in multi_results if r['success']]
    if successful_multi:
        fastest = min(successful_multi, key=lambda x: x['total_time'])
        print()
        print("OPTIMAL MULTIPROCESSING CONFIGURATION:")
        print(f"  Processes: {fastest['num_processes']}")
        print(f"  BLAS threads per process: {fastest['blas_threads']}")
        print(f"  Total time: {fastest['total_time']:.3f}s")

if __name__ == "__main__":
    main()
