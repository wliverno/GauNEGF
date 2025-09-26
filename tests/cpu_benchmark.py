#!/usr/bin/env python3
"""
CPU benchmark comparing SciPy BLAS operations with proper threading.
Tests SciPy's NO_AFFINITY OpenBLAS for SLURM compatibility.
"""

import os
import sys
import time
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Force spawn method for true process isolation
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)


def worker_eig(args):
    """Worker for eigenvalue computation."""
    size, blas_threads = args

    # Clear inherited threading restrictions
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if key in os.environ:
            del os.environ[key]

    # Set threading before numpy import
    os.environ['OMP_NUM_THREADS'] = str(blas_threads)
    os.environ['MKL_NUM_THREADS'] = str(blas_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
    os.environ['BLIS_NUM_THREADS'] = str(blas_threads)

    import numpy as np
    import scipy.linalg as LA
    np.random.seed(42)

    A = np.random.random((size, size)).astype(np.float64)
    A = A @ A.T + np.eye(size) * 0.01  # Symmetric positive definite

    start = time.perf_counter()
    eigenvals = LA.eigvals(A)  # Use SciPy BLAS (NO_AFFINITY)
    return time.perf_counter() - start


def worker_inv(args):
    """Worker for matrix inversion using SciPy BLAS."""
    size, blas_threads = args

    # Clear inherited threading restrictions
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if key in os.environ:
            del os.environ[key]

    # Set threading before numpy import
    os.environ['OMP_NUM_THREADS'] = str(blas_threads)
    os.environ['MKL_NUM_THREADS'] = str(blas_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
    os.environ['BLIS_NUM_THREADS'] = str(blas_threads)

    import numpy as np
    import scipy.linalg as LA
    np.random.seed(42)

    A = np.random.random((size, size)).astype(np.float64)
    A = A @ A.T + np.eye(size) * 0.01  # Symmetric positive definite

    start = time.perf_counter()
    A_inv = LA.inv(A)  # Use SciPy BLAS (NO_AFFINITY)
    return time.perf_counter() - start


def worker_solve(args):
    """Worker for linear system solving using SciPy BLAS."""
    size, blas_threads = args

    # Clear inherited threading restrictions
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if key in os.environ:
            del os.environ[key]

    # Set threading before numpy import
    os.environ['OMP_NUM_THREADS'] = str(blas_threads)
    os.environ['MKL_NUM_THREADS'] = str(blas_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
    os.environ['BLIS_NUM_THREADS'] = str(blas_threads)

    import numpy as np
    import scipy.linalg as LA
    np.random.seed(42)

    A = np.random.random((size, size)).astype(np.float64)
    A = A @ A.T + np.eye(size) * 0.01  # Symmetric positive definite
    b = np.random.random(size).astype(np.float64)

    start = time.perf_counter()
    x = LA.solve(A, b)  # Use SciPy BLAS (NO_AFFINITY)
    return time.perf_counter() - start


def test_operation(operation, size, num_processes, blas_threads):
    """Test a specific operation configuration."""
    worker_func = {
        'eig': worker_eig,
        'inv': worker_inv,
        'solve': worker_solve
    }[operation]

    work_args = [(size, blas_threads) for _ in range(num_processes)]

    start = time.perf_counter()

    if num_processes == 1:
        results = [worker_func(work_args[0])]
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(worker_func, work_args))

    total_time = time.perf_counter() - start
    return total_time


def main():
    size = 1000
    operations = ['eig', 'inv', 'solve']

    print("=" * 80)
    print("SCIPY BLAS OPERATIONS COMPARISON (NO_AFFINITY)")
    print("=" * 80)
    print(f"Matrix size: {size}x{size}")
    print(f"Parent OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print()

    # Test configurations: (processes, blas_threads_per_process)
    configs = [
        (1, 1),   # Single-threaded baseline
        (1, 4),   # Single process, 4 BLAS threads
        (1, 8),   # Single process, 8 BLAS threads
        (2, 4),   # 2 processes, 4 BLAS threads each
        (4, 2),   # 4 processes, 2 BLAS threads each
    ]

    for operation in operations:
        print(f"\n{operation.upper()} OPERATION:")
        print(f"{'Config':<12} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 40)

        baseline = None
        results = []

        for num_processes, blas_threads in configs:
            config_name = f"{num_processes}p x {blas_threads}t"

            try:
                total_time = test_operation(operation, size, num_processes, blas_threads)

                if baseline is None:
                    baseline = total_time
                    speedup = 1.0
                else:
                    speedup = baseline / total_time

                results.append((config_name, total_time, speedup))
                print(f"{config_name:<12} {total_time:<12.3f} {speedup:<10.2f}x")

            except Exception as e:
                print(f"{config_name:<12} FAILED: {e}")

        # Find best config for this operation
        if results:
            best = min(results, key=lambda x: x[1])
            print(f"BEST: {best[0]} ({best[1]:.3f}s, {best[2]:.2f}x speedup)")


if __name__ == "__main__":
    main()