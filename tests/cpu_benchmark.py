#!/usr/bin/env python3
"""
CPU Linear Algebra Benchmark for NEGF-related Operations (eig, inv, solve)

Tests eig(), inv(), and solve() with different thread counts using subprocess approach.
"""

import os
import sys
import time
import subprocess
import tempfile
import argparse
import numpy as np


def create_benchmark_script() -> str:
    """Create the benchmark code that runs in subprocess."""
    return '''
import os
import time
import numpy as np

def create_test_matrices(size: int):
    """Create matrices for NEGF-like problems."""
    np.random.seed(42)

    # Hamiltonian-like matrix (Hermitian)
    H = np.random.random((size, size)) + 1j * np.random.random((size, size))
    H = (H + H.conj().T) / 2
    H += np.eye(size) * 0.1

    # Overlap-like matrix (Hermitian positive definite)
    S_real = np.random.random((size, size)) * 0.1
    S = (S_real + S_real.T) / 2
    S += np.eye(size) * 1.0

    return H.astype(np.complex128), S.astype(np.complex128)

def time_operation(func, iterations: int) -> float:
    """Time an operation over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)

def benchmark_operations(size: int, iterations: int):
    """Benchmark eig, inv, solve operations."""
    H, S = create_test_matrices(size)
    I = np.eye(size, dtype=np.complex128)
    z = 1.0 + 0.0001j
    A = z * S - H  # Typical NEGF linear system

    # Warmup
    _ = np.linalg.eigvals(H)
    _ = np.linalg.inv(A)
    _ = np.linalg.solve(A, I)

    # Benchmark operations
    eig_time, eig_std = time_operation(lambda: np.linalg.eigvals(H), iterations)
    inv_time, inv_std = time_operation(lambda: np.linalg.inv(A), iterations)
    solve_time, solve_std = time_operation(lambda: np.linalg.solve(A, I), iterations)

    return eig_time, eig_std, inv_time, inv_std, solve_time, solve_std

if __name__ == "__main__":
    import sys
    size = int(sys.argv[1])
    iterations = int(sys.argv[2])

    threads = os.environ.get("OMP_NUM_THREADS", "unknown")
    print(f"THREADS:{threads}")

    eig_t, eig_std, inv_t, inv_std, solve_t, solve_std = benchmark_operations(size, iterations)
    print(f"RESULTS:{eig_t:.3e}+/-{eig_std:.3e},{inv_t:.3e}+/-{inv_std:.3e},{solve_t:.3e}+/-{solve_std:.3e}")
'''


def run_benchmark_subprocess(threads: int, size: int, iterations: int):
    """Run benchmark in subprocess with specific thread count."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(create_benchmark_script())
        script_path = f.name

    try:
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "BLIS_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            "MKL_DYNAMIC": "FALSE",
            "OMP_DYNAMIC": "FALSE",
        })

        # Try to remove CPU affinity restrictions for better threading
        try:
            # Get available cores and set affinity to allow using more cores
            available_cores = 0#len(os.sched_getaffinity(0))
            max_cores = min(threads, available_cores)
            core_list = list(range(max_cores))

            # Use taskset if available (more reliable on HPC systems)
            result = subprocess.run(
                ["taskset", "-c", ",".join(map(str, core_list)),
                 sys.executable, script_path, str(size), str(iterations)],
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
        except (FileNotFoundError, OSError):
            # Fallback to normal subprocess if taskset not available
            result = subprocess.run(
                [sys.executable, script_path, str(size), str(iterations)],
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None, None, None

        # Parse results
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith("RESULTS:"):
                times = line.split(":")[1].split(",")
                return float(times[0].split("+/-")[0]), float(times[0].split("+/-")[1]), float(times[1].split("+/-")[0]), float(times[1].split("+/-")[1]), float(times[2].split("+/-")[0]), float(times[2].split("+/-")[1])

        return None, None, None

    finally:
        try:
            os.unlink(script_path)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="CPU benchmark for eig, inv, solve")
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 3000],
                       help="Matrix sizes to test")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                       help="Thread counts to test")
    parser.add_argument("--iters", type=int, default=3,
                       help="Iterations per test")

    args = parser.parse_args()

    print("=" * 60)
    print("CPU LINEAR ALGEBRA BENCHMARK")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print()

    for size in args.sizes:
        print(f"MATRIX SIZE: {size} x {size}")
        print("-" * 70)
        print(f"{'Threads':<8} {'eig (std)':<20} {'inv (std)':<20} {'solve (std)':<20}")
        print("-" * 70)

        for threads in args.threads:
            eig_time, eig_std, inv_time, inv_std, solve_time, solve_std = run_benchmark_subprocess(threads, size, args.iters)

            if eig_time is not None:
                print(f"{threads:<8} {eig_time:<10.3e}({eig_std:<8.2e}) {inv_time:<10.3e}({inv_std:<8.2e}) {solve_time:<10.3e}({solve_std:<8.2e})")
            else:
                print(f"{threads:<8} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10}")

        print()


if __name__ == "__main__":
    main()