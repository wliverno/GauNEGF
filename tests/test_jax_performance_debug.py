#!/usr/bin/env python3
"""
JAX Parallelization Strategy Benchmark

Compares vmap, pmap, and the parallelize.py worker system for NEGF calculations
using physically realistic matrices and actual surfG methods.

Tests multiple matrix sizes to determine optimal parallelization approach.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import time
from jax import jit, vmap, pmap
from typing import Dict, Tuple

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gauNEGF.surfG1D import surfG
from gauNEGF.parallelize import parallelize_energy_calculation

# =============================================================================
# TEST MATRIX CREATION - PHYSICALLY REALISTIC
# =============================================================================

def create_physical_chain(N, coupling=-0.1, overlap=0.1):
    """
    Create physically realistic tridiagonal Fock and overlap matrices.

    Parameters
    ----------
    N : int
        Number of orbitals in the chain
    coupling : float
        Off-diagonal coupling strength (eV)
    overlap : float
        Off-diagonal overlap

    Returns
    -------
    F : ndarray (N, N)
        Fock matrix (tridiagonal)
    S : ndarray (N, N)
        Overlap matrix (tridiagonal)
    """
    # Fock matrix: diagonal=0, off-diagonal=coupling
    F = np.zeros((N, N))
    for i in range(N-1):
        F[i, i+1] = coupling
        F[i+1, i] = coupling

    # Overlap matrix: identity + off-diagonal overlap
    S = np.eye(N)
    for i in range(N-1):
        S[i, i+1] = overlap
        S[i+1, i] = overlap

    return F, S


def create_test_system(N):
    """
    Create test system with realistic F, S, and surfG object.

    Parameters
    ----------
    N : int
        Total number of orbitals (must be even)

    Returns
    -------
    F : ndarray (N, N)
        Fock matrix
    S : ndarray (N, N)
        Overlap matrix
    g : surfG
        Surface Green's function connecting first N/2 to second N/2
    """
    assert N % 2 == 0, "N must be even for splitting into two halves"

    # Create physical chain - keep as NumPy arrays
    F, S = create_physical_chain(N)

    # Create surfG using first half connected to second half
    # Extract blocks for the two regions
    half = N // 2

    # For surfG: first half of orbitals = contact regions
    # Pattern (b): surfG(F, S, indsList, [tau1, tau2], [stau1, stau2])

    # Contact indices in the full system (where contacts connect)
    left_contact = list(range(half))
    right_contact = list(range(half, N))

    # Create surfG with coupling matrices (pattern b)
    # Methods are already JIT-compiled in surfG1D.py
    g = surfG(F, S, [left_contact, right_contact], eta=1e-3)

    return F, S, g


# =============================================================================
# IMPLEMENTATION: SEQUENTIAL BASELINE
# =============================================================================

def sequential_integration(F, S, g, Elist, weights):
    """
    Sequential baseline: compute everything in a simple loop.

    No parallelization, but g.sigma() and g.g() are JIT-compiled internally
    with static_argnums for the contact index, so the expensive iterative
    calculation is optimized.

    Parameters
    ----------
    F : ndarray (N, N)
        Fock matrix
    S : ndarray (N, N)
        Overlap matrix
    g : surfG
        Surface Green's function object (with JIT-compiled methods)
    Elist : ndarray (M,)
        Energy points
    weights : ndarray (M,)
        Integration weights

    Returns
    -------
    Gr : ndarray (N, N)
        Integrated retarded Green's function
    """
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)

    result = np.zeros_like(F, dtype=complex)

    # Sequential loop - but g.sigmaTot(), g.sigma(), g.g() are JIT-compiled
    for E, w in zip(Elist, weights):
        sigma_total = g.sigmaTot(E)  # JIT-compiled internally
        mat = E * S_jax - F_jax - sigma_total
        Gr = jnp.linalg.inv(mat)
        result += w * np.array(Gr)

    return result


# =============================================================================
# IMPLEMENTATION: PARALLELIZE.PY WORKER SYSTEM
# =============================================================================

def worker_integration(F, S, g, Elist, weights):
    """
    Integration using parallelize.py worker queue system.

    Parameters
    ----------
    F : ndarray (N, N)
        Fock matrix
    S : ndarray (N, N)
        Overlap matrix
    g : surfG
        Surface Green's function object
    Elist : ndarray (M,)
        Energy points
    weights : ndarray (M,)
        Integration weights

    Returns
    -------
    Gr : ndarray (N, N)
        Integrated retarded Green's function
    """
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)

    def single_energy_calc(E, F, S, g):
        """Function for single energy point - returns unweighted Gr."""
        sigma_total = g.sigmaTot(E)
        mat = E * S - F - sigma_total
        Gr = jnp.linalg.inv(mat)
        return Gr

    # Use parallelize system
    results_dict = parallelize_energy_calculation(
        Elist,
        single_energy_calc,
        matrix_size=F.shape[0],
        F=F_jax,
        S=S_jax,
        g=g
    )

    # Sum weighted results from all energy points
    result = np.zeros_like(F, dtype=complex)
    for idx in sorted(results_dict.keys()):
        result += weights[idx] * results_dict[idx]['result']

    return result


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_method(method_name, method_func, F, S, g, Elist, weights, num_runs=3):
    """
    Benchmark a single integration method.

    Parameters
    ----------
    method_name : str
        Name of method for display
    method_func : callable
        Function to benchmark
    F, S, g, Elist, weights
        Test system parameters
    num_runs : int
        Number of runs for averaging

    Returns
    -------
    dict
        Timing statistics and result
    """
    print(f"\n{method_name}:")

    times = []
    result = None

    for run in range(num_runs):
        try:
            start = time.perf_counter()
            result = method_func(F, S, g, Elist, weights)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.4f}s")

        except Exception as e:
            print(f"  Run {run+1}: FAILED - {e}")
            times.append(float('inf'))

    valid_times = [t for t in times if t != float('inf')]

    if valid_times:
        avg_time = np.mean(valid_times)
        std_time = np.std(valid_times)
        min_time = np.min(valid_times)

        print(f"  Average: {avg_time:.4f}s +/- {std_time:.4f}s")

        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'times': times,
            'result': result
        }
    else:
        print(f"  All runs failed!")
        return {
            'avg_time': float('inf'),
            'std_time': 0,
            'min_time': float('inf'),
            'times': times,
            'result': None
        }


def run_benchmark_suite():
    """
    Run comprehensive benchmark comparing all methods across multiple sizes.
    """
    print("=" * 80)
    print("JAX PARALLELIZATION STRATEGY BENCHMARK")
    print("=" * 80)

    # Print JAX configuration
    print(f"\nJAX Version: {jax.__version__}")
    print(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX Devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")

    # Test configurations: (matrix_size, num_energies)
    test_cases = [
        (100, 20, "Small: 100x100, 20 energies"),
        (500, 50, "Medium: 500x500, 50 energies"),
        (1000, 100, "Large: 1000x1000, 100 energies"),
        (2000, 200, "X-Large: 2000x2000, 200 energies"),
        (3000, 300, "XX-Large: 3000x3000, 300 energies"),
    ]

    all_results = {}

    for matrix_size, num_energies, test_name in test_cases:
        print(f"\n{'=' * 80}")
        print(f"TEST: {test_name}")
        print(f"{'=' * 80}")

        # Create test system
        print(f"\nCreating test system...")
        F, S, g = create_test_system(matrix_size)

        # Energy grid
        Elist = np.linspace(-1.0, 1.0, num_energies) + 1j * 0.01
        dE = Elist[1] - Elist[0]
        weights = np.ones(num_energies, dtype=complex) * dE

        print(f"Matrix size: {matrix_size}x{matrix_size}")
        print(f"Energy points: {num_energies}")
        print(f"Energy range: {Elist[0].real:.2f} to {Elist[-1].real:.2f} eV")

        # Benchmark each method
        methods = [
            ("Sequential (JIT)", sequential_integration),
            ("Worker Queue", worker_integration),
        ]

        test_results = {}
        baseline_result = None

        for method_name, method_func in methods:
            stats = benchmark_method(method_name, method_func, F, S, g, Elist, weights)
            test_results[method_name] = stats

            # Use first successful result as baseline for accuracy check
            if baseline_result is None and stats['result'] is not None:
                baseline_result = stats['result']
            elif stats['result'] is not None:
                error = np.max(np.abs(stats['result'] - baseline_result))
                print(f"  Max error vs baseline: {error:.2e}")
                stats['error'] = error

        all_results[test_name] = test_results

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    for test_name, results in all_results.items():
        print(f"\n{test_name}:")

        # Find fastest time for speedup calculation
        times = {name: stats['avg_time'] for name, stats in results.items()}
        fastest_time = min(t for t in times.values() if t != float('inf'))

        # Sort by time
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_time'])

        for method_name, stats in sorted_methods:
            avg_time = stats['avg_time']

            if avg_time == float('inf'):
                speedup_str = "FAILED"
            else:
                speedup = fastest_time / avg_time
                if speedup >= 1.0:
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = f"{1/speedup:.2f}x slower"

            print(f"  {method_name:20s}: {avg_time:8.4f}s ({speedup_str})")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("Based on these results:")
    print("1. Choose the fastest method for your typical problem size")
    print("2. Consider memory usage for large systems")
    print("3. pmap benefits from multiple devices (CPUs/GPUs)")
    print("4. Worker queue is most flexible for heterogeneous workloads")


if __name__ == "__main__":
    run_benchmark_suite()
