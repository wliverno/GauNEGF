#!/usr/bin/env python3
"""
Comprehensive JAX Performance Diagnostic Test Suite

This test suite diagnoses why JAX is performing poorly compared to NumPy
on HPC systems. It tests multiple implementation approaches and provides
detailed timing and configuration analysis.

Run this on your HPC cluster to identify the performance bottleneck.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import time
import tracemalloc
import psutil
from jax import jit, vmap
from typing import Dict, List, Tuple, Any

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gauNEGF.integrate import GrInt, GrLessInt

class MockSurfaceGreen:
    """Mock surface Green's function for testing."""
    def __init__(self, size):
        self.size = size
        np.random.seed(42)  # Fixed seed for reproducible results
        self._sigma_base = np.random.randn(size, size) * 0.1
        self._sigma_base = (self._sigma_base + self._sigma_base.T) / 2  # Make Hermitian
        self._sigma_contacts = [
            np.random.randn(size, size) * 0.05 for _ in range(2)
        ]
        # Make contact sigmas Hermitian too
        for i in range(len(self._sigma_contacts)):
            self._sigma_contacts[i] = (self._sigma_contacts[i] + self._sigma_contacts[i].T) / 2

    def sigmaTot(self, E):
        """Total self-energy with small imaginary part."""
        return self._sigma_base + 1j * (0.01 * E * np.eye(self.size) + 0.001)

    def sigma(self, E, ind):
        """Contact-specific self-energy."""
        if ind >= len(self._sigma_contacts):
            ind = 0
        return self._sigma_contacts[ind] + 1j * (0.005 * E * np.eye(self.size) + 0.001)

def print_jax_diagnostics():
    """Print comprehensive JAX configuration information."""
    print("=" * 80)
    print("JAX CONFIGURATION DIAGNOSTICS")
    print("=" * 80)

    try:
        print(f"JAX Version: {jax.__version__}")
        print(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")
        print(f"Device count: {jax.device_count()}")
        print(f"Local device count: {jax.local_device_count()}")
        print(f"Devices: {jax.devices()}")
        print(f"Local devices: {jax.local_devices()}")

        # Test if JAX can actually use multiple cores
        print(f"Process count: {jax.process_count()}")
        print(f"Process index: {jax.process_index()}")

        # Check XLA flags
        import os
        xla_flags = os.environ.get('XLA_FLAGS', 'Not set')
        print(f"XLA_FLAGS: {xla_flags}")

        # Check CPU info
        print(f"System CPU count: {psutil.cpu_count()}")
        print(f"System CPU count (logical): {psutil.cpu_count(logical=True)}")

        # Test simple parallel operation
        print("\nTesting JAX parallelization:")
        test_array = jnp.arange(1000000)
        start = time.perf_counter()
        result = jnp.sum(test_array)
        elapsed = time.perf_counter() - start
        print(f"JAX sum of 1M elements: {elapsed:.6f}s, result: {result}")

        # Test vmap
        def test_func(x):
            return jnp.sum(x**2)

        test_vmap = jax.vmap(test_func)
        test_data = jnp.ones((100, 1000))
        start = time.perf_counter()
        vmap_result = test_vmap(test_data)
        elapsed = time.perf_counter() - start
        print(f"JAX vmap test (100x1000): {elapsed:.6f}s")

    except Exception as e:
        print(f"ERROR in JAX diagnostics: {e}")
        traceback.print_exc()

    print("=" * 80)

def create_test_matrices(size):
    """Create proper test matrices with specified constraints."""
    np.random.seed(42)  # Fixed seed for reproducibility

    # Create Fock matrix F - real and Hermitian
    F_temp = np.random.randn(size, size) * 0.1
    F = (F_temp + F_temp.T) / 2  # Make Hermitian

    # Create overlap matrix S - real Hermitian with specific structure
    S = np.eye(size)
    off_diag = np.random.rand(size, size) * 0.2  # Random values [0, 0.2]
    off_diag = (off_diag + off_diag.T) / 2  # Make symmetric
    np.fill_diagonal(off_diag, 0)  # Zero diagonal
    S += off_diag  # Add to identity matrix

    return F, S

# =============================================================================
# IMPLEMENTATION VARIANTS TO TEST
# =============================================================================

def numpy_serial_gr(F, S, g, Elist, weights):
    """Pure NumPy serial implementation - baseline."""
    result = np.zeros_like(F, dtype=complex)

    for E, w in zip(Elist, weights):
        sigma_total = g.sigmaTot(E)
        mat = E * S - F - sigma_total
        result += w * np.linalg.inv(mat)

    return result

def jax_serial_no_jit(F, S, g, Elist, weights):
    """JAX without JIT compilation - tests JAX overhead without compilation."""
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    result = jnp.zeros_like(F_jax, dtype=complex)

    for E, w in zip(Elist, weights):
        sigma_total = jnp.array(g.sigmaTot(E))
        mat = E * S_jax - F_jax - sigma_total
        result += w * jnp.linalg.inv(mat)

    return np.array(result)

def _jax_matrix_ops_no_jit(E, F, S, sigma_total, weight):
    """JAX matrix operations without JIT."""
    mat = E * S - F - sigma_total
    return weight * jnp.linalg.inv(mat)

def jax_vmap_no_jit(F, S, g, Elist, weights):
    """JAX vmap without JIT compilation."""
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Pre-compute sigma values
    sigma_list = []
    for E in Elist:
        sigma_total = g.sigmaTot(E)
        sigma_list.append(sigma_total)
    sigma_jax = jnp.array(sigma_list)

    # Use vmap without JIT
    vmap_func = vmap(_jax_matrix_ops_no_jit, in_axes=(0, None, None, 0, 0))
    weighted_grs = vmap_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
    return np.array(jnp.sum(weighted_grs, axis=0))

@jit
def _jax_matrix_ops_jit(E, F, S, sigma_total, weight):
    """JIT compiled JAX matrix operations."""
    mat = E * S - F - sigma_total
    return weight * jnp.linalg.inv(mat)

def jax_vmap_with_jit(F, S, g, Elist, weights):
    """JAX vmap with JIT compilation."""
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Pre-compute sigma values
    sigma_list = []
    for E in Elist:
        sigma_total = g.sigmaTot(E)
        sigma_list.append(sigma_total)
    sigma_jax = jnp.array(sigma_list)

    # Use vmap with JIT
    vmap_func = jit(vmap(_jax_matrix_ops_jit, in_axes=(0, None, None, 0, 0)))
    weighted_grs = vmap_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
    return np.array(jnp.sum(weighted_grs, axis=0))

def current_gr_int_forced_vmap(F, S, g, Elist, weights):
    """Current GrInt implementation forced to use vmap path."""
    import gauNEGF.integrate as integrate_module

    orig_threshold = integrate_module.SMALL_MATRIX_THRESHOLD
    orig_memory = integrate_module.MAX_VMAP_MEMORY_GB

    try:
        integrate_module.SMALL_MATRIX_THRESHOLD = 10000
        integrate_module.MAX_VMAP_MEMORY_GB = 100.0
        result = GrInt(F, S, g, Elist, weights)
    finally:
        integrate_module.SMALL_MATRIX_THRESHOLD = orig_threshold
        integrate_module.MAX_VMAP_MEMORY_GB = orig_memory

    return result

def current_gr_int_forced_worker(F, S, g, Elist, weights):
    """Current GrInt implementation forced to use worker path."""
    import gauNEGF.integrate as integrate_module

    orig_threshold = integrate_module.SMALL_MATRIX_THRESHOLD
    orig_memory = integrate_module.MAX_VMAP_MEMORY_GB

    try:
        integrate_module.SMALL_MATRIX_THRESHOLD = 1
        integrate_module.MAX_VMAP_MEMORY_GB = 0.001
        result = GrInt(F, S, g, Elist, weights)
    finally:
        integrate_module.SMALL_MATRIX_THRESHOLD = orig_threshold
        integrate_module.MAX_VMAP_MEMORY_GB = orig_memory

    return result

# =============================================================================
# TIMING AND PROFILING UTILITIES
# =============================================================================

def time_with_memory(func, *args, **kwargs):
    """Time function execution and measure peak memory usage."""
    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Time execution
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    elapsed = end_time - start_time
    memory_delta = end_memory - start_memory
    peak_traced = peak / 1024 / 1024  # Convert to MB

    return result, elapsed, memory_delta, peak_traced

def run_performance_comparison(F, S, g, Elist, weights, test_name: str, num_runs: int = 3):
    """Run comprehensive performance comparison."""
    print(f"\n{test_name}")
    print("-" * len(test_name))

    methods = [
        ("NumPy Serial", numpy_serial_gr),
        ("JAX Serial No JIT", jax_serial_no_jit),
        ("JAX vmap No JIT", jax_vmap_no_jit),
        ("JAX vmap With JIT", jax_vmap_with_jit),
        ("Current vmap", current_gr_int_forced_vmap),
        ("Current Worker", current_gr_int_forced_worker),
    ]

    results = {}
    baseline_result = None

    for method_name, method_func in methods:
        print(f"\nTesting {method_name}:")

        times = []
        memory_deltas = []
        peak_memories = []
        errors = []

        for run in range(num_runs):
            try:
                result, elapsed, memory_delta, peak_memory = time_with_memory(
                    method_func, F, S, g, Elist, weights
                )

                times.append(elapsed)
                memory_deltas.append(memory_delta)
                peak_memories.append(peak_memory)

                # Check accuracy against baseline
                if baseline_result is None:
                    baseline_result = result
                    error = 0.0
                else:
                    error = np.max(np.abs(result - baseline_result))
                errors.append(error)

                print(f"  Run {run+1}: {elapsed:.4f}s, Memory: +{memory_delta:.1f}MB, Peak: {peak_memory:.1f}MB, Error: {error:.2e}")

            except Exception as e:
                print(f"  Run {run+1}: FAILED - {e}")
                times.append(float('inf'))
                memory_deltas.append(0)
                peak_memories.append(0)
                errors.append(float('inf'))

        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            min_time = np.min(valid_times)
            avg_memory = np.mean(memory_deltas)
            avg_peak = np.mean(peak_memories)
            max_error = np.max([e for e in errors if e != float('inf')])
        else:
            avg_time = std_time = min_time = avg_memory = avg_peak = max_error = float('inf')

        results[method_name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'avg_memory': avg_memory,
            'avg_peak': avg_peak,
            'max_error': max_error,
            'times': times
        }

        print(f"  Average: {avg_time:.4f}±{std_time:.4f}s, Memory: {avg_memory:.1f}MB, Error: {max_error:.2e}")

    return results

def print_performance_summary(all_results: Dict[str, Dict]):
    """Print comprehensive performance summary."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    for test_name, results in all_results.items():
        print(f"\n{test_name}:")

        # Find baseline (NumPy Serial) time
        baseline_time = results.get("NumPy Serial", {}).get('avg_time', 1.0)

        # Sort by average time
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_time'])

        for method_name, stats in sorted_methods:
            avg_time = stats['avg_time']
            max_error = stats['max_error']

            if avg_time == float('inf'):
                speedup_str = "FAILED"
            else:
                speedup = baseline_time / avg_time
                if speedup > 1:
                    speedup_str = f"{speedup:.1f}x faster"
                else:
                    speedup_str = f"{1/speedup:.1f}x slower"

            error_str = f"{max_error:.1e}" if max_error != float('inf') else "FAILED"

            print(f"  {method_name:20s}: {avg_time:8.4f}s ({speedup_str:12s}) Error: {error_str}")

def test_compilation_overhead():
    """Test JIT compilation overhead specifically."""
    print("\n" + "=" * 80)
    print("JIT COMPILATION OVERHEAD TEST")
    print("=" * 80)

    size = 100
    F, S = create_test_matrices(size)
    g = MockSurfaceGreen(size)
    Elist = np.linspace(-1.0, 1.0, 8) + 1j * 0.01
    weights = np.ones(8, dtype=complex) * (Elist[1] - Elist[0])

    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    Elist_jax = jnp.array(Elist)
    weights_jax = jnp.array(weights)

    # Pre-compute sigma values
    sigma_list = []
    for E in Elist:
        sigma_total = g.sigmaTot(E)
        sigma_list.append(sigma_total)
    sigma_jax = jnp.array(sigma_list)

    # Test compilation time
    print("Testing JIT compilation overhead:")

    @jit
    def test_func(Elist, F, S, sigma_list, weights):
        def matrix_ops(E, F, S, sigma_total, weight):
            mat = E * S - F - sigma_total
            return weight * jnp.linalg.inv(mat)

        vmap_func = vmap(matrix_ops, in_axes=(0, None, None, 0, 0))
        weighted_grs = vmap_func(Elist, F, S, sigma_list, weights)
        return jnp.sum(weighted_grs, axis=0)

    # First call - includes compilation
    start = time.perf_counter()
    result1 = test_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
    first_call_time = time.perf_counter() - start

    # Second call - should use cached compilation
    start = time.perf_counter()
    result2 = test_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
    second_call_time = time.perf_counter() - start

    # Third call - verify caching
    start = time.perf_counter()
    result3 = test_func(Elist_jax, F_jax, S_jax, sigma_jax, weights_jax)
    third_call_time = time.perf_counter() - start

    print(f"First call (with compilation):  {first_call_time:.4f}s")
    print(f"Second call (cached):           {second_call_time:.4f}s")
    print(f"Third call (cached):            {third_call_time:.4f}s")
    print(f"Compilation overhead:           {first_call_time - second_call_time:.4f}s")
    print(f"Compilation fraction:           {(first_call_time - second_call_time) / first_call_time * 100:.1f}%")

    # Verify results are identical
    error12 = np.max(np.abs(result1 - result2))
    error23 = np.max(np.abs(result2 - result3))
    print(f"Result consistency: {error12:.2e}, {error23:.2e}")

def main():
    """Run comprehensive JAX performance diagnostic."""
    print("JAX Performance Diagnostic Test Suite")
    print("Run this on your HPC cluster to identify performance bottlenecks")

    # Print JAX configuration
    print_jax_diagnostics()

    # Test different problem sizes
    test_cases = [
        (50, 8, "Small Problem (50x50, 8 energies)"),
        (117, 8, "Your Problem (117x117, 8 energies)"),
        (200, 20, "Medium Problem (200x200, 20 energies)"),
        (400, 50, "Large Problem (400x400, 50 energies)"),
    ]

    all_results = {}

    for matrix_size, num_energies, test_name in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING: {test_name}")
        print(f"{'='*80}")

        # Create test data
        F, S = create_test_matrices(matrix_size)
        g = MockSurfaceGreen(matrix_size)
        Elist = np.linspace(-1.0, 1.0, num_energies) + 1j * 0.01
        weights = np.ones(num_energies, dtype=complex) * (Elist[1] - Elist[0])

        # Run performance comparison
        results = run_performance_comparison(F, S, g, Elist, weights, test_name)
        all_results[test_name] = results

    # Test compilation overhead
    test_compilation_overhead()

    # Print summary
    print_performance_summary(all_results)

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("Based on the results above:")
    print("1. If JAX is slower than NumPy across all sizes, check JAX configuration")
    print("2. If compilation overhead dominates, increase matrix/energy thresholds")
    print("3. If memory usage is excessive, consider worker threads over vmap")
    print("4. If JAX vmap is fast for large problems, adjust thresholds accordingly")

if __name__ == "__main__":
    main()