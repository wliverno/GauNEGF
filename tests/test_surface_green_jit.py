#!/usr/bin/env python3
"""
Test the JIT-compiled surface Green's function solver.

Tests the new surface Green's function iteration kernel.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import sys
import os

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the JIT kernel directly
from gauNEGF.surfG1D import _surface_green_iteration_kernel

def create_test_contact(size=10, coupling_strength=0.1):
    """Create test matrices for surface Green's function."""
    np.random.seed(42)

    # Create Hermitian alpha matrix (on-site)
    alpha_real = np.random.randn(size, size)
    alpha = (alpha_real + alpha_real.T) / 2

    # Create Hermitian beta matrix (coupling)
    beta = np.random.randn(size, size) * coupling_strength

    # Create positive definite overlap matrices
    S_temp = np.random.randn(size, size)
    Salpha = S_temp @ S_temp.T
    Salpha -= np.diag(np.diag(Salpha))
    Salpha += np.eye(size)

    S_temp2 = np.random.randn(size, size)
    Sbeta = S_temp2 @ S_temp2.T

    return alpha, Salpha, beta, Sbeta

def manual_iteration(A, B, g_init, conv=1e-6, rel_factor=0.1, max_iter=1000):
    """Manual implementation for comparison."""
    g = g_init.copy()
    B_dag = B.conj().T

    for count in range(max_iter):
        g_prev = g.copy()

        # Main iteration step
        g = np.linalg.inv(A - B @ g @ B_dag)

        # Convergence check
        dg = np.abs(g - g_prev) / np.maximum(np.abs(g), 1e-12)
        diff = np.max(dg)

        # Relaxation mixing
        g = g * rel_factor + g_prev * (1 - rel_factor)

        if diff <= conv:
            return g, True, count + 1

    return g, False, max_iter

def test_surface_green_jit():
    """Test JIT surface Green's function against manual implementation."""
    print("Surface Green's Function JIT Test")
    print("="*50)

    sizes = [5, 10, 20]
    energies = [0.0, 1.0, -1.0, 0.5j]

    for size in sizes:
        print(f"\nTesting {size}x{size} contact matrices:")

        alpha, Salpha, beta, Sbeta = create_test_contact(size)

        for E in energies:
            print(f"  Energy E = {E}")

            # Prepare matrices
            eta = 1e-4
            A = (E + 1j * eta) * Salpha - alpha
            B = (E + 1j * eta) * Sbeta - beta
            g_init = np.zeros_like(A, dtype=complex)

            # Test parameters
            conv = 1e-8
            rel_factor = 0.1
            max_iter = 500

            # Convert to JAX arrays
            A_jax = jnp.array(A)
            B_jax = jnp.array(B)
            g_init_jax = jnp.array(g_init)

            # Time JIT version (including first-time compilation)
            start = time.perf_counter()
            g_jit, conv_jit, iter_jit = _surface_green_iteration_kernel(
                A_jax, B_jax, g_init_jax, conv, rel_factor)
            jit_time = time.perf_counter() - start

            # Time manual version
            start = time.perf_counter()
            g_manual, conv_manual, iter_manual = manual_iteration(
                A, B, g_init, conv, rel_factor, max_iter)
            manual_time = time.perf_counter() - start

            # Compare results
            g_jit_np = np.array(g_jit)
            error = np.max(np.abs(g_jit_np - g_manual))

            print(f"    JIT:     {jit_time:.4f}s, {iter_jit} iters, conv={conv_jit}")
            print(f"    Manual:  {manual_time:.4f}s, {iter_manual} iters, conv={conv_manual}")
            print(f"    Error:   {error:.2e}")

            # Verification
            if error < 1e-10 and conv_jit == conv_manual:
                if jit_time < manual_time:
                    print(f"    [EXCELLENT] {manual_time/jit_time:.1f}x speedup")
                else:
                    print(f"    [GOOD] Correct result")
            else:
                print(f"    [WARNING] Accuracy issue!")

def benchmark_surface_green():
    """Benchmark the surface Green's function with realistic parameters."""
    print(f"\n{'='*50}")
    print("SURFACE GREEN'S FUNCTION BENCHMARK")
    print(f"{'='*50}")

    # Realistic contact size
    size = 30
    num_energies = 50

    print(f"Benchmarking {size}x{size} contact over {num_energies} energy points...")

    alpha, Salpha, beta, Sbeta = create_test_contact(size, coupling_strength=0.2)
    energies = np.linspace(-2.0, 2.0, num_energies)
    eta = 1e-9

    # Test parameters
    conv = 1e-5
    rel_factor = 0.1
    max_iter = 200

    # Warmup JIT
    print("Warming up JIT compilation...")
    E = energies[0]
    A = (E + 1j * eta) * Salpha - alpha
    B = (E + 1j * eta) * Sbeta - beta
    g_init = np.zeros_like(A, dtype=complex)

    A_jax = jnp.array(A)
    B_jax = jnp.array(B)
    g_init_jax = jnp.array(g_init)

    _ = _surface_green_iteration_kernel(A_jax, B_jax, g_init_jax, conv, rel_factor)

    # Benchmark JIT version
    print("Running JIT benchmark...")
    start = time.perf_counter()
    jit_results = []
    total_iterations = 0

    for E in energies:
        A = (E + 1j * eta) * Salpha - alpha
        B = (E + 1j * eta) * Sbeta - beta

        A_jax = jnp.array(A)
        B_jax = jnp.array(B)

        g, converged, iterations = _surface_green_iteration_kernel(
            A_jax, B_jax, g_init_jax, conv, rel_factor)

        jit_results.append(np.array(g))
        total_iterations += iterations
        g_init_jax = g  # Use as initial guess for next energy

    jit_time = time.perf_counter() - start

    # Benchmark manual version
    print("Running manual benchmark...")
    start = time.perf_counter()
    manual_results = []
    manual_iterations = 0
    g_init_manual = np.zeros_like(A, dtype=complex)

    for E in energies:
        A = (E + 1j * eta) * Salpha - alpha
        B = (E + 1j * eta) * Sbeta - beta

        g, converged, iterations = manual_iteration(
            A, B, g_init_manual, conv, rel_factor, max_iter)

        manual_results.append(g)
        manual_iterations += iterations
        g_init_manual = g  # Use as initial guess

    manual_time = time.perf_counter() - start

    # Compare results
    errors = [np.max(np.abs(jit - manual))
              for jit, manual in zip(jit_results, manual_results)]
    max_error = max(errors)

    speedup = manual_time / jit_time

    print(f"\nResults:")
    print(f"  JIT time:       {jit_time:.3f} s ({total_iterations} total iterations)")
    print(f"  Manual time:    {manual_time:.3f} s ({manual_iterations} total iterations)")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  Max error:      {max_error:.2e}")
    print(f"  Avg iterations: {total_iterations/num_energies:.1f}")

    if speedup > 1.5 and max_error < 1e-10:
        print(f"  [EXCELLENT] {speedup:.1f}x speedup with perfect accuracy!")
    elif speedup > 1.1 and max_error < 1e-8:
        print(f"  [GOOD] {speedup:.1f}x speedup with good accuracy")
    else:
        print(f"  [WARNING] {speedup:.1f}x speedup, {max_error:.1e} error")

if __name__ == "__main__":
    test_surface_green_jit()
    benchmark_surface_green()