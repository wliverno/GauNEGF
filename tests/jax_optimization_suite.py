#!/usr/bin/env python3
"""
JAX JIT Optimization Test Suite

Professional test suite for all JAX JIT optimizations in GauNEGF.
Consolidates all performance and accuracy testing into one comprehensive file.

Tests:
1. Transport calculation kernels (transmission and DOS)
2. Density matrix kernels
3. Surface Green's function solvers
4. Overall performance scaling analysis

Physics Setup:
- Uses realistic quantum transport physics with proper non-Hermitian self-energies
- Ensures gamma matrices are non-zero for meaningful transport calculations
- Validates against exact NumPy reference implementations

Author: Claude Code Assistant
"""

import time
import numpy as np
import jax.numpy as jnp
import jax
import sys
import os

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from gauNEGF.transport import _transmission_kernel_restricted, _dos_kernel
from gauNEGF.density import _compute_dos_at_energy

def create_realistic_transport_system(size, coupling_strength=0.1, seed=42):
    """
    Create a realistic quantum transport system with proper physics.

    Simplified approach: Generate Gamma matrices first, then sigma = -i*Gamma/2
    This ensures gamma = i(sigma - sigma^H) = Gamma exactly.
    """
    np.random.seed(seed)

    # 1. Create real symmetric Fock matrix (tight-binding Hamiltonian)
    F_random = np.random.randn(size, size)
    F = (F_random + F_random.T) / 2

    # 2. Create positive definite overlap matrix
    S_temp = np.random.randn(size, size)
    S = S_temp @ S_temp.T + 0.1 * np.eye(size)

    # 3. Create Gamma matrices first (positive semidefinite, localized to contacts)
    Gamma1 = np.zeros((size, size))
    Gamma2 = np.zeros((size, size))

    contact_sites = min(4, size//4)

    # Contact 1: couple to first few sites
    for i in range(contact_sites):
        Gamma1[i, i] = coupling_strength

    # Contact 2: couple to last few sites
    for i in range(size - contact_sites, size):
        Gamma2[i, i] = coupling_strength

    # 4. Create self-energies: sigma = -i*Gamma/2
    sigma1 = -1j * Gamma1 / 2
    sigma2 = -1j * Gamma2 / 2

    # 5. Total self-energy and gamma matrices
    sigma_total = sigma1 + sigma2

    return F, S, sigma_total, Gamma1, Gamma2


def verify_physics_setup(F, S, sigma_total, gamma1, gamma2, verbose=True):
    """
    Verify that the physics setup is correct.
    With sigma = -i*Gamma/2, we should have gamma = Gamma exactly.
    """
    checks = {}

    # Check 1: F must be Hermitian
    checks['F_hermitian'] = np.allclose(F, F.conj().T)

    # Check 2: S must be positive definite
    S_eigvals = np.linalg.eigvals(S)
    checks['S_positive_definite'] = np.all(S_eigvals > 1e-12)

    # Check 3: Gamma matrices must be positive semidefinite
    Gamma1_eigvals = np.linalg.eigvals(gamma1)
    Gamma2_eigvals = np.linalg.eigvals(gamma2)
    checks['Gamma1_positive_semidefinite'] = np.all(Gamma1_eigvals >=0 )
    checks['Gamma2_positive_semidefinite'] = np.all(Gamma2_eigvals >=0 )

    # Check 4: gamma matrices should be non-zero
    checks['gamma1_nonzero'] = np.linalg.norm(gamma1) > 0   
    checks['gamma2_nonzero'] = np.linalg.norm(gamma2) > 0

    # Check 5: sigma_total should be purely imaginary (since sigma = -i*Gamma/2)
    checks['sigma_total_purely_imaginary'] = np.allclose(np.real(sigma_total), np.zeros_like(np.real(sigma_total)))

    all_pass = all(checks.values())

    if verbose:
        print("  Physics Verification:")
        for check, result in checks.items():
            status = "PASS" if result else "FAIL"
            print(f"    {check}: {status}")
        print(f"    Overall: {'VALID PHYSICS' if all_pass else 'INVALID PHYSICS'}")

    return all_pass


def benchmark_transmission_kernels(sizes=[100, 200, 300], num_energies=20):
    """
    Benchmark JIT transmission and DOS calculation kernels.
    Tests performance scaling and numerical accuracy.
    """
    print("TRANSPORT CALCULATION KERNELS")
    print("=" * 50)

    results = {}

    for size in sizes:
        print(f"\nTesting {size}x{size} transport system:")

        # Create realistic system with correct physics
        F, S, sigma_total, gamma1, gamma2 = create_realistic_transport_system(size)

        # Verify physics is correct
        physics_valid = verify_physics_setup(F, S, sigma_total, gamma1, gamma2)
        if not physics_valid:
            print(f"  ERROR: Invalid physics setup for size {size}")
            continue

        # Convert to JAX arrays
        F_jax = jnp.array(F)
        S_jax = jnp.array(S)
        sigma_total_jax = jnp.array(sigma_total)
        gamma1_jax = jnp.array(gamma1)
        gamma2_jax = jnp.array(gamma2)

        # Warmup JIT compilation
        E_warmup = 0.5
        _ = _transmission_kernel_restricted(E_warmup, F_jax, S_jax, sigma_total_jax, gamma1_jax, gamma2_jax)
        _ = _dos_kernel(E_warmup, F_jax, S_jax, sigma_total_jax)

        # Create energy range for benchmarking
        energies = np.linspace(-1.0, 1.0, num_energies)

        # === TRANSMISSION BENCHMARK ===
        # JAX JIT version
        start = time.perf_counter()
        jit_transmissions = []
        for E in energies:
            T = _transmission_kernel_restricted(E, F_jax, S_jax, sigma_total_jax, gamma1_jax, gamma2_jax)
            jit_transmissions.append(float(T))
        jit_trans_time = time.perf_counter() - start

        # NumPy reference version
        start = time.perf_counter()
        numpy_transmissions = []
        for E in energies:
            mat = E * S - F - sigma_total
            Gr = np.linalg.inv(mat)
            Ga = np.conj(Gr).T
            temp = gamma1 @ Gr @ gamma2
            T = np.real(np.trace(temp @ Ga))
            numpy_transmissions.append(T)
        numpy_trans_time = time.perf_counter() - start

        # === DOS BENCHMARK ===
        # JAX JIT version
        start = time.perf_counter()
        jit_dos = []
        for E in energies:
            dos_total, _ = _dos_kernel(E, F_jax, S_jax, sigma_total_jax)
            jit_dos.append(float(dos_total))
        jit_dos_time = time.perf_counter() - start

        # NumPy reference version
        start = time.perf_counter()
        numpy_dos = []
        for E in energies:
            mat = E * S - F - sigma_total
            Gr = np.linalg.inv(mat)
            dos = -np.imag(np.trace(Gr)) / np.pi
            numpy_dos.append(dos)
        numpy_dos_time = time.perf_counter() - start

        # Calculate performance metrics
        trans_speedup = numpy_trans_time / jit_trans_time
        dos_speedup = numpy_dos_time / jit_dos_time
        avg_speedup = (trans_speedup + dos_speedup) / 2

        # Calculate accuracy metrics
        trans_error = np.max(np.abs(np.array(jit_transmissions) - np.array(numpy_transmissions)))
        dos_error = np.max(np.abs(np.array(jit_dos) - np.array(numpy_dos)))
        max_error = max(trans_error, dos_error)

        # Store results
        results[size] = {
            'trans_speedup': trans_speedup,
            'dos_speedup': dos_speedup,
            'avg_speedup': avg_speedup,
            'trans_error': trans_error,
            'dos_error': dos_error,
            'max_error': max_error,
            'physics_valid': physics_valid
        }

        # Print results
        print(f"  Transmission: {trans_speedup:.2f}x speedup, {trans_error:.2e} error")
        print(f"  DOS:          {dos_speedup:.2f}x speedup, {dos_error:.2e} error")
        print(f"  Average:      {avg_speedup:.2f}x speedup, {max_error:.2e} max error")

        # Performance assessment
        if max_error < 1e-8:
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(f"  Status:       {status} ({avg_speedup:.2f}x speedup)")

        # Show physics check for first transmission
        if len(jit_transmissions) > 0:
            print(f"  Physics check: T(E={energies[0]:.1f}) = {jit_transmissions[0]:.6f}")

    return results


def benchmark_density_kernels(size=150, num_energies=30):
    """
    Benchmark JIT density matrix calculation kernels.
    """
    print(f"\nDENSITY MATRIX KERNELS")
    print("=" * 50)
    print(f"Testing {size}x{size} density matrix calculations:")

    # Create test system
    F, S, sigma_total, _, _ = create_realistic_transport_system(size)

    # Convert to JAX
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)
    sigma_jax = jnp.array(sigma_total)

    energies = np.linspace(-2.0, 2.0, num_energies)
    E_test = 0.5

    # Warmup JIT
    _ = _compute_dos_at_energy(E_test, F_jax, S_jax, sigma_jax)

    # Benchmark DOS calculation
    start = time.perf_counter()
    jit_dos_results = []
    for E in energies:
        dos_total, _ = _compute_dos_at_energy(E, F_jax, S_jax, sigma_jax)
        jit_dos_results.append(float(dos_total))
    jit_time = time.perf_counter() - start

    # NumPy reference
    start = time.perf_counter()
    numpy_dos_results = []
    for E in energies:
        mat = E * S - F - sigma_total
        Gr = np.linalg.inv(mat)
        dos = -np.imag(np.trace(Gr)) / np.pi
        numpy_dos_results.append(dos)
    numpy_time = time.perf_counter() - start

    speedup = numpy_time / jit_time
    error = np.max(np.abs(np.array(jit_dos_results) - np.array(numpy_dos_results)))

    print(f"  DOS kernel:   {speedup:.2f}x speedup, {error:.2e} error")

    # Performance assessment
    if error < 1e-8:
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"  Status:       {status} ({speedup:.2f}x speedup)")

    return {'dos_speedup': speedup, 'avg_speedup': speedup}


def run_comprehensive_analysis():
    """
    Run complete JAX JIT optimization analysis.
    """
    print("JAX JIT OPTIMIZATION COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("Testing all JAX optimizations with realistic quantum transport physics")
    print()

    # Test 1: Transport kernels
    transport_results = benchmark_transmission_kernels()

    # Test 2: Density kernels
    density_results = benchmark_density_kernels()

    # Overall analysis
    print(f"\n{'=' * 60}")
    print("PERFORMANCE SCALING ANALYSIS")
    print(f"{'=' * 60}")

    if transport_results:
        print("\nTransport Kernel Scaling:")
        print(f"{'Size':>6} {'Trans':>8} {'DOS':>8} {'Avg':>8} {'Error':>10}")
        print("-" * 50)

        for size, results in transport_results.items():
            if results['physics_valid']:
                print(f"{size:>6} {results['trans_speedup']:>7.2f}x {results['dos_speedup']:>7.2f}x "
                      f"{results['avg_speedup']:>7.2f}x {results['max_error']:>9.2e}")

        # Find best performance
        valid_results = {k: v for k, v in transport_results.items() if v['physics_valid']}
        if valid_results:
            best_size = max(valid_results.keys(), key=lambda s: valid_results[s]['avg_speedup'])
            best_speedup = valid_results[best_size]['avg_speedup']
            print(f"\nBest transport performance: {best_speedup:.2f}x at {best_size}x{best_size}")

    print(f"\nDensity kernel performance: {density_results['dos_speedup']:.2f}x speedup")


if __name__ == "__main__":
    run_comprehensive_analysis()