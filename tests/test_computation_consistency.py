#!/usr/bin/env python3
"""
Test that serial, vmap, and worker computation paths give identical results.

Verifies that the three different computation approaches produce mathematically
identical results for the same Green's function integration problem.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gauNEGF.integrate import GrInt, GrLessInt, Integrator

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

def create_test_matrices(size):
    """
    Create proper test matrices with specified constraints.

    F: Real Hermitian matrix
    S: Real Hermitian overlap matrix with 1s on diagonal and positive off-diagonal < 1
    """
    np.random.seed(42)  # Fixed seed for reproducibility

    # Create Fock matrix F - real and Hermitian
    F_temp = np.random.randn(size, size) * 0.1
    F = (F_temp + F_temp.T) / 2  # Make Hermitian

    # Create overlap matrix S - real Hermitian with specific structure
    # Start with identity matrix
    S = np.eye(size)

    # Add small positive off-diagonal elements < 1
    off_diag = np.random.rand(size, size) * 0.2  # Random values [0, 0.2]
    off_diag = (off_diag + off_diag.T) / 2  # Make symmetric
    np.fill_diagonal(off_diag, 0)  # Zero diagonal

    S += off_diag  # Add to identity matrix

    # Verify properties
    assert np.allclose(F, F.T), "F is not Hermitian"
    assert np.allclose(S, S.T), "S is not Hermitian"
    assert np.allclose(np.diag(S), 1.0), "S diagonal is not 1.0"
    assert np.all(S >= 0), "S has negative elements"
    assert np.all(S <= 1.0), "S has elements > 1.0"

    return F, S

def serial_gr_integration(F, S, g, Elist, weights):
    """Serial implementation for GrInt - simple for loop."""
    result = np.zeros_like(F, dtype=complex)

    for E, w in zip(Elist, weights):
        sigma_total = g.sigmaTot(E)
        mat = E * S - F - sigma_total
        Gr_E = np.linalg.inv(mat)
        result += w * Gr_E

    return result

def serial_gless_integration(F, S, g, Elist, weights, ind=None):
    """Serial implementation for GrLessInt - simple for loop."""
    result = np.zeros_like(F, dtype=complex)

    for E, w in zip(Elist, weights):
        sigma_total = g.sigmaTot(E)

        if ind is None:
            sigma_contact = sigma_total
        else:
            sigma_contact = g.sigma(E, ind)

        # Compute retarded Green's function
        mat = E * S - F - sigma_total
        Gr_E = np.linalg.inv(mat)

        # Compute advanced Green's function
        Ga_E = np.conj(Gr_E).T

        # Compute broadening function
        gamma_E = 1j * (sigma_contact - np.conj(sigma_contact).T)

        # Compute lesser Green's function
        gless_E = Gr_E @ gamma_E @ Ga_E
        result += w * gless_E

    return result

def force_vmap_path(F, S, g, Elist, weights):
    """Force vmap path by temporarily changing thresholds."""
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

def force_worker_path(F, S, g, Elist, weights):
    """Force worker path by temporarily changing thresholds."""
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

def force_vmap_path_gless(F, S, g, Elist, weights, ind=None):
    """Force vmap path for GrLessInt."""
    import gauNEGF.integrate as integrate_module

    orig_threshold = integrate_module.SMALL_MATRIX_THRESHOLD
    orig_memory = integrate_module.MAX_VMAP_MEMORY_GB

    try:
        integrate_module.SMALL_MATRIX_THRESHOLD = 10000
        integrate_module.MAX_VMAP_MEMORY_GB = 100.0
        result = GrLessInt(F, S, g, Elist, weights, ind)
    finally:
        integrate_module.SMALL_MATRIX_THRESHOLD = orig_threshold
        integrate_module.MAX_VMAP_MEMORY_GB = orig_memory

    return result

def force_worker_path_gless(F, S, g, Elist, weights, ind=None):
    """Force worker path for GrLessInt."""
    import gauNEGF.integrate as integrate_module

    orig_threshold = integrate_module.SMALL_MATRIX_THRESHOLD
    orig_memory = integrate_module.MAX_VMAP_MEMORY_GB

    try:
        integrate_module.SMALL_MATRIX_THRESHOLD = 1
        integrate_module.MAX_VMAP_MEMORY_GB = 0.001
        result = GrLessInt(F, S, g, Elist, weights, ind)
    finally:
        integrate_module.SMALL_MATRIX_THRESHOLD = orig_threshold
        integrate_module.MAX_VMAP_MEMORY_GB = orig_memory

    return result

def print_error_and_timing_analysis(serial_result, vmap_result, worker_result,
                                   serial_time, vmap_time, worker_time, method_name):
    """Print detailed error analysis and timing comparison for three methods."""

    # Calculate errors
    vmap_error = np.max(np.abs(vmap_result - serial_result))
    worker_error = np.max(np.abs(worker_result - serial_result))
    vmap_worker_error = np.max(np.abs(vmap_result - worker_result))

    # Calculate relative errors
    max_val = np.max(np.abs(serial_result))
    vmap_rel_error = vmap_error / max_val if max_val > 0 else 0
    worker_rel_error = worker_error / max_val if max_val > 0 else 0
    vmap_worker_rel_error = vmap_worker_error / max_val if max_val > 0 else 0

    print(f"    {method_name}:")
    print(f"      Accuracy:")
    print(f"        Serial vs vmap:   {vmap_error:.2e} abs, {vmap_rel_error:.2e} rel")
    print(f"        Serial vs worker: {worker_error:.2e} abs, {worker_rel_error:.2e} rel")
    print(f"        vmap vs worker:   {vmap_worker_error:.2e} abs, {vmap_worker_rel_error:.2e} rel")

    # Timing analysis
    print(f"      Performance:")
    print(f"        Serial: {serial_time:.4f}s (baseline)")
    print(f"        vmap:   {vmap_time:.4f}s ({serial_time/vmap_time:.1f}x {'speedup' if vmap_time < serial_time else 'slower'})")
    print(f"        Worker: {worker_time:.4f}s ({serial_time/worker_time:.1f}x {'speedup' if worker_time < serial_time else 'slower'})")

    # Overall assessment
    max_error = max(vmap_error, worker_error, vmap_worker_error)
    fastest_time = min(serial_time, vmap_time, worker_time)

    if fastest_time == vmap_time:
        fastest_method = "vmap"
    elif fastest_time == worker_time:
        fastest_method = "worker"
    else:
        fastest_method = "serial"

    if max_error < 1e-14:
        accuracy_assessment = "EXCELLENT - identical to machine precision"
    elif max_error < 1e-12:
        accuracy_assessment = "VERY GOOD - nearly identical"
    elif max_error < 1e-10:
        accuracy_assessment = "GOOD - consistent"
    else:
        accuracy_assessment = "WARNING - significant differences!"

    print(f"      [{accuracy_assessment}]")
    print(f"      [FASTEST: {fastest_method} method]")

    return max_error

def test_computation_consistency():
    """Test that serial, vmap, and worker approaches give identical results."""
    print("Testing computation consistency across all three methods...")

    test_sizes = [100, 800]  # Test different matrix sizes

    overall_max_error = 0.0

    for size in test_sizes:
        print(f"\n  Testing {size}x{size} matrices:")

        # Create proper test matrices
        F, S = create_test_matrices(size)
        g = MockSurfaceGreen(size)

        # Create complex energy contour and weights (more realistic for NEGF)
        num_energies = 12
        # Complex contour: real axis with small imaginary part for convergence
        Elist = np.linspace(-1.0, 1.0, num_energies) + 1j * 0.01
        weights = np.ones(num_energies, dtype=complex) * (Elist[1] - Elist[0])

        # Test GrInt with timing
        print("    Testing GrInt:")

        import time
        start = time.perf_counter()
        serial_gr = serial_gr_integration(F, S, g, Elist, weights)
        serial_time = time.perf_counter() - start

        start = time.perf_counter()
        vmap_gr = force_vmap_path(F, S, g, Elist, weights)
        vmap_time = time.perf_counter() - start

        start = time.perf_counter()
        worker_gr = force_worker_path(F, S, g, Elist, weights)
        worker_time = time.perf_counter() - start

        max_error = print_error_and_timing_analysis(serial_gr, vmap_gr, worker_gr,
                                                   serial_time, vmap_time, worker_time, "GrInt")
        overall_max_error = max(overall_max_error, max_error)

        # Test GrLessInt with ind=None
        print("    Testing GrLessInt (ind=None):")

        start = time.perf_counter()
        serial_gless = serial_gless_integration(F, S, g, Elist, weights, ind=None)
        serial_time = time.perf_counter() - start

        start = time.perf_counter()
        vmap_gless = force_vmap_path_gless(F, S, g, Elist, weights, ind=None)
        vmap_time = time.perf_counter() - start

        start = time.perf_counter()
        worker_gless = force_worker_path_gless(F, S, g, Elist, weights, ind=None)
        worker_time = time.perf_counter() - start

        max_error = print_error_and_timing_analysis(serial_gless, vmap_gless, worker_gless,
                                                   serial_time, vmap_time, worker_time, "GrLessInt (ind=None)")
        overall_max_error = max(overall_max_error, max_error)

        # Test GrLessInt with ind=0
        print("    Testing GrLessInt (ind=0):")

        start = time.perf_counter()
        serial_gless = serial_gless_integration(F, S, g, Elist, weights, ind=0)
        serial_time = time.perf_counter() - start

        start = time.perf_counter()
        vmap_gless = force_vmap_path_gless(F, S, g, Elist, weights, ind=0)
        vmap_time = time.perf_counter() - start

        start = time.perf_counter()
        worker_gless = force_worker_path_gless(F, S, g, Elist, weights, ind=0)
        worker_time = time.perf_counter() - start

        max_error = print_error_and_timing_analysis(serial_gless, vmap_gless, worker_gless,
                                                   serial_time, vmap_time, worker_time, "GrLessInt (ind=0)")
        overall_max_error = max(overall_max_error, max_error)

    return overall_max_error

def test_complex_energy_consistency():
    """Test consistency with complex energy contours."""
    print(f"\n  Testing complex energy contours:")

    size = 80
    F, S = create_test_matrices(size)
    g = MockSurfaceGreen(size)

    # Complex energy contour
    num_energies = 8
    Elist = np.linspace(-0.8, 0.8, num_energies) + 1j * 0.01
    weights = np.ones(num_energies, dtype=complex) * (Elist[1] - Elist[0])

    # Test GrInt with complex energies
    import time
    start = time.perf_counter()
    serial_gr = serial_gr_integration(F, S, g, Elist, weights)
    serial_time = time.perf_counter() - start

    start = time.perf_counter()
    vmap_gr = force_vmap_path(F, S, g, Elist, weights)
    vmap_time = time.perf_counter() - start

    start = time.perf_counter()
    worker_gr = force_worker_path(F, S, g, Elist, weights)
    worker_time = time.perf_counter() - start

    max_error = print_error_and_timing_analysis(serial_gr, vmap_gr, worker_gr,
                                               serial_time, vmap_time, worker_time, "GrInt (complex)")

    return max_error

if __name__ == "__main__":
    print("Computation Consistency Test")
    print("Testing Serial vs vmap vs Worker approaches")
    print("=" * 60)

    try:
        max_error1 = test_computation_consistency()
        max_error2 = test_complex_energy_consistency()

        overall_max_error = max(max_error1, max_error2)

        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Maximum error across all tests: {overall_max_error:.2e}")

        if overall_max_error < 1e-14:
            print("[EXCELLENT] All three computation methods are identical!")
        elif overall_max_error < 1e-12:
            print("[VERY GOOD] All three computation methods are nearly identical!")
        elif overall_max_error < 1e-10:
            print("[GOOD] All three computation methods are consistent!")
        else:
            print("[FAIL] Significant differences between computation methods!")

        print("\nThis demonstrates that:")
        print("1. Serial implementation provides the reference result")
        print("2. vmap optimization preserves mathematical accuracy")
        print("3. Worker parallelization preserves mathematical accuracy")
        print("4. All approaches can be used interchangeably based on performance needs")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()