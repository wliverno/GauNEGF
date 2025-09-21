#!/usr/bin/env python3
"""
GPU Linear Algebra Benchmark for NEGF Calculations

This script benchmarks core linear algebra operations used in Non-Equilibrium
Green's Function (NEGF) quantum transport calculations:

1. Matrix inversion - Computing Green's functions G = (zS - H)^(-1)
2. Matrix multiplication - Computing products like G*Gamma*G†

Tests different precision levels, compares CPU vs GPU performance, and provides
optimization recommendations for large-scale quantum transport simulations.

Requirements:
- CuPy >= 11.0.0 (for matrix operations)
- NumPy for CPU comparisons
- Tests float32 vs float64 precision trade-offs

Author: Adapted for NEGF quantum transport calculations
"""

import time
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional

# GPU availability check
try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CuPy {cp.__version__} detected with CUDA support")
    else:
        print("CuPy found but CUDA not available")
except ImportError:
    CUDA_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy")
    print("Running CPU-only benchmarks...")


def print_with_indent(message: str, level: int = 0):
    """Print message with hierarchical indentation"""
    indent = "  " * level
    print(f"{indent}{message}")


def get_system_info() -> Dict:
    """Get comprehensive system information for performance context"""
    info = {
        'cpu_cores': os.cpu_count(),
        'environment': {}
    }

    # Check threading environment variables that affect CPU performance
    thread_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
    for var in thread_vars:
        info['environment'][var] = os.environ.get(var, 'not set')

    if CUDA_AVAILABLE:
        try:
            info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
            info['gpus'] = []

            for i in range(info['gpu_count']):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    free_mem, total_mem = cp.cuda.Device(i).mem_info

                    gpu_info = {
                        'id': i,
                        'name': props['name'].decode('utf-8'),
                        'memory_gb': total_mem / 1e9,
                        'free_memory_gb': free_mem / 1e9,
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'multiprocessors': props['multiProcessorCount']
                    }
                    info['gpus'].append(gpu_info)

            # CUDA version info
            info['cuda_runtime'] = cp.cuda.runtime.runtimeGetVersion()
            info['cuda_driver'] = cp.cuda.runtime.driverGetVersion()

        except Exception as e:
            info['gpu_error'] = str(e)

    return info


def print_system_info():
    """Display system information in readable format"""
    print_with_indent("=" * 60)
    print_with_indent("NEGF Linear Algebra GPU Benchmark")
    print_with_indent("=" * 60)

    info = get_system_info()

    print_with_indent(f"CPU: {info['cpu_cores']} cores")

    # Print threading configuration
    print_with_indent("Threading configuration:")
    for var, value in info['environment'].items():
        print_with_indent(f"{var}: {value}", 1)

    if CUDA_AVAILABLE and 'gpus' in info:
        print_with_indent(f"\nGPU: {info['gpu_count']} device(s) detected")
        for gpu in info['gpus']:
            print_with_indent(f"GPU {gpu['id']}: {gpu['name']}", 1)
            print_with_indent(f"Memory: {gpu['free_memory_gb']:.1f}GB free / {gpu['memory_gb']:.1f}GB total", 2)
            print_with_indent(f"Compute Capability: {gpu['compute_capability']}", 2)
            print_with_indent(f"Multiprocessors: {gpu['multiprocessors']}", 2)

        print_with_indent(f"\nCUDA Runtime: {info['cuda_runtime']}")
        print_with_indent(f"CUDA Driver: {info['cuda_driver']}")
    else:
        print_with_indent("\nGPU: Not available")


def create_test_matrices(size: int, matrix_type: str = 'hermitian') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test matrices representative of NEGF calculations.

    Parameters:
    -----------
    size : int
        Matrix dimension (typically 1000-5000 for device regions)
    matrix_type : str
        Type of matrix to create ('hermitian', 'complex_general')

    Returns:
    --------
    Tuple of (Hamiltonian-like matrix, Overlap-like matrix)
    """
    np.random.seed(42)  # Reproducible results

    if matrix_type == 'hermitian':
        # Create matrices similar to Hamiltonian (H) and Overlap (S) matrices
        H = np.random.random((size, size)) + 1j * np.random.random((size, size))
        H = (H + H.conj().T) / 2  # Make Hermitian
        H += np.eye(size) * 0.1   # Add small diagonal for numerical stability

        S = np.random.random((size, size)) + 1j * np.random.random((size, size))
        S = (S + S.conj().T) / 2  # Make Hermitian positive definite
        S += np.eye(size) * 1.1   # Ensure positive definiteness

    else:  # complex_general
        H = np.random.random((size, size)) + 1j * np.random.random((size, size))
        S = np.random.random((size, size)) + 1j * np.random.random((size, size))

    return H.astype(np.complex128), S.astype(np.complex128)


def time_cpu_operation(operation_func, *args, num_iterations: int = 5) -> Tuple[float, float]:
    """
    Time a CPU operation with multiple iterations and return mean and std.

    Parameters:
    -----------
    operation_func : callable
        Function to time
    *args : arguments to pass to operation_func
    num_iterations : int
        Number of timing iterations

    Returns:
    --------
    Tuple of (mean_time, std_time)
    """
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = operation_func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def time_gpu_operation(operation_func, *args, num_iterations: int = 5, warmup: bool = True) -> Tuple[float, float]:
    """
    Time a GPU operation with proper synchronization.

    Parameters:
    -----------
    operation_func : callable
        Function to time
    *args : arguments to pass to operation_func
    num_iterations : int
        Number of timing iterations
    warmup : bool
        Whether to do a warmup run

    Returns:
    --------
    Tuple of (mean_time, std_time)
    """
    if not CUDA_AVAILABLE:
        return float('inf'), float('inf')

    # Warmup run
    if warmup:
        try:
            _ = operation_func(*args)
            cp.cuda.Stream.null.synchronize()
        except Exception:
            return float('inf'), float('inf')

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        try:
            result = operation_func(*args)
            cp.cuda.Stream.null.synchronize()
        except Exception:
            return float('inf'), float('inf')
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_matrix_inversion(sizes: List[int], num_iterations: int = 5) -> Dict:
    """
    Benchmark matrix inversion: G = (zS - H)^(-1)

    This is the core operation in NEGF for computing Green's functions.
    Tests both direct inversion and solve-based approaches with different precisions.
    """
    print_with_indent("\n" + "=" * 50)
    print_with_indent("MATRIX INVERSION BENCHMARK")
    print_with_indent("=" * 50)
    print_with_indent("Computing G = (zS - H)^(-1) for Green's function calculation")

    results = {}

    for size in sizes:
        print_with_indent(f"\nTesting {size}x{size} matrices ({num_iterations} iterations)")

        # Create test matrices representing (zS - H)
        H, S = create_test_matrices(size)
        z = 1.0 + 0.01j  # Complex energy point typical in NEGF
        matrix = z * S - H

        size_results = {}

        # Method 1: Direct matrix inversion
        print_with_indent("Method 1: Direct inversion np.linalg.inv()", 1)

        # CPU float64 timing
        def cpu_inv_f64():
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(matrix)

        cpu_inv_mean, cpu_inv_std = time_cpu_operation(cpu_inv_f64, num_iterations=num_iterations)
        print_with_indent(f"CPU: {cpu_inv_mean:.4f}±{cpu_inv_std:.4f}s", 2)

        # GPU float64 timing
        if CUDA_AVAILABLE:
            matrix_gpu = cp.asarray(matrix, dtype=cp.complex128)

            def gpu_inv_f64():
                try:
                    return cp.linalg.inv(matrix_gpu)
                except cp.linalg.LinAlgError:
                    return cp.linalg.pinv(matrix_gpu)

            gpu_inv_mean, gpu_inv_std = time_gpu_operation(gpu_inv_f64, num_iterations=num_iterations)
            speedup_inv = cpu_inv_mean / gpu_inv_mean if gpu_inv_mean > 0 else 0

            print_with_indent(f"GPU: {gpu_inv_mean:.4f}±{gpu_inv_std:.4f}s (speedup: {speedup_inv:.2f}x)", 2)

            size_results['inversion'] = {
                'cpu_time': cpu_inv_mean,
                'cpu_std': cpu_inv_std,
                'gpu_time': gpu_inv_mean,
                'gpu_std': gpu_inv_std,
                'speedup': speedup_inv
            }

        # Method 2: Solve-based approach (often more stable)
        print_with_indent("Method 2: Linear solve (A \\ I)", 1)
        I = np.eye(size, dtype=np.complex128)

        # CPU solve timing
        def cpu_solve_f64():
            try:
                return np.linalg.solve(matrix, I)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(matrix, I, rcond=None)[0]

        cpu_solve_mean, cpu_solve_std = time_cpu_operation(cpu_solve_f64, num_iterations=num_iterations)
        print_with_indent(f"CPU: {cpu_solve_mean:.4f}±{cpu_solve_std:.4f}s", 2)

        if CUDA_AVAILABLE:
            I_gpu = cp.eye(size, dtype=cp.complex128)

            def gpu_solve_f64():
                try:
                    return cp.linalg.solve(matrix_gpu, I_gpu)
                except cp.linalg.LinAlgError:
                    return cp.linalg.lstsq(matrix_gpu, I_gpu, rcond=None)[0]

            gpu_solve_mean, gpu_solve_std = time_gpu_operation(gpu_solve_f64, num_iterations=num_iterations)
            speedup_solve = cpu_solve_mean / gpu_solve_mean if gpu_solve_mean > 0 else 0

            print_with_indent(f"GPU: {gpu_solve_mean:.4f}±{gpu_solve_std:.4f}s (speedup: {speedup_solve:.2f}x)", 2)

            size_results['solve'] = {
                'cpu_time': cpu_solve_mean,
                'cpu_std': cpu_solve_std,
                'gpu_time': gpu_solve_mean,
                'gpu_std': gpu_solve_std,
                'speedup': speedup_solve
            }

            # Compare methods
            if 'inversion' in size_results:
                inv_vs_solve_cpu = cpu_inv_mean / cpu_solve_mean
                inv_vs_solve_gpu = gpu_inv_mean / gpu_solve_mean
                print_with_indent(f"Method comparison:", 2)
                print_with_indent(f"CPU - solve vs inv: {inv_vs_solve_cpu:.2f}x faster", 3)
                print_with_indent(f"GPU - solve vs inv: {inv_vs_solve_gpu:.2f}x faster", 3)

        # Method 3: Float32 precision (potential for major speedups)
        print_with_indent("Method 3: Float32 precision", 1)
        matrix_f32 = matrix.astype(np.complex64)
        I_f32 = I.astype(np.complex64)

        # CPU float32
        def cpu_solve_f32():
            try:
                return np.linalg.solve(matrix_f32, I_f32)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(matrix_f32, I_f32, rcond=None)[0]

        cpu_f32_mean, _ = time_cpu_operation(cpu_solve_f32, num_iterations=num_iterations)
        print_with_indent(f"CPU: {cpu_f32_mean:.4f}s", 2)

        if CUDA_AVAILABLE:
            matrix_f32_gpu = cp.asarray(matrix_f32, dtype=cp.complex64)
            I_f32_gpu = cp.eye(size, dtype=cp.complex64)

            def gpu_solve_f32():
                try:
                    return cp.linalg.solve(matrix_f32_gpu, I_f32_gpu)
                except cp.linalg.LinAlgError:
                    return cp.linalg.lstsq(matrix_f32_gpu, I_f32_gpu, rcond=None)[0]

            gpu_f32_mean, _ = time_gpu_operation(gpu_solve_f32, num_iterations=num_iterations)
            speedup_f32 = cpu_f32_mean / gpu_f32_mean if gpu_f32_mean > 0 else 0

            print_with_indent(f"GPU: {gpu_f32_mean:.4f}s (speedup: {speedup_f32:.2f}x)", 2)

            if 'solve' in size_results:
                precision_speedup = size_results['solve']['gpu_time'] / gpu_f32_mean
                print_with_indent(f"Float32 vs Float64 GPU: {precision_speedup:.2f}x faster", 3)

            size_results['float32'] = {
                'cpu_time': cpu_f32_mean,
                'gpu_time': gpu_f32_mean,
                'speedup': speedup_f32
            }

        results[size] = size_results

    return results


def benchmark_matrix_multiplication(sizes: List[int], num_iterations: int = 5) -> Dict:
    """
    Benchmark matrix multiplication: C = A @ B

    This operation appears frequently in NEGF calculations for computing
    products like G*Gamma*G†, density matrices, and self-energy calculations.
    """
    print_with_indent("\n" + "=" * 50)
    print_with_indent("MATRIX MULTIPLICATION BENCHMARK")
    print_with_indent("=" * 50)
    print_with_indent("Computing C = A @ B for Green's function products")

    results = {}

    for size in sizes:
        print_with_indent(f"\nTesting {size}x{size} matrices ({num_iterations} iterations)")

        # Create test matrices
        A, B = create_test_matrices(size)
        size_results = {}

        # Method 1: Standard matrix multiplication
        print_with_indent("Method 1: Matrix multiplication A @ B", 1)

        # CPU timing
        def cpu_matmul():
            return np.matmul(A, B)

        cpu_matmul_mean, cpu_matmul_std = time_cpu_operation(cpu_matmul, num_iterations=num_iterations)
        print_with_indent(f"CPU: {cpu_matmul_mean:.4f}±{cpu_matmul_std:.4f}s", 2)

        if CUDA_AVAILABLE:
            A_gpu = cp.asarray(A, dtype=cp.complex128)
            B_gpu = cp.asarray(B, dtype=cp.complex128)

            def gpu_matmul():
                return cp.matmul(A_gpu, B_gpu)

            gpu_matmul_mean, gpu_matmul_std = time_gpu_operation(gpu_matmul, num_iterations=num_iterations)
            speedup_matmul = cpu_matmul_mean / gpu_matmul_mean if gpu_matmul_mean > 0 else 0

            print_with_indent(f"GPU: {gpu_matmul_mean:.4f}±{gpu_matmul_std:.4f}s (speedup: {speedup_matmul:.2f}x)", 2)

            size_results['matmul'] = {
                'cpu_time': cpu_matmul_mean,
                'cpu_std': cpu_matmul_std,
                'gpu_time': gpu_matmul_mean,
                'gpu_std': gpu_matmul_std,
                'speedup': speedup_matmul
            }

        # Method 2: Triple product (common in NEGF: G*Gamma*G†)
        print_with_indent("Method 2: Triple product A @ B @ A†", 1)
        A_dagger = A.conj().T

        # CPU timing
        def cpu_triple():
            temp = np.matmul(A, B)
            return np.matmul(temp, A_dagger)

        cpu_triple_mean, cpu_triple_std = time_cpu_operation(cpu_triple, num_iterations=num_iterations)
        print_with_indent(f"CPU: {cpu_triple_mean:.4f}±{cpu_triple_std:.4f}s", 2)

        if CUDA_AVAILABLE:
            A_dagger_gpu = A_gpu.conj().T

            def gpu_triple():
                temp = cp.matmul(A_gpu, B_gpu)
                return cp.matmul(temp, A_dagger_gpu)

            gpu_triple_mean, gpu_triple_std = time_gpu_operation(gpu_triple, num_iterations=num_iterations)
            speedup_triple = cpu_triple_mean / gpu_triple_mean if gpu_triple_mean > 0 else 0

            print_with_indent(f"GPU: {gpu_triple_mean:.4f}±{gpu_triple_std:.4f}s (speedup: {speedup_triple:.2f}x)", 2)

            size_results['triple_product'] = {
                'cpu_time': cpu_triple_mean,
                'cpu_std': cpu_triple_std,
                'gpu_time': gpu_triple_mean,
                'gpu_std': gpu_triple_std,
                'speedup': speedup_triple
            }

        # Method 3: Float32 matrix multiplication (often shows massive GPU speedups)
        print_with_indent("Method 3: Float32 precision", 1)
        A_f32 = A.astype(np.complex64)
        B_f32 = B.astype(np.complex64)

        # CPU float32
        def cpu_matmul_f32():
            return np.matmul(A_f32, B_f32)

        cpu_f32_mean, _ = time_cpu_operation(cpu_matmul_f32, num_iterations=num_iterations)
        print_with_indent(f"CPU: {cpu_f32_mean:.4f}s", 2)

        if CUDA_AVAILABLE:
            A_f32_gpu = cp.asarray(A_f32, dtype=cp.complex64)
            B_f32_gpu = cp.asarray(B_f32, dtype=cp.complex64)

            def gpu_matmul_f32():
                return cp.matmul(A_f32_gpu, B_f32_gpu)

            gpu_f32_mean, _ = time_gpu_operation(gpu_matmul_f32, num_iterations=num_iterations)
            speedup_f32 = cpu_f32_mean / gpu_f32_mean if gpu_f32_mean > 0 else 0

            print_with_indent(f"GPU: {gpu_f32_mean:.4f}s (speedup: {speedup_f32:.2f}x)", 2)

            if 'matmul' in size_results:
                precision_speedup = size_results['matmul']['gpu_time'] / gpu_f32_mean
                print_with_indent(f"Float32 vs Float64 GPU: {precision_speedup:.2f}x faster", 3)

            size_results['matmul_f32'] = {
                'cpu_time': cpu_f32_mean,
                'gpu_time': gpu_f32_mean,
                'speedup': speedup_f32
            }

        results[size] = size_results

    return results


def validate_physics_precision(size: int = 2000) -> Dict:
    """
    Validate that float32 precision preserves essential physics.

    Tests key quantum transport properties:
    - Green's function unitarity
    - Eigenvalue accuracy
    - Hermiticity preservation
    - Density of states calculation
    """
    print_with_indent("\n" + "=" * 60)
    print_with_indent("PHYSICS PRECISION VALIDATION")
    print_with_indent("=" * 60)
    print_with_indent(f"Testing numerical accuracy for {size}x{size} quantum system")

    # Create test system
    H, S = create_test_matrices(size, 'hermitian')
    z = 1.0 + 0.01j  # Energy point
    matrix = z * S - H

    validation_results = {}

    # Test 1: Green's function calculation accuracy
    print_with_indent("\n1. Green's Function Accuracy", 1)

    # Float64 reference
    G_f64 = np.linalg.solve(matrix, np.eye(size, dtype=np.complex128))

    # Float32 calculation
    matrix_f32 = matrix.astype(np.complex64)
    I_f32 = np.eye(size, dtype=np.complex64)
    G_f32 = np.linalg.solve(matrix_f32, I_f32).astype(np.complex128)

    # Compare accuracy
    relative_error = np.linalg.norm(G_f64 - G_f32) / np.linalg.norm(G_f64)
    max_element_error = np.max(np.abs(G_f64 - G_f32))

    print_with_indent(f"Relative error ||G_f64 - G_f32|| / ||G_f64||: {relative_error:.2e}", 2)
    print_with_indent(f"Maximum element error: {max_element_error:.2e}", 2)

    validation_results['greens_function'] = {
        'relative_error': relative_error,
        'max_element_error': max_element_error
    }

    # Test 2: Eigenvalue accuracy (critical for band structure)
    print_with_indent("\n2. Eigenvalue Accuracy (Band Structure)", 1)

    # Float64 eigenvalues
    eigs_f64 = np.linalg.eigvals(H)
    eigs_f64_sorted = np.sort(eigs_f64.real)  # Sort by real part for comparison

    # Float32 eigenvalues
    H_f32 = H.astype(np.complex64)
    eigs_f32 = np.linalg.eigvals(H_f32).astype(np.complex128)
    eigs_f32_sorted = np.sort(eigs_f32.real)

    # Compare band structure accuracy
    eigenval_error = np.mean(np.abs(eigs_f64_sorted - eigs_f32_sorted))
    band_gap_f64 = eigs_f64_sorted[-1] - eigs_f64_sorted[0]  # Energy range
    band_gap_f32 = eigs_f32_sorted[-1] - eigs_f32_sorted[0]
    gap_error = abs(band_gap_f64 - band_gap_f32) / abs(band_gap_f64)

    print_with_indent(f"Mean eigenvalue error: {eigenval_error:.2e} eV", 2)
    print_with_indent(f"Energy range (f64): {band_gap_f64:.6f} eV", 2)
    print_with_indent(f"Energy range (f32): {band_gap_f32:.6f} eV", 2)
    print_with_indent(f"Relative range error: {gap_error:.2e}", 2)

    validation_results['eigenvalues'] = {
        'mean_error': eigenval_error,
        'range_error': gap_error,
        'range_f64': band_gap_f64,
        'range_f32': band_gap_f32
    }

    # Test 3: Hermiticity preservation
    print_with_indent("\n3. Hermiticity Preservation", 1)

    # Test if H remains Hermitian after float32 conversion
    H_f64_herm_error = np.linalg.norm(H - H.conj().T)
    H_f32_herm_error = np.linalg.norm(H_f32 - H_f32.conj().T)

    print_with_indent(f"Hermiticity error (f64): {H_f64_herm_error:.2e}", 2)
    print_with_indent(f"Hermiticity error (f32): {H_f32_herm_error:.2e}", 2)

    validation_results['hermiticity'] = {
        'f64_error': H_f64_herm_error,
        'f32_error': H_f32_herm_error
    }

    # Test 4: Density of States comparison
    print_with_indent("\n4. Density of States Accuracy", 1)

    # DOS at Fermi level (Im[Tr(G)] gives DOS)
    dos_f64 = -np.trace(np.imag(G_f64)) / np.pi
    dos_f32 = -np.trace(np.imag(G_f32)) / np.pi
    dos_error = abs(dos_f64 - dos_f32) / abs(dos_f64)

    print_with_indent(f"DOS (f64): {dos_f64:.6f} states/eV", 2)
    print_with_indent(f"DOS (f32): {dos_f32:.6f} states/eV", 2)
    print_with_indent(f"Relative DOS error: {dos_error:.2e}", 2)

    validation_results['density_of_states'] = {
        'dos_f64': dos_f64,
        'dos_f32': dos_f32,
        'relative_error': dos_error
    }

    # Physics interpretation
    print_with_indent("\n5. Physics Assessment", 1)

    if relative_error < 1e-4:
        print_with_indent("EXCELLENT: Float32 precision preserves quantum transport physics", 2)
    elif relative_error < 1e-3:
        print_with_indent("GOOD: Float32 suitable for most NEGF calculations", 2)
    elif relative_error < 1e-2:
        print_with_indent("CAUTION: Float32 may affect sensitive transport properties", 2)
    else:
        print_with_indent("WARNING: Float32 precision insufficient for accurate physics", 2)

    if eigenval_error < 1e-4:
        print_with_indent("Band structure accuracy preserved", 2)
    else:
        print_with_indent("Band structure may be affected", 2)

    if dos_error < 1e-3:
        print_with_indent("Density of states accuracy preserved", 2)
    else:
        print_with_indent("DOS calculations may be affected", 2)

    return validation_results


def analyze_scaling(inv_results: Dict, matmul_results: Dict):
    """Analyze performance scaling with matrix size"""
    print_with_indent("\n" + "=" * 60)
    print_with_indent("SCALING ANALYSIS: SMALL vs LARGE MATRICES")
    print_with_indent("=" * 60)

    sizes = sorted([s for s in inv_results.keys() if isinstance(s, int)])

    if len(sizes) >= 2:
        small_size = sizes[0]
        large_size = sizes[-1]

        print_with_indent(f"Comparing {small_size}x{small_size} vs {large_size}x{large_size} matrices:")

        # Memory scaling
        small_memory = small_size**2 * 16 / 1e9  # complex128 bytes -> GB
        large_memory = large_size**2 * 16 / 1e9
        memory_ratio = large_memory / small_memory

        print_with_indent(f"\nMemory scaling: {memory_ratio:.1f}x increase", 1)
        print_with_indent(f"Small matrix: {small_memory:.3f} GB", 2)
        print_with_indent(f"Large matrix: {large_memory:.3f} GB", 2)

        # Matrix inversion scaling analysis
        print_with_indent(f"\nMatrix Inversion scaling:", 1)
        if 'solve' in inv_results[small_size] and 'solve' in inv_results[large_size]:
            small_cpu = inv_results[small_size]['solve']['cpu_time']
            large_cpu = inv_results[large_size]['solve']['cpu_time']
            cpu_scaling = large_cpu / small_cpu
            theoretical_scaling = (large_size / small_size) ** 3  # O(n^3) operations
            scaling_efficiency = theoretical_scaling / cpu_scaling

            print_with_indent(f"CPU time ratio: {cpu_scaling:.1f}x (theoretical: {theoretical_scaling:.1f}x)", 2)
            print_with_indent(f"Scaling efficiency: {scaling_efficiency:.2f}", 2)

            if CUDA_AVAILABLE and 'gpu_time' in inv_results[large_size]['solve']:
                small_gpu = inv_results[small_size]['solve']['gpu_time']
                large_gpu = inv_results[large_size]['solve']['gpu_time']
                gpu_scaling = large_gpu / small_gpu
                small_speedup = small_cpu / small_gpu
                large_speedup = large_cpu / large_gpu

                print_with_indent(f"GPU time ratio: {gpu_scaling:.1f}x", 2)
                print_with_indent(f"Speedup scaling: {small_speedup:.1f}x -> {large_speedup:.1f}x", 2)

        # Matrix multiplication scaling analysis
        print_with_indent(f"\nMatrix Multiplication scaling:", 1)
        if 'matmul' in matmul_results[small_size] and 'matmul' in matmul_results[large_size]:
            small_cpu = matmul_results[small_size]['matmul']['cpu_time']
            large_cpu = matmul_results[large_size]['matmul']['cpu_time']
            cpu_scaling = large_cpu / small_cpu
            theoretical_scaling = (large_size / small_size) ** 3
            scaling_efficiency = theoretical_scaling / cpu_scaling

            print_with_indent(f"CPU time ratio: {cpu_scaling:.1f}x (theoretical: {theoretical_scaling:.1f}x)", 2)
            print_with_indent(f"Scaling efficiency: {scaling_efficiency:.2f}", 2)

            if CUDA_AVAILABLE and 'gpu_time' in matmul_results[large_size]['matmul']:
                small_gpu = matmul_results[small_size]['matmul']['gpu_time']
                large_gpu = matmul_results[large_size]['matmul']['gpu_time']
                gpu_scaling = large_gpu / small_gpu
                small_speedup = small_cpu / small_gpu
                large_speedup = large_cpu / large_gpu

                print_with_indent(f"GPU time ratio: {gpu_scaling:.1f}x", 2)
                print_with_indent(f"Speedup scaling: {small_speedup:.1f}x -> {large_speedup:.1f}x", 2)


def print_summary(inv_results: Dict, matmul_results: Dict):
    """Print comprehensive performance summary and recommendations"""
    print_with_indent("\n" + "=" * 60)
    print_with_indent("PERFORMANCE SUMMARY & RECOMMENDATIONS")
    print_with_indent("=" * 60)

    # Find largest matrix size tested
    max_size = max([size for size in inv_results.keys() if isinstance(size, int)])
    print_with_indent(f"\nPerformance at {max_size}x{max_size} (typical device region size):")

    # Matrix inversion summary
    if max_size in inv_results:
        inv_data = inv_results[max_size]
        print_with_indent("\n1. MATRIX INVERSION (Green's functions)")

        if 'solve' in inv_data and 'inversion' in inv_data:
            best_cpu = min(inv_data['solve']['cpu_time'], inv_data['inversion']['cpu_time'])
            best_gpu = min(inv_data['solve']['gpu_time'], inv_data['inversion']['gpu_time'])
            best_method = 'solve' if inv_data['solve']['gpu_time'] < inv_data['inversion']['gpu_time'] else 'inversion'

            print_with_indent(f"   Best method: {best_method}", 1)
            print_with_indent(f"   CPU time: {best_cpu:.3f}s", 1)
            print_with_indent(f"   GPU time: {best_gpu:.3f}s", 1)
            print_with_indent(f"   Speedup: {best_cpu/best_gpu:.1f}x", 1)

            if 'float32' in inv_data:
                f32_speedup = best_gpu / inv_data['float32']['gpu_time']
                print_with_indent(f"   Float32 potential: {f32_speedup:.1f}x additional speedup", 1)

    # Matrix multiplication summary
    if max_size in matmul_results:
        matmul_data = matmul_results[max_size]
        print_with_indent("\n2. MATRIX MULTIPLICATION (G*Gamma*G† products)")

        if 'matmul' in matmul_data:
            print_with_indent(f"   Single multiplication: {matmul_data['matmul']['gpu_time']:.3f}s (speedup: {matmul_data['matmul']['speedup']:.1f}x)", 1)

        if 'triple_product' in matmul_data:
            print_with_indent(f"   Triple product: {matmul_data['triple_product']['gpu_time']:.3f}s (speedup: {matmul_data['triple_product']['speedup']:.1f}x)", 1)

        if 'matmul_f32' in matmul_data:
            f32_speedup = matmul_data['matmul']['gpu_time'] / matmul_data['matmul_f32']['gpu_time']
            print_with_indent(f"   Float32 potential: {f32_speedup:.1f}x additional speedup", 1)

    # Practical recommendations
    print_with_indent("\nRECOMMENDATIONS FOR NEGF CALCULATIONS:")

    # Check if GPU provides significant benefit
    gpu_beneficial = False
    if max_size in inv_results and 'solve' in inv_results[max_size]:
        if inv_results[max_size]['solve']['speedup'] > 2.0:
            gpu_beneficial = True

    if gpu_beneficial:
        print_with_indent("GPU acceleration recommended for this system", 1)

        # Float32 recommendation
        if max_size in inv_results and 'float32' in inv_results[max_size]:
            f32_time = inv_results[max_size]['float32']['gpu_time']
            if f32_time < 0.05:  # Less than 50ms
                print_with_indent("Consider float32 precision for significant speedup", 1)
                print_with_indent("  (Validate numerical accuracy for your specific problem)", 2)

        # Method recommendations
        if max_size in inv_results:
            inv_data = inv_results[max_size]
            if 'solve' in inv_data and 'inversion' in inv_data:
                if inv_data['solve']['gpu_time'] < inv_data['inversion']['gpu_time']:
                    print_with_indent("Use solve() instead of inv() for better performance", 1)
    else:
        print_with_indent("GPU provides limited benefit for this matrix size/system", 1)
        print_with_indent("  Consider CPU optimization or larger problem sizes", 2)

    # Memory usage estimation
    memory_per_matrix = max_size * max_size * 16 / 1e9  # complex128 bytes -> GB
    total_memory = memory_per_matrix * 5  # Estimate for workspace
    print_with_indent(f"\nMemory requirements (estimated):", 1)
    print_with_indent(f"Per matrix: {memory_per_matrix:.2f} GB", 2)
    print_with_indent(f"Total workspace: {total_memory:.2f} GB", 2)


def run_full_benchmark(sizes: Optional[List[int]] = None) -> Dict:
    """Run the complete NEGF benchmark suite"""
    if sizes is None:
        sizes = [1000, 2000, 3000]  # Typical NEGF problem sizes

    print_system_info()

    # Run all benchmarks
    inv_results = benchmark_matrix_inversion(sizes)
    matmul_results = benchmark_matrix_multiplication(sizes)

    # Physics validation (crucial for quantum transport)
    physics_validation = validate_physics_precision(sizes[1] if len(sizes) > 1 else sizes[0])

    # Scaling analysis
    analyze_scaling(inv_results, matmul_results)

    # Print summary
    print_summary(inv_results, matmul_results)

    return {
        'inversion': inv_results,
        'multiplication': matmul_results,
        'physics_validation': physics_validation,
        'system_info': get_system_info()
    }


def main():
    """Main execution function for NEGF linear algebra benchmarks"""
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]
    else:
        sizes = [1000, 2000, 3000]

    print("NEGF Linear Algebra GPU Benchmark")
    print("Testing matrix inversion and multiplication performance")
    print(f"Matrix sizes: {sizes}")

    results = run_full_benchmark(sizes)
    return results


if __name__ == "__main__":
    main()