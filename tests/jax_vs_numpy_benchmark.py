"""
Simple JAX vs NumPy benchmark for surface Green's function sigma computation.
Creates two minimal implementations to isolate the performance difference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

# Enable double precision for fair comparison
jax.config.update("jax_enable_x64", True)


class SimpleSurfGNumPy:
    """Simplified NumPy implementation of surface Green's function."""

    def __init__(self, H, S, contact_indices):
        self.H = H.astype(complex)
        self.S = S.astype(complex)
        self.contact_indices = contact_indices
        self.eta = 1e-6

    def green_function(self, E):
        """Compute surface Green's function at energy E."""
        # Simple convergence loop - no optimization
        matrix_size = len(self.contact_indices[0])
        g = np.eye(matrix_size, dtype=complex)

        for iteration in range(100):  # Fixed iterations for consistency
            A = (E + 1j*self.eta) * np.eye(matrix_size) - self.H[:matrix_size, :matrix_size]
            g_new = np.linalg.inv(A)

            # Simple convergence check
            diff = np.max(np.abs(g_new - g))
            g = g_new
            if diff < 1e-8:
                break

        return g

    def sigma_total(self, E):
        """Compute total sigma matrix."""
        sigma = np.zeros((self.H.shape[0], self.H.shape[1]), dtype=complex)

        # Just use first contact for fair comparison with JAX version
        indices = self.contact_indices[0]
        g = self.green_function(E)
        # Simple sigma contribution
        sigma_contact = 1j * self.eta * g
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                sigma[idx_i, idx_j] += sigma_contact[i, j]

        return sigma


class SimpleSurfGJAX:
    """Simplified JAX implementation of surface Green's function with proper JIT."""

    def __init__(self, H, S, contact_indices):
        self.H = jnp.array(H, dtype=complex)
        self.S = jnp.array(S, dtype=complex)
        self.contact_indices = contact_indices
        self.eta = 1e-6

        # Pre-compile the JIT functions
        matrix_size = len(contact_indices[0])
        self._green_function_jit = jax.jit(self._green_function_impl)
        self._sigma_total_jit = jax.jit(self._sigma_total_impl)

    def _green_function_impl(self, E, H_sub):
        """JIT-compiled Green's function computation."""
        matrix_size = H_sub.shape[0]

        # Use lax.while_loop for proper JIT compilation
        def body_fun(state):
            g, iteration = state
            A = (E + 1j*self.eta) * jnp.eye(matrix_size) - H_sub
            g_new = jnp.linalg.inv(A)
            return g_new, iteration + 1

        def cond_fun(state):
            g, iteration = state
            # Just do fixed iterations for now to keep it simple
            return iteration < 100

        g_init = jnp.eye(matrix_size, dtype=complex)
        g_final, _ = jax.lax.while_loop(cond_fun, body_fun, (g_init, 0))

        return g_final

    def green_function(self, E):
        """Compute surface Green's function at energy E."""
        matrix_size = len(self.contact_indices[0])
        H_sub = self.H[:matrix_size, :matrix_size]
        return self._green_function_jit(E, H_sub)

    def _sigma_total_impl(self, E, H_full):
        """JIT-compiled sigma computation."""
        matrix_size_full = H_full.shape[0]
        matrix_size = len(self.contact_indices[0])

        # Get Green's function
        H_sub = H_full[:matrix_size, :matrix_size]
        g = self._green_function_jit(E, H_sub)

        # Initialize sigma
        sigma = jnp.zeros((matrix_size_full, matrix_size_full), dtype=complex)

        # Simple sigma contribution - just use first contact for now
        sigma_contact = 1j * self.eta * g
        indices = jnp.array(self.contact_indices[0])

        # Use JAX indexing for the update
        sigma = sigma.at[jnp.ix_(indices, indices)].add(sigma_contact)

        return sigma

    def sigma_total(self, E):
        """Compute total sigma matrix."""
        return self._sigma_total_jit(E, self.H)


def create_test_system(matrix_size):
    """Create identical test system for both implementations."""
    np.random.seed(42)  # For reproducible results

    # Random Hermitian Hamiltonian
    H = np.random.random((matrix_size, matrix_size)) + 1j * np.random.random((matrix_size, matrix_size))
    H = (H + H.conj().T) / 2

    # Identity overlap matrix
    S = np.eye(matrix_size, dtype=complex)

    # Simple contact indices
    contact_size = min(3, matrix_size // 4)
    contact_indices = [
        list(range(contact_size)),
        list(range(matrix_size - contact_size, matrix_size))
    ]

    return H, S, contact_indices


def benchmark_serial_comparison():
    """Benchmark JAX vs NumPy for serial sigma computation."""
    print("JAX vs NumPy Serial Comparison")
    print("=" * 50)

    matrix_sizes = [10, 20, 30]
    num_energies = 10

    for matrix_size in matrix_sizes:
        print(f"\nMatrix size: {matrix_size}x{matrix_size}")

        # Create identical test systems
        H, S, contact_indices = create_test_system(matrix_size)

        # Initialize both implementations
        surf_numpy = SimpleSurfGNumPy(H, S, contact_indices)
        surf_jax = SimpleSurfGJAX(H, S, contact_indices)

        # Test energies
        energies = np.linspace(-2, 2, num_energies)

        # Benchmark NumPy
        print("  NumPy implementation...")
        start_time = time.time()
        sigma_results_numpy = []
        for E in energies:
            sigma = surf_numpy.sigma_total(E)
            sigma_results_numpy.append(sigma)
        numpy_time = time.time() - start_time

        # Warm up JAX (compilation)
        print("  JAX warm-up...")
        _ = surf_jax.sigma_total(energies[0])  # Trigger JIT compilation

        # Benchmark JAX
        print("  JAX implementation...")
        start_time = time.time()
        sigma_results_jax = []
        for E in energies:
            sigma = surf_jax.sigma_total(E)
            sigma_results_jax.append(sigma)
        jax_time = time.time() - start_time

        # Compare results for correctness
        max_diff = 0
        for i in range(len(sigma_results_numpy)):
            diff = np.max(np.abs(np.array(sigma_results_jax[i]) - sigma_results_numpy[i]))
            max_diff = max(max_diff, diff)

        # Report results
        speedup = numpy_time / jax_time
        print(f"  Results:")
        print(f"    NumPy time:     {numpy_time:.4f}s")
        print(f"    JAX time:       {jax_time:.4f}s")
        print(f"    JAX speedup:    {speedup:.2f}x")
        print(f"    Max difference: {max_diff:.2e}")

        if max_diff > 1e-10:
            print(f"    WARNING: Large numerical difference!")


if __name__ == "__main__":
    print("Simple JAX vs NumPy Benchmark")
    print("Testing basic sigma computation performance")
    print()

    benchmark_serial_comparison()

    print("\n" + "=" * 50)
    print("This tests the fundamental JAX vs NumPy performance")
    print("before considering any parallelization strategies.")