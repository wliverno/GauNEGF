"""
JAX-powered Green's Functions Integration

Clean, modular integration functions using the generic parallelize.py framework.
Supports both retarded (Gr) and lesser (G<) Green's functions.

Author: William Livernois
"""

import jax
import jax.numpy as jnp
import numpy as np
import threading
from jax import jit
from gauNEGF.parallelize import parallelize_energy_calculation

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)


class Integrator:
    """
    Thread-safe accumulator for weighted matrix integration.

    Provides memory-efficient integration by accumulating results
    as they are computed, avoiding storage of intermediate matrices.
    """

    def __init__(self, shape):
        """
        Initialize integrator with zero matrix.

        Parameters
        ----------
        shape : tuple
            Shape of matrices to accumulate (typically (N, N))
        """
        self.result = np.zeros(shape, dtype=complex)
        self.lock = threading.Lock()

    def accumulate(self, weight, matrix):
        """
        Thread-safe accumulation of weighted matrix.

        Parameters
        ----------
        weight : float or complex
            Integration weight for this matrix
        matrix : ndarray
            Matrix to add to accumulated result

        Returns
        -------
        int
            Success indicator (1) - keeps memory footprint minimal
        """
        with self.lock:
            self.result += weight * matrix
        return 1


@jit
def _compute_gr(E, F, S, sigma_total):
    """
    JIT-compiled function to compute retarded Green's function at energy E.

    Parameters
    ----------
    E : float
        Energy point
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sigma_total : ndarray
        Total self-energy at energy E

    Returns
    -------
    ndarray
        Gr(E) = [E*S - F - Sigma(E)]^-1
    """
    mat = E * S - F - sigma_total
    return jnp.linalg.inv(mat)


@jit
def _compute_gless(E, F, S, sigma_total, sigma_contact):
    """
    JIT-compiled function to compute lesser Green's function at energy E.

    Parameters
    ----------
    E : float
        Energy point
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sigma_total : ndarray
        Total self-energy at energy E
    sigma_contact : ndarray
        Contact self-energy at energy E (sigma_total if ind=None)

    Returns
    -------
    ndarray
        G<(E) = Gr(E) * Γ(E) * Ga(E)
    """
    # Compute retarded Green's function
    mat = E * S - F - sigma_total
    Gr_E = jnp.linalg.inv(mat)

    # Compute advanced Green's function: Ga = Gr dagger
    Ga_E = jnp.conj(Gr_E).T

    # Compute broadening function: Gamma = i[Sigma - Sigma dagger]
    gamma_E = 1j * (sigma_contact - jnp.conj(sigma_contact).T)

    # Compute lesser Green's function: G< = Gr * Gamma * Ga
    return Gr_E @ gamma_E @ Ga_E



def _gr_worker_function(E, F, S, g, integrator, weight_dict):
    """
    Worker function for retarded Green's function calculation.
    Used by parallelize_energy_calculation().

    Parameters
    ----------
    E : float
        Energy point
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    integrator : Integrator
        Thread-safe accumulator for results
    weight_dict : dict
        Dictionary mapping energy -> integration weight

    Returns
    -------
    int
        Success indicator (1) - memory efficient
    """
    weight = weight_dict[E]  # O(1) lookup
    sigma_total = jnp.array(g.sigmaTot(E))
    gr_matrix = _compute_gr(E, F, S, sigma_total)
    return integrator.accumulate(weight, gr_matrix)


def _gless_worker_function(E, F, S, g, integrator, weight_dict, ind=None):
    """
    Worker function for lesser Green's function calculation.
    Used by parallelize_energy_calculation().

    Parameters
    ----------
    E : float
        Energy point
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    integrator : Integrator
        Thread-safe accumulator for results
    weight_dict : dict
        Dictionary mapping energy -> integration weight
    ind : int, optional
        Contact index for partial density calculation

    Returns
    -------
    int
        Success indicator (1) - memory efficient
    """
    weight = weight_dict[E]  # O(1) lookup
    sigma_total = jnp.array(g.sigmaTot(E))

    if ind is None:
        sigma_contact = sigma_total  # Use total self-energy
    else:
        sigma_contact = jnp.array(g.sigma(E, ind))  # Use contact-specific self-energy

    gless_matrix = _compute_gless(E, F, S, sigma_total, sigma_contact)
    return integrator.accumulate(weight, gless_matrix)




def GrInt(F, S, g, Elist, weights):
    """
    Integrate retarded Green's function over energy using JAX parallelization.

    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator with sigmaTot(E) method
    Elist : ndarray
        Array of energies in eV (Mx1)
    weights : ndarray
        Array of weights for each energy (Mx1)

    Returns
    -------
    ndarray
        Integrated retarded Green's function (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    # Convert to JAX arrays for better performance
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)

    # Create integrator and weight dictionary
    integrator = Integrator(F.shape)
    weight_dict = dict(zip(Elist, weights))

    # Use generic parallelization framework with integrator
    # Force worker mode by setting matrix_size large enough to skip vmap
    results = parallelize_energy_calculation(
        Elist, _gr_worker_function, matrix_size=1000,  # Force worker mode
        F=F_jax, S=S_jax, g=g, integrator=integrator, weight_dict=weight_dict
    )

    return integrator.result




def GrLessInt(F, S, g, Elist, weights, ind=None):
    """
    Integrate lesser Green's function over energy using JAX parallelization.

    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator
    Elist : ndarray
        Array of energies in eV (Mx1)
    weights : ndarray
        Array of weights for each energy (Mx1)
    ind : int, optional
        Contact index for partial density calculation (default: None)

    Returns
    -------
    ndarray
        Integrated lesser Green's function (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    # Convert to JAX arrays for better performance
    F_jax = jnp.array(F)
    S_jax = jnp.array(S)

    # Create integrator and weight dictionary
    integrator = Integrator(F.shape)
    weight_dict = dict(zip(Elist, weights))

    # Use generic parallelization framework with integrator
    # Force worker mode by setting matrix_size large enough to skip vmap
    results = parallelize_energy_calculation(
        Elist, _gless_worker_function, matrix_size=1000,  # Force worker mode
        F=F_jax, S=S_jax, g=g, integrator=integrator, weight_dict=weight_dict, ind=ind
    )

    return integrator.result




