"""
Transport calculations for quantum systems using Non-Equilibrium Green's Functions.

This module provides functions for calculating quantum transport properties:
- Coherent transmission through molecular junctions
- Spin-dependent transport calculations
- Current calculations at finite bias
- Density of states calculations

The module supports both energy-independent and energy-dependent self-energies,
with implementations for both spin-restricted and spin-unrestricted calculations.
Spin-dependent transport follows the formalism described in [1].

References
----------
.. [1] Herrmann, C., Solomon, G. C., & Ratner, M. A. J. Chem. Theory Comput. 6, 3078 (2010)
      DOI: 10.1021/acs.jctc.9b01078
"""

import numpy as np
import os
import jax
import jax.numpy as jnp
from jax import jit
import scipy.io as io
from scipy.integrate import trapezoid
from gauNEGF.utils import inv
from gauNEGF.config import ENERGY_STEP, N_KT, TEMPERATURE

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kB = 8.617e-5           # eV/Kelvin
V_to_au = 0.03675       # Volts to Hartree/elementary Charge


class SigmaCalculator:
    """
    Unified interface for energy-dependent and energy-independent self-energies.

    Automatically detects whether sigma inputs are static arrays or energy-dependent
    surface Green's function objects, and provides a consistent interface for both.
    """

    def __init__(self, sig1, sig2=None, energy_dependent=None):
        """
        Initialize sigma calculator.

        Parameters
        ----------
        sig1 : ndarray or surfG object
            Left contact self-energy (static) or surface Green's function calculator
        sig2 : ndarray, optional
            Right contact self-energy (static arrays only)
        energy_dependent : bool, optional
            Force energy dependence detection. If None, auto-detect from sig1 type
        """
        self.sig1 = sig1
        self.sig2 = sig2

        # Auto-detect energy dependence
        if energy_dependent is None:
            # Check if sig1 has methods typical of surfG objects
            self.energy_dependent = hasattr(sig1, 'sigma') and hasattr(sig1, 'sigmaTot')
        else:
            self.energy_dependent = energy_dependent

        if self.energy_dependent and sig2 is not None:
            raise ValueError("For energy-dependent calculations, provide only surfG object as sig1")

        if not self.energy_dependent and sig2 is None:
            raise ValueError("For energy-independent calculations, provide both sig1 and sig2")

    def get_sigma_total(self, E, spin=None, matrix_size=None):
        """Get total self-energy at energy E."""
        if self.energy_dependent:
            sigma_total = self.sig1.sigmaTot(E)
        else:
            # Convert to numpy arrays and handle vector vs matrix cases
            sig1_array = np.asarray(self.sig1)
            sig2_array = np.asarray(self.sig2)

            if sig1_array.ndim == 1:
                sigma_total = np.diag(sig1_array + sig2_array)
            else:
                sigma_total = sig1_array + sig2_array

        # Handle spin expansions for multi-spin calculations
        if spin in ['u', 'ro', 'g'] and matrix_size is not None:
            # Check if matrices are already expanded (detect 2N x 2N case)
            sigma_size = sigma_total.shape[0]
            if matrix_size == 2 * sigma_size:
                # Matrices are already expanded, need to expand sigma to match
                if spin in ['u', 'ro']:
                    # Block diagonal expansion: [sig  0 ]
                    #                          [0  sig]
                    return np.kron(np.eye(2), sigma_total)
                elif spin == 'g':
                    # Kronecker expansion: [sig x I2]
                    return np.kron(sigma_total, np.eye(2))
            # If sizes match, matrices and sigmas are consistent - return as is

        return sigma_total

    def get_sigma(self, E, contact_index, spin=None, matrix_size=None):
        """Get contact-specific self-energy at energy E."""
        if self.energy_dependent:
            sigma = self.sig1.sigma(E, contact_index)
        else:
            if contact_index == 0:
                sig = self.sig1
            elif contact_index == -1 or contact_index == 1:
                sig = self.sig2
            else:
                raise ValueError(f"Invalid contact_index {contact_index}")

            sig_array = np.asarray(sig)
            if sig_array.ndim == 1:
                sigma = np.diag(sig_array)
            else:
                sigma = sig_array

        # Handle spin expansions for multi-spin calculations
        if spin in ['u', 'ro', 'g'] and matrix_size is not None:
            # Check if matrices are already expanded (detect 2N x 2N case)
            sigma_size = sigma.shape[0]
            if matrix_size == 2 * sigma_size:
                # Matrices are already expanded, need to expand sigma to match
                if spin in ['u', 'ro']:
                    # Block diagonal expansion: [sig  0 ]
                    #                          [0  sig]
                    return np.kron(np.eye(2), sigma)
                elif spin == 'g':
                    # Kronecker expansion: [sig x I2]
                    return np.kron(sigma, np.eye(2))
            # If sizes match, matrices and sigmas are consistent - return as is

        return sigma

    def get_gamma(self, E, contact_index, spin=None, matrix_size=None):
        """Get gamma matrix (coupling) at energy E for specified contact."""
        sigma = self.get_sigma(E, contact_index, spin, matrix_size)
        return 1j * (sigma - np.conj(sigma).T)


# JIT-compiled computational kernels for performance
@jit
def _transmission_kernel_restricted(E, F, S, sigma_total, gamma1, gamma2):
    """JIT-compiled kernel for restricted transmission calculation."""
    mat = E * S - F - sigma_total
    Gr = inv(mat)
    Ga = jnp.conj(Gr).T
    temp = gamma1 @ Gr @ gamma2
    return jnp.real(jnp.trace(temp @ Ga))

@jit
def _transmission_kernel_spin_block(E, F, S, sigma_total, gamma1, gamma2, N):
    """JIT-compiled kernel for spin-resolved block transmission calculation."""
    mat = E * S - F - sigma_total
    Gr = inv(mat)
    Ga = jnp.conj(Gr).T

    # Extract spin blocks efficiently
    Gr_blocks = jnp.array([Gr[:N, :N], Gr[:N, N:], Gr[N:, :N], Gr[N:, N:]])
    Ga_blocks = jnp.array([Ga[:N, :N], Ga[:N, N:], Ga[N:, :N], Ga[N:, N:]])
    gamma1_blocks = jnp.array([gamma1[:N, :N], gamma1[:N, :N], gamma1[N:, N:], gamma1[N:, N:]])
    gamma2_blocks = jnp.array([gamma2[:N, :N], gamma2[N:, N:], gamma2[:N, :N], gamma2[N:, N:]])

    def compute_transmission_component(i):
        temp = gamma1_blocks[i] @ Gr_blocks[i] @ gamma2_blocks[i]
        return jnp.real(jnp.trace(temp @ Ga_blocks[i]))

    # Vectorized computation over spin components
    T_spin = jax.vmap(compute_transmission_component)(jnp.arange(4))
    return jnp.sum(T_spin), T_spin

@jit
def _dos_kernel(E, F, S, sigma_total):
    """JIT-compiled kernel for density of states calculation."""
    mat = E * S - F - sigma_total
    Gr = inv(mat)
    dos_per_site = -jnp.imag(jnp.diag(Gr)) / jnp.pi
    total_dos = jnp.sum(dos_per_site)
    return total_dos, dos_per_site


def transmission_single_energy(E, F_jax, S_jax, sigma_calc, spin=None):
    """
    Calculate transmission at a single energy using linalg.py functions.

    Parameters
    ----------
    E : float
        Energy in eV
    F_jax : jax.numpy.ndarray
        Fock matrix (already JAX array)
    S_jax : jax.numpy.ndarray
        Overlap matrix (already JAX array)
    sigma_calc : SigmaCalculator
        Sigma calculator object
    spin : str, optional
        Spin configuration using Gaussian labels:
        'r' - restricted (closed shell)
        'u' - unrestricted (open shell, block diagonal)
        'ro' - restricted open (treated same as 'u')
        'g' - generalized (non-collinear, spinor)
        If None, defaults to 'r'

    Returns
    -------
    float or tuple
        For 'r': transmission value
        For 'u'/'ro'/'g': (total_transmission, [T_up_up, T_up_down, T_down_up, T_down_down])
    """

    if spin is None:
        spin = 'r'  # Always default to restricted

    # Determine N based on spin configuration
    if spin == 'r':
        N = F_jax.shape[0]
    else:
        N = F_jax.shape[0] // 2

    # Get self-energies and gamma matrices (pass matrix size for spin expansion)
    matrix_size = F_jax.shape[0]
    sigma_total = sigma_calc.get_sigma_total(E, spin, matrix_size)
    gamma1 = sigma_calc.get_gamma(E, 0, spin, matrix_size)  # Left contact
    gamma2 = sigma_calc.get_gamma(E, -1, spin, matrix_size)  # Right contact

    # Convert sigma/gamma to JAX arrays (these are energy-dependent, so convert here)
    sigma_total_jax = jnp.asarray(sigma_total)
    gamma1_jax = jnp.asarray(gamma1)
    gamma2_jax = jnp.asarray(gamma2)

    if spin == 'r':
        # Use JIT-compiled restricted transmission kernel
        transmission = _transmission_kernel_restricted(E, F_jax, S_jax, sigma_total_jax, gamma1_jax, gamma2_jax)
        return float(transmission)

    elif spin in ['u', 'ro']:
        # Use JIT-compiled spin block transmission kernel
        total_transmission, T_spin = _transmission_kernel_spin_block(
            E, F_jax, S_jax, sigma_total_jax, gamma1_jax, gamma2_jax, N)
        return float(total_transmission), T_spin.tolist()

    elif spin == 'g':
        # Add JIT kernel for generalized case - implement later if needed
        # For now, fall back to original implementation
        mat = E * S_jax - F_jax - sigma_total_jax
        Gr = inv(mat)
        Ga = jnp.conj(Gr).T

        # Extract spinor indices
        a_indices = jnp.arange(0, 2*N, 2)  # Alpha spin indices
        b_indices = jnp.arange(1, 2*N, 2)  # Beta spin indices

        Gr_blocks = [Gr[jnp.ix_(a_indices, a_indices)], Gr[jnp.ix_(a_indices, b_indices)],
                    Gr[jnp.ix_(b_indices, a_indices)], Gr[jnp.ix_(b_indices, b_indices)]]
        Ga_blocks = [Ga[jnp.ix_(a_indices, a_indices)], Ga[jnp.ix_(a_indices, b_indices)],
                    Ga[jnp.ix_(b_indices, a_indices)], Ga[jnp.ix_(b_indices, b_indices)]]

        # Use diagonal gamma blocks for generalized case
        gamma1_blocks = [gamma1_jax[jnp.ix_(a_indices, a_indices)], gamma1_jax[jnp.ix_(a_indices, a_indices)],
                        gamma1_jax[jnp.ix_(b_indices, b_indices)], gamma1_jax[jnp.ix_(b_indices, b_indices)]]
        gamma2_blocks = [gamma2_jax[jnp.ix_(a_indices, a_indices)], gamma2_jax[jnp.ix_(b_indices, b_indices)],
                        gamma2_jax[jnp.ix_(a_indices, a_indices)], gamma2_jax[jnp.ix_(b_indices, b_indices)]]

        T_spin = []
        for i in range(4):
            temp = gamma1_blocks[i] @ Gr_blocks[i] @ gamma2_blocks[i]
            T_ij = jnp.real(jnp.trace(temp @ Ga_blocks[i]))
            T_spin.append(float(T_ij))

        total_transmission = sum(T_spin)
        return total_transmission, T_spin

    else:
        raise ValueError(f"Unknown spin configuration '{spin}'. Use 'r', 'u', 'ro', or 'g'")


def dos_single_energy(E, F_jax, S_jax, sigma_calc, spin=None):
    """
    Calculate density of states at a single energy using linalg.py functions.

    Parameters
    ----------
    E : float
        Energy in eV
    F_jax : jax.numpy.ndarray
        Fock matrix (already JAX array)
    S_jax : jax.numpy.ndarray
        Overlap matrix (already JAX array)
    sigma_calc : SigmaCalculator
        Sigma calculator object
    spin : str, optional
        Spin configuration using Gaussian labels ('r', 'u', 'ro', 'g')
        If None, defaults to 'r'

    Returns
    -------
    tuple
        For 'r': (total_dos, dos_per_site)
        For 'u'/'ro'/'g': (total_dos, dos_per_site, dos_spin_up, dos_spin_down)
        where dos_per_site includes both spins, dos_spin_up/down are separate
    """
    if spin is None:
        spin = 'r'

    # Get total self-energy for the appropriate spin case
    matrix_size = F_jax.shape[0]
    sigma_total = sigma_calc.get_sigma_total(E, spin, matrix_size)

    # Convert sigma to JAX array (energy-dependent, so convert here)
    sigma_total_jax = jnp.asarray(sigma_total)

    if spin == 'r':
        # Use JIT-compiled DOS kernel
        total_dos, dos_per_site = _dos_kernel(E, F_jax, S_jax, sigma_total_jax)
        return float(total_dos), np.array(dos_per_site)

    elif spin in ['u', 'ro']:
        # Unrestricted/restricted open - split into spin up and down blocks
        N = F_jax.shape[0] // 2

        # Calculate Green's function
        mat = E * S_jax - F_jax - sigma_total_jax
        Gr = inv(mat)
        Gr = np.asarray(Gr)

        # Extract spin-resolved Green's functions
        Gr_up = Gr[:N, :N]      # Up-up block
        Gr_down = Gr[N:, N:]    # Down-down block

        # Calculate spin-resolved DOS
        dos_up_per_site = -np.imag(np.diag(Gr_up)) / np.pi
        dos_down_per_site = -np.imag(np.diag(Gr_down)) / np.pi

        # Total DOS per site (both spins)
        dos_per_site = np.concatenate([dos_up_per_site, dos_down_per_site])

        # Totals
        total_dos_up = np.sum(dos_up_per_site)
        total_dos_down = np.sum(dos_down_per_site)
        total_dos = total_dos_up + total_dos_down

        return total_dos, dos_per_site, dos_up_per_site, dos_down_per_site

    elif spin == 'g':
        # Generalized case - extract spinor components
        N = F_jax.shape[0] // 2

        # Calculate Green's function
        mat = E * S_jax - F_jax - sigma_total_jax
        Gr = inv(mat)
        Gr = np.asarray(Gr)

        # Extract alpha and beta spinor indices
        alpha_indices = np.arange(0, 2*N, 2)
        beta_indices = np.arange(1, 2*N, 2)

        # Extract spin-resolved Green's functions
        Gr_alpha = Gr[np.ix_(alpha_indices, alpha_indices)]
        Gr_beta = Gr[np.ix_(beta_indices, beta_indices)]

        # Calculate spin-resolved DOS
        dos_alpha_per_site = -np.imag(np.diag(Gr_alpha)) / np.pi
        dos_beta_per_site = -np.imag(np.diag(Gr_beta)) / np.pi

        # Total DOS per site (both spins)
        dos_per_site = -np.imag(np.diag(Gr)) / np.pi

        # Totals
        total_dos_alpha = np.sum(dos_alpha_per_site)
        total_dos_beta = np.sum(dos_beta_per_site)
        total_dos = np.sum(dos_per_site)

        return total_dos, dos_per_site, dos_alpha_per_site, dos_beta_per_site

    else:
        raise ValueError(f"Unknown spin configuration '{spin}'. Use 'r', 'u', 'ro', or 'g'")


def calculate_transmission(F, S, sigma_calculator, energy_list,
                          spin=None, checkpoint_file=None, 
                          checkpoint_interval=10):
    """
    Calculate transmission over energy range with checkpointing.
    
    Uses -1 as placeholder for uncalculated energies. Saves checkpoint
    every checkpoint_interval energies. If checkpoint_file exists, resumes
    from first uncalculated energy.
    
    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sigma_calculator : SigmaCalculator
        Sigma calculator object
    energy_list : array_like
        List of energies in eV
    spin : str, optional
        Spin configuration ('r', 'u', 'ro', 'g'). Defaults to 'r'
    checkpoint_file : str, optional
        Path to checkpoint file (.npz format). If None, no checkpointing
    checkpoint_interval : int, optional
        Save checkpoint every N energies (default: 10)
    
    Returns
    -------
    ndarray or tuple
        For restricted: transmission array (N,)
        For open shell: (transmission (N,), spin_transmission (N,4))
    """

    
    energy_list = np.asarray(energy_list)
    n_energies = len(energy_list)
    
    if spin is None:
        spin = 'r'
    
    # Convert F and S to JAX arrays once before the loop (prevents recompilation)
    F_jax = jnp.asarray(F)
    S_jax = jnp.asarray(S)
    
    # Initialize or load checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        data = np.load(checkpoint_file, allow_pickle=True)
        # Check if energy_list matches
        if 'energy_list' in data:
            checkpoint_energies = data['energy_list']
            if not np.allclose(checkpoint_energies, energy_list, rtol=1e-10):
                print(f"Warning: energy_list in checkpoint doesn't match. Starting fresh.")
                transmission = -1 * np.ones(n_energies)
                spin_trans = None
            else:
                transmission = data['transmission'] if 'transmission' in data else -1 * np.ones(n_energies)
                if spin in ['u', 'ro', 'g']:
                    spin_trans = data['spin_transmission'] if 'spin_transmission' in data else -1 * np.ones((n_energies, 4))
                else:
                    spin_trans = None
        else:
            transmission = -1 * np.ones(n_energies)
            spin_trans = None
    else:
        # Pre-allocate with -1 placeholders
        transmission = -1 * np.ones(n_energies)
        if spin in ['u', 'ro', 'g']:
            spin_trans = -1 * np.ones((n_energies, 4))
        else:
            spin_trans = None
    
    # Find remaining energies to calculate
    remaining = np.where(transmission == -1)[0]
    
    # Calculate remaining energies
    for idx, i in enumerate(remaining):
        E = energy_list[i]
        result = transmission_single_energy(E, F_jax, S_jax, sigma_calculator, spin)
        
        # Store result
        if isinstance(result, tuple):
            transmission[i] = result[0]
            spin_trans[i] = np.asarray(result[1])
        else:
            transmission[i] = result
        
        # Save checkpoint periodically
        if checkpoint_file and (idx % checkpoint_interval == 0):
            if spin_trans is not None:
                np.savez(checkpoint_file, transmission=transmission, 
                        spin_transmission=spin_trans, energy_list=energy_list)
            else:
                np.savez(checkpoint_file, transmission=transmission, energy_list=energy_list)
    
    # Final save
    if checkpoint_file:
        if spin_trans is not None:
            np.savez(checkpoint_file, transmission=transmission, 
                    spin_transmission=spin_trans, energy_list=energy_list)
        else:
            np.savez(checkpoint_file, transmission=transmission, energy_list=energy_list)
    
    # Return in expected format
    if spin_trans is not None:
        return transmission, spin_trans
    else:
        return transmission


def calculate_dos(F, S, sigma_calculator, energy_list,
                  spin=None, checkpoint_file=None,
                  checkpoint_interval=10):
    """
    Calculate DOS over energy range with checkpointing.
    
    Uses -1 as placeholder for uncalculated energies. Saves checkpoint
    every checkpoint_interval energies. If checkpoint_file exists, resumes
    from first uncalculated energy.
    
    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sigma_calculator : SigmaCalculator
        Sigma calculator object
    energy_list : array_like
        List of energies in eV
    spin : str, optional
        Spin configuration ('r', 'u', 'ro', 'g'). Defaults to 'r'
    checkpoint_file : str, optional
        Path to checkpoint file (.npz format). If None, no checkpointing
    checkpoint_interval : int, optional
        Save checkpoint every N energies (default: 10)
    
    Returns
    -------
    tuple
        For restricted: (dos_total (N,), dos_per_site (N,M))
        For open shell: (dos_total (N,), dos_per_site (N,M), dos_spin (N,2))
        where M = F.shape[0] (number of sites)
    """
    
    energy_list = np.asarray(energy_list)
    n_energies = len(energy_list)
    n_sites = F.shape[0]  # M sites
    
    if spin is None:
        spin = 'r'
    
    # Convert F and S to JAX arrays once before the loop (prevents recompilation)
    F_jax = jnp.asarray(F)
    S_jax = jnp.asarray(S)
    
    # Initialize or load checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        data = np.load(checkpoint_file, allow_pickle=True)
        # Check if energy_list matches
        if 'energy_list' in data:
            checkpoint_energies = data['energy_list']
            if not np.allclose(checkpoint_energies, energy_list, rtol=1e-10):
                print(f"Warning: energy_list in checkpoint doesn't match. Starting fresh.")
                dos_total = -1 * np.ones(n_energies)
                dos_per_site = -1 * np.ones((n_energies, n_sites))
                dos_spin = None
            else:
                dos_total = data['dos_total'] if 'dos_total' in data else -1 * np.ones(n_energies)
                dos_per_site = data['dos_per_site'] if 'dos_per_site' in data else -1 * np.ones((n_energies, n_sites))
                if spin in ['u', 'ro', 'g']:
                    dos_spin = data['dos_spin'] if 'dos_spin' in data else -1 * np.ones((n_energies, 2))
                else:
                    dos_spin = None
        else:
            dos_total = -1 * np.ones(n_energies)
            dos_per_site = -1 * np.ones((n_energies, n_sites))
            dos_spin = None
    else:
        # Pre-allocate with -1 placeholders
        dos_total = -1 * np.ones(n_energies)
        dos_per_site = -1 * np.ones((n_energies, n_sites))
        if spin in ['u', 'ro', 'g']:
            dos_spin = -1 * np.ones((n_energies, 2))
        else:
            dos_spin = None
    
    # Find remaining energies to calculate
    remaining = np.where(dos_total == -1)[0]
    
    # Calculate remaining energies
    for idx, i in enumerate(remaining):
        E = energy_list[i]
        result = dos_single_energy(E, F_jax, S_jax, sigma_calculator, spin)
        
        # Store result
        if len(result) == 2:
            # Restricted: (total_dos, dos_per_site)
            dos_total[i] = result[0]
            dos_per_site[i] = np.asarray(result[1])
        else:
            # Open shell: (total_dos, dos_per_site, dos_spin_up, dos_spin_down)
            dos_total[i] = result[0]
            dos_per_site[i] = np.asarray(result[1])
            dos_spin[i, 0] = np.sum(result[2])  # dos_up total
            dos_spin[i, 1] = np.sum(result[3])  # dos_down total
        
        # Save checkpoint periodically
        if checkpoint_file and (idx % checkpoint_interval == 0):
            if dos_spin is not None:
                np.savez(checkpoint_file, dos_total=dos_total, 
                        dos_per_site=dos_per_site, dos_spin=dos_spin, 
                        energy_list=energy_list)
            else:
                np.savez(checkpoint_file, dos_total=dos_total, 
                        dos_per_site=dos_per_site, energy_list=energy_list)
    
    # Final save
    if checkpoint_file:
        if dos_spin is not None:
            np.savez(checkpoint_file, dos_total=dos_total, 
                    dos_per_site=dos_per_site, dos_spin=dos_spin, 
                    energy_list=energy_list)
        else:
            np.savez(checkpoint_file, dos_total=dos_total, 
                    dos_per_site=dos_per_site, energy_list=energy_list)
    
    # Return in expected format
    if dos_spin is not None:
        return dos_total, dos_per_site, dos_spin
    else:
        return dos_total, dos_per_site


def calculate_current(F, S, sigma_calculator, fermi, qV, T=TEMPERATURE, spin=None, dE=ENERGY_STEP,
                     **kwargs):
    """
    Calculate current at applied voltage with checkpointing.
    
    Generates energy grid internally based on fermi, qV, T, and dE, then calculates
    transmission over integration window and integrates. Uses transmission checkpointing
    internally. The checkpoint file stores transmission data which can be reused.
    
    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sigma_calculator : SigmaCalculator
        Sigma calculator object
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float, optional
        Temperature in Kelvin (default: TEMPERATURE from config)
    spin : str, optional
        Spin configuration ('r', 'u', 'ro', 'g'). Defaults to 'r'
    dE : float, optional
        Energy step for integration in eV (default: ENERGY_STEP from config)
    **kwargs
        Additional parameters (such as checkpointing parameters) passed to calculate_transmission
    
    Returns
    -------
    float or tuple
        For restricted: current value (float)
        For open shell: (current_total (float), current_spin (4,))
    """
    if fermi is None or qV is None:
        raise ValueError("fermi and qV must be provided for current calculations")
    
    if spin is None:
        spin = 'r'
    
    # Handle negative qV by making dE negative (matches legacy behavior)
    if np.allclose(0, qV):
        return 0.0 if spin == 'r' else [0.0, 0.0, 0.0, 0.0]
    elif qV < 0:
        dE = -1 * abs(dE)
    else:
        dE = abs(dE)
    
    # Calculate chemical potentials
    muL = fermi - qV/2
    muR = fermi + qV/2
    
    # Generate energy grid for integration
    if T == 0:
        # Zero temperature - integrate between chemical potentials
        integration_energies = np.arange(muL, muR, dE)
    else:
        # Finite temperature - need broader energy range
        kT = kB * T
        spread = np.sign(dE) * N_KT * kT
        integration_energies = np.arange(muL - spread, muR + spread, dE)
    
    if len(integration_energies) == 0:
        raise ValueError("No energies in integration window. Check fermi, qV, and dE.")
    
    # Calculate transmission for integration energies    
    transmission_result = calculate_transmission(
        F, S, sigma_calculator, integration_energies,
        spin=spin, **kwargs
    )
    
    if isinstance(transmission_result, tuple):
        transmissions, spin_transmissions = transmission_result
        transmissions = np.asarray(transmissions)
        spin_transmissions = np.asarray(spin_transmissions)
    else:
        transmissions = np.asarray(transmission_result)
        spin_transmissions = None
    
    # Integrate transmission
    if T == 0: 
        # Zero temperature integration
        if spin_transmissions is not None:
            current_spin = [eoverh * trapezoid(spin_transmissions[:, i], integration_energies)
                           for i in range(4)]
            current_total = sum(current_spin)
            return current_total, current_spin
        else:
            current_total = eoverh * trapezoid(transmissions, integration_energies)
            # Apply spin factor for restricted calculations
            if spin == 'r':
                current_total *= 2
            return current_total
    else:
        # Finite temperature integration with Fermi-Dirac distribution
        dfermi = np.abs(1/(np.exp((integration_energies - muR)/(kB*T)) + 1) -
                       1/(np.exp((integration_energies - muL)/(kB*T)) + 1))
        
        if spin_transmissions is not None:
            current_spin = [eoverh * trapezoid(spin_transmissions[:, i] * dfermi, integration_energies)
                           for i in range(4)]
            current_total = sum(current_spin)
            return current_total, current_spin
        else:
            current_total = eoverh * trapezoid(transmissions * dfermi, integration_energies)
            # Apply spin factor for restricted calculations
            if spin == 'r':
                current_total *= 2
            return current_total


## LEGACY FUNCTIONS
def current(F, S, sig1, sig2, fermi, qV, T=TEMPERATURE, spin="r",dE=ENERGY_STEP):
    """
    Calculate coherent current using NEGF with energy-independent self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sig1 : ndarray
        Left contact self-energy (vector or matrix)
    sig2 : ndarray
        Right contact self-energy (vector or matrix)
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float
        Temperature in Kelvin (default: 0)
    spin : str, optional
        Spin configuration ('r' for restricted) (default: 'r')
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    float
        Current in Amperes
    """
    # Create energy grid
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    muL = fermi - qV/2
    muR = fermi + qV/2

    if T == 0:
        Elist = np.arange(muL, muR, dE)
    else:
        kT = kB*T
        spread = np.sign(dE)*N_KT*kT
        Elist = np.arange(muL-spread, muR+spread, dE)

    # Create sigma calculator and use checkpointable current calculation
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    return calculate_current(F, S, sigma_calc, fermi=fermi, qV=qV, T=T, spin=spin, dE=dE)

def currentSpin(F, S, sig1, sig2, fermi, qV, T=TEMPERATURE, spin="r",dE=ENERGY_STEP):
    """
    Calculate coherent spin current using NEGF with energy-independent self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sig1 : ndarray
        Left contact self-energy (vector or matrix)
    sig2 : ndarray
        Right contact self-energy (vector or matrix)
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float
        Temperature in Kelvin (default: 0)
    spin : str, optional
        Spin configuration ('r' for restricted) (default: 'r')
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    list
        Spin-currents (in Amperes) [I↑↑, I↑↓, I↓↑, I↓↓]
    """
    # Create sigma calculator and use checkpointable current calculation
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    result = calculate_current(F, S, sigma_calc, fermi=fermi, qV=qV, T=T, spin=spin, dE=dE)

    if isinstance(result, tuple):
        # Return spin currents if available
        return result[1]
    else:
        # Return zero spin currents for restricted case
        return [0, 0, 0, 0]


def currentE(F, S, g, fermi, qV, T=TEMPERATURE, spin="r",dE=ENERGY_STEP):
    """
    Calculate coherent current using NEGF with energy-dependent self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float
        Temperature in Kelvin (default: 0)
    spin : str, optional
        Spin configuration ('r' for restricted) (default: 'r')
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    float
        Current in Amperes
    """
    # Create sigma calculator and use checkpointable current calculation
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    return calculate_current(F, S, sigma_calc, fermi=fermi, qV=qV, T=T, spin=spin, dE=dE)

def currentF(fn, dE=ENERGY_STEP, T=TEMPERATURE):
    """
    Calculate current from saved SCF matrix file. 

    Parameters
    ----------
    fn : str
        Filename of .mat file containing SCF data
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    float
        Current in Amperes

    Notes
    -----
    The .mat file should contain:
    - F: Fock matrix
    - S: Overlap matrix
    - sig1, sig2: Contact self-energies
    - fermi: Fermi energy
    - qV: Applied voltage
    - spin: Spin configuration
    """
    matfile = io.loadmat(fn)
    return current(matfile["F"], matfile["S"], matfile["sig1"],matfile["sig2"],
            matfile["fermi"][0,0], matfile["qV"][0,0], T, matfile["spin"][0], dE=dE)

## ENERGY INDEPENDENT SIGMA
def cohTrans(Elist, F, S, sig1, sig2):
    """
    Calculate coherent transmission with energy-independent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sig1 : ndarray
        Left contact self-energy (vector or matrix)
    sig2 : ndarray
        Right contact self-energy (vector or matrix)

    Returns
    -------
    list
        Transmission values at each energy

    Notes
    -----
    Supports both vector and matrix self-energies. For vector self-energies,
    diagonal matrices are constructed internally.
    """
    # Create sigma calculator and use checkpointable transmission calculation
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    transmissions = calculate_transmission(F, S, sigma_calc, Elist, spin='r')

    # Print results to match original behavior
    for E, T in zip(Elist, transmissions):
        print("Energy:",E, "eV, Transmission=", T)

    return transmissions.tolist()

def cohTransSpin(Elist, F, S, sig1, sig2, spin='u'):
    """
    Calculate spin-dependent coherent transmission with energy-independent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix (2N x 2N for spin-unrestricted)
    S : ndarray
        Overlap matrix (2N x 2N for spin-unrestricted)
    sig1 : ndarray
        Left contact self-energy (spin independent vector (1xN) or matrix (NxN), 
                                            spin dependent matrix (2Nx2N))
    sig2 : ndarray
        Right contact self-energy (spin independent vector (1xN) or matrix (NxN), 
                                            spin dependent matrix (2Nx2N))
    spin : str, optional
        Spin basis {'r', 'u', 'ro', or 'g'} (default: 'u')

    Returns
    -------
    tuple
        (Tr, Tspin) where:
        - Tr: Total transmission at each energy
        - Tspin: Array of spin-resolved transmissions [T↑↑, T↑↓, T↓↑, T↓↓]

    Notes
    -----
    For collinear spin calculations ('u' or 'ro'), the matrices are arranged in blocks:
    [F↑↑  0 ]  [S↑↑  0 ]
    [0   F↓↓], [0   S↓↓]
    For generalized spin basis ('g'), each orbital contains a 2x2 spinor block:
    [F↑↑  F↑↓]  [S↑↑  S↑↓]
    [F↓↑  F↓↓], [S↓↑  S↓↓]
    which are then combined into a 2Nx2N matrix.
    """
    # Create sigma calculator and use checkpointable transmission calculation
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    result = calculate_transmission(F, S, sigma_calc, Elist, spin=spin)

    if isinstance(result, tuple):
        transmissions, spin_transmissions = result
        # Print results to match original behavior
        for i, E in enumerate(Elist):
            print("Energy:", E, "eV, Transmission=", transmissions[i], ", Tspin=", spin_transmissions[i])
        return (transmissions.tolist(), spin_transmissions)
    else:
        # Restricted case - no spin resolution
        for E, T in zip(Elist, result):
            print("Energy:", E, "eV, Transmission=", T)
        return (result.tolist(), np.zeros((len(Elist), 4)))

# H0 is an NxN matrix, sig1 and sig2 are Nx1 vectors
def DOS(Elist, F, S, sig1, sig2):
    """
    Calculate density of states with energy-independent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate DOS at
    F : ndarray
        Fock matrix, NxN
    S : ndarray
        Overlap matrix, NxN
    sig1 : ndarray
        Left contact self-energy, vector (1xN) or matrix (NxN)
    sig2 : ndarray
        Right contact self-energy, vector (1xN) or matrix (NxN)

    Returns
    -------
    tuple
        (DOS, DOSList) where:
        - DOS: Total density of states at each energy
        - DOSList: Site-resolved DOS at each energy
    """
    # Create sigma calculator and use checkpointable DOS calculation
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    dos_values, dos_per_site_list = calculate_dos(F, S, sigma_calc, Elist, spin='r')
    return dos_values.tolist(), dos_per_site_list

## ENERGY DEPENDENT SIGMA:

def cohTransE(Elist, F, S, g):
    """
    Calculate coherent transmission with energy-dependent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator

    Returns
    -------
    list
        Transmission values at each energy

    Notes
    -----
    Uses the surface Green's function calculator to compute energy-dependent
    self-energies at each energy point.
    """
    # Create sigma calculator and use checkpointable transmission calculation
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    transmissions = calculate_transmission(F, S, sigma_calc, Elist, spin='r')

    # Print results to match original behavior
    for E, T in zip(Elist, transmissions):
        print("Energy:",E, "eV, Transmission=", T)

    return transmissions.tolist()

def cohTransSpinE(Elist, F, S, g, spin='u'):
    """
    Calculate spin-dependent coherent transmission with energy-dependent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    spin : str, optional
        Spin basis {'r', 'u', 'ro', or 'g'} (default: 'u')

    Returns
    -------
    tuple
        (Tr, Tspin) where:
        - Tr: Total transmission at each energy
        - Tspin: Array of spin-resolved transmissions [T↑↑, T↑↓, T↓↑, T↓↓]
    """
    # Create sigma calculator and use checkpointable transmission calculation
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    result = calculate_transmission(F, S, sigma_calc, Elist, spin=spin)

    if isinstance(result, tuple):
        transmissions, spin_transmissions = result
        # Print results to match original behavior
        for i, E in enumerate(Elist):
            print("Energy:", E, "eV, Transmission=", transmissions[i], ", Tspin=", spin_transmissions[i])
        return transmissions, spin_transmissions
    else:
        # Restricted case - no spin resolution
        for E, T in zip(Elist, result):
            print("Energy:", E, "eV, Transmission=", T)
        return result, np.zeros((len(Elist), 4))

                   
def DOSE(Elist, F, S, g):
    """
    Calculate density of states with energy-dependent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate DOS at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator

    Returns
    -------
    tuple
        (DOS, DOSList) where:
        - DOS: Total density of states at each energy
        - DOSList: Site-resolved DOS at each energy
    """
    # Create sigma calculator and use checkpointable DOS calculation
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    dos_values, dos_per_site_list = calculate_dos(F, S, sigma_calc, Elist, spin='r')

    # Print results to match original behavior
    for E, dos in zip(Elist, dos_values):
        print("Energy:",E, "eV, DOS=", dos)

    return dos_values.tolist(), dos_per_site_list
    
