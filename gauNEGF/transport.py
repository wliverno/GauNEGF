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
from numpy import linalg as LA
import scipy.io as io
from gauNEGF.matTools import *
from gauNEGF import linalg

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



def transmission_single_energy(E, F, S, sigma_calc, spin=None):
    """
    Calculate transmission at a single energy using linalg.py functions.

    Parameters
    ----------
    E : float
        Energy in eV
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
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
    F = np.asarray(F)
    S = np.asarray(S)

    if spin is None:
        spin = 'r'  # Always default to restricted

    # Determine N based on spin configuration
    if spin == 'r':
        N = F.shape[0]
    else:
        N = F.shape[0] // 2

    # Get self-energies and gamma matrices (pass matrix size for spin expansion)
    matrix_size = F.shape[0]
    sigma_total = sigma_calc.get_sigma_total(E, spin, matrix_size)
    gamma1 = sigma_calc.get_gamma(E, 0, spin, matrix_size)  # Left contact
    gamma2 = sigma_calc.get_gamma(E, -1, spin, matrix_size)  # Right contact

    # Calculate Green's function using linalg.inv
    mat = E * S - F - sigma_total
    Gr = linalg.inv(mat)

    if spin == 'r':
        # Simple transmission: T = Tr(Gamma1 @ Gr @ Gamma2 @ Ga)
        Ga = np.conj(Gr).T
        # Use two separate multiplications since linalg.times supports max 3 matrices
        temp = linalg.times(gamma1, Gr, gamma2)
        transmission = np.real(np.trace(linalg.times(temp, Ga)))
        return transmission

    elif spin in ['u', 'ro']:
        # Block-diagonal spin structure (treat 'u' and 'ro' identically)
        Ga = np.conj(Gr).T

        # Extract spin blocks
        Gr_blocks = [Gr[:N, :N], Gr[:N, N:], Gr[N:, :N], Gr[N:, N:]]
        Ga_blocks = [Ga[:N, :N], Ga[:N, N:], Ga[N:, :N], Ga[N:, N:]]
        gamma1_blocks = [gamma1[:N, :N], gamma1[:N, :N], gamma1[N:, N:], gamma1[N:, N:]]
        gamma2_blocks = [gamma2[:N, :N], gamma2[N:, N:], gamma2[:N, :N], gamma2[N:, N:]]

        # Calculate spin-resolved transmissions [T_up_up, T_up_down, T_down_up, T_down_down]
        T_spin = []
        for i in range(4):
            # Use two separate multiplications since linalg.times supports max 3 matrices
            temp = linalg.times(gamma1_blocks[i], Gr_blocks[i], gamma2_blocks[i])
            T_ij = np.real(np.trace(linalg.times(temp, Ga_blocks[i])))
            T_spin.append(T_ij)

        total_transmission = sum(T_spin)
        return total_transmission, T_spin

    elif spin == 'g':
        # Generalized spin basis with 2x2 spinor structure
        Ga = np.conj(Gr).T

        # Extract spinor indices
        a_indices = np.arange(0, 2*N, 2)  # Alpha spin indices
        b_indices = np.arange(1, 2*N, 2)  # Beta spin indices

        Gr_blocks = [Gr[np.ix_(a_indices, a_indices)], Gr[np.ix_(a_indices, b_indices)],
                    Gr[np.ix_(b_indices, a_indices)], Gr[np.ix_(b_indices, b_indices)]]
        Ga_blocks = [Ga[np.ix_(a_indices, a_indices)], Ga[np.ix_(a_indices, b_indices)],
                    Ga[np.ix_(b_indices, a_indices)], Ga[np.ix_(b_indices, b_indices)]]

        # Use diagonal gamma blocks for generalized case
        gamma1_blocks = [gamma1[np.ix_(a_indices, a_indices)], gamma1[np.ix_(a_indices, a_indices)],
                        gamma1[np.ix_(b_indices, b_indices)], gamma1[np.ix_(b_indices, b_indices)]]
        gamma2_blocks = [gamma2[np.ix_(a_indices, a_indices)], gamma2[np.ix_(b_indices, b_indices)],
                        gamma2[np.ix_(a_indices, a_indices)], gamma2[np.ix_(b_indices, b_indices)]]

        T_spin = []
        for i in range(4):
            # Use two separate multiplications since linalg.times supports max 3 matrices
            temp = linalg.times(gamma1_blocks[i], Gr_blocks[i], gamma2_blocks[i])
            T_ij = np.real(np.trace(linalg.times(temp, Ga_blocks[i])))
            T_spin.append(T_ij)

        total_transmission = sum(T_spin)
        return total_transmission, T_spin

    else:
        raise ValueError(f"Unknown spin configuration '{spin}'. Use 'r', 'u', 'ro', or 'g'")


def dos_single_energy(E, F, S, sigma_calc, spin=None):
    """
    Calculate density of states at a single energy using linalg.py functions.

    Parameters
    ----------
    E : float
        Energy in eV
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
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
    F = np.asarray(F)
    S = np.asarray(S)

    if spin is None:
        spin = 'r'

    # Get total self-energy for the appropriate spin case
    matrix_size = F.shape[0]
    sigma_total = sigma_calc.get_sigma_total(E, spin, matrix_size)

    # Calculate Green's function using linalg.inv
    mat = E * S - F - sigma_total
    Gr = linalg.inv(mat)

    if spin == 'r':
        # Restricted case - simple DOS calculation
        dos_per_site = -np.imag(np.diag(Gr)) / np.pi
        total_dos = np.sum(dos_per_site)
        return total_dos, dos_per_site

    elif spin in ['u', 'ro']:
        # Unrestricted/restricted open - split into spin up and down blocks
        N = F.shape[0] // 2

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
        N = F.shape[0] // 2

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


def calculate_transport(F, S, sigma_calculator, energy_list, calculation='transmission',
                       spin=None, fermi=None, qV=None, T=0, **kwargs):
    """
    Unified transport calculation dispatcher with automatic parallelization support.

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
    calculation : str, optional
        Type of calculation: 'transmission', 'current', 'dos' (default: 'transmission')
    spin : str, optional
        Spin configuration using Gaussian labels ('r', 'u', 'ro', 'g')
        If None, defaults to 'r'
    fermi : float, optional
        Fermi energy in eV (required for current calculations)
    qV : float, optional
        Applied bias voltage in eV (required for current calculations)
    T : float, optional
        Temperature in Kelvin (default: 0)
    **kwargs
        Additional parameters passed to calculation functions

    Returns
    -------
    ndarray or tuple
        Results depend on calculation type and spin:
        - 'transmission': array of transmission values (+ spin arrays for open shell)
        - 'current': integrated current value (scalar, + spin currents for open shell)
        - 'dos': (dos_values, dos_per_site_list, [+ spin components for open shell])
    """
    F = np.asarray(F)
    S = np.asarray(S)
    energy_list = np.asarray(energy_list)

    # Handle legacy parameter naming (spin_config -> spin)
    if 'spin_config' in kwargs:
        if spin is None:
            # Map from internal names back to Gaussian labels
            spin_config_map = {'restricted': 'r', 'unrestricted': 'u', 'generalized': 'g'}
            spin = spin_config_map.get(kwargs['spin_config'], 'r')
        kwargs.pop('spin_config')  # Remove to avoid conflicts

    if calculation == 'transmission':
        results = []
        spin_results = []

        for E in energy_list:
            result = transmission_single_energy(E, F, S, sigma_calculator, spin)
            if isinstance(result, tuple):
                # Spin-resolved result
                total_trans, spin_trans = result
                results.append(total_trans)
                spin_results.append(spin_trans)
            else:
                # Scalar result
                results.append(result)

        if spin_results:
            return np.array(results), np.array(spin_results)
        else:
            return np.array(results)

    elif calculation == 'current':
        if fermi is None or qV is None:
            raise ValueError("fermi and qV must be provided for current calculations")

        # Calculate energy window for integration
        muL = fermi - qV/2
        muR = fermi + qV/2

        if T == 0:
            # Zero temperature - integrate between chemical potentials
            if qV > 0:
                integration_energies = energy_list[(energy_list >= muL) & (energy_list <= muR)]
            else:
                integration_energies = energy_list[(energy_list >= muR) & (energy_list <= muL)]
        else:
            # Finite temperature - need broader energy range
            kT = kB * T
            spread = 5 * kT
            integration_energies = energy_list[(energy_list >= min(muL, muR) - spread) &
                                             (energy_list <= max(muL, muR) + spread)]

        # Calculate transmission for integration energies
        transmissions = []
        spin_transmissions = []

        for E in integration_energies:
            result = transmission_single_energy(E, F, S, sigma_calculator, spin)
            if isinstance(result, tuple):
                total_trans, spin_trans = result
                transmissions.append(total_trans)
                spin_transmissions.append(spin_trans)
            else:
                transmissions.append(result)

        transmissions = np.array(transmissions)

        if T == 0:
            # Zero temperature integration
            if spin_transmissions:
                spin_transmissions = np.array(spin_transmissions)
                current_spin = [eoverh * np.trapz(spin_transmissions[:, i], integration_energies)
                               for i in range(4)]
                current_total = eoverh * np.trapz(transmissions, integration_energies)
                return current_total, current_spin
            else:
                current_total = eoverh * np.trapz(transmissions, integration_energies)
                # Apply spin factor for restricted calculations
                if spin == 'r' or spin is None:
                    current_total *= 2
                return current_total
        else:
            # Finite temperature integration with Fermi-Dirac distribution
            dfermi = (1/(np.exp((integration_energies - muR)/(kB*T)) + 1) -
                     1/(np.exp((integration_energies - muL)/(kB*T)) + 1))

            if spin_transmissions:
                spin_transmissions = np.array(spin_transmissions)
                current_spin = [eoverh * np.trapz(spin_transmissions[:, i] * dfermi, integration_energies)
                               for i in range(4)]
                current_total = eoverh * np.trapz(transmissions * dfermi, integration_energies)
                return current_total, current_spin
            else:
                current_total = eoverh * np.trapz(transmissions * dfermi, integration_energies)
                # Apply spin factor for restricted calculations
                if spin == 'r' or spin is None:
                    current_total *= 2
                return current_total

    elif calculation == 'dos':
        dos_values = []
        dos_per_site_list = []
        dos_spin_up_list = []
        dos_spin_down_list = []

        for E in energy_list:
            result = dos_single_energy(E, F, S, sigma_calculator, spin)
            if len(result) == 2:
                # Restricted case: (total_dos, dos_per_site)
                total_dos, dos_per_site = result
                dos_values.append(total_dos)
                dos_per_site_list.append(dos_per_site)
            else:
                # Open shell case: (total_dos, dos_per_site, dos_spin_up, dos_spin_down)
                total_dos, dos_per_site, dos_spin_up, dos_spin_down = result
                dos_values.append(total_dos)
                dos_per_site_list.append(dos_per_site)
                dos_spin_up_list.append(dos_spin_up)
                dos_spin_down_list.append(dos_spin_down)

        if dos_spin_up_list:
            # Return spin-resolved DOS for open shell cases
            return (np.array(dos_values), dos_per_site_list,
                   dos_spin_up_list, dos_spin_down_list)
        else:
            # Return simple DOS for restricted case
            return np.array(dos_values), dos_per_site_list

    else:
        raise ValueError(f"Unknown calculation type: {calculation}")


## CURRENT FUNCTIONS
def current(F, S, sig1, sig2, fermi, qV, T=0, spin="r",dE=0.01):
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
        spread = np.sign(dE)*5*kT
        Elist = np.arange(muL-spread, muR+spread, dE)

    # Map spin notation and create sigma calculator
    spin_config = 'restricted' if spin == 'r' else 'unrestricted'
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)

    # Use unified transport system
    return calculate_transport(F, S, sigma_calc, Elist, calculation='current',
                             spin_config=spin_config, fermi=fermi, qV=qV, T=T)

def currentSpin(F, S, sig1, sig2, fermi, qV, T=0, spin="r",dE=0.01):
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
        spread = np.sign(dE)*5*kT
        Elist = np.arange(muL-spread, muR+spread, dE)

    # Map spin notation and create sigma calculator
    spin_map = {'r': 'restricted', 'u': 'unrestricted', 'ro': 'unrestricted', 'g': 'generalized'}
    spin_config = spin_map.get(spin, 'restricted')
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)

    # Use unified transport system
    result = calculate_transport(F, S, sigma_calc, Elist, calculation='current',
                               spin_config=spin_config, fermi=fermi, qV=qV, T=T)

    if isinstance(result, tuple):
        # Return spin currents if available
        return result[1]
    else:
        # Return zero spin currents for restricted case
        return [0, 0, 0, 0]


def currentE(F, S, g, fermi, qV, T=0, spin="r",dE=0.01):
    """
    Calculate coherent current at T=0K using NEGF with energy-dependent self-energies.

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
        spread = np.sign(dE)*5*kT
        Elist = np.arange(muL-spread, muR+spread, dE)

    # Map spin notation and create sigma calculator
    spin_config = 'restricted' if spin == 'r' else 'unrestricted'
    sigma_calc = SigmaCalculator(g, energy_dependent=True)

    # Use unified transport system
    return calculate_transport(F, S, sigma_calc, Elist, calculation='current',
                             spin_config=spin_config, fermi=fermi, qV=qV, T=T)

def currentF(fn, dE=0.01, T=0):
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
    # Create sigma calculator and use unified transport system
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    transmissions = calculate_transport(F, S, sigma_calc, Elist,
                                      calculation='transmission', spin_config='restricted')

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
    # Map spin notation to unified system
    spin_map = {'r': 'restricted', 'u': 'unrestricted', 'ro': 'unrestricted', 'g': 'generalized'}
    spin_config = spin_map.get(spin, 'unrestricted')

    # Create sigma calculator and use unified transport system
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    result = calculate_transport(F, S, sigma_calc, Elist,
                               calculation='transmission', spin_config=spin_config)

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
    # Create sigma calculator and use unified transport system
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    dos_values, dos_per_site_list = calculate_transport(F, S, sigma_calc, Elist, calculation='dos')
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
    # Create sigma calculator and use unified transport system
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    transmissions = calculate_transport(F, S, sigma_calc, Elist,
                                      calculation='transmission', spin_config='restricted')

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
    # Map spin notation to unified system
    spin_map = {'r': 'restricted', 'u': 'unrestricted', 'ro': 'unrestricted', 'g': 'generalized'}
    spin_config = spin_map.get(spin, 'unrestricted')

    # Create sigma calculator and use unified transport system
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    result = calculate_transport(F, S, sigma_calc, Elist,
                               calculation='transmission', spin_config=spin_config)

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
    # Create sigma calculator and use unified transport system
    sigma_calc = SigmaCalculator(g, energy_dependent=True)
    dos_values, dos_per_site_list = calculate_transport(F, S, sigma_calc, Elist, calculation='dos')

    # Print results to match original behavior
    for E, dos in zip(Elist, dos_values):
        print("Energy:",E, "eV, DOS=", dos)

    return dos_values.tolist(), dos_per_site_list
    
