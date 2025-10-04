"""
Density matrix calculation methods for quantum transport simulations.

This module provides functions for calculating density matrices in quantum transport
calculations using various integration methods:

    - Analytical integration for energy-independent self-energies
    - Complex contour integration for equilibrium calculations
    - Real-axis integration for non-equilibrium calculations
    - Parallel processing support for large systems


Notes
-----
The module supports both serial and parallel computation modes, automatically
selecting the most efficient method based on system size and available resources.
Temperature effects are included through Fermi-Dirac statistics.
"""

# Numerical Packages
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# Configuration
from gauNEGF.config import (TEMPERATURE, ADAPTIVE_INTEGRATION_TOL, FERMI_CALCULATION_TOL, 
                            FERMI_SEARCH_CYCLES, SCF_CONVERGENCE_TOL, N_KT, ENERGY_MIN, 
                            MAX_CYCLES, MAX_GRID_POINTS)
from scipy.special import roots_legendre
from scipy.special import roots_chebyu
import matplotlib.pyplot as plt
import warnings

# Parallelization packages
from multiprocessing import Pool
import os

# Developed Packages:
from gauNEGF.fermiSearch import DOSFermiSearch
from gauNEGF.surfG1D import surfG
from gauNEGF.integrate import GrInt, GrLessInt
# Use JAX functions directly - no wrappers needed

# JIT-compiled density matrix kernels
@jit
def _compute_dos_at_energy(E, F, S, sigma_total):
    """JIT-compiled kernel for DOS calculation at single energy."""
    mat = E * S - F - sigma_total
    Gr = jnp.linalg.inv(mat)
    dos_per_site = -jnp.imag(jnp.diag(Gr)) / jnp.pi
    total_dos = jnp.sum(dos_per_site)
    return total_dos, dos_per_site

@jit
def _compute_green_function(E, F, S, sigma_total):
    """JIT-compiled kernel for Green's function calculation."""
    mat = E * S - F - sigma_total
    return jnp.linalg.inv(mat)

@jit
def _fermi_vectorized(E_array, mu, kT):
    """JIT-compiled vectorized Fermi-Dirac distribution."""
    # Handle zero temperature case
    def finite_temp():
        return 1.0 / (jnp.exp((E_array - mu) / kT) + 1.0)

    def zero_temp():
        return (E_array <= mu).astype(jnp.float32)

    return jax.lax.cond(kT == 0.0, zero_temp, finite_temp)

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
kB = 8.617e-5           # eV/Kelvin


## HELPER FUNCTIONS
def fermi(E, mu, T):
    """
    Calculate Fermi-Dirac distribution.

    Parameters
    ----------
    E : float
        Energy in eV
    mu : float
        Chemical potential in eV
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        Fermi-Dirac occupation number
    """
    kT = kB*T
    if kT==0:
        return (E<=mu)*1
    else:
        return 1/(np.exp((E - mu)/kT)+1)

def getANTPoints(N):
    """
    Generate integration points and weights matching ANT.Gaussian implementation.

    Follows the IntCompPlane subroutine in device.F90 from ANT.Gaussian package.
    Uses a modified Gauss-Chebyshev quadrature scheme optimized for transport
    calculations. _Note: Always generates an even number of points._

    Parameters
    ----------
    N : int
        Number of integration points

    Returns
    -------
    tuple
        (points, weights) - Arrays of integration points and weights
    """
    k = np.arange(1,N+1,2)
    theta = k*np.pi/(2*N)
    xs = np.sin(theta)
    xcc = np.cos(theta)

    # Transform points using ANT-like formula
    x = 1.0 + 0.21220659078919378103 * xs * xcc * (3 + 2*xs*xs) - k/(N)
    x = np.concatenate((x,-1*x))
    
    # Generate weights similarly to ANT
    w = xs**4 * 16.0/(3*(N))
    w = np.concatenate((w, w))

    return x, w

def integratePoints(computePointFunc, numPoints, parallel=False, numWorkers=None, 
                   chunkSize=None, debug=False):
    """
    Perform parallel or serial integration for quantum transport calculations.

    This function provides a flexible integration framework that automatically
    chooses between numpy's built-in parallelization for matrix operations
    and process-level parallelization for large workloads.

    Parameters
    ----------
    computePointFunc : callable
        Function that computes a single integration point. Should take an
        integer index i and return a matrix/array.
    numPoints : int
        Total number of points to integrate
    parallel : bool, optional
        Whether to force process-level parallel processing (default: False)
    numWorkers : int, optional
        Number of worker processes. If None, automatically determined based
        on system resources and workload.
    chunkSize : int, optional
        Size of chunks for parallel processing. If None, automatically
        optimized based on numPoints and numWorkers.
    debug : bool, optional
        Whether to print debug information (default: False)

    Returns
    -------
    ndarray
        Sum of all computed points

    Notes
    -----
    - Automatically detects SLURM environment for HPC compatibility
    - Falls back to serial processing if parallel execution fails
    - Uses numpy's built-in parallelization for small workloads
    - Switches to process-level parallelization for:
        * Large number of points (≥100)
        * Many available cores (≥32)
        * When explicitly requested via parallel=True
    """
    # Get SLURM CPU count if available
    numCores = int(os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count()))
    
    if debug:
        print(f'Number of points to integrate: {numPoints}')
        print(f'Number of CPU cores: {numCores}')
    
    # Use process-level parallelization for large workloads when requested
    useProcessParallel = parallel and (
        numPoints >= 100 and numCores >= 32
    )
    
    # Standard case: Use numpy's built-in parallelization
    if not useProcessParallel:
        if debug:
            print('Using numpy built-in parallelization for matrix operations')
        result = np.zeros_like(computePointFunc(0))
        for i in range(int(numPoints)):
            result += computePointFunc(i)
        return result
    
    # Parallel case
    if debug:
        print('Using process-level parallelization')
        
    if numWorkers is None:
        numWorkers = max(1, numCores // 16)
    
    if chunkSize is None:
        chunkSize = max(1, min(numPoints // (numWorkers * 4), 100))
        
    if debug:
        print(f'Workers: {numWorkers}, Chunk size: {chunkSize}')
    
    def process_chunk(points):
        return sum(computePointFunc(i) for i in points)

    # Create chunks of indices
    chunks = [range(i, min(i + chunkSize, numPoints)) 
             for i in range(0, numPoints, chunkSize)]

    with Pool(numWorkers) as pool:
        try:
            results = pool.map(process_chunk, chunks)
            return sum(results)
        except (AttributeError, TypeError):
            # Fallback to sequential processing if parallel fails
            return sum(process_chunk(chunk) for chunk in chunks)
def integratePointsAdaptiveANT(computePoint, tol=ADAPTIVE_INTEGRATION_TOL, maxN=MAX_GRID_POINTS, debug=False):
    """
    Adaptive integration using ANT-modified Gauss-Chebyshev quadrature (IntCompPlane subroutine from ANT.Gaussian package)

    Parameters
    ----------
    computePoint : callable
        Function that computes integral over a list of weights and points. Should return a matrix/array.
    tol : float, optional
        Tolerance for the adaptive integration.
    maxN : int, optional
        Maximum number of points to integrate.
    debug : bool, optional
        Whether to print debug information.

    Returns
    -------
    ndarray
        Integral of the function.
    """
    prev_x = None
    prev_sumW = None
    P = None
    N = 2
    maxDP = 1e10
    while N<=maxN:
        x, w = getANTPoints(N)

        if prev_x is None:
            # first level: no reuse
            P = computePoint(x[0:2], w[0:2])
        else:
            # mark old nodes robustly by value
            old_mask = np.isin(np.round(x, 14), np.round(prev_x, 14))
            # sanity: all previous nodes should be found
            assert int(old_mask.sum()) == prev_x.size, "Old nodes mismatch"

            # exact transfer factor (should be ~1/3)
            ratio = float(np.sum(w[old_mask]) / prev_sumW)

            # scale previous integral + add only new-node contributions
            new_mask = ~old_mask
            new_P = P*ratio
            new_P += computePoint(x[new_mask], w[new_mask])
            maxDP = np.max(np.abs(new_P-P))
            if debug:
                P_debug = computePoint(x, w)
                maxDP_debug = np.max(np.abs(P_debug-P))
                maxDiff = np.max(np.abs(P_debug-new_P))
                print(f"N={N}, nested-weight ratio ~ {ratio:.3f}, maxDP={maxDP:.3e}")
                print(f"Direct Calculation: N={N}, maxDP={maxDP_debug:.3e}, maxDiff={maxDiff:.3e}")
            P = new_P.copy()
            if maxDP<tol:
                print(f'Adaptive integration converged to {maxDP:.3e} in {N} points.')
                return new_P

        # update state for next level
        prev_x = x
        prev_sumW = float(np.sum(w))
        N *= 3
    N/=3
    print(f'Adaptive integration reached full grid ({N} points), final error {maxDP:.3e}')
    return new_P

## ENERGY INDEPENDENT DENSITY FUNCTIONS
def density(V, Vc, D, Gam, Emin, mu):
    """
    Calculate density matrix using analytical integration for energy-independent self-energies.

    Implements the analytical integration method from Eq. 27 in PRB 65, 165401 (2002).
    The method assumes energy-independent self-energies and uses the spectral function
    representation of the density matrix.

    Parameters
    ----------
    V : ndarray
        Eigenvectors of Fock matrix in orthogonalized basis
    Vc : ndarray
        Inverse conjugate transpose of V
    D : ndarray
        Eigenvalues of Fock matrix
    Gam : ndarray
        Broadening matrix Γ = i[Σ - Σ†] in orthogonalized basis
    Emin : float
        Lower bound for integration in eV
    mu : float
        Chemical potential in eV

    Returns
    -------
    ndarray
        Density matrix in orthogonalized basis

    Notes
    -----
    The integration is performed analytically using the residue theorem.
    The result includes contributions from poles below the Fermi energy.
    """
    Nd = len(V)
    DD = np.array([D for i in range(Nd)]).T
    
    #Integral of 1/x is log(x), calculating at lower and upper limit
    logmat = np.array([np.emath.log(1-(mu/D)) for i in range(Nd)]).T
    logmat2 = np.array([np.emath.log(1-(Emin/D)) for i in range(Nd)]).T

    #Compute integral, add prefactor
    invmat = 1/(2*np.pi*(DD-DD.conj().T))
    pref2 = logmat - logmat.conj().T
    pref3 = logmat2-logmat2.conj().T

    prefactor = np.multiply(invmat,(pref2-pref3))

    #Convert Gamma into Fbar eigenbasis, element-wise multiplication
    Gammam = Vc.conj().T@Gam@Vc
    prefactor = np.multiply(prefactor,Gammam)
    
    #Convert back to input basis, return
    den = V@ prefactor @ V.conj().T
    return den

def bisectFermi(V, Vc, D, Gam, Nexp, conv=SCF_CONVERGENCE_TOL, Eminf=ENERGY_MIN):
    """
    Find Fermi energy using bisection method.

    Uses bisection to find the Fermi energy that gives the expected number
    of electrons. The search is performed using the analytical density matrix
    calculation.

    Parameters
    ----------
    V : ndarray
        Eigenvectors of Fock matrix in orthogonalized basis
    Vc : ndarray
        Inverse conjugate transpose of V
    D : ndarray
        Eigenvalues of Fock matrix
    Gam : ndarray
        Broadening matrix Γ = i[Σ - Σ†] in orthogonalized basis
    Nexp : float
        Expected number of electrons
    conv : float, optional
        Convergence criterion for electron number (default: 1e-3)
    Eminf : float, optional
        Lower bound for integration in eV (default: -1e6)

    Returns
    -------
    float
        Fermi energy in eV that gives the expected electron count

    Notes
    -----
    The search is bounded by the minimum and maximum eigenvalues.
    Warns if maximum iterations reached without convergence.
    """
    Emin = min(D.real)
    Emax = max(D.real)
    dN = Nexp
    Niter = 0
    while abs(dN) > conv and Niter<1000:
        fermi = (Emin + Emax)/2
        P = density(V, Vc, D, Gam, Eminf, fermi)
        dN = np.trace(P).real - Nexp
        if dN>0:
            Emax = fermi
        else:
            Emin = fermi
        Niter += 1
    if Niter >= 1000:
        print('Warning: Bisection search timed out after 1000 iterations!')
    print(f'Bisection fermi search converged to {dN:.2E} in {Niter} iterations.')
    return fermi

## ENERGY DEPENDENT DENSITY FUNCTIONS
def densityRealN(F, S, g, Emin, mu, N=100, T=TEMPERATURE, showText=True):
    """
    Calculate equilibrium density matrix using real-axis integration on a specified grid.

    Performs numerical integration along the real energy axis using Gauss-Legendre
    quadrature. Suitable for equilibrium calculations with energy-dependent
    self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    Emin : float
        Lower bound for integration in eV
    mu : float
        Chemical potential in eV
    N : int, optional
        Number of integration points (default: 100)
    T : float, optional
        Temperature in Kelvin (default: 300)
    showText : bool, optional
        Whether to print progress messages (default: True)

    Returns
    -------
    ndarray
        Density matrix
    """
    nKT = N_KT
    kT = kB*T
    Emax = mu + nKT*kT
    mid = (Emax-Emin)/2
    defInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    
    Elist = mid*(x + 1) + Emin
    weights = mid*w*fermi(Elist, mu, T)
    
    if showText:
        print(f'Integrating {N} points along real axis...')

    defInt = GrInt(F, S, g, Elist, weights)

    if showText:
        print('Integration done!')
    
    return (-1+0j)*np.imag(defInt)/(np.pi)

def densityReal(F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, maxN=MAX_CYCLES, debug=False):
    """
    Calculate equilibrium density matrix using adaptive real-axis integration.

    Wrapper for densityRealN() using the tol and maxN specification to determine grid size

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    Emin : float
        Lower bound for integration in eV
    mu : float
        Chemical potential in eV
    tol : float, optional
        Convergence tolerance (default: 1e-3)
    T : float, optional
        Temperature in Kelvin (default: 300)
    maxN : int, optional
        Maximum number of integration points (default: 1000)
    debug : bool, optional
        Whether to print per-iteration diagnostics (default: False)

    Returns
    -------
    ndarray
        Density matrix
    """
    P = np.zeros_like(F)
    N = 1
    maxDP = 1e9
    while N<maxN:
        P_ = P.copy()

        P = densityRealN(F, S, g, Emin, mu, N, T, showText=False)
        maxDP = np.max(np.abs(P - P_))
        if maxDP< tol:
            print(f'Adaptive integration converged to {maxDP:.3e} in {N} points.')
            return P
        N *= 2

    print(f'Warning: adaptive integration not converged after {maxN} points: maxDP={maxDP:.2E}')
    return P
   

def densityGridN(F, S, g, mu1, mu2, ind=None, N=100, T=TEMPERATURE, showText=True):
    """
    Calculate non-equilibrium density matrix using real-axis integration.

    Performs numerical integration for the non-equilibrium part of the density
    matrix when a bias voltage is applied. Uses vectorized integration for efficiency.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    mu1 : float
        Left contact chemical potential in eV
    mu2 : float
        Right contact chemical potential in eV
    ind : int or None, optional
        Contact index (None for total) (default: None)
    N : int, optional
        Number of integration points (default: 100)
    T : float, optional
        Temperature in Kelvin (default: 300)
    showText : bool, optional
        Whether to print progress messages (default: True)

    Returns
    -------
    ndarray
        Non-equilibrium contribution to density matrix
    """
    nKT = N_KT
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    dInt = np.sign(mu2 - mu1) # Sign of bias voltage
    Emax = muHi + nKT*kT
    Emin = muLo - nKT*kT
    mid = (Emax-Emin)/2
    den = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    
    energies = mid*(x + 1) + Emin
    dfermi = fermi(energies, muHi, T) - fermi(energies, muLo, T)
    weights = mid*w*dfermi*dInt

    if showText:
        print(f'Real integration over {N} points...')
    
    den = GrLessInt(F, S, g, energies, weights, ind)

    if showText:
        print('Integration done!')
 
    return den/(2*np.pi)

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGridTrap(F, S, g, mu1, mu2, ind=None, N=100, T=TEMPERATURE):
    """
    Calculate non-equilibrium density matrix using trapezoidal integration on a real energy grid.

    Alternative implementation to densityGridN() using trapezoidal rule instead of 
    Gauss-Legendre quadrature. Provides direct loop-based integration for comparison.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    mu1 : float
        Left contact chemical potential in eV
    mu2 : float
        Right contact chemical potential in eV
    ind : int or None, optional
        Contact index (None for total) (default: None)
    N : int, optional
        Number of integration points (default: 100)
    T : float, optional
        Temperature in Kelvin (default: 300)

    Returns
    -------
    ndarray
        Non-equilibrium contribution to density matrix
    """
    nKT = N_KT
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    dInt = np.sign(mu2 - mu1) # Sign of bias voltage
    Emax = muHi + nKT*kT
    Emin = muLo - nKT*kT
    Egrid = np.linspace(Emin, Emax, N)
    den = np.array(np.zeros(np.shape(F)), dtype=complex)
    print(f'Real integration over {N} points...')
    for i in range(1,N):
        E = (Egrid[i] + Egrid[i-1])/2
        dE = Egrid[i] - Egrid[i-1]
        mat = E * S - F - jnp.array(g.sigmaTot(E))
        GrE = jnp.linalg.inv(mat)
        GaE = GrE.conj().T
        if ind == None:
            sig = g.sigmaTot(E)
        else:
            sig = g.sigma(E, ind)
        Gamma = 1j*(sig - sig.conj().T)
        dFermi = fermi(E, muHi, T) - fermi(E, muLo, T)
        den += (GrE@Gamma@GaE)*dFermi*dE*dInt
    print('Integration done!')
    
    return den/(2*np.pi)

def densityGrid(F, S, g, mu1, mu2, ind=None, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False):
    """
    Calculate non-equilibrium density matrix using real-axis integration.

    Performs numerical integration for the non-equilibrium part of the density
    matrix when a bias voltage is applied. Uses ANT-modified Gauss-Chebyshev quadrature.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    mu1 : float
        Left contact chemical potential in eV
    mu2 : float
        Right contact chemical potential in eV
    ind : int or None, optional
        Contact index (None for total) (default: None)
    tol : float, optional
        Convergence tolerance (default: 1e-3)
    T : float, optional
        Temperature in Kelvin (default: 300)
    debug : bool, optional
        Whether to print debug information (default: False)

    Returns
    -------
    ndarray
        Non-equilibrium contribution to density matrix
    """
    nKT = N_KT
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    dInt = np.sign(mu2 - mu1) # Sign of bias voltage
    Emax = muHi + nKT*kT
    Emin = muLo - nKT*kT
    mid = (Emax-Emin)/2
    den = np.array(np.zeros(np.shape(F)), dtype=complex)
    
    def computePoint(x, w):
        E = mid*(x + 1) + Emin
        dFermi = fermi(E, muHi, T) - fermi(E, muLo, T)
        weights = mid*w*dFermi*dInt
        return GrLessInt(F, S, g, E, weights, ind)
     
    den = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)
    if debug:
        print('Integration done!')

    return den/(2*np.pi)

def densityComplexN(F, S, g, Emin, mu, N=100, T=TEMPERATURE, showText=True, method='ant'):
    """
    Calculate equilibrium density matrix using complex contour integration.

    Performs numerical integration along a complex contour that encloses the
    poles of the Fermi function. More efficient than real-axis integration
    for equilibrium calculations. Uses vectorized integration for efficiency.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    Emin : float
        Lower bound for integration in eV
    mu : float
        Chemical potential in eV
    N : int, optional
        Number of integration points (default: 100)
    T : float, optional
        Temperature in Kelvin (default: 300)
    showText : bool, optional
        Whether to print progress messages (default: True)
    method : {'ant', 'legendre', 'chebyshev'}, optional
        Integration method to use (default: 'ant')

    Returns
    -------
    ndarray
        Equilibrium density matrix

    Notes
    -----
    The 'ant' method uses a modified Gauss-Chebyshev quadrature optimized
    for transport calculations, matching the ANT.Gaussian implementation.
    """
    #Construct circular contour
    nKT= 10
    broadening = nKT*kB*T
    Emax = mu-broadening
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2
    
    if method == 'legendre':
        x, w = roots_legendre(N)
    elif method == 'chebyshev':
        k = np.arange(1, N+1)
        x = np.cos(k * np.pi / (N+1))
        w = (np.pi/(N+1))*(np.sin(k * np.pi / (N+1))**2) /np.sqrt(1-(x**2))
    elif method == 'ant':
        x, w = getANTPoints(N)
    else: # Midpoint rule
        x = np.linspace(-1, 1, N)
        w = 2*np.ones(N)/N
    
    #Integrate along contour
    theta = np.pi/2 * (x + 1)
    Elist = center + r*np.exp(1j*theta)
    dz = 1j * r * np.exp(1j*theta)
    weights = (np.pi/2)*w*fermi(Elist, mu, T)*dz

    if showText:
        print(f'Complex Integration over {N} points...')

    lineInt = GrInt(F, S, g, Elist, weights)
    
    #Add integration points for Fermi Broadening
    if T>0:
        if showText:
            print('Integrating Fermi Broadening')
        Nbroad = int(N//8)
        # Use Legendre or trapezoidal rule for real axis integration
        if method == 'legendre' or method == 'chebyshev' or method == 'ant':
            x_fermi, w_fermi = roots_legendre(Nbroad)
        else: # Trapezoidal rule
            x_fermi = np.linspace(-1, 1, Nbroad)
            w_fermi = 2*np.ones(Nbroad)/Nbroad
        Elist = broadening * (x_fermi) + mu
        weights = broadening*w_fermi*fermi(Elist, mu, T)
        lineInt += GrInt(F, S, g, Elist, weights)

    if showText:
        print('Integration done!')

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi

def densityComplex(F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False):
    """
    Calculate equilibrium density matrix using complex contour integration.

    Performs numerical integration along a complex contour that encloses the
    poles of the Fermi function. More efficient than real-axis integration
    for equilibrium calculations. Uses adaptive integration.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    Emin : float
        Lower bound for integration in eV
    mu : float
        Chemical potential in eV
    tol : float, optional
        Convergence tolerance (default: 1e-3)
    T : float, optional
        Temperature in Kelvin (default: 300)
    debug : bool, optional
        Whether to print debug information (default: False)

    Returns
    -------
    ndarray
        Equilibrium density matrix

    Notes
    -----
    The 'ant' method uses a modified Gauss-Chebyshev quadrature optimized
    for transport calculations, matching the ANT.Gaussian implementation.
    """
    #Construct circular contour
    nKT= 10
    broadening = nKT*kB*T
    Emax = mu-broadening
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2

    # For ANT adaptive integration, compute from point-weight pairs
    def computePoint(x, w):
        theta = np.pi/2 * (x + 1)
        z = center + r*np.exp(1j*theta)
        dz = 1j * r * np.exp(1j*theta)
        weights = (np.pi/2)*w*dz*fermi(z, mu, T)
        return GrInt(F, S, g, z, weights)
    
    print('Complex Contour Integration:')
    lineInt = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)
    
    #Add integration points for Fermi Broadening
    if T>0:
        print('Integrating Fermi Broadening:')
        def computePointBroadening(x, w):
            E = broadening * (x) + mu
            weights = broadening*w*fermi(E, mu, T)
            return GrInt(F, S, g, E, weights)
    
        lineInt += integratePointsAdaptiveANT(computePointBroadening, tol=tol, debug=debug)

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi


## INTEGRATION LIMIT FUNCTIONS
# Calculate Emin using DOS
def calcEmin(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES):
    D,_ = jnp.linalg.eig(jnp.linalg.inv(S)@F)
    Emin = min(D.real.flatten())-5
    counter = 0
    mat = Emin * S - F - jnp.array(g.sigmaTot(Emin))
    gr = jnp.linalg.inv(mat)
    dP = -jnp.imag(jnp.trace(gr)) / jnp.pi
    while dP>tol and counter<maxN:
        Emin -= 1
        mat = Emin * S - F - jnp.array(g.sigmaTot(Emin))
        gr = jnp.linalg.inv(mat)
        dP = abs(-jnp.imag(jnp.trace(gr)) / jnp.pi)
        #print(Emin, dP)
        counter += 1
    if counter == maxN:
        print(f'Warning: Emin still not within tolerance (final value = {dP}) after {maxN} energy samples')
    print(f'Calculated Emin: {Emin} eV, DOS = {dP:.2E}')
    return Emin

def integralFit(F, S, g, mu, Eminf=ENERGY_MIN, tol=FERMI_CALCULATION_TOL, T=TEMPERATURE, maxN=MAX_CYCLES):
    """
    Optimize integration parameters for density calculations.

    Determines optimal integration parameters by iteratively testing
    convergence of the density matrix calculation.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    mu : float
        Equilibrium contact fermi energy in eV
    Eminf : float
        Lower bound for integration in eV (default: -1e6)
    tol : float, optional
        Convergence tolerance (default: 1e-5)
    T : float
        Temperature in Kelvin for Fermi broadening (default: 0)
    maxN : int, optional
        Max grid points and Emin search iterations(default: 1000)

    Returns
    -------
    tuple
        (Emin, N1, N2) - Optimized integration parameters:
        - Emin: Lower bound for complex contour
        - N1: Number of complex contour points
        - N2: Number of real axis points

    Notes
    -----
    The optimization process:
    1. Finds Emin by checking DOS convergence
    2. Optimizes N1 for complex contour integration
    3. Optimizes N2 for real axis integration
    """
    # Calculate Emin using DOS
    Emin = calcEmin(F, S, g, tol, maxN)

    #Determine grid using dP
    Ncomplex = 4
    dP = np.inf
    rho = np.zeros(np.shape(F))
    while dP > tol and Ncomplex < maxN:
        Ncomplex *= 2 # Start with 8 points, double each time
        rho_ = np.real(densityComplexN(F, S, g, Emin,  mu, Ncomplex, T=T))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}, N = {sum(np.diag(rho_).real):2f}")
        rho = rho_
    if dP < tol:
        Ncomplex /= 2
    elif Ncomplex >= maxN and dP > tol:
        print(f'Warning: Ncomplex still not within tolerance (final value = {dP})')
    print(f'Final Ncomplex: {Ncomplex}') 

    #Determine grid using dP
    counter = 0
    Nreal = 8
    dP = np.inf
    rho = np.zeros(np.shape(F))
    while dP > tol and Nreal < maxN:
        Nreal *= 2 # Start with 16 points, double each time
        rho_ = np.real(densityRealN(F, S, g, Eminf, Emin, Nreal, T=0))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}")
        rho = rho_
        counter += 1
    if dP < tol:
        Nreal /= 2
    elif Nreal >=  maxN and dP > tol:
        print(f'Warning: Nreal still not within tolerance (final value = {dP})')
    print(f'Final Nreal: {Nreal}') 

    return Emin, Ncomplex, Nreal

def integralFitNEGF(F, S, g, fermi, qV, Eminf=ENERGY_MIN, tol=FERMI_CALCULATION_TOL, T=TEMPERATURE, maxGrid=MAX_GRID_POINTS):
    """
    Determines number of grid points for non-equilibrium density calculations.

    Same procedure as `integralFit()` but applied to `densityGrid()`

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    fermi : float
        Equilibrium Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    Eminf : float
        Lower bound for integration in eV (default: -1e6)
    tol : float, optional
        Convergence tolerance (default: 1e-5)
    T : float
        Temperature in Kelvin for Fermi broadening (default: 0)
    maxGrid : int, optional
        Maximum number of gridpoints (default: 1000)

    Returns
    -------
    int
        Number of grid points
    """
    #Determine grid using dP
    N = 8
    dP = np.inf
    rho = np.zeros(np.shape(F))
    while dP > tol and N < maxGrid:
        N *= 2 # Start with 16 points, double each time
        rho_ = np.real(densityGridN(F, S, g, fermi, fermi+(qV/2), ind=0, N=N, T=T))
        rho_ += np.real(densityGridN(F, S, g, fermi, fermi-(qV/2), ind=-1, N=N, T=T))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}")
        rho = rho_
    if dP < tol:
        N /= 2
    elif N >= maxGrid and dP > tol:
        print(f'Warning: N still not within tolerance (final value = {dP})')
    print(f'Final Nnegf: {N}') 
    return N


def getFermiContact(g, ne, tol=FERMI_CALCULATION_TOL, Eminf=ENERGY_MIN, maxcycles=MAX_CYCLES, T=TEMPERATURE, nOrbs=0):
    """
    Calculate Fermi energy for a contact.

    Determines the Fermi energy for a contact system (Bethe lattice or 1D chain)
    by matching the electron count.

    Parameters
    ----------
    g : surfG object
        Surface Green's function calculator
    ne : float
        Target number of electrons
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    Eminf : float, optional
        Lower bound for integration (default: -1e6)
    maxcycles : int, optional
        Maximum number of iterations (default: 1000)
    nOrbs : int, optional
        Number of orbitals to consider (0 for all) (default: 0)

    Returns
    -------
    float
        Fermi energy in eV
    """
    # Set up infinite system from contact
    S = g.S
    F = g.F
    orbs, _ = jnp.linalg.eig(jnp.linalg.inv(S)@F)
    orbs = np.sort(np.real(orbs))
    fermi = (orbs[int(ne)-1] + orbs[int(ne)])/2
    Emin, N1, N2 = integralFit(F, S, g, fermi, Eminf, tol, T, maxN=maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, 
                        Eminf, T, tol, maxcycles, nOrbs)[0]

def getFermi1DContact(gSys, ne, ind=0, tol=FERMI_CALCULATION_TOL, Eminf=ENERGY_MIN, T=TEMPERATURE, maxcycles=MAX_CYCLES):
    """
    Calculate Fermi energy for a 1D chain contact.

    Specialized version of getFermiContact for 1D chain contacts, handling
    the periodic boundary conditions correctly.

    Parameters
    ----------
    gSys : surfG object
        Surface Green's function calculator for the full system
    ne : float
        Target number of electrons per unit cell
    ind : int, optional
        Contact index (0 for left, -1 for right) (default: 0)
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    Eminf : float, optional
        Lower bound for integration (default: -1e6)
    maxcycles : int, optional
        Maximum number of iterations (default: 1000)

    Returns
    -------
    tuple
        (fermi, Emin, N1, N2) - Optimized parameters:
        - fermi: Calculated Fermi energy in eV
        - Emin: Lower bound for complex contour
        - N1: Number of complex contour points
        - N2: Number of real axis points
    """
    # Set up infinite system from contact
    F = gSys.aList[ind]
    S = gSys.aSList[ind]
    tau = gSys.bList[ind]
    stau = gSys.bSList[ind]
    inds = np.arange(len(F))
    g = surfG(F, S, [inds], [tau], [stau], eta=1e-6)

    # Initial guess and integral setup using two layers
    Forbs = np.block([[F, tau], [tau.conj().T, F]])
    Sorbs = np.block([[S, stau], [stau.T, S]])
    gorbs = surfG(Forbs, Sorbs, [inds], [tau], [stau], eta=1e-6)
    orbs, _ = jnp.linalg.eig(jnp.linalg.inv(Sorbs)@Forbs)
    orbs = np.sort(np.real(orbs))
    fermi = (orbs[2*int(ne)-1] + orbs[2*int(ne)])/2
    Emin, N1, N2 = integralFit(Forbs, Sorbs, gorbs, fermi, Eminf, tol, T, maxN=maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, Eminf, T, tol, maxcycles)

# Calculate the fermi energy of the surface Green's Function object
def calcFermi(g, ne, Emin, Emax, fermiGuess=0, N1=100, N2=50, Eminf=ENERGY_MIN, T=TEMPERATURE, tol=FERMI_CALCULATION_TOL, maxcycles=MAX_CYCLES, nOrbs=0):
    """
    Calculate Fermi energy using bisection method.

    Parameters
    ----------
    g : surfG object
        Surface Green's function calculator
    ne : float
        Target number of electrons
    Emin : float
        Lower bound for complex contour in eV
    Emax : float
        Upper bound for search in eV
    fermiGuess : float, optional
        Initial guess for Fermi energy in eV (default: 0)
    N1 : int, optional
        Number of complex contour points (default: 100)
    N2 : int, optional
        Number of real axis points (default: 50)
    Eminf : float, optional
        Lower bound for real axis integration in eV (default: -1e6)
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    maxcycles : int, optional
        Maximum number of iterations (default: 20)
    nOrbs : int, optional
        Number of orbitals to consider, 0 for all (default: 0)

    Returns
    -------
    tuple
        (fermi, Emin, N1, N2) - Optimized parameters:
        - fermi: Calculated Fermi energy in eV
        - Emin: Lower bound for complex contour
        - N1: Number of complex contour points
        - N2: Number of real axis points
    """
    # Fermi Energy search using full contact
    mat = Eminf * g.S - g.F - jnp.array(g.sigmaTot(Eminf))
    gr = jnp.linalg.inv(mat)
    dos_eminf = -jnp.imag(jnp.trace(gr)) / jnp.pi
    print(f'Eminf DOS = {dos_eminf}')
    fermi = fermiGuess
    if N2 is None:
        pLow = densityReal(g.F, g.S, g, Eminf, Emin, tol, T, showText=False)
    else:
        pLow = densityRealN(g.F, g.S, g, Eminf, Emin, N2, T, showText=False)
    if nOrbs==0:
        nELow = np.trace(pLow@g.S)
    else:
        nELow = np.trace((pLow@g.S)[-nOrbs:, -nOrbs:])
    print(f'Electrons below lowest onsite energy: {nELow}')
    if nELow >= ne:
        raise Exception('Calculated Fermi energy is below lowest orbital energy!')
    if N1 is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T, showText=False, method='legendre')
    else:
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N1, T, showText=False, method='legendre')
    
    # Fermi search using bisection method (F not changing, highly stable)
    Ncurr = -1
    counter = 0 
    lBound = Emin
    uBound = Emax
    print('Calculating Fermi energy using bisection:')
    while abs(ne - Ncurr) > tol and uBound-lBound > tol/10 and counter < maxcycles:
        g.setF(g.F, fermi, fermi)
        if N2 is None:
            pLow = densityReal(g.F, g.S, g, Eminf, Emin, tol, T=0, showText=False)
        else:
            pLow = densityRealN(g.F, g.S, g, Eminf, Emin, N2, T=0, showText=False)
        p_ = np.real(pLow+pMu(fermi))
        if nOrbs==0:
            Ncurr = np.trace(p_@g.S)
        else:
            Ncurr = np.trace((p_@g.S)[-nOrbs:, -nOrbs:])
        dN = ne-Ncurr
        if dN > 0 and fermi > lBound:
            lBound = fermi
        elif dN < 0 and fermi < uBound:
            uBound = fermi
        fermi = (uBound + lBound)/2
        print("DN:",dN, "Fermi:", fermi, "Bounds:", lBound, uBound)
        counter += 1
    if abs(ne - Ncurr) > tol and counter > maxcycles:
        print(f'Warning: Fermi energy still not within tolerance! Ef = {fermi:.2f} eV, N = {Ncurr:.2f})')
    print(f'Finished after {counter} iterations, Ef = {fermi:.2f}')
    return fermi, Emin, N1, N2

def calcFermiBisect(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, 
                    maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE, debug=True):
    """
    Calculate Fermi energy of system using bisection
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:   
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)
    E = Ef + 0.0
    uBound = None
    lBound = None
    P = None
    Ncurr = ne+0
    dE = tol
    counter = 0
    while None in [uBound, lBound]:
        g.setF(g.F, E, E)
        P = pMu(E)
        Ncurr = np.trace(P@g.S).real
        if counter==maxcycles:
            dE = 1e3
        if Ncurr> ne:
            uBound = E + 0.0
            Ef = uBound
            E -= dE
        if Ncurr< ne:
            lBound = E + 0.0
            Ef = lBound
            E += dE
        if debug==True:
            print(f"DEBUG: Ef={Ef:.2f}, dN={ne-Ncurr:.2E}, dE={dE:.2E}")
        mat = E * g.S - g.F - jnp.array(g.sigmaTot(E))
        gr = jnp.linalg.inv(mat)
        dos = -jnp.imag(jnp.trace(gr)) / jnp.pi
        dE = max(2*abs(Ncurr-ne)/dos, dE)
        counter += 1
    counter = 0
    while abs(ne - Ncurr) > conv and counter < maxcycles:
        g.setF(g.F, Ef, Ef)
        P = pMu(Ef)
        Ncurr = np.trace(P@g.S)
        dN = ne-Ncurr
        if dN > 0 and Ef > lBound:
            lBound = Ef + 0.0
        elif dN < 0 and Ef < uBound:
            uBound = Ef + 0.0
        Ef = (uBound + lBound)/2
        dE = uBound - lBound
        if debug==True:
            print(f"DEBUG: Ef={Ef:.2f}, dN={dN:.2E}, dE={dE:.2E}")
        counter += 1
    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(Ncurr-ne):.2E}')
    return Ef, dE, P

def calcFermiSecant(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, 
                    conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, 
                    T=TEMPERATURE, debug=True):
    """
    Calculate Fermi energy using Secant method, updating dE at each step
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:   
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)
    g.setF(g.F, Ef, Ef)
    P = pMu(Ef)
    nCurr = np.trace(P@g.S).real
    dE = conv
    counter = 0
    while abs(nCurr-ne) > conv and counter < maxcycles:
        Ef += dE
        g.setF(g.F, Ef, Ef)
        P = pMu(Ef)
        nNext = np.trace(P@g.S).real
        if debug==True:
            print(f"DEBUG: Ef={Ef:.2f}, dN={nNext-ne:.2E}, dE={dE:.2E}")
        if abs(nNext - nCurr)<1e-10:
            print('Warning: change in ne low, reducing step size')
            dE *= 0.1
            counter += 1
            continue
        dE = dE*((ne - nCurr)/(nNext-nCurr)) - dE
        nCurr = nNext + 0.0
        counter += 1
    
    Ef += dE  
    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(nCurr-ne):.2E}')
    return Ef, dE, P, abs(nCurr-ne)

def calcFermiMuller(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, 
                    conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, 
                    T=TEMPERATURE, debug=True):
    """
    Calculate Fermi energy using Muller's method, starting with 3 initial points
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    small = 1e-10  # Small value to prevent division by zero
    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:   
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)

    # Initialize three points around initial guess
    E2 = Ef
    E1 = E2 - conv
    E0 = E2 + conv

    # Get initial density matrices and electron counts
    g.setF(g.F, E2, E2)
    P = pMu(E2)
    n2 = np.trace(P@g.S).real - ne
    if abs(n2) < conv:
        return E2, 0, P, abs(n2)
    g.setF(g.F, E1, E1)
    P = pMu(E1)
    n1 = np.trace(P@g.S).real - ne
    if abs(n1) < conv:
        return E1, conv, P, abs(n1)
    g.setF(g.F, E0, E0)
    P = pMu(E0)
    n0 = np.trace(P@g.S).real - ne
    if abs(n0) < conv:
        return E0, delta_E, P, abs(n0)

    counter = 3
    while counter < maxcycles:
        # Calculate differences between points
        h0 = E0 - E2
        h1 = E1 - E2

        # Set up quadratic coefficients
        c = n2
        e0 = n0 - c
        e1 = n1 - c
        
        # Calculate coefficients for the quadratic approximation
        det = h0 * h1 * (h0 - h1)
        a = (e0 * h1 - h0 * e1) / det
        b = (h0 * h0 * e1 - h1 * h1 * e0) / det

        # Calculate discriminant for quadratic formula
        disc = np.sqrt(b * b - 4 * a * c) if b * b > 4 * a * c else 0
        if b < 0:
            disc = -disc

        # Calculate next approximation
        dE = -2 * c / (b + disc)
        Enext = E2 + dE

        # Update points maintaining proper ordering
        if abs(Enext - E1) < abs(Enext - E0):
            E0, E1 = E1, E0
            n0, n1 = n1, n0

        if abs(Enext - E2) < abs(Enext - E1):
            E1 = E2
            n1 = n2

        E2 = Enext
        g.setF(g.F, E2, E2)
        P = pMu(E2)
        n2 = np.trace(P@g.S).real - ne

        # Check convergence
        if abs(n2) < conv:
            break

        if debug==True:
            print(f"DEBUG: Ef={E2:.2f}, dN={n2:.2E}, dE={dE:.2E}")
        counter += 1

    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(n2):.2E}')

    return E2, dE, P, abs(n2)

def calcFermiInversePolynomial(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL,
                    conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, 
                    T=TEMPERATURE, debug=True):
    """
    Calculate Fermi energy using inverse polynomial interpolation with accumulating points.

    Uses all accumulated points to fit a polynomial through (n, E) pairs
    and find where n = ne (target electron count). This is a generalization of Muller's method
    to arbitrary polynomial order.

    Parameters
    ----------
    g : surfG object
        Surface Green's function calculator
    ne : float
        Target number of electrons
    Emin : float
        Lower bound for complex contour in eV
    Ef : float
        Initial guess for Fermi energy in eV
    N : int or None
        Number of integration points (None for adaptive)
    tol : float, optional
        Tolerance for integration (default: 1e-5)
    conv : float, optional
        Convergence criterion for electron count (default: 1e-3)
    maxcycles : int, optional
        Maximum number of iterations (default: 20)
    T : float, optional
        Temperature in Kelvin (default: 300)

    Returns
    -------
    tuple
        (Ef, dE, P, error) - Calculated Fermi energy, last energy step, density matrix, and final error

    Notes
    -----
    This method builds up a history of (n, E) points and uses Lagrange interpolation
    to fit an inverse polynomial (E as a function of n) to find E at n = ne.
    More robust than Muller for well-behaved functions as it uses more information.
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"

    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)

    # Lists to store history of points
    E_pts = []
    n_pts = []

    # Initialize with first point
    E = Ef
    g.setF(g.F, E, E)
    P = pMu(E)
    n = np.trace(P@g.S).real - ne

    if abs(n) < conv:
        return E, 0, P, abs(n)

    E_pts.append(E)
    n_pts.append(n)

    # Enforce monotonicity: Higher E -> Higher N-ne
    step = conv*10
    n_first = n
    counter = 1
    while counter < maxcycles:
        E = Ef + step
        g.setF(g.F, E, E)
        P = pMu(E)
        n = np.trace(P@g.S).real - ne

        if abs(n) < conv:
            return E, step, P, abs(n)

        # Check if we have a meaningful difference from first point
        if n - n_first > 0:
            break

        # Otherwise increase step size and try again
        step *= 10
        counter += 1
        if debug==True:
            print(f'Warning: Tried Ef = {E:2f} eV (too close to {Ef:2f} to get accurate dN {n-n_first:.2E})')

    E_pts.append(E)
    n_pts.append(n)

    dE = step

    while counter < maxcycles:
        # Use Lagrange interpolation to find E where n = 0 (since we store n - ne)
        # We're doing inverse interpolation: E = f(n) instead of n = f(E)

        # Lagrange interpolation: E(n=0) = sum_i E_i * L_i(0)
        # where L_i(n) = prod_{j!=i} (n - n_j) / (n_i - n_j)

        E_next = 0.0
        for i in range(len(n_pts)):
            # Calculate Lagrange basis polynomial L_i(0)
            L_i = 1.0
            for j in range(len(n_pts)):
                if i != j:
                    # Check for duplicate points (would cause division by zero)
                    if abs(n_pts[i] - n_pts[j]) < 1e-14:
                        # Skip this interpolation, use last good estimate
                        E_next = E_pts[-1] + dE * 0.5
                        break
                    L_i *= (0 - n_pts[j]) / (n_pts[i] - n_pts[j])
            else:
                E_next += E_pts[i] * L_i
                continue
            break
    
        # Enforce monotonicity: Higher E -> higher N 
        # If N-nE > 0: need lower E, so E_next must be < E_pts[-1]
        # If N-nE < 0: need higher E, so E_next must be > E_pts[-1]
        if n_pts[-1] > 0 and E_next > E_pts[-1]:
            # Polynomial violated monotonicity - discard it and step in correct direction
            E_next = E_pts[-1] - abs(dE) * 10
            # Remove the last point that led to bad interpolation
            E_pts.pop()
            n_pts.pop()
            counter -= 1  # Don't count this as a valid iteration
            if debug==True:
                print('Warning: monotonicity exception corrected!')
        elif n_pts[-1] < 0 and E_next < E_pts[-1]:
            # Polynomial violated monotonicity - discard it and step in correct direction
            E_next = E_pts[-1] + abs(dE) * 10
            # Remove the last point that led to bad interpolation
            E_pts.pop()
            n_pts.pop()
            counter -= 1  # Don't count this as a valid iteration
            if debug==True:
                print('Warning: monotonicity exception corrected!')

        # Calculate new point
        E = E_next
        g.setF(g.F, E, E)
        P = pMu(E)
        n = np.trace(P@g.S).real - ne

        # Update history
        E_pts.append(E)
        n_pts.append(n)

        # Calculate step size for reporting
        dE = E - E_pts[-2]

        # Check convergence
        if abs(n) < conv:
            break

        if debug==True:
            print(f"Iter {counter}: E = {E:.6f}, n-ne = {n:.3e}, dE = {dE:.3e}, order = {counter+1}")
        counter += 1

    if counter >= maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(n):.2E}')

    return E, dE, P, abs(n)
