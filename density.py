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
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy.special import roots_legendre
from scipy.special import roots_chebyu
import matplotlib.pyplot as plt
import warnings
#import cupy as cp

# Parallelization packages
from multiprocessing import Pool
import os

# Developed Packages:
from fermiSearch import DOSFermiSearch
from surfG1D import surfG

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
kB = 8.617e-5           # eV/Kelvin


## HELPER FUNCTIONS
def Gr(F, S, g, E):
    """
    Calculate retarded Green's function at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    ndarray
        Retarded Green's function G(E) = [ES - F - Σ(E)]^(-1)
    """
    return LA.inv(E*S - F - g.sigmaTot(E))

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

def DOSg(F, S, g, E):
    """
    Calculate density of states at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    float
        Density of states at energy E
    """
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

def getANTPoints(N):
    """
    Generate integration points and weights matching ANT.Gaussian implementation.

    Follows the IntCompPlane subroutine in device.F90 from ANT.Gaussian package.
    Uses a modified Gauss-Chebyshev quadrature scheme optimized for transport
    calculations.

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
    
    # Use process-level parallelization for large workloads or when requested
    useProcessParallel = parallel or (
        numPoints >= 100 and numCores >= 32
    )
    
    # Standard case: Use numpy's built-in parallelization
    if not useProcessParallel:
        if debug:
            print('Using numpy built-in parallelization for matrix operations')
        result = np.zeros_like(computePointFunc(0))
        for i in range(numPoints):
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

def bisectFermi(V, Vc, D, Gam, Nexp, conv=1e-3, Eminf=-1e6):
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
def densityReal(F, S, g, Emin, mu, N=100, T=300, parallel=False,
                numWorkers=None, showText=True):
    """
    Calculate equilibrium density matrix using real-axis integration.

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
    parallel : bool, optional
        Whether to use parallel processing (default: False)
    numWorkers : int, optional
        Number of worker processes for parallel mode
    showText : bool, optional
        Whether to print progress messages (default: True)

    Returns
    -------
    ndarray
        Density matrix
    """
    kT = kB*T
    Emax = mu + 5*kT
    mid = (Emax-Emin)/2
    defInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    
    def computePoint(i):
        E = mid*(x[i] + 1) + Emin
        return mid*w[i]*Gr(F, S, g, E)*fermi(E, mu, T)
    
    if showText:
        print(f'Real integration over {N} points...')

    defInt = integratePoints(computePoint, int(N), parallel, numWorkers)

    if showText:
        print('Integration done!')
    
    return (-1+0j)*np.imag(defInt)/(np.pi)

def densityGrid(F, S, g, mu1, mu2, ind=None, N=100, T=300, parallel=False,
                numWorkers=None, showText=True):
    """
    Calculate non-equilibrium density matrix using real-axis integration.

    Performs numerical integration for the non-equilibrium part of the density
    matrix when a bias voltage is applied. Uses Gauss-Legendre quadrature.

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
    parallel : bool, optional
        Whether to use parallel processing (default: False)
    numWorkers : int, optional
        Number of worker processes for parallel mode
    showText : bool, optional
        Whether to print progress messages (default: True)

    Returns
    -------
    ndarray
        Non-equilibrium contribution to density matrix
    """
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    dInt = np.sign(mu2 - mu1)
    Emax = muHi + 5*kT
    Emin = muLo - 5*kT
    mid = (Emax-Emin)/2
    den = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    
    def computePoint(i):
        E = mid*(x[i] + 1) + Emin
        GrE = Gr(F, S, g, E)
        GaE = GrE.conj().T
        if ind == None:
            sig = g.sigmaTot(E)
        else:
            sig = g.sigma(E, ind)
        Gamma = 1j*(sig - sig.conj().T)
        dFermi = fermi(E, muHi, T) - fermi(E, muLo, T)
        return mid*w[i]*(GrE@Gamma@GaE)*dFermi*dInt
     
    if showText:
        print(f'Real integration over {N} points...')
    
    den = integratePoints(computePoint, int(N), parallel, numWorkers)

    if showText:
        print('Integration done!')
 
    return den/(2*np.pi)

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGridTrap(F, S, g, mu1, mu2, ind=None, N=100, T=300):
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    dInt = np.sign(mu2 - mu1)
    Emax = muHi + 5*kT
    Emin = muLo - 5*kT
    Egrid = np.linspace(Emin, Emax, N)
    den = np.array(np.zeros(np.shape(F)), dtype=complex)
    print(f'Real integration over {N} points...')
    for i in range(1,N):
        E = (Egrid[i] + Egrid[i-1])/2
        dE = Egrid[i] - Egrid[i-1]
        GrE = Gr(F, S, g, E)
        GaE = GrE.conj().T
        if ind == None:
            sig = g.sigmaTot(E)
        else:
            sig = g.sigma(E, ind)
        Gamma = 1j*(sig - sig.conj().T)
        dFermi = fermi(E, muHi, T) - fermi(E, muLo, T)
        den += mid*w[i]*(GrE@Gamma@GaE)*dFermi*dInt
    print('Integration done!')
    
    return den/(2*np.pi)

def densityComplex(F, S, g, Emin, mu, N=100, T=300, parallel=False, numWorkers=None, 
                         showText=True, method='ant'):
    """
    Calculate equilibrium density matrix using complex contour integration.

    Performs numerical integration along a complex contour that encloses the
    poles of the Fermi function. More efficient than real-axis integration
    for equilibrium calculations.

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
    parallel : bool, optional
        Whether to use parallel processing (default: False)
    numWorkers : int, optional
        Number of worker processes for parallel mode
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
    nKT= 5
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
    else:
        x = np.linspace(-1, 1, N)
        w = 2*np.ones(N)/N
    
    #Integrate along contour
    def computePoint(i):
        theta = np.pi/2 * (x[i] + 1)
        z = center + r*np.exp(1j*theta)
        dz = 1j * r * np.exp(1j*theta)
        return (np.pi/2)*w[i]*Gr(F, S, g, z)*fermi(z, mu, T)*dz
    
    if showText:
        print(f'Complex Integration over {N} points...')
    lineInt=integratePoints(computePoint, int(N), parallel, numWorkers)
    
    #Add integration points for Fermi Broadening
    if T>0:
        if showText:
            print('Integrating Fermi Broadening')
        x_fermi,w_fermi = roots_legendre(N//8)
        def computePointBroadening(i):
            E = broadening * (x_fermi[i]) + mu
            return broadening*w[i]*Gr(F, S, g, E)*fermi(E, mu, T)
    
        lineInt += integratePoints(computePoint, int(N//8), parallel, numWorkers)

    if showText:
        print('Integration done!')

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi


# Get density using a complex contour and trapezoidal integration
def densityComplexTrap(F, S, g, Emin, mu, N, T=300):
    #Construct circular contour
    nKT= 5
    broadening = nKT*kB*T
    Emax = mu-broadening
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2
    theta = np.linspace(0, np.pi, N)
    Egrid = r*np.exp(1j*theta)+center

    #Integrate along contour
    print(f'Complex Integration over {N} points...')
    lineInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    for i in range(1,N):
        E = (Egrid[i]+Egrid[i-1])/2
        dS = Egrid[i]-Egrid[i-1]
        lineInt += Gr(F, S, g, E)*fermi(E, mu, T)*dS
    
    #Add integration points for Fermi Broadening
    if T>0:
        print('Integrating Fermi Broadening')
        Egrid2 = np.linspace(Emax, Emax+2*broadening, N//8)
        for i in range(1, N//8):
            E = (Egrid2[i]+Egrid2[i-1])/2
            dS = Egrid2[i]-Egrid2[i-1]
            lineInt += Gr(F, S, g, E)*fermi(E, mu, T)*dS
    print('Integration done!')

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi


## INTEGRATION LIMIT FUNCTIONS
def integralFit(F, S, g, mu, Eminf, tol=1e-6, maxcycles=1000):
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
        Initial guess for chemical potential in eV
    Eminf : float
        Lower bound for integration in eV
    tol : float, optional
        Convergence tolerance (default: 1e-6)
    maxcycles : int, optional
        Maximum number of iterations (default: 1000)

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
    D,_ = LA.eig(LA.inv(S)@F)
    Emin = min(D.real.flatten())-5
    counter = 0
    dP = DOSg(F,S,g,Emin)
    while dP>tol and counter<maxcycles:
        Emin -= 1
        dP = abs(DOSg(F,S,g,Emin))
        print(Emin, dP)
        counter += 1
    if counter == maxcycles:
        print(f'Warning: Emin still not within tolerance (final value = {dP}) after {maxcycles} energy samples')
    print(f'Final Emin: {Emin} eV, DOS = {dP:.2E}') 
    
    #Determine grid using dP
    counter = 0
    Ncomplex = 4
    dP = 100
    rho = np.zeros(np.shape(F))
    while dP > tol and Ncomplex < 1000:
        Ncomplex *= 2 # Start with 8 points, double each time
        rho_ = np.real(densityComplex(F, S, g, Emin,  mu, Ncomplex, T=0))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}, N = {sum(np.diag(rho_).real):2f}")
        rho = rho_
        counter += 1
    if dP < tol:
        Ncomplex /= 2
    elif Ncomplex >  maxcycles and dP > tol:
        print(f'Warning: Ncomplex still not within tolerance (final value = {dP})')
    print(f'Final Ncomplex: {Ncomplex}') 

    #Determine grid using dP
    counter = 0
    Nreal = 8
    dP = 100
    rho = np.zeros(np.shape(F))
    while dP > tol and Nreal < 1000:
        Nreal *= 2 # Start with 16 points, double each time
        rho_ = np.real(densityReal(F, S, g, Eminf, Emin, Nreal, T=0))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}")
        rho = rho_
        counter += 1
    if dP < tol:
        Nreal /= 2
    elif Nreal >  maxcycles and dP > tol:
        print(f'Warning: Nreal still not within tolerance (final value = {dP})')
    print(f'Final Nreal: {Nreal}') 

    return Emin, Ncomplex, Nreal

def getFermiContact(g, ne, tol=1e-4, Eminf=-1e6, maxcycles=1000, nOrbs=0):
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
    orbs, _ = LA.eig(LA.inv(S)@F)
    orbs = np.sort(np.real(orbs))
    fermi = (orbs[int(ne)-1] + orbs[int(ne)])/2
    Emin, N1, N2 = integralFit(g.F, g.S, g, fermi, Eminf, tol, maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, 
                        Eminf, tol, maxcycles, nOrbs)[0]

def getFermi1DContact(gSys, ne, ind=0, tol=1e-4, Eminf=-1e6, maxcycles=1000):
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
    float
        Fermi energy in eV
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
    orbs, _ = LA.eig(LA.inv(Sorbs)@Forbs)
    orbs = np.sort(np.real(orbs))
    fermi = (orbs[2*int(ne)-1] + orbs[2*int(ne)])/2
    Emin, N1, N2 = integralFit(Forbs, Sorbs, gorbs, fermi, Eminf, tol, maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, Eminf, tol, maxcycles)

# Calculate the fermi energy of the surfG() object
def calcFermi(g, ne, Emin, Emax, fermiGuess=0, N1=100, N2=50, Eminf=-1e6, tol=1e-4, maxcycles=20, nOrbs=0):
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
    print(f'Eminf DOS = {DOSg(g.F,g.S,g,Eminf)}')
    fermi = fermiGuess
    pLow = densityReal(g.F, g.S, g, Eminf, Emin, N2, T=0, showText=False)
    if nOrbs==0:
        nELow = np.trace(pLow@g.S)
    else:
        nELow = np.trace((pLow@g.S)[-nOrbs:, -nOrbs:])
    print(f'Electrons below lowest onsite energy: {nELow}')
    if nELow >= ne:
        raise Exception('Calculated Fermi energy is below lowest orbital energy!')
    pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, N1, T=0, showText=False, method='legendre')
    
    # Fermi search using bisection method (F not changing, highly stable)
    Ncurr = -1
    counter = 0 
    lBound = Emin
    uBound = Emax
    print('Calculating Fermi energy using bisection:')
    while abs(ne - Ncurr) > tol and uBound-lBound > tol/10 and counter < maxcycles:
        g.setF(g.F, fermi, fermi)
        pLow = densityReal(g.F, g.S, g, Eminf, Emin, N2, T=0, showText=False)
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

def calcFermiSecant(g, ne, Emin, Ef, N, tol=1e-3, maxcycles=10, T=0):
    """
    Calculate Fermi energy using Secant method, updating dE at each step
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, N, T)
    g.setF(g.F, Ef, Ef)
    P = pMu(Ef)
    nCurr = np.trace(P@g.S).real
    dE = 1.0
    counter = 0
    while abs(nCurr-ne) > tol and counter < maxcycles:
        Ef += dE
        g.setF(g.F, Ef, Ef)
        P = pMu(Ef)
        nNext = np.trace(P@g.S).real
        #print(Ef, dE, nCurr, nNext)
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
    return Ef, dE, P

def calcFermiMuller(g, ne, Emin, Ef, N, tol=1e-3, maxcycles=10, T=0):
    """
    Calculate Fermi energy using Muller's method, starting with 3 initial points
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    small = 1e-10  # Small value to prevent division by zero
    pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, N, T)

    # Initialize three points around initial guess
    E2 = Ef
    E1 = E2 - 0.5
    E0 = E1 - 0.5

    # Get initial density matrices and electron counts
    g.setF(g.F, E2, E2)
    n2 = np.trace(pMu(E2)@g.S).real - ne
    g.setF(g.F, E1, E1)
    n1 = np.trace(pMu(E1)@g.S).real - ne
    g.setF(g.F, E0, E0)
    n0 = np.trace(pMu(E0)@g.S).real - ne

    counter = 0
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
            E2, E1 = E1, E2
            n2, n1 = n1, n2

        E2 = Enext
        g.setF(g.F, E2, E2)
        P = pMu(E2)
        n2 = np.trace(P@g.S).real - ne

        # Check both relative error and absolute convergence
        if abs(n2) < tol:
            break

        #print("E0 - ", E0, n0, "E1 - ", E1, n1, "E2 - ", E2, n2, " dE ", dE)
        counter += 1

    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(n2):.2E}')

    return E2, dE, P

