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
try:
    import cupy as cp
    isCuda = cp.cuda.is_available()
except:
    isCuda = False

# Parallelization packages
from multiprocessing import Pool
import os

# Developed Packages:
from gauNEGF.fermiSearch import DOSFermiSearch
from gauNEGF.surfG1D import surfG

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
    ndarrayh
        Retarded Green's function G(E) = [ES - F - Σ(E)]^(-1)
    """
    return LA.inv(E*S - F - g.sigmaTot(E))

def GrIntVectorized(F, S, g, Elist, weights):
    """
    Integrate retarded Green's function for a list of energies using vectorization.

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

    Returns
    -------
    ndarray
        Retarded Green's function G(E) integrated over the energy grid (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    M = Elist.size  # Number of points in the grid
    N = F.shape[0]  # Assuming F is square (NxN)

    # Use CuPy if cuda available, otherwise numpy
    solver = cp if isCuda else np
    Elist_ = solver.array(Elist)
    weights = solver.array(weights)
    S = solver.array(S)
    F = solver.array(F)

    #Generate vectorized matrices
    S_repeated = solver.tile(S, (M, 1, 1))  # Shape (MxNxN)
    F_repeated = solver.tile(F, (M, 1, 1))  # Shape (MxNxN)
    I = solver.eye(N)  # Shape (NxN)
    I_repeated = solver.tile(I, (M, 1, 1))  # Shape (MxNxN)
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])  # Shape (MxNxN)

    #Solve for retarded Green's function - NOTE: This is the bottleneck for larger systems!
    Gr_vec = solver.linalg.solve(Elist_[:, None, None] * S_repeated - F_repeated - SigmaTot, I_repeated)

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gr_vec*weights[:, None, None], axis=0)
    return Gint.get() if isCuda else Gint

def GrLessVectorized(F, S, g, Elist, weights, ind=None):
    """
    Integrate nonequilibrium Green's function for a list of energies using vectorization.

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
        Nonequilibrium Green's function G<(E) integrated over the energy grid (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"

    M = Elist.size  # Number of points in the grid
    N = F.shape[0]  # Assuming F is square (NxN)

    # Use CuPy if cuda available, otherwise numpy
    solver = cp if isCuda else np
    Elist_ = solver.array(Elist)
    weights = solver.array(weights)
    S = solver.array(S)
    F = solver.array(F)

    #Generate vectorized matrices
    S_repeated = solver.tile(S, (M, 1, 1))  # Shape (MxNxN)
    F_repeated = solver.tile(F, (M, 1, 1))  # Shape (MxNxN)
    I = solver.eye(N)  # Shape (NxN)
    I_repeated = solver.tile(I, (M, 1, 1))  # Shape (MxNxN)
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])  # Shape (MxNxN)

    #Solve for retarded Green's function - NOTE: This is the bottleneck for larger systems!
    Gr_vec = solver.linalg.solve(Elist_[:, None, None] * S_repeated - F_repeated - SigmaTot, I_repeated)
    Ga_vec = solver.conj(Gr_vec).transpose(0, 2, 1) # Shape (MxNxN)
    if ind is not None:
        SigList = SigmaTot
    else:
        SigList = [g.sigma(E, i) for i in range(N)]
    GammaList = [1j*(sig - sig.conj().T) for sig in SigList]

    Gless_vec = solver.matmul(solver.matmul(Gr_vec, GammaList), Ga_vec)

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gless_vec*weights[:, None, None], axis=0)
    return Gint.get() if isCuda else Gint

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
def integratePointsAdaptiveANT(computePoint, tol=1e-3, maxN=1458, debug=False):
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
def densityRealN(F, S, g, Emin, mu, N=100, T=300, showText=True):
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
    nKT= 10
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

    defInt = GrIntVectorized(F, S, g, Elist, weights)

    if showText:
        print('Integration done!')
    
    return (-1+0j)*np.imag(defInt)/(np.pi)

def densityReal(F, S, g, Emin, mu, tol=1e-3, T=0, maxN=1000, debug=False):
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
   

def densityGridN(F, S, g, mu1, mu2, ind=None, N=100, T=300, showText=True):
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
    nKT= 10
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
    
    den = GrLessVectorized(F, S, g, energies, weights, ind)

    if showText:
        print('Integration done!')
 
    return den/(2*np.pi)

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGridTrap(F, S, g, mu1, mu2, ind=None, N=100, T=300):
    nKT= 10
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
        GrE = Gr(F, S, g, E)
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

def densityGrid(F, S, g, mu1, mu2, ind=None, tol=1e-3, T=300, debug=False):
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
    nKT= 10
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
        return GrLessVectorized(F, S, g, E, weights, ind)
     
    den = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)
    if debug:
        print('Integration done!')

    return den/(2*np.pi)

def densityComplexN(F, S, g, Emin, mu, N=100, T=300, showText=True, method='ant'):
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

    lineInt = GrIntVectorized(F, S, g, Elist, weights)
    
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
        lineInt += GrIntVectorized(F, S, g, Elist, weights)

    if showText:
        print('Integration done!')

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi

def densityComplex(F, S, g, Emin, mu, tol=1e-3, T=300, debug=False):
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
        return GrIntVectorized(F, S, g, z, weights)
    
    print('Complex Contour Integration:')
    lineInt = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)
    
    #Add integration points for Fermi Broadening
    if T>0:
        print('Integrating Fermi Broadening:')
        def computePointBroadening(x, w):
            E = broadening * (x) + mu
            weights = broadening*w*fermi(E, mu, T)
            return GrIntVectorized(F, S, g, E, weights)
    
        lineInt += integratePointsAdaptiveANT(computePointBroadening, tol=tol, debug=debug)

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi


## INTEGRATION LIMIT FUNCTIONS
# Calculate Emin using DOS
def calcEmin(F, S, g, tol=1e-5, maxN=1000):
    D,_ = LA.eig(LA.inv(S)@F)
    Emin = min(D.real.flatten())-5
    counter = 0
    dP = DOSg(F,S,g,Emin)
    while dP>tol and counter<maxN:
        Emin -= 1
        dP = abs(DOSg(F,S,g,Emin))
        #print(Emin, dP)
        counter += 1
    if counter == maxN:
        print(f'Warning: Emin still not within tolerance (final value = {dP}) after {maxN} energy samples')
    print(f'Calculated Emin: {Emin} eV, DOS = {dP:.2E}')
    return Emin

def integralFit(F, S, g, mu, Eminf=-1e6, tol=1e-5, T=0, maxN=1000):
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

def integralFitNEGF(F, S, g, fermi, qV, Eminf=-1e6, tol=1e-5, T=0, maxGrid=1000):
    """
    Determines number of  for non-equilibrium density calculations.

    Same procedure as `integralFit()` but applied to `densityGrid()`

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    mu1 : float
        Left contact Fermi energy in eV
    mu2 : float
        Right contact Fermi energy in eV
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


def getFermiContact(g, ne, tol=1e-4, Eminf=-1e6, maxcycles=1000, T=0, nOrbs=0):
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
    Emin, N1, N2 = integralFit(F, S, g, fermi, Eminf, tol, T, maxN=maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, 
                        Eminf, T, tol, maxcycles, nOrbs)[0]

def getFermi1DContact(gSys, ne, ind=0, tol=1e-4, Eminf=-1e6, T=0, maxcycles=1000):
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
    Emin, N1, N2 = integralFit(Forbs, Sorbs, gorbs, fermi, Eminf, tol, T, maxN=maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, Eminf, T, tol, maxcycles)

# Calculate the fermi energy of the surface Green's Function object
def calcFermi(g, ne, Emin, Emax, fermiGuess=0, N1=100, N2=50, Eminf=-1e6, T=0, tol=1e-4, maxcycles=20, nOrbs=0):
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

def calcFermiBisect(g, ne, Emin, Ef, N, tol=1e-4, conv=1e-3, maxcycles=10, T=0):
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
        #print(uBound, lBound, E, dE)
        dE = max(2*abs(Ncurr-ne)/DOSg(g.F, g.S, g, E), dE)
        counter += 1
    counter = 0
    while abs(ne - Ncurr) > conv and counter < maxcycles:
        g.setF(g.F, Ef, Ef)
        P = pMu(Ef)
        Ncurr = np.trace(pMu(Ef)@g.S)
        dN = ne-Ncurr
        if dN > 0 and Ef > lBound:
            lBound = Ef + 0.0
        elif dN < 0 and Ef < uBound:
            uBound = Ef + 0.0
        Ef = (uBound + lBound)/2
        dE = uBound - lBound
        #print(uBound, lBound, dE, E)
        counter += 1
    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(Ncurr-ne):.2E}')
    return Ef, dE, P

def calcFermiSecant(g, ne, Emin, Ef, N, tol=1e-4, conv=1e-3, maxcycles=10, T=0):
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
        #print(Ef, dE, nCurr, nNext)
        if abs(nNext - nCurr)<1e-10:
            print('Warning: change in ne low, reducing step size')
            dE *= 0.1
            counter += 1
            continue
        dE = dE*((ne - nCurr)/(nNext-nCurr)) - dE
        nCurr = nNext + 0.0
        counter += 1
        #print(Ef, dE)
    
    Ef += dE  
    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(nCurr-ne):.2E}')
    return Ef, dE, P

def calcFermiMuller(g, ne, Emin, Ef, N, tol=1e-4, conv=1e-3, maxcycles=10, T=0):
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
    E0 = E1 - conv

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

        # Check convergence
        if abs(n2) < conv:
            break

        #print("E0 - ", E0, n0, "E1 - ", E1, n1, "E2 - ", E2, n2, " dE ", dE)
        counter += 1

    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(n2):.2E}')

    return E2, dE, P

