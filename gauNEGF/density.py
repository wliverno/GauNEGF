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
from gauNEGF.config import (TEMPERATURE, ADAPTIVE_INTEGRATION_TOL, FERMI_CALCULATION_TOL, FERMI_DEBUG,
                            FERMI_SEARCH_CYCLES, N_KT, ENERGY_MIN, MAX_CYCLES, MAX_GRID_POINTS, ETA,
                            EMIN_BUFFER, USE_INERTIA_EMIN)
from scipy.special import roots_legendre
from scipy.special import roots_chebyu
import matplotlib.pyplot as plt
import warnings

# Parallelization packages
from multiprocessing import Pool
import os

# Developed Packages:
from gauNEGF.integrate import GrInt, GrLessInt, GrIntCross, GrLessIntCross

# JIT-compiled functions
from gauNEGF.utils import inv, eig, eigh, fractional_matrix_power

@jit
def _compute_dos_at_energy(E, F, S, sigma_total, Qtot, eta=ETA):
    """JIT-compiled kernel for DOS calculation at single energy."""
    mat = (E + 1j*eta)*S - F - sigma_total
    Gr = inv(mat)
    Q = jnp.zeros_like(S) if Qtot is None else Qtot
    mat_sum = Gr@S - Gr@Q
    return -jnp.imag(jnp.trace(mat_sum)) / jnp.pi


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
    accum = None
    is_tuple = None
    N = 2
    maxDP = 1e10
    while N<=maxN:
        x, w = getANTPoints(N)

        if prev_x is None:
            # first level: no reuse
            result = computePoint(x[0:2], w[0:2])
            is_tuple = isinstance(result, tuple)
            if is_tuple:
                P, accum = result
            else:
                P = result
        else:
            # mark old nodes robustly by value
            old_mask = np.isin(np.round(x, 14), np.round(prev_x, 14))
            # sanity: all previous nodes should be found
            assert int(old_mask.sum()) == prev_x.size, "Old nodes mismatch"

            # exact transfer factor (should be ~1/3)
            ratio = float(np.sum(w[old_mask]) / prev_sumW)

            # scale previous integral + add only new-node contributions
            new_mask = ~old_mask
            new_result = computePoint(x[new_mask], w[new_mask])
            if is_tuple:
                new_mat, new_scl = new_result
                new_P = P * ratio + new_mat
                new_accum = accum * ratio + new_scl
            else:
                new_P = P*ratio + new_result

            maxDP = jnp.max(jnp.abs(new_P-P))
            if is_tuple:
                maxDP = jnp.maximum(maxDP, jnp.abs(new_accum - accum))
            if debug:
                full_result = computePoint(x, w)
                P_debug = full_result[0] if is_tuple else full_result
                maxDP_debug = jnp.max(jnp.abs(P_debug-P))
                maxDiff = jnp.max(jnp.abs(P_debug-new_P))
                print(f"N={N}, nested-weight ratio ~ {ratio:.3f}, maxDP={maxDP:.3e}")
                print(f"Direct Calculation: N={N}, maxDP={maxDP_debug:.3e}, maxDiff={maxDiff:.3e}")
            P = new_P.copy()
            if is_tuple:
                accum = new_accum
            if maxDP<tol:
                print(f'Adaptive integration converged to {maxDP:.3e} in {N} points.')
                if is_tuple:
                    return (new_P, new_accum)
                return new_P

        # update state for next level
        prev_x = x
        prev_sumW = float(jnp.sum(w))
        N *= 3
    N/=3
    print(f'Adaptive integration reached full grid ({N} points), final error {maxDP:.3e}')
    if is_tuple:
        return (new_P, new_accum)
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
    DD = jnp.array([D for i in range(Nd)]).T
    
    #Integral of 1/x is log(x), calculating at lower and upper limit
    logmat = jnp.array([np.emath.log(1-(mu/D)) for i in range(Nd)]).T
    logmat2 = jnp.array([np.emath.log(1-(Emin/D)) for i in range(Nd)]).T

    #Compute integral, add prefactor
    invmat = 1/(2*np.pi*(DD-DD.conj().T))
    pref2 = logmat - logmat.conj().T
    pref3 = logmat2-logmat2.conj().T

    prefactor = jnp.multiply(invmat,(pref2-pref3))

    #Convert Gamma into Fbar eigenbasis, element-wise multiplication
    Gammam = Vc.conj().T@Gam@Vc
    prefactor = jnp.multiply(prefactor,Gammam)
    
    #Convert back to input basis, return
    den = V@ prefactor @ V.conj().T
    return den

def damleCrossTerm(V, D, Vc, Y_eff, Q0, Q1, lo, hi):
    """Analytic lower-contour cross-term delta_N for the Damle method.

    Reuses the eig (V, D, Vc) of Fbar already computed for the density matrix
    (no second eigendecomposition). Q(E) ~ Q0 + Q1*E is linearized between two
    anchors. The unphysical linear-Q 'flat' term b1*(hi-lo) is OMITTED -- only
    the physical in-window-pole term is kept (see
    docs/lower_contour_math_and_probes.md sec 5):

        delta_N = -(1/pi) Im sum_i (b0_i + b1_i*D_i)
                                  * [log(1 - hi/D_i) - log(1 - lo/D_i)]

    with b(E) = Vc^dagger Y_eff Q(E) Y_eff V, b0 = diag(...Q0...),
    b1 = diag(...Q1...). Poles outside [lo, hi] give a real log-difference
    (no contribution); in-window poles pick up the i*pi (the physical count).

    Parameters
    ----------
    V, D, Vc : ndarray
        Eigendecomposition of Fbar: Fbar = V diag(D) Vc^dagger, Vc = inv(V^dagger).
    Y_eff : ndarray
        S_eff^(-1/2).
    Q0, Q1 : ndarray (N, N)
        Linear model of crossTermQTot: Q(E) ~ Q0 + Q1*E.
    lo, hi : float
        Lower-contour energy limits in eV (lo = ENERGY_MIN, hi = Emin).

    Returns
    -------
    float
        Cross-term electron-count correction delta_N.
    """
    Ml = Vc.conj().T @ Y_eff
    Mr = Y_eff @ V
    b0 = jnp.diag(Ml @ jnp.asarray(Q0) @ Mr)
    b1 = jnp.diag(Ml @ jnp.asarray(Q1) @ Mr)
    logdiff = jnp.log(1.0 - hi / D) - jnp.log(1.0 - lo / D)
    contrib = (b0 + b1 * D) * logdiff
    return float(-(1.0 / jnp.pi) * jnp.imag(jnp.sum(contrib)))

def damleLowerDensity(F_eV, Y_eff, Sigma_0, g, Emin, ENERGY_MIN_=ENERGY_MIN):
    """Analytic Damle lower-contour density on [ENERGY_MIN_, Emin].

    Builds Fbar = Y_eff (F + Sigma_0 + i*ETA) Y_eff, diagonalizes it ONCE, and
    returns both the lower-contour density matrix (in the AO basis) and the
    analytic cross-term delta_N (which reuses the same eig; no second
    decomposition).

    The density matrix is negated so its sign matches the densityComplex
    convention. The cross-term uses a linear model of crossTermQTot(E) anchored
    near the band bottom at (Emin, 2*Emin) and DROPS the unphysical linear-Q
    flat term (see damleCrossTerm).

    Parameters
    ----------
    F_eV : ndarray (N, N)
        Device Fock matrix in eV.
    Y_eff : ndarray (N, N)
        S_eff^(-1/2) (from NEGFE._initAsymptoticSigma).
    Sigma_0 : ndarray (N, N)
        Asymptotic constant contact self-energy.
    g : surfG object
        Provides crossTermQTot(E) for the cross-term. If it returns None
        (orthogonal system), delta_N = 0.
    Emin : float
        Upper bound of the lower contour in eV (set by calcEmin upstream).
    ENERGY_MIN_ : float, optional
        Lower bound in eV (default config.ENERGY_MIN).

    Returns
    -------
    tuple (ndarray, float)
        (P_lower, delta_N).
    """
    F_eV = jnp.asarray(F_eV)
    Y_eff = jnp.asarray(Y_eff)
    Sigma_0 = jnp.asarray(Sigma_0)
    N = F_eV.shape[0]
    Emin = float(Emin)
    lo = float(ENERGY_MIN_)

    H_eff = F_eV + Sigma_0 + 1j * ETA * jnp.eye(N)
    Fbar = Y_eff @ H_eff @ Y_eff
    Gam = (H_eff - H_eff.conj().T) * 1j
    GamBar = Y_eff @ Gam @ Y_eff

    D, V = jnp.linalg.eig(Fbar)
    Vc = jnp.linalg.inv(V.conj().T)

    # Density matrix; negate to match densityComplex sign convention.
    P_orth = density(V, Vc, D, GamBar, lo, Emin)
    P_lower = -(Y_eff @ jnp.asarray(P_orth) @ Y_eff)

    # Analytic cross-term: linear Q at near-band anchors (Emin, 2*Emin).
    Qa = g.crossTermQTot(Emin)
    if Qa is None:
        delta_N = 0.0
    else:
        Ea, Eb = Emin, 2.0 * Emin
        Qa = jnp.asarray(Qa)
        Qb = jnp.asarray(g.crossTermQTot(Eb))
        Q1 = (Qb - Qa) / (Eb - Ea)
        Q0 = Qa - Q1 * Ea
        delta_N = damleCrossTerm(V, D, Vc, Y_eff, Q0, Q1, lo, Emin)

    return P_lower, delta_N

def bisectFermi(V, Vc, D, Gam, Nexp, conv=FERMI_CALCULATION_TOL, Eminf=ENERGY_MIN):
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
    # Seed fermi so the function is total: if the entry state already
    # satisfies |dN| <= conv the loop never runs (seen on resume of an
    # already-converged predict-path state, job 36915480) and the return
    # would hit an unbound local. The seed equals what iteration 1 would
    # have tested, so search semantics are unchanged.
    fermi = (Emin + Emax) / 2
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
    tuple (ndarray, float)
        (P, delta_N) where P is the density matrix and delta_N is the
        Mulliken cross-term correction from device-lead overlap.
    """
    nKT = N_KT
    kT = kB*T
    Emax = mu + nKT*kT
    mid = (Emax-Emin)/2
    x,w = roots_legendre(N)
    x = np.real(x)

    Elist = mid*(x + 1) + Emin
    weights = mid*w*fermi(Elist, mu, T)

    if showText:
        print(f'Integrating {N} points along real axis...')

    lineInt, cross_scalar = GrIntCross(F, S, g, Elist, weights)

    if showText:
        print('Integration done!')

    # The standard formula P = -Im(G^R)/pi (see 10.1103/PhysRevB.63.245407, Eq. 19)
    P = (1j/(2*jnp.pi)) * (lineInt - lineInt.conj().T)
    delta_N = float(-(1/jnp.pi) * jnp.imag(cross_scalar))
    return P, delta_N

def densityReal(F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False):
    """
    Calculate equilibrium density matrix using adaptive real-axis ANT integration.

    Uses integratePointsAdaptiveANT with the same Gauss-Chebyshev scheme as
    densityComplex, but integrating along the real axis instead of a complex contour.
    Suitable for zero-DOS regions (below all occupied states) where Im(G^R)=0
    and the integral converges trivially in a small number of points.

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
        Temperature in Kelvin (default: TEMPERATURE from gauNEGF.config)
    debug : bool, optional
        Whether to print per-iteration diagnostics (default: False)

    Returns
    -------
    tuple (ndarray, float)
        (P, delta_N) where P is the density matrix and delta_N is the
        Mulliken cross-term correction from device-lead overlap.
    """
    nKT = N_KT
    kT = kB * T
    Emax = mu + nKT * kT

    mid = (Emax - Emin) / 2

    def computePoint(x, w):
        E = mid * (x + 1) + Emin
        weights = mid * w * fermi(E, mu, T)
        mat, scl = GrIntCross(F, S, g, E, weights)
        # Return density matrix contribution so the adaptive integrator tracks
        # convergence on P directly (G^R - G^A, not the full complex G^R).
        return (1j / (2 * jnp.pi)) * (mat - mat.conj().T), scl

    print('Real Axis Integration (ANT):')
    P, cross_scalar = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)
    delta_N = float(-(1 / jnp.pi) * jnp.imag(cross_scalar))
    return P, delta_N
   

def densityGridN(F, S, g, mu1, mu2, ind, N=100, T=TEMPERATURE, showText=True):
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
    ind : int
        Contact index (-1 for total)
    N : int, optional
        Number of integration points (default: 100)
    T : float, optional
        Temperature in Kelvin (default: 300)
    showText : bool, optional
        Whether to print progress messages (default: True)

    Returns
    -------
    tuple (ndarray, float)
        Non-equilibrium contribution to density matrix and its cross-term
        electron-count correction
    """
    nKT = N_KT
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    if mu1 == mu2:
        return np.array(np.zeros(np.shape(F)), dtype=complex), 0.0
    dInt = np.sign(mu2 - mu1) # Sign of bias voltage
    Emax = muHi + nKT*kT
    Emin = muLo - nKT*kT
    mid = (Emax-Emin)/2
    x,w = roots_legendre(N)
    x = np.real(x)

    energies = mid*(x + 1) + Emin
    dfermi = fermi(energies, muHi, T) - fermi(energies, muLo, T)
    weights = mid*w*dfermi*dInt

    if showText:
        print(f'Real integration over {N} points...')

    den, scl = GrLessIntCross(F, S, g, energies, weights, ind)
    scl = complex(scl)
    # an imaginary window count means an assembly bug, not physics
    assert abs(scl.imag) <= 1e-8 * max(1.0, abs(scl.real)), (
        'window cross scalar has non-negligible imaginary part '
        f'({scl.imag:.3e}): assembly bug, not physics')

    if showText:
        print('Integration done!')

    return den/(2*np.pi), float(scl.real)/(2*np.pi)

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

def densityGrid(F, S, g, mu1, mu2, ind, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False):
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
    ind : int
        Contact index (-1 for total)
    tol : float, optional
        Convergence tolerance (default: 1e-3)
    T : float, optional
        Temperature in Kelvin (default: 300)
    debug : bool, optional
        Whether to print debug information (default: False)

    Returns
    -------
    tuple (ndarray, float)
        Non-equilibrium contribution to density matrix and its cross-term
        electron-count correction
    """
    nKT = N_KT
    kT = kB*T
    muLo = min(mu1, mu2)
    muHi = max(mu1, mu2)
    if mu1 == mu2:
        return np.array(np.zeros(np.shape(F)), dtype=complex), 0.0
    dInt = np.sign(mu2 - mu1) # Sign of bias voltage
    Emax = muHi + nKT*kT
    Emin = muLo - nKT*kT
    mid = (Emax-Emin)/2

    def computePoint(x, w):
        E = mid*(x + 1) + Emin
        dFermi = fermi(E, muHi, T) - fermi(E, muLo, T)
        weights = mid*w*dFermi*dInt
        return GrLessIntCross(F, S, g, E, weights, ind)

    den, scl = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)
    scl = complex(scl)
    # an imaginary window count means an assembly bug, not physics
    assert abs(scl.imag) <= 1e-8 * max(1.0, abs(scl.real)), (
        'window cross scalar has non-negligible imaginary part '
        f'({scl.imag:.3e}): assembly bug, not physics')
    if debug:
        print('Integration done!')

    return den/(2*np.pi), float(scl.real)/(2*np.pi)



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
    tuple (ndarray, float)
        (P, delta_N) where P is the equilibrium density matrix and
        delta_N is the Mulliken cross-term correction from device-lead overlap.

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
    # No fermi factor on the arc: its e^-nKT tail oscillates unresolvably
    # and floors the grid at T>0. Do not reintroduce (2026-07-25 fix commit).
    weights = (np.pi/2)*w*dz

    if showText:
        print(f'Complex Integration over {N} points...')

    lineInt, cross_scalar = GrIntCross(F, S, g, Elist, weights)

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
        broadInt, cross_broad = GrIntCross(F, S, g, Elist, weights)
        # Subtract: the arc runs Emax->Emin (backward), so the forward
        # window enters with a minus (sign bug fixed 2026-07-25).
        lineInt -= broadInt
        cross_scalar -= cross_broad

    if showText:
        print('Integration done!')

    # The standard formula P = -Im(G^R)/pi (see 10.1103/PhysRevB.63.245407, Eq. 19)
    P = (-1j/(2*jnp.pi)) * (lineInt - lineInt.conj().T)
    delta_N = float(-(1/jnp.pi) * jnp.imag(cross_scalar))
    return P, delta_N

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
    tuple (ndarray, float)
        (P, delta_N) where P is the equilibrium density matrix and
        delta_N is the Mulliken cross-term correction from device-lead overlap.

    Notes
    -----
    Uses adaptive integration with GrIntCross to co-accumulate both the
    density matrix and cross-term scalar in a single pass. The cross-term
    converges at the same adaptive grid as the density matrix.
    """
    #Construct circular contour
    nKT= 10
    broadening = nKT*kB*T
    Emax = mu-broadening
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2

    # For ANT adaptive integration, compute from point-weight pairs
    # Returns (matrix, scalar) tuple for co-accumulated cross-term
    def computePoint(x, w):
        theta = np.pi/2 * (x + 1)
        z = center + r*np.exp(1j*theta)
        dz = 1j * r * np.exp(1j*theta)
        # No fermi factor on the arc -- see densityComplexN.
        weights = (np.pi/2)*w*dz
        return GrIntCross(F, S, g, z, weights)

    print('Complex Contour Integration:')
    lineInt, cross_scalar = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug)

    #Add integration points for Fermi Broadening
    if T>0:
        print('Integrating Fermi Broadening:')
        def computePointBroadening(x, w):
            E = broadening * (x) + mu
            weights = broadening*w*fermi(E, mu, T)
            return GrIntCross(F, S, g, E, weights)

        broad_lineInt, broad_cross = integratePointsAdaptiveANT(computePointBroadening, tol=tol, debug=debug)
        # Subtract: backward arc, forward window -- see densityComplexN.
        lineInt -= broad_lineInt
        cross_scalar -= broad_cross

    # The standard formula P = -Im(G^R)/pi (see 10.1103/PhysRevB.63.245407, Eq. 19)
    P = (-1j/(2*jnp.pi)) * (lineInt - lineInt.conj().T)
    delta_N = float(-(1/jnp.pi) * jnp.imag(cross_scalar))
    return P, delta_N

## INTEGRATION LIMIT FUNCTIONS
# Calculate Emin using DOS
def calcEmin(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES, Emin=None, X=None):
    """
    Calculate minimum energy bound for numerical integration.

    Finds the lower energy bound for density matrix integration by iteratively
    lowering Emin until the density of states at that energy falls below the
    specified tolerance. This ensures the integration bounds capture all occupied
    states without including energies too far below the band edge.

    Parameters
    ----------
    F : ndarray
        Fock matrix in eV.
    S : ndarray
        Overlap matrix.
    g : surfG object
        Surface Green's function calculator providing sigmaTot() and
        crossTermQTot() methods for energy-dependent self-energies.
    tol : float, optional
        Convergence tolerance for density of states (default: FERMI_CALCULATION_TOL).
    maxN : int, optional
        Maximum number of expansion iterations (default: MAX_CYCLES).
    Emin : float or None, optional
        Initial lower bound in eV. If None, initialized from minimum eigenvalue
        of X @ F @ X (symmetric Lowdin orthogonalization with X = S^(-1/2))
        minus EMIN_BUFFER (default: None).
    X : ndarray or None, optional
        Lowdin orthogonalizer S^(-1/2). If None (default), computed locally
        via fractional_matrix_power(S, -0.5). Passing the caller's cached X
        avoids recomputing it. Used only when Emin is None.

    Returns
    -------
    float
        Lower energy bound in eV where DOS is below tolerance.

    Notes
    -----
    Prints a warning if maximum iterations are reached without convergence.
    X @ F @ X (X = S^(-1/2)) is used for the initial Emin seed because it
    is Hermitian by construction and its eigenvalues are the true device MO
    energies; eigh(inv(S) @ F) would silently fail for non-trivial S.

    .. deprecated::
        Superseded by find_lower_bound + single contour (USE_INERTIA_EMIN).
        Retained for the default (flag-off) path for one release.
    """
    if X is None:
        X = fractional_matrix_power(S, -0.5)
    if Emin is None:
        D,_ = eigh(X@F@X)
        Emin = min(D.real.flatten()) - EMIN_BUFFER
    counter = 0
    dP = _compute_dos_at_energy(Emin, F, S, g.sigmaTot(Emin), g.crossTermQTot(Emin))
    while dP>tol and counter<maxN:
        Emin -= 10
        dP = _compute_dos_at_energy(Emin, F, S, g.sigmaTot(Emin), g.crossTermQTot(Emin))
        counter += 1
    if counter == maxN:
        print(f'Warning: Emin still not within tolerance (final value = {dP}) after {maxN} energy samples')
    print(f'Calculated Emin: {Emin} eV, DOS = {dP:.2E}')
    return Emin

def calcTSW(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=FERMI_SEARCH_CYCLES,
            Eminf=None, TSW=None, Emin_floor=ENERGY_MIN):
    """Calculate integration bounds by stabilizing total spectral weight.

    Iteratively doubles |Eminf| (Eminf <- 2*Eminf) until Tr(rho @ S) at the
    truncated contour matches a reference TSW computed at the configured
    Emin_floor. Uses a one-sided convergence test (TSW_ref - TSW_new < tol)
    so the loop only widens while the new contour is undercounting relative
    to the reference. dTSW < 0 (reference is *smaller* than the truncated
    contour) is the signature of a deep negative-weight pseudo-pole inside
    the reference contour, induced by asymptotically E-linear Sigma(E)
    from a non-orthogonal contact (see docs/tsw_convergence_notes.md);
    in that case the truncated contour is the physically correct choice
    and convergence is accepted to leave the pseudo-pole excluded.

    The upper contour bound is held at -ENERGY_MIN (the wide config bound)
    independently of Emin_floor so that tightening Emin_floor (e.g. via
    calcPseudoPoleFloor) does not also chop the upper spectrum.

    Parameters
    ----------
    F : ndarray
        Fock matrix in eV.
    S : ndarray
        Overlap matrix.
    g : surfG object
        Surface Green's function calculator.
    tol : float, optional
        Convergence tolerance for relative TSW change (default: FERMI_CALCULATION_TOL).
    maxN : int, optional
        Maximum doubling iterations (default: FERMI_SEARCH_CYCLES).
    Eminf : float or None, optional
        Warm-start lower bound in eV. If None, initialized from min eigenvalue.
    TSW : float or None, optional
        Warm-start spectral weight. If None, recomputed from scratch using
        Emin_floor as the reference contour lower bound.
    Emin_floor : float, optional
        Hard lower limit for Eminf in eV (default: ENERGY_MIN from config).
        Also used as the reference contour lower bound when TSW is recomputed.
        If the doubling loop would expand past this limit without converging,
        the function warns and returns the current Eminf. Tighten this via
        calcPseudoPoleFloor to keep the reference contour from straddling
        deep pseudo-poles.

    Returns
    -------
    tuple (float, float)
        (Eminf, TSW) -- converged lower bound and reference spectral weight.

    .. deprecated::
        Superseded by find_lower_bound + single contour (USE_INERTIA_EMIN).
        Retained for the default (flag-off) path for one release.
    """
    # Capture the warm-start for the postcondition check: the returned Eminf
    # must not be shallower than the caller's warm-start (would invert the
    # caller's lower-contour integration). The doubling loop preserves this for
    # valid negative warm-starts, but the guard catches invalid input (e.g. a
    # positive Eminf) and future algorithm changes.
    Eminf_warm = Eminf

    # Upper contour bound stays at the wide config bound regardless of
    # Emin_floor; only the lower bound moves with pseudo-pole detection.
    Emax = -ENERGY_MIN

    # Initialize from eigenvalues if no warm-start values
    if Eminf is None:
        # X @ F @ X (X = S^(-1/2)) is Hermitian by construction; eigh is correct.
        X = fractional_matrix_power(S, -0.5)
        D, _ = eigh(X @ F @ X)
        eigs = np.real(D).flatten()
        Eminf = float(min(eigs))

    if TSW is None:
        P, _ = densityComplex(F, S, g, Emin_floor, Emax, tol, T=0)
        TSW_ref = np.trace(P @ g.S).real
    else:
        TSW_ref = TSW+0.0

    print(f'Lower Bound Search: TSW={TSW_ref:.2E}')
    for _ in range(maxN):
        P, _ = densityComplex(F, S, g, Eminf, Emax, tol, T=0)
        TSW_new = np.trace(P @ g.S).real
        dTSW = TSW_ref - TSW_new
        # One-sided test: the contour at Eminf has captured all relevant
        # spectral weight when its TSW reaches (or exceeds) the reference.
        # dTSW < 0 means the reference (deeper) contour is *smaller* than
        # the truncated one, which only happens when the reference encloses
        # a deep negative-weight pseudo-pole that the truncated contour
        # excludes. See docs/tsw_convergence_notes.md; the truncated
        # contour is the physically correct choice in that case.
        if dTSW / abs(TSW_ref) < tol:
            if FERMI_DEBUG:
                if dTSW < 0:
                    print(f'calcTSW: dTSW<0 detected at Eminf={Eminf:.2f}, '
                          f'dTSW={dTSW:.2E}. Deep pseudo-pole inside reference '
                          f'contour (likely asymptotically E-linear contact '
                          f'Sigma making S_eff = S - X_L - X_R non-PSD). '
                          f'Truncated contour excludes it; this is the '
                          f'intended outcome.')
                print(f'calcTSW converged: Eminf={Eminf:.2f}, TSW={TSW_new:.4f}')
            if Eminf_warm is not None and Eminf > Eminf_warm:
                print(f'WARNING: calcTSW Eminf={Eminf:.2e} shallower than warm-start {Eminf_warm:.2e}; flooring.')
                Eminf = Eminf_warm
            return Eminf, TSW_new
        elif FERMI_DEBUG:
            print(f'DEBUG: Eminf={Eminf:.2f}, dTSW={dTSW:.2E}')

        # Check that the next doubling stays within Emin_floor.
        next_Eminf = Eminf * 2.0
        if next_Eminf < Emin_floor:
            print(f'Warning: calcTSW would expand Eminf past Emin_floor ({Emin_floor:.2e}).')
            print(f'  Last Eminf={Eminf:.2e}, dTSW={dTSW:.2E}.')
            print(f'  If TSW is still undercounted, lower Emin_floor in config.')
            if Eminf_warm is not None and Eminf > Eminf_warm:
                print(f'WARNING: calcTSW Eminf={Eminf:.2e} shallower than warm-start {Eminf_warm:.2e}; flooring.')
                Eminf = Eminf_warm
            return Eminf, TSW_ref
        Eminf = next_Eminf

    print(f'Warning: calcTSW did not converge after {maxN} iterations')
    print(f'  Last Eminf={Eminf:.2e}, dTSW={dTSW:.2E}')
    if Eminf_warm is not None and Eminf > Eminf_warm:
        print(f'WARNING: calcTSW Eminf={Eminf:.2e} shallower than warm-start {Eminf_warm:.2e}; flooring.')
        Eminf = Eminf_warm
    return Eminf, TSW_ref

def calcPseudoPoleFloor(F, S, g, alpha=0.1, min_buffer=50.0, E1=-1e3, E2=-1e4):
    """Compute a safe Eminf floor by detecting pseudo-poles of the device GF.

    Pseudo-poles are spurious real-axis poles of G_DD^R(E) = (E*S - F - Sigma)^{-1}
    that arise from the asymptotically E-linear piece of Sigma when the contact
    is non-orthogonal. If the contour integration captures them, P_DD acquires
    negative natural-orbital occupations and is no longer DFT-compatible.

    All matrices are treated as complex Hermitian throughout, so this works
    for GHF / SOC where F, S, X, Sigma_0 have nontrivial imaginary parts as
    well as the real-symmetric closed-shell case.

    See docs/pseudo_pole_handling.md for the derivation and worked examples.

    Algorithm:
        1. Two-point asymptotic probe of Sigma:
             X       = (Sigma(E2) - Sigma(E1)) / (E2 - E1)
             Sigma_0 = Sigma(E1) - E1 * X
        2. Form the effective generalized eigenproblem H_eff v = E S_eff v
           that governs deep-|E| poles of G_DD^R:
             H_eff = F + Sigma_0       (Hermitian)
             S_eff = S - X             (Hermitian, possibly indefinite)
        3. Reduce to the standard non-Hermitian eigproblem T = S_eff^{-1} H_eff
           and solve with the JAX general eig (via gauNEGF.utils.eig).
           Eigenvalues are real for Hermitian (H, S) even with indefinite S;
           take Re(...) to drop floating-point imaginary noise.
        4. Classify each eigenvector by its Hermitian S_eff-norm:
             v^H S_eff v > 0  -> physical pole (not returned)
             v^H S_eff v < 0  -> pseudo-pole (returned)
           Number of pseudo-poles equals the number of negative eigenvalues
           of S_eff (Sylvester's law of inertia).
        5. Filter to negative-energy pseudo-poles only (positive ones cannot
           be excluded by Eminf, which is a LOWER bound). For each, compute
           floor_i = E_pp_i + max(alpha * |E_pp_i|, min_buffer); return the
           shallowest (largest) floor_i, or ENERGY_MIN if no negative
           pseudo-poles exist or if the shallowest buffered floor is >= 0.

    Parameters
    ----------
    F : ndarray (N, N)
        Device Fock matrix in eV. Complex Hermitian for GHF/SOC, real-symmetric
        otherwise; both are handled.
    S : ndarray (N, N)
        Device overlap matrix. Complex Hermitian, positive-definite.
    g : surfG-like object
        Surface Green's function calculator with sigmaTot(E) method.
    alpha : float, optional
        Fractional buffer coefficient (default 0.1, i.e. 10%). For each
        pseudo-pole at energy E_pp, the buffer is max(alpha * |E_pp|,
        min_buffer). The fractional component reflects the scale-invariance
        of the asymptotic Sigma ~ E*X regime: pole "width" near the resolvent
        singularity scales with |E|, so a constant-eV buffer is overkill at
        deep poles and inadequate at moderate ones.
    min_buffer : float, optional
        Absolute floor for the buffer in eV (default 50). Prevents the
        fractional formula from placing Eminf arbitrarily close to a
        moderate-depth pseudo-pole. Set both alpha=0 and min_buffer=0
        to disable buffering entirely (useful for tests of raw pole
        detection).
    E1, E2 : float, optional
        Deep probe energies (eV) for asymptotic extraction. Default -1e3, -1e4.

    Returns
    -------
    Eminf_floor : float
        Safe lower bound for Eminf in eV. Returns ENERGY_MIN from config if
        no pseudo-poles are detected.

    Notes
    -----
    Buffer formula: per pseudo-pole, the gap is max(alpha * |E_pp|, min_buffer).
    The fractional term handles scale-invariance (deep poles need proportionally
    larger gaps); min_buffer prevents a meaninglessly tight gap near shallow poles.
    Defaults of 0.1 and 50 eV give ~300 eV gap at -3000 eV poles.

    Positive pseudo-poles are filtered out: they cannot be excluded by Eminf
    (a LOWER bound) and cancel in calcTSW's dTSW comparison.

    A returned floor is GUARANTEED strictly negative (or exactly ENERGY_MIN). If
    the shallowest buffered floor is >= 0 (pseudo-pole mixed with physical states),
    returns ENERGY_MIN and lets calcTSW's dTSW<0 path handle the system.

    F, S, X, Sigma_0 are kept fully complex: taking Re() of a Hermitian matrix
    drops its antisymmetric imaginary part, giving a symmetric matrix with
    different eigenvalues -- this matters for GHF/SOC.
    """
    # Two-point asymptotic probe of Sigma (preserves complex-Hermitian structure).
    Sigma1 = jnp.asarray(g.sigmaTot(E1))
    Sigma2 = jnp.asarray(g.sigmaTot(E2))
    X = (Sigma2 - Sigma1) / (E2 - E1)
    Sigma_0 = Sigma1 - E1 * X

    # Effective generalized (H, S) for the deep-|E| asymptotic G_DD^R.
    H_eff = jnp.asarray(F) + Sigma_0
    S_eff = jnp.asarray(S) - X

    # Reduce (H, S) generalized problem -- where S may be INDEFINITE -- to the
    # standard non-Hermitian eigproblem of T = S^{-1} H. utils.eig (jit-wrapped
    # jnp.linalg.eig) handles complex / non-Hermitian T via LAPACK.
    eigvals_c, eigvecs_c = eig(inv(S_eff) @ H_eff)

    # Eigenvalues are real for Hermitian (H, S) even with indefinite S.
    # Real-part is to discard numerical imaginary noise from JAX's complex-typed
    # return, not a physics approximation.
    eigvals = jnp.real(eigvals_c)

    # Hermitian S_eff-norm: v^H S v (use conj() for the H-conjugate transpose).
    # v^H S v is real for Hermitian S; we wrap in real() just to drop the
    # exact-zero imaginary part from einsum.
    norms = jnp.real(jnp.einsum('in,ij,jn->n',
                                 eigvecs_c.conj(), S_eff, eigvecs_c))

    pp_energies = np.asarray(jnp.sort(eigvals[norms < 0]))

    # Positive pseudo-poles cannot be excluded by Eminf (it's a LOWER bound);
    # only negative-energy pseudo-poles inform the floor.
    pp_neg = pp_energies[pp_energies < 0]
    if len(pp_neg) == 0:
        return float(ENERGY_MIN)

    # Per-pseudo-pole buffer: fractional (alpha * |E_pp|) because the
    # asymptotic E*X regime that creates pseudo-poles is scale-invariant,
    # plus an absolute min_buffer so shallow poles don't get a meaninglessly
    # tight gap. Each pole's required floor is E_pp + buffer(E_pp).
    abs_pp = np.abs(pp_neg)
    buffers = np.maximum(alpha * abs_pp, min_buffer)
    floors = pp_neg + buffers

    # Eminf must clear ALL deep pseudo-poles -> take the shallowest (largest)
    # of the per-pole floors. If even that is non-negative, no safe negative
    # Eminf exists; fall back to ENERGY_MIN and let calcTSW's dTSW<0 path
    # handle the system.
    shallowest = float(np.max(floors))
    if shallowest >= 0:
        return float(ENERGY_MIN)
    return shallowest

def find_lower_bound(F, S, g, maxN=MAX_CYCLES, factor=2.0):
    """Integration lower bound (Eminf) below every pole of G^R = [E S - F -
    Sigma(E)]^{-1}, via a seed + step-down walk referenced to the TOTAL SPECTRAL
    WEIGHT (not the matrix dimension N).

    The number of poles -- states with real spectral weight -- is generally LESS
    than the matrix dimension N for a non-orthogonal basis: the finite overlap and
    the asymptotically E-linear contact self-energy reduce the effective count
    (a handful of eigenvalues of M(E) ~ E*(S - X_asymp) never go negative). So the
    reference is n_ref = n_neg(ENERGY_MIN), the inertia count of the Hermitian part
    of M(E) = E S - F - Sigma(E) at the deepest config cutoff -- the true number of
    poles. The walk seeds at calcEmin's band floor (min generalized eigenvalue of
    (F, S) minus EMIN_BUFFER) and steps DOWN with calcTSW's geometric doubling
    until n_neg(E) reaches n_ref, i.e. E is below every pole. Referencing n_ref
    instead of N is what makes this robust to a (mildly) non-PSD effective overlap;
    the DOS test calcEmin uses would have to guess that count. The seed is usually
    already below every pole, so the common case returns with zero steps. All
    matrix math is JAX (on-device eigh).

    Parameters
    ----------
    F : ndarray
        Fock matrix in eV.
    S : ndarray
        Overlap matrix.
    g : surfG object
        Provides sigmaTot(E) -> contact self-energy at energy E (eV).
    maxN : int, optional
        Maximum number of doubling steps (default: MAX_CYCLES).
    factor : float, optional
        Geometric step-down factor applied to Emin each step (default: 2.0).

    Returns
    -------
    float
        Integration lower bound in eV (n_neg == n_ref there, below every pole).
        Always defined -- n_ref = n_neg(ENERGY_MIN) is reached by ENERGY_MIN at
        the latest.
    """
    Fj = jnp.asarray(F)
    Sj = jnp.asarray(S)

    def _nneg(E):
        # Number of negative eigenvalues of the Hermitian part of M(E) = number
        # of poles below E.
        M = float(E) * Sj - Fj - jnp.asarray(g.sigmaTot(float(E)))
        Mh = 0.5 * (M + M.conj().T)
        return int(jnp.sum(eigh(Mh)[0] < 0.0))

    # Reference: total number of poles = inertia count at the deepest config
    # cutoff (< matrix dimension N for a non-orthogonal basis / finite overlap).
    n_ref = _nneg(float(ENERGY_MIN))

    # Seed at the Lowdin band floor minus EMIN_BUFFER -- calcEmin's seed, on JAX.
    ws, Vs = eigh(Sj)
    X = Vs @ jnp.diag(ws ** -0.5) @ Vs.conj().T
    D = jnp.real(eigh(X @ Fj @ X)[0])
    Emin = float(jnp.min(D)) - EMIN_BUFFER

    # Walk down (calcTSW doubling) until all n_ref poles are below Emin. Emin then
    # sits just below the lowest pole.
    counter = 0
    while _nneg(Emin) < n_ref and counter < maxN and Emin > ENERGY_MIN:
        Emin = max(Emin * factor, float(ENERGY_MIN))
        counter += 1
    # Push a full EMIN_BUFFER below that point so the contour's lower endpoint
    # stays well clear of the poles. The seed's buffer is below the bare band
    # floor (min eig), which the contact self-energy's downward shift of the
    # lowest pole can erode to ~nothing -- so the buffer must be applied here,
    # relative to where the poles actually end.
    return max(float(ENERGY_MIN), Emin - EMIN_BUFFER)

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
        rho_, _ = densityComplexN(F, S, g, Emin,  mu, Ncomplex, T=T)
        rho_ = np.real(rho_)
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
        rho_ = np.real(densityRealN(F, S, g, Eminf, Emin, Nreal, T=0)[0])
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
        Pw, _ = densityGridN(F, S, g, fermi, fermi+(qV/2), ind=0, N=N, T=T)
        rho_ = np.real(Pw)
        Pw, _ = densityGridN(F, S, g, fermi, fermi-(qV/2), ind=-1, N=N, T=T)
        rho_ += np.real(Pw)
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}")
        rho = rho_
    if dP < tol:
        N /= 2
    elif N >= maxGrid and dP > tol:
        print(f'Warning: N still not within tolerance (final value = {dP})')
    print(f'Final Nnegf: {N}') 
    return N


def getFermiContact(g, ne, Emin=None, lBound=None, uBound=None, tol=ADAPTIVE_INTEGRATION_TOL,
                    conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE,
                    useInertiaEmin=None):
    """
    Calculate Fermi energy for a contact using adaptive integration.

    Determines the Fermi energy for a contact system (3D lattice or 1D chain)
    by matching the electron count using bisection with adaptive complex contour integration.

    Parameters
    ----------
    g : surfG object
        Surface Green's function calculator
    ne : float
        Target number of electrons
    Emin : float, optional
        Lower bound for complex contour integration in eV. If None, calculated from DOS.
    lBound : float, optional
        Lower bound for bisection search in eV. If None, estimated from eigenvalues.
    uBound : float, optional
        Upper bound for bisection search in eV. If None, estimated from eigenvalues.
    tol : float, optional
        Tolerance for adaptive integration (default: ADAPTIVE_INTEGRATION_TOL)
    conv : float, optional
        Convergence tolerance for electron count (default: FERMI_CALCULATION_TOL)
    maxcycles : int, optional
        Maximum number of iterations (default: FERMI_SEARCH_CYCLES)
    T : float, optional
        Temperature in Kelvin (default: TEMPERATURE)

    Returns
    -------
    float
        Fermi energy in eV
    """
    # Set up initial guess from eigenvalues
    S = g.S
    F = g.F
    orbs, _ = eig(inv(S)@F)
    orbs = np.sort(np.real(orbs))

    if useInertiaEmin is None:
        useInertiaEmin = USE_INERTIA_EMIN

    if useInertiaEmin:
        # SINGLE-CONTOUR inertia path: find_lower_bound returns the integration
        # floor Eminf directly (seed + inertia step-down); no separate lower
        # contour, no nLower subtraction. calcFermi integrates [Eminf, mu] and
        # targets the full ne.
        Eminf = find_lower_bound(F, S, g)
        if Eminf is not None:
            if lBound is None:
                lBound = min(orbs)
            if uBound is None:
                uBound = max(orbs)
            idx = max(min(int(ne), len(orbs) - 1), 1)
            Ef = (orbs[idx - 1] + orbs[idx]) / 2
            print(f"Single-contour floor Eminf = {Eminf:.2f} eV")
            # trackFermi=False: contact determination holds the leads
            # fixed; only device searches (known fermi0) track trial mu.
            return calcFermi(g, ne, Eminf, Ef, lBound=lBound, uBound=uBound,
                             tol=tol, conv=conv, maxcycles=maxcycles, T=T,
                             trackFermi=False)
        print("find_lower_bound returned None; falling back to legacy two-contour path")

    # Calculate Emin from DOS if not provided
    if Emin is None:
        Emin = calcEmin(F, S, g, tol=conv, maxN=maxcycles)

    # Count electrons below Emin (zero-DOS region: real-axis ANT converges in ~6 pts)
    Eminf, _TSW = calcTSW(F, S, g, Eminf=Emin, tol=conv)
    P, _delta_N_lower = densityComplex(F, S, g, Eminf, Emin, tol, T=0)
    nLower = np.trace(P@g.S).real + _delta_N_lower
    assert nLower < ne, "ne ({ne}) exceeds mininum number of electrons ({nLower:.2f})"
    print(f"{nLower:.2f} electrons below Emin.")
    ne -= nLower # Subtract from total

    # Set bounds from eigenvalues if not provided
    if lBound is None:
        lBound = min(orbs)
    if uBound is None:
        uBound = max(orbs)

    # Initial guess for Fermi energy. Clamp the orbital index to [1, len-1] so a
    # full/near-full count (orbs[int(ne)] out of range) or ne<1 (negative wrap to
    # orbs[-1]) cannot blow up -- Ef is only calcFermi's starting guess.
    idx = max(min(int(ne), len(orbs) - 1), 1)
    Ef = (orbs[idx - 1] + orbs[idx]) / 2

    return calcFermi(g, ne, Emin, Ef, lBound=lBound, uBound=uBound,
                    tol=tol, conv=conv, maxcycles=maxcycles, T=T,
                    trackFermi=False)

# Calculate the fermi energy of the surface Green's Function object
def calcFermi(g, ne, Emin, Ef, lBound=None, uBound=None, tol=ADAPTIVE_INTEGRATION_TOL,
              conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE,
              trackFermi=True):
    """
    Calculate Fermi energy using bisection method with adaptive integration.

    Parameters
    ----------
    g : surfG object
        Surface Green's function calculator
    ne : float
        Target number of electrons
    Emin : float
        Lower bound for complex contour integration in eV
    Ef : float
        Initial guess for Fermi energy in eV
    lBound : float, optional
        Lower bound for bisection search in eV. If None, determined dynamically.
    uBound : float, optional
        Upper bound for bisection search in eV. If None, determined dynamically.
    tol : float, optional
        Tolerance for adaptive integration (default: ADAPTIVE_INTEGRATION_TOL)
    conv : float, optional
        Convergence criterion for electron count (default: FERMI_CALCULATION_TOL)
    maxcycles : int, optional
        Maximum number of iterations (default: FERMI_SEARCH_CYCLES)
    T : float, optional
        Temperature in Kelvin (default: TEMPERATURE)
    Returns
    -------
    float
        Calculated Fermi energy in eV
    """
    assert ne <= len(g.F), "Number of electrons cannot exceed number of basis functions!"

    # Use adaptive complex contour integration
    pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)

    E = Ef
    counter = 0
    dE = tol

    # Initial calculation
    if trackFermi:
        g.setF(g.F, E, E)
    P, _delta_N = pMu(E)
    Ncurr = np.trace(P@g.S).real + _delta_N

    # Establish bounds if not provided
    while None in [uBound, lBound] and counter < maxcycles:
        if counter == maxcycles - 1:
            dE = 1e3
        if Ncurr > ne:
            uBound = E
            Ef = uBound
            E -= dE
        if Ncurr < ne:
            lBound = E
            Ef = lBound
            E += dE
        if FERMI_DEBUG:
            print(f"DEBUG: Ef={Ef:.2f}, dN={ne-Ncurr:.2E}, dE={dE:.2E}")

        # Estimate step size using DOS
        dos = _compute_dos_at_energy(E, g.F, g.S, g.sigmaTot(E), g.crossTermQTot(E), max(g.eta, ETA))
        dE = max(2*abs(Ncurr-ne)/dos, dE)
        counter += 1

        if trackFermi:
            g.setF(g.F, E, E)
        P, _delta_N = pMu(E)
        Ncurr = np.trace(P@g.S).real + _delta_N

    # Bisection search
    print('Calculating Fermi energy using bisection with adaptive integration:')
    while abs(ne - Ncurr) > conv and counter < maxcycles and uBound != lBound:
        dN = ne - Ncurr
        if dN > 0 and Ef > lBound:
            lBound = Ef
        elif dN < 0 and Ef < uBound:
            uBound = Ef
        Ef = (uBound + lBound) / 2
        dE = uBound - lBound
        if FERMI_DEBUG:
            print(f"DEBUG: Ef={Ef:.2f}, dN={dN:.2E}, dE={dE:.2E}")
        counter += 1
        if abs(dN) > conv:
            if trackFermi:
                g.setF(g.F, Ef, Ef)
            P, _delta_N = pMu(Ef)
            Ncurr = np.trace(P@g.S).real + _delta_N

    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(Ncurr-ne):.2E}')
    elif uBound == lBound:
        print(f'Warning: Bisection failed, convergence = {abs(Ncurr-ne):.2E}')
    else:
        print(f'Finished after {counter} iterations, Ef = {Ef:.2f}')

    return Ef

def calcFermiBisect(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, 
                    maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE, uBound=None, lBound=None):
    """
    Calculate Fermi energy of system using bisection
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:   
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)
    E = Ef + 0.0
    Ef0 = Ef + 0.0
    P = None
    Ncurr = ne+0
    dE = tol
    counter = 0
    g.setF(g.F, E, E)
    P, _delta_N = pMu(E)
    Ncurr = np.trace(P@g.S).real + _delta_N
    while None in [uBound, lBound] and counter<maxcycles:
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
        if FERMI_DEBUG:
            print(f"DEBUG: Ef={Ef:.2f}, dN={ne-Ncurr:.2E}, dE={dE:.2E}")
        dos = _compute_dos_at_energy(E, g.F, g.S, g.sigmaTot(E), g.crossTermQTot(E), max(g.eta, ETA))
        dE = max(2*abs(Ncurr-ne)/dos, dE)
        counter += 1
        g.setF(g.F, E, E)
        P, _delta_N = pMu(E)
        Ncurr = np.trace(P@g.S).real + _delta_N
    while abs(ne - Ncurr) > conv and counter < maxcycles and uBound != lBound:
        dN = ne-Ncurr
        if dN > 0 and Ef > lBound:
            lBound = Ef + 0.0
        elif dN < 0 and Ef < uBound:
            uBound = Ef + 0.0
        Ef = (uBound + lBound)/2
        dE = uBound - lBound
        if FERMI_DEBUG:
            print(f"DEBUG: Ef={Ef:.2f}, dN={dN:.2E}, dE={dE:.2E}")
        counter += 1
        if abs(dN) > conv:
            g.setF(g.F, Ef, Ef)
            P, _delta_N = pMu(Ef)
            Ncurr = np.trace(P@g.S).real + _delta_N
    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(Ncurr-ne):.2E}')
        print(f'Reverting to previous Fermi level...')
        Ef = Ef0
    elif uBound == lBound:
        print(f'Warning: Bisection failed, convergence = {abs(Ncurr-ne):.2E}')
        print(f'Reverting to previous Fermi level...')
        Ef = Ef0
    return Ef, dE, P

def calcFermiSecant(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, 
                    conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, 
                    T=TEMPERATURE):
    """
    Calculate Fermi energy using Secant method, updating dE at each step
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:   
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)
    g.setF(g.F, Ef, Ef)
    P, _delta_N = pMu(Ef)
    nCurr = np.trace(P@g.S).real + _delta_N
    dE = conv
    counter = 0
    while abs(nCurr-ne) > conv and counter < maxcycles:
        Ef += dE
        g.setF(g.F, Ef, Ef)
        P, _delta_N = pMu(Ef)
        nNext = np.trace(P@g.S).real + _delta_N
        if FERMI_DEBUG:
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
                    T=TEMPERATURE):
    """
    Calculate Fermi energy using Muller's method, starting with 3 initial points
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"
    small = 1e-10  # Small value to prevent division by zero
    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:   
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)
    Ef0 = Ef + 0.0

    # Initialize three points around initial guess
    E2 = Ef
    E1 = E2 - conv
    E0 = E2 + conv

    uBound = None
    lBound = None

    # Get initial density matrices and electron counts
    nList = []
    for E in [E2, E1, E0]:
        g.setF(g.F, E, E)
        P, _delta_N = pMu(E)
        n = np.trace(P@g.S).real + _delta_N - ne
        if n > 0:
            uBound = min(uBound, E) if uBound is not None else E
        elif n < 0:
            lBound = max(lBound, E) if lBound is not None else E
        if abs(n) < conv:
            return E, 0, P, abs(n), uBound, lBound
        nList.append(n)
    n2, n1, n0 = nList

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
        P, _delta_N = pMu(E2)
        n2 = np.trace(P@g.S).real + _delta_N - ne

        if n2 > 0:
            uBound = min(uBound, E2) if uBound is not None else E2
        elif n2 < 0:
            lBound = max(lBound, E2) if lBound is not None else E2

        # Check convergence
        if abs(n2) < conv:
            break

        if FERMI_DEBUG:
            print(f"DEBUG: Ef={E2:.2f}, dN={n2:.2E}, dE={dE:.2E}")
        counter += 1

    if counter == maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(n2):.2E}')
        print(f'Reverting to previous Fermi level...')
        Ef = Ef0

    return E2, dE, P, abs(n2), uBound, lBound

def calcFermiPolyFit(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL,
                    conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES,
                    T=TEMPERATURE, order=3):
    """
    Calculate Fermi energy using polynomial regression fit with accumulating points.

    Uses all accumulated points to fit a polynomial of specified order through (n, E) pairs
    using least squares regression (polyfit) and finds where n = ne (target electron count).

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
    order : int, optional
        Order of polynomial fit (default: 3)

    Returns
    -------
    tuple
        (Ef, dE, P, error) - Calculated Fermi energy, last energy step, density matrix, and final error

    Notes
    -----
    This method builds up a history of (n, E) points and uses polynomial regression
    to fit an inverse polynomial (E as a function of n) to find E at n = ne.
    Uses numpy.polyfit for regression and numpy.roots to find the solution.
    Polynomial order is min(counter, order) to avoid overfitting with few points.
    """
    assert ne < len(g.F), "Number of electrons cannot exceed number of basis functions!"

    if N is None:
        pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, tol, T)
    else:
        pMu = lambda E: densityComplexN(g.F, g.S, g, Emin, E, N, T)

    # Lists to store history of points
    E_pts = []
    n_pts = []
    
    uBound = None
    lBound = None

    # Initialize with first point
    E = Ef
    g.setF(g.F, E, E)
    P, _delta_N = pMu(E)
    n = np.trace(P@g.S).real + _delta_N - ne

    if abs(n) < conv:
        return E, 0, P, abs(n), uBound, lBound

    E_pts.append(E)
    n_pts.append(n)

    # Enforce monotonicity: Higher E -> Higher N-ne
    step = conv*10
    n_first = n
    counter = 1
    while counter < maxcycles:
        E = Ef + step
        g.setF(g.F, E, E)
        P, _delta_N = pMu(E)
        n = np.trace(P@g.S).real + _delta_N - ne
        if n > 0:
            uBound = min(uBound, E) if uBound is not None else E
        elif n < 0:
            lBound = max(lBound, E) if lBound is not None else E
        if abs(n) < conv:
            return E, step, P, abs(n), uBound, lBound

        # Check if we have a meaningful difference from first point
        if n - n_first > 0:
            break

        # Otherwise increase step size and try again
        step *= 10
        counter += 1
        if FERMI_DEBUG:
            print(f'Warning: Tried Ef = {E:2f} eV (too close to {Ef:2f} to get accurate dN {n-n_first:.2E})')

    E_pts.append(E)
    n_pts.append(n)

    dE = step

    from scipy.optimize import least_squares, minimize
    from scipy.interpolate import PchipInterpolator

    while counter < maxcycles:
        # Use polynomial regression to find E where n = 0 (since we store n - ne)
        # Fit n(E) and solve for roots where n = 0

        # Determine polynomial order: use min(counter, order) to avoid overfitting
        poly_order = min(len(n_pts) - 1, order)

        # PCHIP preserves monotonicity (scipy's shape-preserving cubic interpolator)
        # It won't add spurious oscillations and respects monotonic data
        Esort, nsort = list(zip(*sorted(zip(E_pts,n_pts))))
        pchip = PchipInterpolator(Esort, nsort)
        n_pts_smooth = pchip(E_pts)  # Smooth version at same x points
    
        # Robust polynomial fit with Huber loss
        p0 = np.polyfit(E_pts, n_pts, poly_order) # Initial guess using np.polyfit
        def residuals(coeffs):
            return np.polyval(coeffs, E_pts) - n_pts_smooth
        result = least_squares(residuals, p0, loss='huber', f_scale=ADAPTIVE_INTEGRATION_TOL)

        # Find roots of n(E) = 0
        roots = np.roots(result.x)

        # Find root nearest to last E value (most physically reasonable)
        # Handle complex roots by taking real part
        root_distances = np.abs(roots - E_pts[-1])
        nearest_idx = np.argmin(root_distances)
        E_next = roots[nearest_idx].real

        # Calculate new point
        E = E_next
        g.setF(g.F, E, E)
        P, _delta_N = pMu(E)
        n = np.trace(P@g.S).real + _delta_N - ne
        if n > 0:
            uBound = min(uBound, E) if uBound is not None else E
        elif n < 0:
            lBound = max(lBound, E) if lBound is not None else E

        # Update history
        E_pts.append(E)
        n_pts.append(n)

        # Calculate step size for reporting
        dE = E - E_pts[-2]

        # Check convergence
        if abs(n) < conv:
            break

        if FERMI_DEBUG:
            print(f"Iter {counter}: E = {E:.6f}, n-ne = {n:.3e}, dE = {dE:.3e}, order = {poly_order}")
        counter += 1

    if counter >= maxcycles:
        print(f'Warning: Max cycles reached, convergence = {abs(n):.2E}')

    return E, dE, P, abs(n), uBound, lBound


