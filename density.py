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
    """Green's function calculation (no broadening)"""
    #return cp.asnumpy(cp.linalg.inv(cp.asarray(E*S - F - g.sigmaTot(E))))
    return LA.inv(E*S - F - g.sigmaTot(E))

def fermi(E, mu, T):
    """Fermi function with T=0 convergent case"""
    kT = kB*T
    if kT==0:
        return (E<=mu)*1
    else:
        return 1/(np.exp((E - mu)/kT)+1)

def DOSg(F, S, g, E):
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

def getANTPoints(N):
    """
    Generate integration points and weights matching ANT.Gaussian implementation.
    Following IntCompPlane subroutine in device.F90.
    
    Args:
        N (int): Number of points
    
    Returns:
        tuple: (points, weights) - numpy arrays of integration points and weights
    """
    
    k = np.arange(1,N+1,2)
    theta = k*np.pi/(2*N)
    xs = np.sin(theta)
    xcc = np.cos(theta)

    #print(k,xs,xcc)

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
    Integration of points optimized for quantum transport calculations.
    Defaults to numpy's built-in parallelization for matrix operations.
    
    Parameters:
    -----------
    computePointFunc : callable
        Function that computes a single integration point
    numPoints : int
        Total number of points to integrate
    parallel : bool, optional (default=False)
        Whether to force process-level parallel processing
    numWorkers : int, optional
        Number of workers for parallel processing. If None, automatically determined.
    chunkSize : int, optional
        Size of chunks for parallel processing. If None, automatically determined.
    debug : bool, optional (default=False)
        Whether to print debug information
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
# Requires D, V = eig(H - sigma) and Gam = X@(sigma - sigma.conj().T)@X
# Get density using analytical integration
def density(V, Vc, D, Gam, Emin, mu):
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

# Locate the fermi energy using bisection
def bisectFermi(V, Vc, D, Gam, Nexp, conv=1e-3, Eminf=-1e6):
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
# Get equilibrium density at a single contact (ind) using a real energy grid
def densityReal(F, S, g, Emin, mu, N=100, T=300, parallel=False,
                numWorkers=None, showText=True):
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

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGrid(F, S, g, mu1, mu2, ind=None, N=100, T=300, parallel=False,
                numWorkers=None, showText=True):
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

# Get equilibrium density using a complex contour and a Gaussian quadrature
def densityComplex(F, S, g, Emin, mu, N=100, T=300, parallel=False, numWorkers=None, 
                         showText=True, method='ant'):
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
        #print('Theta=',theta, ', Weight=',w[i])
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
# A simple fitting algorithm for Emin, N1, N2
def integralFit(F, S, g, mu, Eminf, tol=1e-6, maxcycles=1000):

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

# Get the fermi energy of a contact (compatible with Bethe or 1D)
def getFermiContact(g, ne, tol=1e-4, Eminf=-1e6, maxcycles=1000, nOrbs=0):
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

# Get the fermi energy of a 1D contact
def getFermi1DContact(gSys, ne, ind=0, tol=1e-4, Eminf=-1e6, maxcycles=1000):
    # Set up infinite system from contact
    F = gSys.aList[ind]
    S = gSys.aSList[ind]
    tau = gSys.bList[ind]
    stau = gSys.bSList[ind]
    inds = np.arange(len(F))
    #print(S.shape, S.shape, inds.shape, tau.shape, stau.shape)
    g = surfG(F, S, [inds], [tau], [stau], eta=1e-6)

    # Initial guess and integral setup using two layers
    Forbs = np.block([[F, tau], [tau.conj().T, F]])
    Sorbs = np.block([[S, stau], [stau.T, S]])
    gorbs = surfG(Forbs, Sorbs, [inds], [tau], [stau], eta=1e-6)
    orbs, _ = LA.eig(LA.inv(Sorbs)@Forbs)
    orbs = np.sort(np.real(orbs))
    fermi = (orbs[2*int(ne)-1] + orbs[2*int(ne)])/2
    #print(orbs, fermi)
    Emin, N1, N2 = integralFit(Forbs, Sorbs, gorbs, fermi, Eminf, tol, maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, Eminf, tol, maxcycles)

# Calculate the fermi energy of the surfG() object
def calcFermi(g, ne, Emin, Emax, fermiGuess=0, N1=100, N2=50, Eminf=-1e6, tol=1e-4, maxcycles=20, nOrbs=0):
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

