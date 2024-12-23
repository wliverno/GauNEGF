# Python packages
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy.special import roots_legendre
from scipy.special import roots_chebyu
import matplotlib.pyplot as plt

# Developed Packages:
from fermiSearch import DOSFermiSearch
from surfG1D import surfG

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
kB = 8.617e-5           # eV/Kelvin


## HELPER FUNCTIONS
def Gr(F, S, g, E):
    return LA.inv(E*S - F - g.sigmaTot(E)) 

def fermi(E, mu, T):
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
def densityReal(F, S, g, Emin, mu, N=100, T=300, showText=True):
    kT = kB*T
    Emax = mu + 5*kT
    mid = (Emax-Emin)/2
    defInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    if showText:
        print(f'Real integration over {N} points...')
    for i, val in enumerate(x):
        E = mid*(val + 1) + Emin
        defInt += mid*w[i]*Gr(F, S, g, E)*fermi(E, mu, T)
    if showText:
        print('Integration done!')
    
    return (-1+0j)*np.imag(defInt)/(np.pi)

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGrid(F, S, g, mu1, mu2, ind=None, N=100, T=300):
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
    print(f'Real integration over {N} points...')
    for i, val in enumerate(x):
        E = mid*(val + 1) + Emin
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
def densityComplex(F, S, g, Emin, mu, N=100, T=300, showText=True, method='ant'):
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
    if showText:
       print(f'Complex integration over {N} points...')
    lineInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    #x = np.real(x)
    for i, val in enumerate(x):
        theta = np.pi/2 * (val + 1)
        #print('Theta=',theta, ', Weight=',w[i])
        z = center + r*np.exp(1j*theta)
        dz = 1j * r * np.exp(1j*theta)
        lineInt += (np.pi/2)*w[i]*Gr(F, S, g, z)*fermi(z, mu, T)*dz
    
    #Add integration points for Fermi Broadening
    if T>0:
        if showText:
            print('Integrating Fermi Broadening')
        x,w = roots_legendre(N//10)
        x = np.real(x)
        for i, val in enumerate(x):
            E = broadening * (val) + mu
            lineInt += broadening*w[i]*Gr(F, S, g, E)*fermi(E, mu, T)
    if showText:
        print('Integration done!')

    
    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.abs(np.imag(lineInt))/np.pi


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
        Egrid2 = np.linspace(Emax, Emax+2*broadening, N//10)
        for i in range(1, N//10):
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
    Ncomplex = 8
    dP = 100
    rho = np.zeros(np.shape(F))
    while dP > tol and Ncomplex < 1000:
        Ncomplex *= 2 # Start with 16 points, double each time
        rho_ = np.real(densityComplex(F, S, g, Emin,  mu, Ncomplex, T=0))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}, N = {sum(np.diag(rho_).real):2f}")
        rho = rho_
        counter += 1
    if Ncomplex >  maxcycles and dP > tol:
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
    if Nreal >  maxcycles and dP > tol:
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
def calcFermi(g, ne, Emin, Emax, fermiGuess=0, N1=100, N2=50, Eminf=-1e6, tol=1e-4, maxcycles=1000, nOrbs=0):
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
