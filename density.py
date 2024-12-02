import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy.special import roots_legendre
import scipy.io as io
import sys
import time
import matplotlib.pyplot as plt
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

from fermiSearch import DOSFermiSearch
from surfGreen import surfG

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
V_to_au = 0.03675       # Volts to Hartree/elementary Charge
kB = 8.617e-5           # eV/Kelvin


## HELPER FUNCTIONS
def Gr(F, S, g, E):
    return LA.inv(E*S - F - g.sigmaTot(E)) 

def fermi(E, mu, T):
    kT = kB*T
    if kT==0:
        return (E<mu)*1
    else:
        return 1/(np.exp((E - mu)/kT)+1)

def DOSg(F, S, g, E):
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

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
    while abs(dN) > conv:
        fermi = (Emin + Emax)/2
        P = density(V, Vc, D, Gam, Eminf, fermi)
        dN = np.trace(P).real - Nexp
        if dN>0:
            Emax = fermi
        else:
            Emin = fermi
        Niter += 1
    print(f'Bisection fermi search converged to {dN:.2E} in {Niter} iterations.')
    return fermi

## ENERGY DEPENDENT DENSITY FUNCTIONS
# Get equilibrium density at a single contact (ind) using a real energy grid
def densityReal(F, S, g, Emin, mu, N=100, T=300):
    kT = kB*T
    Emax = mu + 5*kT
    mid = (Emax-Emin)/2
    defInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    print(f'Integrating over {N} points...')
    for i, val in enumerate(x):
        E = mid*(val + 1) + Emin
        defInt += mid*w[i]*Gr(F, S, g, E)*fermi(E, mu, T)
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
    print(f'Integrating over {N} points...')
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
    print(f'Integrating over {N} points...')
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
def densityComplex(F, S, g, Emin, mu, N=100, T=300):
    #Construct circular contour
    nKT= 5
    broadening = nKT*kB*T
    Emax = mu-broadening
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2
    
    #Integrate along contour
    print(f'Integrating over {N} points...')
    lineInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    x,w = roots_legendre(N)
    x = np.real(x)
    for i, val in enumerate(x):
        theta = np.pi/2 * (val + 1)
        z = center + r*np.exp(1j*theta)
        dz = 1j * r * np.exp(1j*theta)
        lineInt += (np.pi/2)*w[i]*Gr(F, S, g, z)*fermi(z, mu, T)*dz
    
    #Add integration points for Fermi Broadening
    if T>0:
        print('Integrating Fermi Broadening')
        x,w = roots_legendre(N//10)
        x = np.real(x)
        for i, val in enumerate(x):
            E = broadening * (val) + mu
            lineInt += broadening*w[i]*Gr(F, S, g, E)*fermi(E, mu, T)
    print('Integration done!')

    
    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.abs(np.imag(lineInt))/np.pi


# Get density using a complex contour and trapezoidal integration
def densityComplexTrap(F, S, g, Emin, mu, N, T=300):
    #Construct circular contour
    kT = kB*T
    Emax = mu+(5*kT)
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2
    theta = np.linspace(0, np.pi, N)
    Egrid = r*np.exp(1j*theta)+center

    #Integrate along contour
    print(f'Integrating over {N} points...')
    lineInt = np.array(np.zeros(np.shape(F)), dtype=complex)
    for i in range(1,N):
        E = (Egrid[i]+Egrid[i-1])/2
        dS = Egrid[i]-Egrid[i-1]
        lineInt += Gr(F, S, g, E)*fermi(E, mu, T)*dS
    print('Integration done!')
    
    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi


## INTEGRATION LIMIT FUNCTIONS
# A simple fitting algorithm for Emin, N1, N2
def integralFit(F, S, g, mu, Eminf, tol=1e-6, maxcycles=1000):

    # Calculate Emin using DOS
    D,_ = LA.eig(LA.inv(S)@F)
    Emin = min(D.real.flatten())
    counter = 0
    dP = DOSg(F,S,g,Emin)
    while dP>tol and counter<maxcycles:
        Emin -= 1
        dP = DOSg(F,S,g,Emin)
        print(Emin, dP)
        counter += 1
    if counter == maxcycles:
        print(f'Warning: Emin still not within tolerance (final value = {dP}) after {maxcycles} energy samples')
    print(f'Final Emin: {Emin} eV') 
    
    #Determine grid using dP
    counter = 0
    Ncomplex = 8
    dP = 100
    rho = np.zeros(np.shape(F))
    while dP > tol and Ncomplex < maxcycles:
        Ncomplex *= 2 # Start with 16 points, double each time
        rho_ = np.real(densityComplex(F, S, g, Emin,  mu, Ncomplex, T=0))
        dP = max(abs(np.diag(rho_ - rho)))
        print(f"MaxDP = {dP:.2E}")
        rho = rho_
        counter += 1
    if Ncomplex >  maxcycles and dP > tol:
        print(f'Warning: Ncomplex still not within tolerance (final value = {dP})')
    print(f'Final Ncomplex: {Ncomplex}') 
    #Determine grid using dP
    counter = 0
    Nreal = 8
    dP = 100
    rho = np.ones(np.shape(F))
    while dP > tol and Nreal < maxcycles:
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

# Get the fermi energy of a 1D contact
def getFermiContact(gSys, ne, ind=0, tol=1e-4, Eminf=-1e6, maxcycles=1000):
    # Set up infinite system from contact
    F = gSys.aList[ind]
    S = gSys.aSList[ind]
    tau = gSys.bList[ind]
    stau = gSys.bSList[ind]
    inds = np.arange(len(F))
    #print(S.shape, S.shape, inds.shape, tau.shape, stau.shape)
    g = surfG(F, S, [inds], [tau], [stau], eps=1e-6)

    # Initial guess and integral setup using two layers
    Forbs = np.block([[F, tau], [tau.conj().T, F]])
    Sorbs = np.block([[S, stau], [stau.T, S]])
    gorbs = surfG(Forbs, Sorbs, [inds], [tau], [stau], eps=1e-6)
    orbs, _ = LA.eig(LA.inv(Sorbs)@Forbs)
    orbs = np.sort(np.real(orbs))
    fermi = (orbs[2*int(ne)-1] + orbs[2*int(ne)])/2
    #print(orbs, fermi)
    Emin, N1, N2 = integralFit(Forbs, Sorbs, gorbs, fermi, Eminf, tol, maxcycles)
    Emax = max(orbs)
    return calcFermi(g, ne, Emin, Emax, fermi, N1, N2, Eminf, tol, maxcycles)

# Calculat the fermi energy of the surfG() object
def calcFermi(g, ne, Emin, Emax, fermiGuess=0, N1=100, N2=50, Eminf=-1e6, tol=1e-4, maxcycles=1000):
    # Fermi Energy search using full contact
    print(f'Eminf DOS = {DOSg(g.F,g.S,g,Eminf)}')
    fermi = fermiGuess
    pLow = densityReal(g.F, g.S, g, Eminf, Emin, N2, T=0)
    nELow = np.trace(pLow@g.S)
    print(f'Electrons below lowest onsite energy: {nELow}')
    #if nELow >= ne:
    #    raise Exception('Calculated Fermi energy is below lowest orbital energy!')
    pMu = lambda E: densityComplex(g.F, g.S, g, Emin, E, N1, T=0)
    
    # Fermi search using bisection method (F not changing, highly stable)
    Ncurr = -1
    counter = 0 
    lBound = Emin
    uBound = Emax
    lBoundVal = ne - nELow
    uBoundVal = -lBoundVal
    while abs(ne - Ncurr) > tol and counter < maxcycles:
        p_ = pLow+pMu(fermi)
        Ncurr = np.trace(p_@g.S)
        dN = (ne-Ncurr)
        if dN > 0 and fermi > lBound:
            lBound = fermi
            lBoundVal = dN
        elif dN < 0 and fermi < uBound:
            uBound = fermi
            uBoundVal = dN
        # Weight bisection based on value at endpoints
        weight = -lBoundVal/(uBoundVal - lBoundVal)
        fermi = lBound + (uBound-lBound)*weight
        print("DN:",dN, "Fermi:", fermi, "Bounds:", lBound, uBound)
    if abs(ne - Ncurr) > tol and counter > maxcycles:
        print(f'Warning: Fermi energy still not within tolerance! Ef = {fermi:.2f} eV, N = {Ncurr:.2f})')
    return fermi, Emin, N1, N2
