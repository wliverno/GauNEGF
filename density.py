import numpy as np
from scipy import linalg as LA
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


# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kT = 0.025              # eV @ 20degC
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
        return 1/(np.exp((np.real(E) - mu)/kT)+1)

def DOSg(F, S, g, E):
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

## DENSITY FUNCTIONS
# Get density using analytical integration
def density(V, D, Gam, Emin, mu):
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
    Vc = LA.inv(V.conj().T)
    Gammam = Vc.conj().T@Gam@Vc
    prefactor = np.multiply(prefactor,Gammam)
    
    #Convert back to input basis, return
    den = V@ prefactor @ V.conj().T
    return den

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
    kT = kB*T
    Emax = mu+(5*kT)
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
    print('Integration done!')
    
    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi


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

def integralFit(F, S, g, mu, Eminf, tol=1e-6, maxcycles=1000):

    # Calculate Emin using DOS
    D = LA.eigh(F, S, eigvals_only=True)
    Emin = min(D.real.flatten())
    counter = 0
    dP = DOSg(F,S,g,Emin)
    while dP>tol and counter<maxcycles:
        Emin -= 0.1
        dP = DOSg(F,S,g,Emin)
        #print(Emin, dP)
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
        rho_ = np.real(densityComplex(F, S, g, Emin, mu, Ncomplex, T=0))
        dP = max(np.abs(rho_ - rho).flatten())
        #print(Ncomplex, dP)
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
        dP = max(np.abs(rho_ - rho).flatten())
        #print(Nreal, dP)
        rho = rho_
        counter += 1
    if Nreal >  maxcycles and dP > tol:
        print(f'Warning: Nreal still not within tolerance (final value = {dP})')
    print(f'Final Nreal: {Nreal}') 

    return Emin, Ncomplex, Nreal
