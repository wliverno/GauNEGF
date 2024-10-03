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

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGrid(F, S, g, Emin, mu, ind=None, N=100, T=300):
    kT = kB*T
    Emax = mu + 5*kT
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
        den += mid*w[i]*(GrE@Gamma@GaE)*fermi(E, mu, T)
    print('Integration done!')
    
    # Inverse Lowdin TF
    TF = fractional_matrix_power(S, 0.5)
    den = TF@den@TF
    
    return den/(2*np.pi)

# Get non-equilibrium density at a single contact (ind) using a real energy grid
def densityGridTrap(F, S, g, Emin, mu, ind=None, N=100, T=300):
    kT = kB*T
    Emax = mu + 5*kT
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
        den += (GrE@Gamma@GaE)*fermi(E, mu, T)*dE
    print('Integration done!')
    
    # Inverse Lowdin TF
    TF = fractional_matrix_power(S, 0.5)
    den = TF@den@TF
    
    return den/(2*np.pi)


def Gr(F, S, g, E):
    return LA.inv(E*S - F - g.sigmaTot(E)) 

def fermi(E, mu, T):
    kT = kB*T
    return 1/(np.exp((np.real(E) - mu)/kT)+1)

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
        if T>0:
            fermi = 1/(np.exp((np.real(z)-mu)/kT)+1)
        else:
            fermi = 1
        lineInt += (np.pi/2)*w[i]*Gr(F, S, g, z)*fermi*dz
    print('Integration done!')
    
    # Inverse Lowdin TF
    TF = fractional_matrix_power(S, 0.5)
    lineInt = TF@lineInt@TF

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
        if T>0:
            fermi = 1/(np.exp((np.real(E)-mu)/kT)+1)
        else:
            fermi = 1
        lineInt += Gr(F, S, g, E)*fermi*dS
    print('Integration done!')
    
    # Inverse Lowdin TF
    TF = fractional_matrix_power(S, 0.5)
    lineInt = TF@lineInt@TF

    #Return -Im(Integral)/pi, Equation 19 in 10.1103/PhysRevB.63.245407
    return (1+0j)*np.imag(lineInt)/np.pi

def DOS(F, S, g, E):
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

def integralFit(F, S, g, mu, tol=1e-2, T=300, maxcycles=500):
    kT = kB*T
    Emax = mu + (5*kT)
    D,V = LA.eig(F, S)

    # Calculate Emin using DOS
    Emin = min(D.real.flatten())
    counter = 0
    dN = DOS(F,S,g,Emin)
    while dN>tol and counter<maxcycles:
        Emin -= 0.1
        dN = DOS(F,S,g,Emin)
        print(Emin, dN)
        counter += 1
    if counter == maxcycles:
        print(f'Warning: Emin still not within tolerance after {maxcycles} cycles')
    print(f'Final Emin: {Emin} eV') 
    
    #Determine grid using dN
    counter = 0
    N = 8
    dN = 100
    rho = np.zeros(np.shape(F))
    while dN > tol and counter < maxcycles:
        N *= 2 # Start with 16 points, double each time
        rho_ = np.real(densityComplex(F, S, g, Emin, mu, N, T))
        dN = np.trace(np.abs(rho_ - rho))
        print(N, dN)
        rho = rho_
        counter += 1
    if counter == maxcycles:
        print(f'Warning: dE still not within tolerance after {maxcycles} cycles')
    print(f'Final N: {N}') 
    return Emin, N
