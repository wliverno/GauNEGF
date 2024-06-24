import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import scipy.io as io
import sys
import time
import matplotlib.pyplot as plt
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu


# Matrix Headers
AlphaDen = "ALPHA DENSITY MATRIX"
BetaDen = "BETA DENSITY MATRIX"
AlphaSCFDen = "ALPHA SCF DENSITY MATRIX"
BetaSCFDen = "BETA SCF DENSITY MATRIX"
AlphaFock = "ALPHA FOCK MATRIX"
BetaFock = "BETA FOCK MATRIX"
AlphaMOs = "ALPHA MO COEFFICIENTS"
BetaMOs = "BETA MO COEFFICIENTS"
AlphaEnergies = "ALPHA ORBITAL ENERGIES"
BetaEnergies = "BETA ORBITAL ENERGIES"

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kT = 0.025              # eV @ 20degC
V_to_au = 0.03675       # Volts to Hartree/elementary Charge
kB = 8.617e-5           # eV/Kelvin

#### HELPER FUNCTIONS ####

# Get density using analytical integration
def density(V, D, Gam, Emin, mu):
    Nd = len(V)
    DD = np.array([D for i in range(Nd)]).T
    logmat = np.array([np.emath.log(1-(mu/D)) for i in range(Nd)]).T
    logmat2 = np.array([np.emath.log(1-(Emin/D)) for i in range(Nd)]).T


    invmat = 1/(2*np.pi*(DD-DD.conj().T))
    pref2 = logmat - logmat.conj().T
    pref3 = logmat2-logmat2.conj().T

    prefactor = np.multiply(invmat,(pref2-pref3))


    Vc = LA.inv(V.conj().T)

    Gammam = Vc.conj().T@Gam@Vc

    prefactor = np.multiply(prefactor,Gammam)

    den = V@ prefactor @ V.conj().T

    return den

def denFunc(D, V, Gambar, E):
    V = np.array(V, dtype=complex)
    Ga = np.array(np.diag(1/(E-np.conj(D))))
    Ga = V*Ga*V.conj().T
    Gr = np.array(np.diag(1/(E-D)))
    Gr = V*Gr*V.conj().T
    return Gr*Gambar*Ga

def densityGrid(Fbar, Gambar, Emin, Emax, dE=0.001, T=300):
    kT = kB*T
    Egrid = np.arange(Emin, Emax+(5*kT), dE)
    I = np.array(np.identity(len(Fbar)))
    den = np.array(np.zeros(np.shape(Fbar)), dtype=complex)
    D, V = LA.eig(Fbar)
    print('Starting Integration...')
    for i in range(1,len(Egrid)):
        E = (Egrid[i]+Egrid[i-1])/2
        dE = Egrid[i]-Egrid[i-1]
        fermiFunc = 1/(np.exp((E-Emax)/kT)+1)
        den += denFunc(D, V, Gambar, E)*fermiFunc*dE
    print('Integration done!')
    return den/(2*np.pi)

def densityComplex(Fbar, Gambar, Emin, Emax, dE=0.001):
    #Construct circular contour
    center = (Emin+Emax)/2
    r = (Emax-Emin)/2
    N = int((Emax-Emin)/dE)
    theta = np.linspace(0, np.pi, N)
    Egrid = r*np.exp(1j*theta)+center
    
    #Calculate Residues
    Res = np.array(np.zeros(np.shape(Fbar)), dtype=complex)
    I = np.array(np.identity(len(Fbar)))
    D, V = LA.eig(Fbar)
    V = np.array(V, dtype=complex)
    for ind, E in enumerate(D):
        if abs(E-center) < r:
            #print(E)
            Ga = np.array(np.diag(1/(E-np.conj(D))))
            Ga = V* Ga * V.conj().T 
            Y = V[:, ind] * V.conj().T[ind,:]
            Res += 2j*np.pi*np.conj(Y*Gambar*Ga) #WHY CONJUGATE???
            #Res += 2j*np.pi*denFunc(D, V, Gambar, E+1e-9)*(-1e-9)
    
    #Integrate along contour
    print('Starting Integration...')
    lineInt = np.array(np.zeros(np.shape(Fbar)), dtype=complex)
    for i in range(1,N):
        E = (Egrid[i]+Egrid[i-1])/2
        dS = Egrid[i]-Egrid[i-1]
        lineInt += denFunc(D, V, Gambar, E)*dS
    print('Integration done!')
    
    #Use Residue Theorem
    return (Res-lineInt)/(2*np.pi)

# Form Sigma matrix given self-energy matrix or value (V) and orbital indices (inds)
def formSigma(inds, V, nsto, S=0):
    if isinstance(S, int):  #if overlap is not given
       S = np.eye(nsto)
    sigma = np.array(-1j*1e-9*S,dtype=complex)
    if isinstance(V, (int,complex,float)):  #if sigma is a single value
        for i in inds:
            sigma[i,i] = V
    else:                                     #if sigma is a matrix
        sigma[np.ix_(inds, inds)] = V

    return sigma

# Form Sigma matrix given contact Green's function (g), coupling (tau), electron energy (E) and orbital indices (inds)
def formSigmaE(inds, g, tau, E, stau, nsto):
    t = E*stau - tau
    sig = t*g*t.conj().T
    sigma = formSigma(inds, sig, nsto)
    return sigma

# Build density matrix based on spin type
def getDen(bar, spin):
    # Set up Fock matrix and atom indexing
    # Note: positive ind1ices are alpha/paired orbitals, negative are beta orbitals
    if spin == "r" or spin == "g":
        P = np.array(bar.matlist[AlphaSCFDen].expand())
    elif spin == "ro" or spin == "u":
        PA = np.array(bar.matlist[AlphaSCFDen].expand())
        PB = np.array(bar.matlist[BetaSCFDen].expand())
        P = np.block([[PA, np.zeros(PA.shape)], [np.zeros(PB.shape), PB]])
    else:
        raise ValueError("Spin treatment not recognized!")
    return P

# Build Fock matrix based on spin type, return orbital indices (alpha and beta are +/-)
def getFock(bar, spin):
    # Set up Fock matrix and atom indexing
    # Note: positive ind1ices are alpha/paired orbitals, negative are beta orbitals
    if spin == "r":
        locs = bar.ibfatm
        Fock = np.array(bar.matlist[AlphaFock].expand())
    elif spin == "ro" or spin == "u":
        locs = np.concatenate((bar.ibfatm, bar.ibfatm*-1))
        AFock = np.array(bar.matlist[AlphaFock].expand())
        BFock = np.array(bar.matlist[BetaFock].expand())
        Fock = np.block([[AFock, np.zeros(AFock.shape)], [np.zeros(BFock.shape), BFock]])
    elif spin == "g":
        locs = [loc for pair in zip(bar.ibfatm, bar.ibfatm*-1) for loc in pair]
        Fock = np.array(bar.matlist[AlphaFock].expand())
    else:
        raise ValueError("Spin treatment not recognized!")
    locs = np.array(locs)
    return Fock,locs

# Return energies for each electron in eV
def getEnergies(bar, spin):
    if spin =="r":
        Alevels = np.sort(bar.matlist[AlphaEnergies].expand())
        levels = [level for pair in zip(Alevels, Alevels) for level in pair]
    elif spin=="ro" or spin == "u":
        Alevels = np.sort(bar.matlist[AlphaEnergies].expand())
        Blevels = np.sort(bar.matlist[BetaEnergies].expand())
        levels = [level for pair in zip(Alevels, Blevels) for level in pair]
    elif spin=="g":
        levels = np.sort(bar.matlist[AlphaEnergies].expand())
    else:
        raise ValueError("Spin treatment not recognized!")
    return np.sort(levels)*har_to_eV

# Store density matrix to use in Gaussian
def storeDen(bar, P, spin):
    nsto = len(bar.ibfatm)
    if spin=="r":
        P = np.real(np.array(P))
        PaO = qco.OpMat(AlphaSCFDen,P/2,dimens=(nsto,nsto))
        PaO.compress()
        bar.addobj(PaO)
    elif spin=="ro" or spin=="u":
        P = np.real(np.array(P))
        Pa = P[0:nsto, 0:nsto]
        Pb = P[nsto:, nsto:]
        PaO = qco.OpMat(AlphaSCFDen,Pa,dimens=(nsto,nsto))
        PbO = qco.OpMat(BetaSCFDen,Pb,dimens=(nsto,nsto))
        PaO.compress()
        PbO.compress()
        bar.addobj(PaO)
        bar.addobj(PbO)
    elif spin=="g":
        P = np.complex128(np.array(P))
        PaO = qco.OpMat(AlphaSCFDen,P,dimens=(nsto*2,nsto*2), typed='c')
        PaO.compress()
        bar.addobj(PaO)
    else:
        raise ValueError("Spin treatment not recognized!")


