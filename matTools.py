# Python Packages
import numpy as np

# Gaussian interface packages
from gauopen import QCBinAr as qcb
from gauopen import QCOpMat as qco


# Matrix Headers
AlphaDen = "ALPHA DENSITY MATRIX"
BetaDen = "BETA DENSITY MATRIX"
AlphaSCFDen = "ALPHA SCF DENSITY MATRIX"
BetaSCFDen = "BETA SCF DENSITY MATRIX"
AlphaFock = "ALPHA FOCK MATRIX"
BetaFock = "BETA FOCK MATRIX"

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree

#### HELPER FUNCTIONS ####
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


