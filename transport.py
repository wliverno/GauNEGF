import numpy as np
from numpy import linalg as LA
import scipy.io as io
from matTools import *

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kT = 0.025              # eV @ 20degC
V_to_au = 0.03675       # Volts to Hartree/elementary Charge

# Calculate Coherent Current at T=0K using NEGF
def quickCurrent(F, S, sig1, sig2, fermi, qV, spin="r",dE=0.01):
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    Elist = np.arange(fermi-(qV/2), fermi+(qV/2), dE)
    Tr = cohTrans(Elist, F, S, sig1, sig2)
    curr = eoverh * np.trapz(Tr, Elist)
    if spin=="r":
        curr *= 2
    return curr

# Calculate current from SCF mat file
def qCurrentF(fn, dE=0.01):
    matfile = io.loadmat(fn)
    return quickCurrent(matfile["F"], matfile["S"], matfile["sig1"][0],matfile["sig2"][0],
            matfile["fermi"][0,0], matfile["qV"][0,0], matfile["spin"][0], dE=dE)

# H0 is an NxN matrix, sig1 and sig2 are Nx1 vectors or NxN Matrices
def cohTrans(Elist, F, S, sig1, sig2):
    F = np.array(F)
    N = len(F)
    S = np.array(S)
    gamma1 = -1j*(sig1 - sig1.conj().T)
    gamma2 = -1j*(sig2 - sig2.conj().T)
    Tr = []
    for E in Elist:
        T = 0
        if sig1.shape==F.shape and sig2.shape==F.shape:
            Gr = LA.inv(E*S - F - sig1 - sig2)
            gam1Gr = gamma1@Gr
            gam2Ga = gamma2@Gr.conj().T
        else:
            sig = np.diag(sig1 + sig2)
            Gr = LA.inv(E*S - F - sig)
            gam1Gr = np.array([gamma1*row for row in Gr])
            gam2Ga = np.array([gamma2*row for row in Gr.conj().T])
        for i in range(N):
            T += np.dot(gam1Gr[i, :],gam2Ga[:, i])
        T = np.real(T)
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr

# H0 is an NxN matrix, sig1 and sig2 are Nx1 vectors
def DOS(Elist, F, S, sig1, sig2):
    F = np.array(F)
    S = np.array(S)
    DOS = []
    DOSList = []
    for E in Elist:
        if sig1.shape==F.shape and sig2.shape==F.shape:
            sig = sig1+sig2
        else:
            sig = np.diag(sig1 + sig2)
        Gr = LA.inv(E*S - F - sig)
        DOSList.append(-1*np.imag(np.diag(Gr))/np.pi)
        DOS.append(np.sum(DOSList[-1]))
        print("Energy:",E, "eV, DOS=", DOS[-1])
    return DOS, DOSList
                   
# H0 is an NxN matrix, g is a surfGreen() object
def cohTransE(Elist, F, S, g):
    F = np.array(F)
    S = np.array(S)
    Tr = []
    for E in Elist:
        sig1 = g.sigma(E, 0)
        sig2 = g.sigma(E, 1)
        gamma1 = -1j*(sig1 - sig1.conj().T)
        gamma2 = -1j*(sig2 - sig2.conj().T)
        Gr = LA.inv(E*S - F - sig1 - sig2)
        T = np.real(np.trace(gamma1@Gr@gamma2@Gr.conj().T))
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr
                   
# H0 is an NxN matrix, g is a surfGreen() object
def DOSE(Elist, F, S, g):
    F = np.array(F)
    S = np.array(S)
    DOS = []
    DOSList = []
    for E in Elist:
        sig = g.sigmaTot(E)
        Gr = LA.inv(E*S - F - sig)
        DOSList.append(-1*np.imag(np.diag(Gr))/np.pi)
        DOS.append(np.sum(DOSList[-1]))
        print("Energy:",E, "eV, DOS=", DOS[-1])
    return DOS, DOSList
    
