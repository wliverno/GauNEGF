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
def quickCurrent(H0, sig1, sig2, fermi, qV, spin="r",dE=0.01):
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    Elist = np.arange(fermi-(qV/2), fermi+(qV/2), dE)
    Tr = cohTrans(Elist, H0, sig1, sig2)
    curr = eoverh * np.trapz(Tr, Elist)
    if spin=="r":
        curr *= 2
    return curr

# Calculate current from SCF mat file
def qCurrentF(fn, dE=0.01):
    matfile = io.loadmat(fn);
    return quickCurrent(matfile["H0"],matfile["sig1"][0],matfile["sig2"][0],
            matfile["fermi"][0,0], matfile["qV"][0,0], matfile["spin"][0], dE=dE)

# H0 is an NxN matrix, sig1 and sig2 are Nx1 vectors
def cohTrans(Elist, H0, sig1, sig2):
    H0 = np.array(H0)
    N = len(H0)
    I = np.identity(N);
    gamma1 = np.imag(sig1)*2
    gamma2 = np.imag(sig2)*2
    Tr = []
    for E in Elist:
        sig = np.diag(sig1 + sig2)
        Gr = LA.inv(E*I - H0 - sig)
        T = 0
        gam1Gr = np.array([gamma1*row for row in Gr])
        gam2Ga = np.array([gamma2*row for row in Gr.conj().T])
        for i in range(N):
            T += np.dot(gam1Gr[i, :],gam2Ga[:, i])
        T = np.real(T)
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr

# H0 is an NxN matrix, sig1 and sig2 are Nx1 vectors
def DOS(Elist, H0, sig1, sig2):
    H0 = np.array(H0)
    I = np.identity(len(H0));
    DOS = []
    DOSList = []
    for E in Elist:
        sig = np.diag(sig1 + sig2)
        Gr = LA.inv(E*I - H0 - sig)
        DOSList.append(-1*np.imag(np.diag(Gr))/np.pi)
        DOS.append(np.sum(DOSList[-1]))
        print("Energy:",E, "eV, DOS=", DOS[-1])
    return DOS, DOSList
                   
# H0 is an NxN matrix, g is a surfGreen() object
def cohTransE(Elist, H0, g):
    H0 = np.array(H0)
    I = np.identity(len(H0));
    Tr = []
    for E in Elist:
        sig1 = g.sigma(E, 0)
        sig2 = g.sigma(E, 1)
        gamma1 = np.imag(sig1)*2
        gamma2 = np.imag(sig2)*2
        Gr = LA.inv(E*I - H0 - sig1 - sig2)
        T = np.real(np.trace(gamma1@Gr@gamma2@Gr.conj().T))
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr
                   
# H0 is an NxN matrix, g is a surfGreen() object
def DOSE(Elist, H0, g):
    H0 = np.array(H0)
    I = np.identity(len(H0));
    DOS = []
    DOSList = []
    for E in Elist:
        sig = g.sigmaTot(E)
        Gr = LA.inv(E*I - H0 - sig)
        DOSList.append(-1*np.imag(np.diag(Gr))/np.pi)
        DOS.append(np.sum(DOSList[-1]))
        print("Energy:",E, "eV, DOS=", DOS[-1])
    return DOS, DOSList
    
