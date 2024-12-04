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
    return quickCurrent(matfile["F"], matfile["S"], matfile["sig1"],matfile["sig2"],
            matfile["fermi"][0,0], matfile["qV"][0,0], matfile["spin"][0], dE=dE)

# F,S are NxN matrices, sig1 and sig2 are Nx1 vectors or NxN Matrices
def cohTrans(Elist, F, S, sig1, sig2):
    F = np.array(F)
    N = len(F)
    S = np.array(S)
    gamma1 = -1j*(sig1 - np.conj(sig1).T)
    gamma2 = -1j*(sig2 - np.conj(sig2).T)
    Tr = []
    for E in Elist:
        T = 0
        if sig1.shape==F.shape and sig2.shape==F.shape:
            Gr = LA.inv(E*S - F - sig1 - sig2)
            gam1Gr = gamma1@Gr
            gam2Ga = gamma2@Gr.conj().T
        elif sig1.shape==(N,) and sig2.shape==(N,):
            sig = np.diag(sig1 + sig2)
            Gr = LA.inv(E*S - F - sig)
            gam1Gr = np.array([gamma1*row for row in Gr])
            gam2Ga = np.array([gamma2*row for row in Gr.conj().T])
        else:
            raise Exception('Sigma size mismatch!')
        for i in range(N):
            T += np.dot(gam1Gr[i, :],gam2Ga[:, i])
        T = np.real(T)
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr

# F,S are NxN matrices, sig1 and sig2 are Nx1 vectors or NxN Matrices
def cohTransSpin(Elist, F, S, sig1, sig2, spin='u'):
    F = np.array(F)
    N = int(len(F)/2)
    S = np.array(S)
    gamma1 = -1j*(sig1 - np.conj(sig1).T)
    gamma2 = -1j*(sig2 - np.conj(sig2).T)
    Tr = []
    Tspin = []
    for E in Elist:
        T = 0
        Ts = np.zeros(4)
        # Case: sigmas are NxN Matrices
        if sig1.shape == (N, N) and sig2.shape == (N, N):
            if spin == 'g':
                sigMat = np.kron(sig1+sig2, np.eye(2))
                Gr = LA.inv(E*S - F - sigMat)
                aInds = np.arange(0, 2*N, 2)
                bInds = np.arange(1, 2*N, 2)
                GrQuad = [Gr[np.ix_(aInds, aInds)], Gr[np.ix_(aInds, bInds)],
                          Gr[np.ix_(bInds, aInds)], Gr[np.ix_(bInds, bInds)]]
            else:
                sigMat = np.kron(np.eye(2), sig1+sig2)
                Gr = LA.inv(E*S - F - sigMat)
                GrQuad = [Gr[:N, :N], Gr[:N, N:], Gr[N:, :N], Gr[N:, N:]]
            gam1Gr = [gamma1@Gr for Gr in GrQuad]     #Full Matrix Multiplication
            gam2Ga = [gamma2@Gr.conj().T for Gr in GrQuad]
        #Case sigmas are Nx1 vectors
        elif sig1.shape == (N, ) and sig2.shape == (N, ):
            if spin == 'g':
                sigMat = np.kron(np.diag(sig1+sig2), np.eye(2))
                Gr = LA.inv(E*S - F - sigMat)
                aInds = np.arange(0, 2*N, 2)
                bInds = np.arange(1, 2*N, 2)
                GrQuad = [Gr[np.ix_(aInds, aInds)], Gr[np.ix_(aInds, bInds)],
                          Gr[np.ix_(bInds, aInds)], Gr[np.ix_(bInds, bInds)]]
            else:
                sigMat = np.kron(np.eye(2), np.diag(sig1+sig2))
                Gr = LA.inv(E*S - F - sigMat)
                GrQuad = [Gr[:N, :N], Gr[:N, N:], Gr[N:, :N], Gr[N:, N:]]
            gam1Gr = [np.array([gamma1*row for row in Gr]) for Gr in GrQuad] #Faster algorithm
            gam2Ga = [np.array([gamma2*row for row in Gr.conj().T]) for Gr in GrQuad]
        else:
            raise Exception('Sigma size mismatch!')
        for i in range(N):
            for j in range(4):
                #T_ = np.dot(gam1Gr[j][i,:],gam2Ga[j][:,i])
                T_ = sum(gam1Gr[j][i,:]*gam2Ga[j][:,i])
                #T_ = np.einsum('i,i->',gam1Gr[j][i,:],gam2Ga[j][:,i])
                Ts[j] += T_
                #print(i, j, Ts)
                T+= T_
        T = np.real(T)
        Ts = np.real(Ts)
        print("Energy:",E, "eV, Transmission=", T, ", Tspin=", Ts)
        Tr.append(T)
        Tspin.append(Ts)
    return (Tr, np.array(Tspin))

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
        #print("Energy:",E, "eV, DOS=", DOS[-1])
    return DOS, DOSList

## ENERGY DEPENDENT SIGMA:

# F,S are NxN matrices, g is a surfG() object
def cohTransE(Elist, F, S, g):
    F = np.array(F)
    S = np.array(S)
    Tr = []
    for E in Elist:
        sig1 = g.sigma(E, 0)
        sig2 = g.sigma(E, 1)
        gamma1 = -1j*(sig1 - np.conj(sig1).T)
        gamma2 = -1j*(sig2 - np.conj(sig2).T)
        Gr = LA.inv(E*S - F - sig1 - sig2)
        T = np.real(np.trace(gamma1@Gr@gamma2@Gr.conj().T))
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr

# F,S are 2Nx2N matrices, g is a surfG() object size 2Nx2N
def cohTransSpinE(Elist, F, S, g, spin='u'):
    F = np.array(F)
    N = int(len(F)/2)
    S = np.array(S)
    Tr = np.zeros(len(Elist))
    Tspin = np.zeros((len(Elist),4))
    for ind, E in enumerate(Elist):
        T = 0
        Ts = np.zeros(4)
        sig1 = g.sigma(E, 0)
        sig2 = g.sigma(E, 1)
        gamma1 = -1j*(sig1 - np.conj(sig1).T)
        gamma2 = -1j*(sig2 - np.conj(sig2).T)
        sigMat = sig1+sig2
        Gr = LA.inv(E*S - F - sigMat)
        if spin == 'g':
            aInds = np.arange(0, 2*N, 2)
            bInds = np.arange(1, 2*N, 2)
            GrQuad = [Gr[np.ix_(aInds, aInds)], Gr[np.ix_(aInds, bInds)],
                      Gr[np.ix_(bInds, aInds)], Gr[np.ix_(bInds, bInds)]]
            # Use only main diagonal gamma, ordering from 10.1021/acs.jctc.9b01078
            g1Quad = [gamma1[np.ix_(aInds, aInds)], gamma1[np.ix_(aInds, aInds)],
                      gamma1[np.ix_(bInds, bInds)], gamma1[np.ix_(bInds, bInds)]]
            g2Quad = [gamma2[np.ix_(aInds, aInds)], gamma2[np.ix_(bInds, bInds)],
                      gamma2[np.ix_(aInds, aInds)], gamma2[np.ix_(bInds, bInds)]]
        else:
            GrQuad = [Gr[:N, :N], Gr[:N, N:], Gr[N:, :N], Gr[N:, N:]]
            # Use only main diagonal gamma, ordering from 10.1021/acs.jctc.9b01078
            g1Quad = [gamma1[:N,:N], gamma1[:N,:N], gamma1[N:, N:], gamma1[N:,N:]]
            g2Quad = [gamma2[:N,:N], gamma2[N:,N:], gamma2[:N, :N], gamma2[N:,N:]]
        gam1Gr = [g1Quad[i]@GrQuad[i] for i in range(4)]     #Full Matrix Multiplication
        gam2Ga = [g2Quad[i]@GrQuad[i].conj().T for i in range(4)]
        for i in range(N):
            Ttot =  0
            for j in range(4):
                #T_ = np.dot(gam1Gr[j][i,:],gam2Ga[j][:,i])
                T_ = sum(gam1Gr[j][i,:]*gam2Ga[j][:,i])
                #T_ = np.einsum('i,i->',gam1Gr[j][i,:],gam2Ga[j][:,i])
                Ts[j] += T_
                T+= T_
        T = np.real(T)
        Ts = np.real(Ts)
        print("Energy:",E, "eV, Transmission=", T, ", Tspin=", Ts)
        Tr[ind] = T
        Tspin[ind, :] = Ts
    return Tr, Tspin

                   
# F,S are NxN matrices, g is a surfG() object
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
    
