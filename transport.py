"""
Transport calculations for quantum systems using Non-Equilibrium Green's Functions.

This module provides functions for calculating quantum transport properties:
- Coherent transmission through molecular junctions
- Spin-dependent transport calculations
- Current calculations at finite bias
- Density of states calculations

The module supports both energy-independent and energy-dependent self-energies,
with implementations for both spin-restricted and spin-unrestricted calculations.
Spin-dependent transport follows the formalism described in [1].

References
----------
.. [1] Herrmann, C., Solomon, G. C., & Ratner, M. A. J. Chem. Theory Comput. 6, 3078 (2010)
      DOI: 10.1021/acs.jctc.9b01078
"""

import numpy as np
from numpy import linalg as LA
import scipy.io as io
from matTools import *

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kB = 8.617e-5           # eV/Kelvin
V_to_au = 0.03675       # Volts to Hartree/elementary Charge

## CURRENT FUNCTIONS
def current(F, S, sig1, sig2, fermi, qV, T=0, spin="r",dE=0.01):
    """
    Calculate coherent current using NEGF with energy-independent self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sig1 : ndarray
        Left contact self-energy (vector or matrix)
    sig2 : ndarray
        Right contact self-energy (vector or matrix)
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float
        Temperature in Kelvin (default: 0)
    spin : str, optional
        Spin configuration ('r' for restricted) (default: 'r')
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    float
        Current in Amperes
    """
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    muL = fermi - qV/2
    muR = fermi + qV/2
    if T== 0:
        Elist = np.arange(muL, muR, dE)
        Tr = cohTrans(Elist, F, S, sig1, sig2)
        curr = eoverh * np.trapz(Tr, Elist)
    else:
        kT = kB*T
        spread = np.sign(dE)*5*kT
        Elist = np.arange(muL-spread, muR+spread, dE)
        Tr = cohTrans(Elist, F, S, sig1, sig2)
        dfermi = np.abs(1/(np.exp((Elist - muR)/kT)+1) -  1/(np.exp((Elist-muL)/kT)+1))
        curr = eoverh * np.trapz(Tr*dfermi, Elist)
    if spin=="r":
        curr *= 2
    return curr

def currentSpin(F, S, sig1, sig2, fermi, qV, T=0, spin="r",dE=0.01):
    """
    Calculate coherent spin current using NEGF with energy-independent self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sig1 : ndarray
        Left contact self-energy (vector or matrix)
    sig2 : ndarray
        Right contact self-energy (vector or matrix)
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float
        Temperature in Kelvin (default: 0)
    spin : str, optional
        Spin configuration ('r' for restricted) (default: 'r')
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    list
        Spin-currents (in Amperes) [I↑↑, I↑↓, I↓↑, I↓↓]
    """
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    muL = fermi - qV/2
    muR = fermi + qV/2
    if T== 0:
        Elist = np.arange(muL, muR, dE)
        _, Tspin = cohTransSpin(Elist, F, S, sig1, sig2, spin)
        curr = [eoverh * np.trapz(Tspin[:, i], Elist) for i in range(4)]
    else:
        kT = kB*T
        spread = np.sign(dE)*5*kT
        Elist = np.arange(muL-spread, muR+spread, dE)
        _, Tspin = cohTransSpin(Elist, F, S, sig1, sig2, spin)
        dfermi = np.abs(1/(np.exp((Elist - muR)/kT)+1) -  1/(np.exp((Elist-muL)/kT)+1))
        curr = [eoverh * np.trapz(Tspin[:, i]*dfermi, Elist) for i in range(4)]
    return curr


def currentE(F, S, g, fermi, qV, T=0, spin="r",dE=0.01):
    """
    Calculate coherent current at T=0K using NEGF with energy-dependent self-energies.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    fermi : float
        Fermi energy in eV
    qV : float
        Applied bias voltage in eV
    T : float
        Temperature in Kelvin (default: 0)
    spin : str, optional
        Spin configuration ('r' for restricted) (default: 'r')
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    float
        Current in Amperes
    """
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    muL = fermi - qV/2
    muR = fermi + qV/2
    if T== 0:
        Elist = np.arange(muL, muR, dE)
        Tr = cohTransE(Elist, F, S, g)
        curr = eoverh * np.trapz(Tr, Elist)
    else:
        kT = kB*T
        spread = np.sign(dE)*5*kT
        Elist = np.arange(muL-spread, muR+spread, dE)
        Tr = cohTransE(Elist, F, S, g)
        dfermi = np.abs(1/(np.exp((Elist - muR)/kT)+1) -  1/(np.exp((Elist-muL)/kT)+1))
        curr = eoverh * np.trapz(Tr*dfermi, Elist)
    if spin=="r":
        curr *= 2
    return curr

def currentF(fn, dE=0.01, T=0):
    """
    Calculate current from saved SCF matrix file.

    Parameters
    ----------
    fn : str
        Filename of .mat file containing SCF data
    dE : float, optional
        Energy step for integration in eV (default: 0.01)

    Returns
    -------
    float
        Current in Amperes

    Notes
    -----
    The .mat file should contain:
    - F: Fock matrix
    - S: Overlap matrix
    - sig1, sig2: Contact self-energies
    - fermi: Fermi energy
    - qV: Applied voltage
    - spin: Spin configuration
    """
    matfile = io.loadmat(fn)
    return current(matfile["F"], matfile["S"], matfile["sig1"],matfile["sig2"],
            matfile["fermi"][0,0], matfile["qV"][0,0], T, matfile["spin"][0], dE=dE)

## ENERGY INDEPENDENT SIGMA
def cohTrans(Elist, F, S, sig1, sig2):
    """
    Calculate coherent transmission with energy-independent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sig1 : ndarray
        Left contact self-energy (vector or matrix)
    sig2 : ndarray
        Right contact self-energy (vector or matrix)

    Returns
    -------
    list
        Transmission values at each energy

    Notes
    -----
    Supports both vector and matrix self-energies. For vector self-energies,
    diagonal matrices are constructed internally.
    """
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

def cohTransSpin(Elist, F, S, sig1, sig2, spin='u'):
    """
    Calculate spin-dependent coherent transmission with energy-independent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix (2N x 2N for spin-unrestricted)
    S : ndarray
        Overlap matrix (2N x 2N for spin-unrestricted)
    sig1 : ndarray
        Left contact self-energy (spin independent vector (1xN) or matrix (NxN), 
                                            spin dependent matrix (2Nx2N))
    sig2 : ndarray
        Right contact self-energy (spin independent vector (1xN) or matrix (NxN), 
                                            spin dependent matrix (2Nx2N))
    spin : str, optional
        Spin basis {'r', 'u', 'ro', or 'g'} (default: 'u')

    Returns
    -------
    tuple
        (Tr, Tspin) where:
        - Tr: Total transmission at each energy
        - Tspin: Array of spin-resolved transmissions [T↑↑, T↑↓, T↓↑, T↓↓]

    Notes
    -----
    For collinear spin calculations ('u' or 'ro'), the matrices are arranged in blocks:
    [F↑↑  0 ]  [S↑↑  0 ]
    [0   F↓↓], [0   S↓↓]
    For generalized spin basis ('g'), each orbital contains a 2x2 spinor block:
    [F↑↑  F↑↓]  [S↑↑  S↑↓]
    [F↓↑  F↓↓], [S↓↑  S↓↓]
    which are then combined into a 2Nx2N matrix.
    """
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
                T_ = sum(gam1Gr[j][i,:]*gam2Ga[j][:,i])
                Ts[j] += T_
                T+= T_
        T = np.real(T)
        Ts = np.real(Ts)
        print("Energy:",E, "eV, Transmission=", T, ", Tspin=", Ts)
        Tr.append(T)
        Tspin.append(Ts)
    return (Tr, np.array(Tspin))

# H0 is an NxN matrix, sig1 and sig2 are Nx1 vectors
def DOS(Elist, F, S, sig1, sig2):
    """
    Calculate density of states with energy-independent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate DOS at
    F : ndarray
        Fock matrix, NxN
    S : ndarray
        Overlap matrix, NxN
    sig1 : ndarray
        Left contact self-energy, vector (1xN) or matrix (NxN)
    sig2 : ndarray
        Right contact self-energy, vector (1xN) or matrix (NxN)

    Returns
    -------
    tuple
        (DOS, DOSList) where:
        - DOS: Total density of states at each energy
        - DOSList: Site-resolved DOS at each energy
    """
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
    return DOS, DOSList

## ENERGY DEPENDENT SIGMA:

def cohTransE(Elist, F, S, g):
    """
    Calculate coherent transmission with energy-dependent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator

    Returns
    -------
    list
        Transmission values at each energy

    Notes
    -----
    Uses the surface Green's function calculator to compute energy-dependent
    self-energies at each energy point.
    """
    F = np.array(F)
    S = np.array(S)
    Tr = []
    for E in Elist:
        sig1 = g.sigma(E, 0)
        sig2 = g.sigma(E, -1)
        gamma1 = -1j*(sig1 - np.conj(sig1).T)
        gamma2 = -1j*(sig2 - np.conj(sig2).T)
        Gr = LA.inv(E*S - F - sig1 - sig2)
        T = np.real(np.trace(gamma1@Gr@gamma2@Gr.conj().T))
        print("Energy:",E, "eV, Transmission=", T)
        Tr.append(T)
    return Tr

def cohTransSpinE(Elist, F, S, g, spin='u'):
    """
    Calculate spin-dependent coherent transmission with energy-dependent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate transmission at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    spin : str, optional
        Spin basis {'r', 'u', 'ro', or 'g'} (default: 'u')

    Returns
    -------
    tuple
        (Tr, Tspin) where:
        - Tr: Total transmission at each energy
        - Tspin: Array of spin-resolved transmissions [T↑↑, T↑↓, T↓↑, T↓↓]
    """
    F = np.array(F)
    N = int(len(F)/2)
    S = np.array(S)
    Tr = np.zeros(len(Elist))
    Tspin = np.zeros((len(Elist),4))
    for ind, E in enumerate(Elist):
        T = 0
        Ts = np.zeros(4)
        sig1 = g.sigma(E, 0)
        sig2 = g.sigma(E, -1)
        gamma1 = -1j*(sig1 - np.conj(sig1).T)
        gamma2 = -1j*(sig2 - np.conj(sig2).T)
        sigMat = sig1+sig2
        Gr = LA.inv(E*S - F - sigMat)
        if spin == 'g':
            aInds = np.arange(0, 2*N, 2)
            bInds = np.arange(1, 2*N, 2)
            GrQuad = [Gr[np.ix_(aInds, aInds)], Gr[np.ix_(aInds, bInds)],
                      Gr[np.ix_(bInds, aInds)], Gr[np.ix_(bInds, bInds)]]
            # Use only main diagonal gamma, ordering from JCTCpaper
            g1Quad = [gamma1[np.ix_(aInds, aInds)], gamma1[np.ix_(aInds, aInds)],
                      gamma1[np.ix_(bInds, bInds)], gamma1[np.ix_(bInds, bInds)]]
            g2Quad = [gamma2[np.ix_(aInds, aInds)], gamma2[np.ix_(bInds, bInds)],
                      gamma2[np.ix_(aInds, aInds)], gamma2[np.ix_(bInds, bInds)]]
        else:
            GrQuad = [Gr[:N, :N], Gr[:N, N:], Gr[N:, :N], Gr[N:, N:]]
            # Use only main diagonal gamma, ordering from JCTCpaper
            g1Quad = [gamma1[:N,:N], gamma1[:N,:N], gamma1[N:, N:], gamma1[N:,N:]]
            g2Quad = [gamma2[:N,:N], gamma2[N:,N:], gamma2[:N, :N], gamma2[N:,N:]]
        gam1Gr = [g1Quad[i]@GrQuad[i] for i in range(4)]     #Full Matrix Multiplication
        gam2Ga = [g2Quad[i]@GrQuad[i].conj().T for i in range(4)]
        for i in range(N):
            Ttot =  0
            for j in range(4):
                T_ = sum(gam1Gr[j][i,:]*gam2Ga[j][:,i])
                Ts[j] += T_
                T+= T_
        T = np.real(T)
        Ts = np.real(Ts)
        print("Energy:",E, "eV, Transmission=", T, ", Tspin=", Ts)
        Tr[ind] = T
        Tspin[ind, :] = Ts
    return Tr, Tspin

                   
def DOSE(Elist, F, S, g):
    """
    Calculate density of states with energy-dependent self-energies.

    Parameters
    ----------
    Elist : array_like
        List of energies in eV to calculate DOS at
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator

    Returns
    -------
    tuple
        (DOS, DOSList) where:
        - DOS: Total density of states at each energy
        - DOSList: Site-resolved DOS at each energy
    """
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
    
