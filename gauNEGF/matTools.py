"""
Matrix manipulation utilities for quantum transport calculations.

This module provides helper functions for handling matrices in quantum transport
calculations, with a focus on:
- Self-energy matrix construction
- Density and Fock matrix manipulation
- Spin treatment (restricted, unrestricted, and generalized)
- Integration with Gaussian's matrix formats

The functions handle three types of spin treatments:
- 'r': Restricted (same orbitals for alpha and beta electrons)
- 'ro'/'u': Unrestricted (separate alpha and beta orbitals)
- 'g': Generalized (non-collinear spin treatment)
"""

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
AlphaEnergies = "ALPHA ORBITAL ENERGIES"
BetaEnergies = "BETA ORBITAL ENERGIES"

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree

#### HELPER FUNCTIONS ####
def formSigma(inds, V, nsto, S=0):
    """
    Form a self-energy matrix for specified orbitals.

    Creates a self-energy matrix of size nsto x nsto with values V
    at the specified orbital indices. Can handle both scalar and
    matrix-valued self-energies.

    Parameters
    ----------
    inds : list of int
        Orbital indices where self-energy should be applied
    V : complex or ndarray
        Self-energy value(s) to insert
        - If scalar: Same value used for all specified orbitals
        - If matrix: Must match size of indices
    nsto : int
        Total number of orbitals (size of resulting matrix)
    S : ndarray or int, optional
        Overlap matrix for broadening term. If 0, identity used (default: 0)

    Returns
    -------
    ndarray
        Complex self-energy matrix of size nsto x nsto
    """
    if isinstance(S, int):  #if overlap is not given
       S = np.eye(nsto)
    sigma = np.array(-1j*1e-9*S,dtype=complex)
    if isinstance(V, (int,complex,float)):  #if sigma is a single value
        for i in inds:
            sigma[i,i] = V
    else:                                     #if sigma is a matrix
        sigma[np.ix_(inds, inds)] = V

    return sigma

# Build density matrix based on spin type
def getDen(bar, spin):
    """
    Build density matrix from Gaussian checkpoint file.

    Extracts the density matrix from a Gaussian checkpoint file based on
    the specified spin treatment. Handles restricted, unrestricted, and
    generalized cases.

    Parameters
    ----------
    bar : QCBinAr
        Gaussian interface object
    spin : str
        Spin treatment to use:
        - 'r': Restricted (same orbitals for alpha/beta)
        - 'ro'/'u': Unrestricted (separate alpha/beta)
        - 'g': Generalized (non-collinear)

    Returns
    -------
    ndarray
        Density matrix in appropriate format for spin treatment:
        - Restricted: Single block
        - Unrestricted: Two blocks (alpha/beta)
        - Generalized: Single block with complex entries

    Raises
    ------
    ValueError
        If spin treatment is not recognized
    """
    # Set up Fock matrix and atom indexing
    # Note: positive indices are alpha/paired orbitals, negative are beta orbitals
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
    """
    Build Fock matrix from Gaussian checkpoint file.

    Extracts the Fock matrix and orbital indices from a Gaussian checkpoint
    file based on the specified spin treatment. Handles restricted,
    unrestricted, and generalized cases.

    Parameters
    ----------
    bar : QCBinAr
        Gaussian interface object
    spin : str
        Spin treatment to use:
        - 'r': Restricted (same orbitals for alpha/beta)
        - 'ro'/'u': Unrestricted (separate alpha/beta)
        - 'g': Generalized (non-collinear)

    Returns
    -------
    tuple
        (Fock, locs) where:
        - Fock: ndarray, Fock matrix in appropriate format for spin treatment
        - locs: ndarray, Orbital indices with positive for alpha/paired and
                negative for beta orbitals

    Raises
    ------
    ValueError
        If spin treatment is not recognized
    """
    # Set up Fock matrix and atom indexing
    # Note: positive indices are alpha/paired orbitals, negative are beta orbitals
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
    """
    Get orbital energies from Gaussian checkpoint file.

    Extracts orbital energies from a Gaussian checkpoint file based on
    the specified spin treatment. Converts energies from Hartrees to eV.

    Parameters
    ----------
    bar : QCBinAr
        Gaussian interface object
    spin : str
        Spin treatment to use:
        - 'r': Restricted (same orbitals for alpha/beta)
        - 'ro'/'u': Unrestricted (separate alpha/beta)
        - 'g': Generalized (non-collinear)

    Returns
    -------
    ndarray
        Array of orbital energies in eV, sorted in ascending order.
        Format depends on spin treatment:
        - Restricted: Alternating alpha/beta pairs
        - Unrestricted: Alternating alpha/beta pairs
        - Generalized: Single set of energies

    Raises
    ------
    ValueError
        If spin treatment is not recognized
    """
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
    """
    Store density matrix in Gaussian checkpoint format.

    Converts the density matrix to the appropriate format based on spin
    treatment and stores it in the Gaussian checkpoint file. Handles
    compression and proper typing of matrices.

    Parameters
    ----------
    bar : QCBinAr
        Gaussian interface object
    P : ndarray
        Density matrix to store
    spin : str
        Spin treatment to use:
        - 'r': Restricted (same orbitals for alpha/beta)
        - 'ro'/'u': Unrestricted (separate alpha/beta)
        - 'g': Generalized (non-collinear)

    Notes
    -----
    For restricted calculations, the density is divided by 2 to account
    for double occupation. For generalized calculations, complex matrices
    are used.

    Raises
    ------
    ValueError
        If spin treatment is not recognized
    """
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


