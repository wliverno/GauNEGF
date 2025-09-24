"""
Constant self-energy module for NEGF calculations.

This module provides energy-independent self-energies for NEGF calculations.
While useful for testing and validation, it also serves production calculations
where constant self-energies are appropriate, particularly for non-zero
temperature calculations where energy-dependent effects may be less critical
or when computational efficiency is prioritized over full energy dependence.
"""

import numpy as np
from gauNEGF.matTools import formSigma
from gauNEGF.config import SURFACE_GREEN_CONVERGENCE

class surfGTest:
    """
    Energy-independent self-energy calculator.
    
    This class provides constant (energy-independent) self-energies for
    NEGF calculations. While useful for testing and validation, it also
    enables production calculations with constant self-energies, which
    can be appropriate for non-zero temperature calculations or when
    computational efficiency is prioritized. It implements the same
    interface as other surface Green's function calculators.
    
    Parameters
    ----------
    Fock : ndarray
        Fock matrix for the system
    Overlap : ndarray  
        Overlap matrix for the system
    indsList : list
        List containing [left_contact_indices, right_contact_indices]
    sig1 : complex or array-like, optional
        Self-energy for left contact (default: None)
    sig2 : complex or array-like, optional  
        Self-energy for right contact (default: None)
        
    Attributes
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    X : ndarray
        Inverse square root of overlap matrix
    N : int
        Size of the system matrices
    indsList : list
        Contact orbital indices
    sig : list
        List of self-energy matrices [left, right]
        
    Notes
    -----
    If no self-energies are provided, defaults to -0.05j on contact orbitals.
    This class is compatible with scfE.py for temperature-dependent calculations
    where constant self-energies provide a good approximation. Many method 
    parameters are unused but maintained for interface consistency with other
    surface Green's function calculators.
    """
    def __init__(self, Fock, Overlap, indsList, sig1=None, sig2=None):
        """
        Initialize constant self-energy calculator.
        
        Parameters
        ----------
        Fock : ndarray
            Fock matrix for the system
        Overlap : ndarray
            Overlap matrix for the system  
        indsList : list
            List containing [left_contact_indices, right_contact_indices]
        sig1 : complex or array-like, optional
            Self-energy for left contact (default: None)
        sig2 : complex or array-like, optional
            Self-energy for right contact (default: None)
        """
        self.F = Fock
        self.S = Overlap
        self.N = len(Fock)
        self.indsList = indsList
        self.sig = [np.array(np.zeros((self.N, self.N)), dtype=complex)]*2
        if sig1 is not None:
            self.sig[0] = formSigma(indsList[0], sig1, self.N, self.S)
            if sig2 is None:
                self.sig[1] = formSigma(indsList[1], sig1, self.N, self.S)
            else:
                self.sig[1] = formSigma(indsList[1], sig2, self.N, self.S)
        else:
            self.sig[0][np.ix_(indsList[0], indsList[0])]= np.diag([-0.05j]*self.N)
            self.sig[1][np.ix_(indsList[1], indsList[1])]= np.diag([-0.05j]*self.N)
    
    def sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE):
        """
        Get self-energy matrix for a specific contact.
        
        Parameters
        ----------
        E : float
            Energy point (unused - kept for API compatibility)
        i : int
            Contact index (0 for left, 1 for right)
        conv : float, optional
            Convergence parameter (unused - kept for API compatibility)
            
        Returns
        -------
        ndarray
            Self-energy matrix for contact i
        """
        return self.sig[i]
    def sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE):
        """
        Calculate total self-energy matrix from all contacts.
        
        Parameters
        ----------
        E : float
            Energy point (unused - kept for API compatibility)
        conv : float, optional
            Convergence parameter (unused - kept for API compatibility)
            
        Returns
        -------
        ndarray
            Total self-energy matrix (sum of all contact self-energies)
        """
        sigTot = np.array(np.zeros((self.N, self.N)), dtype=complex)
        for i in range(len(self.indsList)):
            sigTot += self.sigma(E,i,conv)
        return sigTot
    def setF(self, F, mu1, mu2):
        """
        Update Fock matrix and chemical potentials.
        
        Parameters
        ----------
        F : ndarray
            New Fock matrix
        mu1 : float
            Chemical potential for left contact (unused - kept for API compatibility)
        mu2 : float
            Chemical potential for right contact (unused - kept for API compatibility)
            
        Notes
        -----
        Chemical potentials are ignored since self-energies are constant.
        This method exists purely for API compatibility with other surface
        Green's function calculators.
        """
        self.F = F
