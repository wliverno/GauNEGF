"""
Surface Green's function implementation for 1D chain contacts.

This module provides a 1D chain implementation for modeling semi-infinite
contacts in quantum transport calculations. It supports three usage patterns:

a) Fully automatic extraction from Fock matrix:
   surfG1D(F, S, [[contact1], [contact2]], [[contact1connection], [contact2connection]])
   - All parameters extracted from F/S using contact and connection indices

b) Fock matrix with custom coupling:
   surfG1D(F, S, [[contact1], [contact2]], [tau1, tau2], [stau1, stau2])
   - Contact parameters from F/S, but with custom coupling matrices

c) Fully specified contacts:
   surfG1D(F, S, [[contact1], [contact2]], [tau1, tau2], [stau1, stau2],
          [alpha1, alpha2], [salpha1, salpha2], [beta1, beta2], [sbeta1, sbeta2])
   - All contact parameters specified manually

The implementation uses an iterative scheme to calculate surface Green's
functions for 1D chain contacts, with support for both manual parameter
specification and automatic extraction from DFT calculations.
"""

# Python packages
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# Use JAX functions directly

# Configuration
from gauNEGF.config import (ETA, SURFACE_GREEN_CONVERGENCE, SURFACE_RELAXATION_FACTOR)

#Constants
kB = 8.617e-5           # eV/Kelvin

class surfG:
    """
    Surface Green's function calculator for 1D chain contacts.

    This class implements the surface Green's function calculation for 1D chain
    contacts. It supports three usage patterns:

    a) Fully automatic extraction from Fock matrix:
       - Provide contact indices and connection indices
       - All parameters extracted from F/S matrices
       Example: surfG1D(F, S, [[c1], [c2]], [[c1conn], [c2conn]])

    b) Fock matrix with custom coupling:
       - Provide contact indices and coupling matrices
       - Onsite contact parameters from F/S, coupling specified manually
       Example: surfG1D(F, S, [[c1], [c2]], [tau1, tau2], [stau1, stau2])

    c) Fully specified contacts:
       - All contact parameters provided manually
       Example: surfG1D(F, S, [[c1], [c2]], [tau1, tau2], [stau1, stau2],
                [alpha1, alpha2], [salpha1, salpha2], [beta1, beta2], [sbeta1, sbeta2])

    Parameters
    ----------
    Fock : ndarray
        Fock matrix for the extended system
    Overlap : ndarray
        Overlap matrix for the extended system
    indsList : list of lists
        Lists of orbital indices for each contact region
    taus : list or None, optional
        Either coupling matrices or connection indices (default: None)
        - If indices: [[contact1connection], [contact2connection]]
        - If matrices: [tau1, tau2]
    staus : list or None, optional
        Overlap matrices for coupling, required if taus are matrices (default: None)
    alphas : list of ndarray or None, optional
        On-site energies for contacts, required for pattern (c) (default: None)
    aOverlaps : list of ndarray or None, optional
        On-site overlap matrices for contacts, required for pattern (c) (default: None)
    betas : list of ndarray or None, optional
        Hopping matrices between contact unit cells, required for pattern (c) (default: None)
    bOverlaps : list of ndarray or None, optional
        Overlap matrices between contact unit cells, required for pattern (c) (default: None)
    eta : float, optional
        Broadening parameter in eV (default: 1e-9)

    Attributes
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    X : ndarray
        Inverse square root of overlap matrix for orthogonalization
    tauList : list
        Contact coupling matrices
    stauList : list
        Contact coupling overlap matrices
    aList : list
        On-site energy matrices for contacts
    aSList : list
        On-site overlap matrices for contacts
    bList : list
        Hopping matrices between contact unit cells
    bSList : list
        Overlap matrices between contact unit cells
    gPrev : list
        Previous surface Green's functions for convergence
    """
    def __init__(self, Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA):
        """
        Initialize the surface Green's function calculator.

        The initialization follows one of three patterns:
        a) Fully automatic: Only provide Fock, Overlap, indsList, and connection indices in taus
        b) Custom coupling: Provide Fock, Overlap, indsList, coupling matrices in taus, and staus
        c) Fully specified: Provide all parameters including alphas, aOverlaps, betas, bOverlaps

        Parameters
        ----------
        Fock : ndarray
            Fock matrix for the extended system
        Overlap : ndarray
            Overlap matrix for the extended system
        indsList : list of lists
            Lists of orbital indices for each contact region
        taus : list or None, optional
            Either coupling matrices or connection indices (default: None)
            - If indices: [[contact1connection], [contact2connection]]
            - If matrices: [tau1, tau2]
        staus : list or None, optional
            Overlap matrices for coupling, required if taus are matrices (default: None)
        alphas : list of ndarray or None, optional
            On-site energies for contacts, required for pattern (c) (default: None)
        aOverlaps : list of ndarray or None, optional
            On-site overlap matrices for contacts, required for pattern (c) (default: None)
        betas : list of ndarray or None, optional
            Hopping matrices between contact unit cells, required for pattern (c) (default: None)
        bOverlaps : list of ndarray or None, optional
            Overlap matrices between contact unit cells, required for pattern (c) (default: None)
        eta : float, optional
            Broadening parameter in eV (default: 1e-9)

        Notes
        -----
        The initialization will raise an error if:
        - The parameters don't match one of the three usage patterns
        - taus contains matrices but staus is None
        - alphas is provided but aOverlaps is None
        - betas is provided but bOverlaps is None
        """
        # Set up system
        self.F = np.array(Fock)
        self.S = np.array(Overlap)
        self.X = np.array(jnp.linalg.matrix_power(Overlap, -0.5))
        self.indsList = indsList
        self.poleList = len(indsList)*[np.array([], dtype=complex)]
        self.Egrid = len(indsList)*[np.array([], dtype=complex)]
        
        # Set Contact Coupling
        if taus is None:
            taus = [indsList[-1], indsList[0]]
        if len(np.shape(taus[0])) == 1:
           self.tauFromFock = True
           self.tauInds = taus
           self.tauList = [self.F[np.ix_(taus[0],indsList[0])], self.F[np.ix_(taus[1],indsList[-1])]]
           self.stauList = [self.S[np.ix_(taus[0],indsList[0])], self.S[np.ix_(taus[1],indsList[-1])]]
        else:
           self.tauFromFock = False
           self.tauList = taus
           self.stauList = staus
        
        # Set up contact information
        if alphas is None:
            self.contactFromFock = True
            self.setContacts()
        else:
            self.contactFromFock = False
            self.setContacts(alphas, aOverlaps, betas, bOverlaps)
            self.fermiList = [None]*len(indsList)

        # Set up broadening for retarded/advanced Green's function, initialize g
        self.eta = eta
        self.gPrev = [np.zeros(np.shape(alpha)) for alpha in self.aList]
    
    def setContacts(self, alphas=None, aOverlaps=None, betas=None, bOverlaps=None):
        """
        Update contact parameters for the 1D chain.

        This method is used internally during initialization and can be called
        later to update contact parameters. It follows the same patterns as
        initialization:

        a) If self.contactFromFock is True (patterns a and b):
           - Parameters are extracted from F/S matrices
           - Any provided parameters are ignored

        b) If self.contactFromFock is False (pattern c):
           - All parameters must be provided together
           - Partial updates are not supported

        Parameters
        ----------
        alphas : list of ndarray or None, optional
            On-site energies for contacts (default: None)
        aOverlaps : list of ndarray or None, optional
            On-site overlap matrices for contacts (default: None)
        betas : list of ndarray or None, optional
            Hopping matrices between contact unit cells (default: None)
        bOverlaps : list of ndarray or None, optional
            Overlap matrices between contact unit cells (default: None)

        Notes
        -----
        When using pattern (c), all parameters must be provided together.
        Partial updates (providing some parameters but not others) are not
        supported and will raise an error.
        """
        if self.contactFromFock:
            self.aList = []
            self.aSList = []
            for inds in self.indsList:
                self.aList.append(self.F[np.ix_(inds, inds)])
                self.aSList.append(self.S[np.ix_(inds, inds)])
        else:
            self.aList = alphas
            self.aSList = aOverlaps
            
        if self.contactFromFock:
            self.bList = self.tauList
            self.bSList = self.stauList
        else:
            self.bList = betas
            self.bSList = bOverlaps
    
    def g(self, E, i, conv=SURFACE_GREEN_CONVERGENCE, relFactor=SURFACE_RELAXATION_FACTOR):
        """
        Calculate surface Green's function for a contact.

        Uses an iterative scheme to calculate the surface Green's function
        for contact i at energy E. The iteration continues until the change
        in the Green's function is below the convergence criterion.

        Parameters
        ----------
        E : float
            Energy point in eV
        i : int
            Contact index
        conv : float, optional
            Convergence criterion for iteration (default: 1e-5)
        relFactor : float, optional
            Relaxation factor for iteration mixing (default: 0.1)

        Returns
        -------
        ndarray
            Surface Green's function matrix for contact i

        Notes
        -----
        The method uses the previous solution as an initial guess to improve
        convergence. For the first calculation at a given energy, it uses
        zeros as the initial guess. The relaxation factor controls mixing
        between iterations to help convergence.
        """
        alpha = self.aList[i]
        Salpha = self.aSList[i]
        beta = self.bList[i]
        Sbeta = self.bSList[i]

        # Prepare matrices using JAX
        A = jnp.array((E+1j*self.eta)*Salpha - alpha)
        B = jnp.array((E+1j*self.eta)*Sbeta - beta)
        g = jnp.array(self.gPrev[i].copy())
        B_dag = B.conj().T

        # Iterative solution using JAX operations
        MAX_ITER = 2000
        count = 0
        diff = conv + 1

        while diff > conv and count < MAX_ITER:
            g_prev = g

            # Compute new Green's function using JAX operations
            g_new = jnp.linalg.inv(A - B @ g @ B_dag)

            # Compute convergence metric
            dg = jnp.abs(g_new - g) / jnp.maximum(jnp.abs(g_new), 1e-12)
            diff = float(jnp.max(dg))

            # Apply relaxation mixing
            g = g_new * relFactor + g * (1 - relFactor)
            count += 1

        # Check convergence and warn if needed
        if diff > conv:
            print(f'Warning: exceeded max iterations! E: {E}, Conv: {diff}')

        # Convert back to numpy for compatibility
        g = np.array(g)

        # Store result for next iteration initial guess
        self.gPrev[i] = g
        return g
   
    def setF(self, F, mu1=None, mu2=None):
        """
        Update the Fock matrix and contact chemical potentials.

        This method updates the system's Fock matrix and optionally shifts
        the contact chemical potentials. If the contacts are extracted from
        the Fock matrix, their parameters are automatically updated.

        Parameters
        ----------
        F : ndarray
            New Fock matrix for the system
        mu1 : float or None, optional
            Chemical potential for first contact in eV (default: None)
        mu2 : float or None, optional
            Chemical potential for second contact in eV (default: None)

        Notes
        -----
        If chemical potentials are provided, the corresponding contact
        parameters are shifted to align with the new potentials.
        """
        self.F = F
        if self.tauFromFock:
            taus = self.tauInds
            indsList = self.indsList
            self.F[np.ix_(indsList[0], indsList[0])] = self.F[np.ix_(taus[0], taus[0])].copy()
            self.F[np.ix_(indsList[-1], indsList[-1])] = self.F[np.ix_(taus[1], taus[1])].copy()
            self.tauList = [self.F[np.ix_(taus[0],indsList[0])], self.F[np.ix_(taus[1],indsList[-1])]]
            self.stauList = [self.S[np.ix_(taus[0],indsList[0])], self.S[np.ix_(taus[1],indsList[-1])]]
        if not self.contactFromFock:
            if self.fermiList[0] == None:
                self.fermiList[0] = mu1
                self.fermiList[-1] = mu2
            else:
                for i,mu in zip([0,-1], [mu1, mu2]):
                    fermi = self.fermiList[i]
                    if fermi is not None and mu is not None and fermi != mu:
                        dFermi = mu - fermi
                        self.aList[i] += dFermi*np.eye(len(self.aList[i]))
                        self.bList[i] += dFermi*self.bSList[i]
                        self.fermiList[i] = mu
    
    def sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE):
        """
        Calculate self-energy matrix for a contact.

        Computes the self-energy matrix for contact i at energy E using
        the surface Green's function. The self-energy represents the
        effect of the semi-infinite contact on the device region.

        Parameters
        ----------
        E : float
            Energy point in eV
        i : int
            Contact index
        conv : float, optional
            Convergence criterion for surface Green's function (default: 1e-5)

        Returns
        -------
        ndarray
            Self-energy matrix for contact i
        """
        sigma = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        inds = self.indsList[i]
        stau = self.stauList[i]
        tau = self.tauList[i]
        t = E*stau - tau
        sig = t @ self.g(E, i, conv) @ t.conj().T
        sigma[np.ix_(inds, inds)] += sig
        return sigma
    
    def sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE):
        """
        Calculate total self-energy matrix from all contacts.

        Computes the total self-energy matrix at energy E by summing
        contributions from all contacts. This represents the combined
        effect of all semi-infinite contacts on the device region.

        Parameters
        ----------
        E : float
            Energy point in eV
        conv : float, optional
            Convergence criterion for surface Green's functions (default: 1e-5)

        Returns
        -------
        ndarray
            Total self-energy matrix from all contacts
        """
        sigma = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        for i, inds in enumerate(self.indsList):
            stau = self.stauList[i]
            tau = self.tauList[i]
            t = E*stau - tau
            sig = t @ self.g(E, i, conv) @ t.conj().T
            sigma[np.ix_(inds, inds)] += sig
        return sigma
    

    
