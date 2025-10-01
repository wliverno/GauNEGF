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
import jax
import jax.numpy as np
import jax.numpy.linalg as LA
import jax.lax as lax
from jax import jit

# Configuration
from gauNEGF.config import (ETA, SURFACE_GREEN_CONVERGENCE, SURFACE_RELAXATION_FACTOR)
from gauNEGF.utils import fractional_matrix_power

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
        self.X = np.array(fractional_matrix_power(Overlap, -0.5))
        # Keep indsList as Python list - loop unrolls with concrete indices
        self.indsList = [np.array(inds) for inds in indsList]
        
        # Set Contact Coupling
        if taus is None:
            taus = [self.indsList[-1], self.indsList[0]]
        taus = [np.array(tau) for tau in taus]
        if len(np.shape(taus[0])) == 1:
           self.tauFromFock = True
           self.tauInds = taus
           self.tauList = [self.F[np.ix_(taus[0],self.indsList[0])], self.F[np.ix_(taus[1],self.indsList[-1])]]
           self.stauList = [self.S[np.ix_(taus[0],self.indsList[0])], self.S[np.ix_(taus[1],self.indsList[-1])]]
        else:
           self.tauFromFock = False
           self.tauList = [np.array(tau) for tau in taus]
           self.stauList = [np.array(stau) for stau in staus]
        
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
        # gPrev needs to be stacked JAX array for traced indexing
        # aList is now shape (num_contacts, N, N), so gPrev should match
        self.gPrev = [np.zeros_like(a, dtype=complex) for a in self.aList]

        # Store number of contacts for loop bounds
        self.num_contacts = len(indsList)

        # JIT compile g and sigma methods with static contact index
        # This compiles separate versions for each contact (i=0, i=1, etc.)
        # The expensive iterative calculation gets fully optimized
        self.g = jit(self.g, static_argnums=(1,))  # i is argument 1 (after self)
        self.sigma = jit(self.sigma, static_argnums=(1,))  # i is argument 1
    
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
            # Build lists first, then stack into JAX arrays
            aList_temp = []
            aSList_temp = []
            for inds in self.indsList:
                aList_temp.append(self.F[np.ix_(inds, inds)])
                aSList_temp.append(self.S[np.ix_(inds, inds)])
            # Stack into JAX arrays for traced indexing
            self.aList = [np.array(a) for a in aList_temp]
            self.aSList = [np.array(aS) for aS in aSList_temp]
        else:
            # Stack provided matrices into JAX arrays
            self.aList = [np.array(alpha) for alpha in alphas]
            self.aSList = [np.array(aOverlap) for aOverlap in aOverlaps]

        if self.contactFromFock:
            # tauList and stauList should already be lists from initialization
            self.bList = [np.array(tau) for tau in self.tauList]
            self.bSList = [np.array(stau) for stau in self.stauList]
        else:
            self.bList = [np.array(beta) for beta in betas]
            self.bSList = [np.array(bOverlap) for bOverlap in bOverlaps]
    
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
        A = (E+1j*self.eta)*Salpha - alpha
        B = (E+1j*self.eta)*Sbeta - beta
        g = self.gPrev[i]
        B_dag = B.conj().T

        # Iterative solution using jax.lax.while_loop
        MAX_ITER = 2000

        def cond_fun(state):
            count, diff, g, g_prev = state
            return (diff > conv) & (count < MAX_ITER)

        def body_fun(state):
            count, diff, g, g_prev = state

            # Compute new Green's function using JAX operations
            g_new = LA.inv(A - B @ g @ B_dag)

            # Compute convergence metric
            dg = np.abs(g_new - g) / np.maximum(np.abs(g_new), 1e-12)
            diff = np.max(dg)

            # Apply relaxation mixing
            g = g_new * relFactor + g * (1 - relFactor)
            count += 1
            return (count, diff, g, g_prev)

        # Initial state: (count, diff, g, g_prev)
        init_state = (0, np.inf, g, self.gPrev[i])
        count, diff, g, g_prev = lax.while_loop(cond_fun, body_fun, init_state)

        # Check convergence and warn if needed
        if diff>conv:
            jax.debug.print('Warning: exceeded max iterations! E: {E}, Conv: {diff}',
                             E=E, diff=diff, ordered=True)

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
            indsList = self.indsList  # Python list
            self.F = self.F.at[np.ix_(indsList[0], indsList[0])].set(self.F[np.ix_(taus[0], taus[0])].copy())
            self.F = self.F.at[np.ix_(indsList[-1], indsList[-1])].set(self.F[np.ix_(taus[1], taus[1])].copy())
            # Rebuild stacked arrays from new F
            tau_temp = [self.F[np.ix_(taus[0],indsList[0])], self.F[np.ix_(taus[1],indsList[-1])]]
            stau_temp = [self.S[np.ix_(taus[0],indsList[0])], self.S[np.ix_(taus[1],indsList[-1])]]
            self.tauList = [np.array(tau) for tau in tau_temp]
            self.stauList = [np.array(stau) for stau in stau_temp]
        if not self.contactFromFock:
            if self.fermiList[0] == None:
                self.fermiList[0] = mu1
                self.fermiList[-1] = mu2
            else:
                for i,mu in zip([0,-1], [mu1, mu2]):
                    fermi = self.fermiList[i]
                    if fermi is not None and mu is not None and fermi != mu:
                        dFermi = mu - fermi
                        # Use JAX immutable updates
                        self.aList = self.aList.at[i].set(self.aList[i] + dFermi*np.eye(len(self.aList[i])))
                        self.bList = self.bList.at[i].set(self.bList[i] + dFermi*self.bSList[i])
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
        i : int (can be traced)
            Contact index
        conv : float, optional
            Convergence criterion for surface Green's function (default: 1e-5)

        Returns
        -------
        ndarray
            Self-energy matrix for contact i
        """
        sigma = np.zeros(self.F.shape, dtype=complex)
        inds = self.indsList[i]
        stau = self.stauList[i]
        tau = self.tauList[i]
        t = E*stau - tau
        sig = t @ self.g(E, i, conv) @ t.conj().T
        sigma = sigma.at[np.ix_(inds, inds)].add(sig)
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
        # Use Python for loop - JAX unrolls it with concrete indices
        sigma = np.zeros(self.F.shape, dtype=complex)
        for i in range(self.num_contacts):
            sigma = sigma + self.sigma(E, i, conv)
        return sigma
    

    

