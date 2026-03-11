# Python packages
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit

# Configuration
from gauNEGF.config import (ETA, SURFACE_GREEN_CONVERGENCE, SURFACE_RELAXATION_FACTOR,
                             OVERLAP_EIGENVALUE_RATIO)
from gauNEGF.utils import fractional_matrix_power, inv

#Constants

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
       - If staus=None (default), assumes orthonormal coupling
       Example: surfG1D(F, S, [[c1], [c2]], [tau1, tau2])

    c) Fully specified contacts:
       - All contact parameters provided manually
       - If aOverlaps=None, onsite overlap defaults to identity (orthonormal)
       - If bOverlaps=None, hopping overlap defaults to zeros (orthonormal)
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
        Overlap matrices for coupling (default: None = orthonormal coupling)
        None entries trigger de-orthonormalization in sigma()
    alphas : list of ndarray or None, optional
        On-site energies for contacts, required for pattern (c) (default: None)
    aOverlaps : list of ndarray or None, optional
        On-site overlap matrices; defaults to identity when None (orthonormal)
    betas : list of ndarray or None, optional
        Hopping matrices between contact unit cells, required for pattern (c) (default: None)
    bOverlaps : list of ndarray or None, optional
        Overlap matrices between contact unit cells; defaults to zeros when None
    eta : float, optional
        Broadening parameter in eV (default: 1e-9)

    Attributes
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    X : ndarray
        Inverse square root of overlap matrix for orthogonalization (S^-0.5)
    Xi : ndarray
        Square root of overlap matrix for de-orthonormalization (S^+0.5 = inv(X))
    tauList : list
        Contact coupling matrices
    stauList : list
        Contact coupling overlap matrices (None entries -> orthonormal coupling)
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
    def __init__(self, Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA, spin='r'):
        """
        Initialize the surface Green's function calculator.

        The initialization follows one of three patterns:
        a) Fully automatic: Only provide Fock, Overlap, indsList, and connection indices in taus
        b) Custom coupling: Provide Fock, Overlap, indsList, coupling matrices in taus
           - staus=None (default) means orthonormal coupling -> de-ortho applied in sigma()
        c) Fully specified: Provide all parameters including alphas, aOverlaps, betas, bOverlaps
           - aOverlaps=None defaults to identity (orthonormal onsite)
           - bOverlaps=None defaults to zeros (orthonormal hopping)

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
            Overlap matrices for coupling (default: None = orthonormal)
        alphas : list of ndarray or None, optional
            On-site energies for contacts, required for pattern (c) (default: None)
        aOverlaps : list of ndarray or None, optional
            On-site overlap matrices for contacts (default: None = identity)
        betas : list of ndarray or None, optional
            Hopping matrices between contact unit cells, required for pattern (c) (default: None)
        bOverlaps : list of ndarray or None, optional
            Overlap matrices between contact unit cells (default: None = zeros)
        eta : float, optional
            Broadening parameter in eV (default: 1e-9)
        """
        # Set up system
        self.F = jnp.array(Fock)
        self.S = jnp.array(Overlap)
        self.S_orig = jnp.array(Overlap)
        self.spin = spin
        self.X = jnp.array(fractional_matrix_power(Overlap, -0.5))
        self.Xi = jnp.linalg.inv(self.X)
        # Keep indsList as Python list - loop unrolls with concrete indices
        self.indsList = [jnp.array(inds) for inds in indsList]

        # Set Contact Coupling
        if taus is None:
            taus = [self.indsList[-1], self.indsList[0]]
        taus = [jnp.array(tau) for tau in taus]
        if len(jnp.shape(taus[0])) == 1:
            self.tauFromFock = True
            self.tauInds = taus
            taus = [self.F[jnp.ix_(self.tauInds[0],self.indsList[0])], 
                    self.F[jnp.ix_(self.tauInds[1],self.indsList[-1])]]
            staus = [self.S[jnp.ix_(self.tauInds[0],self.indsList[0])], 
                     self.S[jnp.ix_(self.tauInds[1],self.indsList[-1])]]
        else:
            self.tauFromFock = False
        self.tauList = taus
        self.stauList = ([None] * len(taus) if staus is None
                         else [None if stau is None else jnp.array(stau) for stau in staus])

        # Set up contact information
        if alphas is None:
            self.contactFromFock = True
            self._setContacts()
        else:
            self.contactFromFock = False
            self._setContacts(alphas, aOverlaps, betas, bOverlaps)
            self.fermiList = [None]*len(indsList)
        
        # Set up broadening for retarded/advanced Green's function, initialize g
        self.eta = eta

        # Store number of contacts for loop bounds
        self.num_contacts = len(indsList)

        # JIT compile g and sigma methods with static contact index
        # This compiles separate versions for each contact (i=0, i=1, etc.)
        # The expensive iterative calculation gets fully optimized
        self._rejit()

    
    def _setContacts(self, alphas=None, aOverlaps=None, betas=None, bOverlaps=None):
        """Internal: build aList/aSList/bList/bSList and regularize contacts.

        contactFromFock=True: extracts alpha/Salpha from self.F and self.S_orig.
        contactFromFock=False: uses provided alphas/aOverlaps/betas/bOverlaps.
        Calls _regularizeContacts() after setting lists.
        """
        if self.contactFromFock:
            alphas = []
            aOverlaps = []
            for inds in self.indsList:
                alphas.append(self.F[jnp.ix_(inds, inds)])
                aOverlaps.append(self.S_orig[jnp.ix_(inds, inds)])
            self.aList = alphas
            self.aSList = aOverlaps
            self.bList = [jnp.array(tau) for tau in self.tauList]
            self.bSList = [jnp.zeros_like(tau) if stau is None else jnp.array(stau)
                           for tau, stau in zip(self.tauList, self.stauList)]
        else:
            self.aList = [jnp.array(alpha) for alpha in alphas]
            self.bList = [jnp.array(beta) for beta in betas]
            self.aSList = ([jnp.eye(len(alpha)) for alpha in alphas] if aOverlaps is None
                           else [jnp.array(aOverlap) for aOverlap in aOverlaps])
            self.bSList = ([jnp.zeros_like(beta) for beta in betas] if bOverlaps is None
                           else [jnp.zeros_like(beta) if bOverlap is None else jnp.array(bOverlap)
                                 for beta, bOverlap in zip(betas, bOverlaps)])
        self._regularizeContacts()

    def _regularizeContacts(self):
        """Ensure the infinite chain overlap is PSD via congruent eigenvalue clipping.

        For each contact i, diagonalizes S0 = aSList[i]. If the minimum eigenvalue
        is already >= OVERLAP_EIGENVALUE_RATIO * max(eigenvalue), no transform is
        applied. Otherwise builds a congruence transform C that floors small
        eigenvalues and applies C' @ X @ C to all four contact matrices in-place:
            aSList[i], aList[i], bSList[i], bList[i]

        This keeps the basis dimension unchanged and each basis vector as close as
        possible to an original orbital. No downstream sigma correction is needed.
        See docs/plans/2026-03-10-congruent-clipping-design.md.
        """
        import numpy as np
        for i in range(len(self.indsList)):
            S0 = np.array(self.aSList[i])
            n = S0.shape[0]
            eigvals, U = np.linalg.eigh(S0)
            lam_max = float(eigvals[-1])
            lam_min_thresh = OVERLAP_EIGENVALUE_RATIO * lam_max

            if float(eigvals[0]) >= lam_min_thresh:
                # Already PSD -- no transform needed
                continue

            lam_prime = np.maximum(eigvals, lam_min_thresh)
            C = U @ np.diag(np.sqrt(lam_prime / eigvals))
            C = C.astype(self.aSList[i].dtype)

            H0 = np.array(self.aList[i])
            H1 = np.array(self.bList[i])
            S1 = np.array(self.bSList[i])

            self.aSList[i] = jnp.array(C.conj().T @ S0 @ C)
            self.aList[i]  = jnp.array(C.conj().T @ H0 @ C)
            self.bSList[i] = jnp.array(C.conj().T @ S1 @ C)
            self.bList[i]  = jnp.array(C.conj().T @ H1 @ C)

            print(f'Contact overlap regularized (contact {i}): '
                  f'min_eig {eigvals[0]:.4e} -> {lam_min_thresh:.4e}')

    def _rejit(self):
        """Recompile g and sigma to pick up updated contact parameters.

        JAX JIT caches compiled functions keyed on shape/dtype of closed-over
        arrays, not their values. After setF/_setContacts change aList/bList,
        creating fresh JIT wrappers forces a re-trace on next call.
        self.__class__.g always refers to the original class method regardless
        of what self.g currently points to (instance vs class attribute).
        """
        self.g = jit(self.__class__.g.__get__(self), static_argnums=(1,))
        self.sigma = jit(self.__class__.sigma.__get__(self), static_argnums=(1,))

    def g(self, E, i, conv=SURFACE_GREEN_CONVERGENCE, relFactor=0.5):#SURFACE_RELAXATION_FACTOR):
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
            Relaxation factor for iteration mixing (default: 0.5)

        Returns
        -------
        ndarray
            Surface Green's function matrix for contact i
        """
        alpha = self.aList[i]
        Salpha = self.aSList[i]
        beta = self.bList[i]
        Sbeta = self.bSList[i]

        # Prepare matrices using JAX
        A = (E+1j*self.eta)*Salpha - alpha
        B = (E+1j*self.eta)*Sbeta - beta
        B_dag = B.conj().T

        # Iterative solution using jax.lax.while_loop
        MAX_ITER = 10000

        def cond_fun(state):
            count, diff, g = state
            return (diff > conv) & (count < MAX_ITER)

        def body_fun(state):
            count, diff, g = state

            # Compute new Green's function using JAX operations
            g_new = inv(A - B @ g @ B_dag)

            # Compute convergence metric
            dg = jnp.abs(g_new - g) / jnp.maximum(jnp.abs(g_new), 1e-12)
            diff = jnp.max(dg)

            # Apply relaxation mixing
            g = g_new * relFactor + g * (1 - relFactor)
            count += 1
            return (count, diff, g)

        # Initial state: (count, diff, g)
        init_state = (0, jnp.inf, inv(A))
        count, diff, g = lax.while_loop(cond_fun, body_fun, init_state)
        #lax.cond(diff > conv, 
        #        lambda E: jax.debug.print("WARNING: EXCEEDED ITERATIONS at {E:.2f} eV:  count={count}, diff={diff:.2e}", 
        #                                    E=E, count=count, diff=diff), 
        #        lambda E:None, E)
        

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
        """
        self.F = jnp.array(F)
        if self.tauFromFock:
            taus = self.tauInds
            indsList = self.indsList  # Python list
            # Rebuild coupling arrays from new F
            tau_temp = [self.F[jnp.ix_(taus[0],indsList[0])], self.F[jnp.ix_(taus[1],indsList[-1])]]
            stau_temp = [self.S[jnp.ix_(taus[0],indsList[0])], self.S[jnp.ix_(taus[1],indsList[-1])]]
            self.tauList = tau_temp
            self.stauList = stau_temp
        if self.contactFromFock:
            # Rebuild aList/bList from new F and re-trace JIT'd functions
            self._setContacts()
            self._rejit()
        if not self.contactFromFock:
            # Track chemical potentials but do NOT shift aList/bList.
            # The retarded self-energy is independent of chemical potential;
            # mu enters only through the Fermi function in density integration.
            if self.fermiList[0] is None:
                self.fermiList[0] = mu1
                self.fermiList[-1] = mu2
            else:
                for i, mu in zip([0, -1], [mu1, mu2]):
                    if mu is not None:
                        self.fermiList[i] = mu

    def sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE):
        """
        Calculate self-energy matrix for a contact.

        Computes the self-energy matrix for contact i at energy E using
        the surface Green's function. The self-energy represents the
        effect of the semi-infinite contact on the device region.

        When stauList[i] is None (orthonormal coupling), applies de-orthonormalization:
        sig -> Xi[inds,inds] @ sig @ Xi[inds,inds] where Xi = S^+0.5 = inv(X).

        Parameters
        ----------
        E : float
            Energy point in eV
        i : int (static)
            Contact index
        conv : float, optional
            Convergence criterion for surface Green's function (default: 1e-5)

        Returns
        -------
        ndarray
            Self-energy matrix for contact i
        """
        sigma = jnp.zeros(self.F.shape, dtype=complex)
        inds = self.indsList[i]
        stau = self.stauList[i]
        tau = self.tauList[i]
        t = (-tau) if stau is None else (E*stau - tau)
        sig = t @ self.g(E, i, conv) @ t.conj().T
        if stau is None:
            Xi_i = self.Xi[jnp.ix_(inds, inds)]
            sig = Xi_i @ sig @ Xi_i
        sigma = sigma.at[jnp.ix_(inds, inds)].add(sig)
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
        sigma = jnp.zeros(self.F.shape, dtype=complex)
        for i in range(self.num_contacts):
            sigma = sigma + self.sigma(E, i, conv)
        return sigma
