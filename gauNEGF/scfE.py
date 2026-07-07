"""
Energy-dependent extensions to the NEGF class for quantum transport calculations.

This module extends the base NEGF class to handle energy-dependent self-energies,
temperature effects, and advanced Fermi energy search methods. It provides support
for Bethe lattice and 1D chain contacts [1] with proper energy integration.

References
----------
.. [1] Jacob, D. & Palacios, J. J. Chem. Phys. 134, 044118 (2011) 
"""

# Python Packages
import numpy as np
import jax
import jax.numpy as jnp

# Enable double precision for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)

# Developed Packages
from gauNEGF.matTools import *
from gauNEGF.density import *
from gauNEGF.utils import inv, eig, eigh, inv_sqrt_general
from gauNEGF.config import (ETA, TEMPERATURE, ADAPTIVE_INTEGRATION_TOL, FERMI_CALCULATION_TOL, EMIN_BUFFER, ENERGY_MIN, FERMI_DEBUG, USE_INERTIA_EMIN)
from gauNEGF.scf import NEGF
from gauNEGF.surfG1D import surfG
from gauNEGF.surfGBethe import surfGB
from gauNEGF.surfGTester import surfGTest

# Matrix Headers
AlphaDen = "ALPHA DENSITY MATRIX"
BetaDen = "BETA DENSITY MATRIX"
AlphaSCFDen = "ALPHA SCF DENSITY MATRIX"
BetaSCFDen = "BETA SCF DENSITY MATRIX"
AlphaFock = "ALPHA FOCK MATRIX"
BetaFock = "BETA FOCK MATRIX"
AlphaMOs = "ALPHA MO COEFFICIENTS"
BetaMOs = "BETA MO COEFFICIENTS"
AlphaEnergies = "ALPHA ORBITAL ENERGIES"
BetaEnergies = "BETA ORBITAL ENERGIES"

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kB = 8.617e-5           # eV/Kelvin
V_to_au = 0.03675       # Volts to Hartree/elementary Charge


class NEGFE(NEGF):
    """
    Extended NEGF class with energy-dependent self-energies and temperature effects.

    This class extends the base NEGF implementation to handle:
    - Energy-dependent contact self-energies
    - Finite temperature effects
    - Advanced Fermi energy search methods
    - Integration methods for the density matrix
    
    Inherits all attributes and methods from NEGF class.
    """
    # Set energy dependent Bethe lattice contact using surfGB() object
    def setContactBethe(self, contactList, latFile='Au', eta=ETA, T=TEMPERATURE):
        """
        Set energy-dependent Bethe lattice contacts.

        Parameters
        ----------
        contactList : list
            List of atom indices for contacts
        latFile : str, optional
            Lattice parameter file (default: 'Au')
        eta : float, optional
            Broadening parameter in eV (default: 1e-9)
        T : float, optional
            Temperature in Kelvin (default: 0)

        Returns
        -------
        tuple
            Left and right contact orbital indices
        """
        # Set L/R contacts based on atom numbers, use orbital inds for surfG() object
        inds = super().setContacts(contactList[0], contactList[-1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        # Generate surfGB() object for Bethe lattice contacts
        self.g = surfGB(self.F*har_to_eV, self.S, contactList, self.bar, latFile, self.spin, eta, T)

        # Fit asymptotic Sigma and build S_eff / Y_eff once for the new Damle lower contour.
        self._initAsymptoticSigma()

        # Update other variables
        self.setIntegralLimits()
        self.T = T
        return inds

    # Set energy dependent 1D contact using surfG() object
    def setContact1D(self, contactList, tauList=None, stauList=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, neList=None, muList=None, eta=ETA, T=TEMPERATURE, symmetrize_contacts=None):
        """
        Set energy-dependent 1D chain contacts.

        Parameters
        ----------
        contactList : list
            List of atom indices for contacts
        tauList : list or None, optional
            Coupling matrices or atom indices (default: None)
        stauList : list or None, optional
            Overlap matrices (default: None)
        alphas : array_like or None, optional
            On-site energies (default: None)
        aOverlaps : array_like or None, optional
            On-site overlaps (default: None)
        betas : array_like or None, optional
            Hopping energies (default: None)
        bOverlaps : array_like or None, optional
            Hopping overlaps (default: None)
        neList : list or None, optional
            Number of electrons per unit cell (default: None)
        muList : list or None, optional
            fermiEnergy for each contact in eV (default: None)
        eta : float, optional
            Broadening parameter in eV (default: 1e-9)
        T : float, optional
            Temperature in Kelvin (default: 0)
        symmetrize_contacts : bool or None, optional
            Enforce identical on-site parameters for both contacts during SCF.
            True for same-material contacts, False for junctions.
            None (default) auto-detects based on setup pattern.

        Returns
        -------
        tuple
            Left and right contact orbital indices
        """
        # Set L/R contacts based on atom numbers, use orbital inds for surfG() object
        inds = super().setContacts(contactList[0], contactList[-1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        # Auto-detect symmetrization: True only when tauList not provided
        if symmetrize_contacts is not None:
            self._symmetrize_contacts = symmetrize_contacts
        else:
            self._symmetrize_contacts = (tauList is None)

        # One-time overlap check -- warn if contact overlap blocks differ
        if self._symmetrize_contacts:
            lInd, rInd = self.lInd, self.rInd
            SL = self.S[np.ix_(lInd, lInd)]
            SR = self.S[np.ix_(rInd, rInd)]
            if not np.allclose(np.array(SL), np.array(SR)):
                print(
                    'WARNING: symmetrize_contacts=True but contact overlap blocks differ. '
                    'Contacts may not be equivalent materials.'
                )
                print(np.max(abs(SL-SR)))
        # if tauList is a list of atom numbers (rather than a matrix), generate orbital indices
        if tauList is not None:
            if len(np.shape(tauList[0])) == 1:
                ind1 = np.where(np.isin(abs(self.locs), tauList[0]))[0]
                ind2 = np.where(np.isin(abs(self.locs), tauList[-1]))[0]
                tauList = (ind1, ind2)

        # Generate surfG() object for the molecule + contacts and initialize variables
        self.g = surfG(self.F*har_to_eV, self.S, inds, tauList, stauList, alphas, aOverlaps, betas, bOverlaps, eta, self.spin)

        if alphas is not None:
            gList = []
            for a, Sa, b, Sb in zip(self.g.aList, self.g.aSList, self.g.bList, self.g.bSList):
                inds = np.arange(len(a))
                gList.append(surfG(a, Sa, [inds, inds], [b, b.conj().T], [Sb, Sb.conj().T], eta=eta, spin=self.spin))
            if neList is not None:
                muL = getFermiContact(gList[0], neList[0], maxcycles=100)
                if self._symmetrize_contacts:
                    muR=muL+0.0
                else:
                    muR = getFermiContact(gList[-1], neList[-1], maxcycles=100)
            elif muList is not None:
                muL = muList[0]
                muR = muList[-1]
            else:  
                raise Exception('neList or muList must be defined!') 
            self.g.setF(self.g.F, muL, muR)

        # Fit asymptotic Sigma and build S_eff / Y_eff once for the new Damle lower contour.
        self._initAsymptoticSigma()

        # Update other variables
        self.setIntegralLimits()
        self.T = T
        return inds

    def _symmetrize_F(self):
        lInd = self.lInd
        rInd = self.rInd
        if getattr(self.g, 'contactFromFock', False):
            self.F[np.ix_(lInd, lInd)] = np.asarray(self.g.aList[0])/har_to_eV
            self.F[np.ix_(rInd, rInd)] = np.asarray(self.g.aList[-1])/har_to_eV
        if getattr(self, '_symmetrize_contacts', True):
            avg = (self.F[np.ix_(lInd, lInd)] + self.F[np.ix_(rInd, rInd)]) / 2
            self.F[np.ix_(lInd, lInd)] = avg
            self.F[np.ix_(rInd, rInd)] = avg

    def _initAsymptoticSigma(self):
        """Fit asymptotic Sigma(E) ~ Sigma_0 + E * X_asymp and build S_eff, Y_eff.

        Called ONCE per contact setup (from setContact1D / setContactBethe /
        setSigma) after self.g is available. Stores:

          self.Sigma_0   : (N, N) asymptotic constant part of contact self-energy
          self.X_asymp   : (N, N) asymptotic linear coefficient (Sigma' at E -> -inf)
          self.S_eff     : (N, N) complex effective overlap = S - X_asymp (non-Hermitian)
          self.Y_eff     : (N, N) S_eff^(-1/2) via general eig (inv_sqrt_general)
          self.damle_dN_warn : float, default 0.5 e. FockToP warns when any of
                              |tr(P@S)|, |delta_N|, or their sum exceeds it.

        Two-probe fit (see docs/lower_contour_math_and_probes.md). Sigma(E) =
        X_asymp*E + Sigma_0 + O(1/E) is linear only deep below the lead band, so
        the probes are placed DEEP and RELATIVE to the integration window -- never
        at Emin, where band-edge curvature would bias the slope X (which is global:
        it defines S_eff = S - X and hence Y_eff):

          E1 = ENERGY_MIN
          E2 = (Emin_est + ENERGY_MIN) / 2          (the midpoint)

        Emin_est is a cheap band-floor estimate from the device spectrum, since
        self.Emin is not set until setIntegralLimits runs after this. Because
        |ENERGY_MIN| >> |Emin|, the midpoint is ~ ENERGY_MIN/2 regardless, so both
        probes sit safely in the linear regime. Two points determine the two
        parameters (X_asymp, Sigma_0) exactly; the slope error from any residual
        1/E term scales as ~1/(E1*E2) and is negligible at this depth. No third
        probe / linearity residual is computed.

        X_asymp is in general NON-Hermitian (the retarded self-energy carries
        broadening), so S_eff = S - X_asymp is complex/non-Hermitian and
        Y_eff = S_eff^(-1/2) is built with a general (non-symmetric)
        eigendecomposition (inv_sqrt_general), NOT eigh.
        """
        # Cheap band-floor estimate (self.Emin is not set until setIntegralLimits
        # runs after this): min eig of X @ F @ X in eV, minus the standard buffer.
        # X @ F @ X (X = S^(-1/2)) is Hermitian by construction -- eigh is correct.
        # eig runs on jax (GPU); large-matrix eigendecompositions must not use numpy.
        F_eV = np.asarray(self.F) * har_to_eV
        S = np.asarray(self.S)  # consumed below at S_eff = S - X_asymp
        D_dev, _ = eigh(jnp.asarray(self.X) @ jnp.asarray(F_eV) @ jnp.asarray(self.X))
        Emin_est = float(np.real(np.asarray(D_dev)).min()) - EMIN_BUFFER

        # Deep probes, RELATIVE to the integration window [ENERGY_MIN, Emin].
        E1 = float(ENERGY_MIN)
        E2 = 0.5 * (Emin_est + float(ENERGY_MIN))
        Sig1 = np.asarray(self.g.sigmaTot(E1))
        Sig2 = np.asarray(self.g.sigmaTot(E2))

        # 2-point linear fit Sigma(E) = X_asymp*E + Sigma_0 (exact through both
        # probes). Intercept taken at the shallower probe E2 to limit the
        # subtractive cancellation in Sigma_0 = Sig - X*E.
        X_asymp = (Sig2 - Sig1) / (E2 - E1)
        Sigma_0 = Sig2 - X_asymp * E2

        # S_eff = S - X_asymp, kept fully complex. X_asymp is non-Hermitian (the
        # retarded self-energy carries broadening), so no Re()/symmetrization.
        # Y_eff = S_eff^(-1/2) via a general (non-symmetric) eigendecomposition,
        # NOT eigh -- eigh would assume Hermiticity and silently drop Im(X_asymp).
        S_eff = S - X_asymp
        Y_eff = np.asarray(inv_sqrt_general(jnp.asarray(S_eff)))

        self.Sigma_0 = Sigma_0
        self.X_asymp = X_asymp
        self.S_eff = S_eff
        self.Y_eff = Y_eff
        if not hasattr(self, 'damle_dN_warn'):
            # FockToP warns if ANY of |tr(P@S)|, |delta_N|, or their sum
            # (the lower electron count) exceeds this threshold (electrons).
            self.damle_dN_warn = 0.5

        # Diagnostics: non-Hermiticity / non-symmetry fractions of X_asymp
        # (should be small for near-Hermitian contacts; non-zero for SOC/GHF),
        # and the defining-property check Y_eff @ S_eff @ Y_eff = I (must be ~machine precision).
        if FERMI_DEBUG:
            Xn = max(np.linalg.norm(X_asymp), 1e-30)
            imag_frac = np.linalg.norm(np.imag(X_asymp)) / Xn
            herm_frac = np.linalg.norm(X_asymp - X_asymp.conj().T) / Xn
            symm_frac = np.linalg.norm(X_asymp - X_asymp.T) / Xn
            I_defect = np.linalg.norm(Y_eff @ S_eff @ Y_eff - np.eye(S_eff.shape[0]))
            print(f'Asymptotic Sigma fit (2-probe E1={E1:.2e}, E2={E2:.2e} eV): '
                  f'||Sigma_0||_F = {np.linalg.norm(Sigma_0):.3e}, '
                  f'||X_asymp||_F = {np.linalg.norm(X_asymp):.3e}')
            print(f'  X_asymp structure: ||Im X||/||X|| = {imag_frac:.3e}, '
                  f'||X-X^dag||/||X|| = {herm_frac:.3e} (non-Hermiticity), '
                  f'||X-X^T||/||X|| = {symm_frac:.3e} (non-symmetry)')
            print(f'  Y_eff @ S_eff @ Y_eff = I defect: {I_defect:.3e}')

    # Set constant sigma contact for testing or adding non-zero temperature
    def setSigma(self, lContact=None, rContact=None, sig=-0.1j, sig2=None, T=TEMPERATURE):
        """
        Set constant self-energy contacts with temperature.

        Parameters
        ----------
        lContact : list
            Atom numbers for left contact, all atoms if None (default: None)
        rContact : array-like
            Atom numbers for right contact, all atoms if None (default: None)
        sig : complex, optional
            Left contact self-energy (default: -0.1j)
        sig2 : complex or None, optional
            Right contact self-energy (default: None)
        T : float, optional
            Temperature in Kelvin (default: 0)

        Returns
        -------
        tuple
            Left and right contact orbital indices
        """
        super().setSigma(lContact, rContact, sig, sig2)
        inds = (self.lInd, self.rInd)
        self.g = surfGTest(self.F*har_to_eV, self.S, inds, sig, sig2, self.spin)

        # Fit asymptotic Sigma and build S_eff / Y_eff once for the new Damle lower contour.
        # For constant Sigma (surfGTest), X_asymp will be ~0 and S_eff == S.
        self._initAsymptoticSigma()

        # Update other variables
        self.setIntegralLimits()
        self.T = T
        return inds

    # Set up Fermi Search algorithm after setting system Fermi energies
    def setVoltage(self, qV, fermi=None, Emin=None, Eminf=None, fermiMethod=None):
        """
        Set voltage bias and Fermi search method.

        Parameters
        ----------
        qV : float
            Applied voltage in eV
        fermi : float, optional
            Fermi energy in eV (default: None)
        Emin : float, optional
            Minimum energy for integration (default: None)
        Eminf : float, optional
            Minimum energy for Fermi search (default: None)
        fermiMethod : str, optional
            Method for Fermi search: 'muller', 'secant', 'predict' or 'poly' 
            (default: 'muller')
        """
        super().setVoltage(qV, fermi, Emin, Eminf)
        self.g.setF(self.F*har_to_eV, self.mu1, self.mu2)
        # The setF call above shifts each contact's surface Green's function by
        # dFermi_i = mu_i - fermi0_i, which invalidates the cached Sigma_0.
        # X_asymp / S_eff / Y_eff are invariant under rigid-band shifts,
        # but we re-fit them too for simplicity -- cost is dominated by the
        # sigmaTot probes, which converge fast well below the band.
        self._initAsymptoticSigma()
        if self.mu1 != self.mu2 and self.N1 is not None:
            self.Nnegf=50 # Default grid
        if self.updFermi:
            if not hasattr(self, 'fermiMethod'):
                self.fermiMethod = 'muller' if fermiMethod is None else fermiMethod
            elif fermiMethod is not None:
                self.fermiMethod = fermiMethod
    
    def setIntegralLimits(self, N1=None, N2=None, Nnegf=None, tol=ADAPTIVE_INTEGRATION_TOL, Emin=None):

        """
        Set integration parameters for density calculation.

        Two modes:
          - Default (Emin is None, tol given): place Emin below the band via
            calcEmin (true-Sigma DOS loop) and pin the deep bound Eminf to the
            wide config value ENERGY_MIN (TSW unused, set None). calcTSW /
            calcPseudoPoleFloor are NOT on this path -- the lower contour is
            handled analytically by damleLowerDensity, whose cross-term warning
            in FockToP replaces calcTSW's dTSW<0 detection. Used at initial
            setup (called by setContact1D) so the first FockToP cycle has
            sensible bounds; FockToP refreshes these each SCF cycle thereafter.
          - Fixed-grid (N1/N2 given): DEPRECATED for production. The adaptive
            integration in FockToP replaces fixed-grid Emin/Eminf for the
            default tol path; N1/N2 is retained only for code paths that
            explicitly opt into fixed-point quadrature.

        Parameters
        ----------
        N1 : int, optional
            Number of points for complex contour (default: None)
        N2 : int, optional
            Number of points for real axis (default: None)
        Nnegf : int, optional
            Number of points for non-equilibrium integration (default: None)
        tol : float, optional
            Tolerance for integration (default: 1e-4)
        Emin : float or None, optional
            Minimum energy for integration (default: None)
        """
        if Emin is None and tol is not None:
            # Emin from the true-Sigma DOS loop (robust, incl. non-PSD S_eff);
            # the deep bound is the wide config value. calcTSW /
            # calcPseudoPoleFloor are NOT used on this path -- the lower contour
            # is handled analytically by damleLowerDensity, whose cross-term
            # delta_N warning (FockToP) replaces calcTSW's dTSW<0 detection.
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol=tol, X=self.X)
            self.Eminf = ENERGY_MIN
            self.TSW = None
            self.tol = tol
        else:
            self.Emin = Emin
        self.useInertiaEmin = getattr(self, 'useInertiaEmin', USE_INERTIA_EMIN)
        self.N1 = N1
        self.N2 = N2
        self.Nnegf = Nnegf
 
    def integralCheck(self, cycles=10, damp=0.02, pauseFermi=False):
        """
        Check and optimize integration parameters.

        Parameters
        ----------
        cycles : int, optional
            Number of SCF cycles to run (default: 10)
        damp : float, optional
            Damping parameter (default: 0.02)
        pauseFermi : bool, optional
            Whether to pause Fermi updates (default: False)
        """
        if self.updFermi:
            if pauseFermi:
                self.updFermi=False
            if cycles>0:
                print(f'RUNNING SCF FOR {cycles} CYCLES USING DEFAULT GRID: ')
                self.SCF(1e-10,damp,cycles)
            if pauseFermi:
                self.updFermi=True
        else:
            if cycles>0:
                print(f'RUNNING SCF FOR {cycles} CYCLES USING DEFAULT GRID: ')
                self.SCF(1e-10,damp,cycles)
        print('SETTING INTEGRATION LIMITS... ')
        self.Emin, self.N1, self.N2 = integralFit(self.F*har_to_eV, self.S, self.g,
                                                  self.fermi, self.Eminf, self.tol)
        PLower, delta_N_lower = densityRealN(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, self.T)
        nLower = np.trace(self.S@PLower).real + delta_N_lower
        if self.mu1 != self.mu2:
            self.Nnegf = integralFitNEGF(self.F*har_to_eV, self.S, self.g, self.fermi, 
                                         self.qV, self.Eminf, self.tol, self.T)
        if self.updFermi:
                print('CALCULATING FERMI ENERGY')
                ne = self.nae if self.spin == 'r' else self.nae+self.nbe
                self.fermi, dE, P = calcFermiSecant(self.g, ne-nLower, self.Emin, self.fermi, 
                                                    self.N1, tol=self.tol, maxcycles=20)
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                self.setVoltage(self.qV)
                self.P = P
        print('INTEGRATION LIMITS SET!')
        print('#############################')

    
    
    # Get left and right contact self-energies at specified energy
    def getSigma(self, E, E2=None):
        """
        Get contact self-energies at specified energy.

        Parameters
        ----------
        E : float
            Energy in eV

        Returns
        -------
        tuple
            (left_sigma, right_sigma) - Contact self-energies
        """
        if E2 is None:
            E2 = E
        return (self.g.sigma(E, 0), self.g.sigma(E2, -1))

    def spawnNEGF(self, mu1=None, mu2=None):
        if mu1 is None:
            mu1 = self.mu1
        if mu2 is None:
            mu2 = self.mu2
        sig1, sig2 = self.getSigma(mu1, mu2)
        negf = NEGF(self.ifile[:-4], self.basis, self.func, self.spin, False, 
                    self.otherRoute, self.section,len(self.pB)-1)
        negf.setSigma(self.lContact, self.rContact, 
            sig1[np.ix_(self.lInd, self.lInd)], 
            sig2[np.ix_(self.rInd, self.rInd)])
        negf.setDen(self.P)
        qV = mu1-mu2
        fermi = (mu1+mu2)/2
        if self.updFermi:
            negf.setVoltage(qV)
        else:
            negf.setVoltage(qV, fermi)
        return negf
        
    # Updated to use energy-dependent contour integral from surfG()
    def FockToP(self):
        """
        Calculate density matrix using energy-dependent integration.

        Performs complex contour integration for the equilibrium part
        and real axis integration for the non-equilibrium part.
        Updates Fermi energy using specified method if required.

        Returns
        -------
        tuple
            (energies, occupations) - Sorted jnp.linalg.eigenvalues and occupations
        """
        self._symmetrize_F()
        print('Calculating lower density matrix:')
        # Decide the inertia path FIRST, but FALL BACK to the Damle lower contour
        # on None -- never raise (the global constraint requires callers to detect
        # None and fall back, not crash). self.Emin is only mutated on the branch
        # that is actually taken, so a fallback leaves it for calcEmin to set.
        _use_inertia = (self.N2 is None) and getattr(self, 'useInertiaEmin', False)
        _Eminf = None
        if _use_inertia:
            F_eV = self.F * har_to_eV
            _Eminf = find_lower_bound(F_eV, self.S, self.g)
            if _Eminf is None:
                print("find_lower_bound returned None (non-PSD effective overlap "
                      "/ diffuse basis); falling back to the Damle lower-contour "
                      "path for this cycle.")
        if _use_inertia and _Eminf is not None:
            # SINGLE-CONTOUR inertia path: find_lower_bound gives the integration
            # floor directly. No separate lower contour: P starts empty and
            # nLower = 0, so the full contour [self.Emin, mu] is integrated by
            # compContourP2 below and the Fermi search targets the full count.
            self.Emin = _Eminf
            self.Eminf = self.Emin  # keep the predict-path / bisectFermi floor
                                    # (scfE.py:588,599) consistent with the contour
            P = np.zeros(np.shape(self.F), dtype=complex)
            nLower = 0.0
            print(f"Single-contour floor Emin = {self.Emin:.2f} eV; no lower contour.")
        elif self.N2 is None:
            # self.F changes every SCF cycle, and the asymptotic fit
            # (Sigma_0 / S_eff / Y_eff) AND Emin all depend on it -- recompute
            # them on the CURRENT Fock before building the lower contour.
            self._initAsymptoticSigma()
            # Emin is anchored to calcEmin's F-based default (eigh on X @ F @ X
            # minus EMIN_BUFFER), then deepened by the DOS loop using true Sigma(E).
            # A Fbar-based seed is avoided: Fbar's eigenvalues include spurious deep
            # modes from the constant-Sigma_0 approximation outside its asymptotic
            # regime. The F-based anchor depends only on the device spectrum.
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol=self.tol, X=self.X)
            # Lower contour [ENERGY_MIN, Emin] via analytic Damle: damleLowerDensity
            # returns the lower density matrix and the analytic
            # cross-term delta_N. The lower contour should hold ~zero charge --
            # Damle uses the constant asymptotic Sigma_0, valid only in the deep
            # tail. If any term -- the bulk tr(P@S), the cross-term delta_N, or
            # their sum (the lower electron count) -- exceeds damle_dN_warn, real
            # spectral weight sits in the lower contour (Emin too shallow or a
            # pseudo-pole), so Damle is being applied where its constant-Sigma_0
            # approximation is suspect -- warn (replaces calcTSW's dTSW<0
            # detection).
            F_eV = self.F * har_to_eV
            P, delta_N_lower = damleLowerDensity(F_eV, self.Y_eff, self.Sigma_0,
                                                 self.g, self.Emin)
            trLower = np.trace(self.S @ P).real
            nLower = trLower + delta_N_lower
            if (abs(trLower) > self.damle_dN_warn
                    or abs(delta_N_lower) > self.damle_dN_warn
                    or abs(nLower) > self.damle_dN_warn):
                print(f'WARNING: lower-contour holds significant weight at '
                      f'Emin={self.Emin:.2f} eV (tr(P@S)={trLower:.3e}, '
                      f'delta_N={delta_N_lower:.3e}, total={nLower:.3e}); a term '
                      f'exceeding {self.damle_dN_warn:.3e} means Emin is too shallow '
                      f'or a pseudo-pole sits in the lower contour (Damle assumes a '
                      f'negligible deep tail).')
        else:
            # DEPRECATED fixed-grid path. Retained for backward compatibility
            # only. Caller must supply explicit Emin via setIntegralLimits(
            # N2=..., Emin=...); this path uses self.Eminf (ENERGY_MIN) and
            # integrates over [-1e6, Emin] with a fixed point count -- numerically
            # poor. The default (N2=None) Damle path above is preferred.
            P, _delta_N_lower = densityComplexN(self.F*har_to_eV, self.S, self.g,
                                                self.Eminf, self.Emin, self.N2, T=0)
            nLower = np.trace(self.S @ P).real + _delta_N_lower

        # Helper function for densityComplex()
        def compContourP2(mu):
                if self.N1 is not None:
                    P, _ = densityComplexN(self.F*har_to_eV, self.S, self.g, self.Emin,
                                                    mu, N=self.N1, T=self.T)
                else:
                    P, _ = densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin,
                                                    mu, tol=self.tol, T=self.T)
                return P

        # Fermi Energy Update using local self-energy approximation
        if self.updFermi:
            fermi_old = self.fermi+0.0
            ne = self.bar.ne
            if self.spin =='r':
                ne /= 2
            conv= min(self.convLevel, FERMI_CALCULATION_TOL)
            if self.fermiMethod.lower() =='predict':
                # Generate inputs for energy-independent density calculation
                X = jnp.array(self.X)
                sig1, sig2 = self.getSigma(self.fermi)
                Fbar = X @ (self.F*har_to_eV + sig1 + sig2)@ X
                Gam1 = (sig1 - sig1.conj().T)*1j
                Gam2 = (sig2 - sig2.conj().T)*1j
                GamBar = X@(Gam1 + Gam2)@X
                D, V = eig(Fbar)
                Vc = inv(V.conj().T)

                # Number of electrons calculated assuming energy independent
                Ncurr = np.trace(density(V,Vc,D,GamBar,self.Eminf, self.fermi)).real
                
                dN = self.bar.ne - self.nelec
                # Account for factor of 2 for restricted case
                if self.spin=='r':
                    dN /= 2
                dN -= nLower
                #print('Nexp: ', self.bar.ne, ' Nact: ', self.nelec, ' Napprox: ', Ncurr, ' setpoint:', Ncurr+dN)  
                Nsearch = Ncurr + dN
                print('CONSTANT SELF-ENERGY APPROXIMATION:')
                if Nsearch > 0 and Nsearch < len(self.F):
                    self.fermi = bisectFermi(V, Vc, D, GamBar, Ncurr+dN, conv, self.Eminf)
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, shifting by {dN:.2E} electrons ')
                else:
                    print('Warning: Local sigma approximation not valid, Fermi energy not updated...')
                print('Calculating equilibrium density matrix:')
                P += compContourP2(self.mu1)

                # Fix number of electrons
                # TODO: nActual omits cross-term delta_N; acceptable since predict
                # uses constant-sigma approximation (delta_N << approximation error)
                nActual = np.real(np.trace(P @ self.S))
                P *= ne/nActual

            # Full integration methods (progession: muller/secant/poly --> bisect):
            methodFail = False
            uBound = None
            lBound = None
            if self.fermiMethod.lower() =='poly':
                print('POLYNOMIAL REGRESSION METHOD:')
                self.fermi, dE, P2, dN, uBound, lBound = calcFermiPolyFit(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=self.tol, conv=conv, T=self.T)
                print('Setting equilibrium density matrix...')
                methodFail = (dN > conv)
                if methodFail:
                    print(f'Switching to BISECT method (Fermi error = {dE:.2E} eV)')
                    fermi_old = self.fermi + 0.0
                else:
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                    P = P+P2 if self.mu1 == self.mu2 else compContourP2(self.mu1)
            
            if self.fermiMethod.lower() =='muller':
                print('MULLER METHOD:')
                self.fermi, dE, P2, dN, uBound, lBound = calcFermiMuller(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=self.tol, conv=conv, T=self.T)
                print('Setting equilibrium density matrix...') 
                methodFail = (dN > conv)
                if methodFail:
                    print(f'Switching to BISECT method (Fermi error = {dE:.2E} eV)')
                    fermi_old = self.fermi + 0.0
                else:
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                    P = P+P2 if self.mu1 == self.mu2 else compContourP2(self.mu1)

            if self.fermiMethod.lower() =='secant':
                print('SECANT METHOD:')
                self.fermi, dE, P2, dN = calcFermiSecant(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=self.tol, conv=conv, T=self.T)
                print('Setting equilibrium density matrix...') 
                methodFail = (dN > conv)
                if methodFail:
                    print(f'Switching to BISECT method (Fermi error = {dE:.2E} eV)')
                    fermi_old = self.fermi + 0.0
                else:
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                    P = P+P2 if self.mu1 == self.mu2 else compContourP2(self.mu1)

            if self.fermiMethod.lower() =='bisect' or methodFail:
                print('BISECT METHOD:')
                self.fermi, dE, P2 = calcFermiBisect(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=self.tol, conv=conv, T=self.T, uBound=uBound, lBound=lBound)  
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                print('Setting equilibrium density matrix...') 
                P = P+P2 if self.mu1 == self.mu2 else compContourP2(self.mu1)
            
            if self.fermiMethod.lower() not in ['muller', 'secant', 'bisect', 'predict', 'poly']:
                raise Exception('Error: invalid Fermi search method, needs to be \'muller\',' + \
                                                 '\'secant\', \'bisect\' or \'predict\' or \'default\'')
            # Shift Emin, mu1, and mu2 and update contact self-energies
            self.setVoltage(self.qV)
            self.g.setF(self.F*har_to_eV, self.mu1, self.mu2)
        else:
            print('Calculating equilibrium density matrix:')
            P+= compContourP2(self.mu1)
         
        # If bias applied, need to integrate G<
        if self.mu1 != self.mu2:
            print('Calculating non-equilibrium density matrix:')
            if self.Nnegf is not None:
                P += densityGridN(self.F*har_to_eV, self.S, self.g, self.mu1, self.mu2, ind=-1, 
                                    N=self.Nnegf, T=self.T)
            else:
                P += densityGrid(self.F*har_to_eV, self.S, self.g, self.mu1, self.mu2, ind=-1,
                                    tol=self.tol, T=self.T)

        # Calculate Level Occupation, Lowdin TF,  Return
        D,V = eigh(self.X@(self.F*har_to_eV)@self.X)
        self.Xi = inv(self.X)
        pshift = V.conj().T@(self.Xi@P@self.Xi)@ V
        self.P = P.copy()
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)

        return EList[inds], occList[inds]

    
    # Updated to update surfG() Fock matrix
    def PToFock(self):
        """
        Calculate new Fock matrix and update surfG object.

        Returns
        -------
        float
            Energy difference from previous iteration

        Raises
        ------
        RuntimeError
            If called before setVoltage() has defined mu1/mu2. setDen()
            triggers PToFock() via the base class, so on a fresh NEGFE the
            call order must be setVoltage(...) then setDen(...).
        """
        if not hasattr(self, 'mu1'):
            raise RuntimeError(
                'NEGFE.PToFock: mu1/mu2 undefined - call setVoltage() before '
                'setDen()/SCF() on a fresh NEGFE object (setDen -> PToFock -> '
                'g.setF(F, mu1, mu2) requires the chemical potentials).')
        Fock_old = self.F.copy()
        dE = super().PToFock()
        self.F, self.locs = getFock(self.bar, self.spin)
        self._symmetrize_F()
        self.g.setF(self.F*har_to_eV, self.mu1, self.mu2)
        return dE
    
    

