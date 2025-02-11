"""
Self-consistent field (SCF) implementation for Non-Equilibrium Green's Function calculations.

This module provides the base NEGF class for performing self-consistent DFT+NEGF
calculations using Gaussian quantum chemistry package. It implements the energy-independent
self-energy approach developed by Damle et al., which provides an efficient approximation
for molecular transport calculations. The module handles:
    - Integration with Gaussian for DFT calculations
    - SCF convergence with Pulay mixing [2]
    - Contact self-energy calculations
    - Voltage bias and electric field effects
    - Spin-polarized calculations

The implementation follows the standard NEGF formalism where the density matrix
is calculated self-consistently with the Fock matrix from Gaussian DFT calculations.
Convergence is accelerated using the direct inversion in the iterative subspace (DIIS)
method developed by Pulay [2]. The core NEGF-DFT implementation is based on the
ANT.Gaussian approach developed by Palacios et al. [3], which pioneered the integration
of NEGF with Gaussian-based DFT calculations.

References
----------
[1] Damle, P., Ghosh, A. W., & Datta, S. (2002). First-principles analysis of molecular 
    conduction using quantum chemistry software. Chemical Physics, 281(2-3), 171-187. 
    DOI: 10.1016/S0301-0104(02)00496-2

[2] Pulay, P. (1980). Convergence acceleration of iterative sequences. The case of SCF
    iteration. Chemical Physics Letters, 73(2), 393-398.
    DOI: 10.1016/0009-2614(80)80396-4

[3] Palacios, J. J., Pérez-Jiménez, A. J., Louis, E., & Vergés, J. A. (2002).
    Fullerene-based molecular nanobridges: A first-principles study.
    Physical Review B, 66(3), 035322.
    DOI: 10.1103/PhysRevB.66.035322
"""

# Python packages
import numpy as np
from scipy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy import io
import os
import time
import matplotlib.pyplot as plt

# Gaussian interface packages
from gauopen import QCBinAr as qcb

# Developed packages
from matTools import *
from density import * 


# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
V_to_au = 0.03675       # Volts to Hartree/elementary Charge

class NEGF(object):
    """
    Non-Equilibrium Green's Function calculator integrated with Gaussian DFT.

    This class implements the energy-independent NEGF approach developed by Damle et al. [1]
    for efficient molecular transport calculations. It manages the self-consistent field 
    calculation between NEGF transport calculations and DFT electronic structure calculations 
    using Gaussian.

    The energy-independent approximation assumes constant self-energies,
    which significantly reduces computational cost while maintaining accuracy for many
    molecular systems.

    The class handles:
    - Interaction with Gaussian
    - Management of density and Fock matrices
    - Pulay mixing for convergence [2]
    - Constant (energy-independent) contact self-energies
    - Voltage bias effects

    For energy-dependent calculations, see the NEGFE subclass.

    Parameters
    ----------
    fn : str
        Base filename for Gaussian input/output files (without extension)
    basis : str, optional
        Gaussian basis set name (default: 'lanl2dz')
    func : str, optional
        DFT functional to use (default: 'b3pw91')
    spin : {'r', 'u', 'ro', 'g'}, optional
        Spin configuration:
        - 'r': restricted
        - 'u': unrestricted
        - 'ro': restricted open-shell
        - 'g': generalized open-shell (non-collinear)
        (default: 'r')
    fullSCF : bool, optional
        Whether to run full SCF or use Harris approximation (default: True)
    route : str, optional
        Additional Gaussian route commands (default: '')
    nPulay : int, optional
        Number of previous iterations to use in Pulay mixing (default: 4)

    Attributes
    ----------
    F : ndarray
        Fock matrix in eV
    P : ndarray
        Density matrix
    S : ndarray
        Overlap matrix
    fermi : float
        Fermi energy in eV
    nelec : float
        Number of electrons

    References
    ----------
    [1] Damle, P., Ghosh, A. W., & Datta, S. (2002). First-principles analysis of molecular 
        conduction using quantum chemistry software. Chemical Physics, 281(2-3), 171-187.
        DOI: 10.1016/S0301-0104(02)00496-2

    [2] Pulay, P. (1980). DOI: 10.1016/0009-2614(80)80396-4

    """

    def __init__(self, fn, basis="lanl2dz", func="b3pw91", spin="r", fullSCF=True, route="", nPulay=4):
        """
        Initialize NEGF calculator and run initial DFT calculation.

        Sets up the calculator with specified parameters and runs an initial
        DFT calculation using Gaussian to obtain the starting Fock and 
        overlap matrices.

        Parameters are documented in class docstring.
        """
        # Set up variables
        self.ifile = fn + ".gjf"
        self.chkfile = fn + ".chk"
        self.ofile = fn + ".log"
        self.func = func
        self.basis = basis
        self.method= spin+func
        self.otherRoute = route     # Other commands that are needed in Gaussian
        self.spin = spin
        self.energyDep = False;
        self.Total_E_Old=9999.0;
        
        #Default Integration Limits
        self.Eminf = -1e5
        self.fSearch = None
        self.fermi = None
        self.updFermi = False
    
        # Start calculation: Load Initial Matrices from Gaussian
        print('Calculation started at '+str(time.asctime()))
        self.start_time = time.time()
        self.bar = qcb.BinAr(debug=False,lenint=8,inputfile=self.ifile)
        self.bar.write('debug.baf')
        self.runDFT(fullSCF)
        self.nae = int(self.bar.ne/2 + (self.bar.multip-1)/2)
        self.nbe = int(self.bar.ne/2 - (self.bar.multip-1)/2)

        # Prepare self.F, Density, self.S, and TF matrices
        self.P = getDen(self.bar, spin)
        self.F, self.locs = getFock(self.bar, spin)
        self.nsto = len(self.locs)
        Omat = np.array(self.bar.matlist["OVERLAP"].expand())
        if spin == "ro" or spin == "u":
            self.S = np.block([[Omat, np.zeros(Omat.shape)],[np.zeros(Omat.shape),Omat]])
        else:
            self.S = Omat
        self.X = np.array(fractional_matrix_power(self.S, -0.5))
        
        # Set Emin/Emax from orbitals
        orbs, _ = LA.eig(self.X@self.F@self.X)
        self.Emin = min(orbs.real)*har_to_eV - 5
        self.Emax = max(orbs.real)*har_to_eV
        self.convLevel = 9999
        self.MaxDP = 9999 

        # Pulay Mixing Initialization
        self.pList = np.array([self.P for i in range(nPulay)], dtype=complex)
        self.DPList = np.ones((nPulay, self.nsto, self.nsto), dtype=complex)*1e4
        self.pMat = np.ones((nPulay+1, nPulay+1), dtype=complex)*-1
        self.pMat[-1, -1] = 0
        self.pB = np.zeros(nPulay+1)
        self.pB[-1] = -1
        
        # DFT Info dump
        print("ORBS:")
        print(self.locs)
        self.Total_E =  self.bar.scalar("escf")
        self.updateN()
        print('Expecting', str(self.bar.ne), 'electrons')
        print('Actual: ', str(self.nelec), 'electrons')
        print('Charge is:', self.bar.icharg)
        print('Multiplicity is:', self.bar.multip)
        print("Initial SCF energy: ", self.Total_E)
        print('###################################')
    
    def runDFT(self, fullSCF=True):
        """
        Run DFT calculation using Gaussian.

        Performs either a full SCF calculation or generates an initial Harris
        guess using Gaussian. Updates the Fock matrix and orbital indices
        after completion.

        Parameters
        ----------
        fullSCF : bool, optional
            If True, runs full SCF to convergence.
            If False, uses Harris guess only. (default: True)

        Notes
        -----
        - Attempts to load from checkpoint file first
        - Falls back to full calculation if checkpoint fails
        - Updates self.F and self.locs with new Fock matrix
        """
        if fullSCF:
            try:
                self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock=True,chkname=self.chkfile, miscroute=self.otherRoute)
                print('Checking '+self.chkfile+' for saved data...');
            except:
                print('Checkpoint not loaded, running full SCF...');
                self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="scf",chkname=self.chkfile, miscroute=self.otherRoute)
        
            print("Done!")
            self.F, self.locs = getFock(self.bar, self.spin)
            
        else:
            print('Using default Harris DFT guess to initialize...')
            self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="GUESS",chkname=self.chkfile, miscroute=self.otherRoute)
            self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock=True, miscroute=self.otherRoute)
            print("Done!")
            self.F, self.locs = getFock(self.bar, self.spin)
    
    def updateN(self):
        """
        Update the total number of electrons from the density matrix.

        Calculates the number of electrons by tracing the product of
        the density and overlap matrices. For restricted calculations,
        multiplies by 2 to account for spin degeneracy.

        Returns
        -------
        float
            Total number of electrons
        """
        nOcc =  np.real(np.trace(self.P@self.S))
        if self.spin == 'r':
            self.nelec = 2*nOcc
        else:
            self.nelec = nOcc
        return self.nelec

    def setFock(self, F_):
        """
        Set the Fock matrix, converting from eV to atomic units.

        Parameters
        ----------
        F_ : ndarray
            Fock matrix in Hartree units
        """
        self.F = np.array(F_)/har_to_eV

    def setDen(self, P_):
        """
        Set the density matrix and update dependent quantities.

        Updates the density matrix, stores it in Gaussian format,
        recalculates the number of electrons, and updates the Fock matrix.

        Parameters
        ----------
        P_ : ndarray
            New density matrix
        """
        self.P = P_ 
        storeDen(self.bar, self.P, self.spin)
        self.updateN() 
        print(f'Density matrix loaded, nelec = {self.nelec:.2f} electrons')
        self.PToFock()

    def getHOMOLUMO(self):
        """
        Calculate HOMO and LUMO energies.

        Diagonalizes the Fock matrix in orthogonalized basis to get
        orbital energies, then identifies HOMO and LUMO based on
        electron occupation and spin configuration.

        Returns
        -------
        ndarray
            Array of [HOMO, LUMO] energies in eV
        """
        orbs, _ = LA.eig(self.X@self.F@self.X)
        orbs = np.sort(orbs)*har_to_eV
        if self.spin=='r':
            homo_lumo = orbs[self.nae-1:self.nae+1].real
        else:
            homo_lumo = orbs[self.nae+self.nbe-1:self.nae+self.nbe+1].real
        return homo_lumo
                
    def setVoltage(self, qV, fermi=np.nan, Emin=None, Eminf=None):
        """
        Set voltage bias and Fermi energy, updating electric field.

        Applies a voltage bias between contacts and updates the chemical
        potentials and electric field. Can optionally set the Fermi energy
        and integration limits.

        Parameters
        ----------
        qV : float
            Voltage bias in eV
        fermi : float, optional
            Fermi energy in eV. If not provided, will be calculated or
            use existing value (default: np.nan)
        Emin : float, optional
            Minimum energy for integration in eV (default: None)
        Eminf : float, optional
            Lower bound energy in eV (default: None)

        Notes
        -----
        - Requires contacts to be set first
        - Updates chemical potentials as fermi ± qV/2
        - Calculates and applies electric field between contacts
        - If fermi not provided, uses (HOMO+LUMO)/2 for initial guess
        """
        # Check to make sure contacts set 
        assert hasattr(self, 'rInd') and hasattr(self,'lInd'), "Contacts not set!"

        # Set Fermi Energy
        if np.isnan(fermi):
            self.updFermi = True
            if self.fermi is None:
                # Set initial fermi energy as (HOMO + LUMO)/2
                homo_lumo = self.getHOMOLUMO()
                print(f'Setting initial Fermi energy between HOMO ({homo_lumo[0]:.2f} eV) and LUMO ({homo_lumo[1]:.2f} eV)')
                fermi = sum(homo_lumo)/2
            else:
                fermi = self.fermi
        else:
            self.updFermi = False
        
        # Set Integration limits
        if Emin!=None:
            self.Emin = Emin
        if Eminf!=None:
            self.Eminf = Eminf

        self.fermi = fermi
        self.qV = qV
        self.mu1 =  fermi + (qV/2)
        self.mu2 =  fermi - (qV/2)
    
        # Calculate electric field to apply during SCF
        lAtom = self.bar.c[int(self.lContact[0]-1)*3:int(self.lContact[0])*3]
        rAtom = self.bar.c[int(self.rContact[0]-1)*3:int(self.rContact[0])*3]
        vec  = np.array(lAtom-rAtom)
        dist = LA.norm(vec)
        vecNorm = vec/dist
        
        if dist == 0:
            print("WARNING: left and right contact atoms identical, E-field set to zero!")
            field = [0,0,0]
        else:
            field = -1*vecNorm*qV*V_to_au/(dist*0.0001)
        self.bar.scalar("X-EFIELD", int(field[0]))
        self.bar.scalar("Y-EFIELD", int(field[1]))
        self.bar.scalar("Z-EFIELD", int(field[2]))
        if not self.updFermi:
            print("E-field set to "+str(LA.norm(field))+" au")
    
    def setContacts(self, lContact=-1, rContact=-1):
        """
        Set contact atoms and get corresponding orbital indices.

        Identifies the orbital indices corresponding to the specified
        contact atoms. If no contacts specified, uses all orbitals.

        Parameters
        ----------
        lContact : int or array-like, optional
            Atom number(s) for left contact. -1 means all atoms. (default: -1)
        rContact : int or array-like, optional
            Atom number(s) for right contact. -1 means all atoms. (default: -1)

        Returns
        -------
        tuple of ndarrays
            (left_orbital_indices, right_orbital_indices)
        """
        if lContact == -1:
            self.lContact = np.arange(self.nsto)
        else:
            self.lContact=np.array(lContact)
        if rContact == -1:
            self.rContact = np.arange(self.nsto)
        else:
            self.rContact=np.array(rContact)
        lInd = np.where(np.isin(abs(self.locs), self.lContact))[0]
        rInd = np.where(np.isin(abs(self.locs), self.rContact))[0]
        contInds = list(lContact) + list(rContact)
        self.nelecContacts = sum([self.bar.atmchg[i-1] for i in contInds])
        return lInd, rInd
    
    # Set self-energies of left and right contacts (TODO: n>2 terminal device?)
    def setSigma(self, lContact, rContact, sig=-0.1j, sig2=None): 
        """
        Set self-energies for left and right contacts.

        Configures the contact self-energies, handling various input formats
        and spin configurations. Self-energies can be scalar, vector, or matrix,
        with automatic handling of spin degrees of freedom.

        Parameters
        ----------
        lContact : array-like
            Atom numbers for left contact
        rContact : array-like
            Atom numbers for right contact
        sig : scalar or array-like, optional
            Self-energy for left contact. Can be:
            - scalar: same value for all orbitals
            - vector: one value per orbital
            - matrix: full self-energy matrix
            (default: -0.1j)
        sig2 : scalar, array-like, or None, optional
            Self-energy for right contact. If None, uses sig.
            Same format options as sig. (default: None)

        Notes
        -----
        - Handles spin configurations ('r', 'u', 'ro', 'g')
        - Automatically expands scalar/vector inputs to full matrices
        - Verifies matrix dimensions match Fock matrix
        - Updates self.sigma1, self.sigma2, self.sigma12
        
        Raises
        ------
        Exception
            If matrix dimensions don't match or invalid input format
        """
        lInd, rInd = self.setContacts(lContact, rContact)
        #Is there a second sigma matrix? If not, copy the first one
        if sig2 is None:
            sig2 = sig + 0.0
       
        # Sigma can be a value, list, or matrix
        if np.ndim(np.array(sig)) == 0  and np.ndim(np.array(sig2)) == 0:
            pass
        elif np.ndim(sig) == 1 and np.ndim(sig2) == 1:
            if len(sig) == len(lInd) and len(sig2) == len(rInd):
                pass
            elif len(sig) == len(lInd)/2 and len(sig2) == len(rInd)/2:
                if self.spin=='g':
                    sig = np.kron(sig, [1, 1])
                    sig2 = np.kron(sig2, [1, 1])
                elif self.spin=='ro' or self.spin=='u':
                    sig = np.kron([1, 1], sig)
                    sig2 = np.kron([1, 1], sig2)
            else:
                raise Exception('Sigma matrix dimension mismatch!')
        elif np.ndim(sig) == 2 and np.ndim(sig2) == 2:
            if len(sig) == len(lInd) and len(sig2) == len(rInd):
                pass
            elif len(sig) == len(rInd)/2 and len(sig2) == len(rInd)/2:
                if self.spin=='g':
                    sig = np.kron(sig, np.eye(2))
                    sig2 = np.kron(sig2, np.eye(2))
                elif self.spin=='ro' or self.spin=='u':
                    sig = np.kron(np.eye(2), sig)
                    sig2 = np.kron(np.eye(2), sig2)
            else:
                raise Exception('Sigma matrix dimension mismatch!')
            
        else:
            raise Exception('Sigma matrix dimension mismatch!')
        
        # Store Variables
        self.rInd = rInd
        self.lInd = lInd 
        self.sigma1 = formSigma(lInd, sig, self.nsto, self.S)
        self.sigma2 = formSigma(rInd, sig2, self.nsto, self.S)
        
        if self.sigma1.shape != self.F.shape or self.sigma2.shape != self.F.shape:
            raise Exception(f'Sigma size mismatch! F shape={self.F.shape},'+
                            f' sigma shapes={self.sigma1.shape}, {self.sigma2.shape}')
        
        self.sigma12 = self.sigma1 + self.sigma2
    
        print('Max imag sigma:', str(np.max(np.abs(np.imag(self.sigma12)))));
        self.Gam1 = (self.sigma1 - self.sigma1.conj().T)*1j
        self.Gam2 = (self.sigma2 - self.sigma2.conj().T)*1j
        
    def getSigma(self, E=0): #E only used by NEGFE() object, function inherited
        return (self.sigma1, self.sigma2)

    # Calculate density matrix from stored Fock matrix
    def FockToP(self):
        """
        Calculate density matrix from Fock matrix using energy-independent approach.

        This method implements the energy-independent density matrix calculation from
        Damle et al. (2002). By assuming constant self-energies, the density matrix
        can be calculated analytically without energy integration, significantly
        reducing computational cost.

        The method:
        1. Transforms Fock and Gamma matrices to orthogonal basis
        2. Diagonalizes the transformed Fock matrix
        3. Updates Fermi energy if needed
        4. Calculates density matrix analytically
        5. Transforms back to non-orthogonal basis

        Returns
        -------
        tuple
            (eigenvalues, occupations) sorted by energy

        References
        ----------
        [1] Damle, P., Ghosh, A. W., & Datta, S. (2002). First-principles analysis of molecular 
            conduction using quantum chemistry software. Chemical Physics, 281(2-3), 171-187.
            DOI: 10.1016/S0301-0104(02)00496-2
        """
        # Prepare Variables for Analytical Integration
        X = np.array(self.X)
        self.F, self.locs = getFock(self.bar, self.spin)
        Fbar = X @ (self.F*har_to_eV + self.sigma12) @ X
        GamBar1 = X @ self.Gam1 @ X
        GamBar2 = X @ self.Gam2 @ X
        
        
        D,V = LA.eig(np.array(Fbar))
        Vc = LA.inv(V.conj().T)
          
                
        #Update Fermi Energy (if not constant)
        if self.updFermi:
            Nexp = self.bar.ne
            conv = min(self.convLevel, 1e-3)
            if self.spin=='r':
                Nexp/=2
            self.fermi = bisectFermi(V,Vc,D,GamBar1+GamBar2,Nexp,conv, self.Eminf)
            self.setVoltage(self.qV)
            print(f'Fermi Energy set to {self.fermi:.2f} eV')

        #Integrate to get density matrix
        if self.mu1 != self.mu2:
            P = density(V, Vc, D, GamBar1+GamBar2, self.Eminf, self.fermi)
        else:
            P1 = density(V, Vc, D, GamBar1, self.Eminf, self.mu1)
            P2 = density(V, Vc, D, GamBar2, self.Eminf, self.mu2)
            P = P1 + P2 #+ Pw
        
        # Calculate Level Occupation, Lowdin TF,  Return
        pshift = V.conj().T @ P @ V
        self.P = X@P@X
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)
        
        #DEBUG:
        #for pair in zip(occList[inds], EList[inds]):                       
        #    print("Energy=", str(pair[1]), ", Occ=", str(pair[0]))

        return EList[inds], occList[inds]

    def PMix(self, damping, Pulay=False):
        """
        Mix old and new density matrices using damping or Pulay DIIS method [2].

        The Pulay mixing method (also known as DIIS - Direct Inversion in the 
        Iterative Subspace) uses information from previous iterations to predict
        the optimal density matrix. This method is particularly effective for
        systems with challenging convergence behavior, and closely follows ANT.Gaussian approaches.

        Parameters
        ----------
        damping : float
            Mixing parameter between 0 and 1
        Pulay : bool, optional
            Whether to use Pulay mixing (default: False)

        Returns
        -------
        tuple
            (RMSDP, MaxDP) - RMS and maximum density matrix differences

        Notes
        -----
        The Pulay DIIS method [2] minimizes the error in the iterative subspace
        spanned by previous density matrices. This often provides faster and more
        stable convergence compared to simple damping, especially for systems
        with strong electron correlation or near degeneracies [3].

        References
        ----------
        .. [2] Pulay, P. (1980). DOI: 10.1016/0009-2614(80)80396-4
        .. [3] Palacios, J. J., et al. (2002). DOI: 10.1103/PhysRevB.66.035322
        """
        # Store Old Density Info
        Pback = getDen(self.bar, self.spin)
        Dense_old = np.diag(Pback)
        Dense_diff = abs(np.diag(self.P) - Dense_old)
        self.pList[1:, :, :] = self.pList[:-1, :, :]
        self.pList[0,  :, :] = Pback + damping*(self.P - Pback)
        self.DPList[1:, :, :] = self.DPList[:-1, :, :]
        self.DPList[0,  :, :] = self.P - Pback
        
        # Pulay Mixing
        for i, v1 in enumerate(self.DPList):
            for j, v2 in enumerate(self.DPList):
                self.pMat[i,j] = np.sum(v1*v2)
       
        # Apply Damping, store to Gaussian matrix
        if Pulay:
            coeff = LA.solve(self.pMat, self.pB)[:-1]
            print("Applying Pulay Coeff: ", coeff)
            self.P = sum([self.pList[i, :, :]*coeff[i] for i in range(len(coeff))])
            self.pList[0, :, :] = self.P
        else:
            print("Applying Damping value=", damping)
            self.P = self.pList[0, :, :]
        storeDen(self.bar, self.P, self.spin)
        
        # Update counters, print data
        self.updateN() 
        print(f'Total number of electrons (NEGF): {self.nelec:.2f}')
        self.MaxDP = max(Dense_diff)
        RMSDP = np.sqrt(np.mean(Dense_diff**2))
        print(f'MaxDP: {self.MaxDP:.2E} | RMSDP: {RMSDP:.2E}')
        return RMSDP, self.MaxDP

    # Use Gaussian to calculate the density matrix 
    def PToFock(self):
        """
        Calculate new Fock matrix from current density matrix using Gaussian.

        Returns
        -------
        float
            Energy difference from previous iteration
        """
        # Run Gaussian, update SCF Energy
        try:
            self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="DENSITY", miscroute=self.otherRoute)
        except Exception as e:
            print("WARNING: DFT METHOD HAD AN ERROR, CYCLE INVALID:")
            print(e)
            print("CONTINUING TO NEXT CYCLE...")
        self.Total_E_Old = self.Total_E.copy()
        self.Total_E  = self.bar.scalar("escf")
        print("SCF energy: ", self.Total_E)

        # Convergence variables: dE, RMSDP and MaxDP
        dE = self.Total_E-self.Total_E_Old
        print(f'Energy difference is: {dE:.3E}')
        return dE

    # Main SCF loop, runs Fock <-> Density cycle until convergence reached
    # Convergence criteria: dE, RMSDP, and MaxDP < conv, or maxcycles reached
    def SCF(self, conv=1e-5, damping=0.1, maxcycles=100, checkpoint=True, pulay=True):
        """
        Run self-consistent field calculation until convergence.

        The SCF cycle alternates between Fock matrix construction and density matrix
        updates until convergence is reached. Convergence acceleration is achieved
        through either simple damping or Pulay DIIS mixing [2]. The Pulay method
        is applied every nPulay iterations (where nPulay is set in __init__) [3].

        References
        ----------
        .. [3] Palacios, J. J., et al. (2002). DOI: 10.1103/PhysRevB.66.035322

        Parameters
        ----------
        conv : float, optional
            Convergence criterion for energy and density (default: 1e-5)
        damping : float, optional
            Mixing parameter between 0 and 1 (default: 0.1)
        maxcycles : int, optional
            Maximum number of SCF cycles (default: 100)
        checkpoint : bool, optional
            Save density matrix at each iteration and load if job interrupted (default: True)
        pulay : bool, optional
            Whether to use Pulay DIIS mixing [2] (default: True)

        Returns
        -------
        tuple
            (count, PP, TotalE) - cycle number, number of electrons, and DFT energy at each cycle

        Notes
        -----
        Convergence is determined by three criteria:
        1. Energy change (dE)
        2. RMS density matrix difference (RMSDP)
        3. Maximum density matrix difference (MaxDP)
        
        All three must be below the convergence threshold.
        
        The Pulay DIIS method [2] is applied every nPulay iterations when enabled,
        which often provides faster and more stable convergence compared to simple
        damping, especially for challenging systems.

        References
        ----------
        .. [2] Pulay, P. (1980). DOI: 10.1016/0009-2614(80)80396-4
        """
        # Check to make sure contacts and voltage set 
        assert hasattr(self, 'mu1') and hasattr(self, 'mu2'), "Voltage not set!"
        assert hasattr(self, 'rInd') and hasattr(self,'lInd'), "Contacts not set!"
        
        # Find saved data from midrun
        checkpoint_file = self.ifile[:-4]+"_P.mat"
        final_file = self.ifile[:-4]+"_Final.mat"
        if os.path.exists(checkpoint_file) and checkpoint:
            print(f"Found checkpoint file {checkpoint_file}, loading...")
            self.setDen(io.loadmat(checkpoint_file)['den'])

        #Main SCF Loop
        Loop = True
        Niter = 0
        PP=[]
        count=[]
        TotalE=[]
        print('Entering NEGF-SCF loop at: '+str(time.asctime()))
        print('###################################')

        while Loop:
            print()
            print('Iteration '+str(Niter)+':')
            # Run Pulay Kick every nPulay iterations, if turned on
            isPulay = pulay*((Niter+1)%(len(self.pList)+1)==0)
           
            # Fock --> P --> Fock
            EList, occList = self.FockToP()
            RMSDP, MaxDP = self.PMix(damping, isPulay)
            dE = self.PToFock()
            
            # Write monitor variables
            TotalE.append(self.Total_E)
            count.append(Niter)
            PP.append(self.nelec)
            
            # Check 3 convergence criteria
            self.convLevel = max(RMSDP, MaxDP, abs(dE))
            if self.convLevel<conv:
                print('##########################################')
                print('Convergence achieved after '+str(Niter)+' iterations!')
                Loop = False
            elif Niter >= maxcycles:
                print('##########################################')
                print('WARNING: Convergence criterion not met, maxcycles reached!')
                Loop = False

            # Save progress
            if checkpoint:
                io.savemat(checkpoint_file, {'den':self.P}) 
            Niter += 1

        if checkpoint:
            os.system(f'mv {checkpoint_file} {final_file}') 
        print("--- %s seconds ---" % (time.time() - self.start_time))
        print('')
        print('SCF Loop exited at', time.asctime())
        
        homo_lumo = self.getHOMOLUMO()
        print(f'Predicted HOMO: {homo_lumo[0]:.2f} eV , Predicted LUMO {homo_lumo[1]:.2f} eV, Fermi: {self.fermi:0.2f}')
 
        print('=========================')
        print('ENERGY LEVEL OCCUPATION:')
        print('=========================')
        for pair in zip(occList, EList):
            print(f"Energy = {pair[1]:9.3f} eV | Occ = {pair[0]:5.3f}")
        print('=========================')
        return count, PP, TotalE

    def writeChk(self):
        """
        Write current state to Gaussian checkpoint file.
        """
        print('Writing to checkpoint file...') 
        self.bar.writefile(self.chkfile)
        print(self.chkfile+' written!') 
    
    def saveMAT(self, matfile="out.mat"):
        """
        Save calculation results to MATLAB format file.

        Parameters
        ----------
        matfile : str, optional
            Output filename (default: "out.mat")

        Returns
        -------
        ndarray
            Fock matrix in orthogonalized basis
        """
        (sigma1, sigma2) = self.getSigma(self.fermi)
        # Save data in MATLAB .mat file
        matdict = {"F":self.F*har_to_eV, "sig1": sigma1, "sig2": sigma2, 
                  "S": self.S, "fermi": self.fermi, "qV": self.qV, 
                  "spin": self.spin, "den": self.P, "conv": self.convLevel}
        io.savemat(matfile, matdict)
        return self.X@self.F@self.X

