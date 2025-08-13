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
from tkinter import N
import numpy as np
from numpy import linalg as LA

# Gaussian interface packages
from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

# Developed Packages 
from gauNEGF.matTools import *
from gauNEGF.density import *
from gauNEGF.transport import DOS
from gauNEGF.fermiSearch import DOSFermiSearch
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
    def setContactBethe(self, contactList, latFile='Au', eta=1e-9, T=0):
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
        
        # Update other variables
        self.setIntegralLimits()
        self.T = T
        return inds

    # Set energy dependent 1D contact using surfG() object
    def setContact1D(self, contactList, tauList=None, stauList=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, neList=None, eta=1e-9, T=0):
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
        # if tauList is a list of atom numbers (rather than a matrix), generate orbital indices
        if tauList is not None:
            if len(np.shape(tauList[0])) == 1: 
                ind1 = np.where(np.isin(abs(self.locs), tauList[0]))[0]
                ind2 = np.where(np.isin(abs(self.locs), tauList[-1]))[0]
                tauList = (ind1, ind2)

        # Generate surfG() object for the molecule + contacts and initialize variables
        self.g = surfG(self.F*har_to_eV, self.S, inds, tauList, stauList, alphas, aOverlaps, betas, bOverlaps, eta)
        
        if alphas is not None:
            muL = getFermi1DContact(self.g, neList[0], 0)
            muR = getFermi1DContact(self.g, neList[-1], -1)
            self.g.setF(self.g.F, muL, muR)
        # Update other variables
        self.setIntegralLimits()
        self.T = T
        return inds
   
    # Set constant sigma contact for testing or adding non-zero temperature
    def setSigma(self, lContact=None, rContact=None, sig=-0.1j, sig2=None, T=0):
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
        self.g = surfGTest(self.F*har_to_eV, self.S, inds, sig, sig2)
        
        # Update other variables
        self.setIntegralLimits()
        self.T = T
        return inds

    # Set up Fermi Search algorithm after setting system Fermi energies
    def setVoltage(self, qV, fermi=np.nan, Emin=None, Eminf=None, fermiMethod='muller'):
        """
        Set voltage bias and Fermi search method.

        Parameters
        ----------
        qV : float
            Applied voltage in eV
        fermi : float, optional
            Fermi energy in eV (default: np.nan)
        Emin : float, optional
            Minimum energy for integration (default: None)
        Eminf : float, optional
            Minimum energy for Fermi search (default: None)
        fermiMethod : str, optional
            Method for Fermi search: 'muller', 'secant', or 'predict' (default: 'muller')
        """
        super().setVoltage(qV, fermi, Emin, Eminf)
        if self.mu1 != self.mu2:
            self.Nnegf=50 # Default grid
        if self.updFermi:
            self.fermiMethod = fermiMethod
    
    def setIntegralLimits(self, N1=None, N2=None, Nnegf=None, tol=1e-4, Emin=None):
        """
        Set integration parameters for density calculation.

        Parameters
        ----------
        N1 : int
            Number of points for complex contour
        N2 : int
            Number of points for real axis
        Emin : float or None, optional
            Minimum energy for integration (default: None)
        """
        if self.Emin is None and tol is not None:
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol, 1000)
        else:
            self.Emin = Emin
        self.tol = tol
        self.N1 = N1
        self.N2 = N2
        self.Nnegf = Nnegf
 
    def integralCheck(self, cycles=10, damp=0.02, pauseFermi=False):
        """
        Check and optimize integration parameters.

        Parameters
        ----------
        tol : float, optional
            Tolerance for integration (default: 1e-4)
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
        PLower = densityRealN(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, self.T)
        nLower = np.trace(self.S@PLower).real
        if self.mu1 != self.mu2:
            self.Nnegf = integralFitNEGF(self.F*har_to_eV, self.S, self.g, self.fermi, 
                                         self.qV, self.Eminf, self.tol, self.T)
        if self.updFermi:
                print('CALCULATING FERMI ENERGY')
                ne = self.nae if self.spin is 'r' else self.nae+self.nbe
                self.fermi, dE, P = calcFermiSecant(self.g, ne-nLower, self.Emin, self.fermi, 
                                                    self.N1, tol=self.tol, maxcycles=20)
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                self.setVoltage(self.qV, fermiMethod=self.fermiMethod)
                self.P = P
        print('INTEGRATION LIMITS SET!')
        print('#############################')

    
    
    # Get left and right contact self-energies at specified energy
    def getSigma(self, E):
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
        return (self.g.sigma(E, 0), self.g.sigma(E, -1))

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
            (energies, occupations) - Sorted eigenvalues and occupations
        """
        print('Calculating lower density matrix:') 
        P = densityReal(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, T=0)
        nLower = np.trace(self.S@P).real 

        # Fermi Energy Update using local self-energy approximation
        if self.updFermi:
            fermi_old = self.fermi+0.0
            conv= min(self.convLevel, 1e-3)
            if self.fermiMethod.lower() =='predict':
                # Generate inputs for energy-independent density calculation
                X = np.array(self.X)
                sig1, sig2 = self.getSigma(self.fermi)
                Fbar = X@(self.F*har_to_eV + sig1 + sig2)@X
                Gam1 = (sig1 - sig1.conj().T)*1j
                Gam2 = (sig2 - sig2.conj().T)*1j
                GamBar = (X@Gam1@X)+(X@Gam2@X)
                D, V = LA.eig(Fbar)
                Vc = LA.inv(V.conj().T)

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
                if self.N1 is not None:
                    P += densityComplexN(self.F*har_to_eV, self.S, self.g, self.Emin, self.fermi, N=self.N1, T=self.T)
                else:
                    P += densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.fermi, tol=self.tol, T=self.T)

            # Full integration methods (progession: muller/secant --> bisect):
            methodFail = False
            if self.fermiMethod.lower() =='muller':
                ne = self.bar.ne
                if self.spin =='r':
                    ne /= 2
                print('MULLER METHOD:')
                self.fermi, dE, P2 = calcFermiMuller(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=conv, T=self.T)
                print('Setting equilibrium density matrix...') 
                methodFail = (abs(dE) > conv)
                if methodFail:
                    print(f'Switching to BISECT method (Fermi error = {dE:.2E} eV)')
                    fermi_old = self.fermi + 0.0
                else:
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                    P += P2

            if self.fermiMethod.lower() =='secant':
                ne = self.bar.ne
                if self.spin =='r':
                    ne /= 2
                print('SECANT METHOD:')
                self.fermi, dE, P2 = calcFermiSecant(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=self.tol, conv=conv, T=self.T)
                print('Setting equilibrium density matrix...') 
                methodFail = (abs(dE) > conv)
                if methodFail:
                    print(f'Switching to BISECT method (Fermi error = {dE:.2E} eV)')
                    fermi_old = self.fermi + 0.0
                else:
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                    P += P2

            if self.fermiMethod.lower() =='bisect' or methodFail:
                ne = self.bar.ne
                if self.spin =='r':
                    ne /= 2
                print('BISECT METHOD:')
                self.fermi, dE, P2 = calcFermiBisect(self.g, ne-nLower, self.Emin, fermi_old, 
                                            self.N1, tol=self.tol, conv=conv, T=self.T)  
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                print('Setting equilibrium density matrix...') 
                P += P2
            
            if self.fermiMethod.lower() not in ['muller', 'secant', 'bisect', 'predict']:
                raise Exception('Error: invalid Fermi search method, needs to be \'muller\',' + \
                                                 '\'secant\', \'bisect\' or \'predict\'')
            # Shift Emin, mu1, and mu2 and update contact self-energies
            self.setVoltage(self.qV, fermiMethod=self.fermiMethod)
            self.Emin += self.fermi-fermi_old
            self.g.setF(self.F*har_to_eV, self.mu1, self.mu2)
        else:
            print('Calculating equilibrium density matrix:')
            if self.N1 is not None:
                P += densityComplexN(self.F*har_to_eV, self.S, self.g, self.Emin, self.fermi, 
                                    N=self.N1, T=self.T)
            else:
                P += densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.fermi, 
                                    tol=self.tol, T=self.T)
         
        # If bias applied, need to integrate G<
        if self.mu1 != self.mu2:
            print('Calculating left contact non-equilibrium density matrix:')
            if self.Nnegf is not None:
                P += densityGridN(self.F*har_to_eV, self.S, self.g, self.fermi, self.mu1, ind=0, 
                                    N=self.Nnegf, T=self.T)
                print('Calculating right contact non-equilibrium density matrix:')
                P += densityGridN(self.F*har_to_eV, self.S, self.g, self.fermi, self.mu2, ind=-1, 
                                    N=self.Nnegf, T=self.T)
            else:
                P += densityGrid(self.F*har_to_eV, self.S, self.g, self.fermi, self.mu1, ind=0, 
                                    tol=self.tol, T=self.T)                     
                print('Calculating right contact non-equilibrium density matrix:')
                P += densityGrid(self.F*har_to_eV, self.S, self.g, self.fermi, self.mu2, ind=-1, 
                                    tol=self.tol, T=self.T)
            #P2 = self.g.densityComplex(self.Emin, self.mu2, 1)
       
        # Calculate Level Occupation, Lowdin TF,  Return
        D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X)
        self.Xi = LA.inv(self.X)
        pshift = V.conj().T @ (self.Xi@P@self.Xi) @ V
        self.P = P.copy()
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)        
        
        
        #DEBUG:
        #for pair in zip(occList[inds], EList[inds]):                       
        #    print("Energy=", str(pair[1]), ", Occ=", str(pair[0]))

        return EList[inds], occList[inds]

    
    # Updated to update surfG() Fock matrix
    def PToFock(self):
        """
        Calculate new Fock matrix and update surfG object.

        Returns
        -------
        float
            Energy difference from previous iteration
        """
        Fock_old = self.F.copy()
        dE = super().PToFock()
        self.F, self.locs = getFock(self.bar, self.spin)
        self.g.setF(self.F*har_to_eV, self.mu1, self.mu2)
       
        #DEBUG:
        #D,V = LA.eig(self.X@(Fock_old*har_to_eV)@self.X) 
        #EListBefore = np.sort(np.array(np.real(D)).flatten())
        #D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X) 
        #EList = np.sort(np.array(np.real(D)).flatten())
        #for pair in zip(EListBefore, EList):                       
        #    print("Energy Before =", str(pair[0]), ", Energy After =", str(pair[1]))
       
        return dE
    
    
