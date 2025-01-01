# Python Packages
import numpy as np
from numpy import linalg as LA

# Gaussian interface packages
from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

# Developed Packages 
from matTools import *
from density import *
from transport import DOS
from fermiSearch import DOSFermiSearch
from scf import NEGF
from surfG1D import surfG
from surfGBethe import surfGB

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
    # Set energy dependent Bethe lattice contact using surfGB() object
    def setContactBethe(self, contactList, latFile='Au', eta=1e-9, T=300):
        # Set L/R contacts based on atom numbers, use orbital inds for surfG() object
        inds = super().setContacts(contactList[0], contactList[-1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        # Generate surfGB() object for Bethe lattice contacts
        self.g = surfGB(self.F*har_to_eV, self.S, contactList, self.bar, latFile, self.spin, eta, T)
        
        # Update other variables
        self.setIntegralLimits(100, 50)
        self.T = T
        return inds

    # Set energy dependent 1D contact using surfG() object
    def setContact1D(self, contactList, tauList=-1, stauList=-1, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, eta=1e-9, T=300):
        # Set L/R contacts based on atom numbers, use orbital inds for surfG() object
        inds = super().setContacts(contactList[0], contactList[-1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        # if tauList is a list of atom numbers (rather than a matrix), generate orbital indices
        if isinstance(tauList, int):
            pass
        elif len(np.shape(tauList[0])) == 1: 
            ind1 = np.where(np.isin(abs(self.locs), tauList[0]))[0]
            ind2 = np.where(np.isin(abs(self.locs), tauList[-1]))[0]
            tauList = (ind1, ind2)
        # Generate surfG() object for the molecule + contacts and initialize variables
        self.g = surfG(self.F*har_to_eV, self.S, inds, tauList, stauList, alphas, aOverlaps, betas, bOverlaps, eta)
        
        # Update other variables
        self.setIntegralLimits(100, 50)
        self.T = T
        return inds
    
    # Set up Fermi Search algorithm after setting system Fermi energies
    def setVoltage(self, qV, fermi=np.nan, Emin=None, Eminf=None, fermiMethod='muller'):
        super().setVoltage(qV, fermi, Emin, Eminf)
        if self.updFermi:
            self.fermiMethod = fermiMethod
    
    def setIntegralLimits(self, N1, N2, Emin=False):
        self.N1 = N1
        self.N2 = N2
        if Emin:
            self.Emin=Emin
 
    def integralCheck(self, tol=1e-4, cycles=10, damp=0.02, pauseFermi=False):
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
        self.Emin, self.N1, self.N2 = integralFit(self.F*har_to_eV, self.S, self.g, sum(self.getHOMOLUMO())/2, self.Eminf, tol)
        PLower = densityReal(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, T=0)
        nLower = np.trace(self.S@PLower).real
        if self.updFermi:
                print('CALCULATING FERMI ENERGY')
                ne = self.nae if self.spin is 'r' else self.nae+self.nbe
                self.fermi, dE, _ = calcFermiSecant(self.g, ne-nLower, self.Emin, self.fermi, self.N1, tol=tol, maxcycles=100)
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                self.setVoltage(self.qV, fermiMethod=self.fermiMethod)
        print('INTEGRATION LIMITS SET!')
        print('#############################')
    
    # Get left and right contact self-energies at specified energy
    def getSigma(self, E):
        return (self.g.sigma(E, 0), self.g.sigma(E, -1))

    # Updated to use energy-dependent contour integral from surfG()
    def FockToP(self):
        
        print('Calculating lower density matrix:') 
        P = densityReal(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, T=0)
        nLower = np.trace(self.S@P).real 

        # Fermi Energy Update using local self-energy approximation
        if self.updFermi:
            fermi_old = self.fermi+0.0
            conv= min(self.convLevel, 1e-3)
            if self.fermiMethod=='predict':
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
                if Nsearch > 0 and Nsearch < len(self.F):
                    self.fermi = bisectFermi(V, Vc, D, GamBar, Ncurr+dN, conv, self.Eminf, conv=conv)
                    print(f'Fermi Energy set to {self.fermi:.2f} eV, shifting by {dN:.2E} electrons ')
                else:
                    print('Warning: Local sigma approximation not valid, Fermi energy not updated...')
                print('Calculating equilibrium density matrix:') 
                P += densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=self.N1, T=self.T)
            elif self.fermiMethod=='secant':
                ne = self.bar.ne
                if self.spin =='r':
                    ne /= 2
                print('SECANT METHOD:')
                self.fermi, dE, P2 = calcFermiSecant(self.g, ne-nLower, self.Emin, fermi_old, self.N1, tol=conv)  
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                print('Setting equilibrium density matrix...') 
                P += P2
            elif self.fermiMethod=='muller':
                ne = self.bar.ne
                if self.spin =='r':
                    ne /= 2
                print('MULLER METHOD:')
                self.fermi, dE, P2 = calcFermiMuller(self.g, ne-nLower, self.Emin, fermi_old, self.N1, tol=conv)
                print(f'Fermi Energy set to {self.fermi:.2f} eV, error = {dE:.2E} eV ')
                print('Setting equilibrium density matrix...') 
                P += P2
            else:
                raise Exception('Error: invalid Fermi search method, needs to be \'muller\', \'secant\', or \'predict\'')
            # Shift Emin, mu1, and mu2 and update contact self-energies
            self.setVoltage(self.qV, fermiMethod=self.fermiMethod)
            self.Emin += self.fermi-fermi_old
            self.g.setF(self.F*har_to_eV, self.mu1, self.mu2)
        else:
            print('Calculating equilibrium density matrix:') 
            P += densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=self.N1, T=self.T)
         
        # If bias applied, need to integrate G<
        if self.mu1 != self.mu2:
            print('Calculating non-equilibrium density matrix:')
            P += densityGrid(self.F*har_to_eV, self.S, self.g, self.mu1, self.mu2, N=self.N1, T=self.T)
            #P2 = self.g.densityComplex(self.Emin, self.mu2, 1)
       
        # Calculate Level Occupation, Lowdin TF,  Return
        D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X)
        self.Xi = LA.inv(self.X)
        pshift = V.conj().T @ (self.Xi@P@self.Xi) @ V
        self.P = np.abs(P.real) + 1j*P.imag
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)        
        
        
        #DEBUG:
        #for pair in zip(occList[inds], EList[inds]):                       
        #    print("Energy=", str(pair[1]), ", Occ=", str(pair[0]))

        return EList[inds], occList[inds]

    
    # Updated to update surfG() Fock matrix
    def PToFock(self):
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
    
    
