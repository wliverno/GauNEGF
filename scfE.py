import numpy as np
from scipy import linalg as LA
import sys
import time
import matplotlib.pyplot as plt
from PIL import Image
import os


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

from matTools import *
from density import *

from scf2 import NEGF
from surfGreen import surfG

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
    # Set contact as surfG() object
    def setContactE(self, contactList, tauList=-1, stauList=-1, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, eps=1e-9):
        inds = super().setContacts(contactList[0], contactList[1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        self.g = surfG(self.F*har_to_eV, self.S, inds, tauList, stauList, alphas, aOverlaps, betas, bOverlaps, eps)
        self.figures = []
        self.setIntegralLimits(100, 50)
        return inds
    
    def setIntegralLimits(self, N1, N2):
        self.N1 = N1
        self.N2 = N2
 
    def integralCheck(self, tol=1e-4, cycles=10, damp=0.1):
        print(f'RUNNING SCF FOR {cycles} CYCLES USING DEFAULT GRID: ')
        if self.updFermi:
            self.updFermi=False
            self.SCF(1e-10,damp,cycles)
            self.updFermi=True
        else:
            self.SCF(1e-10,damp,cycles)
        print('SETTING INTEGRATION LIMITS... ')
        self.Emin, self.N1, self.N2 = integralFit(self.F*har_to_eV, self.S, self.g, sum(self.getHOMOLUMO())/2, self.Eminf, tol)
        print('INTEGRATION LIMITS SET!')
        print('#############################')
    
    # Get left and right contact self-energies at specified energy
    def getSigma(self, E):
        return (self.g.sigma(E, 0), self.g.sigma(E, 1))

    # Updated to use energy-dependent contour integral from surfG()
    def FockToP(self, T=300, N=100):
        # Density contribution from below self.Emin
        Pw = densityReal(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, T=0)
        print(np.diag(Pw)[:6].real)
        #print(np.diag(Pw)) 
        
        # DEBUG: 
        #Pwalt = self.g.densityComplex(self.Emin=Eminf, Emax=Emin, dE=(Emin-Eminf)/400)
        #print("Comparing Densities:")
        #print(np.diag(Pw)[:10])
        #print(np.diag(Pwalt)[:10])
        #print("--------------------------")
        # Density contribution from above self.Emin
        print('Calculating equilibrium density matrix:') 
        #P1 = densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=10, T=T)
        #P2 = densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=50, T=T)
        P = densityComplex(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=self.N1, T=T)
        print(np.diag(P)[:6].real)
        #P2 = densityReal(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=100, T=T)
        #print(np.diag(P2)[:6].real)
        #P3 = densityReal(self.F*har_to_eV, self.S, self.g, self.Emin, self.mu1, N=1000, T=T)
        #print(np.diag(P3)[:6].real)
        
        # If bias applied, need to integrate G<
        if self.mu1 != self.mu2:
            print('Calculating non-equilibrium density matrix:')
            P += densityGrid(self.F*har_to_eV, self.S, self.g, self.mu1, self.mu2, N=self.N1, T=T)
            #P2 = self.g.densityComplex(self.Emin, self.mu2, 1)
        P+= Pw

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

    
    # Updated to update surfG() Fock matrix and plot integral and residues
    def PToFock(self, debug=False):
        Fock_old = self.F.copy()
        dE = super().PToFock()
        self.F, self.locs = getFock(self.bar, self.spin)
        self.g.setF(self.F*har_to_eV)
        
        # Plot integral path and poles
        if debug==True:
            fig, ax = plt.subplots()
            ax.plot(self.g.Egrid[0].real, self.g.Egrid[0].imag, '-r')
            ax.plot(self.g.poleList[0].real, self.g.poleList[0].imag, 'xr')
            ax.axvline(self.mu1, c = 'r', ls='-')
            ax.plot(self.g.Egrid[1].real, self.g.Egrid[1].imag, '--b')
            ax.plot(self.g.poleList[1].real, self.g.poleList[1].imag,'+b')
            ax.axvline(self.mu2, c = 'b', ls='--')
            ax.set_xlabel('Re(Z) eV')
            ax.set_ylabel('Imag(Z) eV')
            ax.set_title(f'Frame {len(self.figures)}: RMSDP - {RMSDP:.2E}, MaxDP - {MaxDP:.2E}')
            lowBnd = self.Emin - 5*kT
            upBnd = max(self.mu1, self.mu2)+5*kT
            ax.set_xlim(lowBnd, upBnd)
            ax.set_ylim(-1, upBnd-lowBnd)
            self.figures.append(fig)

        # Debug:
        #D,V = LA.eig(self.X@(Fock_old*har_to_eV)@self.X) 
        #EListBefore = np.sort(np.array(np.real(D)).flatten())
        #D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X) 
        #EList = np.sort(np.array(np.real(D)).flatten())
        #for pair in zip(EListBefore, EList):                       
        #    print("Energy Before =", str(pair[0]), ", Energy After =", str(pair[1]))
         
        return dE
    
    # Save integration plots as frame in animated gif
    def plotAnimation(self, gif_path='output.gif'):
        images = []

        for fig in self.figures:
            # Save the figure to a temporary file
            fig_path = f'temp_frame_{self.figures.index(fig)}.png'
            fig.savefig(fig_path)
            plt.close(fig)
            
            # Open the image and append to the list
            images.append(Image.open(fig_path))
            
            # Remove the temporary file
            os.remove(fig_path)

        # Save all frames as a new or updated GIF
        if images:
            images[0].save(gif_path, format='GIF', append_images=images[1:], save_all=True, duration=len(images)*10, loop=0)
            print(f'Saved GIF as {gif_path}')
        else:
            print('No frames to save.')
        
        # Empty the list of figures
        self.figures = []
        print('Figures stack cleared!')
    

