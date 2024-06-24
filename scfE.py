import numpy as np
from numpy import linalg as LA
import sys
import time
import matplotlib.pyplot as plt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

from matTools import *

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
kT = 0.025              # eV @ 20degC
V_to_au = 0.03675       # Volts to Hartree/elementary Charge



class NEGFE(NEGF):
    
    def setContactE(self, contactList, tauList=-1, stauList=-1, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, eps=1e-9):
        inds = super().setContacts(contactList[0], contactList[1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        self.g = surfG(self.F, self.S, inds, tauList, stauList, alphas, aOverlaps, betas, bOverlaps, eps)
        return inds
        
    def getSigma(self, E):
        return (self.g.sigma(E, 0), self.g.sigma(E, 1))

    def FockToP(self):
        # Density contribution from below self.Emin
        sigWVal = -0.00001j #Based on Damle Code
        self.sigmaW1 = formSigma(self.lInd, sigWVal, self.nsto, self.S)
        self.sigmaW2 = formSigma(self.rInd, sigWVal, self.nsto, self.S)
        self.sigmaW12 = self.sigmaW1+self.sigmaW2
    
        self.GamW1 = (self.sigmaW1 - self.sigmaW1.conj().T)*1j
        self.GamW2 = (self.sigmaW2 - self.sigmaW2.conj().T)*1j
        
        FbarW = self.X@(self.F*har_to_eV + self.sigmaW12)@self.X
        GamBarW1 = self.X@self.GamW1@self.X
        GamBarW2 = self.X@self.GamW2@self.X
        Dw,Vw = LA.eig(np.array(FbarW))
        Pw = density(Vw, Dw, GamBarW1+GamBarW2, self.Eminf, self.Emin)
        
        # DEBUG: 
        #Pwalt = self.g.densityComplex(self.Emin=Eminf, Emax=Emin, dE=(Emin-Eminf)/400)
        #print("Comparing Densities:")
        #print(np.diag(Pw)[:10])
        #print(np.diag(Pwalt)[:10])
        #print("--------------------------")
        
        # Density contribution from above self.Emin
        print('Calculating Density for left contact:')
        P1 = self.g.densityComplex(self.Emin, self.mu1, 0)
        print('Calculating Density for right contact:')
        P2 = self.g.densityComplex(self.Emin, self.mu2, 1)
        
        # Sum them Up.
        P = P1 + P2 + Pw
        
        # Calculate Level Occupation, Lowdin TF,  Return
        D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X)
        pshift = V.conj().T @ P @ V
        self.P = np.real(self.X@P@self.X)
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)        
        
        #DEBUG:
        for pair in zip(occList[inds], EList[inds]):                       
            print("Energy=", str(pair[1]), ", Occ=", str(pair[0]))

        return EList[inds], occList[inds]

    
    # Use Gaussian to calculate the Density Matrix
    def PToFock(self, damping, Edamp=False):
        Fock_old = self.F.copy()
        dE, RMSDP, MaxDP = super().PToFock(damping, Edamp)
        self.F, self.locs = getFock(self.bar, self.spin)
        #self.g.setF(self.F)
        
        # Debug:
        #D,V = LA.eig(self.X@(Fock_old*har_to_eV)@self.X) 
        #EListBefore = np.sort(np.array(np.real(D)).flatten())
        #D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X) 
        #EList = np.sort(np.array(np.real(D)).flatten())
        #for pair in zip(EListBefore, EList):                       
        #    print("Energy Before =", str(pair[0]), ", Energy After =", str(pair[1]))
        
        return dE, RMSDP, MaxDP


