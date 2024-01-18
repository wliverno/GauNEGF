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

#From Sajjad/Damle:
Emin = -15
Eminf = -1e5



class NEGFE(NEGF):
#    def __init__(self, fn, basis="lanl2dz", func="b3pw91", spin="r", chkInit=True, route=""):
#        self.NEGF = NEGF(fn, basis, func, spin, chkInit, route)
#        self.F = self.NEGF.F
#        self.S = self.NEGF.S
#        self.X = self.NEGF.X
#    
#    def setVoltage(self, fermi, qV):
#        self.NEGF.setVoltage(fermi, qV)
#        self.fermi = fermi
#        self.qV = qV
#        self.mu1 =  fermi + (qV/2)
#        self.mu2 =  fermi - (qV/2)
    
    def setContactE(self, contactList, tauList, stauList, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, eps=1e-9):
        inds = super().setContacts(contactList[0], contactList[1])
        self.lInd = inds[0]
        self.rInd = inds[1]
        self.g = surfG(self.F, self.S, inds, tauList, stauList, alphas, aOverlaps, betas, bOverlaps, eps)
        return inds
        
    def getSigma(self, E):
        return (self.g.sigma(E, 0), self.g.sigma(E, 1))

    def FockToP(self):
        # Density contribution from below Emin
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
        Pw = density(Vw, Dw, GamBarW1+GamBarW2, Eminf, Emin)
        
        # Density contribution from above Emin
        print('Calculating Density for left contact:')
        P1 = self.g.densityComplex(Emin, self.mu1, 0)
        print('Calculating Density for right contact:')
        P2 = self.g.densityComplex(Emin, self.mu2, 1)
        
        # Sum them Up.
        P = P1 + P2 + Pw
        
        # Calculate Level Occupation, Lowdin TF,  Return
        D,V = LA.eig(self.X@(self.F*har_to_eV)@self.X)
        pshift = V.conj().T @ P @ V
        self.P = np.real(self.X@P@self.X)
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)
        
        # Debug:
        #for pair in zip(occList[inds], EList[inds]):                       
        #    print("Energy =", str(pair[1]), ", Occ =", str(pair[0]))
        
        return EList[inds], occList[inds]

    
    # Use Gaussian to calculate the Density Matrix
    def PToFock(self, damping):
        dE, RMSDP, MaxDP = super().PToFock(damping)
        self.F, self.locs = getFock(self.bar, self.spin)
        self.g.setF(self.F)
        return dE, RMSDP, MaxDP


