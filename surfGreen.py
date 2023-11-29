import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import scipy.io as io
import sys
import time
import matplotlib.pyplot as plt
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu


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


class surfG:
    def __init__(self, Fock, Overlap, indsList, alphas, betas, eps=1e-9):
        self.aList = alphas
        self.bList = betas
        self.F = np.asmatrix(Fock)
        self.S = np.asmatrix(Overlap)
        self.X = np.asmatrix(fractional_matrix_power(Overlap, -0.5))
        self.indsList = indsList
        self.eps = eps
        self.gPrev = 0;
    
    def g(self,E, i, conv=1e-3, relFactor=0.1):
        alpha = np.asmatrix(self.aList[i])*har_to_eV
        beta = np.asmatrix(self.bList[i])*har_to_eV
        if isinstance(self.gPrev, int):
            g = LA.inv(E + 1j*self.eps - alpha)
        else:
            g = self.gPrev.copy()
        count = 0
        maxIter = int(1/conv)
        diff = conv+1
        while diff>conv and count<maxIter:
            g_ = g.copy()
            g = LA.inv(E+1j*self.eps - alpha - beta*g*beta.getH())
            dg = abs(g - g_)/(abs(g).max())
            g = g*relFactor + g_*(1-relFactor)
            diff = dg.max()
            count = count+1
        #print(f'g generated in {count} iterations with convergence {diff}')
        self.gPrev = g
        return g
    
    def sigma(self, E, i, conv=1e-3):
        sigma = np.asmatrix(-1j*1e-9*self.S,dtype=complex)
        inds = self.indsList[i]
        stau = self.S[np.ix_(inds, inds)]
        tau = self.F[np.ix_(inds, inds)]*har_to_eV
        t = E*stau - tau
        sig = t*self.g(E, i, conv)*t.getH()
        sigma[np.ix_(inds, inds)] += sig
        return sigma
    
    def sigmaTot(self, E, conv=1e-3):
        sigma = np.asmatrix(-1j*1e-9*self.S,dtype=complex)
        for i, inds in enumerate(self.indsList):
            stau = self.S[np.ix_(inds, inds)]
            tau = self.F[np.ix_(inds, inds)]*har_to_eV
            t = E*stau - tau
            sig = t*self.g(E, i, conv)*t.getH()
            sigma[np.ix_(inds, inds)] += sig
        return sigma
    

    def denFunc(self, E):
        sig = self.sigmaTot(E)
        Gambar = self.X*(1j*(sig - sig.getH()))*self.X
        Fbar = self.X*(self.F*har_to_eV + sig)*self.X
        D, V = LA.eig(Fbar);
        print(E, D.flatten()[np.argsort(np.real(D).flatten())[18:20]])
        V = np.asmatrix(V, dtype=complex)
        Ga = np.asmatrix(np.diag(1/(E-np.conj(D))))
        Ga = V*Ga*V.getH()
        Gr = np.asmatrix(np.diag(1/(E-D)))
        Gr = V*Gr*V.getH()
        return Gr*Gambar*Ga
    
    def recursiveResidue(self, E, conv=1e-3):
        resid = 1
        while resid>conv:
            #print(E, resid)
            Eprev = E+0.0
            sig = self.sigmaTot(E)
            Fbar = self.X*(self.F*har_to_eV + sig)*self.X
            D, V = LA.eig(Fbar);
            ind = np.argmin(abs(E - D.flatten()))
            E = np.conj(D.flatten()[ind])
            resid = abs((Eprev - E)/Eprev)
        return E

    def densityGrid(self, Emin, Emax, dE=0.001):
        Egrid = np.arange(Emin, Emax, dE)
        den = np.asmatrix(np.zeros(np.shape(self.F)), dtype=complex)
        print('Starting Integration...')
        for i in range(1,len(Egrid)):
            E = (Egrid[i]+Egrid[i-1])/2
            dE = Egrid[i]-Egrid[i-1]
            den += self.denFunc(E)*dE
        print('Integration done!')
        return den/(2*np.pi)

    def densityComplex(self, Emin, Emax, dE=0.001, recursiveResid=False):
        #Construct circular contour
        center = (Emin+Emax)/2
        r = (Emax-Emin)/2
        N = int((Emax-Emin)/dE)
        theta = np.linspace(0, np.pi, N)
        Egrid = r*np.exp(1j*theta)+center

        #Calculate Residues, use center energy
        sig = self.sigmaTot(center)
        Fbar = self.X*(self.F*har_to_eV + sig)*self.X
        Res = np.asmatrix(np.zeros(np.shape(self.F)), dtype=complex)
        I = np.asmatrix(np.identity(len(Fbar)))
        D, V = LA.eig(Fbar)
        Gambar = self.X*(1j*(sig - sig.getH()))*self.X
        for ind, E in enumerate(D):
            if abs(E-center) < r:
                print('Residue, E=', E)
                if recursiveResid:
                    Enew = self.recursiveResidue(E)
                    print('New Residue, E=',Enew)
                    sig = self.sigmaTot(Enew)
                    Fbar = self.X*(self.F*har_to_eV + sig)*self.X
                    Gambar = self.X*(1j*(sig - sig.getH()))*self.X
                    D, V = LA.eig(Fbar)
                Ga = np.asmatrix(np.diag(1/(E-np.conj(D))))
                Ga = V* Ga * V.getH() 
                Y = V[:, ind] * V.getH()[ind,:]
                Res += 2j*np.pi*np.conj(Y*Gambar*Ga) #WHY CONJUGATE???
                #Res += 2j*np.pi*denFunc(D, V, Gambar, E+1e-9)*(-1e-9)

        #Integrate along contour
        print('Starting Integration...')
        lineInt = np.asmatrix(np.zeros(np.shape(self.F)), dtype=complex)
        for i in range(1,N):
            E = (Egrid[i]+Egrid[i-1])/2
            dS = Egrid[i]-Egrid[i-1]
            lineInt += self.denFunc(E)*dS
        print('Integration done!')

        #Use Residue Theorem
        return (Res-lineInt)/(2*np.pi)


