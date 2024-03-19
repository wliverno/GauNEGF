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


class surfG:
    def __init__(self, Fock, Overlap, indsList, taus, staus, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, eps=1e-9):
        
        # Set up system
        self.F = np.array(Fock)
        self.S = np.array(Overlap)
        self.X = np.array(fractional_matrix_power(Overlap, -0.5))
        self.indsList = indsList
        
        # Set Contact Coupling
        if len(np.shape(taus[0])) == 1:
           self.tauFromFock = True
           self.tauInds = taus
           self.tauList = [self.F[np.ix_(taus[0], taus[1])], self.F[np.ix_(taus[1], taus[0])]]
           self.stauList = [self.S[np.ix_(taus[0], taus[1])], self.S[np.ix_(taus[1], taus[0])]]
        else:
           self.tauList = taus
           self.stauList = staus
        
        # Set up contact information
        if isinstance(alphas, int):
            self.contactFromFock = True
            self.setContacts()
        else:
            self.contactFromFock = False
            self.setContacts(alphas, aOverlaps, betas, bOverlaps)

        # Set up broadening for retarded/advanced Green's function, initialize g
        self.eps = eps
        self.gPrev = [np.zeros(np.shape(alpha)) for alpha in self.aList]
    
    def setContacts(self, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1):
        if self.contactFromFock:
            self.aList = []
            self.aSList = []
            for inds in self.indsList:
                self.aList.append(self.F[np.ix_(inds, inds)])
                self.aSList.append(self.S[np.ix_(inds, inds)])
        else:
            self.aList = alphas
            self.aSList = aOverlaps
            
        if self.contactFromFock:
            self.bList = self.tauList
            self.bSList = self.stauList
        else:
            self.bList = betas
            self.bSList = bOverlaps
    
    def g(self,E, i, conv=1e-5, relFactor=0.1):
        alpha = self.aList[i]
        Salpha = self.aSList[i]
        beta = self.bList[i]
        Sbeta = self.bSList[i]
        A = (E+1j*self.eps)*Salpha - alpha
        B = (E+1j*self.eps)*Sbeta - beta
        g = self.gPrev[i].copy()
        count = 0
        maxIter = int(1/conv)
        diff = conv+1
        while diff>conv and count<maxIter:
            g_ = g.copy()
            g = LA.inv(A - B@g@B.conj().T)
            dg = abs(g - g_)/(abs(g).max())
            g = g*relFactor + g_*(1-relFactor)
            diff = dg.max()
            count = count+1
        #print(f'g generated in {count} iterations with convergence {diff}')
        self.gPrev[i] = g
        return g
    
    def setF(self, F):
        self.F = F
        if self.contactFromFock:
            self.setContacts()
        if self.tauFromFock:
           taus = self.tauInds
           self.tauList = [self.F[np.ix_(taus[0], taus[1])], self.F[np.ix_(taus[1], taus[0])]]
           self.stauList = [self.S[np.ix_(taus[0], taus[1])], self.S[np.ix_(taus[1], taus[0])]]
    
    def sigma(self, E, i, conv=1e-5):
        sigma = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        inds = self.indsList[i]
        stau = self.stauList[i]
        tau = self.tauList[i]
        t = E*stau - tau
        sig = t@self.g(E, i, conv)@t.conj().T
        sigma[np.ix_(inds, inds)] += sig
        return sigma
    
    def sigmaTot(self, E, conv=1e-5):
        sigma = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        for i, inds in enumerate(self.indsList):
            stau = self.stauList[i]
            tau = self.tauList[i]
            t = E*stau - tau
            sig = t@self.g(E, i, conv)@t.conj().T
            sigma[np.ix_(inds, inds)] += sig
        return sigma
    

    def denFunc(self, E, ind=-1):
        if ind==-1:
            sig = self.sigmaTot(E)
        else:
            sig = self.sigma(E, ind)
        Gambar = self.X@(1j*(sig - sig.conj().T))@self.X
        Fbar = self.X@(self.F + sig)@self.X
        D, V = LA.eig(Fbar)
        V = np.array(V, dtype=complex)
        Ga = np.array(np.diag(1/(E-np.conj(D))))
        Ga = V@Ga@V.conj().T
        Gr = np.array(np.diag(1/(E-D)))
        Gr = V@Gr@V.conj().T
        return Gr@Gambar@Ga
    
    def recursiveResidue(self, E, ind=-1, conv=1e-3):
        resid = 1
        while resid>conv:
            #print(E, resid)
            Eprev = E+0.0
            if ind==-1:
                sig = self.sigmaTot(E)
            else:
                sig = self.sigma(E, ind)
            Fbar = self.X@(self.F + sig)@self.X
            D, V = LA.eig(Fbar);
            Eind = np.argmin(abs(E - D.flatten()))
            E = np.conj(D.flatten()[Eind])
            resid = abs((Eprev - E)/Eprev)
        return E

    def densityGrid(self, Emin, Emax, ind=-1, dE=0.001):
        Egrid = np.arange(Emin, Emax, dE)
        den = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        print('Starting Integration...')
        for i in range(1,len(Egrid)):
            E = (Egrid[i]+Egrid[i-1])/2
            dE = Egrid[i]-Egrid[i-1]
            den += self.denFunc(E, ind)*dE
        print('Integration done!')
        return den/(2*np.pi)

    def densityComplex(self, Emin, Emax, ind=-1, dE=0.1, recursiveResid=True):
        #Construct circular contour
        center = (Emin+Emax)/2
        r = (Emax-Emin)/2
        N = int((Emax-Emin)/dE)
        theta = np.linspace(0, np.pi, N)
        Egrid = r*np.exp(1j*theta)+center

        #Calculate Residues, use center energy
        if ind==-1:
            sig = self.sigmaTot(center)
        else:
            sig = self.sigma(center, ind)
        Fbar = self.X@(self.F + sig)@self.X
        Res = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        I = np.array(np.identity(len(Fbar)))
        D, V = LA.eig(Fbar)
        Gambar = self.X@(1j*(sig - sig.conj().T))@self.X
        for j, E in enumerate(D):
            if abs(E-center) < r:
                #print('Residue, E=', E)
                if recursiveResid:
                    Enew = self.recursiveResidue(E, ind)
                    #print('New Residue, E=',Enew)
                    if ind==-1:
                        sig = self.sigmaTot(Enew)
                    else:
                        sig = self.sigma(Enew, ind)
                    Fbar = self.X@(self.F + sig)@self.X
                    Gambar = self.X@(1j*(sig - sig.conj().T))@self.X
                    D, V = LA.eig(Fbar)
                Ga = np.array(np.diag(1/(E-np.conj(D))))
                Ga = V@ Ga @ V.conj().T 
                Vrow = np.array([V[j,:]])
                Y = Vrow.T @ Vrow.conj()
                Res += 2j*np.pi*np.conj(Y@Gambar@Ga) #WHY CONJUGATE???
                #Res += 2j*np.pi*denFunc(D, V, Gambar, E+1e-9)*(-1e-9)

        #Integrate along contour
        print('Starting Integration...')
        lineInt = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        for i in range(1,N):
            E = (Egrid[i]+Egrid[i-1])/2
            dS = Egrid[i]-Egrid[i-1]
            lineInt += self.denFunc(E, ind)*dS
        print('Integration done!')

        #Use Residue Theorem
        return (Res-lineInt)/(2*np.pi)


