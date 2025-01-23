# Python packages
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import time
import matplotlib.pyplot as plt
from numpy import savetxt

# Gaussian interface packages
from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

#Constants
kB = 8.617e-5           # eV/Kelvin

class surfG:
    def __init__(self, Fock, Overlap, indsList, taus=-1, staus=-1, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, neList=-1, eta=1e-9):
        # Set up system
        self.F = np.array(Fock)
        self.S = np.array(Overlap)
        self.X = np.array(fractional_matrix_power(Overlap, -0.5))
        self.indsList = indsList
        self.poleList = len(indsList)*[np.array([], dtype=complex)]
        self.Egrid = len(indsList)*[np.array([], dtype=complex)]
        
        # Set Contact Coupling
        if isinstance(taus, int):
            taus = [indsList[-1],indsList[0]]
        if len(np.shape(taus[0])) == 1:
           self.tauFromFock = True
           self.tauInds = taus
           self.tauList = [self.F[np.ix_(taus[0],indsList[0])], self.F[np.ix_(taus[1],indsList[-1])]]
           self.stauList = [self.S[np.ix_(taus[0],indsList[0])], self.S[np.ix_(taus[1],indsList[-1])]]
        else:
           self.tauFromFock = False
           self.tauList = taus
           self.stauList = staus
        
        # Set up contact information
        if isinstance(alphas, int):
            self.contactFromFock = True
            self.setContacts()
        else:
            self.contactFromFock = False
            self.setContacts(alphas, aOverlaps, betas, bOverlaps, neList)
            self.fermiList = [0]*len(indsList)

        # Set up broadening for retarded/advanced Green's function, initialize g
        self.eta = eta
        self.gPrev = [np.zeros(np.shape(alpha)) for alpha in self.aList]
    
    def setContacts(self, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, neList=-1):
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
            self.fermiList = [getFermi1DContact(self, neList[i], i) for i in range(len(indsList))]
    
    def g(self,E, i, conv=1e-5, relFactor=0.1):
        alpha = self.aList[i]
        Salpha = self.aSList[i]
        beta = self.bList[i]
        Sbeta = self.bSList[i]
        A = (E+1j*self.eta)*Salpha - alpha
        B = (E+1j*self.eta)*Sbeta - beta
        g = self.gPrev[i].copy()
        count = 0
        maxIter = int(1/(conv*relFactor))*10
        diff = conv+1
        while diff>conv and count<maxIter:
            g_ = g.copy()
            g = LA.inv(A - B@g@B.conj().T)
            dg = abs(g - g_)/(abs(g).max())
            g = g*relFactor + g_*(1-relFactor)
            diff = dg.max()
            count = count+1
        if diff>conv:
            print(f'Warning: exceeded max iterations! E: {E}, Conv: {diff}')
        #print(f'g generated in {count} iterations with convergence {diff}')
        self.gPrev[i] = g
        return g
   
    # Update Fock matrix and subsequent contacts
    def setF(self, F, mu1, mu2):
        self.F = F
        if self.tauFromFock:
            taus = self.tauInds
            indsList = self.indsList
            self.F[np.ix_(indsList[0], indsList[0])] = self.F[np.ix_(taus[0], taus[0])].copy()
            self.F[np.ix_(indsList[-1], indsList[-1])] = self.F[np.ix_(taus[1], taus[1])].copy()
            self.tauList = [self.F[np.ix_(taus[0],indsList[0])], self.F[np.ix_(taus[1],indsList[-1])]]
            self.stauList = [self.S[np.ix_(taus[0],indsList[0])], self.S[np.ix_(taus[1],indsList[-1])]]
        if not self.contactFromFock:
            for i,mu in zip([0,-1], [mu1, mu2]):
                fermi = self.fermiList[i]
                if fermi!= mu:
                    dFermi = mu - fermi
                    self.alphas[i] += dFermi*np.eye(len(self.alphas[i]))
                    self.betas[i] += dFermi*betaOverlaps[i]
                    self.fermiList[i] = mu
    
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
    

    def denFunc(self, E, ind=-1, mu=-9999, T=300):
        kT = kB*T
        sigTot = self.sigmaTot(E)
        if ind==-1:
            sig = sigTot
        else:
            sig = self.sigma(E, ind)
        Gambar = self.X@(1j*(sig - sig.conj().T))@self.X
        if mu!=-9999:
            Gambar /= (np.exp((np.real(E)-mu)/kT)+1)
        Fbar = self.X@(self.F + sigTot)@self.X
        D, V_lhp = LA.eig(Fbar)
        V_uhp = LA.inv(V_lhp).conj().T
        Gr = np.zeros(np.shape(Fbar), dtype=complex)
        for i in range(len(D)):
            vl = V_lhp[:, i].reshape(-1, 1)
            vu = V_uhp[i, :].reshape(1, -1)
            Gr += (vl@vu)/(E-D[i])
        Ga = Gr.conj().T
        return (Gr@Gambar@Ga, D)
    
    # Density matrix generation using direct integration across the (real) energy axis
    # Not used, for testing purposes only
    def densityGrid(self, Emin, Emax, ind=-1, dE=0.001, mu=-9999, T=300):
        # Create Fermi function if mu given
        kT = kB*T
        fermi = lambda E: 1
        if mu!=-9999:
            Emax += 5*kT
            fermi = lambda E: 1/(np.exp((E-mu)/kT)+1)
        # Direct integration
        Egrid = np.arange(Emin, Emax, dE)
        den = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        print('Starting Integration...')
        for i in range(1,len(Egrid)):
            E = (Egrid[i]+Egrid[i-1])/2
            dE = Egrid[i]-Egrid[i-1]
            den += self.denFunc(E, ind)[0]*fermi(E)*dE
        print('Integration done!')
        den /= 2*np.pi
        return den
    
