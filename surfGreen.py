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

#Constants
kB = 8.617e-5           # eV/Kelvin


class surfG:
    def __init__(self, Fock, Overlap, indsList, taus=-1, staus=-1, alphas=-1, aOverlaps=-1, betas=-1, bOverlaps=-1, eps=1e-6):
        
        # Set up system
        self.F = np.array(Fock)
        self.S = np.array(Overlap)
        self.X = np.array(fractional_matrix_power(Overlap, -0.5))
        self.indsList = indsList
        self.poleList = len(indsList)*[np.array([], dtype=complex)]
        self.Egrid = len(indsList)*[np.array([], dtype=complex)]
        
        # Set Contact Coupling
        if isinstance(taus, int):
            taus = indsList[-1:]+indsList[:-1]
        if len(np.shape(taus[0])) == 1:
           self.tauFromFock = True
           self.tauInds = taus
           self.tauList = [self.F[np.ix_(taus[0], taus[1])], self.F[np.ix_(taus[1], taus[0])]]
           self.stauList = [self.S[np.ix_(taus[0], taus[1])], self.S[np.ix_(taus[1], taus[0])]]
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
    
    def sigma(self, E, i, conv=1e-3):
        E = E.real
        sigma = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        inds = self.indsList[i]
        stau = self.stauList[i]
        tau = self.tauList[i]
        t = E*stau - tau
        sig = t@self.g(E, i, conv)@t.conj().T
        sigma[np.ix_(inds, inds)] += sig
        return sigma
    
    def sigmaTot(self, E, conv=1e-3):
        E = E.real
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
        D, V = LA.eig(Fbar)
        Ga = np.array(np.diag(1/(E-np.conj(D))))
        Ga = V@Ga@V.conj().T
        Gr = np.array(np.diag(1/(E-D)))
        Gr = V@Gr@V.conj().T
        return (Gr@Gambar@Ga, D)
   
    # Function to use recursive algorithm to find poles, no longer used 
    def recursiveResidue(self, E, conv=1e-3):
        resid = 1
        counter = 1
        #print(f'RRes starting for E={E}')
        Eprev = E+0.0
        while resid>conv and counter<50:
            Eprev = E+0.0
            Fbar = self.X@(self.F + self.sigmaTot(Eprev))@self.X
            D, V = LA.eig(Fbar);
            E = D[np.argmin(abs(Eprev - D))].conj()
            resid = abs((Eprev - E)/Eprev)
            #print(E, resid)
            counter += 1
        if counter <50:
            print(f'RRes Done in {counter} iterations, final E={E}')
        else:
            print(f'RRes timed out, conv={resid}, final E={E}')
        return E
    
    # Density matrix generation using direct integration across the (real) energy axis
    def densityGrid(self, Emin, Emax, ind=-1, dE=0.001):
        Egrid = np.arange(Emin, Emax, dE)
        den = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        print('Starting Integration...')
        for i in range(1,len(Egrid)):
            E = (Egrid[i]+Egrid[i-1])/2
            dE = Egrid[i]-Egrid[i-1]
            den += self.denFunc(E, ind)[0]*dE
        print('Integration done!')
        return den/(2*np.pi)
    
    # Density matrix generation using the complex contour and residue theorem
    def densityComplex(self, Emin, Emax, ind=-1, dE=0.1, plotPoles=True, mu=-9999, T=300):
        kT = kB*T
        if plotPoles:
            plt.clf()
        
        #Construct circular contour
        center = (Emin+Emax)/2
        r = (Emax-Emin)/2
        N = int((Emax-Emin)/dE)
        theta = np.linspace(0, np.pi, N)
        Egrid = r*np.exp(1j*theta)+center

        #Integrate along contour, store eigenvalues at each point in Dlist
        print('Starting Integration...')
        lineInt = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        nD = len(self.F)
        Dlist = np.zeros((N, nD), dtype=complex)
        for i in range(N):
            E = (Egrid[i]+Egrid[i-1])/2
            dS = Egrid[i]-Egrid[i-1]
            (F, D) = self.denFunc(E, ind, mu, T)
            Dlist[i,:] = D[np.argsort(D.real)]
            lineInt += F*dS
        print('Integration done!')
        
        #Calculate pole locations from Dlist
        poles = np.zeros(nD, dtype=complex)
        for i in range(nD):
            Di = Dlist[:, i]
            Ediff = np.abs(Egrid-Di.real)
            poles[i] = Di[np.argmin(Ediff)]
            if plotPoles:
                plt.scatter(Di.real, -1*Di.imag, (1 - Ediff/max(Ediff))*20)
        
        #Apply contour cutoff to poles
        poleList = poles.copy().conj()
        #print(poleList)
        poleList = poleList[(abs(poleList-center)<r)&(poleList.imag>0)]
        if plotPoles:
            plt.plot(poleList.real, poleList.imag, '+k')
            plt.plot(Egrid.real, Egrid.imag, '--')
            plt.xlabel('Re(Z)')
            plt.ylabel('Im(Z)')
            plt.title('Variation in Poles')
            plt.xlim(Emin-5*kT, Emax+10*kT)
            plt.ylim(0, r+5*kT)
            plt.savefig('poleFig.png')
            plt.clf()
 
        #Loop through each possible pole and calculate Residue
        Res = np.array(np.zeros(np.shape(self.F)), dtype=complex)
        for j, E in enumerate(poleList):
            print('Residue at E=', E)
            sigTot = self.sigmaTot(E)
            if ind==-1:
                sig = sigTot
            else:
                sig = self.sigma(E, ind)
            Fbar = self.X@(self.F + sigTot)@self.X
            D, V = LA.eig(Fbar)
            Gambar = self.X@(1j*(sig - sig.conj().T))@self.X
            if mu!=-9999:
                Gambar /= (np.exp((np.real(E)-mu)/kT)+1)
            Ga = np.array(np.diag(1/(E-np.conj(D))))
            Ga = V@ Ga @ V.conj().T 
            Vrow = np.array([V[:, j]])
            Y = Vrow.T @ Vrow.conj()
            Res += 2j*np.pi*np.conj(Y@Gambar@Ga)
            #Res += 2j*np.pi*denFunc(D, V, Gambar, E+1e-9)*(-1e-9) #For Testing
        
        # Store Egrid and poleList for access
        self.Egrid[ind] = Egrid
        self.poleList[ind]  = poleList

        #Use Residue Theorem
        return (Res-lineInt)/(2*np.pi)


