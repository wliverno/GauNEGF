import numpy as np
from scipy.linalg import fractional_matrix_power
from matTools import formSigma

# Class for generating an energy-independent green's function for scfE.py
class surfGTest:
    def __init__(self, Fock, Overlap, indsList, sig1=None, sig2=None):
        self.F = Fock
        self.S = Overlap
        self.X = fractional_matrix_power(Overlap, -0.5)
        self.N = len(Fock)
        self.indsList = indsList
        self.sig = [np.array(np.zeros((self.N, self.N)), dtype=complex)]*2
        if sig1 is not None:
            self.sig[0] = formSigma(indsList[0], sig1, self.N, self.S)
            if sig2 is None:
                self.sig[1] = formSigma(indsList[1], sig1, self.N, self.S)
            else:
                self.sig[1] = formSigma(indsList[1], sig2, self.N, self.S)
        else:
            self.sig[0][np.ix_(indsList[0], indsList[0])]= np.diag([-0.05j]*len(inds))
            self.sig[1][np.ix_(indsList[1], indsList[1])]= np.diag([-0.05j]*len(inds))
    
    def sigma(self, E, i, conv=1e-3):
        return self.sig[i]
    def sigmaTot(self, E, conv=1e-3):
        sigTot = np.array(np.zeros((self.N, self.N)), dtype=complex)
        for i in range(len(self.indsList)):
            sigTot += self.sigma(E,i,conv)
        return sigTot
    def setF(self, F, mu1, mu2):
        self.F = F
