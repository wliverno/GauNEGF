import numpy as np
from scipy.linalg import fractional_matrix_power

# Class for generating an energy-independent green's function for testing scfE.py
class surfGTest:
    def __init__(self, Fock, Overlap, indsList, gamma=-0.1):
        self.F = Fock
        self.S = Overlap
        self.X = fractional_matrix_power(Overlap, -0.5)
        self.N = len(Fock)
        self.gamma = gamma
        self.indsList = indsList
    def sigma(self, E, i, conv=1e-3):
        sig = np.array(np.zeros((self.N, self.N)), dtype=complex)
        inds = self.indsList[i]
        sig[np.ix_(inds, inds)]= np.diag([self.gamma*1j/2]*len(inds))
        return sig
    def sigmaTot(self, E, conv=1e-3):
        sig = np.array(np.zeros((self.N, self.N)), dtype=complex)
        for i in range(len(self.indsList)):
            sig += self.sigma(E,i,conv)
        return sig
    def setF(self, F, mu1, mu2):
        self.F = F
