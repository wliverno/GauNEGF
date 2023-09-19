import numpy as np
from scipy import io
import scf
import scf2

fn = "AuSys"
A = io.loadmat('sig_Ef.mat')
sig2 = np.real(A['sig2_Ef']) + (1j*np.imag(A['sig2_Ef'])/4)
sig1 = np.real(A['sig1_Ef']) + (1j*np.imag(A['sig1_Ef'])/4)
sig2 = A['sig2_Ef']
sig1 = A['sig1_Ef']
lContact = [1,2,3,4,5,6,7,8,9,10,11,12]
rContact = [19,20,21,22,23,24,25,26,27,28,29,30]
fermi = -5.1

Vlist = np.concatenate((np.arange(0, 1.1, 0.1), np.arange(1.0, -0.1, -0.1)))
Ilist = []
f = open('IV.log', 'w')
f.write("V (volts), I (Amps)\n")
f.flush()
for V in Vlist:
    matfile = scf.SCF(fn, lContact, rContact, fermi, V, sig1, sig2, 
                       damping=0.01, maxcycles=500, conv=1e-3)
    I = scf.qCurrentF(matfile)
    Ilist.append(I)
    f.write(str(V)+","+str(I)+"\n")
    f.flush()

f.close()
