import numpy as np
from scipy import io
from scf2 import NEGF
from transport import *

fn = "AuSys"
A = io.loadmat('sig_Ef_old.mat')
sig2 = A['sig2_Ef']
sig1 = A['sig1_Ef']
#sig1 = -1.0j
#sig2 = -1.0j

lContact = [1,2,3,4,5,6,7,8,9,10,11,12]
rContact = [19,20,21,22,23,24,25,26,27,28,29,30]
fermi = -5.1

negf = NEGF(fn=fn)
negf.setSigma(lContact, rContact, sig1/4, sig2/4)
Vlist = np.arange(0, 1.1, 0.1)
Ilist = []

print("Initial Run")
negf.setVoltage(fermi, 0)
print("damping=0.05")
negf.SCF(conv=1e-3, damping=0.05, maxcycles=300)
negf.saveMAT(fn+'.mat')
negf.writeChk()

print("Initial convergence done, entering IV loop...")
negf.setSigma(lContact, rContact, sig1, sig2)
f = open('IV.log', 'w')
f.write("V (volts), I (Amps)\n")
f.flush()
for V in Vlist:
    negf.setVoltage(fermi, V)
    negf.SCF(conv=1e-3, damping=0.05, maxcycles=300)
    matfile = f"{fn}DamleSig_{fermi:.2f}_{V:.2f}V.mat"
    negf.saveMAT(matfile)
    I = qCurrentF(matfile)
    Ilist.append(I)
    f.write(str(V)+","+str(I)+"\n")
    f.flush()

f.close()
