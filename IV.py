import numpy as np
from scipy import io
from scf2 import NEGF
from transport import *
from scipy.linalg import block_diag

fn = "AuSys"
A = io.loadmat('sig_Ef_recalc.mat')
sig2 = A['sig2_Ef']
sig1 = A['sig1_Ef']
#sig1 = block_diag(sig1, sig1)
#sig2 = block_diag(sig2, sig2)
#sig1 = -1.0j
#sig2 = -1.0j
sig1 = np.kron(sig1, np.eye(2))
sig2 = np.kron(sig2, np.eye(2))

lContact = [1,2,3,4,5,6,7,8,9,10,11,12]
rContact = [19,20,21,22,23,24,25,26,27,28,29,30]
fermi = -5.1

negf = NEGF(fn=fn, basis="ChkBasis",spin="g", route="integral=dkhso")
negf.setSigma(lContact, rContact, sig1, sig2)
Vlist = np.arange(0, 1.2, 0.2)
Ilist = []

print("Initial Run")
negf.setVoltage(fermi, 0)
m = io.loadmat('AuSysRecalcU_-5.10_0.00V.mat')
negf.setFock(m['F'])
negf.SCF(conv=1e-3, damping=1, maxcycles=0)
print("damping=0.01")
negf.SCF(conv=1e-3, damping=0.01, maxcycles=300)
print("damping=0.001")
negf.SCF(conv=1e-3, damping=0.001, maxcycles=300)
negf.saveMAT(fn+'GSO.mat')
negf.writeChk()

print("Initial convergence done, entering IV loop...")
negf.setSigma(lContact, rContact, sig1, sig2)
f = open('IVAuGSO.log', 'w')
f.write("V (volts), I (Amps)\n")
f.flush()
for V in Vlist:
    negf.setVoltage(fermi, V)
    negf.SCF(conv=1e-3, damping=0.01, maxcycles=300)
    matfile = f"{fn}RecalcGSO_{fermi:.2f}_{V:.2f}V.mat"
    negf.saveMAT(matfile)
    I = qCurrentF(matfile)
    Ilist.append(I)
    f.write(str(V)+","+str(I)+"\n")
    f.flush()

f.close()
