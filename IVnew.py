import numpy as np
from scipy import io
from scf2 import NEGF
import transport

fn = "ethane"
sig1 = -0.2j
sig2 = -0.2j
lContact = [2]
rContact = [6]
fermi = -5.1 

obj = NEGF(fn, spin="u")
Vlist = np.concatenate((np.arange(0, 5.1, 0.1), np.arange(5.0, -0.1, -0.1)))
Ilist = []
f = open('IV3.log', 'w')
f.write("V (volts),I (Amps)\n")
for V in Vlist:
    obj.setSigma(lContact, rContact, fermi, V, sig1, sig2)
    obj.SCF(conv=1e-2, damping=0.1, maxcycles=100)
    matfile = fn+str(fermi)+"_"+str(V)+".mat"
    obj.saveMAT(matfile)
    I = transport.qCurrentF(matfile, 0.001)
    Ilist.append(I)
    f.write(str(V)+","+str(I)+"\n")
    f.flush()
    obj.writeChk()

f.close()
