import numpy as np
import numpy.linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy import io
import matplotlib.pyplot as plt

from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

from gauNEGF.matTools import *
from gauNEGF.scf import NEGF
from gauNEGF.scfE import NEGFE
from gauNEGF.surfG1D import surfG
from gauNEGF.density import *
from gauNEGF.transport import *

hartree_to_eV = 27.211386

# PART 1 - TRANSPORT WITHOUT SCF
# Use a long chain (12 Si atoms) to approximate an infinite chain
# Run DFT calculation using SiNanowire12.gjf input file:
bar = qcb.BinAr(debug=False,lenint=8,inputfile="SiNanowire12.gjf")
bar.update(model='b3lyp', basis='lanl2dz', toutput='out.log',dofock="scf")

# Collect matrices from Gaussian, generate orthogonal H matrix
S = np.array(bar.matlist['OVERLAP'].expand())
P = np.array(bar.matlist['ALPHA SCF DENSITY MATRIX'].expand())
F = np.array(bar.matlist['ALPHA FOCK MATRIX'].expand())*hartree_to_eV
X = np.array(fractional_matrix_power(S, -0.5))
H = np.real(X@F@X)

# Cut out middle 2 Si atoms to use for generation of infinite chain
contactInds = np.arange(0, 8)
onsiteInds = np.arange(8, 16)
PS = P@S
ne = np.trace(PS[40:56, 40:56]).real
F = F[40:56, 40:56]
S = S[40:56, 40:56]
H = H[40:56, 40:56]

# Transport calculations for non-orthogonal case
print('Coherent transport for non-orth case')
g = surfG(F, S, [contactInds, onsiteInds], eta=1e-4) #Added broadening to speed up convergence
fermi = getFermiContact(g, ne)
Elist = np.linspace(-5, 5, 1000)
T = cohTransE(Elist+fermi, F, S, g)

# Transport calculations for non-orthogonal case
print('Coherent transport for orth case')
g = surfG(H, np.eye(len(H)), [contactInds, onsiteInds])
fermi = getFermiContact(g, ne)
Elist = np.linspace(-5, 5, 1000)
Torth = cohTransE(Elist+fermi, H, np.eye(len(H)), g)

io.savemat('SiNanowire_TnoSCF.mat', {'Elist':Elist, 'fermi':fermi, 'T':T, 'Torth':Torth})


#PART 2 - TRANSPORT WITH SCF
print(' ====== PART 2 ====== ')
negf = NEGFE(fn='Si2', func='b3lyp', basis='lanl2dz')
inds = negf.setContact1D([[1],[2]], eta=1e-4) #Again, some broadening to speed up convergence
negf.setVoltage(0, fermiMethod='bisect')
# This type of contact is unstable, setting a low damping value
negf.setIntegralLimits(512, 128, Emin=-24)
negf.SCF(1e-2, 0.005, 200)
negf.saveMAT('SiNanowire_ESCF.mat')

Torth = cohTransE(Elist+negf.fermi, negf.F, negf.S, negf.g)
io.savemat('SiNanowire_TESCF.mat', {'Elist':Elist, 'fermi':negf.fermi, 'T':T})


inds = negf.setContact1D([[1],[2]], T=300, eta=1e-4)
negf.SCF(1e-3, 0.002, 200)
negf.saveMAT('SiNanowire_ESCF_300K.mat')

Torth = cohTransE(Elist+negf.fermi, negf.F, negf.S, negf.g)
io.savemat('SiNanowire_TESCF_300K.mat', {'Elist':Elist, 'fermi':negf.fermi, 'T':T})

