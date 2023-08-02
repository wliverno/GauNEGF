"""

@author: safroosh, willll
"""
import numpy as np
#import unittest
from numpy import linalg as LA
from scipy import linalg
import sys
import time
#import os
import matplotlib.pyplot as plt
#import math
#from goto import goto, label
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu



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
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kT = 0.025              # eV @ 20degC
V_to_au = 0.03675       # Volts to Hartree/elementary Charge

###############################################################################
#
#                             Functions
#
###############################################################################

def density(V, D, Gam, Emin, mu):
    Nd = len(V)
    repdum = np.asmatrix(np.ones(Nd))
    DD = D*repdum

    logmat = np.log(1-np.divide(mu,D))*repdum
    logmat2 = np.log(1-np.divide(Emin,D))*repdum


    invmat = np.divide(1,(2*np.pi*(DD-DD.getH())))
    pref2 = logmat - logmat.getH()
    pref3 = logmat2-logmat2.getH()

    prefactor = np.multiply(invmat,(pref2-pref3))


    Vc = LA.inv(V.getH())

    Gammam = Vc.getH()*Gam*Vc

    prefactor = np.multiply(prefactor,Gammam)

    den = V* prefactor * V.getH()

    return den


def sigmat(inds, value):
    sigma = np.asmatrix(np.zeros((nsto,nsto)),dtype=complex)

    for i in inds:
        sigma[i,i] = value

    return sigma

def getFock(bar, spin):
    # Set up Fock matrix and atom indexing
    # Note: positive indices are alpha/paired orbitals, negative are beta orbitals
    if spin == "r":
        locs = bar.ibfatm
        Fock = np.asmatrix(bar.matlist[AlphaFock].expand())
    elif spin == "ro" or spin == "u":
        locs = np.concatenate((bar.ibfatm, bar.ibfatm*-1))
        AFock = np.asmatrix(bar.matlist[AlphaFock].expand())
        BFock = np.asmatrix(bar.matlist[BetaFock].expand())
        Fock = np.block([[AFock, np.zeros(AFock.shape)], [np.zeros(BFock.shape), BFock]])
    elif spin == "g":
        locs = [loc for pair in zip(bar.ibfatm, bar.ibfatm*-1) for loc in pair]
        Fock = np.asmatrix(bar.matlist[AlphaFock].expand())
    else:
        raise ValueError("Spin treatment not recognized!")
    locs = np.array(locs)
    return Fock,locs

def getEnergies(bar, spin):
    if spin =="r":
        Alevels = bar.matlist[AlphaEnergies].expand()
        levels = [level for pair in zip(Alevels, Alevels) for level in pair]
    elif spin=="ro" or spin == "u":
        Alevels = bar.matlist[AlphaEnergies].expand()
        Blevels = bar.matlist[BetaEnergies].expand()
        levels = [level for pair in zip(Alevels, Blevels) for level in pair]
    elif spin=="g":
        levels = bar.matlist[AlphaEnergies].expand()
    else:
        raise ValueError("Spin treatment not recognized!")
    return np.sort(levels)*har_to_eV

def storeDen(bar, P, spin):
    nsto = len(bar.ibfatm)
    if spin=="r":
        P = np.real(np.array(P))
        PaO = qco.OpMat(AlphaSCFDen,P/2,dimens=(nsto,nsto))
        PaO.compress()
        bar.addobj(PaO)
    elif spin=="ro" or spin=="u":
        P = np.real(np.array(P))
        Pa = P[0:nsto, 0:nsto]
        Pb = P[nsto:, nsto:]
        PaO = qco.OpMat(AlphaSCFDen,Pa,dimens=(nsto,nsto))
        PbO = qco.OpMat(BetaSCFDen,Pb,dimens=(nsto,nsto))
        PaO.compress()
        PbO.compress()
        bar.addobj(PaO)
        bar.addobj(PbO)
    elif spin=="g":
        P = np.complex128(np.array(P))
        PaO = qco.OpMat(AlphaSCFDen,P,dimens=(nsto*2,nsto*2))
        PaO.compress()
        bar.addobj(PaO)
    else:
        raise ValueError("Spin treatment not recognized!")
        


###############################################################################
#
#                              Variables
#
###############################################################################

fn = "AuNanowire"
lContact = [5]
rContact = [6]
infile = fn + ".gjf"
outfile = fn + ".log"

fermi = -5.1
V = 1.0
sig = -5.1j
mu1 =  fermi + (V/2)
mu2 =  fermi - (V/2)

Emin = -15
Eminf = -1e5

damping =  0.1
conv = 1e-5
maxcycles=30

PP=[]
SS=[]
PPP=[]
count=[]
TotalE=[]


basis="lanl2dz"     #e.g. 6-31g(d,p), lanl2dz, sto-3g

spin = "r"          #"r" = restricted, "ro" = restricted open,
                    #"u" = unrestricted", "g" = generalized

func = "b3pw91"      # DFT functional or "hf" for hartree fock

method = spin + func

otherRoute = " 6d 10f"


##############################################################################
#
#                              Starting Code
#
###############################################################################

print('Calculation started at '+str(time.asctime()))
start_time = time.time()

bar = qcb.BinAr(debug=False,lenint=8,inputfile=infile)
print("Running Initial SCF...")

## RUN GAUSSIAN:
bar.update(model=method, basis=basis, toutput=outfile, dofock="scf", miscroute=otherRoute)

print("Done!")

Fock, locs = getFock(bar, spin)

print("ORBS:")
print(locs)
natoms=bar.natoms
icharg=bar.icharg
multip=bar.multip
nelec_i=bar.ne
iopcl=bar.iopcl
#iopcl =-1
c=bar.c
ian=bar.ian
print('Charge is: ', icharg)
print('Multiplicity is: ', multip)

# Calculate electric field
lAtom = bar.c[(lContact[0]-1)*3:lContact[0]*3]
rAtom = bar.c[(rContact[0]-1)*3:rContact[0]*3]
vec  = (rAtom-lAtom) #TODO: rotate coordinate system to make this the x-direction
dist = LA.norm(vec)*np.sign(vec[0])
field = int(V*V_to_au/(dist*0.0001));
print("E-field set to "+str(field)+" au")
if field>=0:
    otherRoute += " field=x+" +str(field)
else:
    otherRoute += " field=x" +str(field)


# Prepare Overlap, Identity, and Lowdin TF matricies
nsto = len(locs)
Omat = np.asmatrix(bar.matlist["OVERLAP"].expand())
if spin == "ro" or spin == "u":
    Overlap = np.block([[Omat, np.zeros(Omat.shape)],[np.zeros(Omat.shape),Omat]])
else:
    Overlap = Omat

X = np.asmatrix(linalg.fractional_matrix_power(Overlap, -0.5))

I = np.asmatrix(np.identity(nsto))

# Prepare Sigma matrices
lInd = np.where(abs(locs)==lContact)[0]
rInd = np.where(abs(locs)==rContact)[0]
sigma1 = sigmat(lInd, sig)
sigma2 = sigmat(rInd, sig)
sigma12 = sigma1 + sigma2

Gam1 = (sigma1 - sigma1.getH())*1j
Gam2 = (sigma2 - sigma2.getH())*1j


sigmaW1 = sigmat(lInd, -0.00001j)
sigmaW2 = sigmat(rInd, -0.00001j)
sigmaW12 = sigmaW1+sigmaW2

GamW1 = (sigmaW1 - sigmaW1.getH())*1j
GamW2 = (sigmaW2 - sigmaW2.getH())*1j

print('###################################')
print('Entering NEGF-SCF loop at: '+str(time.asctime()))
print('###################################')

#sys.exit("BREAK!")

########################     SCF LOOP    ##########################

Loop = True
Niter = 0
#while Niter < 1:
while Loop :

    print(Niter)

    print("SCF energy: ", bar.scalar("escf")) #line 269 of QCBinAr.py

    Total_E =  bar.scalar("escf")


    Fock,locs = getFock(bar, spin)
    Fbar = X * (Fock*har_to_eV + sigma12) * X
    GamBar1 = X * Gam1 * X
    GamBar2 = X * Gam2 * X

    D,V = LA.eig(np.asmatrix(Fbar))
    D = np.asmatrix(D).T
       
    #error check
    err =  np.float_(sum(np.imag(D)))
    if  err > 0:
        print('Imagine elements on diagonal of D are positive ------->  ', err)


    P1 = density(V, D, GamBar1, Emin, mu1)
    P2 = density(V, D, GamBar2, Emin, mu2)

    ################################################

    FbarW = X*(Fock*har_to_eV + sigmaW12)*X
    GamBarW1 = X*GamW1*X
    GamBarW2 = X*GamW2*X
    Dw,Vw = LA.eig(np.asmatrix(FbarW))
    Dw = np.asmatrix(Dw).T

    Pw = density(Vw, Dw, GamBarW1+GamBarW2, Eminf, Emin)

    ################################################

    P=P1 + P2 + Pw
    nelec = 2*np.trace(np.real(P))
    
    pshift = V.getH() * P * V
    occList = np.diag(np.real(pshift)) 
    EList = np.asarray(np.real(D)).flatten()
    inds = np.argsort(EList)

#    for pair in zip(occList[inds], EList[inds]):
#        print("Energy =", str(pair[1]), ", Occ =", str(pair[0]))


    count.append(Niter)
    TotalE.append(nelec)


####################      Check Convergence     ############################## 
    if Niter == 0:
        Dense_old = np.diagonal(P)
        P_old = P
        Total_E_Old = Total_E
        
    else:        
        print('Energy difference is: ', Total_E-Total_E_Old)
        Total_E_Old = Total_E
        
        Dense_diff = abs(np.diagonal(P) - Dense_old)
        MaxDP = max(Dense_diff)
        RMSDP = np.sqrt(np.mean(Dense_diff**2))
        
        print(f'MaxDP: {MaxDP:.2E} | RMSDP: {RMSDP:.2E}')



        PP.append(MaxDP)
        SS.append(Niter)

        if RMSDP<conv and MaxDP<conv:
            print('Convergence achieved!')
            Loop = False
        elif Niter >= maxcycles:
            print('WARNING: Convergence criterion net, maxcycles reached!')
            Loop = False
        
        Dense_old = np.diagonal(P)
        
        ###############################################################
 

        
    P = np.real(X * P * X)

    # APPLY DAMPING:
    if Niter == 0:
        Pback = P

    else:
        P = Pback + damping*(P - Pback)
        Pback = P

    #DEBUG:
#    print("BEFORE")
#    print(np.diag(bar.matlist['ALPHA DENSITY MATRIX'].expand()))
#    print("AFTER")
#    print(np.diag(P))
    storeDen(bar, P, spin)    

    
   
    bar.update(model= method, basis=basis, toutput=outfile, dofock="density", miscroute=otherRoute)
    
#    print(getEnergies(bar,spin))
    print('Total number of electrons: ', nelec)


#    os.system('The calculation end at os system' + time.asctime() + ' >> Log.log')

    Niter += 1

print('The calculation ends at: ',str(time.asctime()))

print('ENERGY LEVEL OCCUPATION:')
for pair in zip(occList[inds], EList[inds]):
    print(f"Energy = {pair[1]:9.3f} eV | Occ = {pair[0]:5.3f}")


############################ End of the Analytical Integral ###################
step = 0.1
E_min = -60
E_max = 120
H0 = X*Fock*har_to_eV*X
Elist=np.arange(E_min, E_max, step)
Tr=[]
DOS = []

for E in Elist:
    
    G = np.linalg.inv(E*I - H0 - sigma12)
    T = np.real(np.trace(Gam1*G*Gam2*G.getH()))
    DOS.append(abs(np.imag(np.trace(G)/np.pi)))
    Tr.append(T)
    

fL = 1/(np.exp((Elist-mu1)/kT) + 1)
fR = 1/(np.exp((Elist-mu2)/kT) + 1)
print("Fermi Energy window = "+str(np.trapz(fL-fR, Elist)) + " eV")

Curr = eoverh * np.trapz(Tr*(fL-fR), Elist)
if spin == 'r':
    Curr=Curr*2;
print(f"Current = {Curr:.3E} A")

savetxt('Iteration.txt', SS, delimiter=',')
savetxt('DM_max.txt', PP, delimiter=',')
savetxt('Electron.txt', TotalE, delimiter=',')
savetxt('Transmission.txt', Tr, delimiter=',')
savetxt('Energy.txt', Elist, delimiter=',')
savetxt('DOS.txt', DOS, delimiter=',')



plt.subplot(221)
plt.title('Fermi level is: '+ str(mu1) + 'eV   sigma is: ' +str(sig) + "eV\n")
plt.ylabel(r'Max Change in $\rho$')
plt.plot(SS, PP, color='g', linestyle='solid' ,linewidth = 1, marker='x')

plt.subplot(223)
plt.ylabel('Total # of electrons')
plt.xlabel('Iteration')
plt.plot(count, TotalE, color='black', linestyle='solid' ,linewidth = 1, marker='o')


plt.subplot(222)
plt.title(' Method: '+ method + '/' + basis + "\n")
plt.ylabel('Transmission')
plt.semilogy(Elist, Tr, color='b', linestyle='solid' ,linewidth = 1, marker='')

for (occ, E) in zip(occList[inds], EList[inds]):
    if occ > 1.0:
        occ = 1.0
    plt.plot(E, 10, color ='r', marker='o', alpha=occ)

plt.xlim([Emin, 0])

plt.subplot(224)
plt.xlabel('Energy (eV)')
plt.ylabel('Density of States')
plt.semilogy(Elist, DOS, color='g', linestyle='solid' ,linewidth = 1, marker='')
plt.plot(Elist, fL-fR+min(DOS))


plt.xlim([Emin, 0])
plt.show()


print('')
print('##########################################')
print("--- %s seconds ---" % (time.time() - start_time))
print('')
print('The calculation end at', time.asctime())


