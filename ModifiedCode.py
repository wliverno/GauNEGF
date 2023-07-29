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
#import matplotlib.pyplot as plt
#import math
#from goto import goto, label
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu



AlphaDensity = "Alpha SCF Density Matrix"
BetaDensity = "Beta SCF Density Matrix"
AlphaFock = "Alpha Fock Matrix"
BetaFock = "Beta Fock Matrix"
AlphaMOs = "Alpha MO Coefficients"
BetaMOs = "Beta MO Coefficients"
AlphaEnergies = "Alpha Orbital Energies"
BetaEnergies = "Beta Orbital Energies"
har_to_eV = 27.211386
e = 1.60216*(1e-19)
h = 6.582*(1e-16)

###############################################################################
#
#                             Functions
#
###############################################################################

def density(V, D, Gam, Emin, mu):
    Nd = len(Fock)
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

def export(key, mat):
    L = mat[np.triu_indices(nsto)].T
    L = np.array(L).reshape((len(L),))
    bar.addobj(qco.OpMat(key, L))

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
    return levels*har_to_eV


###############################################################################
#
#                              Variables
#
###############################################################################

fn = "ethane"
lContact = [1]
rContact = [5]
infile = fn + ".gjf"
outfile = fn + ".log"


mu1 =  -5.1
mu2 =  -5.1
sig = -0.1j

Emin = -15
Eminf = -1e5
damping =  1

PP=[]
SS=[]
PPP=[]
count=[]
TotalE=[]


basis="lanl2dz"     #e.g. 6-31g(d,p), lanl2dz, sto-3g

spin = "g"          #"r" = restricted, "ro" = restricted open,
                    #"u" = unrestricted", "g" = generalized

func = "b3lyp"      # DFT functional or "hf" for hartree fock

method = spin + func

otherRoute = " 6d 10f "


##############################################################################
#
#                              Starting Code
#
###############################################################################

print('The calculation starts at'+str(time.asctime()))
start_time = time.time()

bar = qcb.BinAr(debug=False,lenint=8,inputfile=infile)
print("Running Initial SCF...")

## RUN GAUSSIAN:
bar.update(model=method, basis=basis, toutput=outfile, dofock="scf", miscroute=otherRoute)

print("Done!")
print("IBFATM:")
print(bar.ibfatm)

Fock, locs = getFock(bar, spin)

print("FOCK:")
print(np.diag(Fock))
print(locs)
natoms=bar.natoms
icharg=bar.icharg
multip=bar.multip
nelec_i=bar.ne
iopcl=bar.iopcl
#iopcl =-1
c=bar.c
ian=bar.ian

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
print(np.diag(sigma1))
print(np.diag(sigma2))
sigma12 = sigma1 + sigma2

Gam1 = (sigma1 - sigma1.getH())*1j
Gam2 = (sigma2 - sigma2.getH())*1j


sigmaW1 = sigmat(lInd, -0.00001j)
sigmaW2 = sigmat(rInd, -0.00001j)
sigmaW12 = sigmaW1+sigmaW2

GamW1 = (sigmaW1 - sigmaW1.getH())*1j
GamW2 = (sigmaW2 - sigmaW2.getH())*1j

print('###################################')
print('Entering Analytical Integration at: '+str(time.asctime()))
print('###################################')

#sys.exit("BREAK!")

########################     Analytical Integral     ##########################

Loop = True
Niter = 0
#while Niter < 1:
while Loop :

    print(Niter)

    print("SCF energy: ", bar.scalar("escf")) #line 269 of QCBinAr.py

    Total_E =  bar.scalar("escf")


    Fock = np.asmatrix(bar.matlist["ALPHA FOCK MATRIX"].expand())

    if Niter == 0:
        Fback = Fock

    else:
        Fock = Fback + damping*(Fock - Fback)
        Fback = Fock


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



    count.append(Niter)
    TotalE.append(nelec)


    print('Total electron from NEGF is: ', nelec)
    print('Charge is: ', icharg)
    print('Multiplicity is: ', multip)

####################      Check Convergence     ############################## 
    if Niter == 0:
        Dense_old = np.diagonal(P)
        P_old = P
        Total_E_Old = Total_E
        
    else:        
        print('Energy difference is: ', Total_E-Total_E_Old)
        Total_E_Old = Total_E
        
        Dense_diff = abs(np.diagonal(P) - Dense_old)
        
        print('convergence: ',max(Dense_diff))


        PP.append(max(Dense_diff))
        SS.append(Niter)

        if all(ii<= 1e-5 for ii in Dense_diff):
            print('loop done')
            Loop = False
        
        Dense_old = np.diagonal(P)
        
        ###############################################################
 

        
    P = np.real(X * P *X)
    P = np.array(P)/2
    
    
    PaO = qco.OpMat('Alpha SCF Density Matrix',P,dimens=(nsto,nsto))
    PaO.compress()
    bar.addobj(PaO)
    

    
   
    bar.update(model= method, basis=basis, toutput=outfile, dofock="density")
    
    
    print('Total electron from Gaussian is: ', bar.ne)


#    os.system('The calculation end at os system' + time.asctime() + ' >> Log.log')

    Niter += 1

print('The calculation ends at: ',str(time.asctime()))

############################ End of the Analytical Integral ###################
Count = 1
Curr = 0
step = 0.1
E_min = -60
E_max = 120
N = (E_max - E_min)/step
H0 = X*Fock*har_to_eV*X
E_list=[]
Tr=[]
DOS = []

for E in np.arange(E_min, E_max, step):
    
    G = np.linalg.inv(E*I - H0 - sigma12)
    T = np.trace(Gam1*G*Gam2*G.getH())
    DOS.append(abs(np.imag(np.trace(G)/np.pi)))
    E_list.append(E)
    Tr.append(T)
    
    if E> mu1 and E<=mu2:
        Df12=1
    else:
        Df12=0
    if (count == 1 or count == N):
    
        Curr_tmp = np.real(T)* Df12
    else:
        Curr_tmp = 2*np.real(T)* Df12
#    print('local current is: ', Curr_tmp)

    Curr += Curr_tmp
 #   print(count)
    Count += 1
    

savetxt('Iteration.txt', SS, delimiter=',')
savetxt('DM_max.txt', PP, delimiter=',')
savetxt('Electron.txt', TotalE, delimiter=',')
savetxt('Transmission.txt', Tr, delimiter=',')
savetxt('Energy.txt', E_list, delimiter=',')
savetxt('DOS.txt', DOS, delimiter=',')


eign = LA.eigvals(H0)
'''
plt.figure(1)
plt.title('Fermi level is: '+ str(mu1) + '(eV)   sigma is: ' 
          +str(sig) + '(eV)     Method: '+ method + "\n")
plt.ylabel('Max elemnt in DM matrix')
plt.xlabel('Iteration')
plt.plot(SS, PP, color='g', linestyle='solid' ,linewidth = 1, marker='x')

plt.figure(2)
plt.title('Fermi level is: '+ str(mu1) + '(eV)   sigma is: ' 
          +str(sig) + '(eV)     Method: '+ method + "\n")
plt.ylabel('Total # of electrons')
plt.xlabel('Iteration')
plt.plot(count, TotalE, color='black', linestyle='solid' ,linewidth = 1, marker='o')


plt.figure(3)
plt.ylabel('Transmission')
plt.xlabel('Energy (eV)')
plt.plot(E_list, Tr, color='b', linestyle='solid' ,linewidth = 1, marker='')

plt.figure(4)
plt.plot(E_list, DOS, color='g', linestyle='solid' ,linewidth = 1, marker='')
for i in range (0, len(eign)-1):
    if E_min <= eign[i] <= E_max:

        plt.plot(eign[i], 0, color ='r', marker='o')

plt.show()
'''

print('')
print('##########################################')
print("--- %s seconds ---" % (time.time() - start_time))
print('')
print('The calculation end at', time.asctime())


