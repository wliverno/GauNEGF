#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sajjad
"""
import numpy as np
#import unittest
from numpy import linalg as LA
from scipy import linalg
import time
#import os
#import matplotlib.pyplot as plt
#import math
#from goto import goto, label
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu


df=open('text_file','w')
df.write('The calculation strat at')
df.write(str(time.asctime()))
df.write('\n')
#print('The calculation strat at', time.asctime())
start_time = time.time()

AlphaDensity = "Alpha SCF Density Matrix"
BetaDensity = "Beta SCF Density Matrix"
AlphaFock = "Alpha Fock Matrix"
BetaFock = "Beta Fock Matrix"
AlphaMOs = "Alpha MO Coefficients"
BetaMOs = "Beta MO Coefficients"

###############################################################################
#
#                             Functions
#
###############################################################################
def impor():

  bar = qcb.BinAr(debug=False,lenint=8,inputfile="PDT_cluster.gjf")  
  bar.update(model=method, basis=basis, matfo=fname, dofock="scf")
  
  return bar

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

###############################################################################

def export(key, mat):
    L = mat[np.triu_indices(nsto)].T
    L = np.array(L).reshape((len(L),))
    bar.addobj(qco.OpMat(key, L))

###############################################################################

def Gamma(index, value):
    Gamm = np.asmatrix(np.zeros((nsto,nsto)),dtype=complex)
    
    for i in range(0,index):
        Gamm[i,i] = value
        
    return Gamm
        
###############################################################################
#
#                              Variables
#
###############################################################################
mu1 =  -5.1
mu2 =  -5.1
sig = -0.1j
Nbasis= 66 #12 atoms each has 22 orbitals

Emin = -15
Eminf = -1e5
har_to_eV = 27.212
e = 1.60216*(1e-19)
h = 6.582*(1e-16)
damping =  0.05

PP=[]
SS=[]
PPP=[]
count=[]
TotalE=[]


basis="lanl2dz"  #3-21G  #LanL2DZ
method = "b3pw91" #for DFT = b3lyp, for hartree fock = hf

name = "Transport"     #select a name for file to use (SA)
fname = name + qcu.file_ext() #select the extenstion for the file (SA)

###############################################################################
#
#                              Starting Code
#
###############################################################################

bar = impor()

natoms=bar.natoms
icharg=bar.icharg
multip=bar.multip
nelec_i=bar.ne
iopcl=bar.iopcl
#iopcl =-1
c=bar.c
ian=bar.ian

Overlap = np.asmatrix(bar.matlist["OVERLAP"].expand())
X = np.asmatrix(linalg.fractional_matrix_power(Overlap, -0.5))

nsto = len(bar.matlist["OVERLAP"].expand()) #Size of matrix
I = np.asmatrix(np.identity(nsto))


sigma1 = Gamma(Nbasis, sig)
sigma2 = Gamma(Nbasis, sig)
sigma2 = np.rot90(np.rot90(sigma2))
sigma12 = sigma1 + sigma2

Gam1 = (sigma1 - sigma1.getH())*1j
Gam2 = (sigma2 - sigma2.getH())*1j

sigmaW1 = Gamma(Nbasis, -0.00001j)
sigmaW2 = Gamma(Nbasis, -0.00001j)
sigmaW2 = np.rot90(np.rot90(sigmaW2))
sigmaW12 = sigmaW1+sigmaW2

GamW1 = (sigmaW1 - sigmaW1.getH())*1j
GamW2 = (sigmaW2 - sigmaW2.getH())*1j

df.write('###################################')
df.write('\n')
df.write('Entering Analytical Integration at: ')
df.write(str(time.asctime()))
df.write('\n')
df.write('###################################')
df.write('\n')
df.close()
########################     Analytical Integral     ##########################

Loop = True
Niter = 0
#while Niter < 1:
while Loop :
    df=open('text_file','a')
    
#    print(Niter)

#    print("SCF energy: ", bar.scalar("escf")) #line 269 of QCBinAr.py
    df.write(str(Niter))
    df.write('\n')
    df.write('SCF energy: ')
    df.write(str(bar.scalar("escf")))
    df.write('\n')
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
    
    
    #print('Total electron from NEGF is: ', nelec)
    df.write('Total electron from NEGF is: ')
    df.write(str(nelec))
    df.write('\n')
#    print('Charge is: ', icharg)
#    print('Multiplicity is: ', multip)

#####################      Check Convergence     ############################## 
    if Niter == 0:
        Dense_old = np.diagonal(P)
        P_old = P
        Total_E_Old = Total_E
        
    else:        
        #print('Energy difference is: ', Total_E-Total_E_Old)
        #Total_E_Old = Total_E
        
        Dense_diff = abs(np.diagonal(P) - Dense_old)
        
        #print('convergence: ',max(Dense_diff))
        df.write('convergence: ')
        df.write(str(max(Dense_diff)))
        df.write('\n')

        PP.append(max(Dense_diff))
        SS.append(Niter)

        if all(ii<= 1e-5 for ii in Dense_diff):
            print('loop done')
            df.write('loop done')
            Loop = False
        
        Dense_old = np.diagonal(P)
        
        ###############################################################
 

        
    P = np.real(X * P *X)
    P = np.array(P)/2
    
    
    PaO = qco.OpMat('Alpha SCF Density Matrix',P,dimens=(nsto,nsto))
    PaO.compress()
    bar.addobj(PaO)
    

    
   
    bar.update(model= method, basis=basis, toutput="{0}.txt".format(name), dofock="density")
    
    
    print('Total electron from Gaussian is: ', bar.ne)


    df.write('Total electron from Gaussian is: ')
    df.write(str(bar.ne))
    df.write('\n')
#    os.system('The calculation end at os system' + time.asctime() + ' >> Log.log')

    Niter += 1
    df.close()

df=open('text_file','a')
df.write('The calculation ends at: ')
df.write(str(time.asctime()))
df.write('\n')
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

df.close()
#os.system('The calculation end at os system'+ time.asctime() > Log.log)
#df.close




