import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import sys
import time
import matplotlib.pyplot as plt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

from matTools import *

# Matrix Headers
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

# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kT = 0.025              # eV @ 20degC
V_to_au = 0.03675       # Volts to Hartree/elementary Charge



class NEGF(object):
    def __init__(self, fn, basis="lanl2dz", func="b3pw91", spin="r", chkInit=True, route=""):
        # Set up variables
        self.ifile = fn + ".gjf"
        self.chkfile = fn + ".chk"
        self.ofile = fn + ".log"
        self.func = func
        self.basis = basis
        self.method= spin+func
        self.otherRoute = route     # Other commands that are needed in Gaussian
        self.spin = spin
        self.energyDep = False;
        self.Total_E_Old=9999.0;
        
        #Default Integration Limits (from Damle thesis)
        self.Emin = -15
        self.Eminf = -1e5

    
        # Start calculation: Load Initial Matrices from Gaussian
    
        print('Calculation started at '+str(time.asctime()))
        self.start_time = time.time()
    
        self.bar = qcb.BinAr(debug=False,lenint=8,inputfile=self.ifile)
        self.bar.write('debug.baf')
        self.runDFT(chkInit)

        # Prepare self.F, Density, self.S, and TF matrices
        self.P = getDen(self.bar, spin)
        self.F, self.locs = getFock(self.bar, spin)
        self.nsto = len(self.locs)
        Omat = np.array(self.bar.matlist["OVERLAP"].expand())
        if spin == "ro" or spin == "u":
            self.S = np.block([[Omat, np.zeros(Omat.shape)],[np.zeros(Omat.shape),Omat]])
        else:
            self.S = Omat
        
        self.X = np.array(fractional_matrix_power(self.S, -0.5))
    
        self.I = np.array(np.identity(self.nsto))
        #H0 = self.X*self.F*har_to_eV*self.X.conj().T
    
        print("ORBS:")
        print(self.locs)
        self.Total_E =  self.bar.scalar("escf")
        self.updateN()
        print('Expecting', str(self.bar.ne), 'electrons')
        print('Actual: ', str(self.nelec), 'electrons')
        print('Charge is:', self.bar.icharg)
        print('Multiplicity is:', self.bar.multip)
        print("Initial SCF energy: ", self.Total_E)
        print('###################################')

    def runDFT(self, chkInit):
        if chkInit:
            try:
                self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock=True,chkname=self.chkfile, miscroute=self.otherRoute)
                print('Checking '+self.chkfile+' for saved data...');
            except:
                print('Checkpoint not loaded, running full SCF...');
                self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="scf",chkname=self.chkfile, miscroute=self.otherRoute)
        
            print("Done!")
            self.F, self.locs = getFock(self.bar, self.spin)
            
        else:
            print('Using default Harris DFT guess to initialize...')
            self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="GUESS",chkname=self.chkfile, miscroute=self.otherRoute)
            self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock=True, miscroute=self.otherRoute)
            print("Done!")
            self.F, self.locs = getFock(self.bar, self.spin)
    
    def updateN(self):
        self.nelec = np.real(np.trace(self.P))
        return self.nelec
    def setFock(self, F_):
        self.F = F_ 
    def setVoltage(self, fermi, qV):
        self.fermi = fermi
        self.Emin = fermi-15;
        self.Eminf = fermi-1e5;
        self.qV = qV
        self.mu1 =  fermi + (qV/2)
        self.mu2 =  fermi - (qV/2)
    
    
        # Calculate electric field to apply during SCF, apply between contacts
        # TODO: average location of all contact atoms (currently just first atom)
        lAtom = self.bar.c[int(self.lContact[0]-1)*3:int(self.lContact[0])*3]
        rAtom = self.bar.c[int(self.rContact[0]-1)*3:int(self.rContact[0])*3]
        vec  = np.array(lAtom-rAtom)
        dist = LA.norm(vec)
        vecNorm = vec/dist
        
        if dist == 0:
            print("WARNING: left and right contact atoms identical, E-field set to zero!")
            field = [0,0,0]
        else:
            field = -1*vecNorm*qV*V_to_au/(dist*0.0001)
        self.bar.scalar("X-EFIELD", int(field[0]))
        self.bar.scalar("Y-EFIELD", int(field[1]))
        self.bar.scalar("Z-EFIELD", int(field[2]))
        print("E-field set to "+str(LA.norm(field))+" au")
        
    def setContacts(self, lContact, rContact):
        self.lContact=np.array(lContact)
        self.rContact=np.array(rContact)
        lInd = np.where(np.isin(abs(self.locs), self.lContact))[0]
        rInd = np.where(np.isin(abs(self.locs), self.rContact))[0]
        return lInd, rInd
    
    def setSigma(self, lContact, rContact, sig=-0.1j, sig2=False): 
        lInd, rInd = self.setContacts(lContact, rContact)
        self.sigma1 = formSigma(lInd, sig, self.nsto, self.S)
        if not isinstance(sig2, bool):
            self.sigma2 = formSigma(rInd, sig2, self.nsto, self.S)
        else:
            self.sigma2 = formSigma(rInd, sig, self.nsto, self.S)
        
        self.sigma12 = self.sigma1 + self.sigma2
    
        print('Max imag sigma:', str(np.max(np.abs(np.imag(self.sigma12)))));
        self.Gam1 = (self.sigma1 - self.sigma1.conj().T)*1j
        self.Gam2 = (self.sigma2 - self.sigma2.conj().T)*1j
    
        sigWVal = -0.00001j #Based on Damle Code
        self.sigmaW1 = formSigma(lInd, sigWVal, self.nsto, self.S)
        self.sigmaW2 = formSigma(rInd, sigWVal, self.nsto, self.S)
        self.sigmaW12 = self.sigmaW1+self.sigmaW2
    
        self.GamW1 = (self.sigmaW1 - self.sigmaW1.conj().T)*1j
        self.GamW2 = (self.sigmaW2 - self.sigmaW2.conj().T)*1j
    
    def getSigma(self, E):
        return (self.sigma1, self.sigma2)

    def FockToP(self):
        # Prepare Variables for Analytical Integration
        X = np.array(self.X)
        self.F, self.locs = getFock(self.bar, self.spin)
        Fbar = X @ (self.F*har_to_eV + self.sigma12) @ X
        GamBar1 = X @ self.Gam1 @ X
        GamBar2 = X @ self.Gam2 @ X

        D,V = LA.eig(np.array(Fbar))
           
        err =  np.float_(sum(np.imag(D)))
        if  err > 0:
            print('Imaginary elements on diagonal of D are positive ------->  ', err)

        FbarW = X@(self.F*har_to_eV + self.sigmaW12)@X
        GamBarW1 = X@self.GamW1@X
        GamBarW2 = X@self.GamW2@X
        Dw,Vw = LA.eig(np.array(FbarW))
        
        # Calculate Density
        #P1 = densityGrid(Fbar, GamBar2, self.Emin, self.mu1)
        #P2 = densityGrid(Fbar, GamBar2, self.Emin, self.mu2)
        P1 = density(V, D, GamBar1, self.Emin, self.mu1)
        P2 = density(V, D, GamBar2, self.Emin, self.mu2)
        Pw = density(Vw, Dw, GamBarW1+GamBarW2, self.Eminf, self.Emin)
        
        #P1_ = densityGrid(Fbar, GamBar1, self.Emin, self.mu1)
        #print(np.diag(P1_-P1))
        
        P = P1 + P2 + Pw
        
        # Calculate Level Occupation, Lowdin TF,  Return
        pshift = V.conj().T @ P @ V
        self.P = X@P@X
        occList = np.diag(np.real(pshift)) 
        EList = np.array(np.real(D)).flatten()
        inds = np.argsort(EList)
        
        #DEBUG:
        #for pair in zip(occList[inds], EList[inds]):                       
        #    print("Energy=", str(pair[1]), ", Occ=", str(pair[0]))

        return EList[inds], occList[inds]

    
    # Use Gaussian to calculate the Density Matrix
    def PToFock(self, damping, Edamp):
        # Store Old Density Info
        Pback = getDen(self.bar, self.spin)
        Dense_old = np.diagonal(Pback)
        Dense_diff = abs(np.diagonal(self.P) - Dense_old)
        
        ##DEBUG
        #print('DEBUG: Compare Density')
        #print(np.real(np.diag(self.P)[:10]))
        #print(np.real(np.diag(Pback)[:10]))

        
        # Apply Damping, store to Gaussian matrix
        if self.Total_E_Old<self.Total_E and Edamp:
            print("APPLYING EDAMP...")
            self.P = Pback + 0.1*damping*(self.P - Pback)
        else:
            self.P = Pback + damping*(self.P - Pback)
        storeDen(self.bar, self.P, self.spin) 
        self.updateN()
        print('Total number of electrons: ', self.nelec)
        
        # Run Gaussian, update SCF Energy
        self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="DENSITY", miscroute=self.otherRoute)
        self.Total_E_Old = self.Total_E.copy()
        self.Total_E  = self.bar.scalar("escf")
        print("SCF energy: ", self.Total_E)

        # Convergence variables: dE, RMSDP and MaxDP
        dE = self.Total_E-self.Total_E_Old
        MaxDP = max(Dense_diff)
        RMSDP = np.sqrt(np.mean(Dense_diff**2))
        print('Energy difference is: ', dE)
        print(f'MaxDP: {MaxDP:.2E} | RMSDP: {RMSDP:.2E}')
        return dE, RMSDP, MaxDP

    def SCF(self, conv=1e-5, damping=0.1, maxcycles=100, Edamp=False, plot=False):
        
        Loop = True
        Niter = 0
        PP=[]
        count=[]
        TotalE=[]
        print('Entering NEGF-SCF loop at: '+str(time.asctime()))
        print('###################################')
        while Loop:
            print()
            print('Iteration '+str(Niter)+':')
            # Fock --> P --> Fock
            EList, occList = self.FockToP()
            dE, RMSDP, MaxDP = self.PToFock(damping, Edamp)
            
            # Write monitor variables
            TotalE.append(self.Total_E)
            count.append(Niter)
            PP.append(self.nelec)

            if RMSDP<conv and MaxDP<conv and abs(dE)<conv:
                print('##########################################')
                print('Convergence achieved after '+str(Niter)+' iterations!')
                Loop = False
            elif Niter >= maxcycles:
                print('##########################################')
                print('WARNING: Convergence criterion not met, maxcycles reached!')
                Loop = False
            Niter += 1

        if plot==True:
            # Plot convergence data 
            plt.subplot(311)
            plt.title('Fermi level is: '+ str(self.fermi) + 'eV   $\Delta V=$'+str(self.qV)+'V Method: '+ self.method + '/' + self.basis + "\n")
            plt.ylabel(r'Max Change in $\rho$')
            plt.plot(count, PP, color='g', linestyle='solid' ,linewidth = 1, marker='x')

            plt.subplot(312)
            plt.ylabel('Total # of electrons')
            plt.xlabel('Iteration')
            plt.plot(count, TotalE, color='black', linestyle='solid' ,linewidth = 1, marker='o')

            plt.subplot(313)
            plt.plot(EList, occList, color ='r', marker='o')
            plt.xlabel('Energy')
            plt.ylabel('Occupation')
            plt.xlim([self.Emin, 0])
            plt.savefig('debug.png')
            plt.close()
 
        print("--- %s seconds ---" % (time.time() - self.start_time))
        print('')
        print('SCF Loop exited at', time.asctime())
        
        print('=========================')
        print('ENERGY LEVEL OCCUPATION:')
        print('=========================')
        for pair in zip(occList, EList):
            print(f"Energy = {pair[1]:9.3f} eV | Occ = {pair[0]:5.3f}")
        print('=========================')
        return count, PP, TotalE
    
    def writeChk(self):
        print('Writing to checkpoint file...') 
        self.bar.writefile(self.chkfile)
        print(self.chkfile+' written!') 
    
    def saveMAT(self, matfile="out.mat"):
        (sigma1, sigma2) = self.getSigma(self.fermi)
        # Save data in MATLAB .mat file
        matdict = {"F":self.F*har_to_eV, "sig1": sigma1, "sig2": sigma2, "S": self.S, "fermi": self.fermi, "qV": self.qV, "spin" : self.spin}
        io.savemat(matfile, matdict)
        return self.X@self.F@self.X

