import numpy as np
from scipy import linalg as LA
from scipy.linalg import fractional_matrix_power
import sys
import time
import matplotlib.pyplot as plt

# Gaussian interface packages
from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu

# Other packages
from matTools import *
from transport import DOS
from fermiSearch import DOSFermiSearch
from density import * 


# CONSTANTS:
har_to_eV = 27.211386   # eV/Hartree
eoverh = 3.874e-5       # A/eV
kT = 0.025              # eV @ 20degC
V_to_au = 0.03675       # Volts to Hartree/elementary Charge



class NEGF(object):
    def __init__(self, fn, basis="lanl2dz", func="b3pw91", spin="r", fullSCF=True, route="", nPulay=4):
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
        
        #Default Integration Limits
        self.Eminf = -1e5
        self.fSearch = None
        self.fermi = None
        self.updFermi = False
    
        # Start calculation: Load Initial Matrices from Gaussian
        print('Calculation started at '+str(time.asctime()))
        self.start_time = time.time()
        self.bar = qcb.BinAr(debug=False,lenint=8,inputfile=self.ifile)
        self.bar.write('debug.baf')
        self.runDFT(fullSCF)
        self.nae = int(self.bar.ne/2 + (self.bar.multip-1)/2)
        self.nbe = int(self.bar.ne/2 - (self.bar.multip-1)/2)

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
        
        # Set Emin from orbitals
        orbs, _ = LA.eig(self.X@self.F@self.X)
        self.Emin = min(orbs)*har_to_eV - 5
        self.convLevel = 9999
        self.MaxDP = 9999 

        # Pulay Mixing Initialization
        self.pList = np.array([self.P for i in range(nPulay)], dtype=complex)
        self.DPList = np.ones((nPulay, self.nsto))*1e4
        self.pMat = np.ones((nPulay+1, nPulay+1))*-1
        self.pMat[-1, -1] = 0
        self.pB = np.zeros(nPulay+1)
        self.pB[-1] = -1
        
        # DFT Info dump
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
    
    #Run DFT in Gaussian, default run full SCF to convergence, otherwise use Harris guess only
    def runDFT(self, fullSCF=True):
        if fullSCF:
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
    
    # Update number of Electrons (self.nelec) from density matrix
    def updateN(self):
        nOcc =  np.real(np.trace(self.P@self.S))
        if self.spin == 'r':
            self.nelec = 2*nOcc
        else:
            self.nelec = nOcc
        return self.nelec

    def setFock(self, F_):
        self.F = np.array(F_)/har_to_eV

    def setDen(self, P_):
        self.P = P_ 
        storeDen(self.bar, self.P, self.spin)
        self.updateN() 
        print(f'Density matrix loaded, nelec = {self.nelec:.2f} electrons')
        self.PToFock()

    def getHOMOLUMO(self):
        orbs, _ = LA.eig(self.X@self.F@self.X)
        orbs = np.sort(orbs)*har_to_eV
        if self.spin=='r':
            homo_lumo = orbs[self.nae-1:self.nae+1]
        else:
            homo_lumo = orbs[self.nae+self.nbe-1:self.nae+self.nbe+1]
        return homo_lumo
                
    # Set voltage and fermi energy, update electric field applied and integral limits
    def setVoltage(self, qV, fermi=np.nan, Emin=None, Eminf=None):
        # Set Fermi Energy
        if np.isnan(fermi):
            self.updFermi = True
            if self.fSearch is None:
                if self.fermi is None:
                    # Set initial fermi energy as (HOMO + LUMO)/2
                    homo_lumo = self.getHOMOLUMO()
                    print(f'Setting initial Fermi energy between HOMO ({homo_lumo[0]:.2f} eV) and LUMO ({homo_lumo[1]:.2f} eV)')
                    fermi = sum(homo_lumo)/2
                else:
                    fermi = self.fermi
                self.fSearch = DOSFermiSearch(fermi, self.nae+self.nbe)#,numpoints=1)
            else:
                n_curr = self.updateN()
                dosFunc = lambda E: DOS([E], self.F*har_to_eV, self.S, self.getSigma(E)[0], self.getSigma(E)[1])[0][0]
                fermi = self.fSearch.step(dosFunc, n_curr)
        else:
            self.updFermi = False
            # Set Integration limits
        if Emin!=None:
            self.Emin = Emin
        if Eminf!=None:
            self.Eminf = Eminf

        self.fermi = fermi
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
        if not self.updFermi:
            print("E-field set to "+str(LA.norm(field))+" au")
    
    # Set contacts based on atom orbital locations
    def setContacts(self, lContact=-1, rContact=-1):
        if lContact == -1:
            self.lContact = np.arange(self.nsto)
        else:
            self.lContact=np.array(lContact)
        if rContact == -1:
            self.rContact = np.arange(self.nsto)
        else:
            self.rContact=np.array(rContact)
        lInd = np.where(np.isin(abs(self.locs), self.lContact))[0]
        rInd = np.where(np.isin(abs(self.locs), self.rContact))[0]
        return lInd, rInd
    
    # Set self-energies of left and right contacts (TODO: n>2 terminal device?)
    def setSigma(self, lContact, rContact, sig=-0.1j, sig2=False): 
        lInd, rInd = self.setContacts(lContact, rContact)
        #Is there a second sigma matrix? If not, copy the first one
        if isinstance(sig2, bool):
            sig2 = sig + 0.0
       
        # Sigma can be a value, list, or matrix
        if np.ndim(np.array(sig)) == 0  and np.ndim(np.array(sig2)) == 0:
            pass
        elif np.ndim(sig) == 1 and np.ndim(sig2) == 1:
            if len(sig) == len(lInd) and len(sig2) == len(rInd):
                pass
            elif len(sig) == len(lInd)/2 and len(sig2) == len(rInd)/2:
                if self.spin=='g':
                    sig = np.kron(sig, [1, 1])
                    sig2 = np.kron(sig2, [1, 1])
                elif self.spin=='ro' or self.spin=='u':
                    sig = np.kron([1, 1], sig)
                    sig2 = np.kron([1, 1], sig2)
            else:
                raise Exception('Sigma matrix dimension mismatch!')
        elif np.ndim(sig) == 2 and np.ndim(sig2) == 2:
            if len(sig) == len(lInd) and len(sig2) == len(rInd):
                pass
            elif len(sig) == len(rInd)/2 and len(sig2) == len(rInd)/2:
                if self.spin=='g':
                    sig = np.kron(sig, np.eye(2))
                    sig2 = np.kron(sig2, np.eye(2))
                elif self.spin=='ro' or self.spin=='u':
                    sig = np.kron(np.eye(2), sig)
                    sig2 = np.kron(np.eye(2), sig2)
            else:
                raise Exception('Sigma matrix dimension mismatch!')
            
        else:
            raise Exception('Sigma matrix dimension mismatch!')
        
        # Store Variables
        self.rInd = rInd
        self.lInd = lInd 
        self.sigma1 = formSigma(lInd, sig, self.nsto, self.S)
        self.sigma2 = formSigma(rInd, sig2, self.nsto, self.S)
        
        if self.sigma1.shape != self.F.shape or self.sigma2.shape != self.F.shape:
            raise Exception(f'Sigma size mismatch! F shape={self.F.shape},'+
                            f' sigma shapes={self.sigma1.shape}, {self.sigma2.shape}')
        
        self.sigma12 = self.sigma1 + self.sigma2
    
        print('Max imag sigma:', str(np.max(np.abs(np.imag(self.sigma12)))));
        self.Gam1 = (self.sigma1 - self.sigma1.conj().T)*1j
        self.Gam2 = (self.sigma2 - self.sigma2.conj().T)*1j
        
    def getSigma(self, E=0): #E only used by NEGFE() object, function inherited
        return (self.sigma1, self.sigma2)

    # Calculate density matrix from stored Fock matrix
    def FockToP(self):
        # Prepare Variables for Analytical Integration
        X = np.array(self.X)
        self.F, self.locs = getFock(self.bar, self.spin)
        Fbar = X @ (self.F*har_to_eV + self.sigma12) @ X
        GamBar1 = X @ self.Gam1 @ X
        GamBar2 = X @ self.Gam2 @ X
        
        
        D,V = LA.eig(np.array(Fbar))
        Vc = LA.inv(V.conj().T)
           
        err =  np.float_(sum(np.imag(D)))
        if  err > 0:
            print('Imaginary elements on diagonal of D are positive ------->  ', err)
        
        #Update Fermi Energy (if not constant)
        if self.updFermi:
            Nexp = self.bar.ne
            conv = min(self.convLevel, 1e-3)
            if self.spin=='r':
                Nexp/=2
            self.fermi = bisectFermi(V,Vc,D,GamBar1+GamBar2,Nexp,conv, self.Eminf)
            print(f'Fermi Energy set to {self.fermi:.2f} eV')

        #Integrate to get density matrix
        if self.qV == 0:
            P = density(V, Vc, D, GamBar1+GamBar2, self.Eminf, self.fermi)
        else:
            P1 = density(V, Vc, D, GamBar1, self.Eminf, self.mu1)
            P2 = density(V, Vc, D, GamBar2, self.Eminf, self.mu2)
            P = P1 + P2 #+ Pw
        
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

    def PMix(self, damping, Pulay=False):
        # Store Old Density Info
        Pback = getDen(self.bar, self.spin)
        Dense_old = np.diagonal(Pback)
        Dense_diff = abs(np.diagonal(self.P) - Dense_old)
        self.pList[1:, :, :] = self.pList[:-1, :, :]
        self.pList[0,  :, :] = Pback + damping*(self.P - Pback)
        self.DPList[1:, :] = self.DPList[:-1, :]
        self.DPList[0,  :] = np.diag(np.real(self.P - Pback))
        
        # Pulay Mixing
        for i, v1 in enumerate(self.DPList):
            for j, v2 in enumerate(self.DPList):
                self.pMat[i,j] = np.sum(v1*v2)
       
        # Apply Damping, store to Gaussian matrix
        if Pulay:
            coeff = LA.solve(self.pMat, self.pB)[:-1]
            print("Applying Pulay Coeff: ", coeff)
            self.P = sum([self.pList[i, :, :]*coeff[i] for i in range(len(coeff))])
            self.pList[0, :, :] = self.P
        else:
            print("Applying Damping value=", damping)
            self.P = self.pList[0, :, :]
        storeDen(self.bar, self.P, self.spin)
        
        # Update counters, print data
        self.updateN() 
        print(f'Total number of electrons (NEGF): {self.nelec:.2f}')
        self.MaxDP = max(Dense_diff)
        RMSDP = np.sqrt(np.mean(Dense_diff**2))
        print(f'MaxDP: {self.MaxDP:.2E} | RMSDP: {RMSDP:.2E}')
        return RMSDP, self.MaxDP

    # Use Gaussian to calculate the Density Matrix, apply damping factor
    # Edamp - when True, add multiple additional 0.1 damp when energy increases
    def PToFock(self):
        # Run Gaussian, update SCF Energy
        try:
            self.bar.update(model=self.method, basis=self.basis, toutput=self.ofile, dofock="DENSITY", miscroute=self.otherRoute)
        except Exception as e:
            print("WARNING: DFT METHOD HAD AN ERROR, CYCLE INVALID:")
            print(e)
            print("CONTINUING TO NEXT CYCLE...")
        self.Total_E_Old = self.Total_E.copy()
        self.Total_E  = self.bar.scalar("escf")
        print("SCF energy: ", self.Total_E)

        # Convergence variables: dE, RMSDP and MaxDP
        dE = self.Total_E-self.Total_E_Old
        print(f'Energy difference is: {dE:.3E}')
        return dE

    # Main SCF loop, runs Fock <-> Density cycle until convergence reached
    # Convergence criteria: dE, RMSDP, and MaxDP < conv, or maxcycles reached
    def SCF(self, conv=1e-5, damping=0.1, maxcycles=100, plot=False, pulay=True):
        # Check to make sure contacts and voltage set 
        assert hasattr(self, 'mu1') and hasattr(self, 'mu2'), "Voltage not set!"
        assert hasattr(self, 'rInd') and hasattr(self,'lInd'), "Contacts not set!"

        #Main SCF Loop
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
            # Run Pulay Kick every nPulay iterations, if turned on
            isPulay = pulay*((Niter+1)%(len(self.pList)+1)==0)
           
            # Fock --> P --> Fock
            EList, occList = self.FockToP()
            RMSDP, MaxDP = self.PMix(damping, isPulay)
            dE = self.PToFock()
            
            # Write monitor variables
            TotalE.append(self.Total_E)
            count.append(Niter)
            PP.append(self.nelec)
            
            # Check 3 convergence criteria
            self.convLevel = max(RMSDP, MaxDP, abs(dE))
            if self.convLevel<conv:
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
        
        homo_lumo = self.getHOMOLUMO()
        print(f'Predicted HOMO: {homo_lumo[0]:.2f} eV , Predicted LUMO {homo_lumo[1]:.2f} eV, Fermi: {self.fermi:0.2f}')
 
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
    
    # Save's matlab *.mat file with main variables
    def saveMAT(self, matfile="out.mat"):
        (sigma1, sigma2) = self.getSigma(self.fermi)
        # Save data in MATLAB .mat file
        matdict = {"F":self.F*har_to_eV, "sig1": sigma1, "sig2": sigma2, "S": self.S, "fermi": self.fermi, "qV": self.qV, "spin" : self.spin, "den" : self.P, "conv": self.convLevel}
        io.savemat(matfile, matdict)
        return self.X@self.F@self.X

