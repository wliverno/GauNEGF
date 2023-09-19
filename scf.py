import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import scipy.io as io
import sys
import time
import matplotlib.pyplot as plt
from numpy import savetxt


from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu


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


#### HELPER FUNCTIONS ####

# Get density using analytical integration
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


# Form Sigma matrix given self-energy (value) and orbital indices (inds)
def formSigma(inds, value, nsto):
    sigma = np.asmatrix(np.zeros((nsto,nsto)),dtype=complex)

    for i in inds:
        sigma[i,i] = value

    return sigma


# Build density matrix based on spin type
def getDen(bar, spin):
    # Set up Fock matrix and atom indexing
    # Note: positive ind1ices are alpha/paired orbitals, negative are beta orbitals
    if spin == "r" or spin == "g":
        P = np.asmatrix(bar.matlist[AlphaSCFDen].expand())
    elif spin == "ro" or spin == "u":
        PA = np.asmatrix(bar.matlist[AlphaSCFDen].expand())
        PB = np.asmatrix(bar.matlist[BetaSCFDen].expand())
        P = np.block([[PA, np.zeros(PA.shape)], [np.zeros(PB.shape), PB]])
    else:
        raise ValueError("Spin treatment not recognized!")
    return P

# Build Fock matrix based on spin type, return orbital indices (alpha and beta are +/-)
def getFock(bar, spin):
    # Set up Fock matrix and atom indexing
    # Note: positive ind1ices are alpha/paired orbitals, negative are beta orbitals
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

# Return energies for each electron in eV
def getEnergies(bar, spin):
    if spin =="r":
        Alevels = np.sort(bar.matlist[AlphaEnergies].expand())
        levels = [level for pair in zip(Alevels, Alevels) for level in pair]
    elif spin=="ro" or spin == "u":
        Alevels = np.sort(bar.matlist[AlphaEnergies].expand())
        Blevels = np.sort(bar.matlist[BetaEnergies].expand())
        levels = [level for pair in zip(Alevels, Blevels) for level in pair]
    elif spin=="g":
        levels = np.sort(bar.matlist[AlphaEnergies].expand())
    else:
        raise ValueError("Spin treatment not recognized!")
    return np.sort(levels)*har_to_eV

# Store density matrix to use in Gaussian
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

#### MAIN SCF FUNCTION ######

def SCF(fn, lContact, rContact, fermi, qV, sig=-0.1j, sig2=False, basis="lanl2dz",
    func="b3pw91", spin="r", damping = 0.1, conv = 1e-5, maxcycles=100, save=True):
    
    # Set up variables
    infile = fn + ".gjf"
    chkfile = fn + ".chk"
    outfile = fn + ".log"

    mu1 =  fermi + (qV/2)
    mu2 =  fermi - (qV/2)

    Emin = -15
    Eminf = -1e5

    PP=[]
    count=[]
    TotalE=[]
    
    method= spin+func
    otherRoute = ""     # Other commands that are needed in Gaussian

    # Start Calculation

    print('Calculation started at '+str(time.asctime()))
    start_time = time.time()

    bar = qcb.BinAr(debug=False,lenint=8,inputfile=infile)
    
    
    # Old code for aligning E-field to z-direction through rotation
    #v = np.cross(vecNorm, np.array([0, 0, 1]))
    #c = np.dot(vecNorm, np.array([0, 0, 1]))
    #s = np.linalg.norm(v)
    #kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    #R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)) #Rotation Matrix
    #for i in range(0, len(bar.c), 3):
    #    coord = bar.c[i:i+3]
    #    bar.c[i:i+3] = R.dot(coord)

    ## RUN GAUSSIAN
    try:
        bar.update(model=method, basis=basis, toutput=outfile, dofock="density",chkname=chkfile)
        print('Checking '+chkfile+' for saved data...');
    except:
        print('Checkpoint not loaded, running full SCF...');
        bar.update(model=method, basis=basis, toutput=outfile, dofock="scf",chkname=chkfile)

    print("Done!")

    Fock, locs = getFock(bar, spin)

    print("ORBS:")
    print(locs)
    icharg=bar.icharg
    multip=bar.multip
    print('Expecting', str(bar.ne), 'electrons')
    print('Charge is:', icharg)
    print('Multiplicity is:', multip)

     
    # Calculate electric field to apply during SCF, apply between contacts
    lAtom = bar.c[(lContact[0]-1)*3:lContact[0]*3]
    rAtom = bar.c[(rContact[0]-1)*3:rContact[0]*3]
    vec  = (lAtom-rAtom)
    dist = LA.norm(vec)
    vecNorm = vec/dist

    field = -1*vecNorm*qV*V_to_au/(dist*0.0001)
    bar.scalar("X-EFIELD", int(field[0]))
    bar.scalar("Y-EFIELD", int(field[1]))
    bar.scalar("Z-EFIELD", int(field[2]))
    print("E-field set to "+str(LA.norm(field))+" au")
    #if field>=0:
    #    otherRoute += " field=z+" +str(field)
    #else:
    #    otherRoute += " field=z" +str(field)


    # Prepare Overlap, Identity, and Lowdin TF matricies
    nsto = len(locs)
    Omat = np.asmatrix(bar.matlist["OVERLAP"].expand())
    if spin == "ro" or spin == "u":
        Overlap = np.block([[Omat, np.zeros(Omat.shape)],[np.zeros(Omat.shape),Omat]])
    else:
        Overlap = Omat

    X = np.asmatrix(fractional_matrix_power(Overlap, -0.5))

    I = np.asmatrix(np.identity(nsto))

    # Prepare Sigma matrices
    lInd = np.where(np.isin(locs, lContact))[0]
    rInd = np.where(np.isin(locs, rContact))[0]
    if isinstance(sig, (int,complex,float)):  #if sigma is a single value
        sigma1 = formSigma(lInd, sig, nsto)
        if isinstance(sig2, (int,complex,float)):  #if sigma is a single value
            sigma2 = formSigma(rInd, sig2, nsto)
        else:
            sigma2 = formSigma(rInd, sig, nsto)
    else:                                     #if sigma is a matrix
        sigma1 = np.asmatrix(-1j*1e-9*Overlap,dtype=complex)
        sigma2 = np.asmatrix(-1j*1e-9*Overlap,dtype=complex)
        sigma1[np.ix_(lInd, lInd)] = sig
        if isinstance(sig2, (list, np.ndarray)):
            sigma2[np.ix_(rInd, rInd)] = sig2   
        else:
            sigma2[np.ix_(rInd, rInd)] = sig

    sigma12 = sigma1 + sigma2

    print('Max imag sigma:', str(np.max(np.abs(np.imag(sigma12)))));
    Gam1 = (sigma1 - sigma1.getH())*1j
    Gam2 = (sigma2 - sigma2.getH())*1j


    sigmaW1 = formSigma(lInd, -0.00001j, nsto)
    sigmaW2 = formSigma(rInd, -0.00001j, nsto)
    sigmaW12 = sigmaW1+sigmaW2

    GamW1 = (sigmaW1 - sigmaW1.getH())*1j
    GamW2 = (sigmaW2 - sigmaW2.getH())*1j

    print('Entering NEGF-SCF loop at: '+str(time.asctime()))
    print('###################################')

    #sys.exit("BREAK!")

    ########################     SCF LOOP    ##########################

    Loop = True
    Niter = 0
    while Loop :

        print(Niter)

        print("SCF energy: ", bar.scalar("escf")) #line 269 of QCBinAr.py

        Total_E =  bar.scalar("escf")

        # Prepare Variables for Analytical Integration

        Fock,locs = getFock(bar, spin)
        Fbar = X * (Fock*har_to_eV + sigma12) * X
        GamBar1 = X * Gam1 * X
        GamBar2 = X * Gam2 * X

        D,V = LA.eig(np.asmatrix(Fbar))
        D = np.asmatrix(D).T
           
        err =  np.float_(sum(np.imag(D)))
        if  err > 0:
            print('Imagine elements on diagonal of D are positive ------->  ', err)

        FbarW = X*(Fock*har_to_eV + sigmaW12)*X
        GamBarW1 = X*GamW1*X
        GamBarW2 = X*GamW2*X
        Dw,Vw = LA.eig(np.asmatrix(FbarW))
        Dw = np.asmatrix(Dw).T
        
        # Calculate Density
        P1 = density(V, D, GamBar1, Emin, mu1)
        P2 = density(V, D, GamBar2, Emin, mu2)
        Pw = density(Vw, Dw, GamBarW1+GamBarW2, Eminf, Emin)


        P=P1 + P2 + Pw

        
        pshift = V.getH() * P * V
        occList = np.diag(np.real(pshift)) 
        EList = np.asarray(np.real(D)).flatten()
        inds = np.argsort(EList)

    #    for pair in zip(occList[inds], EList[inds]):                       #DEBUG
    #        print("Energy =", str(pair[1]), ", Occ =", str(pair[0]))



        # Check Convergence 
        if Niter == 0:
            Pback = getDen(bar, spin)
            Dense_old = np.diagonal(Pback)
            Total_E_Old = Total_E
            
        # Convergence variables, currently using RMSDP and MaxDP
        dE = Total_E-Total_E_Old
        Dense_diff = abs(np.diagonal(P) - Dense_old)
        MaxDP = max(Dense_diff)
        RMSDP = np.sqrt(np.mean(Dense_diff**2))
        
        print('Energy difference is: ', dE)
        print(f'MaxDP: {MaxDP:.2E} | RMSDP: {RMSDP:.2E}')



        if RMSDP<conv and MaxDP<conv and abs(dE)<conv:
            print('Convergence achieved after '+str(Niter)+' iterations!')
            Loop = False
        elif Niter >= maxcycles:
            print('WARNING: Convergence criterion net, maxcycles reached!')
            Loop = False
        
        Dense_old = np.diagonal(P)
        Total_E_Old = Total_E
        
        #Lowdin TF Back  
        P = np.real(X * P * X)

        # APPLY DAMPING:
        P = Pback + damping*(P - Pback)
        Pback = P
        
        # Track values for convergence graphic
        nelec = np.trace(np.real(P))
        count.append(Niter)
        PP.append(MaxDP)
        TotalE.append(nelec)


    #    print("BEFORE")                                                    #DEBUG
    #    print(np.diag(bar.matlist['ALPHA DENSITY MATRIX'].expand()))
    #    print("AFTER")
    #    print(np.diag(P))
        storeDen(bar, P, spin)    

        
       
        bar.update(model= method, basis=basis, toutput=outfile, dofock="density", miscroute=otherRoute)
        
        print('Total number of electrons: ', nelec)
    
        Niter += 1
    
    print('Writing to checkpoint file...') 
    bar.writefile(chkfile)
    print(chkfile+' written!') 
    
    print('##########################################')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')
    print('SCF Loop existed at', time.asctime())
    
    print('=========================')
    print('ENERGY LEVEL OCCUPATION:')
    print('=========================')
    for pair in zip(occList[inds], EList[inds]):
        print(f"Energy = {pair[1]:9.3f} eV | Occ = {pair[0]:5.3f}")

    # Final Hamiltonian:
    H0 = X*Fock*har_to_eV*X

    # DEBUGGING: Save run data in text files
#    savetxt('Iteration.txt', count, delimiter=',')
#    savetxt('DM_max.txt', PP, delimiter=',')
#    savetxt('Electron.txt', TotalE, delimiter=',')
    
    # Plot convergence data 
    plt.subplot(311)
    plt.title('Fermi level is: '+ str(fermi) + 'eV   sigma is: ' +str(sig) + "eV\n" + r" $\Delta V=$"+str(qV)+'V Method: '+ method + '/' + basis + "\n")
    plt.ylabel(r'Max Change in $\rho$')
    plt.plot(count, PP, color='g', linestyle='solid' ,linewidth = 1, marker='x')

    plt.subplot(312)
    plt.ylabel('Total # of electrons')
    plt.xlabel('Iteration')
    plt.plot(count, TotalE, color='black', linestyle='solid' ,linewidth = 1, marker='o')


    plt.subplot(313)
    plt.plot(EList[inds], occList[inds], color ='r', marker='o')
    plt.xlabel('Energy')
    plt.ylabel('Occupation')
    plt.xlim([Emin, 0])
    plt.savefig('debug.png')
    plt.close()
 
    if save==True:
        # Save data in MATLAB .mat file
        matdict = {"H0":H0, "sig1": np.diag(sigma1), "sig2": np.diag(sigma2), 
                    "fermi" : fermi, "qV": qV, "X": X, "spin": spin}
        matfile = f"{fn}_{fermi:.2f}_{qV:.2f}V.mat"
        io.savemat(matfile, matdict)
        return matfile
    else:
        return H0

#### TRANSPORT FUNCTIONS ####

# Calculate Coherent Current at T=0K using NEGF
def quickCurrent(H0, sig1, sig2, fermi, qV, spin="r",dE=0.01):
    
    H0 = np.asmatrix(H0)
    sig1 = np.asmatrix(np.diag(sig1))
    sig2 = np.asmatrix(np.diag(sig2))
    gamma1 = np.imag(sig1)*2
    gamma2 = np.imag(sig2)*2
    I = np.asmatrix(np.identity(len(H0)));
    
    if qV < 0:
        dE = -1*abs(dE)
    else:
        dE = abs(dE)
    Elist = np.arange(fermi-(qV/2), fermi+(qV/2), dE)
    Tr = []
    for E in Elist:
        G = LA.inv(E*I - H0 - sig1 - sig2)
        T = np.real(np.trace(gamma1*G*gamma2*G.getH()))
        Tr.append(T)
    
    curr = eoverh * np.trapz(Tr, Elist)
    if spin=="r":
        curr *= 2

    return curr

# Calculate current from SCF mat file
def qCurrentF(fn, dE=0.01):
    matfile = io.loadmat(fn);
    return quickCurrent(matfile["H0"],matfile["sig1"][0],matfile["sig2"][0],
            matfile["fermi"][0,0], matfile["qV"][0,0], matfile["spin"][0], dE=dE)

