# Python packages
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power

# Developed packages
from density import *

#Constants
kB = 8.617e-5           # eV/Kelvin
dim = 9                 # size of single atom matrix: 1s + 3p + 5d
har_to_eV = 27.211386   # eV/Hartree
Eminf = -1e6            # Setting lower bound to -1e6 eV

class surfGB:
    def __init__(self, F, S, contacts, bar,  file='Au', spin='r', eta=1e-9):
        #Read contact/orbital information and store
        self.cVecs = []
        self.latVecs = []
        self.indsLists = []
        self.dirLists = []
        self.N = len(F)
        self.spin = spin
        for contact in contacts:
            indsList = []
            cList = []
            for atom in contact:
                inds = np.where(np.isin(bar.ibfatm, atom))[0]
                cList.append(bar.c[(atom-1)*3:atom*3])
                assert len(inds) == 9, f'Error: Atom {atom} has {len(inds)} basis functions, expecting 9'
                inds = inds[np.argsort(abs(bar.ibftyp[inds])//1000)]
                indsList.append(inds)
            self.indsLists.append(indsList)
            # Calculate plane direction using SVD
            centeredCoords = cList-np.mean(cList, axis=0)
            _, _, Vt = LA.svd(centeredCoords)
            self.cVecs.append(Vt[-1])
            # Calculate one lattice direction for lining up atoms
            vInd = np.argmin([LA.norm(v - cList[0]) for v in cList[1:]])+1
            latVec = cList[vInd]-cList[0]
            self.latVecs.append(latVec/LA.norm(latVec))
            # Calculate rest of lattice directions
            nVecs = self.genNeighbors(Vt[-1], latVec)
            self.dirLists.append(nVecs)

        
        # Read Bethe lattice parameters and generate hopping/overlap matrices
        self.read_bethe_params('Au')
        self.Slists = []
        self.Vlists = []
        for dirList in self.dirLists: 
            # Construct hopping matrices and store to contact
            Slist = []
            Vlist = []
            for d in dirList:
                Slist.append(self.construct_mat(self.Sdict, d))
                Vlist.append(self.construct_mat(self.Vdict, d))
            self.Slists.append(Slist)
            self.Vlists.append(Vlist)
        # Use surfGBAt() object to store the atomic Bethe lattice green's function for each contact
        self.gList = [surfGBAt(H, Slist, Vlist, eta) for H, Slist, Vlist in zip(self.Hlist, self.Slists, self.Vlists)]
        
        for g in self.gList:
            g.calcFermi(self.ne)

        # Store variables
        self.cList = cList #first contact coords, used for testing
        self.F = F
        self.S = S
        self.eta = eta

    def genNeighbors(self, plane_normal, first_neighbor):
        """
        Generate all 12 nearest neighbor unit vectors:
        - 6 in the plane forming a hexagonal pattern
        - 6 in the following plane forming a similar hexagonal pattern
        
        Args:
            plane_normal: Vector normal to the crystal plane (will be normalized)
            first_neighbor: Vector to one nearest neighbor (will be normalized)
            
        Returns:
            Tuple containing:
                - Array of 6 in-plane unit vectors
                - Array of 6 out-of-plane unit vectors
        """
        
        # Project first_neighbor onto plane perpendicular to plane_normal
        proj = first_neighbor - np.dot(first_neighbor, plane_normal) * plane_normal
        first_neighbor = proj / np.linalg.norm(proj)
        
        # Generate in-plane vectors using 60-degree rotations
        in_plane_vectors = []
        rotation_angle = np.pi / 3  # 60 degrees
        
        for i in range(6):
            angle = i * rotation_angle
            # Rodrigues rotation formula
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            
            K = np.array([[0, -plane_normal[2], plane_normal[1]],
                         [plane_normal[2], 0, -plane_normal[0]],
                         [-plane_normal[1], plane_normal[0], 0]])
            
            R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.matmul(K, K)
            rotated_vector = np.dot(R, first_neighbor)
            in_plane_vectors.append(rotated_vector / np.linalg.norm(rotated_vector))
        
        # Generate out-of-plane vectors
        # We'll use a 45-degree angle for the out-of-plane component
        out_of_plane_angle = np.pi / 4  # 45 degrees
        
        # Create base vector for out-of-plane components
        out_of_plane_base = np.cos(out_of_plane_angle) * first_neighbor + \
                           np.sin(out_of_plane_angle) * plane_normal
        
        # Generate 6 vectors in the next plane using the same 60-degree rotations
        out_of_plane_vectors = []
        for i in range(6):
            angle = i * rotation_angle
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            
            K = np.array([[0, -plane_normal[2], plane_normal[1]],
                         [plane_normal[2], 0, -plane_normal[0]],
                         [-plane_normal[1], plane_normal[0], 0]])
            
            R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.matmul(K, K)
            rotated_vector = np.dot(R, out_of_plane_base)
            out_of_plane_vectors.append(rotated_vector / np.linalg.norm(rotated_vector))
        
        return np.array(in_plane_vectors + out_of_plane_vectors)

    # Read parameters from filename.bethe file, check values, store into dicts
    def read_bethe_params(self, filename):
        params = {}

        with open(filename+'.bethe', 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split on comma and strip whitespace
                line = line.replace(' ','')
                key, value = line.split('=')
                params[key] = float(value)
        
        # Check to make sure parameters are all specified
        # Note: set up only for minimal basis with single s, p, and d orbital
        expected_keys = ['ne', 'es', 'ep', 'edd', 'edt', 'sss', 'sps', 'pps', 'ppp',
                        'sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd', 'Ssss', 'Ssps',
                        'Spps', 'Sppp', 'Ssds', 'Spds', 'Spdp', 'Sdds', 'Sddp', 'Sddd']
        assert len(params.keys()) == len(expected_keys) and set(params.keys()) == set(expected_keys), \
             f"Error reading file: Found Bethe parameters: {list(params.keys())}, expected: {expected_keys}"
        
        # sort parameters and convert Hartrees to eV 
        self.ne = params['ne']
        if self.spin=='r':
            print('Restricted --> Halving ne')
            self.ne *= 0.5
        self.Edict = {k[1:]:params[k]*har_to_eV for k in params if k.startswith('e')}
        self.Sdict = {k[1:]:params[k] for k in params if k.startswith('S')}
        self.Vdict = {k:params[k]*har_to_eV for k in params if not k.startswith('e') and not k.startswith('S')}
        # Setup onsite H0 matrix before Fermi level shifting
        H0 = np.diag([self.Edict['s']]+ [self.Edict['p']]*3 + \
                     [self.Edict['dt']]*3 + [self.Edict['dd']]*2)
        self.Hlist = [H0 for indsList in self.indsLists]

    def construct_mat(self, Mdict, dirCosines):
        """
        Construct hopping matrix for octahedral direction (l,m,n)
        Full implementation including d-orbital interactions
        """
        M = np.zeros((dim, dim))
        
        # s-s -- isotropic (o-o)
        M[0,0] = Mdict['sss']
        
        # s-p --  o-8 vs o-∞
        for i in range(3):
            M[0,i+1] = Mdict['sps'] * dirCosines[i]
        
        # p-p  -- 8-8 vs ∞-∞ vs ∞-8
        for i in range(3):
            for j in range(3):
                if i == j: # 8-8 vs ∞-∞
                    dirCos = dirCosines[i]
                    M[i+1,j+1] = Mdict['pps'] * dirCos**2 + Mdict['ppp'] * (1 - dirCos**2)
                else: # 8-∞ also possible
                    M[i+1,j+1] = (Mdict['pps'] - Mdict['ppp']) * dirCosines[i] * dirCosines[j]
        
        # d-orbital angular functions
        (l, m, n) = dirCosines
        dxy = 2*l*m
        dyz = 2*m*n
        dzx = 2*n*l
        dx2y2 = l*l - m*m
        dz2 = (3*n*n - 1)/2
        
        dFuncs = [dxy, dyz, dzx, dx2y2, dz2]
        
        # s-d -- o-X (2x) vs o-+ (2x) vs o-θ
        for i, d_func in enumerate(dFuncs):
            M[0,4+i] = Mdict['sds'] * d_func
            M[4+i,0] = M[0,4+i]
        
        # p-d -- 8-X (4x) vs ∞-X (2x) vs 8-+(4x) vs ∞-+ (2x) vs 8-θ vs ∞-θ
        for i, dCos in enumerate(dirCosines):  # p orbitals
            for j, d_func in enumerate(dFuncs):  # d orbitals
                M[1+i,4+j] = Mdict['pds'] * dCos * d_func + \
                        Mdict['pdp'] * (1 - dCos*d_func)
                M[4+j,1+i] = M[1+i,4+j]
        
        # d-d -- (X, X, +, +, θ) x (X, X, +, +, θ) = 25 combinations
        # delta bond (ddd) = X-X 
        for i in range(5):  # first d orbital
            d_i = dFuncs[i]
            for j in range(5):  # second d orbital
                d_j = dFuncs[j]
                M[4+i,4+j] = Mdict['dds'] * (d_i * d_j) + \
                             Mdict['ddp'] * (1 - (d_i*d_j)) + \
                             Mdict['ddd'] * (1-(d_i**2))*(1-(d_j**2))
                
        return M
    
    def sigma(self, E, i, conv=1e-5):
        sig = np.zeros((self.N, self.N), dtype=complex)
        sigAtom = self.gList[i].sigmaTot(E, conv)
        for inds in self.indsLists[i]:
            sig[np.ix_(inds, inds)] = sigAtom
        return sig
    
    def sigmaTot(self, E, conv=1e-5):
        sig = np.zeros((self.N, self.N), dtype=complex)
        for i in range(len(self.indsLists)):
            sig+= self.sigma(E, i, conv)
        return sig
    
    # Update contact i with a new fermi energy (Ef) by shifting all onsite energies
    def updateFermi(self, i, Ef):
        fermiPrev = self.gList[i].fermi
        print(f'Changing contact {i+1} fermi energy: {fermiPrev} --> {Ef}')
        dFermi = Ef - fermiPrev
        dH = np.eye(dim)*dFermi
        self.gList[i].H += dH 
        self.gList[i].fermi = Ef
    
    # Fermi levels used to update onsite energies, Fock matrix unused 
    def setF(self,F, muL, muR):
        self.F = F
        if self.gList[0].fermi != muL:
            self.updateFermi(0, muL)
        if self.gList[-1].fermi != muR:
            self.updateFermi(-1, muR) 

# Bethe lattice surface Green's function for a single atom
class surfGBAt:
    def __init__(self, H, Slist, Vlist, eta):
        assert np.shape(H) == (dim,dim), f"Error with H dim, should be {dim}x{dim}"
        for S,V in zip(Slist, Vlist):
            assert np.shape(S) == (dim,dim), f"Error with S dim, should be {dim}x{dim}"
            assert np.shape(V) == (dim,dim), f"Error with F dim, should be {dim}x{dim}"
        self.H = H
        self.Slist = Slist
        self.Vlist = Vlist
        self.eta = eta
        self.Eprev = None
        self.gprev = None

        # For compatibility with density methods:
        self.F = self.H
        self.S = np.eye(dim)
    
    # Calculate surface Green's function, i is a dummy variable for compatibility
    def g(self, E, i, conv=1e-5, mix=0.5):
        #Construct matrices for Dyson equation
        A = (E - self.eta*1j)*np.eye(dim) - self.H
        B = np.zeros((dim, dim), dtype=complex)
        #Sum up hopping contributions from each neighbor
        for S, V in zip(self.Slist,self.Vlist):
            B += (E - self.eta*1j)*S - V
        
        # Use previous solution if available and close in energy
        if self.gprev is not None and self.Eprev is not None:
            if abs(E - self.Eprev) < 1: 
                g = self.gprev
            else:                   # Otherwise generate guess with 1 cycle
                g = LA.inv(A)
                g = LA.inv(A - B@g@B.conj().T)
                
        else:
            g = LA.inv(A)
            g = LA.inv(A - B@g@B.conj().T)
        self.Eprev= E
        
        #Self-consistency loop 
        count = 0
        maxIter = int(1/(conv*mix))*10
        diff = np.inf
        while diff > conv and count < maxIter:
            g_ = g.copy()
            g_new = LA.inv(A - B@g@B.conj().T) #Dyson Equation
            dg = abs(g_new - g_)/(abs(g_).max())
            diff = dg.max()
            g = g_new * mix + g_ * (1-mix)
            count += 1
        if diff>conv:
            print(f'Warning: exceeded max iterations! E: {E}, Conv: {diff}')
        #DEBUG:
        if count>1000:
            print(f'gAtom at {E:.2e} converged in {count} iterations: {diff}')
        return g
    
    # Return self-energy matrix associated with single atom
    def sigmaTot(self, E, conv=1e-5):
        sig = np.zeros((dim,dim), dtype=complex)
        g = self.g(E, 0, conv)
        for S, V in zip(self.Slist,self.Vlist):
            B = (E - self.eta*1j)*S - V
            sig += B.conj().T@g@B
        return sig
    
    # Adding this function for compatibility
    def sigma(self, E, i, conv=1e-5):
        return self.sigmaTot(E, conv=1e-5)
   
    # Calculate fermi energy using bisection (to specified tolerance)
    def calcFermi(self, ne, fGuess=5, tol=1e-3):
        Emin = min(np.diag(self.H))
        Emax = max(np.diag(self.H))
        maxCycles = 1000
        cycles = 0
        calcN = 0
        Nint = 200
        calcN_ = np.inf
        # Main loop for calculating Emin/Emax integration limits
        while abs(calcN - calcN_) > tol and cycles < maxCycles:
            calcN_ = calcN +0.0
            Emin -= 10
            Emax += 10
            # recalculate integration limits each iteration to ensure accuracy
            Nint = 8
            MaxDP = np.inf
            rho = np.eye(dim)
            while MaxDP > tol and Nint < 1e3:
                Nint *= 2
                rho_ = rho.copy()
                rho = np.real(densityComplex(self.H, np.eye(dim), self, Emin, Emax, Nint, T=0, showText=False))
                MaxDP = max(np.diag(abs(rho_ - rho)))
            if Nint > 1e3:
                print(f'Warning: MaxDP above tolerance (val = {MaxDP:.3E})')
            calcN = sum(np.diag(rho))
            print(f"Range: {Emin}, {Emax} - Err = {calcN - calcN_}")
        if cycles == maxCycles:
            print(f"Warning: Energy range ({Emin}, {Emax}) not converged (val = {calcN - dim})")
        self.fermi = calcFermi(self, ne, Emin, Emax, fGuess, Nint, 1, Eminf, tol)[0]
        return self.fermi

