# Python packages
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power

#Constants
kB = 8.617e-5           # eV/Kelvin
dim = 9                 # size of single atom matrix: 1s + 3p + 5d
class surfGB:
    def __init__(self, F, S, contacts, bar, file='Au', eta=1e-9):
        #Read contact/orbital information and store
        self.cVecs = []
        self.latVecs = []
        self.indsLists = []
        self.dirLists = []
        self.N = bar.nbasis
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
        
        # Store variables
        self.cList = cList #Used for testing
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
        expected_keys = ['es', 'ep', 'edd', 'edt', 'sss', 'sps', 'pps', 'ppp',
                        'sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd', 'Ssss', 'Ssps',
                        'Spps', 'Sppp', 'Ssds', 'Spds', 'Spdp', 'Sdds', 'Sddp', 'Sddd']
        assert len(params.keys()) == len(expected_keys) and set(params.keys()) == set(expected_keys), \
             f"Error reading file: Found Bethe parameters: {list(params.keys())}, expected: {expected_keys}"
        
        # sort parameters and return values 
        self.Edict = {k[1:]:params[k] for k in params if k.startswith('e')}
        self.Sdict = {k[1:]:params[k] for k in params if k.startswith('S')}
        self.Vdict = {k:params[k] for k in params if not k.startswith('e') and not k.startswith('S')}
        
    def gAtom(self, E, i, conv=1e-5, relFactor=0.9):
        Slist = self.Slists[i]
        Vlist = self.Vlists[i]
        #Construct matrices for Dyson equation
        H0 = np.diag([self.Edict['s']]+ [self.Edict['p']]*3 + \
                     [self.Edict['dt']]*3 + [self.Edict['dd']]*2)
        A = (E - self.eta*1j)*np.eye(dim) - H0
        B = np.zeros((dim, dim), dtype=complex)
        #Sum up hopping contributions from each neighbor
        for k in range(len(Slist)): 
            B += (E - self.eta*1j)*Slist[k] - Vlist[k]
        g = LA.inv(A) #Initial guess
        
        #Self-consistency loop 
        count = 0
        maxIter = int(1/(conv*relFactor))*10
        diff = conv+1
        while diff>conv and count<maxIter:
            g_ = g.copy()
            g = LA.inv(A - B@g@B.conj().T) #Dyson equation
            dg = abs(g-g_)/(abs(g).max())
            g = g*relFactor + g_*(1-relFactor)
            diff = dg.max()
            count = count+1
        if diff>conv:
            print(f'Warning: exceeded max iterations! E: {E}, Conv: {diff}')
        #DEBUG:
        #print(f'gAtom converged in {count} iterations: {diff}')
        return g
            
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
        Slist = self.Slists[i]
        Vlist = self.Vlists[i]
        #Sum up hopping contributions from each neighbor
        gAt = self.gAtom(E, i, conv)
        sigAtom = np.zeros((dim, dim), dtype=complex)
        for k in range(len(Slist)): 
            B = (E - self.eta*1j)*Slist[k] - Vlist[k]
            sigAtom += B.conj().T@gAt@B
        sig = np.zeros((self.N, self.N), dtype=complex)
        for inds in self.indsLists[i]:
            sig[np.ix_(inds, inds)] = sigAtom
        return sig
    
    def sigmaTot(self, E, conv=1e-5):
        sig = np.zeros((self.N, self.N), dtype=complex)
        for i in range(len(self.indsLists)):
            sig+= self.sigma(E, i, conv)
        return sig
   
    # Required to use with other linked packages
    def setF(self,F):
        self.F = F
