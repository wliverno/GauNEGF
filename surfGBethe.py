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
        
        # Spin independent implementation, add generate spin terms during sigma generation
        self.spin = spin
        orbMap = bar.ibfatm[bar.ibfatm>0] 
        orbTyp = bar.ibftyp[bar.ibfatm>0]
        self.N = len(orbMap)

        # Collect contact information
        for contact in contacts:
            indsList = []
            cList = []
            for atom in contact:
                inds = np.where(np.isin(orbMap, atom))[0]
                cList.append(bar.c[(atom-1)*3:atom*3])
                assert len(inds) == 9, f'Error: Atom {atom} has {len(inds)} basis functions, expecting 9'
                inds = inds[np.argsort(abs(orbTyp[inds])//1000)]
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
        self.gList = [surfGBAt(self.H0, Slist, Vlist, eta) for Slist, Vlist in zip(self.Slists, self.Vlists)]
        
        for g in self.gList:
            g.calcFermi(self.ne/2)

        # Store variables
        self.cList = cList #first contact coords, used for testing
        self.F = F
        self.S = S
        self.eta = eta

    def genNeighbors(self, plane_normal, first_neighbor):
        """
        Generate 9 neighbor unit vectors based on an FCC [111] surface:
        - 6 in the plane forming a hexagonal pattern
        - 3 in the following plane forming a triangular pattern
        
        Args:
            plane_normal: Vector normal to the crystal plane (will be normalized)
            first_neighbor: Vector to one nearest neighbor (will be normalized)
            
        Returns:
            Tuple containing:
                - Array of 6 in-plane unit vectors
                - Array of 3 out-of-plane unit vectors
        """
        
        # Project first_neighbor onto plane perpendicular to plane_normal
        proj = first_neighbor - np.dot(first_neighbor, plane_normal) * plane_normal
        first_neighbor = proj / np.linalg.norm(proj)
        
        # Generate in-plane vectors using 60-degree rotations
        in_plane_vectors = []
        rotation_angle = np.pi / 3  # 60 degrees
        
        for i in range(3):
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
        out_of_plane_angle = np.arccos(1/3) / 2  # ~35.26 degrees
        
                # Create vectors pointing up to next plane
        out_of_plane_vectors = []
        out_of_plane_base = np.cos(out_of_plane_angle) * first_neighbor + \
                  np.sin(out_of_plane_angle) * plane_normal
        
        for i in range(3):
            angle = i * 2 * np.pi / 3  # 120 degree rotations
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)

            K = np.array([[0, -plane_normal[2], plane_normal[1]],
                         [plane_normal[2], 0, -plane_normal[0]],
                         [-plane_normal[1], plane_normal[0], 0]])

            R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.matmul(K, K)
            rotated_vector = np.dot(R, out_of_plane_base)
            out_of_plane_vectors.append(rotated_vector)
        
        # Add corresponding opposite vectors at the (k+6)%12 location
        all_vectors = in_plane_vectors + out_of_plane_vectors
        for i in range(6):
            all_vectors.append(-all_vectors[i])

        # Return vectors 
        return all_vectors

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
        self.Edict = {k[1:]:params[k]*har_to_eV for k in params if k.startswith('e')}
        self.Sdict = {k[1:]:params[k] for k in params if k.startswith('S')}
        self.Vdict = {k:params[k]*har_to_eV for k in params if not k.startswith('e') and not k.startswith('S')}
        # Setup onsite H0 matrix before Fermi level shifting
        self.H0 = np.diag([self.Edict['s']]+ [self.Edict['p']]*3 + \
                     [self.Edict['dd']]+ [self.Edict['dt']]*2 + [self.Edict['dd'], self.Edict['dt']])

    def construct_mat(self, Mdict, dirCosines):
        """
        Construct hopping/overlap matrix using Slater-Koster formalism with proper phase factors.
         
        Parameters:
            Mdict: Dictionary of Slater-Koster parameters (ssσ, spσ, etc.)
            dirCosines: Array [l,m,n] of direction cosines
        """

        M = np.zeros((dim, dim))
        
        #Original matrix before rotation - assuming [0,0,1] bond direction
        # s-s coefficient
        M[0,0] = Mdict['sss']
        
        # s-p block
        M[0,3] = Mdict['sps'] #s-pz
        M[3,0] = -Mdict['sps'] #pz-s

        # p-p block
        M[1,1] = M[2,2] = Mdict['ppp'] #px-px, py-py
        M[3,3] = Mdict['pps'] #pz-pz

        # s-d block
        M[0, 4] = M[4, 0] =  Mdict['sds'] #s - d3z²-r²

        # p-d block
        M[1,5] = Mdict['pdp'] #px - dxz
        M[2,6] = Mdict['pdp'] #py - dyz
        M[3,4] = Mdict['pds'] #pz - d3z²-r²
        
        M[5,1] = -Mdict['pdp'] #dxz - px
        M[6,2] = -Mdict['pdp'] #dyz - py
        M[4,3] = -Mdict['pds'] #d3z²-r² - pz

        # d-d block
        M[4,4] = Mdict['dds'] #d3z²-r² - d3z²-r²
        M[5,5] = M[6,6] = Mdict['ddp'] #dxz - dxz, dyz - dyz 
        M[7,7] = M[8,8] = Mdict['ddd'] #dx²-y² - dx²-y², dxy - dxy 
        
        # Initialize 9x9 transformation matrix and polar directions
        tr = np.zeros((9, 9))
        x, y, z = dirCosines
        theta = np.arccos(z)  # polar angle from z-axis
        phi = np.arctan2(y, x)  # azimuthal angle in x-y plane
        
        # s orbital (1x1) at position [0,0] - always 1 since spherically symmetric
        tr[0,0] = 1.0
        
        # p orbitals (3x3) at positions [1:4,1:4]
        # [px,py,pz] block - describes how p orbitals transform under rotation
        tr[1:4,1:4] = np.array([
            [np.cos(theta) * np.cos(phi), -np.sin(phi)  , np.sin(theta)*np.cos(phi)],
            [np.cos(theta) * np.sin(phi),  np.cos(phi)  , np.sin(theta)*np.sin(phi)], 
            [-np.sin(theta)             ,  0            , np.cos(theta)]
        ])
        
        # d orbitals (5x5) at positions [4:9,4:9]
        # [d3z2-r2, dxz, dyz, dx2-y2, dxy] block - transforms the five d orbitals
        d_block = np.zeros((5,5))
        
        # Copying formula from ANT.Gaussian directly
        d_block[0,0] = (3 * z**2 - 1) / 2
        d_block[0,1] = -np.sqrt(3) * np.sin(2*theta) / 2
        d_block[0,3] = np.sqrt(3) * np.sin(theta)**2 / 2
        
        d_block[1,0] = np.sqrt(3) * np.sin(2*theta) * np.cos(phi) / 2
        d_block[1,1] = np.cos(2*theta) * np.cos(phi)
        d_block[1,2] = -np.cos(theta) * np.sin(phi)
        d_block[1,3] = -d_block[1,0] / np.sqrt(3)
        d_block[1,4] = np.sin(theta) * np.sin(phi)
        
        d_block[2,0] = np.sqrt(3) * np.sin(2*theta) * np.sin(phi) / 2
        d_block[2,1] = np.cos(2*theta) * np.sin(phi)
        d_block[2,2] = np.cos(theta) * np.cos(phi)
        d_block[2,3] = -d_block[2,0] / np.sqrt(3)
        d_block[2,4] = -np.sin(theta) * np.cos(phi)
        
        d_block[3,0] = np.sqrt(3) * np.sin(theta)**2 * np.cos(2*phi) / 2
        d_block[3,1] = np.sin(2*theta) * np.cos(2*phi) / 2
        d_block[3,2] = -np.sin(theta) * np.sin(2*phi)
        d_block[3,3] = (1 + np.cos(theta)**2) * np.cos(2*phi) / 2
        d_block[3,4] = -np.cos(theta) * np.sin(2*phi)
        
        d_block[4,0] = np.sqrt(3) * np.sin(theta)**2 * np.sin(2*phi) / 2
        d_block[4,1] = np.sin(2*theta) * np.sin(2*phi) / 2
        d_block[4,2] = np.sin(theta) * np.cos(2*phi)
        d_block[4,3] = (1 + np.cos(theta)**2) * np.sin(2*phi) / 2
        d_block[4,4] = np.cos(theta) * np.cos(2*phi)
        
        tr[4:9,4:9] = d_block
             
        # Apply transformation 
        return tr.T @ M @ tr
    
    def sigma(self, E, i, conv=1e-5):
        sig = np.zeros((self.N, self.N), dtype=complex)
        sigAtom = self.gList[i].sigmaTot(E, conv)
        for inds in self.indsLists[i]:
            sig[np.ix_(inds, inds)] = sigAtom
        if self.spin == 'u' or self.spin == 'ro':
            sig = np.kron(np.eye(2), sig)
        elif self.spin =='g':
            sig = np.kron(sig, np.eye(2))
        return sig
    
    def sigmaTot(self, E, conv=1e-5):
        sigs = [self.sigma(E, i, conv) for i in range(len(self.indsLists))]
        return sum(sigs)
    
    # Update contact i with a new fermi energy (Ef) by shifting all onsite energies
    def updateFermi(self, i, Ef):
        fermiPrev = self.gList[i].fermi
        if i==-1:
            print(f'Changing right contact fermi energy: {fermiPrev} --> {Ef}')
        elif i==0:
            print(f'Changing left contact fermi energy: {fermiPrev} --> {Ef}')
        else:
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

    ## TESTING METHODS FOR SLATER-KOSTER INTERACTIONS:
    def test_d_orbital_functions(self):
        """Test d orbital angular functions using surfGB parameter dictionaries"""
    
        # Use values from the Bethe parameter dictionaries
        Vdict = self.Vdict  # Contains hopping parameters
    
        # Test along x-axis [1,0,0]
        M = self.construct_mat(self.Vdict, [1, 0, 0])
    
        # dxy should be zero along x-axis
        np.testing.assert_almost_equal(M[0,8], 0.0,
            err_msg="dxy not zero along x-axis")
        
        # dx2-y2 should be sqrt(3)/2 * sds along x-axis
        np.testing.assert_almost_equal(M[0,7], np.sqrt(3)/2 * Vdict['sds'],
            err_msg="dx2-y2 incorrect along x-axis")
    
        # dz2 should be -1/2 along x-axis
        np.testing.assert_almost_equal(M[0,4], -0.5 * Vdict['sds'],
            err_msg="dz2 incorrect along x-axis")
    
        print("d orbital angular function tests passed!")
    
    def test_d_orbital_symmetry(self):
        """Test d orbital symmetry properties using surfGB parameters"""
    
        # Test inversion symmetry
        dir1 = [1/np.sqrt(2), 1/np.sqrt(2), 0]
        dir2 = [-1/np.sqrt(2), -1/np.sqrt(2), 0]
    
        M1 = self.construct_mat(self.Vdict, dir1)
        M2 = self.construct_mat(self.Vdict, dir2)
    
        # d-d block should be identical under inversion
        np.testing.assert_array_almost_equal(
            M1[4:,4:], M2[4:,4:],
            err_msg="d-d block not symmetric under inversion")
    
        print("d orbital symmetry tests passed!")
    
    def test_pd_interaction(self):
        """Test p-d orbital interactions using surfGB parameters"""
    
        Vdict = self.Vdict
    
        # Test px-dxy interaction along x-axis
        M = self.construct_mat(Vdict, [1, 0, 0])
    
        # px-dxy should be zero along x-axis
        np.testing.assert_almost_equal(
            M[1,8], 0.0,
            err_msg="px-dxy interaction incorrect along x-axis")
    
        # Test pz-dz2 interaction along z-axis
        M = self.construct_mat(Vdict, [0, 0, 1])
        expected = Vdict['pds']  # Should be pure sigma
        np.testing.assert_almost_equal(
            M[3,4], expected,
            err_msg="pz-dz2 interaction incorrect along z-axis")
    
        print("p-d interaction tests passed!")
    
    def test_dd_interaction(self):
        """Test d-d orbital interactions using surfGB parameters"""
    
        Vdict = self.Vdict
    
        # Test dyz-dyz interaction along x-axis
        M = self.construct_mat(Vdict, [1, 0, 0])
    
        # Should be pure delta interaction
        expected = Vdict['ddd']
        np.testing.assert_almost_equal(
            M[6,6], expected,
            err_msg="dyz-dyz interaction incorrect along x-axis")
    
        # Test dz2-dz2 interaction along x-axis
        M = self.construct_mat(Vdict, [0, 0, 1])
        # Should be pure sigma interaction
        expected = Vdict['dds']
        np.testing.assert_almost_equal(
            M[4,4], expected,
            err_msg="dz2-dz2 interaction incorrect along z-axis")
    
        print("d-d interaction tests passed!")
    def test_hopping_physics(self):
        """
        Test physical properties of hopping matrices for different bond directions.
        Verifies that hopping magnitudes and symmetries are preserved under rotation.
        """
        eps = 1e-10  # Tolerance for floating point comparisons
        
        # Get reference hopping values from [0,0,1] configuration
        s_p_mag = abs(self.Vdict['sps'])  # Magnitude of s-p hopping
        
        # Test set of physically important directions
        test_cases = [
            # Principal axes
            ([0, 0, 1], "z-axis"),
            ([1, 0, 0], "x-axis"),
            ([0, 1, 0], "y-axis"),
            
            # 45-degree rotations
            ([1/np.sqrt(2), 0, 1/np.sqrt(2)], "45° in xz-plane"),
            ([0, 1/np.sqrt(2), 1/np.sqrt(2)], "45° in yz-plane"),
            ([1/np.sqrt(2), 1/np.sqrt(2), 0], "45° in xy-plane"),
        ]
        
        print("\nTesting hopping matrix physics...")
        
        for direction, name in test_cases:
            direction = np.array(direction)
            x, y, z = direction
            
            print(f"\nChecking {name} direction: [{x:.3f}, {y:.3f}, {z:.3f}]")
            V = self.construct_mat(self.Vdict, direction)
            
            # Check s-p hopping antisymmetry
            for i in range(1, 4):  # Check all p orbitals
                assert abs(V[0,i] + V[i,0]) < eps, \
                    f"s-p hopping not antisymmetric for p{i}"
                    
            # Check total s-p hopping magnitude is preserved
            s_p_total = np.sqrt(V[0,1]**2 + V[0,2]**2 + V[0,3]**2)
            assert abs(s_p_total - s_p_mag) < eps, \
                f"s-p hopping magnitude not preserved: {s_p_total:.6f} != {s_p_mag:.6f}"
                
            # Print values for verification
            print(f"s-px: {V[0,1]:.3f}, px-s: {V[1,0]:.3f}")
            print(f"s-py: {V[0,2]:.3f}, py-s: {V[2,0]:.3f}")
            print(f"s-pz: {V[0,3]:.3f}, pz-s: {V[3,0]:.3f}")
            print(f"Total s-p magnitude: {s_p_total:.3f}")
    
        print("\nAll hopping physics tests passed!")    # Update run_all_tests to include new test

    def run_all_tests(self):
        """Run all validation tests for surfGB"""
        print("Running Slater-Koster projection tests...")
        self.test_d_orbital_functions()
        self.test_d_orbital_symmetry()
        self.test_pd_interaction()
        self.test_dd_interaction()
        self.test_hopping_physics()
        print("\nAll tests passed!")

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
        self.NN = len(Slist)
        assert self.NN == 12, "Error: surfGBAt only implemented for FCC using 12 NN"
        self.eta = eta
        self.sigmaKprev = None
        self.Eprev = Eminf

        # For compatibility with density methods:
        self.F = self.H
        self.S = np.eye(dim)
    
    # Calculate bulk atom Green's function, i is a dummy variable for compatibility
    def g(self, E, i, conv=1e-5, mix=0.5):
        #Initialize sigmaK and A matrices for Dyson equation
        if self.sigmaKprev is not None and self.Eprev != Eminf and abs(self.Eprev - E) <1:
            sigmaK = self.sigmaKprev.copy()
        else:
            sigmaK = np.array([np.eye(dim)*(1*self.eta) for k in range(self.NN)], dtype=complex)
        A = (E + self.eta*1j)*np.eye(dim) - self.H
        
        #Self-consistency loop 
        count = 0
        maxIter = int(1/(conv*mix))*10
        diff = np.inf
        while diff > conv and count < maxIter:
            sigmaK_ = sigmaK.copy()
            sigTot = np.sum(sigmaK, axis=0)
           
            for k in range(self.NN):
                pair_k = (k + 6)%12 # Opposite direction vector
                gK = LA.inv(A - sigTot + sigmaK[pair_k]) # subtracted from sigTot
                B = (E + self.eta*1j)*self.Slist[k] - self.Vlist[k]
                sigmaK[k] = (B.conj().T@gK@B) + (1-mix)*sigmaK_[k]
            
            # Convergence Check
            diff = np.max(np.abs(sigmaK - sigmaK_))/np.max(np.abs(sigmaK_))
            count += 1

        if diff>conv:
            print(f'Warning: exceeded max iterations! E: {E}, Conv: {diff}')
        
        #Print statement if took more than 5000 iterations to converge
        if count>5000:
            print(f'gAtom at {E:.2e} converged in {count} iterations: {diff}')
        
        self.sigmaKprev = sigmaK
        self.Eprev= E
        return LA.inv(A -  np.sum(sigmaK, axis=0))
    
    # Return self-energy matrix associated with single surface atom
    def sigmaTot(self, E, conv=1e-5):
        sig = np.zeros((dim,dim), dtype=complex)
        gSurf = self.g(E, 0, conv)
        for k in range(9): #Omitting back plane vectors
            B = (E + self.eta*1j)*self.Slist[k] - self.Vlist[k]
            sig += B.conj().T@gSurf@B
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
            calcN_=calcN+0.0
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
            print(f"Range: {Emin}, {Emax} - calcN = {calcN}")
        if cycles == maxCycles:
            print(f"Warning: Energy range ({Emin}, {Emax}) not converged (val = {calcN - dim})")
        self.fermi = calcFermi(self, ne, Emin, Emax, fGuess, Nint, 1, Eminf, tol)[0]
        return self.fermi

