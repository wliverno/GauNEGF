# Python packages
import numpy as np
from numpy import linalg as LA

# Developed packages
from gauNEGF.density import *
from gauNEGF.config import (ETA, TEMPERATURE, ENERGY_MIN)
from gauNEGF.utils import fractional_matrix_power

#Constants
kB = 8.617e-5           # eV/Kelvin
dim = 9                 # size of single atom matrix: 1s + 3p + 5d
har_to_eV = 27.211386   # eV/Hartree
Eminf = ENERGY_MIN      # Setting lower bound to -1e6 eV

# Bethe lattice surface Green's function for a device with contacts
class surfG3:
    """
    Surface Green's function calculator for 3D lattice. 
    
    Work in progress- need to implement k-space integration (Gamma only)

    Parameters
    ----------
    F : ndarray
        Fock matrix from DFT calculation
    S : ndarray
        Overlap matrix from DFT calculation
    contacts : list of lists
        Lists of atom indices for each contact region
    bar : QCBinAr
        Gaussian interface object containing geometry and orbital information
    latFile : str, optional
        Name of .bethe file containing Slater-Koster parameters (default: 'Au')
    spin : {'r', 'u', 'ro', 'g'}, optional
        Spin treatment: restricted, unrestricted, or generalized (default: 'r')
    eta : float, optional
        Broadening parameter in eV (default: 1e-9)
    T : float, optional
        Temperature in Kelvin (default: 0)

    Attributes
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    gList : list of surfGAt
        List of atomic surface Green's function calculators for each contact
    """
    def __init__(self, F, S, contacts, bar,  latFile='Au', spin='r', eta=ETA, T=TEMPERATURE):
        #Read contact/orbital information and store
        self.cVecs = []
        self.latVecs = []
        self.indsLists = []
        self.dirLists = []
        self.nIndLists = []
        self.Xi = fractional_matrix_power(S, 0.5)
        if spin != 'r':
            self.Xi = self.Xi[::2, ::2]
        
        # Spin independent implementation, add degenerate spin terms during sigma generation
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
                cList.append(np.array(bar.c[(atom-1)*3:atom*3]))
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
            
            # Use lattice vectors to see what nearest neighbors
            nIndList = []
            for c in cList:
                nAtVecs = []
                for c2 in cList:
                    l = LA.norm(c2-c)
                    # if within 1.5*nearest neighbor dist and not the same atom
                    if l < 1.5 * LA.norm(latVec) and not np.allclose(c2, c):
                        nAtVecs.append((c2-c)/l) #Unit vector for that direction
                nInds = []
                for vec in nAtVecs:
                    valList = [np.dot(vec, direction) for direction in nVecs]
                    nInds.append(np.argmax(valList))
                    assert valList[nInds[-1]] > 0.9 and nInds[-1] in [0,1,2,6,7,8], \
                             'Error: Lattice mismatch in atoms!'
                # write neighbor indices for each atom
                nIndList.append(nInds)
            # write direction vectors and neighbors for each contact
            self.nIndLists.append(nIndList)
            self.dirLists.append(nVecs)

        
        # Read Bethe lattice parameters and generate hopping/overlap matrices
        self.readBetheParams(latFile)
        self.Slists = []
        self.Vlists = []
        for dirList in self.dirLists: 
            # Construct hopping matrices and store to contact
            Slist = []
            Vlist = []
            for d in dirList:
                Slist.append(self.constructMat(self.Sdict, d))
                Vlist.append(self.constructMat(self.Vdict, d))
            self.Slists.append(Slist)
            self.Vlists.append(Vlist)
        # Use surfGBAt() object to store the atomic Bethe lattice green's function for each contact
        self.gList = [surfGAt(self.H0.copy(), Slist, Vlist, eta, T) for Slist, Vlist in zip(self.Slists, self.Vlists)]
        
        for g in self.gList:
            g.calcFermi(self.ne/2)

        # Store variables
        self.cList = cList #first contact coords, used for testing
        self.F = F
        self.S = S
        self.eta = eta

    def genNeighbors(self, plane_normal, first_neighbor):
        """
        Generate 12 nearest neighbor unit vectors for an FCC [111] surface.

        Creates a list of unit vectors representing the 12 nearest neighbors in an FCC lattice:
        - 6 in-plane vectors forming a hexagonal pattern (3 pairs of opposite vectors)
        - 6 out-of-plane vectors forming triangular patterns (3 pairs of opposite vectors)

        Parameters
        ----------
        plane_normal : ndarray
            Vector normal to the crystal plane (will be normalized)
        first_neighbor : ndarray
            Vector to one nearest neighbor (will be projected onto plane)

        Returns
        -------
        list
            12 unit vectors representing nearest neighbor directions
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
        out_of_plane_angle = np.arccos(1/np.sqrt(3)) # ~54.74
        
        out_of_plane_vectors = []
        # Add 30° = pi/6 rotation to base vector before going out of plane
        rot_angle = np.pi/6
        K = np.array([[0, -plane_normal[2], plane_normal[1]],
                      [plane_normal[2], 0, -plane_normal[0]],
                      [-plane_normal[1], plane_normal[0], 0]])
        R = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * np.matmul(K, K)
        rotated_first = np.dot(R, first_neighbor)
        out_of_plane_base = np.cos(out_of_plane_angle) * rotated_first + \
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
    def readBetheParams(self, filename):
        """
        Read Slater-Koster parameters from a .bethe file.

        Reads and validates parameters for minimal basis with single s, p, and d orbitals.
        Parameters are stored in dictionaries for onsite energies, hopping integrals,
        and overlap matrices.

        Parameters
        ----------
        filename : str
            Name of the .bethe file (without extension)

        Raises
        ------
        AssertionError
            If parameters are missing or invalid

        Notes
        -----
        Parameters are sorted into:
        - Edict: Onsite energies (converted from Hartrees to eV)
        - Vdict: Hopping parameters (converted from Hartrees to eV)
        - Sdict: Overlap parameters
        """
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

    def constructMat(self, Mdict, dirCosines):
        """
        Construct hopping/overlap matrix using Slater-Koster formalism.

        Builds a 9x9 matrix for s, p, and d orbital interactions based on the
        Slater-Koster two-center approximation. The matrix is first constructed
        assuming a [0,0,1] bond direction, then rotated to the given direction
        using direction cosines.

        Parameters
        ----------
        Mdict : dict
            Dictionary of Slater-Koster parameters (ssσ, spσ, ppσ, etc.)
        dirCosines : ndarray
            Array [l,m,n] of direction cosines for the bond

        Returns
        -------
        ndarray
            9x9 matrix containing orbital interactions in the rotated frame

        Notes
        -----
        Matrix blocks:
        - [0,0]: s-s interaction
        - [0:4,0:4]: s-p block
        - [0:4,4:9]: s-d and p-d blocks
        - [4:9,4:9]: d-d block
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
        return tr @ M @ tr.T
    
    def sigma(self, E, i, conv=1e-5):
        """
        Calculate self-energy matrix for a specific contact.

        Computes the self-energy matrix for contact i by:
        1. Calculating surface self-energies for all 9 directions
        2. Summing contributions from directions not connected to the device
        3. Applying de-orthonormalization if needed
        4. Handling spin configurations

        Parameters
        ----------
        E : float
            Energy point for self-energy calculation (in eV)
        i : int
            Index of the contact to calculate self-energy for
        conv : float, optional
            Convergence criterion for self-energy calculation (default: 1e-5)

        Returns
        -------
        ndarray
            Self-energy matrix for the specified contact, with dimensions:
            - (N, N) for restricted calculations
            - (2N, 2N) for unrestricted or generalized spin calculations

        References
        ----------
        [1] Jacob, D., & Palacios, J. J. (2011). Critical comparison of electrode models 
            in density functional theory based quantum transport calculations.
            The Journal of Chemical Physics, 134(4), 044118.
            DOI: 10.1063/1.3526044  
        """
        sig = np.zeros((self.N, self.N), dtype=complex)
        sigSurf = self.gList[i].sigma(E, None, conv)
        # Apply self energies in first 9 directions that aren't attached to atom
        for nInds, Finds in zip(self.nIndLists[i], self.indsLists[i]):
            sigInds = list(set(range(9)) - set(nInds))
            sigAtom = sum([sigSurf[j] for j in sigInds])
            sig[np.ix_(Finds, Finds)] = sigAtom
        # Apply de-orthonormalization technique from ANT.Gaussian if orthonormal
        if self.Sdict['sss'] == 0:
            sig = times(self.Xi, sig, self.Xi)
        if self.spin == 'u' or self.spin == 'ro':
            sig = np.kron(np.eye(2), sig)
        elif self.spin =='g':
            sig = np.kron(sig, np.eye(2))
        return sig
    
    def sigmaTot(self, E, conv=1e-5):
        """
        Calculate total self-energy matrix for the extended system.

        Computes self-energies for all sites in the extended system (12 neighbors + 1 center).
        The total self-energy is constructed following the Bethe lattice model described in
        Jacob & Palacios [1], which provides an efficient representation of bulk metallic
        electrodes while maintaining proper orbital symmetries.

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)
        conv : float, optional
            Convergence criterion for self-energy calculation (default: 1e-5)

        Returns
        -------
        ndarray
            Total self-energy matrix for the extended system

        References
        ----------
        [1] Jacob, D., & Palacios, J. J. (2011). Critical comparison of electrode models 
            in density functional theory based quantum transport calculations.
            The Journal of Chemical Physics, 134(4), 044118.
            DOI: 10.1063/1.3526044
        """
        sigs = [self.sigma(E, i, conv) for i in range(len(self.indsLists))]
        return sum(sigs)

    def getSigma(self, Elist=[None, None], conv=1e-5):
        """
        Helper method for getting the left and right contact self-energies
 
        Parameters
        ----------
        Elist : tuple, optional
            A list of contact energies for selecting sigma, 
            (default: use contact ermi energy)
        conv: float, optional
            Convergence criterial for the self-energy matrix

        Returns
        -------
        tuple
            A tuple of both self-energy matrices (ndarrays)
        """
        if Elist[0] is None:
            Elist[0] = self.gList[0].fermi
        if Elist[1] is None:
            Elist[1] = self.gList[-1].fermi
        return (self.sigma(Elist[0], 0, conv), self.sigma(Elist[1], -1, conv))

    
    def updateFermi(self, i, Ef):
        """
        Update Fermi energy for a specific contact.

        Shifts the Hamiltonian of contact i to align its Fermi level with
        the specified energy.

        Parameters
        ----------
        i : int
            Contact index
        Ef : float
            New Fermi energy in eV
        """
        fermiPrev = self.gList[i].fermi +0.0
        #if i==-1:
        #    print(f'Changing right contact fermi energy: {fermiPrev} --> {Ef}')
        #elif i==0:
        #    print(f'Changing left contact fermi energy: {fermiPrev} --> {Ef}')
        #else:
        #    print(f'Changing contact {i+1} fermi energy: {fermiPrev} --> {Ef}')
        # Onsite energies
        self.gList[i].updateH(Ef)
    
    def setF(self, F, muL, muR):
        """
        Update Fock matrix and contact chemical potentials.

        Sets the Fock matrix and updates the Fermi levels of the left and
        right contacts if they have changed.

        Parameters
        ----------
        F : ndarray
            New Fock matrix
        muL : float
            Chemical potential for left contact in eV
        muR : float
            Chemical potential for right contact in eV
        """
        self.F = F
        if self.gList[0].fermi != muL:
            self.updateFermi(0, muL)
        if self.gList[-1].fermi != muR:
            self.updateFermi(-1, muR) 

   
    ## TESTING METHODS FOR SLATER-KOSTER INTERACTIONS:
    def testDOrbitalFunctions(self):
        """
        Test d orbital angular functions.

        Validates the angular dependence of d orbital interactions by checking:
        - dxy interaction along x-axis (should be zero)
        - dx2-y2 interaction along x-axis (should be sqrt(3)/2 * sds)
        - dz2 interaction along x-axis (should be -1/2 * sds)
        """
    
        # Use values from the Bethe parameter dictionaries
        Vdict = self.Vdict  # Contains hopping parameters
    
        # Test along x-axis [1,0,0]
        M = self.constructMat(self.Vdict, [1, 0, 0])
    
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
    
    def testDOrbitalSymmetry(self):
        """
        Test d orbital symmetry properties.

        Validates that d orbital interactions respect inversion symmetry
        by comparing interactions along opposite directions.
        """
    
        # Test inversion symmetry
        dir1 = [1/np.sqrt(2), 1/np.sqrt(2), 0]
        dir2 = [-1/np.sqrt(2), -1/np.sqrt(2), 0]
    
        M1 = self.constructMat(self.Vdict, dir1)
        M2 = self.constructMat(self.Vdict, dir2)
    
        # d-d block should be identical under inversion
        np.testing.assert_array_almost_equal(
            M1[4:,4:], M2[4:,4:],
            err_msg="d-d block not symmetric under inversion")
    
        print("d orbital symmetry tests passed!")
    
    def testPDInteraction(self):
        """
        Test p-d orbital interactions.

        Validates p-d orbital interactions by checking:
        - px-dxy interaction along x-axis (should be zero)
        - pz-dz2 interaction along z-axis (should be pure sigma)
        """
    
        Vdict = self.Vdict
    
        # Test px-dxy interaction along x-axis
        M = self.constructMat(Vdict, [1, 0, 0])
    
        # px-dxy should be zero along x-axis
        np.testing.assert_almost_equal(
            M[1,8], 0.0,
            err_msg="px-dxy interaction incorrect along x-axis")
    
        # Test pz-dz2 interaction along z-axis
        M = self.constructMat(Vdict, [0, 0, 1])
        expected = Vdict['pds']  # Should be pure sigma
        np.testing.assert_almost_equal(
            M[3,4], expected,
            err_msg="pz-dz2 interaction incorrect along z-axis")
    
        print("p-d interaction tests passed!")
    
    def testDDInteraction(self):
        """
        Test d-d orbital interactions.

        Validates d-d orbital interactions by checking:
        - dyz-dyz interaction along x-axis (should be pure delta)
        - dz2-dz2 interaction along z-axis (should be pure sigma)
        """
    
        Vdict = self.Vdict
    
        # Test dyz-dyz interaction along x-axis
        M = self.constructMat(Vdict, [1, 0, 0])
    
        # Should be pure delta interaction
        expected = Vdict['ddd']
        np.testing.assert_almost_equal(
            M[6,6], expected,
            err_msg="dyz-dyz interaction incorrect along x-axis")
    
        # Test dz2-dz2 interaction along x-axis
        M = self.constructMat(Vdict, [0, 0, 1])
        # Should be pure sigma interaction
        expected = Vdict['dds']
        np.testing.assert_almost_equal(
            M[4,4], expected,
            err_msg="dz2-dz2 interaction incorrect along z-axis")
    
        print("d-d interaction tests passed!")
    def testHoppingPhysics(self):
        """
        Test physical properties of hopping matrices.

        Validates hopping matrix physics by checking:
        - s-p hopping antisymmetry
        - Conservation of total s-p hopping magnitude
        - Proper angular dependence along principal axes and 45-degree rotations
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
            V = self.constructMat(self.Vdict, direction)
            
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

    def runAllTests(self):
        """
        Run all validation tests for surfG.

        Executes all test methods to validate:
        - d orbital angular functions
        - d orbital symmetry
        - p-d interactions
        - d-d interactions
        - General hopping physics
        """
        print("Running Slater-Koster projection tests...")
        self.testDOrbitalFunctions()
        self.testDOrbitalSymmetry()
        self.testPDInteraction()
        self.testDDInteraction()
        self.testHoppingPhysics()
        print("\nAll tests passed!")

# Bethe lattice surface Green's function for a single atom
class surfGAt:
    """
    Atomic-level Bethe lattice Green's function calculator.

    This class implements the surface Green's function calculation for a single atom
    in the Bethe lattice, handling:
    - Onsite and hopping matrix construction
    - Self-energy calculations for bulk and surface
    - Temperature effects
    - Fermi energy optimization

    Parameters
    ----------
    H : ndarray
        Onsite Hamiltonian matrix (9x9 for minimal basis)
    Slist : list of ndarray
        List of 12 overlap matrices for nearest neighbors
    Vlist : list of ndarray
        List of 12 hopping matrices for nearest neighbors
    eta : float
        Broadening parameter in eV
    T : float, optional
        Temperature in Kelvin (default: 0)

    Attributes
    ----------
    NN : int
        Number of nearest neighbors (fixed to 12 for FCC)
    sigmaKprev : ndarray or None
        Previous bulk self-energy for convergence
    Eprev : float
        Previous energy point for convergence
    fermi : float
        Current Fermi energy
    F : ndarray
        Extended Fock matrix including neighbors
    S : ndarray
        Extended overlap matrix including neighbors
    """
    def __init__(self, H, Slist, Vlist, eta, T=TEMPERATURE):
        """
        Initialize surfGAt with Hamiltonian and neighbor matrices.

        Parameters
        ----------
        H : ndarray
            Onsite Hamiltonian matrix (9x9 for minimal basis)
        Slist : list of ndarray
            List of 12 overlap matrices for nearest neighbors
        Vlist : list of ndarray
            List of 12 hopping matrices for nearest neighbors
        eta : float
            Broadening parameter in eV
        T : float, optional
            Temperature in Kelvin (default: 0)

        Raises
        ------
        AssertionError
            If matrix dimensions are incorrect or number of neighbors != 12
        """
        assert np.shape(H) == (dim,dim), f"Error with H dim, should be {dim}x{dim}"
        for S,V in zip(Slist, Vlist):
            assert np.shape(S) == (dim,dim), f"Error with S dim, should be {dim}x{dim}"
            assert np.shape(V) == (dim,dim), f"Error with F dim, should be {dim}x{dim}"
        self.H = H
        self.Slist = Slist
        self.Vlist = Vlist
        self.NN = len(Slist)
        assert self.NN == 12, "Error: surfGAt only implemented for FCC using 12 NN"
        #self.Slist = [np.zeros((dim,dim)) for n in range(self.NN)] #To match ANT.Gaussian default
        self.eta = eta
        self.T = T
        self.sigmaKprev = None
        self.Eprev = Eminf
        self.fermi = None

        self.updateH()

    def updateH(self, fermi=None):
        """
        Update Hamiltonian and extended matrices.

        Updates onsite and hopping matrices, as well as extended lattice matrices.
        The extended matrices H0x and S0x include 13 sites total (12 neighbor sites
        followed by 1 onsite term). These are stored as F and S for compatibility
        with density.py functions.

        Parameters
        ----------
        fermi : float, optional
            New Fermi energy setpoint in eV (default: None)

        Notes
        -----
        When fermi is provided and different from current value:
        - Shifts onsite energies by the Fermi level difference
        - Updates hopping matrices with overlap contributions
        - Rebuilds extended matrices for the full system
        """
        if fermi is not None and self.fermi is not None and fermi != self.fermi:
            # Shift fermi energy
            fermiPrev = self.fermi
            dFermi =  fermi - fermiPrev
            # Onsite energies
            self.H += dFermi*np.eye(dim)
            # And hopping overlaps
            for j,S in enumerate(self.Slist):
                self.Vlist[j] += dFermi*S
            #print(np.diag(self.H))
            self.fermi = fermi

        H0x = np.kron(np.eye(self.NN+1), self.H)
        S0x = np.eye(dim*(self.NN+1))
        for i in range(self.NN):
            S0x[-dim:, i*dim:(i+1)*dim] = self.Slist[i]
            S0x[i*dim:(i+1)*dim, -dim:] = self.Slist[i].T
            H0x[-dim:, i*dim:(i+1)*dim] = self.Vlist[i]
            H0x[i*dim:(i+1)*dim, -dim:] = self.Vlist[i].conj().T
        self.F = H0x
        self.S = S0x
    
    # Calculate sigmaK for the bulk
    def sigmaK(self, E, conv=1e-5, mix=0.5):
        """
        Calculate bulk self-energies for all 12 lattice directions.

        Computes self-energies for an FCC lattice with the following geometry:
                [3x out of plane dir]
                        \|/  
        [3x plane dir] - o - [3x plane dir]
                        /|\     
                [3x out of plane dir]

        Uses a self-consistent iteration scheme with mixing to solve the Dyson equation.

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)
        conv : float, optional
            Convergence criterion for Dyson equation (default: 1e-5)
        mix : float, optional
            Mixing factor for Dyson equation (default: 0.5)

        Returns
        -------
        ndarray
            Array of 12 self-energy matrices (9x9 each) in order by lattice direction

        Notes
        -----
        Uses previous solution as initial guess when energy point is close to
        previous calculation to improve convergence.
        """
        #Initialize sigmaK and A matrices for Dyson equation
        if self.sigmaKprev is not None and self.Eprev != Eminf and abs(self.Eprev - E) <1:
            sigmaK = self.sigmaKprev.copy()
        else:
            sigmaK = np.array([np.eye(dim)*-1j for k in range(self.NN)], dtype=complex)
        A = (E - self.eta*1j)*np.eye(dim) - self.H
        #Self-consistency loop 
        count = 0
        maxIter = 1000
        diff = np.inf
        while diff > conv and count < maxIter:
            sigmaK_ = sigmaK.copy()
            sigTot = np.sum(sigmaK, axis=0)
            gK = LA.inv(A - sigTot)
           
            for k in range(self.NN):
                B = (E - self.eta*1j)*self.Slist[k] - self.Vlist[k]
                sigmaK[k] = mix*(B@gK@B.conj().T) + (1-mix)*sigmaK_[k]
            
            # Convergence Check
            diff = np.max(np.abs(sigmaK - sigmaK_))/np.max(np.abs(sigmaK_))
            count += 1

        if diff>conv:
            print(f'Warning: sigmaK() exceeded 1000 iterations! E: {E}, Conv: {diff}')
        
        self.sigmaKprev = sigmaK
        self.Eprev= E

        return sigmaK

    def sigma(self, E, inds=None, conv=1e-5, mix=0.5): 
        """
        Calculate surface self-energies for an FCC lattice.

        Computes self-energies for atoms at the surface with the geometry:
        [3x plane dir] - o - [3x plane dir]
                        /|\     
                [3x out of plane dir]

        Uses a self-consistent iteration scheme with mixing to solve the Dyson equation.
        The implementation follows the Bethe lattice approach described in Jacob & Palacios (2011),
        where the self-energy is computed recursively for a semi-infinite tree-like structure
        that preserves the proper coordination number and orbital symmetries of bulk FCC metals.

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)
        inds : list or int, optional
            Indices of the sigma matrix to return. If None, returns full list (default: None)
        conv : float, optional
            Convergence criterion for Dyson equation (default: 1e-5)
        mix : float, optional
            Mixing factor for Dyson equation (default: 0.5)

        Returns
        -------
        list
            List of self-energy matrices for the surface atom. If inds is specified,
            returns only the requested matrices.

        Notes
        -----
        First calculates bulk self-energies using sigmaK, then iterates to find
        surface self-energies for the 9 surface directions. The recursive method
        ensures proper treatment of the metal-molecule interface while maintaining
        computational efficiency.

        References
        ----------
        [1] Jacob, D., & Palacios, J. J. (2011). Critical comparison of electrode models 
            in density functional theory based quantum transport calculations.
            The Journal of Chemical Physics, 134(4), 044118.
            DOI: 10.1063/1.3526044
        """
        sigSurf = self.sigmaK(E, conv, mix)[:9]
        #Self-consistency loop 
        count = 0
        maxIter = 1000
        diff = np.inf                             ## SET THIS TO 0 to BYPASS SECOND LOOP
        A = (E - self.eta*1j)*np.eye(dim) - self.H
        planeVec = [0,1,2,6,7,8] # Location of vectors in plane
        while diff > conv and count < maxIter:
            sigSurf_ = sigSurf.copy()
            sigTot = np.sum(sigSurf, axis=0)
            g = LA.inv(A - sigTot) # subtracted from sigTot
            
            for k in planeVec:
                B = (E - self.eta*1j)*self.Slist[k] - self.Vlist[k]
                sigSurf[k] = mix*(B@g@B.conj().T) + (1-mix)*sigSurf_[k]
            
            # Convergence Check
            diff = np.max(np.abs(sigSurf - sigSurf_))/np.max(np.abs(sigSurf_))
            count += 1

        if diff>conv:
            print(f'Warning: sigma() exceeded 1000 iterations! E: {E}, Conv: {diff}')
        
        if inds is None:
            return sigSurf
        else:
            return [sigSurf[i] for i in inds]
    
    # Empty function for compatibility with density.py methods
    def setF(self, F, mu1, mu2):
        """
        Empty function for compatibility with density.py methods.

        Bethe lattice bulk properties are intrinsic (dependent on TB parameters).

        Parameters
        ----------
        F : ndarray
            Fock matrix (unused)
        mu1 : float
            First chemical potential (unused)
        mu2 : float
            Second chemical potential (unused)
        """
        pass # Bethe lattice bulk properties are intrinsic (dependent on TB parameters)
    
    def sigmaTot(self, E, conv=1e-5):
        """
        Calculate total self-energy matrix for the extended Bethe lattice system.

        Computes self-energies for the full extended system including 12 neighbor sites
        plus 1 central site. This is a wrapper function for compatibility with density.py
        methods that require a single total self-energy matrix.

        Parameters
        ----------
        E : float
            Energy point for self-energy calculation (in eV)
        conv : float, optional
            Convergence criterion for self-energy calculation (default: 1e-5)

        Returns
        -------
        ndarray
            Total self-energy matrix for the extended system ((NN+1)*dim, (NN+1)*dim)

        Notes
        -----
        For each neighbor direction k, the self-energy includes contributions from all
        other directions except the opposite direction (k+6)%12, following the Bethe
        lattice construction.
        """
        sig = np.zeros(((self.NN + 1)*dim, (self.NN+1)*dim), dtype=complex)
        sigK = self.sigmaK(E, conv)
        sigTot = np.sum(sigK, axis=0)
        for k in range(self.NN):
            pair_k = (k + 6)%12 # Opposite direction vector
            sig[k*dim:(k+1)*dim,k*dim:(k+1)*dim] = sigTot - sigK[pair_k]
        return sig
    
    # Get the bulk DOS of the Bethe lattice
    def DOS(self, E):
        """
        Calculate bulk density of states of the Bethe lattice.

        Parameters
        ----------
        E : float
            Energy point for DOS calculation (in eV)

        Returns
        -------
        float
            Density of states at energy E
        """
        Gr = LA.inv((E-1j*self.eta)*np.eye(dim)- self.H - np.sum(self.sigma(E), axis=0))
        return -np.trace(Gr).imag/np.pi

    
    # Calculate fermi energy using bisection (to specified tolerance)
    def calcFermi(self, ne, tol=1e-5):
        """
        Calculate Fermi energy using bisection method.

        Uses getFermiContact from density.py to find the Fermi energy that gives
        the correct number of electrons.

        Parameters
        ----------
        ne : float
            Target number of electrons
        tol : float, optional
            Convergence tolerance (default: 1e-5)

        Returns
        -------
        float
            Calculated Fermi energy in eV

        Notes
        -----
        Previous implementation used ANT.Gaussian approach with complex contour
        integration. Current version uses simpler bisection method from density.py.
        """
        self.fermi = getFermiContact(self, ne, tol, Eminf, 1000, T=self.T, nOrbs=dim)
        return self.fermi

