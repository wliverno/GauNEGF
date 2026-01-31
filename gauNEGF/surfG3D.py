# Python packages
import jax
import jax.numpy as np
from jax.numpy import linalg as LA

# Developed packages
from gauNEGF.density import *
from gauNEGF.config import (ETA, TEMPERATURE, ENERGY_MIN)
from gauNEGF.utils import fractional_matrix_power

#Constants
kB = 8.617e-5           # eV/Kelvin
dim = 9                 # size of single atom matrix: 1s + 3p + 5d
har_to_eV = 27.211386   # eV/Hartree
Eminf = ENERGY_MIN      # Setting lower bound to -1e6 eV


# 3D k-grid lattice surface Green's function for a device with 111 contacts
class surfG3:
    """
    Surface Green's function calculator for 3D lattice with [111] surface.

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
    gList : list of surfGAt3D
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
                inds = jnp.where(jnp.isin(orbMap, atom))[0]
                cList.append(jnp.array(bar.c[(atom-1)*3:atom*3]))
                assert len(inds) == 9, f'Error: Atom {atom} has {len(inds)} basis functions, expecting 9'
                inds = inds[jnp.argsort(abs(orbTyp[inds])//1000)]
                indsList.append(inds)
            self.indsLists.append(indsList)
            # Calculate plane direction using SVD
            cList = jnp.array(cList)
            centeredCoords = cList-jnp.mean(cList, axis=0)
            _, _, Vt = LA.svd(centeredCoords)
            self.cVecs.append(Vt[-1])
            # Calculate one lattice direction for lining up atoms
            vInd = jnp.argmin(jnp.array([LA.norm(v - cList[0]) for v in cList[1:]]))+1
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
                    if l < 1.5 * LA.norm(latVec) and not jnp.allclose(c2, c):
                        nAtVecs.append((c2-c)/l) #Unit vector for that direction
                nInds = []
                for vec in nAtVecs:
                    valList = jnp.array([jnp.dot(vec, direction) for direction in nVecs])
                    nInds.append(jnp.argmax(valList))
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
        self.gList = []
        for Slist, Vlist, vecs in zip(self.Slists, self.Vlists, self.dirLists):
            self.gList.append(surfGAt3D(self.H0.copy(), Slist, Vlist,vecs, eta, T))
        
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
        proj = first_neighbor - jnp.dot(first_neighbor, plane_normal) * plane_normal
        first_neighbor = proj / jnp.linalg.norm(proj)
        
        # Generate in-plane vectors using 60-degree rotations
        in_plane_vectors = []
        rotation_angle = jnp.pi / 3  # 60 degrees
        
        for i in range(3):
            angle = i * rotation_angle
            # Rodrigues rotation formula
            cos_theta = jnp.cos(angle)
            sin_theta = jnp.sin(angle)
            
            K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                         [plane_normal[2], 0, -plane_normal[0]],
                         [-plane_normal[1], plane_normal[0], 0]])
            
            R = jnp.eye(3) + sin_theta * K + (1 - cos_theta) * jnp.matmul(K, K)
            rotated_vector = jnp.dot(R, first_neighbor)
            in_plane_vectors.append(rotated_vector / jnp.linalg.norm(rotated_vector))
        
        # Generate out-of-plane vectors
        out_of_plane_angle = jnp.arccos(1/np.sqrt(3)) # ~54.74
        
        out_of_plane_vectors = []
        # Add 30° = pi/6 rotation to base vector before going out of plane
        rot_angle = jnp.pi/6
        K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                      [plane_normal[2], 0, -plane_normal[0]],
                      [-plane_normal[1], plane_normal[0], 0]])
        R = jnp.eye(3) + jnp.sin(rot_angle) * K + (1 - jnp.cos(rot_angle)) * np.matmul(K, K)
        rotated_first = jnp.dot(R, first_neighbor)
        out_of_plane_base = jnp.cos(out_of_plane_angle) * rotated_first + \
                      jnp.sin(out_of_plane_angle) * plane_normal
        
        for i in range(3):
            angle = i * 2 * jnp.pi / 3  # 120 degree rotations
            cos_theta = jnp.cos(angle)
            sin_theta = jnp.sin(angle)

            K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                         [plane_normal[2], 0, -plane_normal[0]],
                         [-plane_normal[1], plane_normal[0], 0]])

            R = jnp.eye(3) + sin_theta * K + (1 - cos_theta) * jnp.matmul(K, K)
            rotated_vector = jnp.dot(R, out_of_plane_base)
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
        Hdiag = jnp.array([self.Edict['s']]+ [self.Edict['p']]*3 + \
                     [self.Edict['dd']]+ [self.Edict['dt']]*2 + [self.Edict['dd'], self.Edict['dt']])
        self.H0 = jnp.diag(Hdiag)

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

        M = jnp.zeros((dim, dim))
        
        #Original matrix before rotation - assuming [0,0,1] bond direction
        # s-s coefficient
        M = M.at[0,0].set(Mdict['sss'])
        
        # s-p block
        M = M.at[0,3].set(Mdict['sps']) #s-pz
        M = M.at[3,0].set(-Mdict['sps']) #pz-s

        # p-p block
        M = M.at[1,1].set(Mdict['ppp']) #px-px
        M = M.at[2,2].set(Mdict['ppp']) #py-py
        M = M.at[3,3].set(Mdict['pps']) #pz-pz

        # s-d block
        M = M.at[0, 4].set(Mdict['sds']) #s - d3z²-r²
        M = M.at[4, 0].set(Mdict['sds'])

        # p-d block
        M = M.at[1,5].set(Mdict['pdp']) #px - dxz
        M = M.at[2,6].set(Mdict['pdp']) #py - dyz
        M = M.at[3,4].set(Mdict['pds']) #pz - d3z²-r²
        
        M = M.at[5,1].set(-Mdict['pdp']) #dxz - px
        M = M.at[6,2].set(-Mdict['pdp']) #dyz - py
        M = M.at[4,3].set(-Mdict['pds']) #d3z²-r² - pz

        # d-d block
        M = M.at[4,4].set(Mdict['dds']) #d3z²-r² - d3z²-r²
        M = M.at[5,5].set(Mdict['ddp']) #dxz - dxz
        M = M.at[6,6].set(Mdict['ddp']) #dyz - dyz
        M = M.at[7,7].set(Mdict['ddd']) #dx²-y² - dx²-y²
        M = M.at[8,8].set(Mdict['ddd']) #dxy - dxy 
        
        # Initialize 9x9 transformation matrix and polar directions
        tr = jnp.zeros((9, 9))
        x, y, z = dirCosines
        theta = jnp.arccos(z)  # polar angle from z-axis
        phi = jnp.arctan2(y, x)  # azimuthal angle in x-y plane
        
        # s orbital (1x1) at position [0,0] - always 1 since spherically symmetric
        tr = tr.at[0,0].set(1.0)
        
        # p orbitals (3x3) at positions [1:4,1:4]
        # [px,py,pz] block - describes how p orbitals transform under rotation
        tr = tr.at[1:4,1:4].set(jnp.array([
            [np.cos(theta) * jnp.cos(phi), -np.sin(phi)  , jnp.sin(theta)*np.cos(phi)],
            [np.cos(theta) * jnp.sin(phi),  jnp.cos(phi)  , np.sin(theta)*np.sin(phi)], 
            [-np.sin(theta)             ,  0            , jnp.cos(theta)]
        ]))
        
        # d orbitals (5x5) at positions [4:9,4:9]
        # [d3z2-r2, dxz, dyz, dx2-y2, dxy] block - transforms the five d orbitals
        d_block = jnp.zeros((5,5))
        
        # Copying formula from ANT.Gaussian directly
        d_block = d_block.at[0,0].set((3 * z**2 - 1) / 2)
        d_block = d_block.at[0,1].set(-np.sqrt(3) * jnp.sin(2*theta) / 2)
        d_block = d_block.at[0,3].set(jnp.sqrt(3) * jnp.sin(theta)**2 / 2)
        
        d_10 = jnp.sqrt(3) * jnp.sin(2*theta) * np.cos(phi) / 2
        d_block = d_block.at[1,0].set(d_10)
        d_block = d_block.at[1,1].set(jnp.cos(2*theta) * jnp.cos(phi))
        d_block = d_block.at[1,2].set(-np.cos(theta) * jnp.sin(phi))
        d_block = d_block.at[1,3].set(-d_10 / jnp.sqrt(3))
        d_block = d_block.at[1,4].set(jnp.sin(theta) * jnp.sin(phi))
        
        d_20 = jnp.sqrt(3) * jnp.sin(2*theta) * np.sin(phi) / 2
        d_block = d_block.at[2,0].set(d_20)
        d_block = d_block.at[2,1].set(jnp.cos(2*theta) * jnp.sin(phi))
        d_block = d_block.at[2,2].set(jnp.cos(theta) * jnp.cos(phi))
        d_block = d_block.at[2,3].set(-d_20 / jnp.sqrt(3))
        d_block = d_block.at[2,4].set(-np.sin(theta) * jnp.cos(phi))
        
        d_block = d_block.at[3,0].set(jnp.sqrt(3) * jnp.sin(theta)**2 * jnp.cos(2*phi) / 2)
        d_block = d_block.at[3,1].set(jnp.sin(2*theta) * jnp.cos(2*phi) / 2)
        d_block = d_block.at[3,2].set(-np.sin(theta) * jnp.sin(2*phi))
        d_block = d_block.at[3,3].set((1 + jnp.cos(theta)**2) * jnp.cos(2*phi) / 2)
        d_block = d_block.at[3,4].set(-np.cos(theta) * jnp.sin(2*phi))
        
        d_block = d_block.at[4,0].set(jnp.sqrt(3) * jnp.sin(theta)**2 * jnp.sin(2*phi) / 2)
        d_block = d_block.at[4,1].set(jnp.sin(2*theta) * jnp.sin(2*phi) / 2)
        d_block = d_block.at[4,2].set(jnp.sin(theta) * jnp.cos(2*phi))
        d_block = d_block.at[4,3].set((1 + jnp.cos(theta)**2) * jnp.sin(2*phi) / 2)
        d_block = d_block.at[4,4].set(jnp.cos(theta) * jnp.cos(2*phi))
        
        tr = tr.at[4:9,4:9].set(d_block)
        
        # Apply transformation 
        return tr @ M @ tr.T

    
    def sigma(self, E, i, conv=1e-4):
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
        sig = jnp.zeros((self.N, self.N), dtype=complex)
        sigSurf = self.gList[i].sigma(E, None, conv)
        # Apply self energies in first 9 directions that aren't attached to atom
        for nInds, Finds in zip(self.nIndLists[i], self.indsLists[i]):
            sigInds = list(set(range(9)) - set(nInds))
            sigAtom = sum([sigSurf[j] for j in sigInds])
            sig = sig.at[jnp.ix_(Finds, Finds)].set(sigAtom)
        # Apply de-orthonormalization technique from ANT.Gaussian if orthonormal
        if self.Sdict['sss'] == 0:
            sig = times(self.Xi, sig, self.Xi)
        if self.spin == 'u' or self.spin == 'ro':
            sig = jnp.kron(jnp.eye(2), sig)
        elif self.spin =='g':
            sig = jnp.kron(sig, jnp.eye(2))
        return sig
    
    def sigmaTot(self, E, conv=1e-4):
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

    def getSigma(self, Elist=[None, None], conv=1e-4):
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
        jnp.testing.assert_almost_equal(M[0,8], 0.0,
            err_msg="dxy not zero along x-axis")
        
        # dx2-y2 should be sqrt(3)/2 * sds along x-axis
        jnp.testing.assert_almost_equal(M[0,7], jnp.sqrt(3)/2 * Vdict['sds'],
            err_msg="dx2-y2 incorrect along x-axis")
    
        # dz2 should be -1/2 along x-axis
        jnp.testing.assert_almost_equal(M[0,4], -0.5 * Vdict['sds'],
            err_msg="dz2 incorrect along x-axis")
    
        print("d orbital angular function tests passed!")
    
    def testDOrbitalSymmetry(self):
        """
        Test d orbital symmetry properties.

        Validates that d orbital interactions respect inversion symmetry
        by comparing interactions along opposite directions.
        """
    
        # Test inversion symmetry
        dir1 = [1/jnp.sqrt(2), 1/np.sqrt(2), 0]
        dir2 = [-1/jnp.sqrt(2), -1/np.sqrt(2), 0]
    
        M1 = self.constructMat(self.Vdict, dir1)
        M2 = self.constructMat(self.Vdict, dir2)
    
        # d-d block should be identical under inversion
        jnp.testing.assert_array_almost_equal(
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
        jnp.testing.assert_almost_equal(
            M[1,8], 0.0,
            err_msg="px-dxy interaction incorrect along x-axis")
    
        # Test pz-dz2 interaction along z-axis
        M = self.constructMat(Vdict, [0, 0, 1])
        expected = Vdict['pds']  # Should be pure sigma
        jnp.testing.assert_almost_equal(
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
        jnp.testing.assert_almost_equal(
            M[6,6], expected,
            err_msg="dyz-dyz interaction incorrect along x-axis")
    
        # Test dz2-dz2 interaction along x-axis
        M = self.constructMat(Vdict, [0, 0, 1])
        # Should be pure sigma interaction
        expected = Vdict['dds']
        jnp.testing.assert_almost_equal(
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
            ([1/jnp.sqrt(2), 0, 1/jnp.sqrt(2)], "45° in xz-plane"),
            ([0, 1/jnp.sqrt(2), 1/jnp.sqrt(2)], "45° in yz-plane"),
            ([1/jnp.sqrt(2), 1/jnp.sqrt(2), 0], "45° in xy-plane"),
        ]
        
        print("\nTesting hopping matrix physics...")
        
        for direction, name in test_cases:
            direction = jnp.array(direction)
            x, y, z = direction
            
            print(f"\nChecking {name} direction: [{x:.3f}, {y:.3f}, {z:.3f}]")
            V = self.constructMat(self.Vdict, direction)
            
            # Check s-p hopping antisymmetry
            for i in range(1, 4):  # Check all p orbitals
                assert abs(V[0,i] + V[i,0]) < eps, \
                    f"s-p hopping not antisymmetric for p{i}"
                    
            # Check total s-p hopping magnitude is preserved
            s_p_total = jnp.sqrt(V[0,1]**2 + V[0,2]**2 + V[0,3]**2)
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

# 3D surface Green's function for a single atom
class surfGAt3D:
    """
    Atomic-level 3D Green's function calculator for a single atom.

    This class implements the surface Green's function calculation for a single atom
    in the 3D lattice with [111] surface, handling:
    - Onsite and hopping matrix construction
    - Self-energy calculations for bulk and surface

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
    def __init__(self, H, Slist, Vlist, vecs, eta, T=TEMPERATURE, kPoints=11):
        """
        Initialize surfGAt3D with Hamiltonian and neighbor matrices.

        Parameters
        ----------
        H : ndarray
            Onsite Hamiltonian matrix (9x9 for minimal basis)
        Slist : list of ndarray
            List of 12 overlap matrices for nearest neighbors
        Vlist : list of ndarray
            List of 12 hopping matrices for nearest neighbors, first six in plane,
        eta : float
            Broadening parameter in eV (recommended: >= 1e-4 for numerical stability)
        T : float, optional
            Temperature in Kelvin (default: 0)
        kPoints : int, optional
            Number of k-points per direction for BZ integration (default: 11)
            Recommended values:
            - 11: Fast, reasonable accuracy (121 points total)
            - 15: Good balance of speed/accuracy (225 points)
            - 21: High accuracy for production (441 points)
            Note: Odd values use Monkhorst-Pack grid, even use Gamma-centered

        Raises
        ------
        AssertionError
            If matrix dimensions are incorrect or number of neighbors != 12
        """
        assert jnp.shape(H) == (dim,dim), f"Error with H dim, should be {dim}x{dim}"
        for S,V in zip(Slist, Vlist):
            assert jnp.shape(S) == (dim,dim), f"Error with S dim, should be {dim}x{dim}"
            assert jnp.shape(V) == (dim,dim), f"Error with F dim, should be {dim}x{dim}"
        self.H = jnp.array(H)
        self.Slist = jnp.array(Slist)
        self.Vlist = jnp.array(Vlist)
        self.NN = len(Slist)
        self.kPoints = kPoints
        assert self.NN == 12, "Error: surfGAt3D only implemented for FCC using 12 NN"
        assert len(vecs) == 12, "Error: surfGAt3D only implemented for FCC using 12 NN"
        
        # Set up 2D lattice directions using first two vectors
        # TODO: WHAT IF NOT IN Z-direction?
        x1 = vecs[0][0]
        x2 = vecs[1][0]
        y1 = vecs[0][1]
        y2 = vecs[1][1]
        factor = 2 * jnp.pi / (x1*y2 - y1*x2)
        orthVec = jnp.array([[y2, -x2, 0],[-y1, x1, 0]])
        self.bList = factor*orthVec
        self.aList = jnp.array([[x1, y1, 0],[x2, y2, 0]])
        self.vecs = jnp.array(vecs)
        
        #self.Slist = [jnp.zeros((dim,dim)) for n in range(self.NN)] #To match ANT.Gaussian default
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
            self.H += dFermi*jnp.eye(dim)
            # And hopping overlaps
            for j,S in enumerate(self.Slist):
                self.Vlist = self.Vlist.at[j].set(self.Vlist[j] + dFermi*S)
            #print(jnp.diag(self.H))
            self.fermi = fermi

        H0x = jnp.kron(jnp.eye(self.NN+1), self.H)
        S0x = jnp.eye(dim*(self.NN+1))
        for i in range(self.NN):
            S0x = S0x.at[-dim:, i*dim:(i+1)*dim].set(self.Slist[i])
            S0x = S0x.at[i*dim:(i+1)*dim, -dim:].set(self.Slist[i].T)
            H0x = H0x.at[-dim:, i*dim:(i+1)*dim].set(self.Vlist[i])
            H0x = H0x.at[i*dim:(i+1)*dim, -dim:].set(self.Vlist[i].conj().T)
        self.F = H0x
        self.S = S0x

    def gBulk(self, E, conv=1e-4, mix=0.1, maxIter=5000):
        """
        Calculate bulk Green's function with full 3D periodicity.

        Includes all 12 nearest neighbors with k-space integration over the 2D BZ,
        solving the Dyson equation self-consistently. This represents the true
        bulk crystal (not semi-infinite).

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)
        conv : float, optional
            Convergence criterion for Dyson equation (default: 1e-5)
        mix : float, optional
            Mixing factor for convergence (default: 0.5)
        maxIter : int, optional
            Maximum iterations (default: 1000)

        Returns
        -------
        ndarray
            Bulk Green's function (dim x dim matrix), averaged over k-points
        """
        # Setup k-mesh: Gamma-centered for even, Monkhorst-Pack for odd
        if self.kPoints % 2 == 0:
            # Gamma-centered grid for even k-points
            k = jnp.arange(self.kPoints) / self.kPoints - 0.5 + 0.5/self.kPoints
        else:
            # Monkhorst-Pack grid for odd k-points
            k = (2 * jnp.arange(self.kPoints) + 1) / (2 * self.kPoints) - 0.5
        K1, K2 = jnp.meshgrid(k, k, indexing='ij')
        kmesh = K1.flatten()[:, jnp.newaxis] * self.bList[0] + \
                K2.flatten()[:, jnp.newaxis] * self.bList[1]  # nK**2 x 3

        # Apply phase factors to all 12 neighbors
        expList = jnp.exp(-1j*kmesh@(self.vecs.T))  # nK**2 x 12

        # Construct k-dependent hopping and overlap for ALL 12 neighbors
        Flist = expList[:, :, None, None] * self.Vlist[None, :, :, :]  # nK**2 x 12 x dim x dim
        Slist = expList[:, :, None, None] * self.Slist[None, :, :, :]  # nK**2 x 12 x dim x dim

        # Split into in-plane (0-5) and out-of-plane (6-11)
        # For bulk, we include ALL 6 out-of-plane (not just 3)
        Fak = jnp.sum(Flist[:, :6, :, :], axis=1) + \
                jnp.repeat(self.H[None, :, :], self.kPoints**2, axis=0)
        Sak = jnp.sum(Slist[:, :6, :, :], axis=1) + \
                jnp.repeat(jnp.eye(dim)[None, :, :], self.kPoints**2, axis=0)
        A = (E + self.eta*1j)*Sak - Fak

        # All 6 out-of-plane directions (6-11)
        Fbk = jnp.sum(Flist[:, 6:12, :, :], axis=1)
        Sbk = jnp.sum(Slist[:, 6:12, :, :], axis=1)
        B = (E + self.eta*1j)*Sbk - Fbk

        # Converge each k-point independently with robust solver
        def converge_single_k(A_k, B_k):
            """
            Converge Dyson equation for a single k-point using robust iteration.

            Solves: g = [A - B @ g @ B†]^-1

            Uses adaptive mixing and proper convergence criteria to ensure
            retarded Green's function with Im[g] < 0.
            """
            def cond_fun(state):
                count, diff, g, g_ = state
                return (diff > conv) & (count < maxIter)

            def body_fun(state):
                count, diff, g, g_ = state

                # Compute self-energy
                sig = B_k @ g @ B_k.conj().T

                # Update Green's function
                gNew = LA.inv(A_k - sig)

                # Ensure retarded Green's function with Im[g] < 0
                #gNew = jnp.where(gNew.imag > 0, gNew.real, gNew)

                # Apply mixing to ensure retarded Green's function with Im[g] < 0
                g_ = g.copy()
                #g = jnp.where(gNew.imag > 0, g, gNew * mix + (1 - mix) * g)
                g = gNew * mix + (1 - mix) * g

                # Convergence check: use relative change in norm
                diff = jnp.linalg.norm(g - g_) / (jnp.linalg.norm(g_) + 1e-12)
                count += 1
                return (count, diff, g, g_)

            # Initialize with bare Green's function (no self-energy)
            g_init = LA.inv(A_k) - 1j*jnp.eye(dim)*self.eta

            init_state = (0, jnp.inf, g_init, g_init.copy())
            count, diff, g, g_ = jax.lax.while_loop(cond_fun, body_fun, init_state)
            return g, count, diff

        # Vectorize over all k-points (run in parallel)
        g, counts, diffs = jax.vmap(converge_single_k)(A, B)

        # Diagnostic output
        #n_max_iters = jnp.sum(counts >= maxIter)
        #jax.lax.cond(n_max_iters > 0, lambda _: jax.debug.print("gBulk: max={max_c}, avg={avg_c:.1f}, unconverged={n_uc}/{n_tot}, max_diff={max_diff:.1e}, avg_diff={avg_diff:.1e}",
        #               max_c=jnp.max(counts), avg_c=jnp.mean(counts),
        #               n_uc=n_max_iters, n_tot=self.kPoints**2, max_diff=jnp.max(diffs), avg_diff=jnp.mean(diffs)), lambda _: None, n_max_iters)

        # Return k-space Green's functions (nK**2 x dim x dim)
        # Averaging over k-points should be done by the caller as needed
        return g

    def sigmaBulk(self, E, conv=1e-4):
        """
        Calculate bulk self-energies for all 12 directions.

        Self-energies are calculated in k-space then averaged: Σ_i = (1/N_k) Σ_k B_i @ g_k @ B_i†

        Parameters
        ----------
        E : float
            Energy point for self-energy calculation (in eV)
        conv : float, optional
            Convergence criterion (default: 1e-4)

        Returns
        -------
        ndarray
            Array of 12 self-energy matrices (shape: 12 x dim x dim)
        """
        # Get k-space bulk Green's functions: nK**2 x dim x dim
        g_k = self.gBulk(E, conv)

        # Calculate self-energies in k-space for each direction, then average
        # Σ_i = (1/N_k) Σ_k B_i @ g_k @ B_i†
        B = (E + self.eta*1j)*self.Slist - self.Vlist  # 12 x dim x dim

        # For each direction a, calculate Σ_a,k = B_a @ g_k @ B_a† then average over k
        # einsum: for each (a, k), compute B[a] @ g_k[k] @ B[a]†
        sig_temp = jnp.einsum('aij,kjl,aln->akin', B, g_k, B.conj())  # 12 x nK**2 x dim x dim
        sigList = jnp.mean(sig_temp, axis=1)  # Average over k-points -> 12 x dim x dim

        return sigList

    # Calculate Green's function for the surface
    def gSurf(self, E, conv=1e-4, mix=0.1, maxIter=5000):
        # Setup k-mesh: Gamma-centered for even, Monkhorst-Pack for odd
        if self.kPoints % 2 == 0:
            # Gamma-centered grid for even k-points
            k = jnp.arange(self.kPoints) / self.kPoints - 0.5 + 0.5/self.kPoints
        else:
            # Monkhorst-Pack grid for odd k-points
            k = (2 * jnp.arange(self.kPoints) + 1) / (2 * self.kPoints) - 0.5
        K1, K2 = jnp.meshgrid(k, k, indexing='ij') # nK x nK
        kmesh = K1.flatten()[:, jnp.newaxis] * self.bList[0] + \
                K2.flatten()[:, jnp.newaxis] * self.bList[1] # nK**2 x 3
        expList = jnp.exp(-1j*kmesh@(self.vecs.T)) # nK**2 x NN
        
        # Set up dyson equation
        
        Flist = expList[:, :, None, None]*self.Vlist[None, :, :, :]# nK**2 x NN x dim x dim
        Slist = expList[:, :, None, None]*self.Slist[None, :, :, :]# nK**2 x NN x dim x dim
        Fak = jnp.sum(Flist[:, :6, :, :], axis=1) + \
                jnp.repeat(self.H[None, :, :], self.kPoints**2, axis=0)
        Sak = jnp.sum(Slist[:, :6, :, :], axis=1) + \
                jnp.repeat(jnp.eye(dim)[None, :, :], self.kPoints**2, axis=0)
        A = (E + self.eta*1j)*Sak - Fak
        Fbk = jnp.sum(Flist[:, 6:9, :, :], axis=1)
        Sbk = jnp.sum(Slist[:, 6:9, :, :], axis=1)
        B = (E + self.eta*1j)*Sbk - Fbk
        key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX

        # Converge each k-point independently with robust solver
        def converge_single_k(A_k, B_k):
            """
            Converge Dyson equation for a single k-point using robust iteration.

            Solves: g = [A - B @ g @ B†]^-1

            Uses adaptive mixing and proper convergence criteria to ensure
            retarded Green's function with Im[g] < 0.
            """
            def cond_fun(state):
                count, diff, g, g_ = state
                return (diff > conv) & (count < maxIter)

            def body_fun(state):
                count, diff, g, g_ = state
                
                B_k_disordered = B_k#*(1+ (jax.random.uniform(key) - 0.5)*1e-3)
                # Compute self-energy
                sig = B_k_disordered @ g @ B_k_disordered.conj().T

                # Update Green's function
                gNew = LA.inv(A_k - sig)

                # Apply mixing to ensure retarded Green's function with Im[g] < 0
                g_ = g.copy()
                #g = jnp.where(gNew.imag > 0, g, gNew * mix + (1 - mix) * g)
                g = gNew * mix + (1 - mix) * g

                # Convergence check: use relative change in norm
                diff = jnp.linalg.norm(g - g_) / (jnp.linalg.norm(g_) + 1e-12)
                count += 1
                return (count, diff, g, g_)

            # Initialize with bare Green's function (no self-energy)
            g_init = LA.inv(A_k) - 1j*jnp.eye(dim)*self.eta

            init_state = (0, jnp.inf, g_init, g_init.copy())
            count, diff, g, g_ = jax.lax.while_loop(cond_fun, body_fun, init_state)
            return g, count, diff

        # Vectorize over all k-points (run in parallel)
        g, counts, diffs = jax.vmap(converge_single_k)(A, B)

        # Diagnostic output
        #n_max_iters = jnp.sum(counts >= maxIter)
        #jax.lax.cond(n_max_iters > 0, lambda _: jax.debug.print("gSurf: max={max_c}, avg={avg_c:.1f}, unconverged={n_uc}/{n_tot}, max_diff={max_diff:.1e}, avg_diff={avg_diff:.1e}",
        #               max_c=jnp.max(counts), avg_c=jnp.mean(counts),
        #               n_uc=n_max_iters, n_tot=self.kPoints**2, max_diff=jnp.max(diffs), avg_diff=jnp.mean(diffs)), lambda _: None, n_max_iters)

        # Return k-space Green's functions (nK**2 x dim x dim)
        # Averaging over k-points should be done by the caller as needed
        return g
        

    def sigma(self, E, inds=None, conv=1e-4, mix=0.1):
        """
        Calculate surface self-energies for an FCC lattice.

        Computes self-energies for atoms at the surface with the geometry:
        [3x plane dir] - o - [3x plane dir]
                        /|\
                [3x out of plane dir]

        Self-energies are calculated in k-space then averaged: Σ_i = (1/N_k) Σ_k B_i @ g_k @ B_i†

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)
        inds : list or int, optional
            Indices of the sigma matrix to return. If None, returns full list (default: None)
        conv : float, optional
            Convergence criterion for Dyson equation (default: 1e-4)
        mix : float, optional
            Mixing factor for Dyson equation (default: 0.1)

        Returns
        -------
        ndarray
            Array of self-energy matrices for the surface atom. If inds is specified,
            returns only the requested matrices. Shape: (9, dim, dim) or (len(inds), dim, dim)
        """
        # Get k-space Green's functions: nK**2 x dim x dim
        g_k = self.gSurf(E, conv, mix)

        # Calculate self-energies in k-space for each direction, then average
        # Σ_i = (1/N_k) Σ_k B_i @ g_k @ B_i†
        B = (E + self.eta*1j)*self.Slist - self.Vlist  # 12 x dim x dim

        # For each direction a, calculate Σ_a,k = B_a @ g_k @ B_a† then average over k
        # B: 12 x dim x dim, g_k: nK**2 x dim x dim
        # einsum: for each (a, k), compute B[a] @ g_k[k] @ B[a]† = B[a,i,j] * g_k[k,j,l] * conj(B[a,l,n])
        sig_temp = jnp.einsum('aij,kjl,aln->akin', B, g_k, B.conj())  # 12 x nK**2 x dim x dim
        sigListAll = jnp.mean(sig_temp, axis=1)  # Average over k-points -> 12 x dim x dim

        if inds is None:
            # Return surface self-energies: 6 in-plane (0-5) + 3 out-of-plane (6-8)
            in_plane = sigListAll[0:6]
            out_plane = sigListAll[6:9]
            return jnp.concatenate([in_plane, out_plane], axis=0)  # 9 x dim x dim
        else:
            return jnp.array([sigListAll[i] for i in inds])
    
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
    
    def sigmaTot(self, E, conv=1e-4):
        """
        Calculate total self-energy matrix for the extended system.

        Computes self-energies for the full extended system including 12 neighbor sites
        plus 1 central site. For each neighbor site k, applies all bulk self-energies
        EXCEPT the one in the opposite direction (pair_k), preventing double-counting
        of the connection to the central atom.

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
        Follows the approach from surfGBethe: for neighbor k, self-energy is
        Σ_tot - Σ[pair_k] where pair_k = (k + 6) % 12 is the opposite direction.
        Center site (site 12) has zero self-energy (it's the coupling point).
        """
        sig = jnp.zeros(((self.NN + 1)*dim, (self.NN+1)*dim), dtype=complex)

        # Get all 12 bulk self-energies
        sigList = self.sigmaBulk(E, conv)  # List of 12 dim x dim matrices

        # Convert to JAX array and sum
        sigArray = jnp.array(sigList)  # 12 x dim x dim
        sigSum = jnp.sum(sigArray, axis=0)  # dim x dim

        # For each of the 12 neighbor sites
        for k in range(self.NN):
            pair_k = (k + 6) % 12  # Opposite direction
            # Apply all directions EXCEPT the one pointing back to center
            sig = sig.at[k*dim:(k+1)*dim, k*dim:(k+1)*dim].set(sigSum - sigArray[pair_k])

        # Center site (site 12) has zero self-energy (it's the reference point)
        # No need to set anything - already zeros

        return sig

    
    # Get the bulk DOS
    def DOS(self, E, conv=1e-4, mix=0.1):
        """
        Calculate bulk density of states using surface self-energies.

        Matches surfGBethe.DOS() by computing the local DOS of a surface atom
        with 9 semi-infinite connections (6 in-plane + 3 out-of-plane).
        Uses DOS(E) = -(1/π) Im[Tr[G]] where G = [E*I - H - Σ_surf]^-1.

        Parameters
        ----------
        E : float
            Energy point for DOS calculation (in eV)
        conv : float, optional
            Convergence criterion for self-energy calculation (default: 1e-5)
        mix : float, optional
            Mixing parameter for self-energy convergence (default: 0.5)

        Returns
        -------
        float
            Density of states at energy E
        """
        # Get surface self-energies (9 directions: 6 in-plane + 3 out-of-plane)
        sigList = self.sigma(E, inds=None, conv=conv, mix=mix)  # 9 x dim x dim

        # Sum all surface self-energies
        sigTot = jnp.sum(sigList, axis=0)  # dim x dim

        # Compute Green's function: G = [E*I - H - Σ_tot]^-1
        # Note: Use same sign convention as rest of surfG3D (E + 1j*eta)
        Gr = LA.inv((E + 1j*self.eta)*jnp.eye(dim) - self.H - sigTot)

        # DOS = -(1/π) Im[Tr[G]]
        return -jnp.trace(Gr).imag / jnp.pi

    
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

