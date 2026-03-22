# Developed packages
from gauNEGF.density import *
from gauNEGF.config import (ETA, TEMPERATURE, ENERGY_MIN, FERMI_DEBUG,shard_array)
from gauNEGF.utils import fractional_matrix_power, eig

# Python packages
import jax
import jax.numpy as jnp
from jax.numpy import linalg as LA

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
        self.num_contacts = len(self.indsLists)
        # Use surfGBAt() object to store the atomic Bethe lattice green's function for each contact
        self.gList = []
        for Slist, Vlist, vecs in zip(self.Slists, self.Vlists, self.dirLists):
            self.gList.append(surfGAt3D(self.H0.copy(), Slist, Vlist,vecs, eta, T))
        
        # Calculate fermi level 
        fermi = self.gList[0].calcFermi(self.ne/2)
        for g in self.gList:
            g.fermi = fermi

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
        out_of_plane_angle = jnp.arccos(1/jnp.sqrt(3)) # ~54.74
        
        out_of_plane_vectors = []
        # Add 30° = pi/6 rotation to base vector before going out of plane
        rot_angle = jnp.pi/6
        K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                      [plane_normal[2], 0, -plane_normal[0]],
                      [-plane_normal[1], plane_normal[0], 0]])
        R = jnp.eye(3) + jnp.sin(rot_angle) * K + (1 - jnp.cos(rot_angle)) * jnp.matmul(K, K)
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
            [jnp.cos(theta) * jnp.cos(phi), -jnp.sin(phi)  , jnp.sin(theta)*jnp.cos(phi)],
            [jnp.cos(theta) * jnp.sin(phi),  jnp.cos(phi)  , jnp.sin(theta)*jnp.sin(phi)], 
            [-jnp.sin(theta)             ,  0            , jnp.cos(theta)]
        ]))
        
        # d orbitals (5x5) at positions [4:9,4:9]
        # [d3z2-r2, dxz, dyz, dx2-y2, dxy] block - transforms the five d orbitals
        d_block = jnp.zeros((5,5))
        
        # Copying formula from ANT.Gaussian directly
        d_block = d_block.at[0,0].set((3 * z**2 - 1) / 2)
        d_block = d_block.at[0,1].set(-jnp.sqrt(3) * jnp.sin(2*theta) / 2)
        d_block = d_block.at[0,3].set(jnp.sqrt(3) * jnp.sin(theta)**2 / 2)
        
        d_10 = jnp.sqrt(3) * jnp.sin(2*theta) * jnp.cos(phi) / 2
        d_block = d_block.at[1,0].set(d_10)
        d_block = d_block.at[1,1].set(jnp.cos(2*theta) * jnp.cos(phi))
        d_block = d_block.at[1,2].set(-jnp.cos(theta) * jnp.sin(phi))
        d_block = d_block.at[1,3].set(-d_10 / jnp.sqrt(3))
        d_block = d_block.at[1,4].set(jnp.sin(theta) * jnp.sin(phi))
        
        d_20 = jnp.sqrt(3) * jnp.sin(2*theta) * jnp.sin(phi) / 2
        d_block = d_block.at[2,0].set(d_20)
        d_block = d_block.at[2,1].set(jnp.cos(2*theta) * jnp.sin(phi))
        d_block = d_block.at[2,2].set(jnp.cos(theta) * jnp.cos(phi))
        d_block = d_block.at[2,3].set(-d_20 / jnp.sqrt(3))
        d_block = d_block.at[2,4].set(-jnp.sin(theta) * jnp.cos(phi))
        
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
        G_AB = self.gList[i].gSurf(E, conv)
        # Apply self energies in first 9 directions that aren't attached to atom
        for nInds, Finds in zip(self.nIndLists[i], self.indsLists[i]):
            sigInds = list(set(range(9)) - {int(x) for x in nInds})
            sigAtom = self.gList[i].sigma(E, active_dirs=sigInds, G_AB=G_AB)
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

    def crossTermQ(self, E, i, conv=1e-4):
        """Cross-term Q_sym for contact i in full device basis.

        Mirrors surfG3.sigma: computes G_AB via gSurf, then for each atom
        in contact i assembles Q using active directions, applies Xi.
        """
        sig = jnp.zeros((self.N, self.N), dtype=complex)
        G_AB = self.gList[i].gSurf(E, conv)
        for nInds, Finds in zip(self.nIndLists[i], self.indsLists[i]):
            sigInds = list(set(range(9)) - {int(x) for x in nInds})
            Q_atom = self.gList[i].crossTermQ(E, active_dirs=sigInds, G_AB=G_AB)
            sig = sig.at[jnp.ix_(Finds, Finds)].set(Q_atom)

        if self.Sdict['sss'] == 0:
            sig = times(self.Xi, sig, self.Xi)
        if self.spin == 'u' or self.spin == 'ro':
            sig = jnp.kron(jnp.eye(2), sig)
        elif self.spin == 'g':
            sig = jnp.kron(sig, jnp.eye(2))
        return sig

    def crossTermQTot(self, E, conv=1e-4):
        """Total cross-term Q_sym from all contacts."""
        qs = [self.crossTermQ(E, i, conv) for i in range(len(self.indsLists))]
        return sum(qs)

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
        self.H0 = jnp.array(H)          # immutable reference (never mutated)
        self.Slist = jnp.array(Slist)
        self.Vlist = jnp.array(Vlist)
        self.Vlist0 = jnp.array(Vlist)  # immutable reference (never mutated)
        self.NN = len(Slist)
        self.kPoints = kPoints
        assert self.NN == 12, "Error: surfGAt3D only implemented for FCC using 12 NN"
        assert len(vecs) == 12, "Error: surfGAt3D only implemented for FCC using 12 NN"

        self.vecs = jnp.array(vecs)

        # Define 3D lattice vectors for bulk periodicity
        # For FCC [111]: vecs[0,1,2] are in-plane (z approx 0), vecs[3,4,5] are out-of-plane (+z)
        self.a1 = self.vecs[0]  # First in-plane vector
        self.a2 = self.vecs[1]  # Second in-plane vector
        self.a3 = self.vecs[3]  # First out-of-plane vector (upward)

        # Reciprocal lattice vectors computed separately for 2D (surface) and 3D (bulk)
        # See _setup_kmesh_2D() and _setup_kmesh_3D()

        #self.Slist = [jnp.zeros((dim,dim)) for n in range(self.NN)] #To match ANT.Gaussian default
        self.eta = eta
        self.T = T
        self.sigmaKprev = None
        self.Eprev = Eminf
        self.fermi = None
        self.fermi0 = None    # reference Fermi level
        self.dFermi = 0.0    # shift from reference

        # Pre-compute k-mesh for surface (2D) and bulk (3D)
        self._setup_kmesh_2D()
        self._setup_kmesh_3D()

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
        if fermi is not None and fermi != self.fermi:
            self.fermi = fermi
            self.dFermi = fermi if self.fermi0 is None else fermi - self.fermi0
            if self.fermi0 is None:
                self.fermi0 = fermi

        # Build H and Vlist from H0/Vlist0 plus current dFermi (for non-JIT callers)
        self.H = self.H0 + self.dFermi * jnp.eye(dim)
        self.Vlist = jnp.array([self.Vlist0[j] + self.dFermi * self.Slist[j]
                                 for j in range(self.NN)])

        H0x = jnp.kron(jnp.eye(self.NN+1), self.H)
        S0x = jnp.eye(dim*(self.NN+1))
        for i in range(self.NN):
            S0x = S0x.at[-dim:, i*dim:(i+1)*dim].set(self.Slist[i])
            S0x = S0x.at[i*dim:(i+1)*dim, -dim:].set(self.Slist[i].T)
            H0x = H0x.at[-dim:, i*dim:(i+1)*dim].set(self.Vlist[i])
            H0x = H0x.at[i*dim:(i+1)*dim, -dim:].set(self.Vlist[i].conj().T)
        self.F = H0x
        self.S = S0x

    def _setup_kmesh_2D(self):
        """
        Pre-compute 2D k-mesh and phase factors for surface Green's function.

        Computes 2D reciprocal lattice vectors using surface normal approach:
        - Surface normal n = a1 × a2 (perpendicular to surface)
        - 2D reciprocal vectors lie in surface plane, perpendicular to n
        - This ensures k_z = 0 for all k-points (no periodicity perpendicular to surface)

        Sets up:
        - self.b1_2D, self.b2_2D: 2D reciprocal lattice vectors (z-component = 0)
        - self.kmesh_2D: 2D Cartesian k-points in reciprocal space (nK^2 x 3)
        - self.expList_2D: Phase factors exp(+ik·r) for all vectors (nK^2 x 12)
        """
        # Compute surface normal (perpendicular to a1 and a2)
        surface_normal = jnp.cross(self.a1, self.a2)
        n_hat = surface_normal / jnp.linalg.norm(surface_normal)

        # Compute 2D reciprocal vectors using surface normal
        # These lie in the plane perpendicular to n_hat
        # Formula: b1 = 2π * (a2 × n) / [a1 · (a2 × n)]
        #          b2 = 2π * (n × a1) / [a2 · (n × a1)]
        cross_a2_n = jnp.cross(self.a2, n_hat)
        cross_n_a1 = jnp.cross(n_hat, self.a1)

        denom1 = jnp.dot(self.a1, cross_a2_n)
        denom2 = jnp.dot(self.a2, cross_n_a1)

        self.b1_2D = 2 * jnp.pi * cross_a2_n / denom1
        self.b2_2D = 2 * jnp.pi * cross_n_a1 / denom2

        # Verify orthogonality: n · b1 = 0, n · b2 = 0 (perpendicular to surface normal)
        # This ensures k-points lie in surface plane

        # Setup k-mesh: Gamma-centered for even, Monkhorst-Pack for odd
        if self.kPoints % 2 == 0:
            k = jnp.arange(self.kPoints) / self.kPoints - 0.5 + 0.5/self.kPoints
        else:
            k = (2 * jnp.arange(self.kPoints) + 1) / (2 * self.kPoints) - 0.5

        K1, K2 = jnp.meshgrid(k, k, indexing='ij')
        self.kmesh_2D = K1.flatten()[:, jnp.newaxis] * self.b1_2D + \
                        K2.flatten()[:, jnp.newaxis] * self.b2_2D  # nK^2 x 3

        # Phase factors: exp(+ik·r) for forward Fourier transform
        self.expList_2D = jnp.exp(+1j * self.kmesh_2D @ (self.vecs.T))  # nK^2 x 12

    def _setup_kmesh_3D(self):
        """
        Pre-compute 3D k-mesh and phase factors for bulk Green's function.

        Computes 3D reciprocal lattice vectors from a1, a2, a3:
        - Full 3D periodicity (all 12 neighbors included)
        - Reciprocal vectors may have non-zero components in all directions

        Sets up:
        - self.b1_3D, self.b2_3D, self.b3_3D: 3D reciprocal lattice vectors
        - self.kmesh_3D: 3D Cartesian k-points in reciprocal space (nK^3 x 3)
        - self.expList_3D: Phase factors exp(+ik·r) for all vectors (nK^3 x 12)
        """
        # Compute 3D reciprocal lattice vectors
        # Formula: b_i = 2π * (a_j × a_k) / [a_i · (a_j × a_k)]
        vol = jnp.dot(self.a1, jnp.cross(self.a2, self.a3))
        self.b1_3D = 2 * jnp.pi * jnp.cross(self.a2, self.a3) / vol
        self.b2_3D = 2 * jnp.pi * jnp.cross(self.a3, self.a1) / vol
        self.b3_3D = 2 * jnp.pi * jnp.cross(self.a1, self.a2) / vol

        # Setup k-mesh: Gamma-centered for even, Monkhorst-Pack for odd
        if self.kPoints % 2 == 0:
            k = jnp.arange(self.kPoints) / self.kPoints - 0.5 + 0.5/self.kPoints
        else:
            k = (2 * jnp.arange(self.kPoints) + 1) / (2 * self.kPoints) - 0.5

        K1, K2, K3 = jnp.meshgrid(k, k, k, indexing='ij')
        # Flatten and stack into nK^3 x 3 array
        kmesh_flat = jnp.stack([K1.flatten(), K2.flatten(), K3.flatten()], axis=1)  # nK^3 x 3

        # Convert fractional coordinates to Cartesian using 3D reciprocal lattice
        self.kmesh_3D = (kmesh_flat[:, 0:1] * self.b1_3D +
                         kmesh_flat[:, 1:2] * self.b2_3D +
                         kmesh_flat[:, 2:3] * self.b3_3D)  # nK^3 x 3

        # Phase factors: exp(+ik·r) for forward Fourier transform
        self.expList_3D = jnp.exp(+1j * self.kmesh_3D @ (self.vecs.T))  # nK^3 x 12

    def gBulk(self, E):
        """
        Calculate bulk Green's function with full 3D periodicity via direct inversion.

        For bulk with all neighbors included, no Dyson equation needed:
        g(k) = [(E + iη)S(k) - H(k)]^-1

        Uses 3D k-space mesh with full periodicity in all directions.

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)

        Returns
        -------
        ndarray
            Real-space G_AB propagator matrix (12*dim x 12*dim).
            Block (A,B) at G_AB[A*dim:(A+1)*dim, B*dim:(B+1)*dim] gives
            the propagator G(R_A - R_B) between directions A and B.
        """
        # Construct H(k) = sum_R V(R)*exp(+ik*R) and S(k) = sum_R S(R)*exp(+ik*R)
        # Use 3D k-mesh
        Flist = self.expList_3D[:, :, None, None] * self.Vlist0[None, :, :, :]  # nK^3 x 12 x dim x dim
        Slist = self.expList_3D[:, :, None, None] * self.Slist[None, :, :, :]  # nK^3 x 12 x dim x dim

        # Sum over ALL 12 neighbors (in-plane + out-of-plane) + onsite
        Hk = jnp.sum(Flist, axis=1) + jnp.repeat(self.H0[None, :, :], self.kPoints**3, axis=0)
        Sk = jnp.sum(Slist, axis=1) + jnp.repeat(jnp.eye(dim)[None, :, :], self.kPoints**3, axis=0)

        # Shard k-point data across devices for parallel computation
        Hk_sharded = shard_array(Hk, axis=0)
        Sk_sharded = shard_array(Sk, axis=0)

        # Direct inversion: g(k) = [(E + i*eta)S(k) - H(k)]^-1
        # Vectorized over all k-points, automatically parallelized across devices
        E_eff = E - self.dFermi
        g_k = jax.vmap(lambda H, S: LA.inv((E_eff + self.eta*1j)*S - H))(Hk_sharded, Sk_sharded)

        # Inverse FT: build 108x108 real-space propagator G_AB
        # G_AB[A,B] = (1/Nk) sum_k exp(+ik*R_A) * g_k * exp(-ik*R_B)
        phases = self.expList_3D  # nK^3 x 12
        nK3 = self.kPoints**3
        G_AB_blocks = jnp.einsum('ka,kij,kb->abij', phases, g_k, phases.conj()) / nK3
        G_AB = G_AB_blocks.transpose(0, 2, 1, 3).reshape(12 * dim, 12 * dim)
        return G_AB

    def sigmaBulk(self, E):
        """
        Calculate bulk self-energies for all 12 directions using G_AB propagator.

        For each direction k, computes the self-energy from 11 directions
        (all except the reverse direction pair_k = (k+6)%12):
            Sigma_k = tau_k @ G_sub_k @ tau_k'
        where tau_k is the horizontal concat of phase-free coupling matrices
        and G_sub_k is the sub-block of G_AB for those 11 directions.

        Parameters
        ----------
        E : float
            Energy point for self-energy calculation (in eV)

        Returns
        -------
        ndarray
            Array of 12 self-energy matrices (shape: 12 x dim x dim)
        """
        G_AB = self.gBulk(E)  # 108 x 108

        sigList = []
        for k in range(12):
            pair_k = (k + 6) % 12  # Reverse direction
            active_dirs = [d for d in range(12) if d != pair_k]  # 11 dirs

            # Build tau: horizontal concat of phase-free couplings
            # active_dirs must be a static Python list (not a traced JAX value)
            E_eff = E - self.dFermi
            z = E_eff + self.eta*1j
            tau_blocks = [z * self.Slist[a] - self.Vlist0[a] for a in active_dirs]
            tau = jnp.concatenate(tau_blocks, axis=1)  # dim x (11*dim)
            bar_tau_blocks = [z * self.Slist[a].conj().T - self.Vlist0[a].conj().T for a in active_dirs]
            bar_tau = jnp.concatenate(bar_tau_blocks, axis=0)  # (11*dim) x dim

            # Extract sub-block of G_AB for active directions
            row_idx = jnp.concatenate([jnp.arange(a*dim, (a+1)*dim) for a in active_dirs])
            G_sub = G_AB[jnp.ix_(row_idx, row_idx)]  # (11*dim) x (11*dim)

            sigList.append(tau @ G_sub @ bar_tau)

        return jnp.array(sigList)  # 12 x dim x dim

    # Calculate Green's function for the surface
    def gSurf(self, E, conv=1e-4, mix=0.1, maxIter=5000):
        # Set up dyson equation using pre-computed 2D phase factors
        # Use 2D k-mesh for surface
        Flist = self.expList_2D[:, :, None, None]*self.Vlist0[None, :, :, :]# nK**2 x NN x dim x dim
        Slist = self.expList_2D[:, :, None, None]*self.Slist[None, :, :, :]# nK**2 x NN x dim x dim

        E_eff = E - self.dFermi
        # A matrix: in-plane neighbors only (vecs 0,1,2,6,7,8)
        # These are the 6 in-plane directions with z=0
        in_plane_indices = jnp.array([0, 1, 2, 6, 7, 8])
        Fak = jnp.sum(Flist[:, in_plane_indices, :, :], axis=1) + \
                jnp.repeat(self.H0[None, :, :], self.kPoints**2, axis=0)
        Sak = jnp.sum(Slist[:, in_plane_indices, :, :], axis=1) + \
                jnp.repeat(jnp.eye(dim)[None, :, :], self.kPoints**2, axis=0)
        A = (E_eff + self.eta*1j)*Sak - Fak

        # B matrix: out-of-plane neighbors pointing UP (vecs 3,4,5)
        # These connect surface to bulk above
        out_plane_indices = jnp.array([3, 4, 5])
        Fbk = jnp.sum(Flist[:, out_plane_indices, :, :], axis=1)
        Sbk = jnp.sum(Slist[:, out_plane_indices, :, :], axis=1)
        z = E_eff + self.eta*1j
        B = z*Sbk - Fbk
        B_bar = z*jnp.conj(Sbk).transpose(0,2,1) - jnp.conj(Fbk).transpose(0,2,1)

        # Converge each k-point independently with robust solver
        def converge_single_k(A_k, B_k, B_bar_k):
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
                sig = B_k @ g @ B_bar_k

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

        # Shard k-point data across devices for parallel computation
        A_sharded = shard_array(A, axis=0)
        B_sharded = shard_array(B, axis=0)
        B_bar_sharded = shard_array(B_bar, axis=0)

        # Vectorize over all k-points, automatically parallelized across devices
        g_k, counts, diffs = jax.vmap(converge_single_k)(A_sharded, B_sharded, B_bar_sharded)

        # Inverse Fourier transform: build 81x81 real-space propagator G_AB
        # G_AB[A*dim:(A+1)*dim, B*dim:(B+1)*dim] = (1/Nk) sum_k exp(+ik*R_A) * g_k * exp(-ik*R_B)
        # Only surface directions [0..8]
        surf_dirs = jnp.arange(9)
        phases = self.expList_2D[:, surf_dirs]  # nK^2 x 9
        nK2 = self.kPoints**2
        G_AB_blocks = jnp.einsum('ka,kij,kb->abij', phases, g_k, phases.conj()) / nK2
        # Reshape from (9, 9, dim, dim) block form to (81, 81) matrix
        G_AB = G_AB_blocks.transpose(0, 2, 1, 3).reshape(9 * dim, 9 * dim)
        return G_AB

    def sigma(self, E, active_dirs=None, conv=1e-4, mix=0.1, G_AB=None):
        """
        Calculate surface self-energy using G_AB propagator.

        Computes Sigma = tau @ G_S @ tau' where:
        - G_S is the sub-block of G_AB for active directions
        - tau = [tau_0 | tau_1 | ...] is the horizontal concatenation
          of phase-free coupling matrices tau_a = (E+i*eta)*S_a - V_a

        Parameters
        ----------
        E : float
            Energy point for Green's function calculation (in eV)
        active_dirs : list of int, optional
            Direction indices to include. Default None means all 9
            surface directions [0..8].
        conv : float, optional
            Convergence criterion for Dyson equation (default: 1e-4)
        mix : float, optional
            Mixing factor for Dyson equation (default: 0.1)
        G_AB : ndarray, optional
            Pre-computed G_AB propagator from gSurf(). If None,
            gSurf() is called internally.

        Returns
        -------
        ndarray
            Self-energy matrix of shape (dim, dim).
        """
        if active_dirs is None:
            active_dirs = list(range(9))

        if G_AB is None:
            G_AB = self.gSurf(E, conv, mix)  # 81 x 81

        # Build tau: horizontal concat of phase-free coupling matrices
        # active_dirs must be a static Python list (not a traced JAX value)
        E_eff = E - self.dFermi
        z = E_eff + self.eta*1j
        tau_blocks = [z * self.Slist[a] - self.Vlist0[a] for a in active_dirs]
        tau = jnp.concatenate(tau_blocks, axis=1)  # dim x (nDirs*dim)
        bar_tau_blocks = [z * self.Slist[a].conj().T - self.Vlist0[a].conj().T for a in active_dirs]
        bar_tau = jnp.concatenate(bar_tau_blocks, axis=0)  # (nDirs*dim) x dim

        # Extract sub-block of G_AB for active directions
        # G_AB is 81x81 with block structure [A*dim:(A+1)*dim, B*dim:(B+1)*dim]
        row_idx = jnp.concatenate([jnp.arange(a*dim, (a+1)*dim) for a in active_dirs])
        col_idx = row_idx  # square sub-block
        G_sub = G_AB[jnp.ix_(row_idx, col_idx)]  # (nDirs*dim) x (nDirs*dim)

        # Self-energy: tau @ G_sub @ bar_tau
        return tau @ G_sub @ bar_tau  # dim x dim

    def crossTermQ(self, E, active_dirs=None, conv=1e-4, mix=0.1, G_AB=None):
        """Symmetrized cross-term Q_sym using G_AB propagator.

        Q_sym = (tau @ G_sub @ S_LD + S_DL @ G_sub @ tau^dagger) / 2

        where tau is the same phase-free coupling used in sigma().
        """
        if active_dirs is None:
            active_dirs = list(range(9))

        if G_AB is None:
            G_AB = self.gSurf(E, conv, mix)

        E_eff = E - self.dFermi

        tau_blocks = [(E_eff + self.eta * 1j) * self.Slist[a] - self.Vlist0[a]
                      for a in active_dirs]
        tau = jnp.concatenate(tau_blocks, axis=1)  # (dim, nDirs*dim)

        # S_LD: stack of S[a]^H for each active direction, shape (nDirs*dim, dim)
        S_LD = jnp.concatenate([self.Slist[a].conj().T for a in active_dirs], axis=0)
        # S_DL: row stack of S[a] for each active direction, shape (dim, nDirs*dim)
        S_DL = jnp.concatenate([self.Slist[a] for a in active_dirs], axis=1)
        # bar_tau: right-side coupling, uses z (not z*) with S^H, V^H
        bar_tau = jnp.concatenate([(E_eff + self.eta * 1j) * self.Slist[a].conj().T
                                    - self.Vlist0[a].conj().T
                                    for a in active_dirs], axis=0)

        row_idx = jnp.concatenate([jnp.arange(a * dim, (a + 1) * dim) for a in active_dirs])
        G_sub = G_AB[jnp.ix_(row_idx, row_idx)]

        Q_fwd = tau @ G_sub @ S_LD       # (dim, dim)
        Q_rev = S_DL @ G_sub @ bar_tau   # (dim, dim)
        return (Q_fwd + Q_rev) / 2

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
    
    def sigmaTot(self, E):
        """
        Calculate total self-energy matrix for the extended system.

        Computes self-energies for the full extended system including 12 neighbor sites
        plus 1 central site. For each neighbor site k, applies the bulk self-energy
        from 11 directions (excluding the direction back to center).

        Used for calculating Fermi energy of the bulk system.

        Parameters
        ----------
        E : float
            Energy point for self-energy calculation (in eV)

        Returns
        -------
        ndarray
            Total self-energy matrix for the extended system ((NN+1)*dim, (NN+1)*dim)
        """
        sig = jnp.zeros(((self.NN + 1)*dim, (self.NN+1)*dim), dtype=complex)

        # sigmaBulk returns 12 self-energies, each from 11 dirs (excluding reverse)
        sigList = self.sigmaBulk(E)  # 12 x dim x dim

        # Place each on the diagonal of the extended system
        for k in range(self.NN):
            sig = sig.at[k*dim:(k+1)*dim, k*dim:(k+1)*dim].set(sigList[k])

        # Center site (site 12) has zero self-energy
        return sig

    def DOS(self, E, conv=1e-4, mix=0.1):
        """
        Use surface Green's function to calculate density of states.

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
        sig = self.sigma(E, conv=conv, mix=mix)  # dim x dim
        Gr = LA.inv((E - self.dFermi + self.eta*1j)*jnp.eye(dim) - self.H0 - sig)
        return -jnp.trace(Gr).imag / jnp.pi

    def generate_band_plot(self, E_fermi=0.0, n_points=40, plot=True, save_path=None):
        """
        Generate band structure plot along high-symmetry path for FCC.

        Uses the pre-computed reciprocal lattice vectors (b1_3D, b2_3D, b3_3D)
        to convert fractional k-coordinates to Cartesian k-space.

        Parameters
        ----------
        E_fermi : float, optional
            Fermi energy to shift bands relative to (default: 0.0)
        n_points : int, optional
            Number of points between each high-symmetry point (default: 40)
        plot : bool, optional
            Whether to generate matplotlib plot (default: True)
        save_path : str, optional
            Path to save plot (default: None, display only)

        Returns
        -------
        dict
            Dictionary containing:
            - 'distances': 1D array of k-path distances
            - 'bands': 2D array (n_kpoints x n_bands) of eigenvalues
            - 'k_labels': List of high-symmetry point labels
            - 'k_positions': Positions of high-symmetry points along path
        """
        # Compute correct FCC high-symmetry k-points for the [111]-frame
        # rhombohedral primitive cell (a1=vecs[0], a2=vecs[1], a3=vecs[3])
        B_inv = jnp.linalg.inv(jnp.column_stack([self.b1_3D, self.b2_3D, self.b3_3D]))
        ex = jnp.array([1, -1, 0]) / jnp.sqrt(2)
        ey = jnp.array([1, 1, -2]) / jnp.sqrt(6)
        ez = jnp.array([1, 1, 1]) / jnp.sqrt(3)
        R = jnp.array([ex, ey, ez])
        scale = 2 * jnp.pi / jnp.sqrt(2)  # nearest-neighbor distance = 1 -> a = sqrt(2)
        def _kpt(cubic_vec):
            return B_inv @ (R @ (scale * jnp.array(cubic_vec)))
        k_points_special = {
            'G': jnp.zeros(3),
            'X': _kpt([1.0, 0.0, 0.0]),
            'W': _kpt([1.0, 0.5, 0.0]),
            'L': _kpt([0.5, 0.5, 0.5]),
            'K': _kpt([0.75, 0.75, 0.0]),
        }

        path = ['G', 'X', 'W', 'L', 'G', 'K']

        # Generate k-path
        k_path_frac = []
        k_labels = []
        k_positions = []
        distance = 0.0

        for i in range(len(path) - 1):
            k_start = k_points_special[path[i]]
            k_end = k_points_special[path[i+1]]

            if i > 0:
                k_labels.append('')
                k_positions.append(distance)

            k_labels.append(path[i])
            k_positions.append(distance)

            for j in range(n_points):
                t = j / (n_points - 1)
                k_frac = k_start + t * (k_end - k_start)
                k_path_frac.append(k_frac)

                if j > 0 and len(k_path_frac) > 1:
                    # Convert to Cartesian k-space for distance calculation
                    k_cart_prev = (k_path_frac[-2][0] * self.b1_3D +
                                   k_path_frac[-2][1] * self.b2_3D +
                                   k_path_frac[-2][2] * self.b3_3D)
                    k_cart = (k_frac[0] * self.b1_3D +
                              k_frac[1] * self.b2_3D +
                              k_frac[2] * self.b3_3D)
                    dk = jnp.linalg.norm(k_cart - k_cart_prev)
                    distance += dk

        k_labels.append(path[-1])
        k_positions.append(distance)

        # Calculate band structure
        bands = []
        distances = jnp.zeros(len(k_path_frac))

        for idx, k_frac in enumerate(k_path_frac):
            # Convert fractional to Cartesian k-space
            k_cart = (k_frac[0] * self.b1_3D +
                      k_frac[1] * self.b2_3D +
                      k_frac[2] * self.b3_3D)

            # Build H(k) and S(k) using Bloch sum
            H_k = jnp.array(self.H, dtype=complex)
            S_k = jnp.eye(dim, dtype=complex)

            for i, vec in enumerate(self.vecs):
                phase = jnp.exp(1j * jnp.dot(k_cart, vec))
                H_k += phase * self.Vlist[i]
                S_k += phase * self.Slist[i]

            # Solve generalized eigenvalue problem: H|psi> = E S|psi>
            # First transform to standard eigenvalue problem: S^(-1/2) H S^(-1/2) |phi> = E |phi>
            S_inv = LA.inv(S_k)
            H_transformed = S_inv @ H_k
            evals, _ = LA.eigh(H_transformed)
            bands.append(jnp.sort(jnp.real(evals)) - E_fermi)

            # Calculate cumulative distance
            if idx > 0:
                k_cart_prev = (k_path_frac[idx-1][0] * self.b1_3D +
                               k_path_frac[idx-1][1] * self.b2_3D +
                               k_path_frac[idx-1][2] * self.b3_3D)
                dk = jnp.linalg.norm(k_cart - k_cart_prev)
                distances = distances.at[idx].set(distances[idx-1] + dk)

        bands = jnp.array(bands)  # Shape: (n_kpoints, n_bands)

        # Generate plot if requested
        if plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot all bands
                for i in range(dim):
                    ax.plot(distances, bands[:, i], 'b-', linewidth=1.5)

                # Fermi level
                ax.axhline(0, color='r', linestyle='--', linewidth=2, label='E_F')

                # High-symmetry point labels
                ax.set_xticks(k_positions)
                ax.set_xticklabels(k_labels, fontsize=12)

                # Vertical lines at high-symmetry points
                for pos in k_positions:
                    ax.axvline(pos, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

                ax.set_ylabel('Energy - E$_F$ (eV)', fontsize=12)
                ax.set_title('Band Structure (surfGAt3D)', fontsize=14, fontweight='bold')
                ax.set_ylim([-8, 5])
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(fontsize=10)

                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Band structure plot saved to: {save_path}")
                else:
                    plt.show()

            except ImportError:
                print("Warning: matplotlib not available, skipping plot generation")

        return {
            'distances': distances,
            'bands': bands,
            'k_labels': k_labels,
            'k_positions': k_positions,
        }

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
        print('Calculating Bulk Lattice Fermi Energy...')
        self.fermi = getFermiContact(self, ne, conv=tol, maxcycles=1000, T=self.T, nOrbs=dim)
        return self.fermi

