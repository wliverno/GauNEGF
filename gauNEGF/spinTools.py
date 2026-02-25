"""Spin manipulation tools for non-collinear NEGF calculations.

This module provides utilities for handling spin degrees of freedom in
non-collinear (generalized) spin-polarized transport calculations. It
includes:
    - Pauli matrix definitions as module-level constants
    - SU(2) rotation matrix generation via matrix exponential
    - Orthogonal spin direction generation for spin-torque analysis
    - Hirshfeld charge and magnetic moment extraction from Gaussian logs

Rotation matrices are constructed using the standard spin-1/2
representation: R(n, omega) = exp(-i * omega/2 * (n . sigma)), where n is
the rotation axis unit vector and sigma are the Pauli matrices.
"""

import numpy as np
import numpy.linalg as LA
import re
from scipy.linalg import expm

# CONSTANTS: Pauli matrices and 2x2 identity (spin-1/2 basis)
sig0 = np.eye(2, dtype=complex)
sigx = np.array([[0, 1],  [1, 0]],   dtype=complex)
sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigz = np.array([[1, 0],  [0, -1]],  dtype=complex)


def get_hirshfeld_data(filename):
    """
    Extract Hirshfeld charges and magnetic moments from a Gaussian log file.

    Parses the last occurrence of the Hirshfeld partition block in the log
    file, returning per-atom net charges and 3D magnetic moment vectors.

    Parameters
    ----------
    filename : str
        Path to the Gaussian output (.log) file.

    Returns
    -------
    charges : ndarray of shape (N,)
        Hirshfeld net charges for each atom.
    mag_moments : ndarray of shape (N, 3)
        Hirshfeld magnetic moment vectors (Mx, My, Mz) for each atom.
    None
        Returned if no Hirshfeld section is found in the file.
    """
    with open(filename, 'r') as f:
        content = f.read()

    # Find last occurrence of Hirshfeld section
    matches = list(re.finditer(r"Atomic charges and magnetic moments \(Hirshfeld partition\):", content))
    if not matches:
        return None

    # Parse atom lines after the last match
    start = matches[-1].end()
    lines = content[start:].split('\n')

    charges, mag_moments = [], []
    pattern = r'IAtom=\s*\d+\s+N=\s*([-+]?\d*\.?\d+)\s+M=\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            charges.append(float(match.group(1)))
            mag_moments.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
        elif line.strip().startswith('Total:'):
            break

    return np.array(charges), np.array(mag_moments)

def spinorTF(rho_initial, rho_target):
        """
        Find unitary transformation U such that U @ rho_initial @ U.conj().T = rho_target using Bloch sphere rotation.

        Parameters
        ----------
        rho_initial : ndarray
            Initial density matrix
        rho_target : ndarray
            Target density matrix

        Returns
        -------
        ndarray
            Unitary transformation matrix U
        """
        # Extract Bloch vectors
        def get_bloch_vector(rho):
            x = np.trace(rho @ sigx).real
            y = np.trace(rho @ sigy).real
            z = np.trace(rho @ sigz).real
            return np.array([x, y, z])
        
        r_init = get_bloch_vector(rho_initial)
        r_targ = get_bloch_vector(rho_target)
        
        # Normalize to unit vectors
        r_init_norm = LA.norm(r_init)
        r_targ_norm = LA.norm(r_targ)
        
        if r_init_norm < 1e-12 or r_targ_norm < 1e-12:
            return np.eye(2, dtype=complex)  # One is maximally mixed
        
        n_init = r_init / r_init_norm
        n_targ = r_targ / r_targ_norm
        
        # Find rotation axis and angle
        cross = np.cross(n_init, n_targ)
        dot = np.dot(n_init, n_targ)
        
        if LA.norm(cross) < 1e-12:
            return np.eye(2, dtype=complex)  # Already aligned
        
        axis = cross / LA.norm(cross)
        angle = np.arccos(np.clip(dot, -1, 1))
        
        # U = exp(-i * angle/2 * axis · sigma) = exp (A)
        # Construct A = -i * angle/2 * axis · sigma
        A = -1j * angle/2 * (axis[0]*sigx + axis[1]*sigy + axis[2]*sigz)
        
        # Eigendecomposition: A = V @ D @ V^(-1)
        # For anti-Hermitian A, eigenvalues are purely imaginary
        D, V = LA.eig(A)
        
        # exp(A) = V @ diag(exp(D)) @ V^(-1)
        return V @ np.diag(np.exp(D)) @ LA.inv(V)


def genRot(n, omega):
    """
    Generate an SU(2) rotation matrix for spin-1/2.

    Constructs R = exp(-i * omega/2 * (n . sigma)) using the module-level
    Pauli matrices. This represents a rotation by angle omega about axis n
    in spin space.

    Parameters
    ----------
    n : array_like of shape (3,)
        Unit vector defining the rotation axis in (x, y, z) coordinates.
    omega : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of shape (2, 2)
        Complex-valued SU(2) rotation matrix.
    """
    return expm(-0.5j * omega * (n[0]*sigx + n[1]*sigy + n[2]*sigz))

def genRotsGrid(spinVec, dphi=np.pi/4):
    """
    Generate a grid of spin rotations around a given axis. Starts at
    0 rads and goes to 2*pi-dphi rads in steps of dphi.

    Parameters
    ----------
    spinVec : array_like of shape (3,)
        Unit vector defining the rotation axis in (x, y, z) coordinates.
    dphi : float, optional
        Step size in radians. Default is np.pi/4.

    Returns
    -------
    rotations : list of ndarray, each of shape (2, 2)
        SU(2) rotation matrices.
    angles : ndarray of shape (n,)
        Angles in radians corresponding to each rotation.
    """
    angles = np.arange(0, 2*np.pi, dphi)
    rotations = [genRot(spinVec, angle) for angle in angles]
    return rotations, angles

def genOrthRots(spinVec):
    """
    Generate a set of orthogonal spin rotations for spin-torque analysis.

    Given an input spin quantization axis, constructs 7 rotation matrices
    mapping to: the original axis (identity), two pairs of orthogonal axes
    (+/-vec1, +/-vec2), and two independent 180 degree rotations to the 
    anti-parallel directions (-original via vec1, -original via vec2). The 
    orthogonal basis is built to avoid singularities when the input is aligned 
    with any coordinate axis.

    Notes
    -----
    Indices 5 and 6 both target -original but rotate around different axes
    (vec1 and vec2 respectively). 
    """
    unitVec = np.array(spinVec, dtype=float)
    unitVec = unitVec / LA.norm(unitVec)

    # Robust orthogonal vector construction -- avoids degeneracy when
    # unitVec is aligned with the y-axis
    if abs(unitVec[0]) > 0 or abs(unitVec[2]) > 0:
        vec1 = np.array([unitVec[2], 0, -unitVec[0]])
    else:
        vec1 = np.array([0, unitVec[2], -unitVec[1]])

    vec1 = vec1 / LA.norm(vec1)
    vec2 = np.cross(unitVec, vec1)

    # Verify orthogonality
    assert abs(np.dot(unitVec, vec1)) < 1e-10
    assert abs(np.dot(unitVec, vec2)) < 1e-10
    assert abs(np.dot(vec1, vec2)) < 1e-10

    # pi/2 rotations to orthogonal directions; pi rotation to anti-parallel
    rot_to_vec1      = genRot(vec2,  np.pi/2)   # original -> +vec1
    rot_to_neg_vec1  = genRot(vec2, -np.pi/2)   # original -> -vec1
    rot_to_vec2      = genRot(vec1, -np.pi/2)   # original -> +vec2
    rot_to_neg_vec2  = genRot(vec1,  np.pi/2)   # original -> -vec2
    rot_to_neg_orig  = genRot(vec1,  np.pi)     # original -> -original (via vec1)
    rot_to_neg_orig2 = genRot(vec2,  np.pi)     # original -> -original (via vec2)

    rotations  = [sig0, rot_to_vec1, rot_to_neg_vec1,
                  rot_to_vec2, rot_to_neg_vec2,
                  rot_to_neg_orig, rot_to_neg_orig2]
    directions = np.array([unitVec, vec1, -vec1, vec2, -vec2, -unitVec, -unitVec])

    return rotations, directions

def genOrthRotFile(filename):
    """
    Use genOrthRots() to generate a set of orthogonal spin rotations based on the
    direction of maximum magnetic moment in a Gaussian output file.

    Parameters
    ----------
    filename : str
        Path to the Gaussian output (.log) file.

    Returns
    -------
    rotations : list of ndarray, each of shape (2, 2)
        7 SU(2) rotation matrices in order:
        [identity, +vec1, -vec1, +vec2, -vec2, -original (via vec1),
        -original (via vec2)].
    directions : ndarray of shape (7, 3)
        Unit vectors corresponding to each rotation target.
    max_moment_index : int
        Index of the atom with the maximum magnetic moment.
    """
    hirsh_res = get_hirshfeld_data(filename)
    if hirsh_res is None:
        raise ValueError(f"No Hirshfeld data found in file: {filename}")
    _, mag_moments = hirsh_res
    max_moment_index = np.argmax(LA.norm(mag_moments, axis=1))
    max_moment_direction = mag_moments[max_moment_index]
    rotations, directions = genOrthRots(max_moment_direction)
    return rotations, directions, max_moment_index

def genOrthRotGrids(spinVec=None, filename=None, dphi=np.pi/4):
    """
    Generate a grid of orthogonal spin rotations using genOrthRots() from the 
    spinVec or filename, depending on which is provided. A list of 2 grids are
    returned, one for each of the orthogonal directions (+vec1, +vec2). 

    Parameters
    ----------
    spinVec : array_like, shape (3,), optional
        Input spin quantization axis as a unit vector.
    filename : str, optional
        If given, the direction is determined from the file's largest magnetization.
    dphi : float, optional
        Step size in radians. Default is np.pi/4.

    Returns
    -------
    rotations_grids : list of list of ndarray
        Each entry is a list of SU(2) rotation matrices (shape [n,2,2]). There are two entries,
        corresponding to the two orthogonal directions (+vec1, +vec2).
    angles_grids : list of ndarray
        Angles (in radians) used to generate the respective rotation grid.
    """
    if spinVec is not None:
        _, directions = genOrthRots(spinVec)
    elif filename is not None:
        _, directions, _ = genOrthRotFile(filename)
    else:
        raise ValueError("Must provide either spinVec or filename for rotation grid generation.")
    # Use +vec1 and +vec2 (directions[1] and directions[3])
    rotations_grids = []
    angles_grids = []
    for dir_vec in (directions[1], directions[3]):
        rot_grid, ang_grid = genRotsGrid(dir_vec, dphi)
        rotations_grids.append(rot_grid)
        angles_grids.append(ang_grid)
    return rotations_grids, angles_grids

def constructSOCterm(lambdas):
    """
    Construct spin-orbit coupling term for s, p and d orbitals.

    Uses the correct L·S operator construction via transformation from
    spherical harmonic basis to real orbital basis, following the verified
    approach in calcSOCOrbs.py.

    Parameters
    ----------
    lambdas : list
        List of spin-orbit coupling parameters [lambda_s, lambda_p, lambda_d]
        for s, p and d orbitals respectively

    Returns
    -------
    ndarray
        18x18 matrix containing spin-orbit coupling terms in the basis
        (s_up, s_dn, px_up, px_dn, py_up, py_dn, pz_up, pz_dn,
         d3z2r2_up, d3z2r2_dn, dxz_up, dxz_dn, dyz_up, dyz_dn,
         dx2y2_up, dx2y2_dn, dxy_up, dxy_dn)
    """

    def LOps(l):
        """Generate angular momentum operators in spherical harmonic basis."""
        m = np.arange(-l, l+1)
        Lz = np.diag(m) + 0j
        Lp = np.zeros((2*l+1, 2*l+1), dtype=complex)
        for i, mi in enumerate(m):
            for j, mj in enumerate(m):
                if mj == mi+1:
                    Lp[j, i] = np.sqrt((l-mi)*(l+mi+1))
        Lm = Lp.T
        Lx = 0.5*(Lp+Lm)
        Ly = -0.5j*(Lp-Lm)
        return Lx, Ly, Lz

    def genOrbList(l):
        """Generate transformation matrix from spherical to real orbitals."""
        m = np.zeros(2*l+1, dtype=int)
        for i in range(2*l+1):
            if i == 0:
                m[i] = 0
            elif i % 2 == 1:
                m[i] = (i + 1) // 2
            else:
                m[i] = -((i + 1) // 2)
        V = np.zeros((2*l+1, 2*l+1), dtype=complex)
        for i, mi in enumerate(m):
            if mi == 0:
                V[i, l] = 1.0
            elif mi > 0:
                V[i, l+mi] = ((-1.0)**mi)/np.sqrt(2)
                V[i, l-mi] = 1.0/np.sqrt(2)
            else:
                V[i, l+mi] = -((-1.0)**mi)*1j/np.sqrt(2)
                V[i, l-mi] = 1j/np.sqrt(2)
        return V

    def genLSMatrix(l):
        """Generate L·S matrix for angular momentum l."""
        Lx, Ly, Lz = LOps(l)
        orbs = genOrbList(l)
        # Transform to real orbital basis using Hermitian conjugate (dagger)
        Lx = orbs.conj().T @ Lx @ orbs
        Ly = orbs.conj().T @ Ly @ orbs
        Lz = orbs.conj().T @ Lz @ orbs
        # Construct L·S = 0.5 * [[Lz, L-], [L+, -Lz]]
        # where L- = Lx - i*Ly and L+ = Lx + i*Ly
        return 0.5*np.block([[Lz, Lx-1j*Ly], [Lx+1j*Ly, -Lz]])

    # s (l=0): L·S is always zero
    LdotS_s = genLSMatrix(0)  # 2x2 zero matrix

    # p (l=1): L·S in real orbital basis
    LdotS_p = genLSMatrix(1)  # 6x6 matrix

    # d (l=2): L·S in real orbital basis
    LdotS_d = genLSMatrix(2)  # 10x10 matrix

    # Construct block diagonal 18x18 matrix
    Hsoc = np.block([[lambdas[0]*LdotS_s, np.zeros((2,6)), np.zeros((2,10))],
                     [np.zeros((6,2)), lambdas[1]*LdotS_p, np.zeros((6,10))],
                     [np.zeros((10,2)), np.zeros((10,6)), lambdas[2]*LdotS_d]])
    return Hsoc
