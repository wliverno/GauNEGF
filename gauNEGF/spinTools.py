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
    unitVec = unitVec / np.linalg.norm(unitVec)

    # Robust orthogonal vector construction -- avoids degeneracy when
    # unitVec is aligned with the y-axis
    if abs(unitVec[0]) > 0 or abs(unitVec[2]) > 0:
        vec1 = np.array([unitVec[2], 0, -unitVec[0]])
    else:
        vec1 = np.array([0, unitVec[2], -unitVec[1]])

    vec1 = vec1 / np.linalg.norm(vec1)
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
    max_moment_index = np.argmax(np.linalg.norm(mag_moments, axis=1))
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