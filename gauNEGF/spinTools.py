from gauNEGF.scfE import NEGFE
from gauNEGF.scf import NEGF
from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu
from scipy import io
from scipy.linalg import expm
import numpy as np
import re


def get_hirshfeld_data(filename):
    """Extract Hirshfeld charges and magnetic moments from Gaussian log file."""
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
    """Generate rotation matrix for spin-1/2 around axis n by angle omega"""
    sigx = np.array([[0, 1], [1, 0]], dtype=complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigz = np.array([[1, 0], [0, -1]], dtype=complex)
    return expm(-0.5j * omega * (n[0]*sigx + n[1]*sigy + n[2]*sigz))

def genRots(spinVec):
    """Generate 5 orthogonal spin rotations: ±vec1, ±vec2, -original"""
    unitVec = np.array(spinVec, dtype=float)
    unitVec = unitVec / np.linalg.norm(unitVec)
    
    # Robust orthogonal vector construction
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
    
    print(f"Original: {unitVec}")
    print(f"Vec1: {vec1}")  
    print(f"Vec2: {vec2}")
    
    # Generate rotations (π/2 rotations to orthogonal directions)
    rot_to_vec1 = genRot(vec2, np.pi/2)      # Original → vec1
    rot_to_neg_vec1 = genRot(vec2, -np.pi/2)  # Original → -vec1
    rot_to_vec2 = genRot(vec1, -np.pi/2)         # vec1 → vec2  
    rot_to_neg_vec2 = genRot(vec1, np.pi/2)    # vec1 → -vec2
    rot_to_neg_orig = genRot(vec1, np.pi)       # Original → -original
    rot_to_neg_orig2 = genRot(vec2, np.pi)       # Original → -original
    
    return [np.eye(2), rot_to_vec1, rot_to_neg_vec1, rot_to_vec2, rot_to_neg_vec2, rot_to_neg_orig, rot_to_neg_orig2], \
            np.array([unitVec, vec1, -vec1, vec2, -vec2, -unitVec, -unitVec])


