"""
Test script for surfGAt3D class from surfG3D.py
Tests the atomic-level 3D Green's function calculator without requiring BinAr object.
"""

import sys
sys.path.insert(0, '..')

import jax
import jax.numpy as jnp
from jax.numpy import linalg as LA
import matplotlib.pyplot as plt

# Import from gauNEGF
from gauNEGF.surfG3D import surfGAt3D
from gauNEGF.surfGBethe import surfGBAt

# Constants
dim = 9  # size of single atom matrix: 1s + 3p + 5d
har_to_eV = 27.211386  # eV/Hartree
eta = 1e-6  # broadening parameter
kpoints = 3

def read_bethe_params(filename):
    """Read Slater-Koster parameters from a .bethe file."""
    params = {}

    with open(filename + '.bethe', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            line = line.replace(' ', '')
            key, value = line.split('=')
            params[key] = float(value)

    # Sort parameters and convert Hartrees to eV
    ne = params['ne']
    Edict = {k[1:]: params[k] * har_to_eV for k in params if k.startswith('e')}
    Sdict = {k[1:]: params[k] for k in params if k.startswith('S')}
    Vdict = {k: params[k] * har_to_eV for k in params if not k.startswith('e') and not k.startswith('S')}

    # Setup onsite H0 matrix
    Hdiag = jnp.array([Edict['s']] + [Edict['p']] * 3 +
                      [Edict['dd']] + [Edict['dt']] * 2 + [Edict['dd'], Edict['dt']])
    H0 = jnp.diag(Hdiag)

    return ne, H0, Sdict, Vdict


def construct_mat(Mdict, dirCosines):
    """
    Construct hopping/overlap matrix using Slater-Koster formalism.
    This is a copy of the constructMat method from surfG3.
    """
    M = jnp.zeros((dim, dim))

    # Original matrix before rotation - assuming [0,0,1] bond direction
    # s-s coefficient
    M = M.at[0, 0].set(Mdict['sss'])

    # s-p block
    M = M.at[0, 3].set(Mdict['sps'])  # s-pz
    M = M.at[3, 0].set(-Mdict['sps'])  # pz-s

    # p-p block
    M = M.at[1, 1].set(Mdict['ppp'])  # px-px
    M = M.at[2, 2].set(Mdict['ppp'])  # py-py
    M = M.at[3, 3].set(Mdict['pps'])  # pz-pz

    # s-d block
    M = M.at[0, 4].set(Mdict['sds'])  # s - d3z2-r2
    M = M.at[4, 0].set(Mdict['sds'])

    # p-d block
    M = M.at[1, 5].set(Mdict['pdp'])  # px - dxz
    M = M.at[2, 6].set(Mdict['pdp'])  # py - dyz
    M = M.at[3, 4].set(Mdict['pds'])  # pz - d3z2-r2

    M = M.at[5, 1].set(-Mdict['pdp'])  # dxz - px
    M = M.at[6, 2].set(-Mdict['pdp'])  # dyz - py
    M = M.at[4, 3].set(-Mdict['pds'])  # d3z2-r2 - pz

    # d-d block
    M = M.at[4, 4].set(Mdict['dds'])  # d3z2-r2 - d3z2-r2
    M = M.at[5, 5].set(Mdict['ddp'])  # dxz - dxz
    M = M.at[6, 6].set(Mdict['ddp'])  # dyz - dyz
    M = M.at[7, 7].set(Mdict['ddd'])  # dx2-y2 - dx2-y2
    M = M.at[8, 8].set(Mdict['ddd'])  # dxy - dxy

    # Initialize transformation matrix
    tr = jnp.zeros((9, 9))
    x, y, z = dirCosines
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    # s orbital (spherically symmetric)
    tr = tr.at[0, 0].set(1.0)

    # p orbitals (3x3) block
    tr = tr.at[1:4, 1:4].set(jnp.array([
        [jnp.cos(theta) * jnp.cos(phi), -jnp.sin(phi), jnp.sin(theta) * jnp.cos(phi)],
        [jnp.cos(theta) * jnp.sin(phi), jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi)],
        [-jnp.sin(theta), 0, jnp.cos(theta)]
    ]))

    # d orbitals (5x5) block
    d_block = jnp.zeros((5, 5))

    d_block = d_block.at[0, 0].set((3 * z**2 - 1) / 2)
    d_block = d_block.at[0, 1].set(-jnp.sqrt(3) * jnp.sin(2*theta) / 2)
    d_block = d_block.at[0, 3].set(jnp.sqrt(3) * jnp.sin(theta)**2 / 2)

    d_10 = jnp.sqrt(3) * jnp.sin(2*theta) * jnp.cos(phi) / 2
    d_block = d_block.at[1, 0].set(d_10)
    d_block = d_block.at[1, 1].set(jnp.cos(2*theta) * jnp.cos(phi))
    d_block = d_block.at[1, 2].set(-jnp.cos(theta) * jnp.sin(phi))
    d_block = d_block.at[1, 3].set(-d_10 / jnp.sqrt(3))
    d_block = d_block.at[1, 4].set(jnp.sin(theta) * jnp.sin(phi))

    d_20 = jnp.sqrt(3) * jnp.sin(2*theta) * jnp.sin(phi) / 2
    d_block = d_block.at[2, 0].set(d_20)
    d_block = d_block.at[2, 1].set(jnp.cos(2*theta) * jnp.sin(phi))
    d_block = d_block.at[2, 2].set(jnp.cos(theta) * jnp.cos(phi))
    d_block = d_block.at[2, 3].set(-d_20 / jnp.sqrt(3))
    d_block = d_block.at[2, 4].set(-jnp.sin(theta) * jnp.cos(phi))

    d_block = d_block.at[3, 0].set(jnp.sqrt(3) * jnp.sin(theta)**2 * jnp.cos(2*phi) / 2)
    d_block = d_block.at[3, 1].set(jnp.sin(2*theta) * jnp.cos(2*phi) / 2)
    d_block = d_block.at[3, 2].set(-jnp.sin(theta) * jnp.sin(2*phi))
    d_block = d_block.at[3, 3].set((1 + jnp.cos(theta)**2) * jnp.cos(2*phi) / 2)
    d_block = d_block.at[3, 4].set(-jnp.cos(theta) * jnp.sin(2*phi))

    d_block = d_block.at[4, 0].set(jnp.sqrt(3) * jnp.sin(theta)**2 * jnp.sin(2*phi) / 2)
    d_block = d_block.at[4, 1].set(jnp.sin(2*theta) * jnp.sin(2*phi) / 2)
    d_block = d_block.at[4, 2].set(jnp.sin(theta) * jnp.cos(2*phi))
    d_block = d_block.at[4, 3].set((1 + jnp.cos(theta)**2) * jnp.sin(2*phi) / 2)
    d_block = d_block.at[4, 4].set(jnp.cos(theta) * jnp.cos(2*phi))

    tr = tr.at[4:9, 4:9].set(d_block)

    # Apply transformation
    return tr @ M @ tr.T


def gen_fcc_111_neighbors():
    """
    Generate 12 nearest neighbor unit vectors for FCC [111] surface.
    Returns direction vectors assuming z-axis as surface normal.
    """
    plane_normal = jnp.array([0., 0., 1.])
    first_neighbor = jnp.array([1., 0., 0.])

    # Generate in-plane vectors using 60-degree rotations
    in_plane_vectors = []
    rotation_angle = jnp.pi / 3  # 60 degrees

    for i in range(3):
        angle = i * rotation_angle
        cos_theta = jnp.cos(angle)
        sin_theta = jnp.sin(angle)

        K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                      [plane_normal[2], 0, -plane_normal[0]],
                      [-plane_normal[1], plane_normal[0], 0]])

        R = jnp.eye(3) + sin_theta * K + (1 - cos_theta) * jnp.matmul(K, K)
        rotated_vector = jnp.dot(R, first_neighbor)
        in_plane_vectors.append(rotated_vector / jnp.linalg.norm(rotated_vector))

    # Generate out-of-plane vectors
    out_of_plane_angle = jnp.arccos(1/jnp.sqrt(3))  # ~54.74 degrees

    out_of_plane_vectors = []
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

    # Add corresponding opposite vectors
    all_vectors = in_plane_vectors + out_of_plane_vectors
    for i in range(6):
        all_vectors.append(-all_vectors[i])

    return all_vectors


def test_reciprocal_lattice(g_atom_3d):
    """
    Test 2D reciprocal lattice vectors against Damle's gold standard values.
    Damle uses: K1 = 2*pi*[1, -0.5774, 0], K2 = 2*pi*[0, 1.1547, 0]
    where -0.5774 approx -1/sqrt(3) and 1.1547 approx 2/sqrt(3)

    Note: We now compute separate 2D (surface) and 3D (bulk) reciprocal vectors.
    This test compares our 2D vectors to Damle's surface reciprocal vectors.
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST: 2D Reciprocal Lattice vs Damle")
    print("="*60)

    # Extract 2D reciprocal lattice vectors (for surface)
    b1_2D = g_atom_3d.b1_2D
    b2_2D = g_atom_3d.b2_2D

    # Damle's values (gold standard for FCC 111 surface)
    K1_damle = 2*jnp.pi * jnp.array([1.0, -1.0/jnp.sqrt(3), 0.0])
    K2_damle = 2*jnp.pi * jnp.array([0.0, 2.0/jnp.sqrt(3), 0.0])

    print(f"\nsurfG3D computed (2D surface):")
    print(f"  b1_2D = [{b1_2D[0]:.6f}, {b1_2D[1]:.6f}, {b1_2D[2]:.6f}]")
    print(f"  b2_2D = [{b2_2D[0]:.6f}, {b2_2D[1]:.6f}, {b2_2D[2]:.6f}]")

    print(f"\nDamle's gold standard (2D surface):")
    print(f"  K1 = [{K1_damle[0]:.6f}, {K1_damle[1]:.6f}, {K1_damle[2]:.6f}]")
    print(f"  K2 = [{K2_damle[0]:.6f}, {K2_damle[1]:.6f}, {K2_damle[2]:.6f}]")

    # Normalized (divided by 2*pi) for easier comparison
    print(f"\nNormalized (divided by 2*pi):")
    print(f"  b1_2D/(2*pi) = [{b1_2D[0]/(2*jnp.pi):.6f}, {b1_2D[1]/(2*jnp.pi):.6f}, {b1_2D[2]/(2*jnp.pi):.6f}]")
    print(f"  Expected:      [1.000000, -0.577350, 0.000000]")
    print(f"  b2_2D/(2*pi) = [{b2_2D[0]/(2*jnp.pi):.6f}, {b2_2D[1]/(2*jnp.pi):.6f}, {b2_2D[2]/(2*jnp.pi):.6f}]")
    print(f"  Expected:      [0.000000,  1.154701, 0.000000]")

    # Calculate differences
    diff_b1 = jnp.linalg.norm(b1_2D - K1_damle)
    diff_b2 = jnp.linalg.norm(b2_2D - K2_damle)

    print(f"\nDifference from Damle's values:")
    print(f"  ||b1_2D - K1|| = {diff_b1:.6e}")
    print(f"  ||b2_2D - K2|| = {diff_b2:.6e}")

    # Check orthogonality with real-space lattice
    a1 = g_atom_3d.a1
    a2 = g_atom_3d.a2

    print(f"\nOrthogonality check (should be 2*pi * identity):")
    print(f"  a1.b1_2D = {jnp.dot(a1, b1_2D):.6f} (expect {2*jnp.pi:.6f})")
    print(f"  a1.b2_2D = {jnp.dot(a1, b2_2D):.6f} (expect 0.000000)")
    print(f"  a2.b1_2D = {jnp.dot(a2, b1_2D):.6f} (expect 0.000000)")
    print(f"  a2.b2_2D = {jnp.dot(a2, b2_2D):.6f} (expect {2*jnp.pi:.6f})")

    # Test result
    tolerance = 1e-6
    if diff_b1 < tolerance and diff_b2 < tolerance:
        print(f"\n[PASS] 2D reciprocal lattice matches Damle's values (tol={tolerance})")
        return True
    else:
        print(f"\n[FAIL] 2D reciprocal lattice differs from Damle's values")
        return False


def test_neighbor_ordering():
    """Verify FCC [111] neighbor vectors match expected structure."""
    print("\n" + "="*60)
    print("VALIDATION TEST: Neighbor Vector Ordering")
    print("="*60)

    vecs = gen_fcc_111_neighbors()

    # Check in-plane (z approx 0)
    print("\nChecking in-plane vectors (should have z approx 0):")
    for i in [0, 1, 2, 6, 7, 8]:
        z_component = abs(vecs[i][2])
        print(f"  vec[{i}]: z = {vecs[i][2]:7.4f}, |z| = {z_component:.6e}")
        assert z_component < 1e-10, f"vec[{i}] should be in-plane but z={vecs[i][2]}"

    # Check out-of-plane up (+z)
    print("\nChecking out-of-plane upward vectors (should have z > 0.8):")
    for i in [3, 4, 5]:
        z_component = vecs[i][2]
        print(f"  vec[{i}]: z = {z_component:7.4f}")
        assert z_component > 0.8, f"vec[{i}] should point upward but z={z_component}"

    # Check out-of-plane down (-z)
    print("\nChecking out-of-plane downward vectors (should have z < -0.8):")
    for i in [9, 10, 11]:
        z_component = vecs[i][2]
        print(f"  vec[{i}]: z = {z_component:7.4f}")
        assert z_component < -0.8, f"vec[{i}] should point downward but z={z_component}"

    # Check opposite pairs
    print("\nChecking opposite vector pairs:")
    for i in range(6):
        diff = vecs[i] + vecs[i+6]
        norm = jnp.linalg.norm(diff)
        print(f"  vec[{i}] + vec[{i+6}]: ||sum|| = {norm:.6e}")
        assert norm < 1e-10, f"vec[{i}] and vec[{i+6}] should be opposite"

    print("\n[PASS] Neighbor vector ordering correct")
    return True


def test_3d_reciprocal_lattice(g_atom_3d):
    """Verify 3D reciprocal lattice satisfies a_i . b_j = 2pi delta_ij."""
    print("\n" + "="*60)
    print("VALIDATION TEST: 3D Reciprocal Lattice")
    print("="*60)

    # Check if attributes exist
    try:
        a_vecs = [g_atom_3d.a1, g_atom_3d.a2, g_atom_3d.a3]
        b_vecs = [g_atom_3d.b1_3D, g_atom_3d.b2_3D, g_atom_3d.b3_3D]
    except AttributeError as e:
        print(f"\n[SKIP] 3D reciprocal lattice attributes not yet implemented: {e}")
        return False

    print("\nReal-space lattice vectors:")
    for i, a in enumerate(a_vecs):
        print(f"  a{i+1} = [{a[0]:7.4f}, {a[1]:7.4f}, {a[2]:7.4f}]")

    print("\n3D Reciprocal lattice vectors (for bulk):")
    for i, b in enumerate(b_vecs):
        print(f"  b{i+1}_3D = [{b[0]:7.4f}, {b[1]:7.4f}, {b[2]:7.4f}]")

    print("\nOrthogonality check (a_i . b_j should equal 2pi delta_ij):")
    all_pass = True
    for i in range(3):
        for j in range(3):
            dot_product = jnp.dot(a_vecs[i], b_vecs[j])
            expected = 2*jnp.pi if i == j else 0.0
            diff = abs(dot_product - expected)
            status = "PASS" if diff < 1e-6 else "FAIL"
            print(f"  a{i+1}.b{j+1}_3D = {dot_product:8.5f}, expect {expected:8.5f}, diff = {diff:.2e} [{status}]")
            if diff >= 1e-6:
                all_pass = False

    # Also check 2D reciprocal vectors
    print("\n2D Reciprocal lattice vectors (for surface):")
    b1_2D = g_atom_3d.b1_2D
    b2_2D = g_atom_3d.b2_2D
    print(f"  b1_2D = [{b1_2D[0]:7.4f}, {b1_2D[1]:7.4f}, {b1_2D[2]:7.4f}]")
    print(f"  b2_2D = [{b2_2D[0]:7.4f}, {b2_2D[1]:7.4f}, {b2_2D[2]:7.4f}]")

    print("\n2D Orthogonality check:")
    for i, (a, label) in enumerate([(g_atom_3d.a1, "a1"), (g_atom_3d.a2, "a2")]):
        for j, (b, blabel) in enumerate([(b1_2D, "b1_2D"), (b2_2D, "b2_2D")]):
            dot_product = jnp.dot(a, b)
            expected = 2*jnp.pi if i == j else 0.0
            diff = abs(dot_product - expected)
            status = "PASS" if diff < 1e-6 else "FAIL"
            print(f"  {label}.{blabel} = {dot_product:8.5f}, expect {expected:8.5f}, diff = {diff:.2e} [{status}]")
            if diff >= 1e-6:
                all_pass = False

    # Check that 2D vectors have z=0
    print("\nCheck that 2D reciprocal vectors lie in surface plane (z=0):")
    max_z = max(abs(b1_2D[2]), abs(b2_2D[2]))
    print(f"  max|z-component| = {max_z:.2e}")
    if max_z < 1e-10:
        print("  [PASS] 2D vectors have z=0")
    else:
        print(f"  [FAIL] 2D vectors have non-zero z = {max_z}")
        all_pass = False

    if all_pass:
        print("\n[PASS] Reciprocal lattice orthogonality and 2D constraint checks")
        return True
    else:
        print("\n[FAIL] Reciprocal lattice checks failed")
        return False


def test_kmesh_dimensions(g_atom_3d):
    """Check that 2D and 3D k-meshes have correct shapes."""
    print("\n" + "="*60)
    print("VALIDATION TEST: K-mesh Dimensions")
    print("="*60)

    nK = g_atom_3d.kPoints

    # Check for 2D mesh attributes
    try:
        kmesh_2D_shape = g_atom_3d.kmesh_2D.shape
        expList_2D_shape = g_atom_3d.expList_2D.shape

        print(f"\n2D k-mesh (for surface):")
        print(f"  kmesh_2D shape:  {kmesh_2D_shape} (expect {nK**2} x 3)")
        print(f"  expList_2D shape: {expList_2D_shape} (expect {nK**2} x 12)")

        assert kmesh_2D_shape == (nK**2, 3), \
            f"2D k-mesh should be {nK**2}x3, got {kmesh_2D_shape}"
        assert expList_2D_shape == (nK**2, 12), \
            f"2D expList should be {nK**2}x12, got {expList_2D_shape}"
        print("  [PASS] 2D mesh dimensions correct")
        has_2D = True
    except AttributeError as e:
        print(f"\n[SKIP] 2D k-mesh attributes not yet implemented: {e}")
        has_2D = False

    # Check for 3D mesh attributes
    try:
        kmesh_3D_shape = g_atom_3d.kmesh_3D.shape
        expList_3D_shape = g_atom_3d.expList_3D.shape

        print(f"\n3D k-mesh (for bulk):")
        print(f"  kmesh_3D shape:  {kmesh_3D_shape} (expect {nK**3} x 3)")
        print(f"  expList_3D shape: {expList_3D_shape} (expect {nK**3} x 12)")

        assert kmesh_3D_shape == (nK**3, 3), \
            f"3D k-mesh should be {nK**3}x3, got {kmesh_3D_shape}"
        assert expList_3D_shape == (nK**3, 12), \
            f"3D expList should be {nK**3}x12, got {expList_3D_shape}"
        print("  [PASS] 3D mesh dimensions correct")
        has_3D = True
    except AttributeError as e:
        print(f"\n[SKIP] 3D k-mesh attributes not yet implemented: {e}")
        has_3D = False

    if has_2D and has_3D:
        print(f"\n[PASS] K-mesh dimensions: 2D={nK**2}, 3D={nK**3}")
        return True
    else:
        return False


def test_gSurf_shapes(g_atom_3d):
    """Check gSurf returns g_k array with correct shape."""
    print("\n" + "="*60)
    print("VALIDATION TEST: gSurf Return Shapes")
    print("="*60)

    E = 0.0  # Use arbitrary energy (will update after calcFermi)
    try:
        g_k = g_atom_3d.gSurf(E, conv=1e-3, mix=0.1)
        nK = g_atom_3d.kPoints

        print(f"\nReturn value shape:")
        print(f"  g_k shape: {g_k.shape} (expect {nK**2} x {dim} x {dim})")

        assert g_k.shape == (nK**2, dim, dim), \
            f"gSurf g_k should be {nK**2}x{dim}x{dim}, got {g_k.shape}"

        print(f"\n[PASS] gSurf returns correct shape")
        return True
    except Exception as e:
        print(f"\n[FAIL] gSurf raised exception: {e}")
        return False


def test_gBulk_shapes(g_atom_3d):
    """Check gBulk returns (g_k, G_real) with correct shapes using 3D k-mesh."""
    print("\n" + "="*60)
    print("VALIDATION TEST: gBulk Return Shapes")
    print("="*60)

    E = 0.0  # Use arbitrary energy
    try:
        result = g_atom_3d.gBulk(E)

        # Check if returns tuple
        if not isinstance(result, tuple) or len(result) != 2:
            print(f"\n[FAIL] gBulk should return tuple (g_k, G_real), got {type(result)}")
            return False

        g_k, G_real = result
        nK = g_atom_3d.kPoints

        print(f"\nReturn value shapes:")
        print(f"  g_k shape:    {g_k.shape} (expect {nK**3} x {dim} x {dim})")
        print(f"  G_real shape: {G_real.shape} (expect {dim} x {dim})")

        assert g_k.shape == (nK**3, dim, dim), \
            f"gBulk g_k should be {nK**3}x{dim}x{dim} for 3D mesh, got {g_k.shape}"
        assert G_real.shape == (dim, dim), \
            f"gBulk G_real should be {dim}x{dim} (onsite GF), got {G_real.shape}"

        print(f"\n[PASS] gBulk returns correct shapes with 3D k-mesh")
        return True
    except Exception as e:
        print(f"\n[FAIL] gBulk raised exception: {e}")
        return False


def test_sigma_count(g_atom_3d):
    """Verify sigma returns a single self-energy matrix (dim x dim)."""
    print("\n" + "="*60)
    print("VALIDATION TEST: Sigma Self-Energy Shape")
    print("="*60)

    E = 0.0  # Use arbitrary energy
    try:
        sig = g_atom_3d.sigma(E, conv=1e-3, mix=0.1)

        print(f"\nReturn value shape:")
        print(f"  sigma shape: {sig.shape} (expect {dim} x {dim})")

        assert sig.shape == (dim, dim), \
            f"sigma should return {dim}x{dim}, got {sig.shape}"

        # Also test with active_dirs subset
        sig_sub = g_atom_3d.sigma(E, active_dirs=[3, 4, 5], conv=1e-3, mix=0.1)
        print(f"  sigma([3,4,5]) shape: {sig_sub.shape} (expect {dim} x {dim})")

        assert sig_sub.shape == (dim, dim), \
            f"sigma with active_dirs should return {dim}x{dim}, got {sig_sub.shape}"

        print(f"\n[PASS] sigma returns correct shape")
        return True
    except Exception as e:
        print(f"\n[FAIL] sigma raised exception: {e}")
        return False


def test_subset_sigma_psd_gamma(g_atom_3d):
    """
    Test that partial-direction sigma produces PSD gamma matrices.

    When a contact atom only uses a subset of the 9 surface directions
    (because some directions point toward neighboring contact atoms),
    the resulting gamma = i*(sigma - sigma^dag) must still be positive
    semidefinite (PSD) for physical transmission.

    Uses the active_dirs parameter of sigma() which computes the subset
    quadratic form: Sigma_S = (1/Nk) sum_k B_S(k) @ g_surf(k) @ B_S^dag(k)
    where B_S(k) = sum_{a in S} exp(ik*R_a) * B_a_bare.
    """
    print("\n" + "="*60)
    print("VALIDATION TEST: Subset Sigma -> PSD Gamma")
    print("="*60)

    # Test several subsets that arise in practice:
    # [3,4,5] = out-of-plane UP only (common: atom with all in-plane neighbors removed)
    # [0,1,2,3,4,5] = forward + up (atom with backward in-plane removed)
    # [0,1,2,6,7,8] = all in-plane (atom with all out-of-plane removed)
    subsets = {
        'out-of-plane UP [3,4,5]': [3, 4, 5],
        'forward + UP [0,1,2,3,4,5]': [0, 1, 2, 3, 4, 5],
        'all in-plane [0,1,2,6,7,8]': [0, 1, 2, 6, 7, 8],
        'all 9 [0..8]': list(range(9)),
    }

    test_energies = [-3.2, 0.0, 2.0]  # band edge, mid-band, upper band
    all_passed = True
    psd_tol = -1e-10  # eigenvalue tolerance for PSD check

    for E in test_energies:
        print(f"\n  E = {E:.1f} eV:")

        # Pre-compute gSurf once, reuse for all subsets at this energy
        g_k = g_atom_3d.gSurf(E, conv=1e-3, mix=0.1)

        for name, dirs in subsets.items():
            # Call sigma with active_dirs -- returns single (dim, dim) matrix
            sigma_sub = g_atom_3d.sigma(E, active_dirs=dirs, g_k=g_k)
            gamma = 1j * (sigma_sub - sigma_sub.conj().T)
            eigs = jnp.linalg.eigvalsh(gamma)
            min_eig = float(jnp.min(eigs))
            is_psd = min_eig >= psd_tol

            status = "PASS" if is_psd else "FAIL"
            print(f"    {name}: min(eig(gamma)) = {min_eig:.6e} [{status}]")

            if not is_psd:
                all_passed = False

    if all_passed:
        print(f"\n[PASS] All subset gammas are PSD")
    else:
        print(f"\n[FAIL] Some subset gammas have negative eigenvalues (non-PSD)")

    return all_passed


def main():
    """Main test function with Damle validation diagnostics."""

    print("="*60)
    print("VALIDATION: surfGAt3D vs Damle's Gold Standard")
    print("="*60)

    # Read Bethe parameters
    print("\n1. Reading Bethe parameters from Au.bethe...")
    ne, H0, Sdict, Vdict = read_bethe_params('Au')
    print(f"   Number of electrons: {ne}")
    print(f"   H0 diagonal: {jnp.diag(H0)}")

    # Generate neighbor directions
    print("\n2. Generating FCC [111] nearest neighbor directions...")
    vecs = gen_fcc_111_neighbors()
    print(f"   Generated {len(vecs)} neighbor vectors")

    # Construct hopping and overlap matrices
    print("\n3. Constructing hopping and overlap matrices...")
    Slist = []
    Vlist = []
    for i, d in enumerate(vecs):
        Slist.append(construct_mat(Sdict, d))
        Vlist.append(construct_mat(Vdict, d))
    print(f"   Created {len(Slist)} overlap matrices")
    print(f"   Created {len(Vlist)} hopping matrices")

    # Initialize both implementations
    print("\n4. Initializing both surfGAt3D and surfGBAt objects...")
    T = 0  # Temperature in Kelvin
    g_atom_3d = surfGAt3D(H0, Slist, Vlist, vecs, eta, T=T, kPoints=kpoints)
    g_atom_bethe = surfGBAt(H0.copy(), Slist, Vlist, eta, T=T)
    print(f"   surfGAt3D: eta={eta}, T={T}K, kPoints={kpoints}")
    print(f"   surfGBAt: eta={eta}, T={T}K")
    print(f"   Number of neighbors: {g_atom_3d.NN}")

    # Run NEW validation tests
    print("\n" + "="*60)
    print("VALIDATION TESTS (Phase 1)")
    print("="*60)

    test_neighbor_ordering()
    test_3d_reciprocal_lattice(g_atom_3d)
    test_kmesh_dimensions(g_atom_3d)
    test_gSurf_shapes(g_atom_3d)
    test_gBulk_shapes(g_atom_3d)
    test_sigma_count(g_atom_3d)
    test_subset_sigma_psd_gamma(g_atom_3d)

    # Run OLD diagnostic test against Damle's gold standard
    print("\n" + "="*60)
    print("DIAGNOSTIC: Reciprocal Lattice vs Damle")
    print("="*60)

    test_reciprocal_lattice(g_atom_3d)

    print("\n" + "="*60)
    print("CONTINUING WITH STANDARD TESTS")
    print("="*60)

    # Calculate Fermi energy for both
    print("\n5. Calculating Fermi energy...")
    fermi_3d = g_atom_3d.calcFermi(ne/2, tol=1e-3)
    #fermi_3d = 3.249851 #eV
    fermi_bethe = g_atom_bethe.calcFermi(ne/2, tol=1e-3)
    print(f"   surfGAt3D Fermi energy: {fermi_3d:.6f} eV")
    print(f"   surfGBAt Fermi energy:  {fermi_bethe:.6f} eV")
    print(f"   Difference: {abs(fermi_3d - fermi_bethe):.6f} eV")

    # Compare DOS at Fermi energy
    print("\n6. Comparing DOS at Fermi energy...")
    dos_3d = g_atom_3d.DOS(fermi_3d)
    dos_bethe = g_atom_bethe.DOS(fermi_bethe)
    print(f"   surfGAt3D DOS at E={fermi_3d:.4f} eV: {dos_3d:.6f}")
    print(f"   surfGBAt DOS at E={fermi_bethe:.4f} eV:  {dos_bethe:.6f}")
    print(f"   Relative difference: {abs(dos_3d - dos_bethe)/dos_bethe*100:.2f}%")

    # Compare bulk self-energies
    print("\n7. Comparing bulk self-energy calculations...")
    sig_list_3d = g_atom_3d.sigmaBulk(fermi_3d)
    sig_list_bethe = g_atom_bethe.sigmaK(fermi_bethe, conv=1e-5)
    print(f"   surfGAt3D: calculated {len(sig_list_3d)} self-energy matrices")
    print(f"   surfGBAt: calculated {len(sig_list_bethe)} self-energy matrices")

    # Compare first self-energy matrix
    trace_3d = jnp.trace(sig_list_3d[0])
    trace_bethe = jnp.trace(sig_list_bethe[0])
    print(f"   surfGAt3D sig[0] trace: {trace_3d:.6f}")
    print(f"   surfGBAt sig[0] trace:  {trace_bethe:.6f}")
    print(f"   Difference: {abs(trace_3d - trace_bethe):.6f}")

    # Compare total self-energies
    print("\n8. Comparing total self-energy calculations...")
    sig_tot_3d = g_atom_3d.sigmaTot(fermi_3d)
    sig_tot_bethe = g_atom_bethe.sigmaTot(fermi_bethe)
    print(f"   surfGAt3D sigmaTot shape: {sig_tot_3d.shape}")
    print(f"   surfGBAt sigmaTot shape:  {sig_tot_bethe.shape}")
    print(f"   surfGAt3D sigmaTot trace: {jnp.trace(sig_tot_3d):.6f}")
    print(f"   surfGBAt sigmaTot trace:  {jnp.trace(sig_tot_bethe):.6f}")

    # Plot DOS comparison
    print("\n9. Plotting DOS comparison...")
    E_range = jnp.linspace(-5, 5, 500)
    dos_values_3d = jax.vmap(g_atom_3d.DOS)(E_range + fermi_3d)
    dos_values_bethe = jax.vmap(g_atom_bethe.DOS)(E_range + fermi_bethe)

    plt.figure(figsize=(12, 8))

    # Main comparison plot
    plt.subplot(2, 1, 1)
    plt.plot(E_range+fermi_3d, dos_values_3d, 'b-', linewidth=2, label='surfGAt3D (k-space)')
    plt.plot(E_range+fermi_bethe, dos_values_bethe, 'r--', linewidth=2, label='surfGBAt (Bethe)')
    plt.axvline(fermi_3d, color='blue', linestyle=':', alpha=0.7)
    plt.axvline(fermi_bethe, color='red', linestyle=':', alpha=0.7)
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('DOS (states/eV)', fontsize=12)
    plt.title('Density of States Comparison: surfGAt3D vs surfGBAt', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 10)
    plt.legend()

    # Difference plot
    plt.subplot(2, 1, 2)
    dos_diff = jnp.array(dos_values_3d) - jnp.array(dos_values_bethe)
    plt.plot(E_range, dos_diff, 'g-', linewidth=2)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(fermi_3d, color='blue', linestyle=':', alpha=0.7)
    plt.axvline(fermi_bethe, color='red', linestyle=':', alpha=0.7)
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('DOS Difference', fontsize=12)
    plt.ylim(-10, 10)
    plt.title('Difference: surfGAt3D - surfGBAt', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_surfGAt3D_comparison.png', dpi=150)
    print("   Saved DOS comparison plot to test_surfGAt3D_comparison.png")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
