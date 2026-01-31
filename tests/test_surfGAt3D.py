"""
Test script for surfGAt3D class from surfG3D.py
Tests the atomic-level 3D Green's function calculator without requiring BinAr object.
"""

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
eta = 1e-4  # broadening parameter

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


def main():
    """Main test function comparing surfGAt3D and surfGBAt."""

    print("="*60)
    print("Comparing surfGAt3D (k-space) vs surfGBAt (Bethe lattice)")
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
    g_atom_3d = surfGAt3D(H0, Slist, Vlist, vecs, eta, T=T, kPoints=3)
    g_atom_bethe = surfGBAt(H0.copy(), Slist, Vlist, eta, T=T)
    print(f"   surfGAt3D: eta={eta}, T={T}K, kPoints=5")
    print(f"   surfGBAt: eta={eta}, T={T}K")
    print(f"   Number of neighbors: {g_atom_3d.NN}")

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
    sig_list_3d = g_atom_3d.sigmaBulk(fermi_3d, conv=1e-5)
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
    E_range = jnp.linspace(fermi_3d - 5, fermi_3d + 5, 500)
    dos_values_3d = jax.vmap(g_atom_3d.DOS)(E_range)
    E_range_bethe = jnp.linspace(fermi_bethe - 5, fermi_bethe + 5, 500)
    dos_values_bethe = jax.vmap(g_atom_bethe.DOS)(E_range_bethe)

    plt.figure(figsize=(12, 8))

    # Main comparison plot
    plt.subplot(2, 1, 1)
    plt.plot(E_range, dos_values_3d, 'b-', linewidth=2, label='surfGAt3D (k-space)')
    plt.plot(E_range_bethe, dos_values_bethe, 'r--', linewidth=2, label='surfGBAt (Bethe)')
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
