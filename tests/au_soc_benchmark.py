"""
SOC Benchmarking: Compare Au.bethe (scalar) vs AuSOC.bethe (with spin-orbit)

Shows how spin-orbit coupling splits the d-bands in gold.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec

from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors

# Constants
dim = 9
har_to_eV = 27.211386


def read_soc_params(filename):
    """Read SOC parameters from a .bethe file (returns lambda_p, lambda_d in eV)."""
    params = {}
    with open(filename + '.bethe', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            line = line.replace(' ', '')
            key, value = line.split('=')
            params[key] = float(value)
    soc_p = params.get('soc_p', 0.0) * har_to_eV
    soc_d = params.get('soc_d', 0.0) * har_to_eV
    return soc_p, soc_d


def build_Hsoc(soc_p, soc_d):
    """Build 18x18 SOC Hamiltonian (L.S term)."""
    from gauNEGF.spinTools import constructSOCterm
    lambdas = [0.0, soc_p, soc_d]
    return constructSOCterm(lambdas)


def compute_reciprocal_vectors(vecs):
    """Compute 3D reciprocal lattice vectors from real-space primitive vectors.

    Uses vecs[0], vecs[1], vecs[3] (two in-plane + one out-of-plane) as the
    rhombohedral primitive cell of the FCC lattice in the [111] frame.
    """
    a1, a2, a3 = np.array(vecs[0]), np.array(vecs[1]), np.array(vecs[3])
    vol = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / vol
    b2 = 2 * np.pi * np.cross(a3, a1) / vol
    b3 = 2 * np.pi * np.cross(a1, a2) / vol
    return b1, b2, b3


def compute_kpoints_111(vecs):
    """Compute FCC high-symmetry k-points in fractional coordinates for the
    [111]-frame rhombohedral primitive cell (a1=vecs[0], a2=vecs[1], a3=vecs[3]).

    The standard FCC k-points (X, L, W, K) are defined in the conventional
    cubic frame. We transform them to the [111] rotated frame and express
    them in the reciprocal basis of the rhombohedral cell.
    """
    a1, a2, a3 = np.array(vecs[0]), np.array(vecs[1]), np.array(vecs[3])
    vol = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / vol
    b2 = 2 * np.pi * np.cross(a3, a1) / vol
    b3 = 2 * np.pi * np.cross(a1, a2) / vol
    B_inv = np.linalg.inv(np.column_stack([b1, b2, b3]))

    # Rotation from cubic to [111] frame
    ex = np.array([1, -1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, -2]) / np.sqrt(6)
    ez = np.array([1, 1, 1]) / np.sqrt(3)
    R = np.array([ex, ey, ez])

    # FCC with nearest-neighbor distance = 1 -> a = sqrt(2)
    scale = 2 * np.pi / np.sqrt(2)

    # High-symmetry points in cubic Cartesian k-space
    cubic_pts = {
        'G': np.array([0.0, 0.0, 0.0]),
        'X': scale * np.array([1.0, 0.0, 0.0]),
        'L': scale * np.array([0.5, 0.5, 0.5]),
        'W': scale * np.array([1.0, 0.5, 0.0]),
        'K': scale * np.array([0.75, 0.75, 0.0]),
    }

    # Transform to [111] frame, then to fractional coordinates
    k_special = {}
    for name, k_cubic in cubic_pts.items():
        k_111 = R @ k_cubic
        k_frac = B_inv @ k_111
        k_special[name] = k_frac

    return k_special


def frac_to_cart(k_frac, b1, b2, b3):
    return k_frac[0] * b1 + k_frac[1] * b2 + k_frac[2] * b3


def get_eigenvalues_scalar(H0, Vlist, Slist, vecs, k_frac, b1, b2, b3):
    """Eigenvalues for scalar (no SOC) case -- 9 bands."""
    k_cart = frac_to_cart(k_frac, b1, b2, b3)
    H_k = np.array(H0, dtype=complex).copy()
    S_k = np.eye(dim, dtype=complex)
    for i, vec in enumerate(vecs):
        phase = np.exp(1j * np.dot(k_cart, vec))
        H_k += phase * np.array(Vlist[i])
        S_k += phase * np.array(Slist[i])
    return np.sort(eigh(H_k, S_k, eigvals_only=True).real)


def get_eigenvalues_soc(H0, Vlist, Slist, Hsoc, vecs, k_frac, b1, b2, b3):
    """Eigenvalues for SOC case -- 18 bands."""
    k_cart = frac_to_cart(k_frac, b1, b2, b3)
    # Spin-double everything: kron with eye(2)
    H_k = np.kron(np.array(H0, dtype=complex), np.eye(2)) + Hsoc
    S_k = np.kron(np.eye(dim, dtype=complex), np.eye(2))
    for i, vec in enumerate(vecs):
        phase = np.exp(1j * np.dot(k_cart, vec))
        H_k += phase * np.kron(np.array(Vlist[i], dtype=complex), np.eye(2))
        S_k += phase * np.kron(np.array(Slist[i], dtype=complex), np.eye(2))
    return np.sort(eigh(H_k, S_k, eigvals_only=True).real)


def find_fermi(eigfunc, args, n_electrons, n_k_sample=20):
    """Find Fermi level by sampling k-space."""
    np.random.seed(42)
    k_points = np.random.rand(n_k_sample**3, 3) - 0.5
    all_evals = []
    for k in k_points:
        all_evals.extend(eigfunc(*args, k))
    all_evals = np.sort(all_evals)
    fermi_idx = int(n_electrons * len(k_points))
    return all_evals[min(fermi_idx, len(all_evals) - 1)]


def band_structure(eigfunc, args, E_fermi, k_special, n_points=60):
    """Calculate band structure along high-symmetry path."""
    path = ['G', 'X', 'W', 'L', 'G', 'K']
    b1, b2, b3 = args[-3], args[-2], args[-1]

    k_path = []
    for i in range(len(path) - 1):
        ks, ke = k_special[path[i]], k_special[path[i+1]]
        for j in range(n_points):
            t = j / (n_points - 1)
            k_path.append(ks + t * (ke - ks))

    # Compute distances
    distances = np.zeros(len(k_path))
    for i in range(1, len(k_path)):
        kc_prev = frac_to_cart(k_path[i-1], b1, b2, b3)
        kc = frac_to_cart(k_path[i], b1, b2, b3)
        distances[i] = distances[i-1] + np.linalg.norm(kc - kc_prev)

    # Label positions
    label_pos = [distances[0]]
    for i in range(1, len(path) - 1):
        label_pos.append(distances[i * n_points])
    label_pos.append(distances[-1])

    # Eigenvalues
    bands = np.array([eigfunc(*args, k) - E_fermi for k in k_path])

    return distances, bands, path, label_pos


def calculate_dos(eigfunc, args, E_fermi, n_k_sample=30):
    """Calculate DOS by k-space sampling."""
    np.random.seed(42)
    k_points = np.random.rand(n_k_sample**3, 3) - 0.5
    all_evals = []
    for k in k_points:
        all_evals.extend(eigfunc(*args, k) - E_fermi)
    all_evals = np.array(all_evals)
    dos_hist, bin_edges = np.histogram(all_evals, bins=300, density=True)
    dos_energies = (bin_edges[:-1] + bin_edges[1:]) / 2
    return dos_energies, dos_hist


def plot_bands(ax, dist, bands, labels, lpos, color, title):
    """Helper to plot a band structure panel."""
    for i in range(bands.shape[1]):
        ax.plot(dist, bands[:, i], color=color, linewidth=0.9, alpha=0.8)
    ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
    ax.set_ylabel('E - E_F (eV)')
    ax.set_title(title)
    ax.set_xticks(lpos)
    ax.set_xticklabels([('$\\Gamma$' if l == 'G' else l) for l in labels])
    ax.set_ylim([-8, 5])
    ax.grid(True, alpha=0.3, axis='y')
    for p in lpos:
        ax.axvline(p, color='k', linewidth=0.5, alpha=0.3)


def main():
    # Load parameters
    ne, H0, Sdict, Vdict = read_bethe_params('Au')
    vecs = gen_fcc_111_neighbors()
    Vlist = [construct_mat(Vdict, vec) for vec in vecs]
    Slist = [construct_mat(Sdict, vec) for vec in vecs]
    b1, b2, b3 = compute_reciprocal_vectors(vecs)
    k_special = compute_kpoints_111(vecs)

    soc_p, soc_d = read_soc_params('AuSOC')
    print(f"SOC parameters (eV): lambda_p = {soc_p:.3f}, lambda_d = {soc_d:.3f}")

    Hsoc = build_Hsoc(soc_p, soc_d)

    # Args tuples -- layout: (H0, Vlist, Slist, [Hsoc,] vecs, b1, b2, b3)
    scalar_args = (H0, Vlist, Slist, vecs, b1, b2, b3)
    soc_args = (H0, Vlist, Slist, Hsoc, vecs, b1, b2, b3)

    def eig_scalar(*a):
        return get_eigenvalues_scalar(a[0], a[1], a[2], a[3], a[7], a[4], a[5], a[6])
    def eig_soc(*a):
        return get_eigenvalues_soc(a[0], a[1], a[2], a[3], a[4], a[8], a[5], a[6], a[7])

    # Fermi levels (scalar: ne/2 per spin, SOC: ne total in doubled basis)
    print("\nFinding Fermi levels...")
    Ef_scalar = find_fermi(eig_scalar, scalar_args, ne / 2)
    Ef_soc = find_fermi(eig_soc, soc_args, ne)
    print(f"  Scalar: {Ef_scalar:.3f} eV")
    print(f"  SOC:    {Ef_soc:.3f} eV")

    # Band structures
    print("Computing band structures...")
    dist_s, bands_s, labels, lpos = band_structure(eig_scalar, scalar_args, Ef_scalar, k_special)
    dist_soc, bands_soc, _, _ = band_structure(eig_soc, soc_args, Ef_soc, k_special)

    # DOS
    print("Computing DOS...")
    dos_e_s, dos_s = calculate_dos(eig_scalar, scalar_args, Ef_scalar)
    dos_e_soc, dos_soc = calculate_dos(eig_soc, soc_args, Ef_soc)

    # Gamma eigenvalues for table
    k_G = k_special['G']
    eG_s = get_eigenvalues_scalar(H0, Vlist, Slist, vecs, k_G, b1, b2, b3) - Ef_scalar
    eG_soc = get_eigenvalues_soc(H0, Vlist, Slist, Hsoc, vecs, k_G, b1, b2, b3) - Ef_soc

    # -- Plotting: 2 rows x 2 cols --
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: Scalar band structure
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bands(ax1, dist_s, bands_s, labels, lpos, 'blue', 'Scalar (no SOC) - 9 bands')

    # Top-right: SOC band structure
    ax2 = fig.add_subplot(gs[0, 1])
    plot_bands(ax2, dist_soc, bands_soc, labels, lpos, 'red',
               f'SOC (lp={soc_p:.1f}, ld={soc_d:.2f} eV) - 18 bands')

    # Bottom-left: Overlay
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(bands_s.shape[1]):
        lbl = 'Scalar' if i == 0 else None
        ax3.plot(dist_s, bands_s[:, i], 'b-', linewidth=1.5, alpha=0.4, label=lbl)
    for i in range(bands_soc.shape[1]):
        lbl = 'SOC' if i == 0 else None
        ax3.plot(dist_soc, bands_soc[:, i], 'r-', linewidth=0.7, alpha=0.8, label=lbl)
    ax3.axhline(0, color='k', linestyle=':', linewidth=0.8)
    ax3.set_ylabel('E - E_F (eV)')
    ax3.set_title('Overlay: Scalar vs SOC')
    ax3.set_xticks(lpos)
    ax3.set_xticklabels([('$\\Gamma$' if l == 'G' else l) for l in labels])
    ax3.set_ylim([-8, 5])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    for p in lpos:
        ax3.axvline(p, color='k', linewidth=0.5, alpha=0.3)

    # Bottom-right: DOS + eigenvalue summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(dos_e_s, 0, dos_s, alpha=0.2, color='blue')
    ax4.plot(dos_e_s, dos_s, 'b-', linewidth=1.5, label='Scalar')
    ax4.fill_between(dos_e_soc, 0, dos_soc, alpha=0.15, color='red')
    ax4.plot(dos_e_soc, dos_soc, 'r-', linewidth=1.5, label='SOC')
    ax4.axvline(0, color='k', linestyle=':', linewidth=1)
    ax4.set_xlabel('E - E_F (eV)')
    ax4.set_ylabel('DOS (arb. units)')
    ax4.set_title('DOS comparison')
    ax4.set_xlim([-8, 5])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Au SOC Parameter Benchmark', fontsize=14, fontweight='bold')
    plt.savefig('au_soc_benchmark.png', dpi=200, bbox_inches='tight')
    print("Saved: au_soc_benchmark.png")

    # Print Gamma-point eigenvalue table
    print("\nGamma-point eigenvalues (eV, rel. to E_F)")
    print("-" * 40)
    print(f"{'Scalar':>10s}  {'SOC (pair)':>14s}")
    print("-" * 40)
    for i in range(9):
        s_val = f"{eG_s[i]:+.2f}"
        soc_a = f"{eG_soc[2*i]:+.2f}"
        soc_b = f"{eG_soc[2*i+1]:+.2f}"
        print(f"{s_val:>10s}  {soc_a:>6s}, {soc_b:>6s}")
    print("-" * 40)
    print(f"lambda_p: {soc_p:.3f} eV")
    print(f"lambda_d: {soc_d:.3f} eV")
    print(f"d-band SOC splitting ~ {soc_d*5/2:.2f} eV (exp. ~ 1.5 eV)")

    # L.S matrix verification
    print("\n--- L.S matrix verification ---")
    Hsoc_unit = build_Hsoc(1.0, 1.0)
    print(f"  Hermitian: {np.allclose(Hsoc_unit, Hsoc_unit.conj().T)}")
    print(f"  s-block zero: {np.allclose(Hsoc_unit[:2, :2], 0)}")
    p_eigs = np.sort(np.linalg.eigvalsh(Hsoc_unit[2:8, 2:8]))
    d_eigs = np.sort(np.linalg.eigvalsh(Hsoc_unit[8:18, 8:18]))
    print(f"  p-block eigenvalues: {np.array2string(p_eigs, precision=3)}")
    print(f"    (expect: -1, -1, +1/2 x4)")
    print(f"  d-block eigenvalues: {np.array2string(d_eigs, precision=3)}")
    print(f"    (expect: -3/2 x4, +1 x6)")


if __name__ == '__main__':
    main()
