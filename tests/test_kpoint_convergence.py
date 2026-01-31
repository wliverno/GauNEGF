import numpy as np
import matplotlib.pyplot as plt
from gauNEGF.surfG3D import surfGAt3D
import jax.numpy as jnp

# Read Bethe lattice parameters for Au
params = {}
with open('Au.bethe', 'r') as f:
    for line in f:
        if not line.strip():
            continue
        line = line.replace(' ', '')
        key, value = line.split('=')
        params[key] = float(value)

# Convert Hartrees to eV
har_to_eV = 27.211386

# Build onsite Hamiltonian
Hdiag = jnp.array([params['es']] + [params['ep']]*3 +
                  [params['edd']] + [params['edt']]*2 + [params['edd'], params['edt']]) * har_to_eV
H0 = jnp.diag(Hdiag)

# Build hopping and overlap parameter dictionaries
Edict = {k[1:]: params[k]*har_to_eV for k in params if k.startswith('e')}
Sdict = {k[1:]: params[k] for k in params if k.startswith('S')}
Vdict = {k: params[k]*har_to_eV for k in params if not k.startswith('e') and not k.startswith('S')}

# Define 12 nearest neighbor directions for FCC [111] surface
# 6 in-plane hexagonal neighbors
in_plane = [
    np.array([1, 0, 0]),
    np.array([0.5, np.sqrt(3)/2, 0]),
    np.array([-0.5, np.sqrt(3)/2, 0]),
    np.array([-1, 0, 0]),
    np.array([-0.5, -np.sqrt(3)/2, 0]),
    np.array([0.5, -np.sqrt(3)/2, 0])
]

# 6 out-of-plane neighbors (3 up + 3 down)
out_angle = np.arccos(1/np.sqrt(3))
out_of_plane = []
for i in range(3):
    angle = i * 2 * np.pi / 3 + np.pi/6
    base = np.array([np.cos(angle), np.sin(angle), 0])
    out_of_plane.append(np.cos(out_angle) * base + np.sin(out_angle) * np.array([0, 0, 1]))
for i in range(3):
    out_of_plane.append(-out_of_plane[i])

vecs = in_plane + out_of_plane

# Helper function to construct hopping/overlap matrices
def constructMat(Mdict, dirCosines):
    dim = 9
    M = jnp.zeros((dim, dim))

    # s-s
    M = M.at[0,0].set(Mdict['sss'])

    # s-p block
    M = M.at[0,3].set(Mdict['sps'])
    M = M.at[3,0].set(-Mdict['sps'])

    # p-p block
    M = M.at[1,1].set(Mdict['ppp'])
    M = M.at[2,2].set(Mdict['ppp'])
    M = M.at[3,3].set(Mdict['pps'])

    # s-d block
    M = M.at[0,4].set(Mdict['sds'])
    M = M.at[4,0].set(Mdict['sds'])

    # p-d block
    M = M.at[1,5].set(Mdict['pdp'])
    M = M.at[2,6].set(Mdict['pdp'])
    M = M.at[3,4].set(Mdict['pds'])
    M = M.at[5,1].set(-Mdict['pdp'])
    M = M.at[6,2].set(-Mdict['pdp'])
    M = M.at[4,3].set(-Mdict['pds'])

    # d-d block
    M = M.at[4,4].set(Mdict['dds'])
    M = M.at[5,5].set(Mdict['ddp'])
    M = M.at[6,6].set(Mdict['ddp'])
    M = M.at[7,7].set(Mdict['ddd'])
    M = M.at[8,8].set(Mdict['ddd'])

    # Transformation matrix
    tr = jnp.zeros((9, 9))
    x, y, z = dirCosines
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    tr = tr.at[0,0].set(1.0)

    tr = tr.at[1:4,1:4].set(jnp.array([
        [np.cos(theta) * jnp.cos(phi), -np.sin(phi), jnp.sin(theta)*np.cos(phi)],
        [np.cos(theta) * jnp.sin(phi), jnp.cos(phi), np.sin(theta)*np.sin(phi)],
        [-np.sin(theta), 0, jnp.cos(theta)]
    ]))

    d_block = jnp.zeros((5,5))
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

    return tr @ M @ tr.T

# Build hopping and overlap lists
Slist = [constructMat(Sdict, v) for v in vecs]
Vlist = [constructMat(Vdict, v) for v in vecs]

# Test parameters
eta = 1e-4
T = 0
test_energy = 3.249851  # eV (near Fermi level for Au)
energy_samples = test_energy + np.linspace(-0.1, 0.1, 5)  # Sample around test energy
kpoint_values = [1, 3, 4, 5, 6, 7]

print(f"Testing k-point convergence for DOS near E = {test_energy} eV")
print(f"Energy samples: {energy_samples}")
print(f"K-point values: {kpoint_values}\n")

# Store results
dos_results = np.zeros((len(kpoint_values), len(energy_samples)))

for i, nk in enumerate(kpoint_values):
    print(f"Testing with {nk} k-points ({nk**2} total)...")

    # Create surfGAt3D object with specific k-point count
    g = surfGAt3D(H0.copy(), Slist, Vlist, vecs, eta, T, kPoints=nk)

    # Calculate DOS at each energy sample
    for j, E in enumerate(energy_samples):
        dos = g.DOS(E, conv=1e-4, mix=0.1)
        dos_results[i, j] = dos
        print(f"  E = {E:6.3f} eV: DOS = {dos:.6f}")

    print()

# Calculate convergence metrics
print("\nConvergence Analysis:")
print("=" * 60)

for j, E in enumerate(energy_samples):
    dos_values = dos_results[:, j]
    reference_dos = dos_results[-1, j]  # Use highest k-point as reference

    print(f"\nEnergy E = {E:.3f} eV:")
    print(f"Reference DOS ({kpoint_values[-1]} k-points): {reference_dos:.6f}")

    for i, nk in enumerate(kpoint_values[:-1]):
        rel_error = abs(dos_values[i] - reference_dos) / abs(reference_dos) * 100
        print(f"  {nk:2d} k-points: DOS = {dos_values[i]:.6f}, rel. error = {rel_error:5.2f}%")

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: DOS vs k-points for each energy
for j, E in enumerate(energy_samples):
    axes[0].plot(kpoint_values, dos_results[:, j], 'o-', label=f'E = {E:.2f} eV')

axes[0].set_xlabel('Number of k-points per direction')
axes[0].set_ylabel('DOS (states/eV)')
axes[0].set_title('DOS Convergence with k-points')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Relative error vs k-points
for j, E in enumerate(energy_samples):
    reference_dos = dos_results[-1, j]
    rel_errors = np.abs(dos_results[:, j] - reference_dos) / abs(reference_dos) * 100
    axes[1].semilogy(kpoint_values, rel_errors, 'o-', label=f'E = {E:.2f} eV')

axes[1].set_xlabel('Number of k-points per direction')
axes[1].set_ylabel('Relative Error (%)')
axes[1].set_title(f'Convergence Error (ref: {kpoint_values[-1]} k-points)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% error')

plt.tight_layout()
plt.savefig('test_kpoint_convergence.png', dpi=300)
print(f"\nPlot saved as test_kpoint_convergence.png")
plt.show()
