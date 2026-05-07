# Setting Up a System with 1D Contacts in gauNEGF

This guide walks through how to set up an NEGF transport calculation using
energy-dependent 1D chain contacts via the `NEGFE` class and `surfG` surface
Green's function.

## Overview

The workflow has three stages:

1. Run a large DFT cluster calculation in Gaussian to get bulk-like matrices
2. Extract the Fock, overlap, and coupling matrices for the device and contacts
3. Set up the `NEGFE` object and run the self-consistent field (SCF) loop

---

## 1. Prepare Gaussian Input Files

You need two Gaussian input files (`.gjf`):

### a) Large cluster (for matrix extraction)

A cluster with many unit cells (e.g., 11) so that the interior cells are
bulk-like. This is used *only* to extract initial Fock/overlap matrices and
coupling parameters. Hydrogen caps at the edges are fine -- they will be
discarded during extraction.

Example: `CNT33_11_minbasis.gjf` -- an 11-cell (3,3) armchair CNT with STO-3G.

### b) Device cluster (for SCF)

A smaller cluster matching the device region you will simulate. This file
defines the geometry that Gaussian uses during the SCF loop. It should:

- Contain exactly the atoms in your device (contact cells + interior cells)
- Use the same basis set as the large cluster
- Have NO hydrogen caps (the semi-infinite leads replace them)
- Have coordinates taken from the interior of the large cluster

Example: `CNTPeriodic5.gjf` -- 5 cells extracted from the interior of the
11-cell system, STO-3G, no caps.

---

## 2. Understand the Unit Cell Structure

For a periodic 1D system, identify:

- **Atoms per unit cell** (`CPerLayer`): e.g., 12 for a (3,3) armchair CNT
- **Orbitals per atom** (`Corbs`): e.g., 5 for carbon with STO-3G (1s, 2s, 2px, 2py, 2pz)
- **Orbitals per unit cell** (`orbsPerCell = Corbs * CPerLayer`): e.g., 60

The device is partitioned as:

```
|  Left   |         Device          |  Right  |
| Contact |  Buffer ... Buffer      | Contact |
| 1 cell  |  (N-2) cells            | 1 cell  |
```

The contact cells define the repeating unit of the semi-infinite leads.
Buffer cells screen the contact self-energies from the active region.

**Recommendation**: Use at least 5 total cells (1 contact + 3 device + 1
contact). More buffer cells improve accuracy.

---

## 3. Extract Matrices from the Large Cluster

```python
import numpy as np
from scipy.linalg import fractional_matrix_power
from gauopen import QCBinAr as qcb

har_to_eV = 27.211386
Corbs = 5
CPerLayer = 12
orbsPerCell = Corbs * CPerLayer  # 60

# Run Gaussian on the large cluster
bar = qcb.BinAr(debug=False, lenint=8, inputfile="CNT33_11_minbasis.gjf")
bar.update(model='b3lyp', basis='STO-3G', toutput='out.log',
           chkname="CNT33_11_minbasis.chk", dofock=True)

S_full = np.array(bar.matlist['OVERLAP'].expand())
P_full = np.array(bar.matlist['ALPHA SCF DENSITY MATRIX'].expand())
F_full = np.array(bar.matlist['ALPHA FOCK MATRIX'].expand()) * har_to_eV
```

### a) Extract the device block

Choose the most central cells. For a 10-cell system extracting 5 cells:

```python
nCells = 5
startCell = 3  # cells 3,4,5,6,7

ind1 = startCell * orbsPerCell            # 180
ind2 = (startCell + nCells) * orbsPerCell  # 480

F = F_full[ind1:ind2, ind1:ind2]
S = S_full[ind1:ind2, ind1:ind2]
```

### b) Count electrons

```python
PS_full = P_full @ S_full
ne = np.trace(PS_full[ind1:ind2, ind1:ind2]).real
```

This gives the Mulliken electron count in the device block from the
full DFT calculation. Divide by `nCells` to get the electron count per
unit cell.

### c) Extract coupling matrices from the deep interior

Use the two most central cells of the large cluster for the most bulk-like
coupling. These define the semi-infinite 1D chain:

```python
# Interior cells 4 and 5 (deep inside the 10-cell cluster)
ic1 = np.arange(4 * orbsPerCell, 5 * orbsPerCell)
ic2 = np.arange(5 * orbsPerCell, 6 * orbsPerCell)

tau_F = F_full[np.ix_(ic1, ic2)]     # inter-cell coupling (Fock)
tau_S = S_full[np.ix_(ic1, ic2)]     # inter-cell coupling (overlap)
alpha_F = F_full[np.ix_(ic2, ic2)]   # on-site energy (Fock)
alpha_S = S_full[np.ix_(ic2, ic2)]   # on-site overlap
```

**Key definitions:**

| Matrix | Meaning | Size |
|--------|---------|------|
| `alpha` | On-site Fock for one bulk unit cell | orbsPerCell x orbsPerCell |
| `alpha_S` | On-site overlap for one bulk unit cell | orbsPerCell x orbsPerCell |
| `tau` | Coupling Fock from cell n to cell n+1 | orbsPerCell x orbsPerCell |
| `tau_S` | Coupling overlap from cell n to cell n+1 | orbsPerCell x orbsPerCell |
| `beta` | Hopping between bulk cells (= tau for periodic) | orbsPerCell x orbsPerCell |

For a periodic system, `beta = tau` because every inter-cell coupling is
identical.

---

## 4. Set Up NEGFE

```python
from gauNEGF.scfE import NEGFE

negf = NEGFE(fn='CNTPeriodic5', func='b3lyp', basis='STO-3G', fullSCF=False)
negf.setFock(F)
```

- `fn`: stem of the device `.gjf` file (no extension)
- `fullSCF=False`: use Harris guess for initial DFT (faster)
- `setFock(F)`: override the initial Fock with the one extracted from the
  large cluster

---

## 5. Set Up 1D Contacts

```python
# Atom indices (1-indexed) for left and right contact cells
leftAtoms  = np.arange(CPerLayer) + 1                       # [1, ..., 12]
rightAtoms = np.arange((nCells-1)*CPerLayer, nCells*CPerLayer) + 1  # [49, ..., 60]

inds = negf.setContact1D(
    [leftAtoms, rightAtoms],          # contact atom indices
    [tau_F, tau_F.conj().T],          # tau (L: left->right, R: right->left)
    [tau_S, tau_S.conj().T],          # stau (overlap coupling)
    [alpha_F, alpha_F],               # alpha (on-site Fock, same for both)
    [alpha_S, alpha_S],               # aOverlap (on-site overlap)
    [tau_F, tau_F.conj().T],          # beta = tau for periodic systems
    [tau_S, tau_S.conj().T],          # bOverlap
    neList=[ne/nCells, ne/nCells],    # electrons per unit cell per contact
    symmetrize_contacts=True,         # True when both contacts are same material
    eta=1e-3                          # broadening (eV)
)
```

### Argument details

**contactList** `[leftAtoms, rightAtoms]`

1-indexed atom numbers within the device `.gjf` file. The left contact is the
first unit cell, the right contact is the last.

**tauList** `[tau_L, tau_R]`

The coupling matrices between the contact surface and the adjacent device cell.
For the left contact, tau goes left-to-right; for the right contact, it goes
right-to-left. In a symmetric periodic system: `tau_R = tau_L.conj().T`.

**stauList** `[stau_L, stau_R]`

Overlap matrices corresponding to the coupling. Same symmetry as tau. If your
system uses an orthogonal basis, pass `None` instead.

**alphas** `[alpha_L, alpha_R]`

On-site Fock matrix for one bulk unit cell of each contact. For same-material
contacts, these are identical.

**aOverlaps** `[salpha_L, salpha_R]`

On-site overlap for each contact's unit cell.

**betas** `[beta_L, beta_R]`

Hopping Fock between adjacent bulk unit cells in the lead chain. For periodic
systems, `beta = tau`.

**bOverlaps** `[sbeta_L, sbeta_R]`

Hopping overlap. Same symmetry as beta/tau.

**neList** `[ne_L, ne_R]`

Number of electrons per unit cell for each contact. Used to compute the contact
Fermi level via `getFermiContact`. For a periodic system, this is
`total_electrons / nCells`.

**symmetrize_contacts** `True/False`

Set `True` when both contacts are the same material (e.g., periodic nanotube,
nanowire). This averages the on-site Fock blocks during SCF to prevent
artificial symmetry breaking from the opposite directional coupling signs in
sigma_L vs sigma_R.

**eta**

Broadening parameter in eV. Larger values (1e-3) speed up surface Green's
function convergence but smear sharp features. Smaller values (1e-5 to 1e-9)
are more accurate but may require more iterations.

---

## 6. Run the SCF

```python
# Set voltage and initial Fermi level from contact calculation
negf.setVoltage(0.0, negf.g.fermiList[0])

# Run SCF: tolerance, damping, max iterations
negf.SCF(1e-3, 0.02, 1000)

# Save results
negf.saveMAT('output.mat')
```

### SCF parameters

- **tolerance** (1e-3): convergence criterion on the density matrix change
- **damping** (0.02): linear mixing parameter -- smaller is more stable but
  slower. Range: 0.01 to 0.1.
- **max iterations** (1000): safety cap

---

## 7. Post-SCF: Transmission

```python
from gauNEGF.transport import cohTransE
from scipy import io

Elist = np.linspace(-5, 5, 500)
T = cohTransE(Elist + negf.fermi, negf.F * har_to_eV, negf.S, negf.g)
io.savemat('transmission.mat', {'Elist': Elist, 'fermi': negf.fermi, 'T': T})
```

---

## Common Pitfalls

1. **Too few cells**: Using only 3 cells (1+1+1) gives no buffer between
   contacts. The self-energies overlap on the single device cell. Use at
   least 5 cells.

2. **Edge-contaminated coupling**: Extracting tau/alpha from the edge of the
   large cluster gives non-bulk-like values. Always extract from the deep
   interior.

3. **Mismatched geometries**: The device `.gjf` must have coordinates taken
   from the same large cluster used for matrix extraction. Do not mix
   independently generated geometries.

4. **Wrong basis in device .gjf**: The NEGFE constructor overrides the basis
   via the `basis` argument, but using a checkpoint from a different basis
   can cause issues. Delete stale `.chk` files when changing basis sets.

5. **Forgetting .conj().T for right contact**: In a periodic system, the
   right contact coupling is the Hermitian conjugate of the left. Passing
   the same matrix for both creates an asymmetric system.

6. **H caps in device .gjf**: The device file should have bare dangling bonds
   at the edges. The semi-infinite leads replace the caps. Including H atoms
   adds spurious states.
