# Benchmark Systems for gauNEGF Validation Against QuantumATK/TranSIESTA

## Context

gauNEGF's NEGF-SCF loop has shown instability (oscillating electron count and
RMSDP) on metallic systems -- gold nanowires and (3,3) armchair CNTs. These are
metallic with high density of states at the Fermi level, causing charge sloshing
during SCF. We need simpler, semiconducting benchmark systems where:

- SCF converges reliably (band gap stabilizes charge)
- Published T(E) reference data exists in the literature
- The system can also be run in QuantumATK and/or TranSIESTA for direct comparison

The primary goal is **transmission validation**: compare T(E) curves at zero
bias to verify the Green's function and self-energy implementation is correct.

---

## System A: Carbon Chain (Polyyne)

### Physical System

Linear carbon chain with alternating bond lengths (polyyne: ...=C-C=C-C=...).
The bond length alternation opens a band gap of ~1-2 eV depending on bond
lengths and functional, making SCF stable.

- **Unit cell:** 2 carbon atoms (short bond ~1.28 A, long bond ~1.32 A)
- **Symmetry:** D_inf_h, strictly 1D
- **Electronic character:** Semiconducting (gap from Peierls distortion)

### Why This System

1. Simplest nontrivial 1D periodic system
2. Appears in the original TranSIESTA paper (Brandbyge et al., PRB 65, 165401, 2002)
3. Common QuantumATK tutorial system
4. Very few orbitals per cell -- fast iteration, easy debugging
5. Band gap ensures SCF stability

### Setup Parameters

| Parameter | STO-3G run | 6-31G* run |
|-----------|-----------|------------|
| Orbs/unit cell | 10 | 28 |
| Large cluster | 21 cells (42 C atoms) | 21 cells |
| Device | 5 cells (10 C atoms) | 5 cells |
| Extract from | Cells 8-12 (interior) | Cells 8-12 |
| ATK/SIESTA basis match | SZP | DZP |
| Functional | B3LYP or PBE | PBE |

**Note on functional:** PBE is preferred for comparison since ATK and TranSIESTA
use PBE by default. B3LYP is Gaussian's default but is a hybrid functional not
available in standard SIESTA. Use PBE for the comparison runs.

### Gaussian Input

Large cluster (21 cells, no caps needed -- just dangling bonds at edges):
- Linear chain of 42 C atoms along z-axis
- Alternating bond lengths: 1.28 A / 1.32 A
- `#p PBE/6-31G* Force NoSymm`

Device cluster (5 cells):
- 10 C atoms extracted from interior of large cluster
- Same basis, no H caps
- Coordinates from cells 8-12 of the large cluster

### Expected T(E)

- Gap in transmission centered at E_F (~1-2 eV wide)
- T(E) = 2 (two pi channels) outside the gap in the first subband
- Step structure at higher energies from additional subbands
- Quantized transmission (integer or near-integer plateaus)

### Reference Data

- Brandbyge et al., PRB 65, 165401 (2002) -- TranSIESTA paper, carbon chain benchmark
- QuantumATK carbon chain tutorial (available in ATK documentation)
- Lang & Avouris, PRL 81, 3515 (1998) -- early carbon chain transport

---

## System B: (8,0) Zigzag Carbon Nanotube

### Physical System

(8,0) zigzag single-wall carbon nanotube. Semiconducting with band gap
~0.55-0.8 eV (depends on functional and basis).

- **Unit cell:** 32 carbon atoms
- **Diameter:** ~6.3 A
- **Translational period:** ~4.26 A along tube axis
- **Electronic character:** Semiconducting (mod(n-m, 3) != 0)

### Why This System

1. Most commonly benchmarked CNT in the NEGF literature
2. Semiconducting -- band gap stabilizes SCF (unlike metallic (3,3))
3. Physically meaningful system (real nanotubes are fabricated)
4. Extensive published T(E) data for validation
5. Tests the code with larger matrix blocks than the carbon chain

### Setup Parameters

| Parameter | STO-3G run | 6-31G* run |
|-----------|-----------|------------|
| Orbs/unit cell | 160 | 448 |
| Large cluster | 11 cells (352 C atoms) | 11 cells |
| Device | 5 cells (160 C atoms) | 5 cells |
| Extract from | Cells 3-7 (interior) | Cells 3-7 |
| ATK/SIESTA basis match | SZP | DZP |
| Functional | PBE | PBE |

**Note:** 6-31G* with 448 orbs/cell will be computationally expensive. Consider
6-31G (no polarization, 288 orbs/cell) as an intermediate step.

### Gaussian Input

Large cluster (11 cells):
- Generate (8,0) CNT coordinates with TubeGen, ASE, or similar tool
- 11 unit cells along the tube axis (352 C atoms)
- No hydrogen caps
- `#p PBE/STO-3G Force NoSymm` (start with minimal basis)

Device cluster (5 cells):
- 160 C atoms from interior of large cluster
- Same basis, no caps

### Expected T(E)

- Clear gap in transmission around E_F (~0.6 eV wide)
- Van Hove singularity steps at subband edges
- T(E) increases in integer steps as each subband opens
- First plateau: T = 2 (doubly degenerate subband)

### Reference Data

- Nardelli, PRB 60, 7828 (1999) -- iterative surface GF method with CNT examples
- Areshkin & White, Nano Lett. 7, 3253 (2007) -- (8,0) CNT transport
- QuantumATK CNT tutorial (available in ATK documentation)
- Many groups have published (8,0) T(E) -- search "zigzag nanotube transport NEGF"

---

## Sequencing Plan

### Phase 1: Carbon chain with STO-3G
- Fastest possible iteration (~10 orbs/cell)
- Validate the end-to-end workflow: Gaussian cluster -> matrix extraction -> NEGFE setup -> T(E)
- Debug any issues with small matrices

### Phase 2: Carbon chain with 6-31G* (PBE)
- Basis-matched comparison with TranSIESTA DZP / ATK DZP
- Generate reference T(E) in ATK or TranSIESTA
- Overlay and compare T(E) curves

### Phase 3: (8,0) CNT with STO-3G
- Scale up to a real nanotube geometry
- Verify SCF convergence is stable (should be, given the band gap)
- Compare T(E) shape with published results

### Phase 4: (8,0) CNT with 6-31G* (PBE)
- Publication-quality benchmark
- Direct comparison with ATK DZP results
- This is the "headline" result for validating gauNEGF

---

## Verification

For each system/basis combination:

1. **Gaussian convergence:** Large cluster DFT must converge. Check total energy and
   forces are reasonable.
2. **Matrix extraction:** Verify that alpha, tau from interior cells are bulk-like
   (compare cells 4-5 vs 5-6 -- should be nearly identical).
3. **Contact Fermi level:** getFermiContact should find a Fermi level inside the
   band gap for semiconducting systems.
4. **SCF convergence:** RMSDP and electron count should decrease monotonically
   (or at least converge, not oscillate).
5. **T(E) shape:** Should show a clear gap, integer plateaus, and match published
   results qualitatively.
6. **Quantitative comparison:** Overlay gauNEGF T(E) with ATK/TranSIESTA T(E) on
   the same energy grid. Differences should be small (< 5-10%) and attributable
   to basis set differences.

---

## Notes

- **Functional choice:** Use PBE throughout for apples-to-apples comparison with
  SIESTA/ATK. Avoid B3LYP for benchmarks since it is not available in standard
  SIESTA.
- **Coordinate generation:** Use ASE (`ase.build.nanotube`) for CNT coordinates.
  For carbon chains, manually specify or use a simple script.
- **Symmetrize contacts:** Set `symmetrize_contacts=True` for both systems since
  left and right contacts are the same material.
- **Eta:** Start with eta=1e-3 for robustness, reduce to 1e-5 for final T(E) plots
  to resolve sharp features.
