# Phase III Source Inventory

Date: 2026-05-11
Source: gauNEGF/ contact + surfG infrastructure

## Contact entry points

### NEGF.setSigma (gauNEGF.scf)

Signature: `setSigma(self, lContact=None, rContact=None, sig=-0.1j, sig2=None)`
Instantiates: None (creates static self-energy matrices only; use for energy-independent contacts)
Config consumed: None (uses contact orbital indices from self.locs)
Attributes set: self.sigma1, self.sigma2, self.sigma12, self.Gam1, self.Gam2, self.lInd, self.rInd
When to use: Set constant (energy-independent) contact self-energies for fast calculations without energy integration.
Physical scenario: Energy-independent self-energies; handles scalar, vector, or matrix inputs with automatic spin expansion.

### NEGF.setContacts (gauNEGF.scf)

Signature: `setContacts(self, lContact=None, rContact=None)`
Instantiates: None (utility method, no class instantiation)
Config consumed: None
Attributes set: self.lContact, self.rContact, self.lInd, self.rInd, self.nelecContacts
When to use: Identify and store orbital indices for left and right contact atoms before setting sigma or voltage.
Physical scenario: Maps atom numbers to orbital indices in current basis; required before setSigma, setVoltage, or NEGFE contact setup.

### NEGFE.setSigma (gauNEGF.scfE)

Signature: `setSigma(self, lContact=None, rContact=None, sig=-0.1j, sig2=None, T=TEMPERATURE)`
Instantiates: surfGTest (energy-independent test object wrapping constant sigma)
Config consumed: TEMPERATURE (from gauNEGF.config)
Attributes set: self.g (surfGTest), self.T, self.lInd, self.rInd
When to use: Set constant self-energies for NEGFE with explicit temperature control; inherits NEGF.setSigma then wraps in surfGTest.
Physical scenario: Temperature-dependent constant self-energies; calls parent NEGF.setSigma then wraps in surfGTest interface for compatibility with energy-dependent density integration.

### NEGFE.setContact1D (gauNEGF.scfE)

Signature: `setContact1D(self, contactList, tauList=None, stauList=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, neList=None, muList=None, eta=ETA, T=TEMPERATURE, symmetrize_contacts=None)`
Instantiates: surfG (1D chain contact object)
Config consumed: ETA, TEMPERATURE, ADAPTIVE_INTEGRATION_TOL, FERMI_CALCULATION_TOL (from gauNEGF.config)
Attributes set: self.g (surfG), self.T, self.lInd, self.rInd, self._symmetrize_contacts
When to use: Set energy-dependent 1D quasi-infinite chain contacts with full frequency-dependent self-energy; supports three usage patterns (automatic extraction, custom coupling, fully specified).
Physical scenario: Semi-infinite 1D chain electrodes; supports orthogonal/non-orthogonal overlaps and optional contact symmetrization for identical material contacts.

### NEGFE.setContactBethe (gauNEGF.scfE)

Signature: `setContactBethe(self, contactList, latFile='Au', eta=ETA, T=TEMPERATURE)`
Instantiates: surfGB (Bethe lattice contact object)
Config consumed: ETA, TEMPERATURE (from gauNEGF.config)
Attributes set: self.g (surfGB), self.T, self.lInd, self.rInd
When to use: Set energy-dependent Bethe lattice contacts for FCC [111] metallic electrodes with Slater-Koster parameters.
Physical scenario: Semi-infinite metallic contacts modeled as Bethe lattice with FCC [111] surface; uses orbital-dependent hopping and overlap from Slater-Koster files.

## surfG class interfaces

### surfG (gauNEGF.surfG1D)

Constructor: `surfG(Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA, spin='r')`
Physical scenario: Semi-infinite 1D chain contact; supports three setup patterns: (a) automatic extraction from Fock/Overlap, (b) custom coupling with automatic onsite, (c) fully specified contact parameters.
SOC/spin handling: Single spin value for all orbitals (spin='r' assumed); no explicit SOC but architecture permits extension.
Key methods:
- `sigma(E, i, conv=...)`: Returns self-energy matrix for contact i at energy E via recursive surface Green's function.
- `sigmaTot(E, conv=...)`: Returns total self-energy (sum of left and right contact contributions).
- `crossTermQ(E, i, conv=...)`: Returns overlap-coupling Q matrix for contact i (non-orthogonal overlaps only).
- `crossTermQTot(E, conv=...)`: Returns total overlap-coupling Q matrix.
- `setF(F, mu1, mu2)`: Update Fock matrix and chemical potentials for contacts.
- `g(E, i, conv=..., relFactor=...)`: Compute surface Green's function for contact i using iterative recursion.
- `_setContacts(...)`: Internal: extract or set onsite/hopping matrices.
- `_regularizeContacts()`: Enforce positive-semi-definite overlap via congruent eigenvalue clipping.
- `_rejit()`: Internal: JIT-compile surface Green's function methods for each contact index.

### surfG3 (gauNEGF.surfG3D)

Constructor: `surfG3(F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE)`
Physical scenario: 3D k-grid lattice contact with FCC [111] surface; uses bar object to extract orbital information and reads Slater-Koster parameters.
SOC/spin handling: Spin-independent implementation (spin='r' assumed); adds degenerate spin terms during sigma generation for 'u'/'ro'/'g'.
Key methods:
- `sigma(E, i, conv=...)`: Return self-energy for contact i at energy E via atomic surface Green's function.
- `sigmaTot(E, conv=...)`: Total self-energy from all contacts.
- `crossTermQ(E, i, conv=...)`: Overlap-coupling Q matrix for contact i.
- `crossTermQTot(E, conv=...)`: Total Q matrix.
- `setF(F, muL, muR)`: Update Fock and chemical potentials.
- `getSigma(Elist=[...], conv=...)`: Pre-compute self-energies on energy grid.
- `updateFermi(i, Ef)`: Update Fermi level for contact i.
- `genNeighbors(plane_normal, first_neighbor)`: Generate 12 nearest neighbor vectors for FCC [111].
- `readBetheParams(filename)`: Load Slater-Koster parameters from file.
- `constructMat(Mdict, dirCosines, SOC=False)`: Build hopping/overlap matrices in arbitrary direction.
- (test methods: `testDOrbitalFunctions`, `testDOrbitalSymmetry`, `testPDInteraction`, `testDDInteraction`, `testHoppingPhysics`, `runAllTests`)

### surfGAt3D (gauNEGF.surfG3D)

Constructor: `surfGAt3D(H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False)`
Physical scenario: Atomic surface Green's function for 3D k-grid; maintains Fermi level and self-energy across energy range.
SOC/spin handling: Explicit SOC parameter; no explicit spin expansion (spin='r' assumed in surfG3 wrapper).
Key methods:
- `sigma(E, i, conv=...)`: Contact-specific self-energy at energy E.
- `sigmaTot(E, conv=...)`: Total self-energy.
- `crossTermQ(E, i, conv=...)`: Overlap Q for contact i.
- `crossTermQTot(E, conv=...)`: Total Q.
- `setF(F, mu1, mu2)`: Update Fock and potentials.
- `sigmaK(E, conv=..., mix=...)`: Surface Green's function in k-space with mixing.
- `sigmaSurf(E, conv=..., mix=...)`: Surface contribution to self-energy.
- `crossTermQSurf(E, sigInds=None, conv=..., mix=...)`: Surface Q contribution.
- `crossTermQBulk(E, conv=..., mix=...)`: Bulk Q contribution.
- `updateH(fermi=None)`: Update Hamiltonian based on Fermi level.
- `DOS(E)`: Density of states at energy E.
- `calcFermi(ne, tol=...)`: Find Fermi level for ne electrons.

### surfGB (gauNEGF.surfGBethe)

Constructor: `surfGB(F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE)`
Physical scenario: Bethe lattice contact for FCC [111] metallic surface; uses bar object to extract orbital information.
SOC/spin handling: Spin-independent implementation; adds degenerate spin terms during sigma generation.
Key methods:
- `sigma(E, i, conv=...)`: Self-energy for contact i at energy E via atomic Bethe surface Green's function.
- `sigmaTot(E, conv=...)`: Total self-energy from all contacts.
- `crossTermQ(E, i, conv=...)`: Overlap-coupling Q matrix for contact i.
- `crossTermQTot(E, conv=...)`: Total Q.
- `setF(F, muL, muR)`: Update Fock and chemical potentials.
- `getSigma(Elist=[...], conv=...)`: Pre-compute self-energies on grid.
- `updateFermi(i, Ef)`: Update Fermi level for contact i.
- `genNeighbors(plane_normal, first_neighbor)`: Generate 12 FCC [111] nearest neighbor directions.
- `readBetheParams(filename)`: Load Slater-Koster hopping/overlap parameters.
- `constructMat(Mdict, dirCosines, SOC=False)`: Build hopping/overlap in arbitrary direction.
- (test methods: `testDOrbitalFunctions`, `testDOrbitalSymmetry`, `testPDInteraction`, `testDDInteraction`, `testHoppingPhysics`, `runAllTests`)

### surfGBAt (gauNEGF.surfGBethe)

Constructor: `surfGBAt(H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False)`
Physical scenario: Atomic surface Green's function for Bethe lattice; manages recursion and Fermi level across energy range.
SOC/spin handling: Explicit SOC parameter; spin='r' assumed.
Key methods:
- `sigma(E, i, conv=...)`: Contact-specific self-energy.
- `sigmaTot(E, conv=...)`: Total self-energy.
- `crossTermQ(E, i, conv=...)`: Contact-specific Q.
- `crossTermQTot(E, conv=...)`: Total Q.
- `setF(F, mu1, mu2)`: Update Fock and potentials.
- `sigmaK(E, conv=..., mix=...)`: Bethe lattice surface Green's function with mixing.
- `sigmaSurf(E, conv=..., mix=...)`: Surface component.
- `crossTermQSurf(E, sigInds=None, conv=..., mix=...)`: Surface Q.
- `crossTermQBulk(E, conv=..., mix=...)`: Bulk Q.
- `updateH(fermi=None)`: Update internal Hamiltonian.
- `DOS(E)`: Density of states.
- `calcFermi(ne, tol=...)`: Find Fermi level for given electron count.

### SigmaCalculator (gauNEGF.transport)

Constructor: `SigmaCalculator(sig1, sig2=None, energy_dependent=None)`
Physical scenario: Unified dispatcher for energy-dependent (surfG-type) or energy-independent (static matrix) self-energies; auto-detects interface via duck typing.
SOC/spin handling: Expands static matrices for 'u'/'ro'/'g' spin configurations; energy-dependent objects handle internally.
Key methods:
- `get_sigma_total(E, spin=None, matrix_size=None)`: Total self-energy at energy E with automatic spin expansion.
- `get_Q_tot(E, spin=None, matrix_size=None)`: Total overlap-coupling Q at energy E (None if energy-independent or no coupling).
- `get_sigma(E, contact_index, spin=None, matrix_size=None)`: Contact-specific self-energy.
- `get_gamma(E, contact_index, spin=None, matrix_size=None)`: Gamma matrix (broadening) = i*(Sigma - Sigma^dagger).

### surfGTest (gauNEGF.surfGTester)

Constructor: `surfGTest(Fock, Overlap, indsList, sig1=None, sig2=None, spin='r')`
Physical scenario: Energy-independent (constant) self-energy object for testing or production use; wraps static matrices in surfG-compatible interface.
SOC/spin handling: No explicit spin; static matrices only; interface compatible with all spin modes.
Key methods:
- `sigma(E, i, conv=...)`: Return constant self-energy matrix for contact i (ignores E, conv).
- `sigmaTot(E, conv=...)`: Sum of all contact self-energies (constant, ignores E).
- `setF(F, mu1, mu2)`: Update Fock matrix (ignores mu1, mu2 since sigma is constant).
- `crossTermQ(E, i, conv=...)`: Always None (no overlap coupling for constant sigma).
- `crossTermQTot(E, conv=...)`: Always None.

## Cross-reference table

| Entry point | Module | surfG class used | Config constants consumed | Use case |
|-------------|--------|-----------------|--------------------------|----------|
| NEGF.setSigma | scf | None | None | Static energy-independent self-energy |
| NEGF.setContacts | scf | None | None | Identify contact orbital indices |
| NEGFE.setSigma | scfE | surfGTest | TEMPERATURE | Constant self-energy with temperature |
| NEGFE.setContact1D | scfE | surfG | ETA, TEMPERATURE, ADAPTIVE_INTEGRATION_TOL, FERMI_CALCULATION_TOL | 1D chain contact with energy dependence |
| NEGFE.setContactBethe | scfE | surfGB | ETA, TEMPERATURE | Bethe lattice contact with energy dependence |
| SigmaCalculator.__init__ | transport | surfG, surfGB, surfGTest (duck-typed) | None | Dispatch to correct sigma interface |

## Decision tree (plain English)

**Choosing a contact setup method:**

1. Do you want energy-dependent self-energies (frequency-dependent coupling)?
   - **No** (constant self-energy): Use `NEGF.setSigma()` or `NEGFE.setSigma()` with `sig=` scalar/vector/matrix.
     - Just constant sigma? Use `NEGF.setSigma()` directly.
     - Want temperature effects? Use `NEGFE.setSigma()` with temperature parameter.
     - Return object is static matrices (sigma1, sigma2, Gam1, Gam2).
   
   - **Yes** (frequency-dependent): Use `NEGFE.setContact1D()` or `NEGFE.setContactBethe()`.
     - Semi-infinite 1D chain electrode? Use `NEGFE.setContact1D()`.
       - Automatic extraction from Fock/Overlap: pass only `contactList` and connection indices in `tauList`.
       - Custom coupling matrices: pass `tauList=[tau1, tau2]` and optional `stauList=[stau1, stau2]`.
       - Fully specified contact parameters: provide `alphas`, `aOverlaps`, `betas`, `bOverlaps`.
     - Metallic contact with Slater-Koster parameters? Use `NEGFE.setContactBethe()`.
       - Provide atom list, Slater-Koster file name (default 'Au'), and temperature.
       - Returns `surfGB` object for energy integration; use in transport calculations.
     - Return object is surfG or surfGB (with sigma(E, i), sigmaTot(E), crossTermQ methods).

2. After setting contacts, do you need to compute transport properties (transmission, DOS, current)?
   - **Yes**: Wrap contact object in `SigmaCalculator()` for unified interface.
     - Automatically detects energy-dependent (surfG/surfGB) vs energy-independent (static sigma).
     - Use `SigmaCalculator.get_sigma_total(E)`, `get_gamma(E, contact_idx)` for transport functions.
   - **No**: Use contact object directly for self-energy queries via `sigma()`, `sigmaTot()`.

3. Spin configuration:
   - **Restricted ('r')**: All contact methods handle automatically.
   - **Unrestricted ('u'), restricted open ('ro'), or generalized ('g')**: 
     - Static sigma (NEGF.setSigma): Pass sigma as full matrix or use kronecker expansion.
     - Energy-dependent (NEGFE methods): Pass spin parameter; surfG classes expand internally.
     - Transport: SigmaCalculator.get_sigma_total() auto-expands for spin-resolved calculations.

4. Temperature effects:
   - **Zero temperature**: Use any method; most default to T=0.
   - **Finite temperature**: Use NEGFE methods (setContact1D, setContactBethe, setSigma) with T parameter; affects contact Fermi level and occupation.

## Summary of entry point and surfG class relationships

**Total contact setup entry points:** 5 (NEGF.setSigma, NEGF.setContacts, NEGFE.setSigma, NEGFE.setContact1D, NEGFE.setContactBethe)

**Total surfG-compatible classes:** 6 (surfG, surfG3, surfGAt3D, surfGB, surfGBAt, surfGTest; plus SigmaCalculator as unified dispatcher)

**Decision tree shape:** Binary fork at energy dependence, then three branches for static/1D/Bethe, then convergence to SigmaCalculator for transport.

**Key findings from source code:**

1. **Duck-typed overload:** SigmaCalculator auto-detects energy dependence via `hasattr(sig1, 'sigma') and hasattr(sig1, 'sigmaTot')` rather than explicit type checking; any object implementing this interface (surfG, surfGB, surfGTest, surfG3, surfGAt3D, surfGBAt) is compatible.

2. **Three surfG1D patterns:** surfG constructor accepts minimal indices (pattern a), coupling matrices (pattern b), or full specifications (pattern c); controlled by argument presence and shape detection (1D vs 2D array for taus).

3. **Spin implementation splits:** surfG1D implements explicit spin expansion ('r', 'u', 'ro', 'g'); surfG3/surfGB/surfGBAt are spin-independent internally, add degenerate terms during sigma generation, require explicit handling in transport layer.

4. **Contact overlap handling:** surfG and NEGF.setSigma both implement congruent eigenvalue clipping (_regularizeContacts) to enforce positive-semi-definite infinite-chain overlap; no downstream sigma correction needed.

5. **NEGFE inheritance:** NEGFE inherits NEGF.__init__ without explicit __init__ override; NEGFE methods call super() to reuse setContacts and setSigma, then wrap result in energy-dependent objects.

6. **Cross-term Q coupling:** Not all contacts support crossTermQ; surfGTest always returns None (orthogonal basis); surfG returns Q only if staus is non-None (non-orthogonal coupling); surfG3/surfGB/surfGBAt compute Q from orbital overlap structure.

7. **Bethe lattice coordinate handling:** Both surfG3 and surfGB compute FCC [111] surface normal via SVD of contact atom positions, then generate 12 nearest-neighbor vectors via Rodrigues rotation; latVec orientation auto-corrects via dot product check.

