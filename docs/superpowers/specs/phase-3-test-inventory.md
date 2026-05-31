# Phase III Test Inventory

Date: 2026-05-11
Source: tests/ + ../NEGFTests/, priority files A-C

## Contact-setup call patterns

### 1D Chain (setContact1D / surfG1D)

| File | System | Key args (verbatim) | Workflow notes |
|------|--------|---------------------|----------------|
| CNT33.py | (3,3) CNT, 5-cell device, 12 atoms/unit cell | setContact1D([leftAtoms, rightAtoms], [tau_F, tau_F.conj().T], [None, None], [alpha_F, alpha_F], [alpha_S, alpha_S], [tau_F, tau_F.conj().T], [tau_S, tau_S.T], neList=[ne/nCells, ne/nCells], symmetrize_contacts=True) | DFT->matrix extraction->setFock->setContact1D with tau/alpha/beta matrices (pattern A: full contact setup). Sets voltage, runs SCF 2x (cold+warm), saves MAT, computes transmission |
| CNTCont.py | CNT (3,3), 2-cell, self-consistent contact | setContact1D([np.arange(CPerLayer)+1, np.arange(CPerLayer, 2*CPerLayer)+1], neList=[ne/nCells, ne/nCells], symmetrize_contacts=True) | Auto-extraction from DFT (no explicit tau/alpha). setContact1D called with only contact atoms list + neList. fermiMethod='poly'. SCF 1000 cycles |
| CNTTest.py (old) | 1D chain 3-site test (obsolete) | surfG(F, S, [[0], [2]], [alpha, alpha], [beta, beta.getH()], [beta, beta.getH()], [S0, S0]) | Legacy Jupyter code using surfGreen (pre-gauNEGF). Does NOT use NEGFE class |
| CNanowire.py | CNanowire 6 atoms, spin='u' | setContact1D([[1,2,3],[4,5,6]], eta=1e-4) + setContact1D(..., eta=1e-4, T=300) | Two separate contact setups: 0K and 300K. Same atoms, different temperature. Both setVoltage(0.0), SCF with pulay=False on second run |
| CNanowire_ESCF.py | (3,3) CNT, 3-cell device, orthogonal basis | setContact1D([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12], [13,14,15,16,17,18,19,20,21,22,23,24]], [[13,...,24], [13,...,24]], symmetrize_contacts=True, eta=1e-4) + (second call) setContact1D(..., eta=1e-4, T=300) | Two calls: cold (symmetrize_contacts=True with atom-pair list for tau), warm (temp variant T=300K). setFock(F) explicit. fermiMethod='poly' |
| CNTPDT.py | CNT from previous run (PDT device), periodic | setContact1D([leftAtoms, rightAtoms], [tau_F, tau_F.conj().T], [tau_S, tau_S.conj().T], [alpha_F, alpha_F], [alpha_S, alpha_S], [tau_F, tau_F.conj().T], [tau_S, tau_S.conj().T], muList=[fermi, fermi]) | Full pattern: loads saved F/S from prior run, extracts tau/alpha/beta from center, passes muList (not neList). setVoltage(0.0, fermi) with explicit fermi arg |
| test_surfG1D_features.py | Chain 6-site, 1D-TB | surfG(F, S, [[0, 1], [4, 5]], taus=[tau, tau], staus=[stau, stau], alphas=[alpha, alpha], aOverlaps=[Salpha, Salpha], betas=[beta, beta], bOverlaps=[Sbeta, Sbeta]) | Direct surfG init (not via NEGFE.setContact1D): tests pattern (c) with explicit contact matrices. staus=None variant tests orthonormal coupling. aOverlaps/bOverlaps defaults |

### Bethe Lattice (setContactBethe / surfGBethe)

| File | System | Key args (verbatim) | Workflow notes |
|------|--------|---------------------|----------------|
| test_bethe_cross_term_fermi.py | Au 9x9 single cell, FCC [111] | setContactBethe(contactList, latFile='Au', eta=ETA, T=TEMPERATURE) NOT CALLED; uses surfGBAt constructor directly: surfGBAt(H0, Slist, Vlist, eta=eta, T=0) | No NEGFE wrapper. Tests surfGBAt directly. Contact geometry implicit in Vlist/Slist (12 neighbor hopping/overlap matrices) |
| test_soc.py | Au with SOC, 9x9->18x18 with spinors | surfGBAt(H0, Slist, Vlist, eta=eta, T=0, SOC=True [inferred]) | Reads Au.bethe with soc_p, soc_d params. constructMat expands to 18x18 when SOC=True |

### 3D Contacts (surfG3D / surfGAt3D)

| File | System | Key args (verbatim) | Workflow notes |
|------|--------|---------------------|----------------|
| test_surfGAt3D.py | Au FCC [111] surface + bulk, 12 neighbors | surfGAt3D(H0, Slist, Vlist, vecs, eta, T=T, kPoints=kpoints) | Constructor-level: no setContact call. vecs = 12 normalized direction vectors. kPoints=31 (k-space mesh for bulk DOS). gSurf/gBulk distinguish surface vs bulk |
| test_kpoint_convergence.py | Au FCC [111], k-point sweep | surfGAt3D(H0, jnp.array(Slist), jnp.array(Vlist), jnp.array(fcc_vecs, dtype=float), eta=1e-3, kPoints=3) | Similar to test_surfGAt3D but kPoints=3 for convergence test |

### Energy-Independent Sigma (setSigma with matrix)

| File | System | Key args (verbatim) | Workflow notes |
|------|--------|---------------------|----------------|
| test_transport_checkpointing.py | Nanowire 1D TB | SigmaCalculator(sig1_matrix) where sig1_matrix = ndarray | Tests checkpointing loop over energy list. SigmaCalculator wraps static matrix |

### Energy-Dependent Sigma (setSigma with surfG object / SigmaCalculator)

| File | System | Key args (verbatim) | Workflow notes |
|------|--------|---------------------|----------------|
| CNT33.py (end) | After SCF convergence | T = cohTransE(Elist + negf.fermi, negf.F * har_to_eV, negf.S, negf.g) | Transport via negf.g (surfG object). cohTransE accepts g (energy-dependent sigma) |
| test_transport_dos_crossterm.py | Orthogonal + non-orthogonal chains | sigma_calc = SigmaCalculator(g) where g is surfG | Wraps surfG into SigmaCalculator; used with dos_single_energy |
| test_cross_term.py | Orthogonal 1D chain, 4-site | g = surfG(F, S, indsList, eta=1e-3) passed directly to crossTermQ(E, i) | Tests Q_sym computation without SigmaCalculator wrapper |

## Workflow patterns

### Pattern A: Full Contact Setup with Explicit Matrices (4 occurrences)

DFT cluster (Gaussian) -> extract F/S/P -> compute H = S^-0.5 @ F @ S^-0.5 (orthogonal) -> extract tau/tau_S (hopping), alpha/alpha_S (on-site), beta/beta_S (periodic copies) from interior cells -> setFock(F) -> setContact1D([atoms, atoms], [tau, tau.T], [tau_S, tau_S.T], [alpha, alpha], [alpha_S, alpha_S], [beta, beta.T], [beta_S, beta_S.T], neList=[ne/2, ne/2], symmetrize_contacts=True) -> setVoltage(0) -> SCF(tol=1e-3, damp=0.02, cycles=100 or 1000, checkpoint=False).

Representative code (CNT33.py lines 26-87):
```
bar = qcb.BinAr(..., inputfile="CNT33_11.gjf")
bar.update(model='b3lyp', basis='6-31g(d,p)', ..., dofock=True)
S_full = np.array(bar.matlist['OVERLAP'].expand())
P_full = np.array(bar.matlist['ALPHA SCF DENSITY MATRIX'].expand())
F_full = np.array(bar.matlist['ALPHA FOCK MATRIX'].expand()) * har_to_eV
X_full = fractional_matrix_power(S_full, -0.5)
H_full = np.real(X_full @ F_full @ X_full)
[extract central block for device, interior cells for couplings]
negf = NEGFE(fn='...', func='b3lyp', basis='6-31g(d,p)', fullSCF=False, route='...')
negf.setFock(F)
inds = negf.setContact1D(
    [leftAtoms, rightAtoms],
    [tau_F, tau_F.conj().T],
    [None, None],  # or [tau_S, tau_S.conj().T]
    [alpha_F, alpha_F],
    [alpha_S, alpha_S],
    [tau_F, tau_F.conj().T],
    [tau_S, tau_S.T],
    neList=[ne / nCells, ne / nCells],
    symmetrize_contacts=True,
)
negf.setVoltage(0.0)
negf.SCF(1e-3, 0.02, 1000)
```

### Pattern B: Auto-Extraction Contact (2 occurrences)

DFT cluster -> NEGFE loads checkpoint -> setContact1D([atoms_L, atoms_R], neList=[ne/2, ne/2], symmetrize_contacts=True) with NO tau/alpha/beta (auto-extracted from DFT checkpoint) -> setVoltage(0.0, fermiMethod='poly') -> SCF(checkpoint=False). Fermi search uses polynomial fitting rather than ANT quadrature.

Representative code (CNTCont.py lines 59-76):
```
negf = NEGFE(fn='CNTCont', func='b3lyp', basis='STO-3G', fullSCF=False, ...)
inds = negf.setContact1D(
    [np.arange(CPerLayer)+1, np.arange(CPerLayer, 2*CPerLayer)+1],
    neList=[ne / nCells, ne / nCells],
    symmetrize_contacts=True
)
negf.setVoltage(0.0, fermiMethod='poly')
negf.SCF(1e-3, 0.02, 1000, checkpoint=False)
```

### Pattern C: Multi-Temperature Sweep (2 occurrences)

Single device geometry, contact setup called twice: first at T=0K, second at T=300K. Both go through full SCF cycle independently. Used to study temperature-dependent transmission/DOS.

Representative code (CNanowire.py lines 21-40):
```
negf.setContact1D([[1,2,3],[4,5,6]], eta=1e-4)  # T=0 (default)
negf.setVoltage(0.0)
negf.SCF(1e-3, 0.02, 100)
negf.SCF(1e-3, 0.02, 1000, pulay=False)
[save/plot transmission at 0K]
negf.setContact1D([[1,2,3],[4,5,6]], eta=1e-4, T=300)  # T=300K
negf.setVoltage(0.0)
negf.SCF(1e-3, 0.02, 100)
negf.SCF(1e-3, 0.02, 1000, pulay=False)
[save/plot transmission at 300K]
```

### Pattern D: Periodic (Lead-Like) Device (1 occurrence)

Device extracted from 11-cell system, treated as periodic (tau and beta are identical hopping matrices). Contact atoms extracted from prior SCF run via io.loadmat. muList (not neList) passed to setContact1D, allowing asymmetric Fermi levels.

Representative code (CNTPDT.py lines 25-65):
```
A = io.loadmat('CNTCont_2cell_selfESCF.mat')
tau_F = A['F'][:orbsPerCell, orbsPerCell:]
tau_S = A['S'][:orbsPerCell, orbsPerCell:]
alpha_F = A['F'][:orbsPerCell, :orbsPerCell]
alpha_S = A['S'][:orbsPerCell, :orbsPerCell]
fermi = A['fermi'][0][0]
negf = NEGFE(fn='CNTPDT', func='b3lyp', basis='chkbasis', route='...')
inds = negf.setContact1D(
    [leftAtoms, rightAtoms],
    [tau_F, tau_F.conj().T],
    [tau_S, tau_S.conj().T],
    [alpha_F, alpha_F],
    [alpha_S, alpha_S],
    [tau_F, tau_F.conj().T],
    [tau_S, tau_S.conj().T],
    muList=[fermi, fermi],
)
negf.setVoltage(0.0, fermi)
negf.SCF(1e-3, 0.02, 50, pulay=False)
negf.SCF(1e-3, 0.01, 1000)
```

### Pattern E: Direct surfG1D Construction (for unit tests)

Skip NEGFE entirely. Build F/S by hand (TB tight-binding), call surfG() constructor directly with all contact parameters (taus, staus, alphas, aOverlaps, betas, bOverlaps, eta, spin). Test individual methods (sigma, sigmaTot, crossTermQ, g) on the surfG object.

Representative code (test_surfG1D_features.py lines 78-85):
```
g = surfG(F, S,
          [[n_contact, n_contact + 1], [N - n_contact - 2, N - n_contact - 1]],
          taus=[tau, tau],
          staus=[stau, stau],
          alphas=[alpha, alpha],
          aOverlaps=None,
          betas=[beta, beta],
          bOverlaps=None)
assert g.aSList[i] == identity  # test aOverlaps defaulting
```

### Pattern F: Bethe Lattice Direct Construction (for unit tests & validation)

Build surfGBAt or surfGAt3D directly from H0/Slist/Vlist (12 neighbor hopping/overlap matrices for Au FCC [111]). Compute Fermi energy via calcFermi, DOS via DOS(), and compare against reference implementations. No NEGFE involved; pure geometric contact model.

Representative code (test_bethe_cross_term_fermi.py lines 44-53):
```
ne, H0, Slist, Vlist = au_params  # read Au.bethe + construct matrices
gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
fermi = gBAt.calcFermi(ne / 2)
assert abs(fermi - AU_BULK_FERMI_EV) < 0.02
```

### Pattern G: Transport Calculation with Checkpointing (1 occurrence)

After SCF convergence, loop over energy list with checkpoint_file argument. Saves intermediate transmission/DOS every checkpoint_interval energies. Used for long energy scans.

Representative code (test_transport_checkpointing.py):
```
sigma_calc = SigmaCalculator(negf.g)
T = calculate_transmission(negf.F, negf.S, sigma_calc, Elist, 
                           checkpoint_file='T_checkpoint.h5', 
                           checkpoint_interval=10)
```

## Configuration deviations

| Constant | Default value | Override value | File | Reason (from comment if available) |
|----------|--------------|----------------|------|-------------------------------------|
| ETA | config.ETA (~1e-2) | 1e-4 | CNT33.py, CNTCont.py, CNanowire.py, CNanowire_ESCF.py, CNTPDT.py | "broadening to speed up convergence" (comment in CNanowire.py) |
| ETA | config.ETA | 1e-6 | test_bethe_cross_term_fermi.py, test_soc.py, test_surfGAt3D.py | Fine broadening for accurate DOS & Fermi energy in single-cell Bethe tests |
| ETA | config.ETA | 5e-4 | test_surfG1D_features.py (CNT33 tests) | "Golden standard" for CNT (3,3) integer transmission |
| TEMPERATURE | config.TEMPERATURE (0K) | 300 | CNanowire.py, CNanowire_ESCF.py (second setContact1D call) | Finite-T transport: setContact1D(..., T=300) |
| fermiMethod | (adaptive ANT default) | 'poly' | CNTCont.py line 69, CNanowire_ESCF.py line 47 | Polynomial Fermi fit instead of ANT quadrature |
| SCF cycles | SCF_MAX_CYCLES (~100) | 1000 | CNT33.py, CNTCont.py, CNanowire_ESCF.py line 66 | Dense convergence for multi-cell devices |
| checkpoint | checkpoint=True (default) | checkpoint=False | CNTCont.py, CNanowire_ESCF.py line 50 | Disable checkpointing for smaller systems |
| pulay | pulay=True (default) | pulay=False | CNanowire.py, CNanowire_ESCF.py (second SCF call) | Disable Pulay mixing after initial convergence |
| symmetrize_contacts | (None, inferred True for 2-cell auto-extract) | True (explicit) | CNT33.py, CNTCont.py, CNanowire_ESCF.py | Force contact symmetrization for parity-symmetric systems (sp chain, CNT) |
| spin | spin='r' (restricted, default) | spin='u' (unrestricted) | CNanowire.py | Unrestricted calculation for open-shell molecule |
| basis | chkbasis (default, read from checkpoint) | 'STO-3G', '6-31g(d,p)', 'lanl2dz' | CNT33.py (6-31g), CNTCont.py (STO-3G), CNTPDT.py (chkbasis) | Basis set specified in DFT update() or passed to NEGFE() |

## Notable test files

1. **test_surfG1D_features.py** - Comprehensive 1D chain tests: Xi attribute, aOverlaps/bOverlaps defaults, setF Fermi tracking, de-orthonormalization with staus=None, composite regularization (congruent clipping). Tests the full sigma-correction approach for non-orthogonal overlap. Integer transmission benchmarks for CNT (3,3).

2. **test_surfGAt3D.py** - Atomic 3D contact validation: reciprocal lattice construction (2D surface + 3D bulk k-point meshes), neighbor geometry (FCC [111] 12 vectors), subset sigma with active_dirs (for partial contact atoms), PSD gamma checks. Compares surfGAt3D vs surfGBAt DOS & Fermi energy.

3. **test_bethe_cross_term_fermi.py** - Bethe lattice single-cell tests: cross-term electron count consistency, Q_sym = (tau @ g^R @ S_LD + S_DL @ g^R @ tau^dag) / 2, Fermi energy agreement between independent instances. Validates SurfGProtocol compliance.

4. **CNT33.py** - Full workflow: 11-cell DFT system (Gaussian) -> extract 5-cell device + interior couplings -> NEGFE.setContact1D with tau/alpha/beta -> SCF 1000 cycles -> transmission via cohTransE. Demonstrates pattern A (full matrix extraction).

5. **CNTCont.py** - Auto-extraction workflow: checkpoint loaded by NEGFE -> setContact1D([atoms, atoms], neList=..., symmetrize_contacts=True) with no matrices -> fermiMethod='poly' -> SCF 1000 cycles. Demonstrates pattern B.

6. **CNanowire.py** - Multi-temperature workflow: two consecutive setContact1D calls (0K, 300K) with same atoms but different T parameter -> separate SCF convergence for each -> transmission saved at both temperatures.

7. **test_cross_term.py** - Cross-term formulas: Q_sym computation across surfGTest (returns None), surfG1D orthogonal (None), and non-orthogonal chains. Validates Q_tot = sum([Q_L, Q_R]) integration for delta_N = -(1/pi) Im(sum_k w_k Tr(G^R @ Q_tot)).

8. **test_transport_checkpointing.py** - Checkpointing infrastructure: SigmaCalculator wrapper + calculate_transmission with checkpoint_file and checkpoint_interval. Tests energy loop robustness for long scans.

9. **test_calcTSW.py** - Fermi energy bounds: calcTSW convergence, Eminf placement, warm-start behavior. Uses surfGBAt directly for Au single cell.

10. **test_soc.py** - Spin-orbit coupling: readBetheParams with soc_p, soc_d parameters, constructSOCterm integration, 18x18 spinor expansion, surfGBAt with SOC=True flag.

## Coverage gaps

Entry point calls NOT found in any test file:

| Function/Method | Module | Status |
|-----------------|--------|--------|
| NEGF.__init__ (non-NEGFE subclass) | scf | Only NEGFE instances tested; base NEGF never instantiated standalone |
| NEGF.setContacts (non-energy-dependent, base class) | scf | Only NEGFE.setContact1D/setContactBethe tested (energy-dependent subclass methods) |
| NEGF.setSigma (static matrix sigma via base NEGF) | scf | Only NEGFE.setSigma with energy-dependent g tested; NEGF.setSigma(sig=-0.1j, sig2=None) not exercised |
| NEGF.SCF (base class) | scf | Only NEGFE.SCF tested; base NEGF.SCF never called |
| NEGF.spawnNEGF | scfE | No test creates NEGF instance from NEGFE via spawnNEGF |
| setContactBethe (via NEGFE) | scfE | Only direct surfGBAt tested; NEGFE.setContactBethe never called in any workflow |
| setContact1D with tau=None (auto-extract full) | scfE | CNTCont.py passes tau implicitly, but explicit tau=None not tested |
| setSigma (pattern a: setSigma on base NEGF after setContacts) | scf | Only NEGFE.setSigma (pattern b: energy-dependent) tested |
| density.densityGrid, densityGridN, densityGridTrap, densityComplexN, densityComplex | density | Used internally by setVoltage; not called directly by user code in tests |
| currentSpin (spin-specific current, pattern u/d vs r) | transport | currentE (energy-dependent) used; currentSpin never exercised |
| cohTrans, cohTransSpin (legacy transmission via setSigma) | transport | cohTransE (energy-dependent via surfG) used; cohTrans/cohTransSpin (static sigma) never called |
| DOSE (legacy DOS via setSigma) | transport | No user call to DOSE; only new transport.dos_single_energy / density._compute_dos_at_energy tested |
| surfG3.readBetheParams, constructMat | surfG3D | Both methods called indirectly via test_surfGAt3D helpers, but never used in workflow |
| surfG3.genNeighbors | surfG3D | Neighbor geometry hard-coded in test_surfGAt3D; genNeighbors method not invoked |
| surfGB.testDOrbitalFunctions, testDOrbitalSymmetry, testPDInteraction, testDDInteraction | surfGBethe | Test helper methods never called in any workflow |
| integralFit, integralFitNEGF | density | Advanced Fermi fit variants; only calcFermi (bisection/secant/Muller/poly) tested |
| GrInt, GrIntCross, GrLessInt | integrate | Used internally; never called by user code |

---

## Summary

Scanned 23 test files total: 7 priority-A unit tests (surfG1D, surfGAt3D, Bethe cross-terms, cross-term formulas, transport DOS, SOC, ANT quadrature), 7 priority-B full workflows (CNT33, CNTCont, CNTTest/legacy, CNanowire x2, CNTPDT), 9 priority-C unit tests (calcTSW, densityReal, transport checkpointing, constructSOCterm, k-point convergence, Fermi shift JIT, and 3 benchmark/legacy).

**Contact types found:**
- 1D Chain (setContact1D): 6 tests with full matrix extraction, auto-extraction, multi-temperature, periodic device patterns
- Bethe Lattice (setContactBethe/direct surfGBAt): 2 unit tests + 1 legacy code
- 3D Atomic (surfGAt3D): 1 validation test vs Bethe, 1 k-point study
- Direct surfG1D construction (unit tests): 5 parametric tests

**Most common workflow pattern:** Full matrix extraction (Pattern A) -> DFT checkpoint load -> explicit tau/alpha/beta -> setContact1D -> SCF. Found in CNT33.py, CNTPDT.py, CNanowire_ESCF.py. 4 instances of auto-extraction (Pattern B) also present.

**Top config deviations:** ETA (1e-4 to 1e-6 from default ~1e-2 for faster/accurate convergence), TEMPERATURE (0K vs 300K), fermiMethod='poly' (vs ANT), symmetrize_contacts=True (for parity-symmetric systems), pulay=False (after warm start), checkpoint=False (for small systems).
