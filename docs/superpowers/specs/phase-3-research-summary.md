# Phase III Research Summary

Date: 2026-05-11
Source: phase-3-test-inventory.md + phase-3-source-inventory.md

---

## Coverage matrix

| Contact entry point | Tested in | Existing doc | Test exhaustiveness |
|--------------------|-----------|--------------|---------------------|
| NEGFE.setContact1D | CNT33.py, CNTCont.py, CNanowire.py, CNanowire_ESCF.py, CNTPDT.py, test_surfG1D_features.py | docs/1D_contact_setup_guide.md (301-line markdown) | High -- all 3 usage patterns exercised |
| NEGFE.setContactBethe | NOT CALLED via NEGFE in any test file. Only direct surfGBAt construction tested (test_bethe_cross_term_fermi.py, test_soc.py, test_calcTSW.py, test_surfGAt3D.py). | None | Zero via NEGFE wrapper; medium coverage of underlying surfGBAt |
| NEGFE.setSigma | Indirectly: test_transport_checkpointing wraps static matrix via SigmaCalculator. NEGFE.setSigma (surfGTest) itself not explicitly invoked. | None | Low |
| NEGF.setSigma | Zero direct tests. Only NEGFE subclass tested. | None | Zero |
| NEGF.setContacts | Zero direct tests. Internal to NEGFE methods. | None | Zero (internal use only) |
| surfGAt3D (direct) | test_surfGAt3D.py, test_kpoint_convergence.py | None | Medium -- sigma and DOS, not full workflow |
| surfGB / surfGBAt | test_bethe_cross_term_fermi.py, test_soc.py, test_calcTSW.py | None | Medium -- geometry and Fermi energy, not full NEGF workflow |
| SigmaCalculator | test_transport_checkpointing.py, test_transport_dos_crossterm.py | None | Low -- only used as wrapper, not standalone |
| Transport (calculate_transmission, calculate_dos) | test_transport_dos_crossterm.py, test_transport_checkpointing.py | None (README only) | Medium |

---

## Workflow pattern frequency

Ranked by occurrence in tests and full-workflow files:

1. **Pattern A: Full matrix extraction** (4 occurrences -- CNT33.py, CNanowire_ESCF.py, CNTPDT.py + partial in others)
   - DFT cluster -> F/S/P extraction -> tau/alpha/beta from interior cells -> NEGFE.setContact1D with all matrices -> setVoltage -> SCF -> transmission
   - Most realistic user workflow; involves Gaussian .chk checkpoint loading
   - Covered by existing 1D guide (sections 1-3)

2. **Pattern B: Auto-extraction contact** (2 occurrences -- CNTCont.py, part of CNanowire_ESCF.py)
   - NEGFE loads .chk -> setContact1D([atoms, atoms], neList=..., symmetrize_contacts=True) with no tau/alpha
   - gauNEGF extracts coupling matrices internally from DFT Fock block
   - Existing guide mentions this but does not walk through it step-by-step

3. **Pattern C: Multi-temperature sweep** (2 occurrences -- CNanowire.py, CNanowire_ESCF.py)
   - Same geometry, setContact1D called twice (T=0 then T=300)
   - Two independent SCF loops, two transmission curves compared
   - NOT in existing guide

4. **Pattern D: Periodic / muList device** (1 occurrence -- CNTPDT.py)
   - Loads saved F/S from prior SCF run (NEGF chained from a converged contact calc)
   - Passes muList (not neList) to setContact1D for asymmetric Fermi levels
   - Closest to a "real" IV bias preparation step
   - NOT in existing guide

5. **Pattern E: Direct surfG1D construction** (unit tests only -- test_surfG1D_features.py)
   - Builds F/S by hand (tight-binding), instantiates surfG() directly without NEGFE
   - Used for testing surfG methods in isolation
   - Not a user-facing workflow

6. **Pattern F: Bethe lattice direct construction** (2 occurrences -- test_bethe_cross_term_fermi.py, test_soc.py)
   - surfGBAt(H0, Slist, Vlist, eta, T=0) with Au parameters; calcFermi, DOS, crossTermQ tested
   - No NEGFE wrapper -- validates contact model standalone
   - Not a user-facing workflow per se, but is how users would validate a Bethe contact before NEGF

7. **Pattern G: Transport checkpointing** (1 occurrence -- test_transport_checkpointing.py)
   - SigmaCalculator(negf.g) -> calculate_transmission with checkpoint_file + checkpoint_interval
   - Enables resumable long energy scans
   - NOT in existing guide

---

## Feature-without-example gaps

Entry points with NO test coverage via the expected user-facing call:

| Gap | Severity | Interview question |
|-----|----------|-------------------|
| NEGFE.setContactBethe called via NEGFE wrapper | HIGH -- no test drives the full Bethe workflow through NEGFE.SCF | Batch C: "Do you have a real workflow that uses setContactBethe via NEGFE? What does the Gaussian/contact setup look like?" |
| NEGF base class (non-NEGFE) standalone | MEDIUM -- may be intentional (NEGFE is the recommended path) | Batch B: "Is NEGF standalone still used, or is NEGFE the always-recommended class?" |
| surfG3 with bar object (vs surfGAt3D direct) | MEDIUM -- surfG3 needs a qcb.BinAr bar object, which requires Gaussian output | Batch C: "When does a user use surfG3 vs surfGAt3D? Is surfG3 the high-level wrapper for production use?" |
| Energy-independent sigma NEGF.setSigma (static, no NEGFE) | LOW -- rare in practice, probably test-only | Batch C: "Is NEGF.setSigma (static matrix, base class) still useful for users, or is surfGTest the intended interface?" |
| IV curve / voltage sweep workflow | HIGH -- no test sweeps voltage systematically | Batch C: "Is there a canonical voltage-sweep workflow? How do users sweep setVoltage + rerun SCF?" |
| SigmaCalculator standalone (not via negf.g) | LOW -- advanced use case | Not interview-critical |

---

## Expert workflow candidates

These test patterns showcase notable combinations a guide should feature:

1. **setContact1D with tau=None auto-extraction** (CNTCont.py):
   - Most user-friendly entry point; guide should highlight this for new users
   - fermiMethod='poly' speeds convergence for small systems

2. **Multi-temperature sweep** (CNanowire.py):
   - Two independent SCF runs same geometry; temperature-dependent transmission curves
   - Demonstrates T parameter on setContact1D

3. **Periodic lead-like device with muList** (CNTPDT.py):
   - Shows how to chain two gauNEGF runs: first get a self-consistent contact, then load for device
   - muList vs neList distinction deserves its own documentation section

4. **SOC with Bethe lattice** (test_soc.py):
   - surfGBAt with SOC=True parameter and readBetheParams with soc_p, soc_d fields
   - 18x18 spinor expansion (from 9x9 Au d-orbital basis)
   - Expert topic: spin-orbit coupling in metallic contacts

5. **Transport checkpointing** (test_transport_checkpointing.py):
   - Resumable energy scans via checkpoint_file and checkpoint_interval
   - Important for users running long transport calculations on HPC

6. **surfGAt3D k-point convergence** (test_kpoint_convergence.py):
   - kPoints parameter controls reciprocal lattice mesh for 3D contacts
   - Shows convergence behavior as function of kPoints

7. **Three surfG1D usage patterns** (test_surfG1D_features.py + CNTCont.py + CNT33.py):
   - Pattern (a): minimal args (contactList, neList) -- auto-extracts everything
   - Pattern (b): contactList + tauList -- custom coupling, auto onsite
   - Pattern (c): full spec -- tauList, stauList, alphas, aOverlaps, betas, bOverlaps
   - Each pattern deserves a code block in the 1D contacts guide

---

## Configuration deviation patterns

Most common overrides from gauNEGF.config defaults:

| Constant | Default | Most common override | Context |
|----------|---------|---------------------|---------|
| ETA | (default from config; ~1e-2 or similar) | 1e-4 (workflows), 1e-6 (Bethe/3D tests) | Lower ETA for more accurate Fermi energy; higher for faster convergence |
| TEMPERATURE | 0 | 300 | Finite-temperature transport in setContact1D / setContactBethe |
| fermiMethod | ANT quadrature | 'poly' | Polynomial Fermi fit; faster for small systems or low convergence tolerance |
| SCF cycles | ~100 | 1000 | Dense convergence for multi-cell CNT devices |
| checkpoint | True | False | Disable checkpointing for small systems (overhead not worth it) |
| pulay | True | False | Disable Pulay mixing on warm-start SCF for stability |
| symmetrize_contacts | None | True | Force symmetrization for parity-symmetric electrodes (sp-chain, armchair CNT) |
| spin | 'r' (restricted) | 'u' (unrestricted) | Open-shell device (CNanowire.py) |

ETA is the most commonly tuned parameter. The config_tuning guide should explain the tradeoff between ETA precision and convergence speed, and give recommended values for different contact types.

---

## Existing 1D guide assessment (for migration / interview Batch E)

docs/1D_contact_setup_guide.md (301 lines, Claude-generated):
- Covers: Gaussian input prep, large cluster vs device cluster, matrix extraction steps,
  setContact1D with explicit tau/alpha/beta (Pattern A), setVoltage, SCF loop, saving results.
- Missing: auto-extraction pattern (Pattern B), multi-temperature (Pattern C), muList/periodic (Pattern D),
  voltage sweep workflow, fermiMethod option, symmetrize_contacts option, neList vs muList distinction,
  transport checkpointing, SigmaCalculator usage after SCF.
- Accuracy risk: code examples are synthesized (no citation to test files), written without the
  Phase II truth table available. Verifier pass should check all code blocks before migration.
- Recommendation: Use as structural skeleton for contacts_1d.rst; replace code blocks with
  test-file-lifted examples where available; add Pattern B and C sections.

---

## Research summary

4 contact types documented in source (1D chain, Bethe lattice, 3D atomic, constant sigma).
7 workflow patterns found in tests (A through G).
5 feature-without-example gaps surface for interview (setContactBethe via NEGFE, IV sweep, NEGF standalone,
surfG3 vs surfGAt3D, NEGF.setSigma static path).
Top config deviations: ETA, TEMPERATURE, fermiMethod.

R1 found 23 test files scanned; R2 found 5 contact entry points and 7 surfG-compatible classes.
