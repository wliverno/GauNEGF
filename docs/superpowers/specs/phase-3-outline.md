# Phase III Guide Outline

Date: 2026-05-11
Source: phase-3-interview-notes.md + phase-3-research-summary.md + api-truth.md

Approved guides: 5
Audience: Experts new to gauNEGF (know NEGF theory; need the API and workflow)

---

## Guide 1: Choosing a Contact Type

**File:** docs/source/guides/contact_choice.rst
**Length estimate:** Short (1-2 pages)

### Target audience

API experts deciding which contact class to use. Assumes NEGF knowledge.

### Sub-sections

1. Overview of contact types in gauNEGF
   - Content: 4 contact types (energy-independent, 1D chain, Bethe lattice, 3D atomic),
     mapped to gauNEGF classes (NEGF.setSigma, NEGFE.setContact1D, NEGFE.setContactBethe,
     surfGAt3D)
   - Key example: rst list-table (decision matrix), no code block
   - Cross-refs: API sections Contact Models, Core Modules

2. Decision tree: Which contact type for your system?
   - Content: rst list-table with 3 columns: Scenario / Contact type / gauNEGF class
     Rows:
     - "Quick test or benchmarking" -> energy-independent -> NEGF.setSigma
     - "Molecular junction with metallic electrodes (Au, Au+SOC)" -> Bethe lattice -> NEGFE.setContactBethe
     - "Periodic 1D electrode (CNT, nanowire)" -> 1D chain -> NEGFE.setContact1D
     - "FCC bulk surface with k-grid" -> 3D atomic -> surfGAt3D (advanced)
   - No code block; prose + table only

3. CRITICAL: Fermi energy for 1D auto-extract contacts
   - Content: "If you use setContact1D with atom indices only (no explicit tau matrices),
     you MUST specify a fermiMethod in setVoltage. Without a fermiSearch method, the electron
     count in the contact region is not physically meaningful."
   - Contrast with Bethe contacts: "For setContactBethe, the Bethe lattice provides its
     own Fermi reference; fermiMethod='poly' is recommended but the Bethe Fermi energy
     can be used directly."
   - Key example: two-line code block showing setVoltage(0.0, fermiMethod='poly') for 1D
     vs setVoltage(0.0) for Bethe
   - Cross-refs: Guide config_tuning.rst (fermiSearch method table)

4. After choosing: what to read next
   - Content: cross-reference to each contact guide

### Code examples planned

- E1: setVoltage fermiMethod comparison (new, 2 lines, illustrative)
- E2: list-table decision matrix (rst table, no Python code)

### Cross-refs to API

- :class:`gauNEGF.scf.NEGF` (Core Modules: NEGF Base Class)
- :class:`gauNEGF.scfE.NEGFE` (Core Modules: Energy-Dependent NEGF)
- :class:`gauNEGF.surfG1D.surfG` (Contact Models: 1D Chain)
- :class:`gauNEGF.surfGBethe.surfGB` (Contact Models: Bethe Lattice)
- :doc:`contacts_1d`, :doc:`contacts_bethe`, :doc:`config_tuning`

### Cross-refs to other guides

- All 4 other guides (decision tree links out to each)

---

## Guide 2: 1D Chain Contacts

**File:** docs/source/guides/contacts_1d.rst
**Length estimate:** Medium (3-5 pages)
**Migration:** docs/1D_contact_setup_guide.md (301 lines, mostly RST format conversion)

### Target audience

Users setting up NEGFE transport with 1D chain contacts (CNT, nanowire, periodic electrode).

### Sub-sections

1. Overview (migrate from existing guide "Overview" section)
   - Content: 3-stage workflow summary; when to use 1D contacts
   - Cross-refs: :doc:`contact_choice`

2. Prepare Gaussian input files (migrate sections 1a and 1b)
   - Content: large cluster for matrix extraction; device cluster for SCF
   - Key example: Lifted from CNT33.py context (CNT33_11.gjf description)
   - Notes from existing guide are accurate per user review

3. Understanding unit cell structure (migrate section 2)
   - Content: What is nCells, CPerLayer, ne; how to map DFT atoms to NEGF indices

4. Extract matrices from the large cluster (migrate section 3a/3b/3c)
   - Content: Extract F/S/P blocks, count electrons, extract tau/alpha/beta from interior cells
   - Key example: Adapted from CNT33.py lines 26-65 (matrix extraction block)
   - Source: phase-3-test-inventory.md Pattern A code

5. Set up NEGFE (migrate section 4)
   - Content: NEGFE constructor with fn, func, basis, spin, fullSCF, route parameters
   - Key example: Lifted from CNT33.py: NEGFE(fn=..., func='b3lyp', basis='6-31g(d,p)', ...)

6. Three usage patterns for setContact1D (EXPAND from existing guide section 5)
   - Content: Three patterns with code examples each:
     a. Full specification (tau, stau, alpha, aOverlap, beta, bOverlap, neList)
        Lifted from CNT33.py setContact1D call
     b. Custom coupling with auto onsite (tauList only, no alphas/betas)
        New illustrative example (synthesized, needs_review)
     c. Auto-extraction from DFT (contactList + neList only, no matrices)
        Lifted from CNTCont.py setContact1D call
   - CRITICAL WARNING box: "Pattern (c) auto-extraction requires fermiMethod in setVoltage"
   - Cross-ref: :doc:`config_tuning` for fermiSearch method selection

7. Run the SCF (migrate section 6)
   - Content: SCF parameters (tol, damp, cycles, checkpoint, pulay)
   - Key example: negf.SCF(1e-3, 0.02, 1000) from CNT33.py

8. Post-SCF: transmission and current (migrate section 7, expand)
   - Content: SigmaCalculator(negf.g) -> calculate_transmission / calculate_current
   - Key example: from CNT33.py cohTransE call or AuBetheFerrocene.py SigmaCalculator
   - Cross-ref: :doc:`workflow_recipes` for IV curve sweep

9. Common pitfalls (migrate existing section, add new ones)
   - symmetrize_contacts=True for parity-symmetric systems
   - fermiSearch method requirement for auto-extract (1D contacts)
   - pulay=False on warm-start second SCF pass

### Code examples planned

- E1: CNT33.py matrix extraction block (lifted, ~20 lines)
- E2: NEGFE constructor (lifted from CNT33.py, ~3 lines)
- E3a: Full setContact1D (lifted from CNT33.py, ~10 lines)
- E3b: Custom coupling (synthesized, needs_review)
- E3c: Auto-extract setContact1D (lifted from CNTCont.py, ~5 lines)
- E4: SCF call (lifted, 1 line)
- E5: SigmaCalculator + calculate_transmission (lifted from AuBetheFerrocene.py or tests, ~5 lines)

### Cross-refs to API

- :class:`gauNEGF.scfE.NEGFE` (Energy-Dependent NEGF)
- :class:`gauNEGF.surfG1D.surfG` (Contact Models: 1D Chain)
- :func:`gauNEGF.scfE.NEGFE.setContact1D`
- :func:`gauNEGF.transport.SigmaCalculator`
- :func:`gauNEGF.transport.calculate_transmission`

### Cross-refs to other guides

- :doc:`contact_choice` (overview)
- :doc:`config_tuning` (fermiSearch)
- :doc:`workflow_recipes` (IV curve, multi-temperature)

---

## Guide 3: Bethe Lattice Contacts

**File:** docs/source/guides/contacts_bethe.rst
**Length estimate:** Medium (3-5 pages)
**Canonical sources:** AuStudies/AuBetheFerrocene.py, HemeStudies/AuFe.py (WRITER MUST READ BOTH)

### Target audience

Users setting up metallic contacts (Au, Fe+Au) using Bethe lattice self-energies.
Covers standard spin and SOC (spin='g') workflows.

### Sub-sections

1. Overview: What is a Bethe lattice contact?
   - Content: FCC [111] surface Bethe lattice approximation, Slater-Koster parameters,
     when to use Bethe vs 1D chain contacts
   - Cross-ref: :doc:`contact_choice`

2. Available latFile options
   - Content: 'Au' (standard) vs 'AuSOC' (with spin-orbit coupling)
   - Note: latFile controls Slater-Koster hopping/overlap parameter set
   - Cross-ref: :class:`gauNEGF.surfGBethe.surfGB`

3. Setting up NEGFE with Bethe contacts (spin='r')
   - Content: Standard (non-SOC) Bethe contact workflow
   - Key example: Adapted from HemeStudies/AuFe.py (simpler case):
     negf = NEGFE(fn=..., func='b3lyp', basis='chkbasis', spin='g', fullSCF=False)
     negf.setContactBethe([[1,2,3],[7,8,9]], 'Au')
     negf.setVoltage(0.0)
     negf.SCF(1e-3, 0.1, 1000, pulay=False)
   - Note: Bethe contacts do not require fermiMethod in setVoltage (Bethe provides Fermi reference)

4. SOC with Bethe contacts (spin='g')
   - Content: Using 'AuSOC' latFile, spin='g' (generalized spinor), 18x18 matrix expansion
   - Key example: From AuBetheFerrocene.py:
     negf = NEGFE(fn=fn, func='b3lyp', basis='chkbasis', spin='g', fullSCF=False, route=...)
     negf.setContactBethe([[1,2,3,4,5,6], [36,...,41]], 'AuSOC')
     negf.setVoltage(0.0, fermiMethod='poly')
   - Note: spin='g' doubles the orbital dimension (SOC mixes up and down)

5. Warm-start from prior run
   - Content: Loading saved density matrix + Fermi energy to warm-start SCF
   - Key example: From AuBetheFerrocene.py:
     A = io.loadmat(f"{fn}_{extra}.mat")
     negf.setDen(A['den'])
     negf.setVoltage(0.0, A['fermi'][0][0])
     negf.SCF(1e-3, 0.02, 200, checkpoint=False)

6. Testing the Bethe contact standalone (advanced)
   - Content: Direct surfGBAt construction for validating contact parameters
   - Key example: From test_bethe_cross_term_fermi.py Pattern F code
   - Cross-ref: :class:`gauNEGF.surfGBethe.surfGBAt`

7. Pitfalls
   - latFile choice (Au vs AuSOC)
   - spin='g' doubles orbital dimension (memory, cost)
   - setVoltage with Bethe: fermiMethod optional, Bethe Fermi energy can be used directly

### Code examples planned

- E1: HemeStudies/AuFe.py workflow (lifted, ~10 lines, simpler/non-SOC)
- E2: AuBetheFerrocene.py SOC setup (lifted, ~8 lines)
- E3: AuBetheFerrocene.py warm-start (lifted, ~5 lines)
- E4: surfGBAt direct construction (lifted from test_bethe_cross_term_fermi.py, ~5 lines)

### Cross-refs to API

- :class:`gauNEGF.scfE.NEGFE` (Energy-Dependent NEGF)
- :class:`gauNEGF.surfGBethe.surfGB` (Contact Models: Bethe Lattice)
- :class:`gauNEGF.surfGBethe.surfGBAt`
- :func:`gauNEGF.scfE.NEGFE.setContactBethe`
- :doc:`workflow_recipes` (IV curve)

### Cross-refs to other guides

- :doc:`contact_choice`
- :doc:`workflow_recipes` (for full IV sweep using Bethe contacts)

---

## Guide 4: Configuration and Tuning

**File:** docs/source/guides/config_tuning.rst
**Length estimate:** Short-Medium (2-3 pages)

### Target audience

Users who need to tune convergence, Fermi energy search, or broadening for their system.

### Sub-sections

1. Overview: gauNEGF.config constants
   - Content: Brief intro -- all defaults live in gauNEGF.config; can be overridden locally
     or via environment. Cross-ref: :class:`gauNEGF.config`

2. Fermi energy search method (fermiMethod)
   - Content: fermiMethod parameter in setVoltage controls the Fermi SEARCH ALGORITHM
     (not the integration method). Options:
     | Method | Stability | Speed | When to use |
     |--------|-----------|-------|-------------|
     | bisect | highest | slow | Always safe; use when other methods fail |
     | muller | high | medium | Good default for most systems |
     | poly | medium | medium-fast | Common in production (CNTCont.py, AuBetheFerrocene.py) |
     | secant | low | fast | Use only on well-behaved systems with good initial guess |
     | predict | lowest | fastest | Advanced; unstable for most systems |
   - CRITICAL NOTE: "For setContact1D with atom indices only (no explicit tau matrices),
     you MUST set fermiMethod. Not setting it yields non-physical electron counts."
   - Cross-ref: :doc:`contact_choice`, :doc:`contacts_1d`

3. SCF_DAMPING
   - Content: Controls density matrix mixing between SCF iterations.
     Default: 0.02. Range seen in production: 0.01 to 0.1.
     - Tighten (0.01): oscillating convergence; system near a phase transition
     - Loosen (0.1): slow convergence; aggressive damping to find basin first
   - Key example: negf.SCF(1e-3, 0.02, 1000) vs negf.SCF(1e-3, 0.1, 1000, pulay=False)

4. ETA (broadening)
   - Content: Controls imaginary part of energy (broadening of spectral peaks).
     Default in config: see gauNEGF.config.ETA.
     - 1e-4: typical for full DFT workflows (CNT33.py, CNanowire.py)
     - 1e-6: accurate Fermi energy search for Bethe/3D contacts
     - Larger ETA speeds convergence but broadens features
   - Note: Set locally via setContact1D(eta=...) or setContactBethe(eta=...)

5. TEMPERATURE
   - Content: Default temperature for contact Fermi functions.
     Pass T= to setContact1D or setContactBethe for finite-T contacts.
     Setting T=300 models room-temperature Fermi distribution in the electrodes.

6. Pointer to IntegralDemo
   - "For a hands-on comparison of integration methods (complex contour vs real-axis),
     see examples/IntegralDemo.ipynb -- a step-by-step notebook that can be run on a
     compute node."

### Code examples planned

- E1: fermiMethod comparison 2-liner (new, illustrative)
- E2: SCF_DAMPING comparison (new, 2 lines)
- E3: ETA override on setContact1D (new, 1 line)

### Cross-refs to API

- :class:`gauNEGF.config` (Configuration Reference)
- :func:`gauNEGF.scfE.NEGFE.setContact1D`
- :func:`gauNEGF.scfE.NEGFE.setContactBethe`

### Cross-refs to other guides

- :doc:`contact_choice` (fermiSearch critical rule)
- :doc:`contacts_1d` (ETA and fermiMethod usage)
- :doc:`contacts_bethe` (ETA and TEMPERATURE for metallic contacts)

---

## Guide 5: Workflow Recipes

**File:** docs/source/guides/workflow_recipes.rst
**Length estimate:** Short-Medium (2-3 pages)
**Canonical source:** AuStudies/AuBetheFerrocene.py (WRITER MUST READ), CNanowire.py,
  test_transport_checkpointing.py

### Target audience

Users who have a working SCF and need to compute IV curves, sweep temperature,
or handle long energy-scan runs.

### Sub-sections

1. Recipe: IV curve sweep
   - Content: Loop over voltage list, setVoltage(V, negf.fermi), SCF, calculate_current,
     save per-voltage MAT, log I values.
   - Key example: Lifted from AuBetheFerrocene.py lines 38-61 (voltage loop with resume
     from checkpoint mat file)
   - Notes: How to resume a partial IV sweep from saved MAT files (os.path.exists check)
   - Cross-refs: :func:`gauNEGF.transport.calculate_current`, :class:`gauNEGF.transport.SigmaCalculator`

2. Recipe: Multi-temperature sweep
   - Content: Call setContact1D (or setContactBethe) twice with different T values,
     run independent SCF each time, compare transmission curves.
   - Key example: Lifted from CNanowire.py multi-T pattern (Pattern C)

3. Recipe: Transport checkpointing
   - Content: For long energy scans, use checkpoint_file + checkpoint_interval arguments
     in calculate_transmission so the scan can be resumed after interruption.
   - Key example: Lifted from test_transport_checkpointing.py SigmaCalculator + calculate_transmission
   - Cross-ref: :func:`gauNEGF.transport.calculate_transmission`

4. Recipe: Warm-start SCF from prior run
   - Content: io.loadmat to load density matrix + Fermi energy from a prior SCF run,
     then setDen + setVoltage with saved Fermi to hot-start.
   - Key example: From AuBetheFerrocene.py lines 34-37

5. Pointer to IntegralDemo
   - "For a comparison of the integration approaches (complex contour vs real-axis),
     see examples/IntegralDemo.ipynb."

### Code examples planned

- E1: AuBetheFerrocene.py IV loop (lifted, ~20 lines)
- E2: CNanowire.py multi-T sweep (lifted, ~15 lines)
- E3: test_transport_checkpointing.py checkpoint scan (lifted, ~8 lines)
- E4: AuBetheFerrocene.py warm-start (lifted, ~5 lines)

### Cross-refs to API

- :class:`gauNEGF.transport.SigmaCalculator` (Transport Module)
- :func:`gauNEGF.transport.calculate_transmission` (Transport Module)
- :func:`gauNEGF.transport.calculate_current` (Transport Module)
- :func:`gauNEGF.scfE.NEGFE.setVoltage` (Energy-Dependent NEGF)
- :func:`gauNEGF.scf.NEGF.SCF` (NEGF Base Class)

### Cross-refs to other guides

- :doc:`contacts_1d` (contact setup before recipe)
- :doc:`contacts_bethe` (contact setup before recipe)

---

## Coverage check: research gaps vs outline

| Feature-without-example gap | Disposition |
|-----------------------------|-------------|
| NEGFE.setContactBethe via NEGFE wrapper | COVERED -- contacts_bethe.rst (full guide with production examples) |
| NEGF base class standalone | DEFERRED -- mentioned in contact_choice.rst as "energy-independent contacts" |
| surfG3 vs surfGAt3D decision | DEFERRED -- contact_choice.rst table mentions 3D as "advanced"; no dedicated guide per user decision |
| NEGF.setSigma static sigma | DEFERRED -- contact_choice.rst table row; no dedicated guide per user decision |
| IV curve / voltage sweep | COVERED -- workflow_recipes.rst Recipe 1 (IV curve sweep) |
| fermiSearch for 1D auto-extract | COVERED -- contact_choice.rst critical section + contacts_1d.rst warning + config_tuning.rst table |

All 3 actively user-relevant gaps are covered. 3 gaps deferred per user prioritization.

---

## Cross-ref targets for guide writers (Phase II API index sections)

Use these EXACT Sphinx section titles in :doc: cross-refs within guides:

API index sections (docs/source/api/index.rst):
- Core Modules > NEGF Base Class (gauNEGF.scf)
- Core Modules > Energy-Dependent NEGF (gauNEGF.scfE)
- Core Modules > Density Module (gauNEGF.density)
- Core Modules > Transport Module (gauNEGF.transport)
- Contact Models > Bethe Lattice (gauNEGF.surfGBethe)
- Contact Models > 1D Chain (gauNEGF.surfG1D)
- Contact Models > 3D Contacts (gauNEGF.surfG3D)
- Contact Models > Constant Self Energy (gauNEGF.surfGTester)
- Utilities > Matrix Tools (gauNEGF.matTools)
- Utilities > Integration Tools (gauNEGF.integrate)
- Utilities > Spin Tools (gauNEGF.spinTools)
- Utilities > JIT / Linear Algebra Helpers (gauNEGF.utils)
- Configuration Reference (gauNEGF.config)
- Developer / Extensibility Reference (gauNEGF.protocols)
