# Phase III Interview Notes

Date: 2026-05-11
Round: 1 (complete -- no Round 2 needed; outline shape fully determined)

---

## Batch A: Audience priority

User answer: Experts new to gauNEGF (Recommended)
Implication for outline: Assume NEGF theory knowledge. Focus guides on API, workflow, and
configuration. No need to explain NEGF physics or Dyson equations. Code-first with enough
prose to explain parameter choices.

---

## Batch B: Guide topic confirmation

User selected 4 of 8 candidate topics:
1. Choosing a Contact Type (decision tree)
2. Bethe Lattice Contacts
3. 1D Chain Contacts (migrate existing markdown)
4. Configuration and Tuning

Plus Round 1 Batch C / IV question added:
5. Workflow Recipes (IV curve sweep, checkpointing, multi-temperature)

NOT selected (defer or skip):
- Energy-Independent Contacts (guide #2) -- NEGF.setSigma pattern not prioritized
- 3D Contacts (guide #5) -- surfGAt3D not a priority; decision tree guide will mention when to use
- Spin and SOC Calculations (guide #6) -- not selected; SOC usage in Bethe guide can be a section

IntegralDemo fate: Keep as Jupyter notebook (do not convert). It is a foundational test for
implementation validation, meant to run in Jupyter, not documentation. A brief pointer in docs
is fine (e.g., a note in the workflow guide: "See examples/IntegralDemo.ipynb for integration
comparison."). No writer slot needed.

---

## Batch C: Per-contact-type questions

### NEGFE.setContactBethe -- production examples found

User confirmed setContactBethe IS used in production:
- AuStudies/AuBetheFerrocene.py: setContactBethe([[1,2,3,4,5,6], [36,...,41]], 'AuSOC')
  with spin='g' (generalized / SOC). Also shows full IV curve voltage sweep.
- HemeStudies/AuFe.py: setContactBethe([[1,2,3],[7,8,9]], 'Au') with spin='g'.

Implication: Bethe guide is a FULL production guide. Use AuBetheFerrocene.py as the
canonical code example. Both files need to be read by the Bethe guide writer.

Canonical Bethe workflow (from AuBetheFerrocene.py):
  negf = NEGFE(fn=fn, func='b3lyp', basis='chkbasis', spin=spin, fullSCF=False, route=...)
  negf.setContactBethe([[leftAtoms], [rightAtoms]], 'AuSOC')
  negf.setVoltage(0.0, fermiMethod='poly')
  negf.setDen(A['den'])       # warm start from prior run
  negf.setVoltage(0.0, A['fermi'][0][0])
  negf.SCF(1e-3, 0.02, 200, checkpoint=False)
  # IV sweep follows...

### IV curve sweep -- confirmed as 5th guide

AuBetheFerrocene.py shows a full production IV sweep pattern:
  Vlist = np.concatenate((...)
  for i, V in enumerate(Vlist):
      negf.setVoltage(V, negf.fermi)
      if os.path.exists(matName):  # resume from checkpoint mat file
          A = io.loadmat(matName)
          negf.setDen(A['den'])
          negf.setVoltage(V, A['fermi'][0][0])
      negf.SCF(conv, 0.01, 1000, checkpoint=False)
      negf.saveMAT(matName)
      I, Is = calculate_current(negf.F*har_to_eV, negf.S, SigmaCalculator(negf.g), negf.fermi, V, spin=spin)

This is the primary "Workflow Recipes" guide content. Also shows SigmaCalculator(negf.g) usage
and calculate_current with spin='g' (spin-resolved current).

### Contact pitfalls to feature prominently

User selected:
1. symmetrize_contacts=True vs False
   - Critical for parity-symmetric systems (CNT armchair, sp-chain).
   - Missing this causes incorrect Fermi level search (breaks symmetry).

2. CRITICAL NEW INSIGHT (user provided, NOT in any existing doc):
   "For 1D contacts with indices only you must use the fermiSearch methods or your electron
   count will not be physical."
   
   Expansion (from user): When using Bethe contacts, the Fermi energy from the Bethe lattice
   itself is available and can be used directly (statically). For 1D contacts specified with
   only contact atom indices (auto-extract pattern, no explicit tau/alpha matrices), the Fermi
   energy MUST be found via a Fermi search method (bisect, muller, poly, secant, predict)
   because no precomputed Fermi reference is available -- using a static Fermi energy with
   auto-extract 1D contacts gives non-physical electron counts.

   This rule needs to appear prominently in:
   - Guide 1 (Choosing a Contact Type): "If using 1D auto-extract, you must set fermiMethod"
   - Guide 3 (1D Chain guide): Warning box in the auto-extraction section
   - Guide 4 (Bethe guide): Note that Bethe Fermi energy can be used static
   - Guide 5 (Config & Tuning): Fermi search method selection table

User did NOT select: neList vs muList confusion, ETA tuning, Bethe latFile choice.
Include these as secondary items, not featured warnings.

---

## Batch D: Config tuning

### CRITICAL CORRECTION

User clarified: fermiMethod is a FERMI SEARCH ALGORITHM, NOT an integration method.
ANT quadrature is a complex contour integration method (internal to gauNEGF).
These are completely different:
- fermiMethod: how to find the Fermi energy (bisect, muller, poly, secant, predict)
- integration method: complex contour (ANT-style) vs real-axis (used internally)

This was a research-summary error -- the phase-3-research-summary.md config deviation table
incorrectly described fermiMethod as "Polynomial Fermi fit instead of ANT quadrature".
The correct description: "fermiSearch method: poly (balanced speed/stability) vs default (bisect)."

Fermi search method ranking (from user):
- bisect: most stable (use when stability is needed)
- muller: stable (Muller's method)
- poly: balanced stability and speed (commonly used in production)
- secant: much faster but unstable (use with caution)
- predict: very fast but very unstable (advanced use only)

### Config constants that drive real confusion

User selected: SCF_DAMPING (0.02 default)
- Production runs use 0.01 or 0.1 (seen in AuBetheFerrocene.py)
- When to tighten (0.01) vs loosen (0.1) is not documented anywhere

User additional note confirms: fermiSearch method choice is hard and underdocumented.

Config & Tuning guide should cover:
1. fermiSearch method selection (bisect vs poly vs muller vs secant vs predict)
2. SCF_DAMPING -- when to tighten vs loosen
3. The "1D auto-extract requires fermiSearch" rule (see Batch C critical insight)
4. ETA (mention but not the top confusion source)
5. TEMPERATURE (mention for finite-T contact setup)

---

## Batch E: Existing-docs meta-feedback

1D guide (docs/1D_contact_setup_guide.md): "Mostly fine -- just needs RST formatting"
- User selected this option (format conversion is the main task)
- Implication: migrate with format conversion + add auto-extraction section (missing in current guide)
  + add pointer to Workflow Recipes for IV sweep / multi-temperature

Other Claude-generated docs: none flagged by user (no specific concerns raised)
Plan: verifier pass will catch any code errors during Task 14-15.

---

## Batch F: IntegralDemo fate

Keep as Jupyter notebook. No writer slot. Brief mention in docs:
"See examples/IntegralDemo.ipynb for a step-by-step comparison of integration approaches."
Add this reference in the workflow recipes guide or config & tuning guide.

---

## Round 1 outline shape (final)

5 guides approved:

1. **contact_choice.rst** -- Choosing a Contact Type (decision tree)
   - Decision tree: energy-dependent vs static, then 1D vs Bethe vs 3D
   - Critical rule: "1D auto-extract requires fermiSearch method"
   - Cross-refs to all contact guides
   - Medium length

2. **contacts_1d.rst** -- 1D Chain Contacts (migrate existing markdown)
   - Format conversion from existing guide (mostly fine)
   - Add: auto-extraction section (missing in current guide)
   - Add: pointer to Workflow Recipes for IV/multi-T
   - Code examples lifted from CNT33.py, CNTCont.py, CNanowire.py
   - Medium length

3. **contacts_bethe.rst** -- Bethe Lattice Contacts
   - Full production guide
   - Canonical: AuBetheFerrocene.py (SOC, spin='g', IV sweep)
   - Also: HemeStudies/AuFe.py (simpler, spin='g', no IV)
   - SOC section: readBetheParams with soc_p/soc_d, spin='g'
   - Note about surfGBAt direct construction for testing
   - Medium-Long length

4. **config_tuning.rst** -- Configuration and Tuning
   - fermiSearch method table (bisect/muller/poly/secant/predict with stability ratings)
   - SCF_DAMPING: when to tighten vs loosen
   - ETA: contact-type recommended values
   - "1D auto-extract requires fermiSearch" rule
   - Pointer to IntegralDemo.ipynb for integration method comparison
   - Short-Medium length

5. **workflow_recipes.rst** -- Workflow Recipes
   - IV curve sweep: lifted from AuBetheFerrocene.py
   - Multi-temperature sweep: lifted from CNanowire.py
   - Transport checkpointing: lifted from test_transport_checkpointing.py
   - Warm-start SCF (setDen from prior run): from AuBetheFerrocene.py
   - Short-Medium length

---

## Open questions for Round 2

NONE -- outline shape is fully determined. No Round 2 needed.

Key facts for Task 7:
- 5 guides, not 8
- Contact-choice guide has the critical "1D auto-extract requires fermiSearch" rule
- Bethe guide uses AuBetheFerrocene.py as canonical (writer must read this file)
- Config guide corrects the fermiMethod = fermiSearch (not ANT quadrature) confusion
- Workflow guide centers on IV sweep from AuBetheFerrocene.py
- fermiSearch methods: bisect > muller > poly > secant > predict (stability order)
- SCF_DAMPING: underdocumented; default 0.02, production range 0.01 to 0.1
