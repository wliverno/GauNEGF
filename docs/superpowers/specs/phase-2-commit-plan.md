# Phase II Commit Plan

This file lists draft commit messages for the file groups produced during
Phase II of the GauNEGF documentation overhaul. Claude does not run git;
the user reviews diffs and runs commits in the order suggested below.

Suggested commit order is documented in Task 13; entries appear here in
the order they are produced during execution and may be reordered by the
user.

---

## Commit candidate: Phase II gap inventory

Files: docs/superpowers/specs/phase-2-gaps.md

Suggested message:
docs(spec): add Phase II gap inventory for docstring coverage audit

Per-module docstring status (present/thin/stub/absent), detected
docstring style, and API index registration. Drives Phase II Step 3
writer dispatch + Step 5 API index update.

---

## Commit candidate: Phase II Step 2 decisions

Files: docs/superpowers/specs/phase-2-decisions.md

Suggested message:
docs(spec): record Phase II Step 2 API index decisions

User-confirmed decisions for the 6 previously-unregistered modules
(surfG3D, spinTools, config, protocols, utils, fermiSearch). Drives
the api/index.rst update in Task 9.

---

## Commit candidate: D1 group docstring additions

Files: gauNEGF/density.py

Suggested message:
docs(gauNEGF): add docstring for density.calcEmin (Phase II D1)

Phase II Step 3 writer output applied to D1 group (scf, scfE, density,
transport). Only one gap surfaced: density.calcEmin lacked any
docstring. Numpy-style docstring added covering Parameters, Returns,
Notes (DOS-tolerance iteration behavior). No logic changes.

---

## Commit candidate: D2 group docstring additions

Files: gauNEGF/surfG1D.py

Suggested message:
docs(gauNEGF): add module-level docstring for surfG1D (Phase II D2)

Phase II Step 3 writer output applied to D2 group (surfG1D, surfG3D,
surfGBethe, surfGTester). Only one gap surfaced: surfG1D lacked a
module-level docstring. Added one covering the surfG class purpose,
the three usage patterns, and key features (1D chain Green's function,
JAX JIT acceleration, congruent regularization).

---

## Commit candidate: D3 group docstring additions

Files: gauNEGF/utils.py

Suggested message:
docs(gauNEGF): add docstrings for utils.inv/eig/eigh (Phase II D3)

Phase II Step 3 writer output applied to D3 group (matTools, integrate,
spinTools, utils, protocols, config, fermiSearch). Three gaps surfaced,
all in utils.py: inv, eig, eigh JAX wrapper functions had no
docstrings. Added numpy-style docstrings noting the JAX backend, the
inv-via-solve approach, and the eig vs eigh choice for Hermitian
matrices.

---

## Commit candidate: Phase II verifier corrections

Files: gauNEGF/scf.py, gauNEGF/density.py, gauNEGF/transport.py

Suggested message:
docs(gauNEGF): apply verifier corrections to existing docstrings (Phase II)

Phase II Step 4 postwrite verifier surveyed all 15 modules and the
5 newly-written docstrings. The new ones verified clean. Three
mechanical mismatches surfaced in pre-existing docstrings and were
applied:

- scf.py NEGF class docstring: added missing `section` parameter
  (present in __init__ signature, absent from class docstring).
- density.py densityReal docstring: removed extraneous `maxN`
  parameter (not in signature, not used in body).
- transport.py currentF docstring: added missing `T` parameter
  (in signature, used to pass to current()).

Zero degrading fixes, zero judgment items, zero code bugs. Per
phase-2-corrections.md.

---

## Commit candidate: API index update (Phase II Step 5)

Files: docs/source/api/index.rst

Suggested message:
docs(sphinx): register Phase II module decisions in API index

Adds:
- gauNEGF.surfG3D under Contact Models (between 1D Chain and Constant Self Energy)
- gauNEGF.spinTools under Utilities (Spin Tools subsection)
- gauNEGF.utils under Utilities (JIT / Linear Algebra Helpers subsection,
  flipped from spec default per Phase I usage scan)
- Configuration Reference top-level section for gauNEGF.config
- Developer / Extensibility Reference top-level section for gauNEGF.protocols

fermiSearch is excluded per user decision in Task 3 (deprecated).

Resolves the Phase I "missing modules" gap surfaced by Group C verifier.

---

## Commit candidate: pre-rendered HTML deletion (Phase II Step 6)

Files: 18 deleted -- docs/{index,installation,quickstart,genindex,search}.html,
       docs/objects.inv, docs/searchindex.js,
       docs/api/index.html,
       docs/examples/{IntegralDemo,advanced_examples,ethane,index,silicon_nanowire}.html,
       docs/theory/{best_practices,index,introduction,negf_dft,transport}.html

Suggested message:
docs: remove pre-rendered HTML; rely on GitHub Pages Sphinx build

GitHub Pages now builds from docs/source via a workflow (user
confirmed in Phase II Task 10), so the committed pre-rendered HTML
under docs/*.html, docs/api/*.html, docs/examples/*.html,
docs/theory/*.html, docs/objects.inv, docs/searchindex.js are dead
weight that drift relative to source.

docs/.nojekyll is intentionally preserved per the existing .gitignore
`!docs/.nojekyll` un-ignore rule (zero-byte sentinel).

The .gitignore already has comprehensive patterns covering all of
these (lines 11-25), so no .gitignore edit is needed; future local
sphinx-build runs will not re-introduce them via accidental commit.

---

## Commit candidate: API truth table refresh (Phase II Step 7)

Files: docs/superpowers/specs/api-truth.md
       docs/superpowers/specs/api-truth.phase1-snapshot.md (delete after Phase III if clean)

Suggested message:
docs(spec): refresh API truth table with upgraded extraction conventions

Re-runs extraction with five upgrades surfaced by the Phase I retrospective:
- Class inheritance constructors (NEGFE inherits NEGF)
- Dynamic attributes (NEGF.sigma1/sigma2/sigma12/Gam1/Gam2, NEGFE.g)
- Variable-shape returns (calculate_transmission, calculate_dos)
- Method overloads via duck typing (SigmaCalculator)
- Nested-list parameter conventions (setContactBethe contactList)

Also corrects a Phase I error: surfGBethe section had wrong class names
(surfGTest) -- replaced with actual surfGB and surfGBAt classes from
surfGBethe.py. surfG3D section correct (surfG3, surfGAt3D).

All absent docstrings now 0 (Phase II additions reflected).
Line count: 1023 (was 917).

api-truth.phase1-snapshot.md preserves the Phase I version for
traceability; can be deleted at end of Phase III.

---
