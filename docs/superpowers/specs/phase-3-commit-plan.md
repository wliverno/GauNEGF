# Phase III Commit Plan

This file lists draft commit messages for Phase III outputs.
Claude does not run git; the user stages and commits.

---

## Commit candidate: Phase III research outputs

Files: docs/superpowers/specs/phase-3-test-inventory.md
       docs/superpowers/specs/phase-3-source-inventory.md
       docs/superpowers/specs/phase-3-research-summary.md

Suggested message:
docs(spec): record Phase III research outputs (tests + source contact infrastructure)

R1 surveyed 23 test files across tests/ and ../NEGFTests/ for every
contact-setup call, system topology, configuration deviation, and
workflow pattern. R2 surveyed gauNEGF contact + surfG infrastructure.
Aggregated summary cross-references coverage and surfaces 5 feature-
without-example gaps and 7 expert workflow candidates for the interview.

---

## Commit candidate: Phase III interview notes

Files: docs/superpowers/specs/phase-3-interview-notes.md

Suggested message:
docs(spec): record Phase III interview notes (Round 1, complete)

User decisions: 5 guides (contact_choice, contacts_1d, contacts_bethe,
config_tuning, workflow_recipes). Audience: experts new to gauNEGF.
Key findings: fermiSearch method selection is the top underdocumented
pitfall (1D auto-extract requires fermiSearch or electron count is
non-physical). Production Bethe files found (AuBetheFerrocene.py,
HemeStudies/AuFe.py). IntegralDemo kept as Jupyter notebook.

---

## Commit candidate: Phase III outline

Files: docs/superpowers/specs/phase-3-outline.md

Suggested message:
docs(spec): write Phase III guide outline (5 guides)

Per-guide outline derived from interview + research summary. 5 guides:
contact_choice (decision tree), contacts_1d (1D chain migration),
contacts_bethe (Bethe lattice with AuBetheFerrocene.py as canonical),
config_tuning (fermiSearch table + SCF_DAMPING), workflow_recipes
(IV curve sweep, multi-T, checkpointing). Cross-refs target Phase II
API index sections per phase-2-decisions.md.

---

## Commit candidate: Guide -- Choosing a Contact Type

Files: docs/source/guides/contact_choice.rst
       docs/superpowers/specs/phase-3-todos.md

Suggested message:
docs(guides): add Choosing a Contact Type decision tree guide

Phase III Step 4. Decision table (4 contact types vs. scenarios), critical
warning for 1D auto-extract requiring fermiMethod, and Where to Go Next
cross-refs. 1 synthesized code block. 0 sections flagged for review.

---

## Commit candidate: Guide -- 1D Chain Contacts

Files: docs/source/guides/contacts_1d.rst
       docs/1D_contact_setup_guide.md (DELETED)

Suggested message:
docs(guides): migrate 1D contact setup guide from markdown to Sphinx RST

Phase III Step 4 + Step 5. Migrated and expanded docs/1D_contact_setup_guide.md
to RST. Added three usage patterns section (auto-extract, custom coupling,
full spec) with lifted code from NEGFTests/CNT33.py and CNTCont.py. Added
critical fermiMethod warning for Pattern C. 11 code blocks (2 lifted).
1 section flagged for review (Pattern B synthesized example). Original
markdown deleted.

---

## Commit candidate: Guide -- Bethe Lattice Contacts

Files: docs/source/guides/contacts_bethe.rst

Suggested message:
docs(guides): add Bethe Lattice Contacts guide

Phase III Step 4. Full production guide using AuStudies/AuBetheFerrocene.py
and HemeStudies/AuFe.py as canonical examples. Covers standard and SOC
(spin='g', AuSOC latFile) workflows, warm-start SCF, standalone surfGBAt
testing. 11 code blocks (4 lifted from production files). 0 sections flagged.

---

## Commit candidate: Guide -- Configuration and Tuning

Files: docs/source/guides/config_tuning.rst

Suggested message:
docs(guides): add Configuration and Tuning guide

Phase III Step 4. fermiSearch method stability table (bisect/muller/poly/
secant/predict), SCF_DAMPING guidance, ETA per contact type, TEMPERATURE
for finite-T contacts. Pointer to examples/IntegralDemo.ipynb. 3 synthesized
illustrative code blocks. 0 sections flagged.

---

## Commit candidate: Guide -- Workflow Recipes

Files: docs/source/guides/workflow_recipes.rst

Suggested message:
docs(guides): add Workflow Recipes guide

Phase III Step 4. IV curve sweep (lifted from AuStudies/AuBetheFerrocene.py),
multi-temperature sweep (lifted from NEGFTests/CNanowire.py), transport
checkpointing (lifted from tests/test_transport_checkpointing.py), warm-start
SCF (lifted from AuBetheFerrocene.py). 4 code blocks, all lifted. 0 flagged.

---

## Commit candidate: 1D guide migration

(Bundled with the contacts_1d.rst commit above.)

---

## Commit candidate: Sphinx integration (toctree + cross-refs)

Files: docs/source/guides/index.rst (created),
       docs/source/index.rst (modified),
       docs/source/api/index.rst (modified)

Suggested message:
docs(sphinx): wire Phase III guides into toctree + add API cross-refs

guides/index.rst lists all 5 Phase III guides in a maxdepth-2 toctree.
Top-level index.rst now includes guides/index between Theory and Examples.
api/index.rst gains See-also directives linking Energy-Dependent NEGF,
Transport Module, Bethe Lattice, 1D Chain, and Configuration Reference
to their companion guides.

---

## Commit candidate: Verifier corrections (none needed)

Files: docs/superpowers/specs/phase-3-findings/guides-postwrite.json
       docs/superpowers/specs/phase-3-corrections.md

Suggested message:
docs(spec): record Phase III verifier pass results (all clean)

Postwrite verifier checked all 5 guide RST files against api-truth.md.
Zero findings (mechanical, judgment, code bug, or degrading fix).
All code blocks verified: NEGFE inheritance, setContactBethe nested list,
SigmaCalculator duck-typing, variable-shape transport returns.

---

## Commit candidate: Sphinx build dry-run fixes

Files: (to be determined after user runs sphinx-build -W)
       docs/superpowers/specs/phase-3-sphinx-warnings.md (if warnings found)

Suggested message:
docs(sphinx): fix warnings surfaced by sphinx-build -W dry-run

Phase III Task 16. Sphinx-build -W run on compute node.
[Fill in: N warnings fixed across M files, or "Build succeeded clean
with zero warnings" if no fixes needed.]

---
