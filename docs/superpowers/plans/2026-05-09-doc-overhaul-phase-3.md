# GauNEGF Doc Overhaul - Phase III Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a comprehensive set of user-facing guides covering every
contact type, configuration setting, and end-to-end workflow that gauNEGF
supports. Migrate the existing 1D markdown guide into Sphinx. Decide and
execute on the IntegralDemo notebook fate. Validate the rebuilt Sphinx
docs build cleanly via dry-run.

**Architecture:** 2 parallel research subagents (haiku) survey tests/ and
gauNEGF/ contact code; main session conducts a multi-batch interview
with the user (capped at 2 follow-up rounds); main session drafts the
guide outline; user approves; 3-4 parallel guide-writer subagents per
batch produce .rst guide content; main session merges drafts; verifier
pass on all code blocks; main session applies corrections; sphinx-build
dry-run on a compute node validates clean build; user runs all commits.

**Tech Stack:** haiku-class subagents, AskUserQuestion for interview +
outline approval, Edit/Write for application, the refreshed truth table
from Phase II Task 12 as API ground truth, sphinx-build executed by user
on a compute node (login node lacks the env).

**Parent spec:** `docs/superpowers/specs/2026-05-09-doc-overhaul-design.md`
(Phase III Steps 1-8, plus the Phase I Retrospective section).

**Hard rules** (carried from spec, repeated):

- Claude does NOT run `git commit`, `git add`, `git push`, or any git
  mutation. User stages and commits everything.
- Subagents are dispatched with `model: "haiku"` per global user pref.
- Verification is static + AST. No execution of doc snippets by Claude.
- The user runs `sphinx-build` on a compute node (login node lacks the
  env). Phase III plan tells the user the exact command + expected
  output.
- ASCII only in any file Claude writes.
- Phase III does NOT begin until Phase II commits are landed (Task 1
  verifies, with user-waiver fallback).

**Phase II lessons baked into this plan (carryover):**

- All subagent prompts include the Verifier Guardrails Block (SP-5 from
  Phase II plan).
- Quality-degrading-fix detection (SP-6 from Phase II) gates every
  mechanical-fix application in Tasks 11 and 15.
- Judgment items surface to user EARLY (preview before application).
- Refreshed truth table from Phase II Task 12 is the API ground truth.
- Main session retraction culture: when verifier and existing/proposed
  doc disagree about API shape and the doc is internally coherent,
  trust the doc and ask the user.

Phase II execution notes (resolved):
- The upgraded extraction prompt caught all five categories. The truth
  table refresh had to be done manually (haiku agent lacked sufficient
  context to handle 15 modules + 5 new annotation types in one pass).
  For Phase III: split large agent tasks into read-phase and write-phase.
- SP-6 (degrading-fix detection) triggered zero times -- the patterns
  listed are appropriate and no new patterns surfaced.
- The writer JSON schema (kind/anchor_text/docstring) via SP-4 worked
  cleanly for all 3 writer groups. No SP-4 revision needed.

---

## Shared procedures

### SP-1, SP-2, SP-3 (carry from Phase I/II)

Reused verbatim. Apply mechanical fix via Edit; re-verify file group;
append commit-plan entry.

### SP-5 (Verifier Guardrails Block, carry from Phase II -- SHORTENED)

All 5 blind spots are now resolved in the refreshed truth table
(api-truth.md). The shortened block for Phase III is:

```
=== VERIFIER GUARDRAILS (DO NOT VIOLATE) ===

The API truth table (docs/superpowers/specs/api-truth.md) now
explicitly documents all of the following. Trust the table:

1. INHERITANCE CONSTRUCTORS: table has "Inherits constructor from:"
   lines. NEGFE inherits NEGF.__init__ -- NEGFE('molecule', basis=...)
   is valid.
2. DYNAMIC ATTRIBUTES: table has "Dynamic attributes:" sections.
   NEGF has sigma1/sigma2/sigma12/Gam1/Gam2 (set by setSigma).
   NEGFE has self.g (set by setContactBethe, setContact1D, setSigma).
3. VARIABLE-SHAPE RETURNS: table has "Variable returns:" lines.
   calculate_transmission and calculate_dos both have spin-conditional
   return shapes. Do NOT flag tuple-unpacking as a bug.
4. OVERLOADS: table has "Overloads:" lines. SigmaCalculator accepts
   ndarray or surfG-type object. Do NOT flag SigmaCalculator(negf.g).
5. NESTED-LIST PARAMS: table has "Parameter shapes:" lines.
   setContactBethe contactList is [[atoms...], [atoms...]] per contact.

If you suspect a discrepancy between guide code and the truth table,
report it as a finding -- do NOT auto-fix.

=== END VERIFIER GUARDRAILS ===
```

### SP-6 (quality-degrading-fix detection, carry from Phase II)

Reused verbatim. No new degrading-fix patterns surfaced in Phase II
(SP-6 triggered zero times). Existing patterns remain correct.

### SP-7 (NEW for Phase III): Apply guide writer's draft to .rst

Used by Task 11 to merge guide-writer output into
`docs/source/guides/*.rst` (or other Sphinx source paths per outline).
Writer agents produce one JSON per guide containing the full .rst
content. Application:

- If guide file does not exist: use Write tool to create it from the
  writer's `content` field.
- If guide file exists (e.g., the migrated 1D guide): use Edit tool to
  replace specific sections per the writer's `edits` array.
- Sections flagged by writer with `needs_review: true`: insert
  `.. note:: This section needs author review.` at the top of the
  section in the .rst output. Surface to user during Task 15 verifier
  application.

### SP-8 (NEW for Phase III): Format an interview question batch

Used by Tasks 5-6 to construct AskUserQuestion calls. Per spec line
~376-405, the interview is multi-batch with question categories. Each
batch covers one category at a time:
- Audience priority
- Guide topic confirmation
- Per-contact-type questions (split by type if needed)
- Settings explanations
- Existing-docs meta-feedback
- IntegralDemo fate

Each batch is 1-3 AskUserQuestion calls (each with up to 4 questions).
Cap on follow-up rounds: 2 (per spec). After 2 follow-ups, if outline
shape still unresolved, STOP and surface "structural problem -- need
to replan with user before proceeding to Step 3".

---

## File Structure (what gets created in this phase)

| Path | Created in | Purpose |
|------|------------|---------|
| `docs/superpowers/specs/phase-3-test-inventory.md` | Task 2 | R1 output: every contact-setup call + topology + setting + workflow pattern across tests/ and ../NEGFTests/. |
| `docs/superpowers/specs/phase-3-source-inventory.md` | Task 2 | R2 output: every public contact-setup entry point + surfG class interface across gauNEGF source. |
| `docs/superpowers/specs/phase-3-research-summary.md` | Task 3 | Aggregated research summary; cross-references R1 patterns to R2 source code. |
| `docs/superpowers/specs/phase-3-interview-notes.md` | Tasks 5-6 | User answers + main-session inferences from interview batches. |
| `docs/superpowers/specs/phase-3-outline.md` | Task 7 | Approved outline: title + sub-sections + key code examples + cross-refs per guide. |
| `docs/superpowers/specs/phase-3-guides-output/<guide>.json` | Tasks 9-10 | One JSON per guide writer (intermediate). |
| `docs/superpowers/specs/phase-3-todos.md` | Task 11 | Guides where writer flagged sections as `needs_review`. |
| `docs/superpowers/specs/phase-3-findings/guides-postwrite.json` | Task 14 | Verifier pass output on guide code blocks. |
| `docs/superpowers/specs/phase-3-corrections.md` | Task 15 | Verifier corrections + judgment-item preview. |
| `docs/superpowers/specs/phase-3-sphinx-warnings.md` | Task 16 | Sphinx-build dry-run warnings (user posts back). |
| `docs/superpowers/specs/phase-3-commit-plan.md` | Task 17 | Draft commit messages for Phase III commits. |

Files MODIFIED or CREATED in working tree:
- `docs/source/guides/*.rst` -- one per approved guide (likely 6-8 files)
- `docs/source/guides/index.rst` -- new toctree
- `docs/source/index.rst` -- toctree update to include guides/index
- `docs/source/api/index.rst` -- cross-refs added between automodule
  entries and relevant guides
- (possibly) `docs/source/examples/IntegralDemo.rst` if interview chose
  rewrite-as-rst path

Files DELETED (working tree):
- `docs/1D_contact_setup_guide.md` -- migrated to
  `docs/source/guides/contacts_1d.rst` and original deleted (Task 12)
- (possibly) `examples/IntegralDemo.ipynb` -- if interview chose
  replace path; otherwise kept

---

## Task 1: Verify prerequisites + Phase II close-out

**Files:** none modified; read-only checks.

- [ ] **Step 1: Confirm Phase II commits landed (or user waiver)**

Run:
```bash
cd /gscratch/anantram/willll/NEGFCode && git log --oneline -25
```
Expected: recent commits include Phase II work (gap inventory,
docstring additions per group, API index update, HTML deletion,
truth-table refresh).

If working tree dirty with Phase II edits not committed: ASK user via
AskUserQuestion (same pattern as Phase II Task 1 Step 1).

- [ ] **Step 2: Confirm refreshed truth table is the Phase II version**

Run:
```bash
grep -c "Dynamic attributes:\|Variable returns:\|Overloads:\|Inherits constructor" \
  /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
```
Expected: > 0 (refreshed table has these markers; Phase I version did
not). If 0: STOP -- you are reading the Phase I snapshot. Surface to
user.

- [ ] **Step 3: Confirm Phase II artifacts exist**

Run:

```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-{gaps,decisions,corrections,commit-plan}.md
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md \
        /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.phase1-snapshot.md
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-writer-output/
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-findings/
```
Expected: all present. writer-output has D1.json, D2.json, D3.json.
findings has group-D-postwrite.json. phase-2-todos.md and
phase-2-skipped.md do NOT exist (Phase II left no TODOs and no
degrading-fix skips).

- [ ] **Step 4: Confirm research input paths**

Run:
```bash
ls -d /gscratch/anantram/willll/NEGFCode/tests /gscratch/anantram/willll/NEGFTests
```
Expected: both directories exist.

- [ ] **Step 5: Read Phase II carryover artifacts**

Use Read tool on:
- `docs/superpowers/specs/api-truth.md` (refreshed)
- `docs/superpowers/specs/phase-2-decisions.md` (drives cross-ref
  targets in Task 7 outline)
- `docs/superpowers/specs/phase-2-todos.md` (if exists -- modules with
  unfilled docstrings need extra interview attention)
- `docs/superpowers/specs/phase-1-judgment-fixes.md` (existing user
  decisions about API patterns)

Key Phase II findings for Phase III:
- All 5 docstring gaps filled (calcEmin, inv, eig, eigh, surfG1D
  module-level). No TODOs remain; Batch G of the interview is SKIP.
- API index now includes surfG3D (Contact Models), spinTools and utils
  (Utilities), config (Configuration Reference), protocols (Developer
  Reference). fermiSearch excluded (deprecated). These section names
  are the cross-ref targets for guide See-also directives in Task 13.
- phase-2-judgment-fixes.md does NOT exist -- all verifier corrections
  were mechanical (no user judgment needed in Phase II).
- phase-1-judgment-fixes.md DOES exist; read it for prior API decisions.

- [ ] **Step 6: Create research output + writer-output + findings dirs**

Run:
```bash
mkdir -p /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-guides-output
mkdir -p /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-findings
```

---

## Task 2: Dispatch 2 parallel research agents

**Files:**
- Create: `docs/superpowers/specs/phase-3-test-inventory.md`
- Create: `docs/superpowers/specs/phase-3-source-inventory.md`

- [ ] **Step 1: Dispatch R1 (test inventory) and R2 (source inventory) in parallel**

Use the Agent tool TWICE in ONE assistant message. Both
`subagent_type: "general-purpose"`, `model: "haiku"`,
`run_in_background: false`.

**R1 prompt (test inventory):**

```
You are surveying the gauNEGF test suites to extract every workflow
pattern, contact setup, and configuration deviation that the user
actually exercises.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. API truth table: docs/superpowers/specs/api-truth.md (READ FIRST)
2. Test trees:
   - /gscratch/anantram/willll/NEGFCode/tests/*.py and *.ipynb
   - /gscratch/anantram/willll/NEGFTests/*.py and *.ipynb
   - Subdirectories of both, recursively.

[INSERT VERIFIER GUARDRAILS BLOCK HERE -- verbatim from SP-5]

Method, applied to every test file:

1. Read the file in full.
2. Extract:
   a. **Contact setups**: every call to setSigma, setContacts,
      setContact1D, setContactBethe. Capture full args, surrounding
      comments, and the system geometry context (e.g., "11-cell CNT
      armchair", "ethane molecule with carbon contacts").
   b. **System topologies**: number of atoms / cells, basis set, spin
      mode, special features (SOC, periodic, finite-T).
   c. **Configuration deviations**: any line that overrides a
      `gauNEGF.config` constant (e.g., setting ETA to 1e-6 instead of
      default).
   d. **Workflow patterns**: ordering of steps -- DFT cluster ->
      matrix extraction -> NEGF setup -> SCF -> transmission/current.
      Note variants (Harris guess vs full SCF, voltage sweep, IV
      curve, etc.).

3. Group findings by contact type (1D, Bethe, 3D, constant sigma,
   Sigma-NEGFE) and workflow shape.

Output: write to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-test-inventory.md using Write tool.

File format:

# Phase III Test Inventory

Date: <today>
Source: tests/ + ../NEGFTests/, recursively

## Contact-setup call patterns

### 1D Chain (setContact1D)

| File | System | Args (key ones) | Workflow notes |
|------|--------|-----------------|----------------|
| ... | ... | ... | ... |

### Bethe Lattice (setContactBethe)

(same shape)

### 3D (surfG3D / NEGFE with surfGB)

(same shape)

### Energy-Independent (setSigma constant)

(same shape)

## Workflow patterns

### Pattern A: Full SCF + transmission (most common)

(One paragraph + key code skeleton lifted near-verbatim from one
representative test.)

### Pattern B: Harris guess + transmission

(...)

### Pattern C: IV curve sweep

(...)

(More patterns as found.)

## Configuration deviations

| Constant | Default | Test override | Why (from comment if available) |
|----------|---------|---------------|---------------------------------|
| ETA | 1e-9 | 1e-3 | Speed up convergence |
| ... | ... | ... | ... |

## Notable test files for reference

(List of 5-10 most "canonical" test files that span the major
patterns. These are the files guide writers should preferentially
lift code from.)

ASCII only. Use Read for inputs and Write for the output.

When done, return a one-paragraph summary: total tests scanned,
contact types found, most common workflow pattern, top 3 most
important config deviations.
```

**R2 prompt (source inventory):**

```
You are inventorying the gauNEGF source-side contact infrastructure
to map "what the package can do" against "what the tests exercise"
(R1 output).

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. API truth table: docs/superpowers/specs/api-truth.md (READ FIRST)
2. Source files:
   - gauNEGF/scf.py
   - gauNEGF/scfE.py
   - gauNEGF/surfG1D.py
   - gauNEGF/surfG3D.py
   - gauNEGF/surfGBethe.py
   - gauNEGF/surfGTester.py
   - gauNEGF/protocols.py

[INSERT VERIFIER GUARDRAILS BLOCK HERE -- verbatim from SP-5]

Method:

1. For each contact-setup entry point (NEGF.setSigma, NEGF.setContacts,
   NEGFE.setSigma, NEGFE.setContact1D, NEGFE.setContactBethe), document:
   - Full signature
   - Which surfG class it instantiates internally
   - What configuration constants it consumes
   - What attributes it sets on self (per Phase II truth-table dynamic
     attributes data)
   - When to use it (one-line plain-English summary inferred from the
     code body and docstring)

2. For each surfG* class, document:
   - Constructor args
   - Key methods (sigma, sigmaTot, crossTermQ, setF, etc.)
   - What physical scenario the class is meant for
   - How it handles SOC / spin / temperature

3. Cross-reference: for each contact entry point in (1), note the
   surfG class it uses, and for each surfG class in (2), note which
   entry point(s) instantiate it.

Output: write to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-source-inventory.md using Write tool.

File format:

# Phase III Source Inventory

Date: <today>
Source: gauNEGF/ contact + surfG infrastructure

## Contact entry points

### NEGF.setSigma

(One section per entry point with the fields above.)

### NEGF.setContacts

(...)

### NEGFE.setContact1D

(...)

(etc.)

## surfG class interfaces

### surfG (gauNEGF.surfG1D)

(One section per class.)

### surfGB / surfGBAt (gauNEGF.surfG3D)

(...)

(etc.)

## Cross-reference table

| Entry point | surfG class used | Configuration consumed |
|-------------|------------------|------------------------|
| NEGFE.setContact1D | surfG | ETA, TEMPERATURE |
| ... | ... | ... |

ASCII only. Use Read for inputs and Write for the output.

When done, return a one-paragraph summary: total entry points, total
surfG classes, the "decision tree shape" the user will see (which
entry point for which scenario).
```

- [ ] **Step 2: Wait for both research agents to return**

Both `run_in_background: false` -- parallel dispatch blocks until
complete.

If either fails: re-dispatch only the failed one.

- [ ] **Step 3: Validate outputs exist and are non-trivial**

Run:
```bash
wc -l /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-test-inventory.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-source-inventory.md
```
Expected: both > 50 lines. If either < 30, the agent likely produced a
stub. Re-dispatch.

- [ ] **Step 4: No commit (intermediate; rolls into Task 3 commit)**

---

## Task 3: Aggregate research outputs

**Files:**
- Create: `docs/superpowers/specs/phase-3-research-summary.md`

- [ ] **Step 1: Read both inventories**

Use Read tool on `phase-3-test-inventory.md` and
`phase-3-source-inventory.md`.

- [ ] **Step 2: Cross-reference and identify gaps**

For each entry point in R2 (source), find the matching test patterns
in R1. Note any entry point with NO test coverage (these are
"feature-without-example" gaps for the interview to confirm).

For each test pattern in R1, find the entry point(s) in R2. Note any
test pattern that uses an undocumented or hard-to-find combination
(these are "expert workflow" candidates for guide writers to feature).

- [ ] **Step 3: Write phase-3-research-summary.md**

Use Write tool. Format:

```markdown
# Phase III Research Summary

Date: <today>
Source: phase-3-test-inventory.md + phase-3-source-inventory.md

## Coverage matrix

| Contact entry point | Tested in | Documented in (current) | Test exhaustiveness |
|--------------------|-----------|--------------------------|---------------------|
| NEGFE.setContact1D | <test files> | docs/1D_contact_setup_guide.md (markdown) | High |
| NEGFE.setContactBethe | <test files> | docs/source/theory/best_practices.rst (snippet) | Medium |
| ... | ... | ... | ... |

## Workflow pattern frequency

(Patterns ranked by occurrence count in tests, hinting at user
priority.)

## Feature-without-example gaps

(Entry points with no test coverage. Interview asks user whether
these are real features to document or implementation details to
exclude.)

## Expert workflow candidates

(Test patterns showcasing notable combinations -- SOC + 3D contact,
voltage sweep with full SCF reload, etc.)

## Configuration deviation patterns

(Most common config overrides, with frequency. Drives the
"Configuration & Tuning Guide" content.)
```

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append to `docs/superpowers/specs/phase-3-commit-plan.md` (create if
needed):

```
## Commit candidate: Phase III research outputs

Files: docs/superpowers/specs/phase-3-test-inventory.md
       docs/superpowers/specs/phase-3-source-inventory.md
       docs/superpowers/specs/phase-3-research-summary.md

Suggested message:
docs(spec): record Phase III research outputs (tests + source contact infrastructure)

R1 surveyed tests/ and ../NEGFTests/ for every contact-setup call,
system topology, configuration deviation, and workflow pattern.
R2 surveyed gauNEGF/ contact + surfG infrastructure. Aggregated
summary cross-references coverage and surfaces feature-without-
example gaps + expert workflow candidates for the interview.
```

- [ ] **Step 5: Hand off**

Tell user: "Research complete. <N> contact patterns mapped, <M> config
deviations cataloged, <K> feature-without-example gaps. Interview
batches next."

---

## Task 4: Interview prep

**Files:** none modified (read + drafting in main session).

- [ ] **Step 1: Read research summary + Phase II carryover**

Read `phase-3-research-summary.md`. Re-read (already in context from
Task 1):
- `phase-2-decisions.md` (cross-ref targets)
- `phase-2-todos.md` if exists (modules with unfilled docstrings)
- Existing `docs/1D_contact_setup_guide.md` (for migration context)

- [ ] **Step 2: Draft interview question batches**

Per spec line ~376-405, the interview covers these categories. For each
category, prepare 1-3 AskUserQuestion calls (each up to 4 questions
with up to 4 options).

Phase II output incorporated below in each batch.

**Batch A: Audience priority (1 question)**
- "Who is the primary audience for these guides? Researchers new to
  NEGF / Experts new to gauNEGF / Contributors / All three (with
  different sections marked for each)."

**Batch B: Guide topic confirmation (1-2 questions)**
- Present the spec's best-guess guide list (8 items). User confirms,
  adds, removes.
- Follow-up: "If we add a guide, what topic? If we remove, which?"

Phase II created a Configuration Reference automodule page (gauNEGF.config
in docs/source/api/index.rst). Ask the user how they want to split content:
the automodule page covers the public API (shard_array + module constants);
the Config & Tuning Guide should cover "when to change what and why."

**Batch C: Per-contact-type questions (3-4 questions)**
- "When do you reach for Bethe vs 1D vs 3D vs constant sigma?"
- "What pitfalls were learned the hard way per contact type?"
- "Canonical worked example per type?"

Phase II API index decisions (confirmed, include in Batch C framing):
surfGBethe (Bethe Lattice), surfG1D (1D Chain), surfG3D (3D Contacts),
surfGTester (Constant Self Energy) -- all under Contact Models. Ask
whether these section names match the user's mental model for guide
content placement, and whether surfG3D has enough real-world use to
warrant its own full guide vs. a section in contacts_bethe.rst.

**Batch D: Settings explanations (1-2 questions)**
- "Which config.py settings have been tuned in real work?"
- "What's the story behind SCF_DAMPING=0.02 in the README?"

Phase II left no placeholders in config.py -- the module has one public
function (shard_array) and module-level constants (ETA, TEMPERATURE,
SCF_CONVERGENCE_TOL, SCF_DAMPING, etc.). Focus Batch D questions on
which constants users have tuned in practice and what drove the defaults.

**Batch E: Existing-docs meta-feedback (1 question)**
- "What does the existing 1D_contact_setup_guide.md get wrong? Known
  soft spots?"
- "Any other Claude-generated docs in the repo that need extra
  scrutiny during migration?"

**Batch F: IntegralDemo fate (1 question, with all 3 outcomes pre-staged)**
- "examples/IntegralDemo.ipynb is old and Sphinx-Jupyter integration
  is spotty. Three options:"
  - Rewrite as .rst (writer slot reserved in Task 9-10)
  - Replace with fresh integration-methods .rst guide (writer slot reserved)
  - Keep / re-execute later (no writer slot needed; user re-executes
    on compute node post-Phase III)

**Batch G: TODO modules (SKIP)**
Phase II filled all 5 docstring gaps with no TODOs remaining. Batch G
does not apply -- skip it entirely.

- [ ] **Step 3: No commit (drafting; user input drives next task)**

---

## Task 5: Conduct interview Round 1

**Files:**
- Create: `docs/superpowers/specs/phase-3-interview-notes.md`

- [ ] **Step 1: Send Batches A, B, C in sequence**

Use AskUserQuestion three times (one per batch). Wait between calls
for user response. Record all answers verbatim into a working notes
buffer (will be written to file in Step 3).

For Batch C, if user reveals a contact type with notable depth, plan
to follow up in Round 2 (Task 6) for more detail.

- [ ] **Step 2: Send Batches D, E, F, G**

Same pattern. Batch G is conditional -- skip if no TODO modules
existed in Phase II.

- [ ] **Step 3: Write phase-3-interview-notes.md (Round 1 draft)**

Use Write tool. Format:

```markdown
# Phase III Interview Notes

Date: <today>
Round: 1

## Batch A: Audience priority

User answer: <verbatim>
Implication for outline: <one sentence>

## Batch B: Guide topic confirmation

(Per question, user answer + main-session interpretation.)

(... etc per batch.)

## Round 1 outline shape (preliminary)

Based on Round 1 answers, the outline will look like:
- <Guide name>: <sub-sections>
- ...

## Open questions for Round 2 (if any)

- <question>
- ...
```

- [ ] **Step 4: Decide whether Round 2 is needed**

If outline shape from Round 1 is unambiguous (every guide has clear
content + structure): SKIP Task 6 (no Round 2). Mark Task 6 complete.

If 1-3 specific questions remain about depth or structure: PROCEED to
Task 6 with those targeted follow-ups.

If MORE THAN 3 unanswered questions remain: per spec cap (2 follow-up
rounds), Round 2 is not enough. STOP and surface to user:
"Interview revealed <N> structural questions. The outline is not
converging. Recommend pausing Phase III to revise the spec or re-scope
guides before continuing."

- [ ] **Step 5: No commit yet (Task 6 may add to interview notes)**

---

## Task 6: (Conditional) Conduct interview Round 2

**Files:**
- Update: `docs/superpowers/specs/phase-3-interview-notes.md`

Skip this task entirely if Task 5 Step 4 marked "no Round 2 needed".

- [ ] **Step 1: Send up-to-2 follow-up AskUserQuestion calls**

Each targeting 1-4 specific Round 1 ambiguities.

- [ ] **Step 2: Append Round 2 notes to phase-3-interview-notes.md**

Add a "## Round 2" section with question-answer pairs.

- [ ] **Step 3: Re-decide outline shape**

If outline now unambiguous: proceed to Task 7.

If still unclear after 2 rounds (per spec cap): STOP. "Interview
exceeded 2-round cap without converging on outline. Pausing Phase III
for replan."

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append (covering both rounds):
```
## Commit candidate: Phase III interview notes

Files: docs/superpowers/specs/phase-3-interview-notes.md

Suggested message:
docs(spec): record Phase III interview notes (Round 1 + 2)

User decisions on audience, guide topics, contact-type depth,
configuration explanation depth, existing-docs soft spots, and
IntegralDemo fate. Drives Task 7 outline draft.
```

---

## Task 7: Write outline doc

**Files:**
- Create: `docs/superpowers/specs/phase-3-outline.md`

- [ ] **Step 1: Read all carryover artifacts**

Use Read tool on:
- `phase-3-research-summary.md`
- `phase-3-interview-notes.md`
- `phase-2-decisions.md` (for cross-ref targets)
- `api-truth.md` (refreshed)

- [ ] **Step 2: Draft per-guide outline**

For each guide approved in the interview (typically 6-8 items), draft:

```markdown
## Guide: <title>

### Target audience

(From Batch A.)

### Sub-sections

1. <section name>: <one-sentence content summary>
   - Key example: lifted from <test file:line range> or new
   - Cross-refs: <list of API symbols and other guides>
2. ...

### Code examples planned

(Each example: source = test file or new; status = lifted /
adapted / new; verified-against-truth-table after writing.)

### Cross-refs to API

List automodule directives the guide references. Phase II API index
sections (use these exact section titles for cross-refs):
- Core Modules: NEGF Base Class, Energy-Dependent NEGF, Density Module, Transport Module
- Contact Models: Bethe Lattice (surfGBethe), 1D Chain (surfG1D), 3D Contacts (surfG3D), Constant Self Energy (surfGTester)
- Utilities: Matrix Tools, Integration Tools, Spin Tools, JIT / Linear Algebra Helpers
- Configuration Reference (gauNEGF.config)
- Developer / Extensibility Reference (gauNEGF.protocols)

### Cross-refs to other guides

(For decision-tree guides especially.)

### Length estimate

Short (1-2 pages) / Medium (3-5) / Long (6+).
```

Cross-ref targets confirmed from Phase II API index:
"Configuration Reference" EXISTS (gauNEGF.config).
"Developer / Extensibility Reference" EXISTS (gauNEGF.protocols).
Both are top-level sections in docs/source/api/index.rst.

- [ ] **Step 3: Add a top-level navigation guide**

Per spec, Guide #1 is "Choosing a Contact Type" -- a decision tree.
Make sure the outline shows this guide explicitly with branches and
which downstream guide each branch points to.

- [ ] **Step 4: Cross-check outline against research summary**

For each "feature-without-example gap" from R1 research: confirm the
outline either (a) covers it, (b) explicitly defers it, or (c) marks
it as "no guide planned -- consult source / API page".

- [ ] **Step 5: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: Phase III outline

Files: docs/superpowers/specs/phase-3-outline.md

Suggested message:
docs(spec): write Phase III guide outline

Per-guide outline derived from interview + research summary.
Covers <N> guides + IntegralDemo decision. Cross-refs target
existing automodule sections per phase-2-decisions.md.
```

- [ ] **Step 6: Hand off**

Tell user: "Outline drafted. <N> guides planned. Approval gate next."

---

## Task 8: User approval gate on outline

**Files:** none modified.

This is the most important checkpoint in Phase III per spec. If the
outline is wrong, every writer downstream produces wasted work.

- [ ] **Step 1: Present the outline to user for review**

Tell user: "Outline written to `phase-3-outline.md`. Please review
end-to-end. Any changes before guide writers run?"

Wait for user response.

- [ ] **Step 2: If user requests changes, apply them**

For changes affecting structure (add/remove a guide, restructure
sub-sections): edit `phase-3-outline.md` per user input. If the
changes are large enough that the outline-cross-research-summary
check (Task 7 Step 4) changes outcome, RE-RUN that check.

For minor edits (rename, reword): apply directly.

- [ ] **Step 3: Get explicit approval**

Use AskUserQuestion: "Outline approved as written? Any final tweaks?"
Header: "Outline approval"
multiSelect: false
options:
1. "Approved -- proceed to writer dispatch (Recommended)"
2. "One more revision pending -- I'll comment specific changes"
3. "Pause Phase III -- structural concern"
4. "Revise interview rounds first"

Only proceed to Task 9 on option 1.

- [ ] **Step 4: No commit (outline already drafted in Task 7's commit
candidate; this task only resolves changes which roll into that
commit).**

---

## Task 9: Dispatch guide writers Batch 1

**Files:**
- Create: `docs/superpowers/specs/phase-3-guides-output/<guide>.json` (3-4 files)

Writers are batched 3-4 at a time per spec. Batch 1 covers the first
3-4 guides from the outline.

- [ ] **Step 1: Determine Batch 1 guides**

`[UPDATE AFTER PHASE II / Task 8]`: the order depends on outline
priority. Default ordering: most-foundational first.
- Guide 1: Choosing a Contact Type (decision tree)
- Guide 2: Energy-Independent Contacts
- Guide 3: Bethe Lattice Contacts
- Guide 4: 1D Chain Contacts (migration of existing markdown)

If the outline approved a different priority order, adjust.

- [ ] **Step 2: Read inputs into context for diagnosis**

Use Read tool on outline, research summary, refreshed truth table.
Required so a failed writer can be diagnosed.

- [ ] **Step 3: Dispatch 3-4 writer agents in a single message**

Use the Agent tool 3-4 times in ONE assistant message (parallel).
All `subagent_type: "general-purpose"`, `model: "haiku"`,
`run_in_background: false`.

Per-dispatch parameter table:

| Dispatch | Guide | Output Path |
|----------|-------|-------------|
| W1 | Choosing a Contact Type | `phase-3-guides-output/contact_choice.json` |
| W2 | Energy-Independent Contacts | `phase-3-guides-output/contacts_constant_sigma.json` |
| W3 | Bethe Lattice Contacts | `phase-3-guides-output/contacts_bethe.json` |
| W4 | 1D Chain Contacts | `phase-3-guides-output/contacts_1d.json` |

Shared writer prompt template:

```
You are writing a Sphinx-rst guide for the gauNEGF documentation,
for Phase III Step 4 of the documentation overhaul.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. Outline: docs/superpowers/specs/phase-3-outline.md (READ -- find
   the section for [GUIDE TITLE])
2. Research summary: docs/superpowers/specs/phase-3-research-summary.md
   (for canonical-test-pattern lifts)
3. Test inventory: docs/superpowers/specs/phase-3-test-inventory.md
   (for code lifts)
4. Source inventory: docs/superpowers/specs/phase-3-source-inventory.md
   (for "when to use" guidance)
5. Refreshed truth table: docs/superpowers/specs/api-truth.md
6. Interview notes: docs/superpowers/specs/phase-3-interview-notes.md
   (for user direction)
7. Phase II decisions: docs/superpowers/specs/phase-2-decisions.md
   (for cross-ref targets in API index)
8. Existing markdown (for migration guides only):
   docs/1D_contact_setup_guide.md (treat as DRAFT to verify, not
   ground truth)

[INSERT VERIFIER GUARDRAILS BLOCK HERE -- verbatim from SP-5]

[UPDATE AFTER PHASE II: insert any additional guardrails surfaced
by Phase II.]

You are writing: [GUIDE TITLE]

Method:

1. Read the outline section for your guide. Identify required
   sub-sections, code examples, cross-refs.
2. Read the research summary for context on what users actually do.
3. For each sub-section, write rst content matching the outline's
   shape. Keep prose grounded in real usage; do not fabricate
   scenarios.
4. For each code example: PREFER to lift verbatim or near-verbatim
   from a real test file (R1 inventory has the canonical files).
   Cite the test file with a `.. code-block:: python` directive
   followed by an inline comment `# adapted from tests/<file>:<lines>`
   on the first line of the lifted block. Verify every API call in
   the lifted code against the truth table; correct typos but do
   not "improve" working code.
5. For each example you are NOT confident about (no test analog
   exists, OR the test analog uses a deprecated pattern, OR the
   example is synthesized to fill a teaching gap): mark with
   `.. note:: This section needs author review.` at the top of the
   subsection containing the example, AND set `needs_review: true`
   on the JSON entry for that section.
6. Cross-refs use sphinx syntax: `:func:`gauNEGF.scf.NEGF.setSigma``,
   `:class:`gauNEGF.scfE.NEGFE``, `:doc:`contacts_bethe`` (relative
   doc path within docs/source/guides/).
7. For decision-tree-style guides: use rst tables or
   `.. list-table::` directive with the decision branches. Don't
   use prose-only for decision content (hard to scan).

Output: write JSON to [OUTPUT_PATH] using Write tool.

JSON schema:

{
  "guide_title": "[GUIDE TITLE]",
  "target_path": "docs/source/guides/<filename>.rst",
  "content": "VERBATIM FULL RST CONTENT for the guide, including
              the title, all sub-sections, all code blocks, all
              cross-refs. Indented properly for rst.",
  "needs_review_sections": [
    {"section": "<section title>", "reason": "<one sentence>"}
  ],
  "code_block_count": <int>,
  "lifted_blocks": <int -- code blocks lifted from tests/>,
  "synthesized_blocks": <int -- code blocks not directly lifted>,
  "cross_refs": [
    {"type": "func"|"class"|"doc", "target": "...", "context": "..."}
  ]
}

DO NOT modify any source file. Use Write tool ONLY for the output
JSON. Use Read tool for inputs.

ASCII only.

When done, return a one-paragraph summary: guide title, total
length (lines of rst), code blocks (lifted vs synthesized), sections
flagged for review, cross-refs used.
```

- [ ] **Step 4: Wait for all writers to return**

Block until all done.

- [ ] **Step 5: Validate each output JSON parses**

Run:
```bash
for f in /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-guides-output/*.json; do
  python3 -c "import json; d=json.load(open('$f')); print(f\"{d['guide_title']}: {d.get('code_block_count',0)} blocks ({d.get('lifted_blocks',0)} lifted), {len(d.get('needs_review_sections',[]))} review-flagged\")"
done
```
Expected: one line per Batch 1 guide.

- [ ] **Step 6: No commit (intermediate; rolls into Task 11 commit)**

---

## Task 10: Dispatch guide writers Batch 2

**Files:** same shape as Task 9; remaining 3-4 guides.

`[UPDATE AFTER TASK 8]`: depends on number of guides approved.

Default Batch 2 guides:

| Dispatch | Guide | Output Path |
|----------|-------|-------------|
| W5 | 3D Contacts | `phase-3-guides-output/contacts_3d.json` |
| W6 | Spin and SOC Calculations | `phase-3-guides-output/spin_soc.json` |
| W7 | Configuration & Tuning | `phase-3-guides-output/config_tuning.json` |
| W8 | Workflow Recipes | `phase-3-guides-output/workflow_recipes.json` |

Plus, conditionally per Task 5 Batch F outcome:
- W9: IntegralDemo as .rst (only if interview chose "rewrite as rst"
  or "replace")

- [ ] **Steps mirror Task 9 Steps 1-6 with the new dispatch list.**

---

## Task 11: Apply guide writer outputs to .rst files

**Files:**
- Create: `docs/source/guides/*.rst` (one per guide)
- Create: `docs/superpowers/specs/phase-3-todos.md` (if any
  needs_review sections)

- [ ] **Step 1: Read all writer JSONs from both batches**

Run:
```bash
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-guides-output/
```

Read each JSON file with the Read tool.

- [ ] **Step 2: Apply each writer's draft via SP-7**

For each guide JSON:
- Use Write tool to create the file at `target_path`.
- The content is the verbatim `content` field; no further processing.

- [ ] **Step 3: Aggregate needs_review_sections into phase-3-todos.md**

Use Write tool. Format:

```markdown
# Phase III Sections Flagged for Author Review

Date: <today>

(One section per flagged subsection.)

## <guide_title> -- <section title>

**Reason flagged:** <writer's reason>
**Suggested action:** <writer's note OR "user reviews and revises
in-place">
```

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append per-guide entries:
```
## Commit candidate: Guide -- [GUIDE TITLE]

Files: docs/source/guides/<filename>.rst
       docs/superpowers/specs/phase-3-todos.md (combined for all)

Suggested message:
docs(guides): add [GUIDE TITLE] guide

Phase III Step 4 writer output. Code examples lifted from <list of
test files>; <K> sections flagged for author review (see
phase-3-todos.md). Cross-refs target Phase II API index sections.

(Repeat per guide. User can squash if preferred.)
```

- [ ] **Step 5: Hand off**

Tell user: "<N> guides written to docs/source/guides/. <K> sections
flagged for review. Migration of 1D markdown next."

---

## Task 12: Migrate 1D markdown guide to .rst

**Files:**
- Create: `docs/source/guides/contacts_1d.rst` (if not already created
  by writer in Task 9 Batch 1 -- depending on how Guide #4 was
  scoped, this task may be a no-op)
- Delete: `docs/1D_contact_setup_guide.md`

`[UPDATE AFTER PHASE II / TASK 9]`: if Guide #4 (1D Chain Contacts)
writer in Task 9 already produced the migrated version, this task
just deletes the old .md file. Otherwise, this task does the
migration.

- [ ] **Step 1: Check writer output for the migrated file**

Run:
```bash
ls /gscratch/anantram/willll/NEGFCode/docs/source/guides/contacts_1d.rst 2>/dev/null
```
If present (already created by Task 11): proceed to Step 3.
If absent: proceed to Step 2.

- [ ] **Step 2 (conditional): Migrate manually**

Read `docs/1D_contact_setup_guide.md`. For each section:
- Convert markdown headers (`#`, `##`) to rst headers (`===`, `---`).
- Convert code fences (` ```python ... ``` `) to
  `.. code-block:: python` directives with proper indentation.
- Convert markdown tables to rst tables (use list-table directive
  for complex ones; simple grids for short ones).
- Convert inline code (`` `code` ``) to `` ``code`` ``.

Use Write tool to create `docs/source/guides/contacts_1d.rst`.

Verify every code block against the truth table per Verifier
Guardrails.

- [ ] **Step 3: Delete the original markdown**

Run:
```bash
rm /gscratch/anantram/willll/NEGFCode/docs/1D_contact_setup_guide.md
```

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: 1D guide migration

Files: docs/source/guides/contacts_1d.rst (created or modified),
       docs/1D_contact_setup_guide.md (deleted)

Suggested message:
docs: migrate 1D contact setup guide from markdown to Sphinx rst

Per Phase III Step 5. Original markdown was Claude-generated and
not fully verified ground truth; migration verified all code blocks
against the refreshed truth table and adapted prose to fit the
broader Phase III contacts guide structure.

Original .md deleted; rst version is now the single source.
```

---

## Task 13: Sphinx integration

**Files:**
- Create: `docs/source/guides/index.rst`
- Modify: `docs/source/index.rst`
- Modify: `docs/source/api/index.rst`

- [ ] **Step 1: Create guides toctree**

Use Write tool:

```rst
Guides
======

In-depth, task-oriented guides for using gauNEGF in different scenarios.

.. toctree::
   :maxdepth: 2

   contact_choice
   contacts_constant_sigma
   contacts_bethe
   contacts_1d
   contacts_3d
   spin_soc
   config_tuning
   workflow_recipes
```

`[UPDATE AFTER TASK 11]`: order entries to match outline approval; add
`integration_demo` only if interview chose IntegralDemo rewrite path.

- [ ] **Step 2: Update docs/source/index.rst toctree**

Read existing top-level toctree. Insert `guides/index` line between
the Theory section and the API Reference section. Use Edit tool.

- [ ] **Step 3: Add cross-references from API automodule entries to guides**

For each automodule entry that has a relevant guide, add a brief
"See also" rst directive linking to the guide.

`[UPDATE AFTER PHASE II / TASK 11]`: specific cross-refs to add depend
on which guides exist. Default mapping:
- `gauNEGF.scfE.NEGFE.setContactBethe` -> `:doc:`/guides/contacts_bethe``
- `gauNEGF.scfE.NEGFE.setContact1D` -> `:doc:`/guides/contacts_1d``
- `gauNEGF.surfG3D` -> `:doc:`/guides/contacts_3d``
- `gauNEGF.spinTools` -> `:doc:`/guides/spin_soc``
- `gauNEGF.config` -> `:doc:`/guides/config_tuning``

These cross-refs go in `docs/source/api/index.rst`, NOT in the
docstrings (docstring rewrites are Phase II territory).

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: Sphinx integration (toctree + cross-refs)

Files: docs/source/guides/index.rst,
       docs/source/index.rst,
       docs/source/api/index.rst

Suggested message:
docs(sphinx): wire Phase III guides into toctree + add API cross-refs

guides/index.rst lists all <N> Phase III guides in a maxdepth-2
toctree. Top-level index.rst now includes guides/index between
Theory and API Reference. api/index.rst gains See-also directives
linking each automodule to its companion guide.
```

- [ ] **Step 5: Hand off**

Tell user: "Sphinx integration done. Verifier pass on guide code
blocks next."

---

## Task 14: Verifier pass on guide code

**Files:**
- Create: `docs/superpowers/specs/phase-3-findings/guides-postwrite.json`

- [ ] **Step 1: Dispatch the verifier**

Use the Agent tool with `subagent_type: "general-purpose"`,
`model: "haiku"`, `run_in_background: false`.

Prompt is structurally identical to Phase II Task 7 verifier, but
scoped to `docs/source/guides/*.rst` AND
`docs/source/examples/IntegralDemo.rst` (if it exists from Task 9-10).

Include the Verifier Guardrails Block (SP-5) and SP-6 (degrading-fix
detection).

Output JSON path: `phase-3-findings/guides-postwrite.json`.

- [ ] **Step 2: Validate output**

Run:
```bash
python3 -c "
import json
d = json.load(open('/gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-findings/guides-postwrite.json'))
s = d['summary']
print(f'Guides postwrite: {s[\"total_findings\"]} findings ({s[\"mechanical_count\"]} mech, {s[\"judgment_count\"]} judg, {s[\"code_bug_count\"]} bug, {s[\"degrading_fix_count\"]} degrading)')
"
```

- [ ] **Step 3: No commit (intermediate)**

---

## Task 15: Apply verifier corrections (with judgment preview)

**Files:**
- Create: `docs/superpowers/specs/phase-3-corrections.md`
- Modify: `docs/source/guides/*.rst` (where corrections needed)

Identical shape to Phase II Task 8. Real mechanical fixes apply via
SP-1 + SP-6 gate; judgment items preview via AskUserQuestion.

- [ ] **Step 1: Read postwrite findings**
- [ ] **Step 2: Categorize per SP-6 + spec rules**
- [ ] **Step 3: Write phase-3-corrections.md preview**
- [ ] **Step 4: Apply real mechanical fixes**
- [ ] **Step 5: Surface judgment fixes via AskUserQuestion**
- [ ] **Step 6: Update phase-3-corrections.md with dispositions**
- [ ] **Step 7: Append commit-plan entry (use SP-3)**

(Each step mirrors Phase II Task 8 with phase-3 paths.)

---

## Task 16: Sphinx build dry-run + fixes

**Files:**
- Create: `docs/superpowers/specs/phase-3-sphinx-warnings.md`
- (Possibly) Modify: `docs/source/**/*.rst` per warning fixes

The user runs sphinx-build on a compute node since the login node
lacks the env. Per spec line ~478-487: `sphinx-build -W` treats
warnings as errors.

- [ ] **Step 1: Prepare the exact build command for user**

Tell user:

"Phase III dry-run requires `sphinx-build -W -b html docs/source
/tmp/sphinx-build-test` on a compute node.

`-W` treats warnings as errors -- catches:
- Broken cross-references (`:doc:` to non-existent files,
  `:func:` to symbols not in the API)
- Malformed rst (table syntax, unclosed directives, indentation)
- Missing toctree entries
- Duplicate labels

Expected outputs (success): build completes; tail of output says
`build succeeded.`

Expected outputs (failure): warning lines per problem; non-zero exit.

Please run the command on a compute node (gXXXX/nXXXX/zXXXX) and
post back either:
- 'Build succeeded.' (clean) -- proceed to Step 4.
- The full warning/error tail (1-50 lines typically) -- proceed to
  Step 2."

- [ ] **Step 2: User-supplied warnings -> phase-3-sphinx-warnings.md**

When user pastes the warnings, write them verbatim into
`phase-3-sphinx-warnings.md` for traceability.

- [ ] **Step 3: Apply warning fixes**

For each warning:
- Identify which guide / rst file is responsible.
- Apply fix via Edit tool.
- Common patterns:
  - Cross-ref to non-existent symbol: typo (Edit) or symbol exists
    in different module (Edit with corrected fully-qualified name)
  - Indentation error in code-block: re-indent (Edit)
  - Missing toctree entry: add to guides/index.rst (Edit)
  - Duplicate label: rename (Edit, tracking the rename across rst
    files that reference the label)

- [ ] **Step 4: Ask user to re-run sphinx-build**

If fixes were applied: "Please re-run the same sphinx-build command.
Iterating until clean."

Loop Step 3 + Step 4 until user reports "Build succeeded."

If after 3 iterations the build is still not clean: STOP. Surface
remaining warnings to user with a "I cannot diagnose these without
a working sphinx environment" note. User decides whether to defer
fixes or escalate.

- [ ] **Step 5: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: Sphinx build dry-run fixes

Files: <rst files modified per warning fixes>,
       docs/superpowers/specs/phase-3-sphinx-warnings.md

Suggested message:
docs(sphinx): fix warnings surfaced by sphinx-build -W dry-run

Phase III Task 16 ran sphinx-build with -W (warnings as errors)
on a compute node. <N> warnings fixed across <M> files; full
warning tail recorded in phase-3-sphinx-warnings.md for
traceability.

Build now succeeds clean.
```

---

## Task 17: Final commit-plan + Phase III exit

**Files:**
- Update: `docs/superpowers/specs/phase-3-commit-plan.md`

- [ ] **Step 1: Verify all Phase III artifacts exist**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-*.md
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-guides-output/
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-3-findings/
ls /gscratch/anantram/willll/NEGFCode/docs/source/guides/
```
Expected: all primary artifacts present; guides directory has the rst
files.

- [ ] **Step 2: Verify recommended commit ordering**

Read `phase-3-commit-plan.md`. Expected order:

1. Phase III research outputs (one commit)
2. Phase III interview notes (one commit)
3. Phase III outline (one commit, possibly bundled with #2)
4. Per-guide commits (~6-8 commits, one per guide for individual review)
5. 1D guide migration (one commit, possibly bundled with the
   contacts_1d guide commit)
6. Sphinx integration toctree + cross-refs (one commit)
7. Verifier corrections (one commit)
8. Sphinx-build dry-run fixes (one commit, may overlap with #7)

`[UPDATE AFTER TASK 11]`: number of commits depends on how many guides
landed; user can squash to fewer.

- [ ] **Step 3: Print final summary for user**

Output to chat:

```
=== Phase III Complete (pending your commits) ===

Research:           docs/superpowers/specs/phase-3-research-summary.md
Interview notes:    docs/superpowers/specs/phase-3-interview-notes.md
Outline:            docs/superpowers/specs/phase-3-outline.md
Guides:             docs/source/guides/*.rst (<N> files)
Sphinx integration: guides/index.rst, top-level index.rst toctree,
                    api/index.rst cross-refs
1D migration:       docs/source/guides/contacts_1d.rst (created),
                    docs/1D_contact_setup_guide.md (deleted)
Author-review TODO: docs/superpowers/specs/phase-3-todos.md (<K> sections)
Verifier corrections: docs/superpowers/specs/phase-3-corrections.md
Sphinx warnings:    docs/superpowers/specs/phase-3-sphinx-warnings.md
                    (build now clean per dry-run)
Commit plan:        docs/superpowers/specs/phase-3-commit-plan.md

What you do next:
1. Review each guide individually before committing.
2. Run commits in suggested order (or squash to your preference).
3. Address author-review TODOs at your leisure post-commit.
4. Done -- the doc overhaul is complete!

Reminder: I have not run any git command and will not.
```

- [ ] **Step 4: Phase III exit**

When user confirms commits landed, the doc overhaul is complete.

If desired, delete `docs/superpowers/specs/api-truth.phase1-snapshot.md`
(no longer needed for traceability since refreshed table is now
authoritative).

---

## Self-Review (executor: confirm before starting)

Before launching Task 1, verify:

- [ ] Spec at `docs/superpowers/specs/2026-05-09-doc-overhaul-design.md`
  contains the Phase I Retrospective.
- [x] All `[UPDATE AFTER PHASE II]` markers in this plan have been
  walked through and resolved (Phase II output integrated, draft
  status removed from header).
- [ ] Phase II commits are landed (or user explicitly waived per
  Task 1 Step 1).
- [ ] You understand: Claude does NOT commit; user does.
- [ ] You understand: subagents are haiku.
- [ ] You understand: every subagent prompt that does verification or
  writing in Phase III includes the (possibly Phase-II-shortened)
  Verifier Guardrails Block (SP-5).
- [ ] You understand: SP-6 gates every mechanical-fix application.
- [ ] You understand: the user runs sphinx-build, not Claude.
- [ ] You have access to AskUserQuestion (used heavily in Tasks 5-8,
  also in Tasks 10, 11, 15).

If any of these is uncertain, stop and ask the user before
dispatching the research agents.

---

## Plan-level self-review (drafter: completed during draft writing)

- **Spec coverage:** Phase III spec Steps 1-8 all map to plan tasks.
  Step 1 -> Tasks 2-3 (research). Step 2 -> Tasks 4-6 (interview).
  Step 3 -> Tasks 7-8 (outline + approval). Step 4 -> Tasks 9-11
  (writers + apply). Step 5 -> Task 12 (1D migration). Step 6 ->
  Task 13 (sphinx integration). Step 7 -> Tasks 14-15 (verifier +
  apply). Step 8 -> Tasks 16-17 (sphinx-build + exit).

- **Phase II carryover:** Task 1 Step 5 reads phase-2 carryover.
  Task 7 Step 2 cross-refs Phase II decisions. Task 13 Step 3
  cross-refs Phase II API index sections. SP-5 + SP-6 carry from
  Phase II. `[UPDATE AFTER PHASE II]` markers at every point where
  Phase II output specifically drives Phase III.

- **Placeholders:** none in the live tasks. The `[UPDATE AFTER
  PHASE II]` markers ARE explicit placeholders, which is intentional
  for a draft.

- **Type consistency:** finding fields (`current_snippet`,
  `proposed_fix`, `confidence`, `code_bug`, `degrading_fix`) carry
  through. Guide writer JSON schema (`guide_title`, `target_path`,
  `content`, `needs_review_sections`) consistent across Tasks 9-11.

- **Commit discipline:** every task with commit-worthy output ends
  with append-to-commit-plan via SP-3. No git mutation anywhere.

- **Bite-sized check:** longest single step is Task 11 Step 2
  (apply each writer's draft via SP-7). Acceptable: tight loop over
  N guides.

- **DRAFT markers:** All 13 `[UPDATE AFTER PHASE II]` markers resolved
  during Phase II close-out walk-through (2026-05-11). Plan is
  non-draft and ready for execution.
