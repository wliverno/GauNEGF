# GauNEGF Documentation Overhaul - Design Spec

Date: 2026-05-09
Status: Draft (awaiting user review)
Authors: William Livernois (direction), Claude (drafting)

## Context

GauNEGF's published documentation has accumulated drift relative to the source.
The README's quickstart calls a legacy positional API (`transport.current(...)`)
that has been superseded by `calculate_current(F, S, SigmaCalculator(...), ...)`.
The Sphinx best-practices guide demonstrates `setContact1D` with two positional
arguments when its real signature requires up to eleven, and shows
`setContactBethe` with a nested-list argument shape that does not match the
actual function. The Sphinx API index documents 8 modules; the package ships 14,
including 1,576 lines of `surfG3D` with no public entry. Module docstrings are
absent from `surfG1D.py` and `surfG3D.py`. Pre-rendered HTML is committed to
`docs/`, which guarantees drift any time sources change.

This spec defines a three-phase overhaul to bring the published docs back in
line with the source, fill the documentation gaps, and produce comprehensive
guides that go beyond what currently exists, especially for the different
contact types and configuration settings the package supports.

The spec intentionally favors token-efficient orchestration via parallel haiku
subagents anchored on a single API truth table extracted up front, rather than
ad-hoc per-file investigation that would re-derive the same source knowledge
repeatedly.

---

## Goals

1. Every code block, signature reference, and import path in published docs
   matches the actual `gauNEGF` source. Static + AST verified, no runtime
   execution required for the verification itself.
2. Every public module/class/function in `gauNEGF/*.py` either has accurate
   docstring coverage or is explicitly marked deprecated/internal.
3. The published docs site (built from Sphinx sources via GitHub Pages) is a
   single navigable home for: install, quickstart, theory, guides (including
   all contact types), API reference, examples.
4. Phase III delivers a comprehensive contacts-and-settings guide that
   improves on what exists today, sourced from the test suites and an
   interview with the user.

## Non-goals

1. **No code changes to `gauNEGF/*.py` to chase doc consistency.** If a doc
   snippet references a function that does not exist, the doc is wrong, not
   the code. If we find an actual code bug incidentally, it gets logged for
   the user to triage, not silently fixed.
2. **No physics correctness review.** The package implements cutting-edge
   transport physics. Verifying that the equations themselves are right is
   not in scope. We do verify internal consistency where cheap (dimensions
   referenced in docstrings vs. function signature shapes; symbol names in
   formulas vs. parameter names).
2a. **No docstring style harmonization across existing docstrings.** Phase
   II detects per-module style and writers match the dominant style of
   that module (default numpy-style only when there is no dominant style
   to match). Existing docstrings are not rewritten just to normalize
   style across modules.
3. **No restructuring of the Sphinx theme, conf.py, or build pipeline**
   beyond what is needed (e.g., enabling a guides toctree, registering
   missing modules in the API index).
4. **No execution of doc snippets** as part of the doc work. Verification is
   static + AST. The user runs the rebuilt `sphinx-build` separately on a
   compute node when each phase lands.

## Ground rules (non-negotiable)

1. **Claude does not run `git commit` or `git add` at any point.** Every phase
   produces clean file-level diffs and a draft commit message per file group.
   The user reviews diffs and runs all commits.
2. **Subagents default to haiku** for token efficiency. Escalate only if a
   haiku agent reports being stuck on something it cannot reason about.
3. **The truth table is the single source of API ground truth.** Built once
   at the start of Phase I from `gauNEGF/*.py` AST inspection. Stored at
   `docs/superpowers/specs/api-truth.md`. Refreshed at the end of Phase II
   for use in Phase III.
4. **Verification bar = static + AST.** Each verifier agent uses Python's
   `ast` module to parse code blocks, resolves imports against the truth
   table, checks argument names, order, and keyword usage. No execution.
5. **Findings file per phase.** Each phase writes
   `docs/superpowers/specs/phase-N-findings.md` (and a separate
   `phase-N-bugs.md` if incidental code bugs surface) summarizing what was
   changed, what was deferred with reason, and what was logged for human
   triage.
6. **Review checkpoints.** End of each phase = stop, user reviews, user
   commits, then we proceed. No phase begins until the prior phase is
   committed and the user has signed off.

---

## Scope

### Files in scope (the audit set)

- `README.md`
- `docs/source/**/*.rst` (all Sphinx sources)
- `docs/source/conf.py` (only if a Phase III guides toctree needs registering
  or HTML cleanup needs config touch)
- `docs/1D_contact_setup_guide.md` (migrated to .rst in Phase III)
- `gauNEGF/*.py` (docstrings only; module, class, function)
- `examples/SiNEGF.py` (code/comments only)
- `examples/IntegralDemo.ipynb` (code cells and markdown cells)

### Files explicitly out of scope for content edits

- `tests/**` and `../NEGFTests/**` -- read for Phase III research, never
  edited as docs.
- `gauNEGF/*.py` non-docstring code -- bug findings get logged in
  `phase-N-bugs.md` for user triage, not edited.

### Files to be deleted

- All committed pre-rendered HTML under `docs/`: `docs/*.html`,
  `docs/genindex.html`, `docs/objects.inv`, `docs/searchindex.js`,
  `docs/.nojekyll`, `docs/theory/*.html`, `docs/examples/*.html`,
  `docs/api/*.html`. GitHub Pages builds from Sphinx source, so these are
  dead weight.
- Keep: `docs/source/**`, `docs/source/examples/IntegralDemo_files/*.png`
  (referenced from rst), and `docs/1D_contact_setup_guide.md` until Phase
  III migrates and deletes it.
- Done as a single deletion commit at the end of Phase II, with `.gitignore`
  updated to prevent re-commit.

### Caveat on existing markdown content

The existing `docs/1D_contact_setup_guide.md` was Claude-generated in a prior
session. It has been "mostly checked" by the user but is not ground truth.
Phase III treats it as a draft to verify against the truth table during
migration, not as authoritative source material. The Phase III interview will
ask the user to flag known soft spots so writers know where to push back
hardest.

The same caveat applies to any other Claude-generated markdown found in
`docs/` during Phase I exploration.

---

## Phase I - Correctness Audit

**Goal:** every code reference in the scope-set docs matches `gauNEGF` source.
Findings produced as structured fixes the user can review before they land.

### Step 1: Build the API truth table (1 haiku agent, sequential)

- Input: `gauNEGF/*.py`.
- Method: AST inspection of every module. Extract every public class/function
  with full signature (positional args, kwargs with defaults, `*args` /
  `**kwargs`), note module path, note docstring presence (yes/no/stub), flag
  deprecation markers in module headers.
- Output: `docs/superpowers/specs/api-truth.md` -- one section per module,
  one entry per public symbol, plus a flat-table appendix for fast lookup.
- Committed by user when Phase I lands. Note: downstream phases read this
  file from disk; the commit is for versioning, not for unblocking
  consumers. Phase II still gates on user sign-off of Phase I as a whole.

### Step 2: Parallel verifiers, one per file group (6 haiku agents in parallel)

Each agent receives: (a) the truth table, (b) its assigned files, (c) the
same verification prompt. Output: structured JSON findings. No edits at this
step.

| Group | Files |
|-------|-------|
| A. User-entry        | `README.md`, `examples/SiNEGF.py`, `examples/IntegralDemo.ipynb` |
| B. Sphinx top + theory | `docs/source/quickstart.rst`, `installation.rst`, `theory/{introduction,negf_dft,transport,best_practices}.rst` |
| C. Sphinx examples + API index | `docs/source/examples/{ethane,silicon_nanowire,advanced_examples,index}.rst`, `docs/source/api/index.rst` |
| D1. Docstrings: core | `gauNEGF/{scf,scfE,density,transport}.py` |
| D2. Docstrings: surface Greens | `gauNEGF/{surfG1D,surfG3D,surfGBethe,surfGTester}.py` |
| D3. Docstrings: utilities | `gauNEGF/{matTools,integrate,spinTools,utils,protocols,config,fermiSearch}.py` |

Each verifier reports per finding:

- File path
- Line range
- Current snippet
- What is wrong (referencing the truth table)
- Proposed fix
- **Confidence label**: `mechanical` (signature/import/arg-order mismatches)
  or `judgment` (cases requiring choice between equivalent APIs or rewriting
  against a different abstraction)

Group C also produces a "missing modules" sub-finding listing every module in
`gauNEGF/` that is absent from `api/index.rst`. This feeds Phase II Step 2.

Group D verifiers focus on internal consistency: does the docstring's
"Parameters" list match the actual signature, do referenced symbols in
embedded formulas match the parameter names, do dimension annotations agree
with how the function is called elsewhere in the package. Physics correctness
is out of scope.

### Step 3: Consolidation + diff preparation (main session)

- Aggregate all 6 findings JSONs into
  `docs/superpowers/specs/phase-1-findings.md` (human-readable).
- Apply `mechanical`-confidence fixes via `Edit`, file by file, grouped by
  file group for clean commits.
- Surface `judgment` fixes to the user in a single batch: each one shown
  with current/proposed/why; user approves or revises before any edit lands.
- Anything that looks like an actual code bug (not a doc bug) gets written
  to `phase-1-bugs.md` with no edit. User triages.

### Step 4: Review checkpoint (user)

- Output handed over: 4 commit-ready file groups (`README + examples`,
  `sphinx top + theory`, `sphinx examples + API index`, `docstrings`), each
  with a draft commit message in `phase-1-commit-plan.md`.
- User reviews diffs, edits/discards messages, runs commits at their pace.
- `phase-1-findings.md`, `phase-1-bugs.md`, and `api-truth.md` also get
  committed by the user as a versioned record.

### Phase I exit criteria

- `api-truth.md` committed.
- All `mechanical`-confidence fixes applied; all `judgment` fixes either
  applied or explicitly deferred with reason logged.
- `phase-1-findings.md` and `phase-1-bugs.md` committed.
- User has signed off on all four file-group commits.

---

## Phase I Retrospective (added post-execution, 2026-05-09)

Phase I executed against this spec on 2026-05-09. The audit produced 47
findings, of which 17 mechanical fixes were applied directly, 9 judgment
fixes were user-approved and applied, 12 verifier-flagged "judgment"
items were rejected by the user as not actually broken, and 8
"code_bug" flags turned out to be 1 doc typo + 7 verifier
misclassifications. Lessons that change how Phase II and Phase III
should run:

### Verifier (truth-table) blind spots

The truth-table extractor (a haiku agent doing AST inspection) got the
basic signature correct in every case but missed five categories of
information that downstream verifiers then leaned on incorrectly:

1. **Class inheritance constructors.** `class NEGFE(NEGF):` was
   recorded as `Signature: NEGFE(NEGF)`, which reads to a downstream
   verifier as "constructor takes a NEGF instance". Real behavior:
   NEGFE inherits NEGF's `__init__`, so `NEGFE('molecule', basis=...)`
   works. The truth table needs to either inline the inherited
   `__init__` signature or explicitly note "inherits Y constructor".
2. **Dynamic attributes.** Attributes set by methods other than
   `__init__` (e.g., `NEGF.setSigma` sets `self.sigma1`, `self.sigma2`,
   `self.Gam1`, `self.Gam2`; `NEGFE.setContactBethe` /
   `setContact1D` / `setSigma` all set `self.g`) were absent from the
   truth table. Downstream verifiers then flagged any reference to
   these attributes as "does not exist".
3. **Variable-shape returns.** Functions with branching returns
   (e.g., `calculate_transmission` returns scalar normally but
   2-tuple when `spin` is set; `calculate_dos` always returns 2+
   tuple) were extracted with only the basic signature, no return
   shape annotation. Verifiers then flagged correct tuple-unpack
   usages as bugs.
4. **Method overloads via duck typing.** Classes whose first-arg
   type changes behavior were not noted. `SigmaCalculator` accepts
   either `(sig1, sig2)` matrices or `(surfG_object,)`; verifiers
   only saw the matrix form.
5. **Nested-list parameter conventions.** Where a parameter
   documented as `list` actually expects `list-of-lists` (one inner
   list per contact), the convention was not surfaced. E.g.,
   `setContactBethe(contactList=[[1,2,3], [4,5,6]])` is correct
   for two contacts; verifier proposed flattening to a single list.

**Fix:** Phase II Step 7 (truth table refresh) uses an upgraded
extraction prompt that explicitly checks for and documents each of
these five categories. See `docs/superpowers/specs/phase-1-bugs.md`
for the canonical "truth-table generator upgrades" punch list.

### Verifier confidence calibration

The Phase I verifiers were over-aggressive. Of 17 verifier-flagged
"mechanical" items, 6 were no-ops (cosmetic-only proposed changes
like adding `spin=None` when None was already the default, or
reordering kwargs that work in either order). Of 16 "judgment"
items, 11 were rejected by the user as not bugs. Of 8 "code_bug"
flags, 7 were verifier mis-reads of the truth table. Net: only
~30% of findings represented real fixes.

**Fix for Phase II/III:**
- Verifier prompts include an explicit "if your proposed fix would
  degrade the docs (e.g., comment out a working example, add
  decorative kwargs, paraphrase prose without changing meaning),
  flag as judgment instead of mechanical".
- Verifier prompts include the 5 known truth-table blind spots above
  so verifiers can self-suppress those false-positive shapes.
- The `code_bug` classification gets a tighter rule: only set true
  when (a) the doc reference matches a documented public API and
  (b) the source's actual behavior differs from what the docs
  describe. Truth table absence alone is no longer sufficient.

### Main-session (Claude) assumption errors

Beyond the verifier issues, the main session itself made a real
mistake during Phase I: applied an "NEGFE wrap pattern" fix in
`silicon_nanowire.rst` based on the same wrong reading of the truth
table that the verifier had. The mistake was caught only after the
user pushed back during Task 9 (judgment review), and the change
was reverted. About 12 additional fixes were queued that would have
similarly damaged the docs; user input prevented them from landing.

**Fix for Phase II/III:**
- Surface judgment items to the user EARLY, not at the end of a long
  application phase. Phase II/III plans should include a
  "preview judgment items" step before any application step that
  derives from possibly-shaky verifier output.
- When the verifier and a doc example disagree about API shape and
  the doc looks coherent, default to "trust the doc, suspect the
  verifier", not the other way around. Surface to user with a
  question like "verifier says X is broken; doc says it works
  this way for these reasons; which is right?".
- Apply the "if a fix degrades docs, push back" rule from the
  verifier guardrail to the main session itself.

### Phase II Step 2 default change

Per Phase I exploration, `gauNEGF.utils` is used user-facing in
`examples/SiNEGF.py` (`from gauNEGF.utils import fractional_matrix_power`)
and in three test files (`tests/test_calcTSW.py`,
`tests/test_transport_dos_crossterm.py`, `tests/test_surfG1D_features.py`
all import `inv` / `eigh` / `fractional_matrix_power`). Per the
spec's own pre-check rule ("greps examples/ and tests for any
user-facing call to utils symbols before locking; if any are
found, include in API"), the `utils` default flips from EXCLUDE to
INCLUDE. The Phase II plan reflects this.

### Process change: artifact ordering

Phase I produced its `phase-1-judgment-fixes.md` artifact at the
END of the application phase. In retrospect, partial drafts of
that artifact would have been more useful at the start of judgment
review — the user could see all proposed judgment fixes in one
file before answering questions. Phase II Step 8 should produce
the equivalent file as a draft DURING the writer-output review,
not after.

---

## Phase II - Gap Fill

**Goal:** every public symbol either has accurate docstring coverage or is
explicitly excluded; every non-deprecated module is registered in the Sphinx
API index; dead pre-rendered HTML is removed.

### Step 1: Gap inventory (1 haiku agent, sequential)

- Input: the truth table from Phase I.
- Output: `docs/superpowers/specs/phase-2-gaps.md`. Per module, a list of:
  - Module-level docstring: present / absent.
  - Per-symbol docstring status: present / stub / absent.
  - API index status: registered / missing.
  - Detected docstring style (numpy / google / sphinx-rst). Writers in Step
    3 follow whatever style dominates that module; if mixed, default to
    numpy-style.

### Step 2: API index decisions (user, single batch)

Defaults locked in this spec; user re-confirms during execution with module
summaries in hand:

| Module | Decision | Rationale |
|--------|----------|-----------|
| `surfG3D`     | Include in main API | 1576 lines of production code, public contact type. |
| `spinTools`   | Include in main API | User-facing for non-collinear / SOC users. |
| `config`      | Include as "Configuration Reference" page | Already user-facing via README table; Sphinx page makes it discoverable + cross-referenceable. |
| `protocols`   | Include in separate "Developer / Extensibility Reference" section | Contributor-facing interface contract; documenting it enables adding new contact types. |
| `utils`       | Include in main API (REVISED post-Phase-I) | Originally defaulted to Exclude. Phase I exploration found `utils.fractional_matrix_power`, `utils.inv`, `utils.eigh` used user-facing in `examples/SiNEGF.py` and three test files. Per the spec's own pre-check rule, this flipped to Include. Phase II Step 2 will still confirm with the user. |
| `fermiSearch` | Exclude | Deprecated per its own docstring header. Module docstring gets a clear "DEPRECATED -- use X" pointer. |

The two flagged for re-confirmation at execution time:

- `protocols`'s placement (separate Developer section vs. flat under main
  API) is stylistic.
- `utils` exclusion is the "implementation detail" call. Phase II Step 2
  greps `examples/` and tests for any user-facing call to `utils` symbols
  before locking; if any are found, include in API.

### Step 3: Parallel docstring writers (3 haiku agents in parallel)

Splits match Phase I D1/D2/D3 (core / surface Greens / utilities). Each
writer:

- Reads truth table + gap list + assigned modules.
- For each missing or stub docstring, writes:
  - One-line summary derived from function body inspection. Not fabricated.
  - `Parameters` section (mechanical from signature, types from annotations
    or call-site inspection).
  - `Returns` section (mechanical where possible).
  - `Raises` section if explicit `raise` calls present in the body.
  - `Notes` section ONLY when there are non-obvious side effects visible in
    the body (mutates `self`, writes to disk, JAX-traced behavior, etc.).
- **Writer never fabricates usage examples in this phase.** Examples are
  Phase III territory.
- If a function is too complex to summarize accurately from inspection,
  writer leaves a stub `# TODO(human): one-line summary needed` and flags
  in findings. Honest TODO over confident hallucination.
- **TODO ceiling: 3 per module.** If a writer would leave more than 3
  TODOs in a single module, the module is flagged for human-written
  docstrings instead, and that decision surfaces to the user in Step 8
  review. Prevents silent accumulation of unfilled stubs.

### Step 4: Verifier pass on new docstrings (1 haiku agent, sequential)

- Input: all newly written docstrings + the truth table.
- Method: same AST verification as Phase I -- every cross-reference, every
  parameter name, every return type tag must match the actual signature.
- Output: list of any drift between writer output and truth (caught
  hallucinations, typo'd parameter names, etc.).
- **Correction workflow:** main session applies the verifier's corrections
  directly via `Edit`. No re-dispatch to writers. The verifier output is
  precise enough (file path + line + before/after) for mechanical fixes
  against the truth table. If a verifier finding requires judgment (e.g.,
  the writer's description of behavior is wrong, not just the parameter
  name), it goes to the user in Step 8 review.

### Step 5: API index update

- Apply Step 2 decisions to `docs/source/api/index.rst`.
- Add headings for new module groups if needed (e.g., a "Configuration"
  section, a "Developer Reference" section).

### Step 6: Pre-rendered HTML deletion (last commit of Phase II)

- Delete: all pre-rendered HTML and Sphinx build artifacts under `docs/` per
  the scope section.
- **Pre-check:** verify `.gitignore` exists at repo root and read its
  current contents. If absent, create it. Append the new patterns only if
  not already present (avoid duplicate entries).
- Add `docs/*.html`, `docs/**/*.html`, `docs/genindex.html`, `docs/objects.inv`,
  `docs/searchindex.js`, `docs/.nojekyll` to `.gitignore`.
- **Sequencing safety:** this deletion is the LAST commit of Phase II,
  after Sphinx sources are clean. Before this commit, the user confirms
  GitHub Pages is configured to build from Sphinx (Action/workflow) rather
  than serve `/docs` directly. If it is the latter, hold deletion until
  Pages config is changed.

### Step 7: Truth table refresh

After all Phase II edits, regenerate `api-truth.md` so Phase III works
against current state.

### Step 8: Review checkpoint (user)

- Commit groupings: (i) new module docstrings per writer group, (ii) new
  function docstrings per writer group, (iii) API index update, (iv) HTML
  deletion + `.gitignore`. Each with draft commit messages in
  `phase-2-commit-plan.md`.
- User commits at their pace.

### Phase II exit criteria

- API index decisions confirmed; new entries land in `api/index.rst`.
- All gap-list docstrings written and verifier-passed (or explicitly stubbed
  with TODOs the user has seen).
- Pre-rendered HTML deleted; `.gitignore` updated; GitHub Pages source-config
  precondition confirmed.
- Truth table refreshed.

### Risk

Auto-written docstrings can be subtly wrong even with verification. The
real safety net is the user's eyes on the surfG3D and density docstrings
during Phase III interview, when those modules come up in context.

---

## Phase III - Comprehensive Guides + Interview

This is the longest phase and the lowest-verifiability one. Quality control =
AST-verify all code blocks the writers produce, plus the user's eyes on every
guide before it commits. Phase III is iterative, not one-shot.

### Step 1: Research subagents (2 haiku agents in parallel)

- **Agent R1: tests/ and ../NEGFTests/ survey.** Reads every `.py` and
  `.ipynb`. Extracts: every contact-setup call (which method, full args),
  every system topology (atoms/cells/basis), every setting deviating from
  `config.py` defaults, every workflow pattern (extract matrices, SCF,
  transmission, full-SCF vs Harris guess, voltage sweeps, etc.). Output:
  `docs/superpowers/specs/phase-3-test-inventory.md` -- structured tables.
- **Agent R2: source-side contact inventory.** Reads `scf.py`, `scfE.py`,
  `surfG*.py`. Extracts: every public contact-setup entry point, every
  `surfG*` class's interface (constructor args, key methods like `sigma()`,
  `crossTermQ()`), what physical scenario each is meant for. Output:
  `docs/superpowers/specs/phase-3-source-inventory.md`.

Together these answer "what can the package actually do, and what does the
user actually use it for".

### Step 2: Interview prep + interview (main session)

Main session reads both research outputs and drafts an interview question
batch. Questions go through `AskUserQuestion` (multiple choice where
possible, free-text where not). Categories:

- **Audience priority**: researchers new to NEGF / experts new to gauNEGF /
  contributors / all three.
- **Guide topic confirmation**: from research, the best-guess topic list
  below; user confirms, adds, removes.
- **Per-contact-type questions**: when does the user reach for Bethe vs 1D
  vs 3D vs constant sigma? What pitfalls were learned the hard way? What is
  the canonical worked example for each?
- **Settings explanations**: the `config.py` table is descriptive, not
  didactic. Which settings have been tuned in real work and why?
  (e.g., `SCF_DAMPING=0.02` in the README -- what is the story?)
- **What the existing docs get wrong** -- meta-feedback. The user has noted
  the existing `1D_contact_setup_guide.md` is Claude-generated and "mostly
  checked" but not trusted as ground truth. Interview will ask for known
  soft spots.
- **IntegralDemo.ipynb status**: the user has noted this notebook is old
  and Sphinx integration of Jupyter notebooks is "spotty at best". Decide
  in interview: re-execute, rewrite, or replace with a comparable .rst
  walkthrough.

Interview is 1-2 batches with follow-ups as needed. **Cap on follow-up
rounds: 2.** If after 2 follow-up rounds the outline shape is still
unresolved, that is a signal something is structurally wrong (e.g., the
research outputs missed a major usage pattern, or the topic decomposition
itself is off). Stop, replan with the user, do not proceed to Step 3.
Output: `docs/superpowers/specs/phase-3-interview-notes.md`.

### Best-guess guide list (user confirms in interview)

1. **Choosing a Contact Type** -- decision tree + when-to-use-what.
2. **Energy-Independent Contacts (constant sigma)** -- the simplest, for
   screening / testing.
3. **Bethe Lattice Contacts** -- metallic leads via `setContactBethe` +
   `surfGBethe`.
4. **1D Chain Contacts** -- periodic chains via `setContact1D` + `surfG1D`.
   Migrated and expanded from the existing `1D_contact_setup_guide.md`.
5. **3D Contacts** -- `surfG3D` workflow. Brand new content.
6. **Spin and SOC Calculations** -- `spinTools`, `spin='u'` / `'g'`, SOC
   integration patterns.
7. **Configuration & Tuning Guide** -- `config.py` settings explained
   didactically (when to change SCF_DAMPING, ETA, FERMI_SEARCH_CYCLES,
   etc.).
8. **Workflow Recipes** -- end-to-end NEGF-DFT with checkpointing, voltage
   sweeps, current/transmission/DOS analysis.

### Step 3: Outline + user approval gate

Main session drafts `docs/superpowers/specs/phase-3-outline.md` -- title +
sub-sections + key code examples planned + cross-refs to API for each
guide. **User reviews the whole outline before any writer runs.** This is
the most important checkpoint in Phase III; if the outline is wrong, every
writer downstream produces wasted work.

### Step 4: Parallel guide writers (one haiku per guide, batched 3-4 at a time)

Each writer receives: outline section for its guide, research outputs,
interview notes, current truth table. Writes `.rst` directly. Writers
prefer code examples lifted verbatim or near-verbatim from existing tests
(known-working code) over fabricated examples. Each writer flags any code
example it is not confident about with `.. note:: needs review`.

**IntegralDemo.ipynb fate handling:** the interview decides one of three
outcomes. Each is pre-staged so scheduling is not ambiguous when the
decision arrives:

- *Rewrite as .rst*: a writer slot is reserved for converting the
  notebook content into a guide, with code lifted from the notebook's
  cells (verified against truth table) and prose adapted from the
  notebook markdown. Slot activates if interview chooses this.
- *Replace*: a writer slot is reserved for a fresh integration-methods
  guide, sourcing examples from `tests/` rather than the stale notebook.
- *Keep / re-execute later*: notebook is left as-is; user re-executes on
  a compute node post-Phase III. No writer slot needed.

### Step 5: Migrate the 1D guide

Convert `docs/1D_contact_setup_guide.md` to
`docs/source/guides/contacts_1d.rst` (or whatever the outline names it).
Mostly mechanical -- markdown tables to rst tables, code fences to
`.. code-block:: python` directives. Then expand against the new outline
(the existing guide is good but narrow). Treat the original as a draft to
verify, not source-of-truth. Delete the `.md` after migration is verified
and committed.

### Step 6: Sphinx integration

- Create `docs/source/guides/index.rst` with toctree of all new guides.
- Add `guides/index` to the main `docs/source/index.rst` toctree (likely
  right after Theory, before API Reference).
- Add cross-references from API automodule entries to relevant guides
  (e.g., `surfGBethe` automodule links to "Bethe Lattice Contacts" guide).

### Step 7: Verifier pass on guide code

Same AST verification as Phase I, applied to every code block in every new
guide. Catches writer hallucinations. 1 haiku agent.

### Step 8: Review checkpoint (user)

- **Sphinx build dry-run.** Before any per-guide commits, the user runs
  `sphinx-build -W -b html docs/source /tmp/sphinx-build-test` on a
  compute node (`-W` treats warnings as errors, catching broken
  cross-references, malformed rst, missing toctree entries). Main
  session prepares the exact command and lists the expected outputs;
  user executes and reports back. Any errors block the commit and feed
  back to the relevant writer slot for correction.
- Per-guide commits (user commits each guide individually so they can
  review/revise per topic).
- Final commits: guides toctree + index.rst toctree update + 1D markdown
  deletion.
- `phase-3-findings.md` summarizes what was written, what was verified,
  what is flagged for follow-up.

### Phase III exit criteria

- Research outputs committed.
- Interview notes committed.
- Outline approved by user.
- All guides written, verifier-passed, individually committed.
- 1D markdown migrated to rst and original deleted.
- `docs/source/index.rst` toctree updated; cross-refs in place.

### Risks

- Phase III generates the most novel content; even with verification,
  prose can be wrong. User review is the safety net.
- The interview drives outline shape. If interview reveals topics not
  anticipated, outline updates before writing, not after.
- 1D guide migration may reveal rst formatting edge cases (rst tables
  are fussy). Budget a small revision pass post-conversion.
- IntegralDemo.ipynb resolution depends on interview outcome. If decision
  is "rewrite as .rst", the rewrite happens as one of the guide writers'
  outputs.

---

## Orchestration & Sequencing

### Phase-level sequencing

**Strictly sequential.** Each phase ends with the user committing, then we
move on. Reasons:

- Phase II's writers depend on the truth table from Phase I.
- Phase III's writers depend on accurate Phase II docstrings (cross-references
  via Sphinx, example patterns lifted from updated module docstrings).
- Sequential phasing keeps review load bounded.

### Within-phase parallelism

- Phase I: 6 parallel verifiers + 1 sequential truth-table builder.
- Phase II: 3 parallel writers + 1 sequential gap inventory + 1 sequential
  verifier.
- Phase III: 2 parallel research agents, then guide writers batched 3-4 at
  a time.

### Token budget posture

All subagent work is haiku unless the model explicitly reports a blocker.
The truth-table-first design ensures no agent re-derives signature info
that another agent already computed. Estimated total: well under one
main-session-equivalent of token use across all phases.

---

## Success Criteria (meta-level)

1. A new user reading the published docs front-to-back never encounters a
   code snippet that will not import / will not run / has wrong arg names.
2. Every public symbol in `gauNEGF` has an accurate Sphinx page with at
   minimum a one-line summary and parameter list.
3. The "I want to use a [Bethe / 1D / 3D / constant-sigma] contact"
   question has a dedicated guide, sourced from real test patterns, not
   synthesized.
4. The repo no longer carries pre-rendered HTML; the published site is
   rebuilt from Sphinx source via GitHub Pages.

---

## Open questions (to revisit during execution; not blockers)

- Does Sphinx need `myst-parser` or another extension for any reason? Phase
  I exploration of conf.py will tell us. The user has noted Jupyter notebook
  integration is currently "spotty"; decide in Phase III interview whether
  to invest in fixing or replace IntegralDemo with rst.
- Will Sphinx need version bumps or new extensions for cross-references
  between guides and automodule entries? Will check during Phase II
  API-index work.
- The `examples/IntegralDemo.ipynb` may have stale output cells. Phase I
  verifies input cells; output staleness is a separate Phase III decision.

---

## Appendix A - Concrete known issues (inventory from initial exploration)

These are real issues found while drafting this spec. Phase I will find
many more, but these are the anchor cases verifying that the audit is
necessary, not theoretical.

1. **README.md quickstart** calls `transport.current(negf.F*harToEV, negf.S,
   negf.sig1, negf.sig2, negf.fermi, negf.qV)` -- legacy positional API.
   The current canonical API is `calculate_current(F, S,
   SigmaCalculator(sig1, sig2), fermi=..., qV=...)`. The legacy `current()`
   still exists at `transport.py:750` for back-compat, but using it in the
   README onboarding example sets the wrong norm.
2. **README.md quickstart** uses `negf.setContacts([1], [2])` then refers
   to `negf.sig1` / `negf.sig2` which `setContacts` does not produce
   (`setSigma` does). Verify which method the README actually means.
3. **docs/source/theory/best_practices.rst** shows
   `setContact1D([[2],[3]], [[1],[4]], eta=1e-5, T=300)`. Real signature
   takes 11 args including `tauList`, `stauList`, `alphas`, `aOverlaps`,
   `betas`, `bOverlaps`, `neList`. This call would crash.
4. **docs/source/theory/best_practices.rst** shows
   `setContactBethe([[1,2,3],[6,7,8]], latFile='Au2', ...)` -- nested
   list. Real signature is `setContactBethe(contactList, latFile='Au',
   ...)` -- single contactList. Also missing `negf.` prefix on the call.
5. **docs/source/theory/transport.rst** spin-resolved example uses `negf.F`
   (Hartree) while other examples in the same file use `negf.F*27.211386`
   (eV). Internally inconsistent.
6. **docs/source/api/index.rst** documents 8 modules. Missing from Sphinx:
   `surfG3D` (1576 lines), `spinTools` (378 lines), `protocols`, `utils`,
   `config`, `fermiSearch`. Decisions per the Phase II Step 2 table.
7. **gauNEGF/surfG1D.py** opens with bare imports, no module docstring.
8. **gauNEGF/surfG3D.py** opens with bare imports, no module docstring.
9. **docs/source/quickstart.rst** uses `np.linspace` without importing
   numpy in the same code block (imports happen earlier; rst readers
   copy-pasting individual blocks may miss this).

---

## Appendix B - What we ruled out and why

- **Per-file independent agents that re-derive signatures.** High token
  cost, risk of inconsistent verdicts between agents. Truth-table approach
  wins on both.
- **Sequential single-agent walkthrough.** Slowest, no parallelism win.
  Only useful if we wanted maximally cohesive narrative output, which is
  not the goal.
- **Markdown-via-myst integration for guides.** Mixed-format burden, weaker
  rst extension support. All guides rst keeps the format story simple.
- **Standalone markdown guides outside Sphinx.** Poor discoverability on
  the published site.
- **Maintaining pre-rendered HTML in the repo.** GitHub Pages handles it;
  committed HTML guarantees drift any time sources change.
- **Modifying gauNEGF/*.py code to make docs match.** Code is source of
  truth; doc bugs are doc bugs. Code bugs found incidentally are logged
  for user triage.
- **Physics correctness review.** Out of scope per user direction; cutting-
  edge physics, both author and reviewer would be guessing.
