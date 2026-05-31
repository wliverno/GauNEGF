# GauNEGF Doc Overhaul - Phase II Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill every public-symbol docstring gap in `gauNEGF/*.py`, register
the missing modules in the Sphinx API index per locked Phase II Step 2
decisions, delete pre-rendered HTML now that GitHub Pages builds from
Sphinx source, and refresh the API truth table with the upgraded
extraction conventions surfaced by the Phase I retrospective.

**Architecture:** Sequential gap inventory (1 haiku); user re-confirms Step 2
decisions via `AskUserQuestion` (defaults pre-loaded from spec, with the
utils flip already applied); 3 parallel docstring writer agents (haiku) per
writer group D1/D2/D3; sequential verifier pass on writer output; main
session applies verifier corrections directly; manual API index edit; user
confirms GitHub Pages config before HTML deletion; refreshed truth table
uses an upgraded extraction prompt. User runs all commits.

**Tech Stack:** haiku-class subagents via the Agent tool, AskUserQuestion
for Step 2 confirmation, Edit/Write tools for application, Bash for HTML
deletion + .gitignore update, the refreshed truth table at
`docs/superpowers/specs/api-truth.md` (overwriting the Phase I version
once Phase II edits land).

**Parent spec:** `docs/superpowers/specs/2026-05-09-doc-overhaul-design.md`
(Phase II Steps 1-8, plus the Phase I Retrospective section added
post-Phase-I).

**Hard rules** (carried from spec, repeated here so the executor cannot
miss them):

- Claude does NOT run `git commit`, `git add`, `git push`, or any git
  mutation. User stages and commits everything. Each task that produces
  commit-worthy output ends with appending a draft commit message to
  `docs/superpowers/specs/phase-2-commit-plan.md`.
- Subagents are dispatched with `model: "haiku"` per global user pref.
- Verification is static + AST. No execution of doc snippets.
- ASCII only in any file Claude writes. No unicode.
- Phase II does NOT begin until Phase I commits are landed (or user
  explicitly waives this gate). Task 1 verifies this.

**Phase I lessons baked into this plan:**

- All subagent prompts (writer + verifier) include the "Verifier
  Guardrails Block" (see Shared Procedures) listing the 5 truth-table
  blind spots so subagents do not re-flag the same false positives.
- The truth-table refresh task uses an UPGRADED extraction prompt
  documenting class inheritance constructors, dynamic attributes,
  variable-shape returns, duck-typed overloads, and nested-list
  parameter conventions.
- Judgment items surface to the user EARLY (Task 8 has a preview step
  before any application).
- The `utils` Step 2 default flipped from Exclude to Include based on
  Phase I exploration finding user-facing usage in `examples/SiNEGF.py`
  and three test files. Step 2 still asks the user to confirm.
- Main session retraction culture: when the verifier's proposed fix
  would degrade documentation (comment out a working example, add
  decorative kwargs, paraphrase prose without changing meaning), DO
  NOT apply; reroute to user judgment review.

---

## Shared procedures

### SP-1: Apply a mechanical fix via Edit tool

(Same as Phase I.) For each fix from a verifier finding:

- `Edit` tool with `file_path`, `old_string` = verbatim `current_snippet`,
  `new_string` = verbatim `proposed_fix`, `replace_all = false`.
- If `Edit` fails on non-unique `old_string`: read file at finding's
  `line_range`, expand `old_string` with surrounding context until
  unique, re-apply.
- If `Edit` fails on missing `old_string`: skip and add to "skipped"
  list; do not guess.
- For `.rst` files: preserve indentation; sphinx default for
  `code-block::` is 3-space block-quote indentation.
- For `.py` docstring edits: if `proposed_fix` would touch a function
  body or signature, reclassify as code-bug and add to
  `phase-2-bugs.md` (do NOT apply).

### SP-2: Re-verify a file group after fixes

(Same as Phase I.) Dispatch a single haiku agent (`general-purpose`,
`haiku`, `run_in_background: false`) with the prompt template below,
filling bracketed values:

```
Re-verify these files against the API truth table:
[FILE_LIST_OF_GROUP]

Truth table: docs/superpowers/specs/api-truth.md (READ THIS FIRST)

Method: same as the original Phase I/II verifier prompt:
1. Read each file in full.
2. Find every Python code reference (rst code-block, .py docstrings).
3. AST-equivalent inspection of each reference; resolve names against
   truth table; check signatures, arg names, arg order, kwargs.
4. Apply the Verifier Guardrails Block (below) -- do NOT re-flag the
   5 known truth-table blind spots.
5. Flag any mismatch with file_path, line_range, current_snippet,
   issue, proposed_fix, confidence, code_bug.

[VERIFIER GUARDRAILS BLOCK -- inserted verbatim from SP-5]

Output: write JSON to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-findings/[OUTPUT_FILENAME] with the same schema as Phase I.

If zero findings: still write the JSON with empty findings array.
ASCII only.

Return one paragraph: files audited, total findings, breakdown by
confidence.
```

### SP-3: Append commit-plan entry

(Same as Phase I.) Use Edit (or Write if file does not exist) on
`docs/superpowers/specs/phase-2-commit-plan.md`. Format:

```
## Commit candidate: <short title>

Files: <paths>

Suggested message:
<one-line subject>

<optional 2-4 line body>
```

### SP-4: Apply a writer's draft docstring to a source file (NEW for Phase II)

Used by Task 6 to merge writer-agent output into `gauNEGF/*.py` source.
Writer agents produce a JSON file containing a list of docstring inserts
or replacements per symbol. For each entry:

- If `kind == "insert"`: the symbol currently has no docstring. Use
  Edit tool to find the symbol's `def` line + colon, then add the new
  docstring (triple-quoted block) on the next line indented one level
  deeper. The Edit `old_string` is the `def ...:` line plus a small
  amount of trailing context to keep it unique; `new_string` is the
  same `def ...:` line followed by the indented docstring then the
  trailing context.

- If `kind == "replace_stub"`: the symbol has a stub or absent docstring
  recorded in the gap inventory but the writer chose to replace existing
  text. Use Edit tool with the existing-stub text as `old_string` and
  the writer's new docstring as `new_string`.

- If `kind == "todo"`: the writer left a TODO marker because the
  function was too complex to summarize from inspection. Use Edit to
  insert `# TODO(human): one-line summary needed for [symbol]` as a
  code comment immediately above the `def` line. Do NOT insert a
  partial docstring.

- TODO ceiling: at most 3 TODOs per module. Task 6 enforces this; if
  a writer's output for one module exceeds 3 TODOs, the entire module
  is flagged in `phase-2-todos.md` and the user decides per-module
  whether to (a) write the docstrings by hand, (b) accept all TODOs
  knowingly, or (c) defer the module to Phase III. SP-4 then applies
  only the under-ceiling subset for that module.

### SP-5: Verifier Guardrails Block (NEW for Phase II)

This block is inserted verbatim into EVERY subagent prompt that does
verification or writing in Phase II. Skip ONLY for the truth-table
refresh agent (which has its own guardrails as part of the upgraded
extraction prompt).

```
=== VERIFIER GUARDRAILS (DO NOT VIOLATE) ===

The Phase I truth-table extractor missed five categories of
information. Do NOT flag findings that fall into any of these,
unless the issue is genuinely separate from the blind spot:

1. CLASS INHERITANCE CONSTRUCTORS. The truth table records
   `class NEGFE(NEGF):` as `Signature: NEGFE(NEGF)`. This DOES NOT
   mean NEGFE's constructor takes a NEGF instance. NEGFE INHERITS
   NEGF's __init__, so `NEGFE('molecule', basis=...)` is valid.
   When a class signature is `X(Y)` and you suspect inheritance,
   check Y's __init__ in the truth table -- the class likely
   inherits Y's constructor.

2. DYNAMIC ATTRIBUTES. Many attributes are set by methods other
   than __init__. Examples currently absent from the truth table:
   - `NEGF.setSigma` sets `self.sigma1`, `self.sigma2`,
     `self.sigma12`, `self.Gam1`, `self.Gam2`.
   - `NEGFE.setContactBethe`, `setContact1D`, `setSigma` all set
     `self.g`.
   Do NOT flag references to these attributes as "does not exist"
   without first reading the relevant setter method to confirm.

3. VARIABLE-SHAPE RETURNS. Some functions return scalar in one
   mode and tuple in another. Confirmed examples:
   - `calculate_transmission(F, S, sc, El, spin=...)` returns
     a 2-tuple when spin is set; scalar otherwise.
   - `calculate_dos(F, S, sc, El, ...)` always returns at least
     a 2-tuple `(dos_total, dos_per_site)`, sometimes a 4-tuple
     including spin components.
   Do NOT flag tuple-unpacking as a bug without inspecting the
   function body for the return statement.

4. METHOD OVERLOADS VIA DUCK TYPING. Some classes accept different
   first-arg types and dispatch internally. Confirmed:
   - `SigmaCalculator(sig1, sig2)` accepts two matrices OR
     `SigmaCalculator(surfG_object)` accepts a surfG instance.
   Do NOT flag `SigmaCalculator(negf.g)` as a bug.

5. NESTED-LIST PARAMETER CONVENTIONS. Some parameters documented
   as `list` actually expect `list-of-lists`, one inner list per
   contact. Confirmed:
   - `setContactBethe(contactList=[[1,2,3], [4,5,6]])` is the
     correct two-contact form. Do NOT flatten.

=== END VERIFIER GUARDRAILS ===
```

### SP-6: Quality-degrading-fix detection (NEW for Phase II)

Before applying any verifier-flagged "mechanical" fix in Tasks 6 and 8,
check whether the proposed_fix would degrade the documentation. Common
degradation patterns:

- The proposed_fix comments out a code block (lines starting with `#`
  added to wrap a block).
- The proposed_fix adds a kwarg whose value is the existing default
  (e.g., `spin=None` when `spin=None` is the default).
- The proposed_fix only reorders kwargs without changing the call
  semantics.
- The proposed_fix paraphrases prose without altering its information
  content.
- The proposed_fix replaces working code with a NotImplementedError or
  placeholder.

If a proposed_fix matches any of these patterns:
- Do NOT apply via SP-1.
- Add the finding to `phase-2-skipped.md` with the pattern name.
- Do NOT auto-reroute to user judgment review unless the underlying
  issue (separate from the bad fix) is real -- often the verifier's
  premise was wrong and there is no underlying issue.

---

## File Structure (what gets created in this phase)

| Path | Created in | Purpose |
|------|------------|---------|
| `docs/superpowers/specs/phase-2-gaps.md` | Task 2 | Gap inventory: per-module docstring status, style detection, API index status. |
| `docs/superpowers/specs/phase-2-decisions.md` | Task 3 | User's API index decisions per Step 2 with rationale. |
| `docs/superpowers/specs/phase-2-writer-output/group-D1.json` | Task 5 | Writer-agent draft for D1 modules. |
| `docs/superpowers/specs/phase-2-writer-output/group-D2.json` | Task 5 | Writer-agent draft for D2 modules. |
| `docs/superpowers/specs/phase-2-writer-output/group-D3.json` | Task 5 | Writer-agent draft for D3 modules. |
| `docs/superpowers/specs/phase-2-todos.md` | Task 6 | Modules where writer exceeded 3-TODO ceiling, with per-module user decision. |
| `docs/superpowers/specs/phase-2-skipped.md` | Tasks 6, 8 | Findings skipped per SP-6 (degrading fixes). |
| `docs/superpowers/specs/phase-2-findings/group-D-postwrite.json` | Task 7 | Verifier pass after writer outputs land. |
| `docs/superpowers/specs/phase-2-corrections.md` | Task 8 | Verifier corrections + judgment-item preview. |
| `docs/superpowers/specs/phase-2-commit-plan.md` | Task 13 | Draft commit messages for Phase II commits. |
| `docs/superpowers/specs/api-truth.md` | Task 12 | OVERWRITTEN: refreshed truth table with upgraded extraction conventions. |

Files MODIFIED in this phase (in working tree, not committed by Claude):
- `gauNEGF/*.py` (docstring additions / replacements only; no logic
  changes -- modules touched depend on gap inventory output)
- `docs/source/api/index.rst` (new module entries per Task 9)
- `.gitignore` (HTML patterns per Task 11)

Files DELETED in this phase (working tree):
- `docs/index.html`, `docs/installation.html`, `docs/quickstart.html`,
  `docs/genindex.html`, `docs/search.html`
- `docs/api/*.html`, `docs/examples/*.html`, `docs/theory/*.html`
- `docs/objects.inv`, `docs/searchindex.js`, `docs/.nojekyll` (if present)
- (Total ~16-19 files, confirmed by Task 10 pre-check.)

---

## Task 1: Verify prerequisites + Phase I close-out

**Files:** none modified; read-only checks.

- [ ] **Step 1: Confirm Phase I commits landed (or user waiver)**

Run:
```bash
cd /gscratch/anantram/willll/NEGFCode && git log --oneline -20
```

Expected: recent commits include the Phase I work (truth table, findings,
group fixes, judgment fixes). Or, if working tree is dirty with Phase I
edits not yet committed, ASK the user via `AskUserQuestion`:
"Phase I changes are in working tree but not committed. Phase II's
inputs (api-truth.md, modified .py files, etc.) depend on them being
the right state. Do you want to (a) commit Phase I now and resume,
(b) proceed with the working-tree state as-is, or (c) pause Phase II
until Phase I commits are reviewed?"

If user picks (b): proceed but note in Task 13 final summary.
If user picks (a) or (c): STOP and wait for user.

- [ ] **Step 2: Confirm spec is the post-retrospective version**

Run:
```bash
grep -c "Phase I Retrospective" /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/2026-05-09-doc-overhaul-design.md
```
Expected: `1` (the section exists). If `0`, the spec was reverted; STOP
and surface to user.

- [ ] **Step 3: Confirm api-truth.md exists**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
```
Expected: file exists, non-empty.

- [ ] **Step 4: Confirm gauNEGF source modules intact**

Run:
```bash
ls /gscratch/anantram/willll/NEGFCode/gauNEGF/*.py | wc -l
```
Expected: `>=14` (excluding `__init__.py` and `testANT.py`).

- [ ] **Step 5: Create writer-output and findings subdirectories**

Run:
```bash
mkdir -p /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-writer-output
mkdir -p /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-findings
```
Expected: directories created or already exist.

- [ ] **Step 6: Confirm current branch**

Run:
```bash
git -C /gscratch/anantram/willll/NEGFCode branch --show-current
```
Expected: prints current branch (likely `testing` per session start).
DO NOT mutate; read only.

---

## Task 2: Build the gap inventory

**Files:**
- Create: `docs/superpowers/specs/phase-2-gaps.md`

- [ ] **Step 1: Dispatch the gap-inventory agent**

Use the Agent tool with `subagent_type: "general-purpose"`, `model: "haiku"`,
`run_in_background: false`. Prompt verbatim:

```
You are auditing docstring coverage and style across the gauNEGF Python
package for Phase II of the documentation overhaul.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. API truth table: docs/superpowers/specs/api-truth.md (READ THIS FIRST)
2. The 14 in-scope modules:
   gauNEGF/{scf,scfE,density,transport,surfG1D,surfG3D,surfGBethe,
   surfGTester,matTools,integrate,spinTools,utils,protocols,config,
   fermiSearch}.py

Method, applied per module:

1. Read the source file. Note the module-level docstring status:
   "present" / "absent" / "stub" (stub = under 20 chars, generic, or
   placeholder text like "TODO").

2. For every public symbol (class, top-level function, public method)
   in the truth table, check the actual current docstring in the
   source file. Categorize each as:
   - present (substantive, parameters documented)
   - thin (one-line summary only, no Parameters section)
   - stub (placeholder, "TODO", under 20 chars)
   - absent (no docstring at all)

3. Detect docstring style for the module. Look at the dominant style
   used for substantive docstrings:
   - "numpy" -- uses Parameters / Returns headers with dash underlines
   - "google" -- uses Args: / Returns: with colon notation
   - "rst" -- uses :param X:, :returns:, sphinx field list style
   - "mixed" -- multiple styles in same module
   - "none" -- not enough docstrings to determine
   Note the dominant style. If "mixed", note which styles are present.

4. For modules in the truth table's missing_modules list (surfG3D,
   spinTools, utils, protocols, config, fermiSearch -- per Group C
   recheck), confirm api/index.rst still does NOT register them.
   For modules already registered (scf, scfE, density, transport,
   surfG1D, surfGBethe, surfGTester, matTools, integrate), note
   "registered".

Output: write to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-gaps.md using the Write tool.

File format (exact structure required):

# Phase II Gap Inventory

Date: <today>
Source: gauNEGF/*.py + docs/superpowers/specs/api-truth.md
Method: AST-equivalent inspection by haiku agent

## Module summary

| Module | Module docstring | Style | API index | Public symbols (P/T/S/A) |
|--------|------------------|-------|-----------|--------------------------|
| scf    | present          | numpy | registered | 16 / 0 / 0 / 0          |
| ...    | ...              | ...   | ...        | ...                     |

(P = present, T = thin, S = stub, A = absent. Total per row should
match truth-table public symbol count for that module.)

## Per-module detail

### gauNEGF.scf

- Module docstring: present | absent | stub
- Style detected: numpy | google | rst | mixed | none
- API index: registered | missing
- Symbol gaps:
  - `NEGF.someMethod`: absent
  - `NEGF.otherMethod`: stub
  - (omit symbols with status "present" -- they are not gaps)

(Repeat per module. Modules with zero gaps still get a one-line entry
saying "No docstring gaps; style detected = X".)

## Aggregate gap counts

- Total absent docstrings across all modules: <int>
- Total stub docstrings: <int>
- Total thin docstrings: <int>
- Modules with absent module-level docstring: [list]
- Modules with mixed style: [list]

When done, return a one-paragraph summary: total modules audited,
total gap count, modules with the largest gap counts (top 3), any
modules where style detection failed.

Do NOT modify any source file. ASCII only in the output. Use Read
tool for inputs and Write tool for the gaps file. Do NOT execute code.
```

Run as: single Agent tool invocation.

Expected runtime: 5-10 minutes for haiku.

- [ ] **Step 2: Validate gap inventory output**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-gaps.md
wc -l /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-gaps.md
grep -c "^### gauNEGF\." /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-gaps.md
```
Expected: file exists; line count > 50; per-module section count = 14.

If line count < 30 or section count != 14: re-dispatch the agent with
a corrective prompt naming the discrepancy.

- [ ] **Step 3: Spot-check known-absent docstrings against truth table**

The Phase I truth table noted these absent docstrings:
- `gauNEGF.utils.inv` (function)
- `gauNEGF.utils.eig` (function)
- `gauNEGF.utils.eigh` (function)
- `gauNEGF.density.calcEmin` (function)

Read `phase-2-gaps.md` and confirm all 4 appear in their respective
module's "Symbol gaps" section with status "absent" or "stub".

If any is missing: the agent missed it; re-dispatch with the missing
list as additional context.

- [ ] **Step 4: Spot-check known-absent module docstrings**

The Phase I truth table noted module-level docstring "absent" for
`gauNEGF.surfG1D`. Confirm `phase-2-gaps.md` lists `surfG1D` with
"Module docstring: absent".

If missing: re-dispatch agent.

- [ ] **Step 5: Append commit-plan entry (use SP-3)**

Append to `docs/superpowers/specs/phase-2-commit-plan.md` (create if
needed) the following draft:

```
## Commit candidate: Phase II gap inventory

Files: docs/superpowers/specs/phase-2-gaps.md

Suggested message:
docs(spec): add Phase II gap inventory for docstring coverage audit

Per-module docstring status (present/thin/stub/absent), detected
docstring style, and API index registration. Drives Phase II Step 3
writer dispatch + Step 5 API index update.
```

- [ ] **Step 6: Hand off**

Tell user: "Gap inventory written to phase-2-gaps.md. <N> total gaps
across <M> modules. Reviewing decisions next."

---

## Task 3: API index decisions

**Files:**
- Create: `docs/superpowers/specs/phase-2-decisions.md`

This task confirms the Phase II Step 2 decision table from the spec.
Defaults are pre-loaded (incl. the post-Phase-I `utils` flip from
Exclude to Include). The user can override per module.

- [ ] **Step 1: Read gap inventory and prepare summaries**

Read `docs/superpowers/specs/phase-2-gaps.md` to get current per-module
state. For each of the 6 modules under decision (`surfG3D`, `spinTools`,
`config`, `protocols`, `utils`, `fermiSearch`), prepare a one-sentence
summary based on the gap inventory + the Phase I usage scan.

- [ ] **Step 2: Present decisions to user via AskUserQuestion**

Three questions, batched in one `AskUserQuestion` call:

Question 1: "Confirm the API index decisions for the 6 modules not
currently in api/index.rst. Defaults are pre-loaded from the spec
(with utils flipped to Include after Phase I exploration confirmed
user-facing usage in examples + 3 tests). Override any?"
Header: "API index defaults"
multiSelect: false
options:
1. "Accept all defaults (Recommended)" -- "surfG3D, spinTools, config,
   utils -> Include in main API; protocols -> Developer Reference;
   fermiSearch -> Exclude (deprecated). Apply as-is."
2. "Accept all but reconsider one" -- "I'll specify which one needs a
   different decision in a follow-up."
3. "Reconsider multiple" -- "I want to walk through each module
   individually."
4. "Use a totally different scheme" -- "Let me describe the index
   layout I want."

Question 2: "Where should the protocols module appear?"
Header: "protocols placement"
multiSelect: false
options:
1. "Separate Developer Reference section (Recommended)" -- "Per spec
   default. Distinguishes contributor interface from user-facing API."
2. "Flat under main API" -- "Treat like any other module; users
   browsing the index see it inline."
3. "Bottom of API index with collapsible section" -- "Visible but
   visually distinct."
4. "Exclude" -- "Internal, not for documentation."

Question 3: "fermiSearch is deprecated per its own docstring header.
Decision?"
Header: "fermiSearch decision"
multiSelect: false
options:
1. "Exclude from API docs (Recommended)" -- "Module docstring still
   gets a clear 'DEPRECATED -- use density.calcFermi*' pointer for
   in-source readers, but no Sphinx page."
2. "Include with prominent DEPRECATED banner" -- "Render in API but
   make deprecated status visually obvious."
3. "Include silently" -- "Just register it like any module."
4. "Delete entirely" -- "Remove the module from the codebase. NOTE:
   out of Phase II scope; would require Phase II amendment."

- [ ] **Step 3: Apply user decisions to the decision artifact**

Use Write tool to create `docs/superpowers/specs/phase-2-decisions.md`
with the resolved per-module decisions. Format:

```markdown
# Phase II Step 2 Decisions

Date: <today>
Source: User answers via AskUserQuestion in Phase II Task 3.

| Module | Decision | Notes |
|--------|----------|-------|
| surfG3D | Include in main API | <user note or default rationale> |
| spinTools | Include in main API | <...> |
| config | Include as Configuration Reference page | <...> |
| protocols | <user choice> | <...> |
| utils | Include in main API (flipped from spec default per Phase I usage scan) | <...> |
| fermiSearch | <user choice> | <...> |

Drives Task 9 (API index update).
```

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: Phase II Step 2 decisions

Files: docs/superpowers/specs/phase-2-decisions.md

Suggested message:
docs(spec): record Phase II Step 2 API index decisions

User-confirmed decisions for the 6 previously-unregistered modules
(surfG3D, spinTools, config, protocols, utils, fermiSearch). Drives
the api/index.rst update in Task 9.
```

- [ ] **Step 5: Hand off**

Tell user: "Decisions recorded. Writers next; they will produce
docstring drafts for symbols missing or stubbed."

---

## Task 4: Pre-check the utils inclusion is solid

**Files:** none modified.

- [ ] **Step 1: Re-grep for user-facing utils usage**

Run:
```bash
cd /gscratch/anantram/willll/NEGFCode && grep -rn "from gauNEGF.utils\|gauNEGF\.utils\." examples/ tests/ docs/ 2>/dev/null
```
Expected: at least the known references (examples/SiNEGF.py,
tests/test_calcTSW.py, tests/test_transport_dos_crossterm.py,
tests/test_surfG1D_features.py) plus any new ones found.

- [ ] **Step 2: If user picked "Exclude utils" in Task 3, surface conflict**

If Task 3 resolved utils to anything other than "Include", and Step 1
above found user-facing usage, ASK the user:
"You picked <decision> for utils, but Phase I + this re-check found
user-facing usage (<list of files>). Reaffirm exclude (orphans those
users), or flip to Include?"

If reaffirm exclude: proceed with that decision. If flip: update
phase-2-decisions.md accordingly.

- [ ] **Step 3: No commit (sanity-check task only)**

---

## Task 5: Dispatch 3 parallel docstring writers

**Files:**
- Create: `docs/superpowers/specs/phase-2-writer-output/group-D{1,2,3}.json`

This task dispatches one writer per group. Writers produce a JSON
manifest of docstring inserts/replacements; SP-4 in Task 6 applies
them.

- [ ] **Step 1: Read inputs into context**

Use Read tool on:
- `docs/superpowers/specs/api-truth.md`
- `docs/superpowers/specs/phase-2-gaps.md`
- `docs/superpowers/specs/phase-2-decisions.md`

(Required so that if a writer dispatch fails, you can diagnose.)

- [ ] **Step 2: Dispatch all 3 writer agents in a single message**

Use the Agent tool 3 times in ONE assistant message (parallel execution).
All 3 use `subagent_type: "general-purpose"`, `model: "haiku"`,
`run_in_background: false`.

Per-dispatch parameter table:

| Dispatch | [GROUP] | [MODULES] | [OUTPUT_PATH] |
|----------|---------|-----------|---------------|
| W1 | D1 | gauNEGF/{scf,scfE,density,transport}.py | docs/superpowers/specs/phase-2-writer-output/group-D1.json |
| W2 | D2 | gauNEGF/{surfG1D,surfG3D,surfGBethe,surfGTester}.py | docs/superpowers/specs/phase-2-writer-output/group-D2.json |
| W3 | D3 | gauNEGF/{matTools,integrate,spinTools,utils,protocols,config,fermiSearch}.py | docs/superpowers/specs/phase-2-writer-output/group-D3.json |

Shared writer prompt template:

```
You are writing docstrings for previously-undocumented or stub-documented
public symbols in the gauNEGF Python package, for Phase II Step 3 of the
documentation overhaul.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. API truth table: docs/superpowers/specs/api-truth.md (READ FIRST)
2. Gap inventory: docs/superpowers/specs/phase-2-gaps.md (READ FIRST,
   filter to your group's modules)
3. Modules to write for: [MODULES]

[INSERT VERIFIER GUARDRAILS BLOCK HERE -- verbatim from SP-5 in plan]

Method, applied to each module in your list:

1. Read the source file in full.
2. Read the gap inventory entries for this module to find which
   symbols need work.
3. For each symbol with status "absent" or "stub":
   a. Read the function/class body to understand what it does.
   b. Detect the dominant docstring style for this module from the
      gap inventory. If style is "numpy", write numpy-style. If
      "google", google-style. If "rst", sphinx-rst. If "mixed",
      default to numpy. If "none" (no other docstrings to model on),
      default to numpy.
   c. Write a docstring with these sections in order:
      - One-line summary (imperative or declarative; match module
        style).
      - Optional 1-3 paragraph extended description if the function
        has non-obvious behavior.
      - Parameters section: every parameter from the actual signature,
        with type from annotation if present, default value if any,
        one-line description from inferred behavior.
      - Returns section: ONLY if the function returns a value (not
        just None / mutates state). Type + brief description. For
        variable-shape returns (see guardrail #3), document each
        case.
      - Raises section: ONLY if the function body has explicit
        `raise X(...)` calls. List each exception type and the
        condition.
      - Notes section: ONLY when there are non-obvious side effects
        visible in the body (mutates self, writes to disk,
        JAX-traced behavior, etc.). NEVER add Notes for filler.
   d. DO NOT fabricate usage examples. Examples are Phase III.
   e. If a function is too complex to summarize accurately from
      inspection, leave a TODO marker (see TODO ceiling below).

4. For each symbol with status "thin" (one-line only, no Parameters):
   Augment with the missing sections per (3c) but PRESERVE the
   existing one-line summary if it is accurate.

5. For modules with status "absent" / "stub" module-level docstring:
   Write a module-level docstring with:
   - One-line summary of the module's purpose.
   - 1-3 paragraph extended description listing the main exports
     and the use case.
   - DO NOT include "Last updated" or "Author" lines (style noise).

TODO ceiling: at most 3 TODO markers per module. If your output for
one module would exceed 3 TODOs, STOP at 3 and emit `"todo_overflow":
true` for that module. The main session will surface this to the
user for per-module decision (write by hand, accept all TODOs, or
defer to Phase III).

Output: write JSON to [OUTPUT_PATH] using the Write tool.

JSON schema (exact):

{
  "group": "[GROUP]",
  "modules_processed": ["path1", "path2", ...],
  "entries": [
    {
      "file_path": "gauNEGF/scf.py",
      "symbol": "NEGF.someMethod" or "module-level",
      "kind": "insert" or "replace_stub" or "todo",
      "anchor_line": <int -- 1-indexed line where the def or class
                     declaration lives, OR 0 for module-level>,
      "anchor_text": "VERBATIM def line including trailing colon and
                      surrounding context (the next 1-2 lines of body)
                      so SP-4 can locate the insertion point uniquely",
      "docstring": "VERBATIM new docstring text including the
                    triple-quotes (\"\"\" both sides), with proper
                    indentation matching the function body",
      "style": "numpy" or "google" or "rst",
      "todo_count_so_far": <int -- only set if kind=todo, the running
                            todo count for this module after this entry>
    }
  ],
  "summary": {
    "total_entries": <int>,
    "inserts": <int>,
    "replaces": <int>,
    "todos": <int>,
    "modules_with_todo_overflow": ["modulename", ...]
  }
}

If a module has zero gaps to fill: still list it in modules_processed,
emit zero entries for it.

Do NOT modify any source file. Use Write tool ONLY for the JSON
output. Use Read tool for inputs.

ASCII only in the JSON output. No unicode characters (no smart quotes,
no em-dashes, no math symbols).

When done, return a one-paragraph summary: modules processed, total
entries, breakdown by kind, modules with todo_overflow, any modules
where docstring style detection failed and you defaulted to numpy.
```

- [ ] **Step 3: Wait for all 3 writers to return**

Since `run_in_background: false`, the parallel dispatch blocks until
all complete.

If any one fails: note which one, dispatch a single replacement.

- [ ] **Step 4: Validate each output JSON parses**

Run:
```bash
for g in D1 D2 D3; do
  python3 -c "import json; d=json.load(open('/gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-writer-output/group-${g}.json')); s=d['summary']; print(f'{g}: {s[\"total_entries\"]} entries ({s[\"inserts\"]} ins, {s[\"replaces\"]} repl, {s[\"todos\"]} todos); todo_overflow modules: {s[\"modules_with_todo_overflow\"]}')"
done
```
Expected: 3 lines, one per group, valid JSON.

If any group's JSON does not parse: re-dispatch that one writer with
"Your previous output was not valid JSON. Re-emit with strict JSON syntax."

- [ ] **Step 5: No commit (intermediate artifacts; bundled with applied
docstrings in Task 6 commit)**

---

## Task 6: Apply writer output to source files

**Files:**
- Modify: `gauNEGF/*.py` (per writer entries)
- Create: `docs/superpowers/specs/phase-2-todos.md` (if any todo_overflow)

- [ ] **Step 1: Read all 3 writer output JSONs**

Use Read tool on each of `group-{D1,D2,D3}.json`.

- [ ] **Step 2: Handle todo_overflow modules first**

For each module in `modules_with_todo_overflow` across the 3 groups,
ASK the user via `AskUserQuestion`:

"Module <X> would have N TODOs (>3 ceiling). Decision?"
Header: "TODO overflow <module>"
multiSelect: false
options:
1. "Accept all N TODOs" -- "Apply every entry as TODO; user fills in later."
2. "Defer module entirely to Phase III" -- "Skip all entries for this
   module in Phase II. Module's gaps remain."
3. "I'll write the docstrings by hand" -- "Skip all entries; user
   will fill in personally before Phase III."
4. "Apply only entries marked 'insert' / 'replace_stub' (skip todos)" --
   "Drop the TODOs but accept the substantive entries the writer
   produced."

Record decisions in `docs/superpowers/specs/phase-2-todos.md`:

```markdown
# Phase II Module TODO-Overflow Decisions

Date: <today>

## Module: gauNEGF/<X>.py

Writer produced N TODOs (ceiling 3). User decision: <choice>
Action taken: <description>

(Repeat per overflow module.)

## Modules under ceiling

(List per-module todo counts for modules NOT in overflow.)
```

- [ ] **Step 3: For each entry per group, apply via SP-4**

Process entries in this order: D1 first, then D2, then D3 (so that
inter-module cross-references between writer outputs land in
dependency order if any exist).

Within each group, for each entry NOT skipped per Step 2:
- If `kind == "insert"`: use SP-4 insert procedure.
- If `kind == "replace_stub"`: use SP-4 replace_stub procedure.
- If `kind == "todo"`: use SP-4 todo procedure.

Track applied / skipped count per group.

- [ ] **Step 4: Append commit-plan entries (use SP-3)**

Append three entries to `phase-2-commit-plan.md`, one per writer group:

```
## Commit candidate: D1 group docstring additions

Files: gauNEGF/scf.py, gauNEGF/scfE.py, gauNEGF/density.py, gauNEGF/transport.py

Suggested message:
docs(gauNEGF): fill docstring gaps in core modules (D1)

Phase II Step 3 writer output applied to scf, scfE, density, transport.
<N> docstrings inserted, <M> stubs replaced, <K> TODOs left for human
review. Style matched per-module dominant convention (numpy / google /
rst). No logic changes.

(Repeat per group D2, D3.)
```

- [ ] **Step 5: Hand off**

Tell user: "Writer output applied. <N total> docstrings landed across
3 groups. <K total> TODOs. Verifier pass next."

---

## Task 7: Verifier pass on new docstrings

**Files:**
- Create: `docs/superpowers/specs/phase-2-findings/group-D-postwrite.json`

- [ ] **Step 1: Dispatch the verifier agent**

Use the Agent tool with `subagent_type: "general-purpose"`,
`model: "haiku"`, `run_in_background: false`. Prompt verbatim:

```
You are verifying that newly-written docstrings in the gauNEGF Python
package match the actual function signatures and behavior, for Phase II
Step 4.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. API truth table: docs/superpowers/specs/api-truth.md (READ FIRST)
2. The 14 in-scope modules:
   gauNEGF/{scf,scfE,density,transport,surfG1D,surfG3D,surfGBethe,
   surfGTester,matTools,integrate,spinTools,utils,protocols,config,
   fermiSearch}.py
3. Phase II writer outputs to focus your attention on:
   docs/superpowers/specs/phase-2-writer-output/group-D{1,2,3}.json

[INSERT VERIFIER GUARDRAILS BLOCK HERE -- verbatim from SP-5 in plan]

Method, applied to each module:

1. Read the source file in full.
2. For every symbol that appears as an entry in the writer outputs
   (focused attention) AND every other symbol with a docstring (full
   sweep), verify:
   - Parameters section names match the actual function signature
     EXACTLY (no extras, no missing).
   - Return section type and shape claims match what the function
     body actually returns (apply guardrail #3).
   - Cross-references (`:func:`, `:class:`, See Also lists) name
     real symbols in the truth table.
   - For embedded code examples (>>> doctests, .. code-block:: python),
     AST-equivalent inspection per Phase I verifier method, with the
     guardrails block applied.

3. Apply SP-6 (quality-degrading-fix detection): if your proposed_fix
   would degrade the docs (comment out, decorative kwarg, paraphrase),
   flag the finding as `confidence: "judgment"` with a note
   `degrading_fix: true` so the main session can route appropriately.

4. For EVERY mismatch you find, produce a finding:
   - file_path
   - line_range
   - current_snippet (verbatim)
   - issue (one short sentence)
   - proposed_fix (verbatim corrected text)
   - confidence: "mechanical" | "judgment"
   - degrading_fix: true | false (set true per SP-6)
   - code_bug: true ONLY if (a) the docstring matches a documented
     public API and (b) the source's actual behavior differs. NEVER
     set true based on truth-table absence alone.

Output: write JSON to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-findings/group-D-postwrite.json with this schema:

{
  "phase": "II-postwrite",
  "files_audited": [...],
  "findings": [
    {
      "file_path": "...",
      "line_range": [<start>, <end>],
      "current_snippet": "...",
      "issue": "...",
      "proposed_fix": "...",
      "confidence": "mechanical",
      "degrading_fix": false,
      "code_bug": false
    }
  ],
  "summary": {
    "total_findings": <int>,
    "mechanical_count": <int>,
    "judgment_count": <int>,
    "code_bug_count": <int>,
    "degrading_fix_count": <int>
  }
}

If zero findings: write the JSON with empty findings array.

ASCII only. Do NOT modify any source file. Use Write tool ONLY for
the output JSON.

When done, return a one-paragraph summary: files audited, total
findings, breakdown by confidence + degrading_fix, code_bug findings
highlighted (rare; the truth-table guardrails should suppress most
false positives).
```

- [ ] **Step 2: Validate output JSON parses**

Run:
```bash
python3 -c "
import json
d = json.load(open('/gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-findings/group-D-postwrite.json'))
s = d['summary']
print(f'Postwrite: {s[\"total_findings\"]} findings ({s[\"mechanical_count\"]} mech, {s[\"judgment_count\"]} judg, {s[\"code_bug_count\"]} bug, {s[\"degrading_fix_count\"]} degrading)')
"
```
Expected: one line, valid JSON.

- [ ] **Step 3: No commit (intermediate; rolls into Task 8 corrections commit)**

---

## Task 8: Apply verifier corrections + judgment preview

**Files:**
- Create: `docs/superpowers/specs/phase-2-corrections.md`
- Modify: `gauNEGF/*.py` (only where corrections needed)

This task is split: real mechanical fixes apply via SP-1 directly;
judgment items get previewed to user via AskUserQuestion BEFORE any
application (per the "surface judgment items early" Phase I lesson).

- [ ] **Step 1: Read postwrite findings**

Use Read tool on
`docs/superpowers/specs/phase-2-findings/group-D-postwrite.json`.

- [ ] **Step 2: Categorize findings**

Mentally split:
- **Real mechanical**: confidence=="mechanical" AND degrading_fix==false AND code_bug==false
- **Degrading mechanical (skip)**: degrading_fix==true (regardless)
- **Judgment**: confidence=="judgment" AND degrading_fix==false AND code_bug==false
- **Code bug**: code_bug==true (rare given guardrails; goes to phase-2-bugs.md)

Count per bucket.

- [ ] **Step 3: Write phase-2-corrections.md PREVIEW**

Use Write tool. Format:

```markdown
# Phase II Verifier Corrections (preview before application)

Date: <today>
Source: postwrite verifier pass at phase-2-findings/group-D-postwrite.json

## Summary

- Real mechanical fixes (will be applied automatically): N
- Degrading-fix skips (per SP-6): K
- Judgment fixes (require user approval): M
- Code-bug flags (NOT auto-applied): J

## Real mechanical fixes

(One block per fix, current_snippet + proposed_fix verbatim.)

## Judgment fixes (require user approval)

(Same format with "Why this needs judgment" line.)

## Degrading-fix skips

(Same format with "Why skipped" line citing the SP-6 pattern matched.)

## Code-bug flags

(Same format. These are NOT auto-applied; user triages.)
```

- [ ] **Step 4: Apply real mechanical fixes via SP-1**

For each fix in the "Real mechanical" bucket: apply via SP-1.
Track applied / skipped count.

- [ ] **Step 5: Surface judgment fixes to user via AskUserQuestion**

If zero judgment fixes: skip this step.

Otherwise: present judgment fixes 4-at-a-time per `AskUserQuestion`
call. Per fix, options:
- "Approve" -- apply proposed_fix verbatim
- "Approve with edit" -- ask user for corrected text in follow-up
- "Reject" -- do not apply
- "Defer" -- log for later, do not apply now

For each "Approve" / "Approve with edit": apply via SP-1.

- [ ] **Step 6: Update phase-2-corrections.md with applied/rejected/deferred status**

Re-write the file with each fix annotated by its disposition.

- [ ] **Step 7: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: Phase II verifier corrections

Files: gauNEGF/<modules touched in Tasks 6+8>

Suggested message:
docs(gauNEGF): apply verifier corrections to new docstrings (Phase II)

After Phase II Step 3 writers landed new docstrings, the Step 4
verifier found <N> mechanical mismatches (mostly parameter name
typos and type-tag drift) which were applied directly. <M> judgment
items were user-approved or rejected per
phase-2-corrections.md. <K> degrading-fix proposals were skipped
per the SP-6 quality guard.
```

- [ ] **Step 8: Hand off**

Tell user: "Corrections applied. <N> mechanical fixes, <M> judgment
approvals, <K> skipped degrading fixes. Truth table refresh next."

---

## Task 9: API index update

**Files:**
- Modify: `docs/source/api/index.rst`

- [ ] **Step 1: Read current api/index.rst**

Use Read tool on `/gscratch/anantram/willll/NEGFCode/docs/source/api/index.rst`.

- [ ] **Step 2: Read decisions artifact**

Use Read tool on `/gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-decisions.md`.

- [ ] **Step 3: Plan the new structure**

Based on user decisions in Task 3:

- Modules to ADD as automodule under existing sections:
  - `surfG3D` -- under "Contact Models" (3D Contacts subsection)
  - `spinTools` -- under "Utilities" (Spin Tools subsection)
  - `utils` -- under "Utilities" (JIT/Linear Algebra Helpers
    subsection)

- Sections to potentially CREATE:
  - "Configuration Reference" (top-level after API Reference) -- if
    config decision was Include.
  - "Developer / Extensibility Reference" (top-level, after
    Configuration Reference) -- if protocols decision was that.

- Modules to DOCUMENT WITH DEPRECATION BANNER:
  - `fermiSearch` -- if user picked include with banner; otherwise
    skip.

- [ ] **Step 4: Apply the api/index.rst edits**

Use Edit tool to add entries. Use the existing automodule pattern:

```rst
3D Contacts
----------

.. automodule:: gauNEGF.surfG3D
   :members:
   :undoc-members:
   :show-inheritance:
```

Apply each addition as a separate Edit call (one per new automodule
block) so the diff stays reviewable.

For new top-level sections (Configuration Reference,
Developer Reference), use the existing top-level section heading
pattern (=== underline).

- [ ] **Step 5: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: API index update (Phase II Step 5)

Files: docs/source/api/index.rst

Suggested message:
docs(sphinx): register Phase II module decisions in API index

Adds:
- gauNEGF.surfG3D under Contact Models
- gauNEGF.spinTools under Utilities
- gauNEGF.utils under Utilities (flipped from spec default per Phase I usage scan)
- Configuration Reference section for gauNEGF.config
- Developer / Extensibility Reference for gauNEGF.protocols
- (fermiSearch handling per user decision, see phase-2-decisions.md)

Resolves the Phase I "missing modules" gap surfaced by Group C verifier.
```

- [ ] **Step 6: Hand off**

Tell user: "API index updated. <N> new module entries + <M> new
sections."

---

## Task 10: Pre-HTML-deletion check

**Files:** none modified (read + ask user).

- [ ] **Step 1: List the HTML files to be deleted**

Run:
```bash
cd /gscratch/anantram/willll/NEGFCode && find docs -maxdepth 2 -name "*.html" -o -name "objects.inv" -o -name "searchindex.js" -o -name ".nojekyll" 2>/dev/null | sort
```
Expected: list of ~16-19 files. Note: confirm no `docs/source/**/*.html`
in the list (those would be source files, not build output).

- [ ] **Step 2: Confirm GitHub Pages config builds from Sphinx (not /docs)**

ASK the user via `AskUserQuestion`:
"Phase II Step 6 deletes the pre-rendered HTML under docs/. This is
safe ONLY if GitHub Pages is configured to build from Sphinx source
(via a GitHub Action / workflow), NOT to serve /docs directly.
Confirm Pages config?"
Header: "Pages config"
multiSelect: false
options:
1. "Pages builds from Sphinx via a workflow (Recommended -- safe to delete)"
2. "Pages serves /docs directly (UNSAFE -- block deletion)"
3. "I don't know -- let me check first"
4. "Pages is disabled / I don't use Pages"

If user picks option 1 or 4: proceed to Task 11.
If user picks option 2: STOP. Surface "Phase II HTML deletion blocked
until Pages config is changed. Phase II remains in incomplete state
until then."
If user picks option 3: STOP. Wait for user to check.

- [ ] **Step 3: Read .gitignore current state**

Use Read tool on `/gscratch/anantram/willll/NEGFCode/.gitignore`.

- [ ] **Step 4: No commit (sanity-check task)**

---

## Task 11: HTML deletion + .gitignore update

**Files:**
- Delete: pre-rendered HTML files per Task 10 list
- Modify: `.gitignore`

- [ ] **Step 1: Delete pre-rendered HTML**

Run:
```bash
cd /gscratch/anantram/willll/NEGFCode && rm -f \
  docs/index.html docs/installation.html docs/quickstart.html \
  docs/genindex.html docs/search.html \
  docs/objects.inv docs/searchindex.js docs/.nojekyll \
  docs/api/*.html docs/examples/*.html docs/theory/*.html
```

Expected: silently completes (rm -f tolerates missing files).

Confirm with:
```bash
find /gscratch/anantram/willll/NEGFCode/docs -maxdepth 2 -name "*.html" -o -name "objects.inv" -o -name "searchindex.js" -o -name ".nojekyll" 2>/dev/null
```
Expected: empty output. Any remaining files: re-list and surface to
user.

- [ ] **Step 2: Update .gitignore**

Append HTML patterns ONLY if not already present. Use Edit tool to
add the following block at the end of `.gitignore` (or create the
file if absent):

```
# Sphinx build artifacts (Pages builds from source, not committed)
docs/*.html
docs/**/*.html
docs/genindex.html
docs/objects.inv
docs/searchindex.js
docs/.nojekyll
```

Pre-check: read .gitignore first; if any of those patterns are
already present, skip duplicates.

- [ ] **Step 3: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: pre-rendered HTML deletion (Phase II Step 6)

Files: 16-19 deleted HTML/build artifacts under docs/, .gitignore

Suggested message:
docs: remove pre-rendered HTML; rely on GitHub Pages Sphinx build

GitHub Pages now builds from docs/source via a workflow, so the
committed pre-rendered HTML under docs/*.html, docs/api/*.html,
docs/examples/*.html, docs/theory/*.html, docs/objects.inv,
docs/searchindex.js, docs/.nojekyll are dead weight that drift
relative to source.

Adds matching patterns to .gitignore to prevent re-commit on local
sphinx-build runs.

Pages config confirmed source-built per Phase II Task 10 user check.
```

- [ ] **Step 4: Hand off**

Tell user: "Pre-rendered HTML deleted, .gitignore updated. Truth
table refresh next."

---

## Task 12: Truth table refresh with upgraded extraction

**Files:**
- Modify (overwrite): `docs/superpowers/specs/api-truth.md`

This task uses an UPGRADED extraction prompt that addresses the 5
truth-table blind spots from the Phase I retrospective.

- [ ] **Step 1: Backup existing truth table**

Run:
```bash
cp /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md \
   /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.phase1-snapshot.md
```
Expected: silent success. The snapshot is for traceability; will be
deleted in Phase III if the refresh validates clean.

- [ ] **Step 2: Dispatch the upgraded truth-table builder**

Use the Agent tool with `subagent_type: "general-purpose"`,
`model: "haiku"`, `run_in_background: false`. Prompt verbatim:

```
You are extracting a complete API reference for the gauNEGF Python
package, with upgraded extraction conventions added after the Phase I
truth table revealed five blind spots.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs to read: every file matching gauNEGF/*.py (use the Read tool;
do NOT execute any code). 14 modules: scf, scfE, density, transport,
surfG1D, surfG3D, surfGBethe, surfGTester, matTools, integrate,
spinTools, utils, protocols, config, fermiSearch. Skip __init__.py
and testANT.py.

UPGRADED EXTRACTION CONVENTIONS (must be applied):

1. CLASS INHERITANCE CONSTRUCTORS. When a class is defined as
   `class X(Y):` and has no explicit `__init__`, document its
   constructor as `inherited from Y.__init__`. INLINE the parent's
   __init__ signature into the per-module section so downstream
   readers do not have to cross-reference.

2. DYNAMIC ATTRIBUTES. Add a "Dynamic attributes" sub-section under
   each class. List every attribute set by methods OTHER than
   __init__, with a one-line note on which method sets it.

   Example detection: read each method body. Any line of form
   `self.<name> = <expr>` that is NOT `self._<name>` (private
   convention) and NOT in __init__ is a dynamic attribute.

3. VARIABLE-SHAPE RETURNS. For each function, in addition to the
   signature, examine the return statements. If the function has
   multiple `return` statements with different shapes (scalar vs
   tuple, different tuple arities), add a "Returns" sub-line:
   "Variable shape: <case> -> <shape>" for each branch. If all
   returns have the same shape, no extra line needed.

4. METHOD OVERLOADS VIA DUCK TYPING. For each function/class, scan
   the body for `isinstance` checks, `hasattr` checks, or other
   type-dispatching code. If the function dispatches on the first
   argument's type, add an "Overloads" sub-line listing the
   supported types.

5. NESTED-LIST PARAMETER CONVENTIONS. Where a parameter docstring
   mentions "list of contacts" or similar, AND the function body
   indexes the parameter as `param[i][j]`, note the nested
   convention in a "Parameter shapes" sub-line.

For each .py file:

a. Read the file in full.
b. Note module-level docstring presence.
c. For every public class and top-level function (name does NOT
   start with underscore):
   - Symbol name
   - Full signature (positional, defaults, kwargs, *args, **kwargs,
     type annotations)
   - Decorators
   - Docstring presence: present | absent | stub
   - Module-level deprecation marker
d. For every class, public methods + the upgrades above.
e. For every function, the upgrades above (where applicable).

Output: write to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md using the Write tool. OVERWRITE the existing file.

File format (per-module section):

### Module: gauNEGF.<module>

- Module docstring: present | absent | stub
- Module-level deprecation marker: none | "<text>"
- Public classes: N
- Public functions: N

#### class ClassName

Signature: `ClassName(arg1, arg2=default, ...)`
Inherits constructor from: <parent class with full inherited signature, OR "explicit __init__">
Docstring: present | absent | stub
Deprecated: true | false
Decorators: [list or empty]

Methods:
- `__init__(self, ...)` -- docstring: present|absent|stub
  (omit if inherited unmodified)
- `methodName(self, arg, ...)` -- docstring: present|absent|stub
  - Overloads: <list, if any>
  - Variable returns: <list, if any>
  - Parameter shapes: <list, if any>
- ...

Dynamic attributes:
- `self.attr1` (set by `methodA`)
- `self.attr2` (set by `methodB`, `methodC`)
- (omit subsection entirely if no dynamic attrs)

#### function functionName

Signature: `functionName(arg1, ...)`
Docstring: present | absent | stub
Deprecated: true | false
Decorators: [list or empty]
Overloads: <list, if any>
Variable returns: <list, if any>
Parameter shapes: <list, if any>

(Repeat per module.)

Then a Flat Lookup Table (same format as Phase I).

Then an "Upgrade conventions applied" footer noting which modules
had inheritance constructors, dynamic attributes, variable-shape
returns, overloads, or nested-list params detected.

ASCII only. Use Read for inputs and Write for the output.

When done, return a one-paragraph summary: total modules, total
classes, total functions, total deprecated symbols, total absent
docstrings, COUNT of detected (a) inheritance constructors, (b)
dynamic attribute sets, (c) variable-shape returns, (d) overloads,
(e) nested-list params.
```

- [ ] **Step 3: Validate refreshed truth table**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
wc -l /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
```
Expected: line count > 950 (Phase I was 917; upgrades add lines).

- [ ] **Step 4: Spot-check the 5 upgrade categories landed**

Search the refreshed truth table for at least one example of each
upgrade category. Run:

```bash
T=/gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
echo "Inheritance: $(grep -c 'Inherits constructor' $T)"
echo "Dynamic attrs: $(grep -c 'Dynamic attributes:' $T)"
echo "Variable returns: $(grep -c 'Variable returns:' $T)"
echo "Overloads: $(grep -c 'Overloads:' $T)"
echo "Param shapes: $(grep -c 'Parameter shapes:' $T)"
```

Expected: all five > 0. Specifically:
- Inheritance: at least 1 (NEGFE inherits NEGF)
- Dynamic attrs: at least 2 (NEGF.setSigma sets sigma1/sigma2; NEGFE
  setters set self.g)
- Variable returns: at least 2 (calculate_transmission, calculate_dos)
- Overloads: at least 1 (SigmaCalculator)
- Param shapes: at least 1 (setContactBethe nested-list)

If any category shows 0: re-dispatch the agent with a corrective
prompt naming which category it missed.

- [ ] **Step 5: Append commit-plan entry (use SP-3)**

Append:
```
## Commit candidate: API truth table refresh (Phase II Step 7)

Files: docs/superpowers/specs/api-truth.md
       docs/superpowers/specs/api-truth.phase1-snapshot.md (delete after Phase III if clean)

Suggested message:
docs(spec): refresh API truth table with upgraded extraction conventions

Re-runs the AST-equivalent extraction with five upgrades surfaced by
the Phase I retrospective:
- Class inheritance constructors (NEGFE inherits NEGF)
- Dynamic attributes (e.g. NEGF.sigma1, NEGFE.g)
- Variable-shape returns (calculate_transmission, calculate_dos)
- Method overloads via duck typing (SigmaCalculator)
- Nested-list parameter conventions (setContactBethe contactList)

These were the five categories that caused over-aggressive verifier
flagging in Phase I. Phase III verifiers and writers consume the
refreshed table.

api-truth.phase1-snapshot.md preserves the Phase I version for
traceability; can be deleted at end of Phase III.
```

- [ ] **Step 6: Hand off**

Tell user: "Truth table refreshed. Backup at api-truth.phase1-snapshot.md.
Final Phase II summary next."

---

## Task 13: Final commit-plan + Phase II exit

**Files:**
- Update: `docs/superpowers/specs/phase-2-commit-plan.md`

- [ ] **Step 1: Verify all Phase II artifacts exist**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-gaps.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-decisions.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-corrections.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-commit-plan.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-writer-output/
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-2-findings/
```
Expected: all primary artifacts present; writer-output dir has 3 JSONs;
findings dir has 1 postwrite JSON. `phase-2-todos.md` and
`phase-2-skipped.md` exist only if applicable.

- [ ] **Step 2: Verify recommended commit ordering**

Read `phase-2-commit-plan.md`. Expected order (top to bottom):

1. Gap inventory (one commit)
2. Step 2 decisions (one commit)
3. D1 group docstring additions (one commit)
4. D2 group docstring additions (one commit)
5. D3 group docstring additions (one commit)
6. Verifier corrections (one commit, optional bundle with 3-5)
7. API index update (one commit)
8. Pre-rendered HTML deletion + .gitignore (one commit, MUST be after
   Pages config confirmed)
9. Truth table refresh (one commit, ideally last so Phase III
   consumers see the upgraded version)

If ordering is different in the file: rewrite to match.

- [ ] **Step 3: Print final summary for user**

Output to chat:

```
=== Phase II Complete (pending your commits) ===

Gap inventory:    docs/superpowers/specs/phase-2-gaps.md
Decisions:        docs/superpowers/specs/phase-2-decisions.md
Writer outputs:   docs/superpowers/specs/phase-2-writer-output/{D1,D2,D3}.json
TODOs:            docs/superpowers/specs/phase-2-todos.md (if any overflow)
Postwrite verify: docs/superpowers/specs/phase-2-findings/group-D-postwrite.json
Corrections:      docs/superpowers/specs/phase-2-corrections.md
Commit plan:      docs/superpowers/specs/phase-2-commit-plan.md (~9 commits)
Truth table:      docs/superpowers/specs/api-truth.md (refreshed; backup at api-truth.phase1-snapshot.md)

What you do next:
1. Review diffs (git diff per file group; HTML deletion is a single
   `git rm` review).
2. Run commits in the order suggested.
3. Confirm sphinx builds clean against the refreshed sources.
4. When all Phase II commits are landed, signal me to write the
   Phase III plan.

Reminder: I have not run any git command and will not.

Carryover for Phase III:
- Refreshed truth table consumed by Phase III research + guide writers.
- Verifier guardrails block (SP-5) becomes shared infrastructure;
  Phase III plan should reuse it verbatim.
- Any Phase II TODOs left in source code feed back to Phase III as
  "modules needing user-written summary" (or get filled during
  Phase III interview).
```

- [ ] **Step 4: Phase II exit**

When user confirms commits are landed, Phase II is complete.

Note for the executor: do NOT proceed to write Phase III plan
automatically. Spec exit gate requires user sign-off. Wait for
explicit user instruction to write Phase III.

---

## Self-Review (executor: confirm before starting)

Before launching Task 1, verify:

- [ ] Spec at `docs/superpowers/specs/2026-05-09-doc-overhaul-design.md`
  contains the "Phase I Retrospective" section.
- [ ] Phase I commits are landed (or user explicitly waived per Task 1
  Step 1).
- [ ] You understand: Claude does NOT commit; user does.
- [ ] You understand: subagents are haiku.
- [ ] You understand: every subagent prompt that does verification or
  writing in Phase II includes the Verifier Guardrails Block (SP-5)
  verbatim. The truth-table refresh agent is the exception (its prompt
  has different built-in guardrails).
- [ ] You understand: SP-6 (quality-degrading-fix detection) gates
  every mechanical-fix application. Skip if degrading.
- [ ] You have access to AskUserQuestion (used in Tasks 3, 4, 6, 8, 10).

If any of these is uncertain, stop and ask the user before
dispatching the gap inventory agent.

---

## Plan-level self-review (drafter: completed during writing)

- **Spec coverage:** Phase II spec Steps 1-8 all map to plan tasks
  (Step 1 -> Task 2; Step 2 -> Tasks 3-4; Step 3 -> Tasks 5-6;
  Step 4 -> Tasks 7-8; Step 5 -> Task 9; Step 6 -> Tasks 10-11;
  Step 7 -> Task 12; Step 8 -> Task 13).

- **Phase I retrospective coverage:** Five truth-table blind spots
  inlined into SP-5 + Task 12 upgraded extraction prompt. Verifier
  calibration changes inlined into SP-6. Main-session retraction
  culture noted in hard rules + Task 8 preview step. Utils flip
  applied in spec Step 2 table + plan Task 3 + Task 4.

- **Placeholders:** none. Every prompt is verbatim. Every command
  is exact. The "anchor_text" mechanism in SP-4 is the lone
  abstraction; Task 6 Step 3 spells out how the executor uses it
  (find anchor_text in source via Read, then Edit).

- **Type consistency:** finding fields (`current_snippet`,
  `proposed_fix`, `confidence`, `code_bug`, NEW: `degrading_fix`)
  used identically in writer JSON, verifier JSON, corrections file,
  and apply-fix tasks.

- **Commit discipline:** every task that produces commit-worthy
  output ends with "append to commit-plan; user runs commit". No
  `git commit` invocation anywhere in the plan.

- **Token efficiency:** gap inventory (Task 2) reuses Phase I truth
  table. Writers (Task 5) parallel-batched. Verifier (Task 7) runs
  once across all 14 modules instead of per-group. Truth table
  refresh (Task 12) is the one big new compute.

- **Bite-sized check:** longest single step is Task 6 Step 3
  (apply all writer entries). Acceptable because it's a tight loop
  over a known-shape JSON, not exploratory work.
