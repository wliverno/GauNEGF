# GauNEGF Doc Overhaul - Phase I Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete correctness audit of all in-scope GauNEGF documentation. Build an API truth table from `gauNEGF/*.py` source, dispatch 6 parallel haiku verifiers to find every code-reference mismatch, apply mechanical fixes directly, surface judgment fixes to the user for approval. Output: truth table + per-file-group fixes applied to working tree + findings/bugs files + commit plan, all on disk for user review and commit.

**Architecture:** Single sequential truth-table builder agent (haiku, ~1 run); 6 parallel verifier agents (haiku, dispatched in single batch); main session aggregates findings, applies mechanical fixes via Edit, surfaces judgment fixes via AskUserQuestion. User runs all commits; Claude never invokes `git commit` or `git add`.

**Tech Stack:** Python `ast` module (used inside subagent prompts for AST inspection), haiku-class subagents via the Agent tool, AskUserQuestion for judgment-fix approval, Edit/Write tools for fix application, the truth table at `docs/superpowers/specs/api-truth.md` as single source of API ground truth.

**Parent spec:** `docs/superpowers/specs/2026-05-09-doc-overhaul-design.md`

**Hard rules** (carried from spec, repeated here so executor cannot miss them):
- Claude does NOT run `git commit`, `git add`, `git push`, or any git mutation. User stages and commits everything. Each task that produces commit-worthy output ends with "prepare commit message for user".
- Subagents are dispatched with `model: "haiku"` per global user pref.
- Verification is static + AST. No execution of doc snippets.
- Output is ASCII only. No unicode characters in any file written by Claude.

---

## Shared procedures (referenced from multiple tasks)

### SP-1: Apply a mechanical fix via Edit tool

Used by Tasks 5, 6, 7, 8, and 9 to apply individual fixes from the verifier output. For each fix:

- Use `Edit` tool with:
  - `file_path` = absolute path under `/gscratch/anantram/willll/NEGFCode/`
  - `old_string` = the verbatim `current_snippet` value from the finding's JSON
  - `new_string` = the verbatim `proposed_fix` value from the finding's JSON
  - `replace_all` = false (always; if the verifier flagged the same snippet at multiple line ranges, treat as separate findings, each with its own Edit call)

- If `Edit` fails due to non-unique `old_string`: read the file at the finding's `line_range`, expand `old_string` with surrounding context until unique, then re-apply with the expanded form. Adjust `new_string` to include the same expanded context unchanged outside the actual edit region.

- If `Edit` fails because `old_string` is not found in the file: the finding's `current_snippet` may have been transformed by an earlier fix in the same task. Skip and add to a "skipped" list. Do not guess or pattern-match; the verifier was explicit.

- For `examples/IntegralDemo.ipynb` (notebook cells are JSON-embedded): the `Edit` tool can still apply if the `current_snippet` is exact (which the verifier ensures). If a notebook cell needs structural editing that Edit can't express cleanly, fall back to: Read the full notebook with the Read tool, use Write to rewrite with the corrected cell. Do this per-cell so the diff stays reviewable.

- For .rst files specifically: rst code-blocks have indentation that matters. The `current_snippet` from the verifier preserves indentation; copy it verbatim. The `proposed_fix` should match the same indentation. If indentation in the proposed_fix is off, normalize to 3-space block-quote indentation (sphinx default for `code-block::` directives).

- For .py docstring edits in Tasks 5-8: do NOT edit code outside docstrings. If a finding's `proposed_fix` would touch a function body or signature, that is a code change, not a doc change. Skip and re-classify as code-bug; add to `phase-1-bugs.md`.

### SP-2: Re-verify a file group after fixes

Used by Tasks 5, 6, 7, 8 after applying mechanical fixes. Dispatch a single haiku agent (subagent_type `general-purpose`, model `haiku`, run_in_background `false`) with this prompt template, filling the bracketed values:

```
Re-verify these documentation files against the API truth table:
[FILE_LIST_OF_GROUP]

Truth table: docs/superpowers/specs/api-truth.md (READ THIS FIRST)

Method: same as the original Phase I verifier prompt:
1. Read each file in full.
2. Find every Python code reference (rst code-block, markdown fence, .py docstrings, .ipynb code cells).
3. AST-equivalent inspection of each reference; resolve names against truth table; check signatures, arg names, arg order, kwargs, imports.
4. Flag any mismatch with file_path, line_range, current_snippet, issue, proposed_fix, confidence, code_bug.

Output: write JSON to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-1-findings/[OUTPUT_FILENAME] with the same schema as before:
{
  "group": "[GROUP_LABEL]-recheck",
  "files_audited": [...],
  "findings": [...],
  "summary": {"total_findings": <int>, "mechanical_count": <int>, "judgment_count": <int>, "code_bug_count": <int>}
}

If zero findings: still write the JSON with empty findings array.

Do NOT modify any source file. ASCII only in the JSON.

Goal: confirm zero mechanical findings remain. If any remain, the issue is either (a) a fix that did not land cleanly or (b) a new mismatch surfaced by an applied fix. List them clearly so the consumer can address.

Return one paragraph: files audited, total findings, breakdown by confidence.
```

Wait for return. Read the recheck JSON.

Expected: `mechanical_count == 0`. If `> 0`, surface to user as anomalies and decide per-finding whether to fix-and-rerun or accept and note.

### SP-3: Append commit-plan entry

Used by Tasks 2, 4, 5, 6, 7, 8, 9 to add to the per-task commit-message draft list. Use Edit tool to append a new section to `docs/superpowers/specs/phase-1-commit-plan.md`. If the file does not yet exist, use Write to create it with a top-level header `# Phase I Commit Plan` followed by the new section. Each section format:

```
## Commit candidate: <short descriptive title>

Files: <paths, comma or newline separated>

Suggested message:
<one-line subject>

<optional 2-4 line body>
```

Do not write commit hashes, dates, or "applied at <timestamp>" -- those are git-side and the user adds them. Just the message draft.

---

## File Structure (what gets created in this phase)

| Path | Created in | Purpose |
|------|------------|---------|
| `docs/superpowers/specs/api-truth.md` | Task 2 | Versioned API truth table; reused in Phase II/III. |
| `docs/superpowers/specs/phase-1-findings/group-A.json` | Task 3 | Verifier output for README + examples. |
| `docs/superpowers/specs/phase-1-findings/group-B.json` | Task 3 | Verifier output for sphinx top + theory. |
| `docs/superpowers/specs/phase-1-findings/group-C.json` | Task 3 | Verifier output for sphinx examples + API index. |
| `docs/superpowers/specs/phase-1-findings/group-D1.json` | Task 3 | Verifier output for core docstrings. |
| `docs/superpowers/specs/phase-1-findings/group-D2.json` | Task 3 | Verifier output for surface-Greens docstrings. |
| `docs/superpowers/specs/phase-1-findings/group-D3.json` | Task 3 | Verifier output for utility docstrings. |
| `docs/superpowers/specs/phase-1-findings.md` | Task 4 | Human-readable consolidation. |
| `docs/superpowers/specs/phase-1-bugs.md` | Task 4 (conditional) | Code-bug findings for user triage. |
| `docs/superpowers/specs/phase-1-judgment-fixes.md` | Task 9 | Judgment fixes the user approved/rejected. |
| `docs/superpowers/specs/phase-1-commit-plan.md` | Task 10 | Draft commit messages for the 4 file groups + artifacts. |

Files MODIFIED in this phase (in working tree, not committed by Claude):
- `README.md`
- `examples/SiNEGF.py`
- `examples/IntegralDemo.ipynb`
- `docs/source/quickstart.rst`
- `docs/source/installation.rst`
- `docs/source/theory/{introduction,negf_dft,transport,best_practices}.rst`
- `docs/source/examples/{ethane,silicon_nanowire,advanced_examples,index}.rst`
- `docs/source/api/index.rst`
- `gauNEGF/{scf,scfE,density,transport,surfG1D,surfG3D,surfGBethe,surfGTester,matTools,integrate,spinTools,utils,protocols,config,fermiSearch}.py` (docstring-only edits)

---

## Task 1: Verify prerequisites

**Files:** none modified; read-only checks.

- [ ] **Step 1: Confirm spec file exists and is the approved version**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/2026-05-09-doc-overhaul-design.md
```
Expected: file exists, non-empty.
If missing: STOP. Do not proceed; flag to user.

- [ ] **Step 2: Confirm gauNEGF source modules are intact**

Run:
```bash
ls /gscratch/anantram/willll/NEGFCode/gauNEGF/*.py
```
Expected: 14 files at minimum: `__init__.py`, `config.py`, `density.py`, `fermiSearch.py`, `integrate.py`, `matTools.py`, `protocols.py`, `scf.py`, `scfE.py`, `spinTools.py`, `surfG1D.py`, `surfG3D.py`, `surfGBethe.py`, `surfGTester.py`, `transport.py`, `utils.py`.
If any missing: STOP. The spec was written against this set; any missing file means the project drifted and the spec needs revision.

- [ ] **Step 3: Confirm in-scope doc files exist**

Run:
```bash
for f in README.md docs/source/quickstart.rst docs/source/installation.rst \
  docs/source/theory/introduction.rst docs/source/theory/negf_dft.rst \
  docs/source/theory/transport.rst docs/source/theory/best_practices.rst \
  docs/source/examples/ethane.rst docs/source/examples/silicon_nanowire.rst \
  docs/source/examples/advanced_examples.rst docs/source/examples/index.rst \
  docs/source/api/index.rst examples/SiNEGF.py examples/IntegralDemo.ipynb; do
  if [ ! -f "/gscratch/anantram/willll/NEGFCode/$f" ]; then
    echo "MISSING: $f"
  fi
done
```
Expected: no MISSING lines. If any missing: STOP and surface to user.

- [ ] **Step 4: Create findings subdirectory**

Run:
```bash
mkdir -p /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-1-findings
```
Expected: directory created or already exists.

- [ ] **Step 5: Confirm clean git state on `testing` branch (or current branch)**

Run:
```bash
git -C /gscratch/anantram/willll/NEGFCode branch --show-current
```
Expected: prints current branch (likely `testing` per session start).

DO NOT run any git mutation. This is a read check only.

---

## Task 2: Build the API truth table

**Files:**
- Create: `docs/superpowers/specs/api-truth.md`

- [ ] **Step 1: Dispatch the truth-table builder agent**

Use the Agent tool with `subagent_type: "general-purpose"` and `model: "haiku"`. Prompt verbatim:

```
You are extracting a complete API reference for the gauNEGF Python package.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs to read: every file matching gauNEGF/*.py (use the Read tool; do NOT execute any code).

Method:
1. For each .py file in gauNEGF/, read the file in full.
2. Mentally parse it as Python (you may use ast logic, but you do not run python). For every top-level class and function whose name does NOT start with an underscore, extract:
   - Symbol name
   - Full signature (positional args with defaults, kwargs, *args, **kwargs, type annotations if present)
   - First line of the docstring (or the literal token "absent" if there is no docstring, or "stub" if the docstring is < 20 chars or just a placeholder)
   - Decorators (especially @jit, @jax.jit, @deprecated, @staticmethod, @classmethod, @property)
3. For every class, also extract public methods (same rules: name does NOT start with underscore, except __init__ which IS included).
4. Detect deprecation markers: look for the literal strings "[DEPRECATED]", "deprecated", or "DEPRECATED" in the module docstring or in the function/class docstring. Set deprecated=true if any match is found in the docstring of that symbol.
5. Note module-level docstring presence: "present" / "absent" / "stub".

Output: write the complete result to /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md using the Write tool.

File format (exact structure required):

# GauNEGF API Truth Table

Generated: <today's date>
Source: gauNEGF/*.py
Method: AST-equivalent inspection by haiku agent

## Per-Module Detail

### Module: gauNEGF.<modulename>

- **Module docstring:** present | absent | stub
- **Module-level deprecation marker:** none | "<exact text matched>"
- **Public classes:** N
- **Public functions:** N

#### class ClassName

Signature: `ClassName(arg1, arg2=default, *args, **kwargs)`
Docstring: present | absent | stub
Deprecated: true | false
Decorators: [list or empty]

Methods:
- `__init__(self, ...)` -- docstring: present|absent|stub
- `methodName(self, arg, ...)` -- docstring: present|absent|stub
- ...

#### function functionName

Signature: `functionName(arg1, arg2=default, ...)`
Docstring: present | absent | stub
Deprecated: true | false
Decorators: [list or empty]

(Repeat per module.)

## Flat Lookup Table

| Symbol | Module | Type | Signature | Docstring | Deprecated |
|--------|--------|------|-----------|-----------|-----------|
| NEGF | gauNEGF.scf | class | (fn, func='b3lyp', ...) | present | false |
| NEGF.setContacts | gauNEGF.scf | method | (self, lContact=None, rContact=None) | present | false |
| ...

Be exhaustive. Every public symbol must appear in both the per-module section and the flat table.

Do NOT write code. Do NOT run code. Use Read tool only for inputs and Write tool for the output. ASCII only -- no unicode in the output file.

When done, return a one-paragraph summary: total modules, total classes, total functions, total deprecated symbols, total absent docstrings.
```

Run as: single Agent tool invocation. Background = false (we wait for the result).

Expected runtime: 5-10 minutes for haiku across 14 modules.

- [ ] **Step 2: Validate truth table output**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
wc -l /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md
```
Expected: file exists, line count > 200 (rough sanity floor; 14 modules with multiple symbols each must be substantial).

If line count < 100: the agent likely failed or produced a stub. Re-dispatch with the same prompt.

- [ ] **Step 3: Spot-check three known signatures**

Read the truth table and verify these three known-good symbols appear correctly:

1. `NEGF.setContacts` should show signature `(self, lContact=None, rContact=None)` (matches `gauNEGF/scf.py:424`).
2. `NEGFE.setContact1D` should show 11 args ending in `..., eta=ETA, T=TEMPERATURE, symmetrize_contacts=None)` (matches `gauNEGF/scfE.py:96`).
3. `calculate_current` (in transport.py) should show signature with `sigma_calculator` parameter, NOT positional `sig1, sig2` (matches `gauNEGF/transport.py:636`).

If any of the three are wrong: re-dispatch the agent with a corrective prompt naming the discrepancy.

- [ ] **Step 4: Prepare commit message draft (do not commit)**

Append to `docs/superpowers/specs/phase-1-commit-plan.md` (create if needed) the following draft:

```
## Commit candidate: API truth table

Files: docs/superpowers/specs/api-truth.md

Suggested message:
docs(spec): add API truth table for Phase I correctness audit

Generated by haiku agent via AST-equivalent inspection of gauNEGF/*.py.
Single source of API ground truth for Phase I verifiers, Phase II
docstring writers, and Phase III guide writers. Versioned for traceability.
```

- [ ] **Step 5: Hand off**

Tell the user: "Truth table written to `docs/superpowers/specs/api-truth.md`. You can review it now or wait until the end of Phase I to review the full bundle. Either way, no commits yet."

---

## Task 3: Dispatch 6 parallel verifiers

**Files:**
- Create: `docs/superpowers/specs/phase-1-findings/group-{A,B,C,D1,D2,D3}.json`

- [ ] **Step 1: Read the truth table into context**

Use Read tool on `docs/superpowers/specs/api-truth.md`.

(Required so that if a verifier fails, we can diagnose what it should have seen.)

- [ ] **Step 2: Dispatch all 6 verifier agents in a single message**

Use the Agent tool 6 times in ONE assistant message (parallel execution). All 6 use `subagent_type: "general-purpose"`, `model: "haiku"`, `run_in_background: false`.

The prompt template below is shared. The bracketed `[GROUP_X_FILES]` and `[GROUP_X_LABEL]` and `[OUTPUT_PATH]` differ per dispatch; everything else is identical.

**Shared prompt template:**

```
You are verifying that documentation code references match a known API.

Working directory: /gscratch/anantram/willll/NEGFCode/

Inputs:
1. API truth table: docs/superpowers/specs/api-truth.md (READ THIS FIRST)
2. Files to verify: [GROUP_X_FILES]

Method, applied to each file in your list:
1. Read the file in full.
2. Find every Python code reference. These appear as:
   - rst code-block:: python directives
   - markdown ```python fences
   - .py files (every line)
   - .ipynb code cells (parse JSON; each cell with cell_type=code has source list of lines)
   - Code snippets inside docstrings (look for >>> doctest-style or :code:: directives)
3. For each code reference, treat the contents as a Python AST root.
4. Walk the AST. For every Name, Attribute, Call, ImportFrom, Import:
   - Resolve the symbol against the truth table.
   - Check: does the symbol exist? If a Call, do the arg names match the signature? Is arg order valid? Are kwargs valid for that signature? If the symbol is marked deprecated in the truth table, flag this as a judgment finding.
5. For prose text adjacent to code blocks, also check: do referenced symbols (names mentioned in the text) match the symbols actually used in the code block?

For EVERY mismatch you find, produce a finding with these fields:
- file_path: relative to working dir, e.g. "README.md"
- line_range: [start_line, end_line] (1-indexed; for ipynb use [cell_index, line_in_cell])
- current_snippet: VERBATIM copy of the offending text including surrounding code-block boundaries (so the consumer can match it exactly for replacement)
- issue: one short sentence describing what is wrong
- proposed_fix: VERBATIM corrected snippet
- confidence: "mechanical" if the fix is a name/arg substitution that can be applied without judgment; "judgment" if it requires choosing between equivalent APIs or rewriting against a different abstraction
- code_bug: true ONLY if the issue looks like a bug in gauNEGF source (a function the docs reference correctly that the code itself implements wrongly), not a doc bug; false otherwise

ALSO: if you encounter a docstring (in a .py file) that references parameters not in its function's signature, flag as mechanical fix.

ALSO: if you encounter a doc that references a symbol the truth table does NOT contain (e.g., a function name that does not exist anywhere in gauNEGF), flag as mechanical with proposed_fix indicating the closest matching real symbol if there is an obvious one, or "remove or replace -- symbol does not exist" if not.

Output: write JSON to [OUTPUT_PATH] using the Write tool.

JSON schema (exact):
{
  "group": "[GROUP_X_LABEL]",
  "files_audited": ["path1", "path2", ...],
  "findings": [
    {
      "file_path": "...",
      "line_range": [<start>, <end>],
      "current_snippet": "...",
      "issue": "...",
      "proposed_fix": "...",
      "confidence": "mechanical",
      "code_bug": false
    }
  ],
  "summary": {
    "total_findings": <int>,
    "mechanical_count": <int>,
    "judgment_count": <int>,
    "code_bug_count": <int>
  }
}

If there are zero findings: still write the JSON with empty findings array and summary.total_findings = 0.

Do NOT modify any source file (no edits to README, docs/, gauNEGF/, examples/). Use Write tool ONLY to create the output JSON. Use Read tool for inputs.

ASCII only in the JSON output. No unicode characters.

When done, return a one-paragraph summary: files audited, total findings, breakdown by confidence, any "code_bug" findings highlighted.
```

**Per-dispatch parameter table:**

| Dispatch | [GROUP_X_LABEL] | [GROUP_X_FILES] | [OUTPUT_PATH] |
|----------|-----------------|-----------------|---------------|
| A | A | README.md, examples/SiNEGF.py, examples/IntegralDemo.ipynb | docs/superpowers/specs/phase-1-findings/group-A.json |
| B | B | docs/source/quickstart.rst, docs/source/installation.rst, docs/source/theory/introduction.rst, docs/source/theory/negf_dft.rst, docs/source/theory/transport.rst, docs/source/theory/best_practices.rst | docs/superpowers/specs/phase-1-findings/group-B.json |
| C | C | docs/source/examples/ethane.rst, docs/source/examples/silicon_nanowire.rst, docs/source/examples/advanced_examples.rst, docs/source/examples/index.rst, docs/source/api/index.rst | docs/superpowers/specs/phase-1-findings/group-C.json |
| D1 | D1 | gauNEGF/scf.py, gauNEGF/scfE.py, gauNEGF/density.py, gauNEGF/transport.py | docs/superpowers/specs/phase-1-findings/group-D1.json |
| D2 | D2 | gauNEGF/surfG1D.py, gauNEGF/surfG3D.py, gauNEGF/surfGBethe.py, gauNEGF/surfGTester.py | docs/superpowers/specs/phase-1-findings/group-D2.json |
| D3 | D3 | gauNEGF/matTools.py, gauNEGF/integrate.py, gauNEGF/spinTools.py, gauNEGF/utils.py, gauNEGF/protocols.py, gauNEGF/config.py, gauNEGF/fermiSearch.py | docs/superpowers/specs/phase-1-findings/group-D3.json |

**Important: Group C verifier additionally produces "missing modules" output.** Append this paragraph to Group C's prompt:

```
ADDITIONAL CHECK FOR GROUP C ONLY:
After verifying docs/source/api/index.rst, also produce a "missing_modules" sub-section in your output JSON:

"missing_modules": [
  {"module": "gauNEGF.surfG3D", "reason": "not registered in api/index.rst", "size_lines": <int>, "deprecated": false}
]

Compare the modules referenced in api/index.rst against the truth table's module list. Every truth-table module not in api/index.rst goes in missing_modules. Note deprecated status from the truth table.
```

(For executor: this means Group C's prompt is the shared template + this extra block. Other 5 groups use the shared template only.)

- [ ] **Step 3: Wait for all 6 to return**

Since `run_in_background: false`, the parallel dispatch blocks until all complete. No polling needed.

If any one fails (returns an error rather than completing): note which one, dispatch a single replacement agent with the same prompt. Do not abandon the other 5's results.

- [ ] **Step 4: Validate each output JSON parses**

Run:
```bash
for g in A B C D1 D2 D3; do
  python3 -c "import json; d=json.load(open('/gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-1-findings/group-${g}.json')); print(f'{g}: {d[\"summary\"][\"total_findings\"]} findings ({d[\"summary\"][\"mechanical_count\"]} mech, {d[\"summary\"][\"judgment_count\"]} judg, {d[\"summary\"][\"code_bug_count\"]} bug)')"
done
```
Expected: 6 lines, one per group, each parsing as valid JSON. NB: this only RUNS the python interpreter to parse JSON; it does not execute gauNEGF code. Safe on a login node.

If any group's JSON does not parse: re-dispatch that one verifier with the prompt addendum "Your previous output was not valid JSON. Re-emit with strict JSON syntax."

- [ ] **Step 5: Sanity check Group C's missing_modules list**

Read `group-C.json`. Confirm `missing_modules` exists and contains entries for at least: `gauNEGF.surfG3D`, `gauNEGF.spinTools`, `gauNEGF.config`, `gauNEGF.protocols`, `gauNEGF.utils`, `gauNEGF.fermiSearch` (these are known-missing per the spec's exploration findings).

If the list is empty or missing these: re-dispatch Group C's verifier.

- [ ] **Step 6: No commit (intermediate artifacts; will commit with consolidated findings)**

These JSONs are not committed individually. They roll up into `phase-1-findings.md` in Task 4.

---

## Task 4: Aggregate findings into human-readable consolidation

**Files:**
- Create: `docs/superpowers/specs/phase-1-findings.md`
- Create: `docs/superpowers/specs/phase-1-bugs.md` (only if any code_bug findings)

- [ ] **Step 1: Read all 6 group JSONs**

Use Read tool on each of:
- `docs/superpowers/specs/phase-1-findings/group-A.json`
- `docs/superpowers/specs/phase-1-findings/group-B.json`
- `docs/superpowers/specs/phase-1-findings/group-C.json`
- `docs/superpowers/specs/phase-1-findings/group-D1.json`
- `docs/superpowers/specs/phase-1-findings/group-D2.json`
- `docs/superpowers/specs/phase-1-findings/group-D3.json`

- [ ] **Step 2: Categorize findings**

Mentally split every finding into one of three buckets:
- **Mechanical**: confidence == "mechanical" AND code_bug == false
- **Judgment**: confidence == "judgment" AND code_bug == false
- **Code bug**: code_bug == true (regardless of confidence)

Count totals per bucket per group.

- [ ] **Step 3: Write `phase-1-findings.md`**

Use Write tool. Format:

```markdown
# Phase I Findings - GauNEGF Documentation Audit

Date: <today>
Source: 6 parallel haiku verifiers + truth table at api-truth.md

## Summary

| Group | Files | Mechanical | Judgment | Code Bug | Total |
|-------|-------|------------|----------|----------|-------|
| A     | 3     | <int>      | <int>    | <int>    | <int> |
| B     | 6     | ...        | ...      | ...      | ...   |
| ...   | ...   | ...        | ...      | ...      | ...   |
| **Total** | 26 | ... | ... | ... | ... |

## Group C: Missing modules from api/index.rst

(From Group C's missing_modules sub-section.)

| Module | Size (lines) | Deprecated | Phase II default decision |
|--------|--------------|------------|---------------------------|
| gauNEGF.surfG3D | <int> | false | Include in main API |
| ... | ... | ... | ... |

## Mechanical Fixes (will be applied directly by main session)

### Group A

#### Fix A.1: README.md:88
**Issue:** <issue>
**Current:**
```
<current_snippet verbatim>
```
**Proposed:**
```
<proposed_fix verbatim>
```

#### Fix A.2: ...

(Repeat per finding, organized by group.)

## Judgment Fixes (require user approval)

### Group A

#### Judgment A.1: README.md:99-101
**Issue:** <issue>
**Current:**
```
<current_snippet verbatim>
```
**Proposed:**
```
<proposed_fix verbatim>
```
**Why this needs judgment:** <one sentence -- e.g., "Two equivalent APIs exist; this picks calculate_current over the legacy current(). User confirms direction.">

#### Judgment A.2: ...

(Repeat per finding.)
```

NB: every "current_snippet" and "proposed_fix" is copied verbatim from the JSON. Do not paraphrase, edit, or reformat. The mechanical fix step in later tasks matches these strings literally for `Edit` calls.

- [ ] **Step 4: If any code bugs, write `phase-1-bugs.md`**

Only if any group's `code_bug_count > 0`. Use Write tool. Format:

```markdown
# Phase I Code Bug Findings (NOT FIXED -- for user triage)

Date: <today>
These are findings the verifiers flagged as `code_bug: true`. They appear to be bugs in `gauNEGF/*.py` source code, not in the docs. Per Phase I non-goals, code bugs are NOT fixed silently. Listed here for the user to triage and address separately.

## Bug 1: <file>:<line>

**Issue:** <issue>
**Current code:**
```python
<current_snippet>
```
**What the docs/tests assume:** <proposed_fix>
**Verifier reasoning:** <if available from issue text>

(Repeat per code-bug finding.)
```

If no code bugs: skip this step. Note in `phase-1-findings.md` summary: "No code bugs flagged."

- [ ] **Step 5: Append commit-plan entry**

Append to `docs/superpowers/specs/phase-1-commit-plan.md`:

```
## Commit candidate: Phase I findings + bugs

Files: docs/superpowers/specs/phase-1-findings.md
       docs/superpowers/specs/phase-1-bugs.md  (if exists)
       docs/superpowers/specs/phase-1-findings/group-A.json
       docs/superpowers/specs/phase-1-findings/group-B.json
       docs/superpowers/specs/phase-1-findings/group-C.json
       docs/superpowers/specs/phase-1-findings/group-D1.json
       docs/superpowers/specs/phase-1-findings/group-D2.json
       docs/superpowers/specs/phase-1-findings/group-D3.json

Suggested message:
docs(spec): record Phase I doc audit findings

Six parallel haiku verifiers compared every code reference in the
in-scope docs against the API truth table. Findings split into
mechanical (applied directly) and judgment (user-approved).
Includes raw per-group JSON for traceability and a phase-1-bugs.md
inventory of incidental code-bug findings for separate triage.
```

- [ ] **Step 6: Hand off**

Tell user: "Findings consolidated. <N> mechanical, <M> judgment, <K> code-bug. Mechanical fixes will be applied automatically next; judgment fixes will be surfaced for your approval after."

---

## Task 5: Apply mechanical fixes to Group A files

**Files:**
- Modify: `README.md`
- Modify: `examples/SiNEGF.py`
- Modify: `examples/IntegralDemo.ipynb`

- [ ] **Step 1: Read the mechanical-fix list for Group A from `phase-1-findings.md`**

- [ ] **Step 2: For each Group A mechanical fix, apply via Edit**

Follow shared procedure SP-1 for each finding listed in `phase-1-findings.md` under Group A's mechanical fixes section. The `IntegralDemo.ipynb` notes in SP-1 apply specifically to one of this group's files.

- [ ] **Step 3: Re-verify Group A files**

Follow shared procedure SP-2 with:
- `[FILE_LIST_OF_GROUP]` = `README.md`, `examples/SiNEGF.py`, `examples/IntegralDemo.ipynb`
- `[GROUP_LABEL]` = `A`
- `[OUTPUT_FILENAME]` = `group-A-recheck.json`

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

Append to `phase-1-commit-plan.md`:

```
## Commit candidate: Group A doc fixes (README + examples)

Files: README.md, examples/SiNEGF.py, examples/IntegralDemo.ipynb

Suggested message:
docs: fix code references in README and examples to match current API

- README quickstart now uses calculate_current with SigmaCalculator
  instead of legacy positional transport.current()
- examples/SiNEGF.py code paths verified against gauNEGF source
- examples/IntegralDemo.ipynb code cells verified

Mechanical fixes only; judgment fixes (if any) in separate commits.
```

- [ ] **Step 5: Hand off**

Tell user: "Group A mechanical fixes applied. <N> edits across README + examples. Re-verified clean. Stage and commit when ready."

---

## Task 6: Apply mechanical fixes to Group B files

**Files:**
- Modify: `docs/source/quickstart.rst`
- Modify: `docs/source/installation.rst`
- Modify: `docs/source/theory/introduction.rst`
- Modify: `docs/source/theory/negf_dft.rst`
- Modify: `docs/source/theory/transport.rst`
- Modify: `docs/source/theory/best_practices.rst`

- [ ] **Step 1: Read the mechanical-fix list for Group B from `phase-1-findings.md`**

- [ ] **Step 2: For each Group B mechanical fix, apply via Edit**

Follow shared procedure SP-1 for each finding listed under Group B. All Group B files are .rst; the rst-indentation note in SP-1 applies.

- [ ] **Step 3: Re-verify Group B files**

Follow shared procedure SP-2 with:
- `[FILE_LIST_OF_GROUP]` = `docs/source/quickstart.rst`, `docs/source/installation.rst`, `docs/source/theory/introduction.rst`, `docs/source/theory/negf_dft.rst`, `docs/source/theory/transport.rst`, `docs/source/theory/best_practices.rst`
- `[GROUP_LABEL]` = `B`
- `[OUTPUT_FILENAME]` = `group-B-recheck.json`

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

```
## Commit candidate: Group B doc fixes (sphinx top + theory)

Files: docs/source/quickstart.rst, docs/source/installation.rst,
       docs/source/theory/{introduction,negf_dft,transport,best_practices}.rst

Suggested message:
docs(sphinx): fix code references in top-level + theory pages

- best_practices.rst: setContact1D and setContactBethe call shapes
  corrected to match real signatures
- transport.rst: Hartree/eV unit consistency in spin-resolved example
- quickstart.rst: import paths and snippet completeness verified
- All theory examples re-checked against API truth table

Mechanical fixes only.
```

- [ ] **Step 5: Hand off**

Tell user: "Group B mechanical fixes applied. <N> edits across 6 sphinx files. Re-verified clean."

---

## Task 7: Apply mechanical fixes to Group C files

**Files:**
- Modify: `docs/source/examples/ethane.rst`
- Modify: `docs/source/examples/silicon_nanowire.rst`
- Modify: `docs/source/examples/advanced_examples.rst`
- Modify: `docs/source/examples/index.rst`
- Modify: `docs/source/api/index.rst`

- [ ] **Step 1: Read the mechanical-fix list for Group C from `phase-1-findings.md`**

Note: Group C also has the `missing_modules` sub-section. That data is preserved for Phase II (not applied here). Phase I touches `api/index.rst` only for mechanical fixes (typos, malformed automodule directives, dead cross-refs). New module ENTRIES are a Phase II decision.

- [ ] **Step 2: For each Group C mechanical fix, apply via Edit**

Follow shared procedure SP-1 for each finding listed under Group C.

- [ ] **Step 3: Re-verify Group C files**

Follow shared procedure SP-2 with:
- `[FILE_LIST_OF_GROUP]` = `docs/source/examples/ethane.rst`, `docs/source/examples/silicon_nanowire.rst`, `docs/source/examples/advanced_examples.rst`, `docs/source/examples/index.rst`, `docs/source/api/index.rst`
- `[GROUP_LABEL]` = `C`
- `[OUTPUT_FILENAME]` = `group-C-recheck.json`

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

```
## Commit candidate: Group C doc fixes (sphinx examples + API index)

Files: docs/source/examples/{ethane,silicon_nanowire,advanced_examples,index}.rst,
       docs/source/api/index.rst

Suggested message:
docs(sphinx): fix code references in examples + API index

- examples/*.rst snippets verified against API truth table
- api/index.rst automodule directives validated
  (note: missing modules tracked for Phase II, not added in this commit)

Mechanical fixes only.
```

- [ ] **Step 5: Hand off**

Tell user: "Group C mechanical fixes applied. Missing-module list preserved in `group-C.json` for Phase II."

---

## Task 8: Apply mechanical fixes to Groups D1, D2, D3 (docstrings)

**Files:**
- Modify: `gauNEGF/scf.py`, `gauNEGF/scfE.py`, `gauNEGF/density.py`, `gauNEGF/transport.py` (D1)
- Modify: `gauNEGF/surfG1D.py`, `gauNEGF/surfG3D.py`, `gauNEGF/surfGBethe.py`, `gauNEGF/surfGTester.py` (D2)
- Modify: `gauNEGF/matTools.py`, `gauNEGF/integrate.py`, `gauNEGF/spinTools.py`, `gauNEGF/utils.py`, `gauNEGF/protocols.py`, `gauNEGF/config.py`, `gauNEGF/fermiSearch.py` (D3)

This is one task because all three groups touch only docstrings (no logic edits) and the same procedure applies. We commit them as a single bundle for review.

- [ ] **Step 1: Read mechanical-fix lists for D1, D2, D3 from `phase-1-findings.md`**

- [ ] **Step 2: For each D-group mechanical fix, apply via Edit**

Follow shared procedure SP-1 for each finding listed under D1, D2, and D3. The "do not edit code outside docstrings" note in SP-1 is the load-bearing constraint here -- every edit must land inside a `"""..."""` block.

- [ ] **Step 3: Re-verify Groups D1+D2+D3 in a single agent run**

Follow shared procedure SP-2 with:
- `[FILE_LIST_OF_GROUP]` = `gauNEGF/scf.py`, `gauNEGF/scfE.py`, `gauNEGF/density.py`, `gauNEGF/transport.py`, `gauNEGF/surfG1D.py`, `gauNEGF/surfG3D.py`, `gauNEGF/surfGBethe.py`, `gauNEGF/surfGTester.py`, `gauNEGF/matTools.py`, `gauNEGF/integrate.py`, `gauNEGF/spinTools.py`, `gauNEGF/utils.py`, `gauNEGF/protocols.py`, `gauNEGF/config.py`, `gauNEGF/fermiSearch.py`
- `[GROUP_LABEL]` = `D`
- `[OUTPUT_FILENAME]` = `group-D-recheck.json`

- [ ] **Step 4: Append commit-plan entry (use SP-3)**

```
## Commit candidate: docstring fixes across gauNEGF/

Files: gauNEGF/{scf,scfE,density,transport,surfG1D,surfG3D,surfGBethe,
       surfGTester,matTools,integrate,spinTools,utils,protocols,config,
       fermiSearch}.py

Suggested message:
docs(gauNEGF): fix docstring references to match function signatures

Docstring-only edits across 14 modules. Parameter names, type tags,
formula symbols, and cross-references brought into agreement with the
API truth table. No logic changes.

Modules touched: <list of modules with non-zero findings>
```

- [ ] **Step 5: Hand off**

Tell user: "Docstring mechanical fixes applied across <N> modules. <M> total edits. Re-verified clean."

---

## Task 9: Surface judgment fixes to user

**Files:**
- Create: `docs/superpowers/specs/phase-1-judgment-fixes.md`

- [ ] **Step 1: Read the judgment-fix list from `phase-1-findings.md`**

If zero judgment fixes: skip Task 9 entirely. Tell user "No judgment fixes; proceeding to Task 10."

- [ ] **Step 2: Group judgment fixes by file group for batched review**

Group fixes by which file group they're in (A/B/C/D1/D2/D3). User reviews one group at a time.

- [ ] **Step 3: For each group with judgment fixes, present via AskUserQuestion**

Per group, present up to 4 fixes per `AskUserQuestion` call (the tool's max). If a group has more than 4 judgment fixes, paginate.

For each fix, the question is:

```
question: "Approve this judgment fix in <file>:<line>? <issue>"
header: "Group X fix N"
multiSelect: false
options: [
  { "label": "Approve", "description": "Apply the proposed fix as shown." },
  { "label": "Approve with edit", "description": "Apply, but I'll edit the proposed text first." },
  { "label": "Reject", "description": "Do not apply this fix; leave the doc as-is." },
  { "label": "Defer", "description": "Add to a follow-up list; do not apply now, do not block the phase." }
]
```

For "Approve with edit": after the user answers, ask follow-up "What's the corrected text?" via free-text. Apply user's text instead of `proposed_fix`.

- [ ] **Step 4: Apply approved judgment fixes**

For each fix marked "Approve" or "Approve with edit": apply via shared procedure SP-1 (same Edit-based application as mechanical fixes; for "Approve with edit" the user-supplied text replaces the proposed_fix value).

For each fix marked "Defer": record in a deferred list.

For each fix marked "Reject": record in a rejected list with reason if user gave one.

- [ ] **Step 5: Write `phase-1-judgment-fixes.md`**

Use Write tool. Format:

```markdown
# Phase I Judgment Fixes - User Decisions

Date: <today>

## Approved and applied (<N>)

### <file>:<line> — Group X
**Issue:** <issue>
**Applied fix:**
```
<final text applied>
```

## Approved with edit (<N>)

(Same format; "Applied fix" shows user's edited version.)

## Deferred (<N>)

### <file>:<line> — Group X
**Issue:** <issue>
**Original proposal:**
```
<proposed_fix>
```
**Reason for deferral:** <user note or "no reason given">

## Rejected (<N>)

(Same format; "Reason for rejection".)
```

- [ ] **Step 6: Append commit-plan entry**

Add a single entry to `phase-1-commit-plan.md` covering all approved-and-applied judgment fixes, grouped by file group:

```
## Commit candidates: Group <X> judgment fixes

Files: <list per file group>

Suggested message:
docs(<scope>): apply user-approved judgment fixes

These are fixes flagged as requiring judgment by the Phase I verifiers
(typically choosing between equivalent APIs or rewriting against the
canonical abstraction). User reviewed and approved each one; rejected
and deferred items recorded in phase-1-judgment-fixes.md.
```

- [ ] **Step 7: Hand off**

Tell user: "Judgment fixes resolved. <N> approved and applied, <M> deferred, <K> rejected. Decisions logged in `phase-1-judgment-fixes.md`."

---

## Task 10: Final commit-plan + Phase I exit

**Files:**
- Update: `docs/superpowers/specs/phase-1-commit-plan.md`

- [ ] **Step 1: Verify all Phase I artifacts exist**

Run:
```bash
ls -la /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/api-truth.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-1-findings.md \
       /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-1-commit-plan.md
ls /gscratch/anantram/willll/NEGFCode/docs/superpowers/specs/phase-1-findings/
```
Expected: all primary artifacts present; `phase-1-findings/` contains 6 group JSONs plus 4 -recheck JSONs (one per task that re-verified). `phase-1-bugs.md` and `phase-1-judgment-fixes.md` present only if applicable (if there were code bugs / judgment fixes).

- [ ] **Step 2: Verify the recommended commit ordering in `phase-1-commit-plan.md`**

Read `phase-1-commit-plan.md`. Expected commit order (top to bottom in the file, suggesting you commit them in this order):

1. API truth table (one commit)
2. Phase I findings + bugs + JSONs (one commit)
3. Group A doc fixes (one commit)
4. Group B doc fixes (one commit)
5. Group C doc fixes (one commit)
6. Docstring fixes (one commit covering D1+D2+D3)
7. Judgment fixes per file group (zero to several commits, one per file group with approvals)

If the ordering in the file is wrong: rewrite to match the above.

- [ ] **Step 3: Print the final summary for the user**

Output to chat:

```
=== Phase I Complete (pending your commits) ===

Truth table: docs/superpowers/specs/api-truth.md
Findings:    docs/superpowers/specs/phase-1-findings.md  (<N> total)
              - Mechanical applied: <N>
              - Judgment approved: <N>
              - Judgment deferred: <N>
              - Judgment rejected: <N>
              - Code bugs logged for triage: <N>
Commit plan: docs/superpowers/specs/phase-1-commit-plan.md  (<N> commits suggested)

What you do next:
1. Review diffs in your working tree (git diff per file group).
2. Run commits in the order suggested in phase-1-commit-plan.md.
3. When all Phase I commits are landed, signal me to write the Phase II plan.

Reminder: I have not run any git command and will not. Every commit is yours.
```

- [ ] **Step 4: Phase I exit**

When user confirms commits are landed (or chooses to defer some commits), Phase I is complete.

Note for the executor: do NOT proceed to write Phase II plan automatically. The spec requires user sign-off as the phase exit gate. Wait for explicit user instruction to write Phase II.

---

## Self-Review (executor: confirm before starting)

Before launching Task 1, verify:

- [ ] Spec at `docs/superpowers/specs/2026-05-09-doc-overhaul-design.md` is the version dated 2026-05-09 with revisions per haiku review (TODO ceiling, interview cap, verifier feedback workflow, sphinx dry-run).
- [ ] You understand: Claude does NOT commit; user does.
- [ ] You understand: subagents are haiku.
- [ ] You understand: every fix is verbatim string replacement against the verifier's `current_snippet` field; no paraphrasing.
- [ ] You have access to AskUserQuestion (used in Task 9).

If any of these is uncertain, stop and ask the user before dispatching the truth-table builder.

---

## Plan-level self-review (drafter: completed during writing)

- **Spec coverage:** Phase I steps 1-4 from spec map to plan Tasks 2-9. Phase I exit criteria (truth table committed, mechanical fixes applied, judgment fixes resolved or deferred, findings committed) all addressed.
- **Placeholders:** none. Every prompt is verbatim. Every command is exact.
- **Type consistency:** finding fields (`current_snippet`, `proposed_fix`, `confidence`, `code_bug`) used identically in the verifier prompt, the JSON schema, the consolidation, and the apply-fix tasks.
- **Commit discipline:** every task that produces commit-worthy output ends with "append to commit-plan; user runs commit". No `git commit` invocation anywhere in the plan.
- **Token efficiency:** truth table built once (Task 2); all 6 verifiers reference it (Task 3); re-verifications target only their changed file group (Tasks 5-8); no agent re-derives signature info from gauNEGF/*.py.
