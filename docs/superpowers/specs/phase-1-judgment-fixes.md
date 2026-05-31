# Phase I Judgment Fixes - User Decisions

Date: 2026-05-09

This file records every judgment-level finding from Phase I and the
user's decision on each. Many findings turned out to be verifier
over-interpretation rather than real bugs; those are marked
"REJECTED -- not actually broken" with the user's clarification.

---

## Approved and applied

### scf.py NEGF.setFock + class docstring (D1.1, originally code-bug)

**Issue:** Doc/code/class-docstring inconsistency about whether F_ input is
eV or Hartree, and whether self.F is stored as eV or Hartree.
**User decision:** "Input is eV, self.F is in Hartree internally."
**Applied fixes:**
- scf.py NEGF class docstring (line 114): `F : Fock matrix in eV` ->
  `F : Fock matrix in Hartree (multiply by 27.211386 for eV)`
- scf.py NEGF.setFock summary: `converting from Hartree to eV` ->
  `converting from eV to Hartree`
- Param doc `F_ : Fock matrix in eV units` left unchanged (was correct).

### scf.py NEGF.setDen spinLockList (D1.J1)

**Issue:** Docstring text cut off mid-word: "List of atoms to apply sp"
**User decision:** Apply verifier's guess.
**Applied fix:** "List of atoms to apply spin-locking to"

### transport.py currentSpin spin description (D1.J2)

**Issue:** Spin parameter description says 'r' for restricted on a
spin-dependent function -- paradoxical wording.
**User decision:** Apply.
**Applied fix:** Change to "Spin configuration for spin-dependent
calculations (default: 'r')"

### surfG1D.py surfG.g i parameter JIT note (D2.J1)

**Issue:** Parameter `i` documented as "Contact index" without noting
JAX JIT static-argument behavior.
**User decision:** Apply both this and currentSpin.
**Applied fix:** Add "(static argument for JAX JIT compilation)" to
the i parameter description.

### NEGFE-only feature prose changes (C.3, C.4, C.5, B.J11)

**Issue:** Multiple docs use `negf` (presumed NEGF) with NEGFE-only
features (T= on setSigma, setContactBethe, setContact1D).
**User decision:** Switch the example construction to show NEGFE.
**Applied fixes:**
- best_practices.rst:127-133 (Add Temperature): Now imports NEGFE,
  constructs `negf = NEGFE('molecule', basis='lanl2dz')`, and uses
  setSigma with kwargs.
- advanced_examples.rst Temperature Effects section: Prose now reads
  "Or locally on an NEGFE object", code block now imports NEGFE and
  constructs the instance.
- advanced_examples.rst Energy-Dependent Contacts section: Prose now
  notes "NEGFE provides setContactBethe and setContact1D", code block
  imports and constructs NEGFE.

### best_practices.rst missing negf. prefix (B.J9, B.J10 partial)

**Issue:** Two examples called bare `setContactBethe(...)` /
`setContact1D(...)` without the `negf.` prefix.
**Applied fixes:**
- best_practices.rst:54: `inds = setContactBethe(...)` ->
  `inds = negf.setContactBethe(...)`
- best_practices.rst:63: `inds = setContact1D(...)` ->
  `inds = negf.setContact1D(...)`
(The verifier's separate proposal to flatten the nested-list shape
was REJECTED -- see below.)

### README.md sig1/sig2 typo (originally Bug 1, code-bug)

**Issue:** README references `negf.sig1`/`negf.sig2` (typo); real
attribute names are `negf.sigma1`/`negf.sigma2`.
**User decision:** Use `getSigma()` -- preferred for either NEGF/NEGFE.
**Applied fix:** Replace the bare attribute access with
`sig1, sig2 = negf.getSigma()` then pass into `transport.current(...)`.

---

## Rejected -- not actually broken

### NEGFE constructor pattern (B.J1, B.J4, B.J9 partial, B.J10 partial)

**Verifier claim:** `negf = NEGFE('molecule', basis='lanl2dz')` is
broken because NEGFE signature is `(NEGF)`.
**User clarification:** "NEGFE inherits NEGF" -- the NEGFE constructor
accepts the same `(fn, ...)` form as NEGF. Truth table only extracted
the class-declaration line (`class NEGFE(NEGF):`) which shows the
parent class, not the constructor signature.
**Action:** No fix applied. Truth table extraction needs refinement
for the Phase II refresh.
**Locations untouched:** negf_dft.rst:100, 198; best_practices.rst:52, 61.
**Note:** A previously-applied "fix" in silicon_nanowire.rst:106-107
that wrapped NEGFE in NEGF(NEGFE(...)) was REVERTED based on this
clarification.

### setContactBethe nested-list shape (B.J2, B.J5, B.J9 partial)

**Verifier claim:** `setContactBethe([[1,2,3], [4,5,6]], ...)` is
broken because signature shows single `contactList` parameter.
**User clarification:** "It's DEFINITELY a nested list! It's a list
of lists, each list in the list is for a different contact (listing
the atom numbers)."
**Action:** No fix applied.

### SigmaCalculator(surfG_object) -- the "negf.g" pattern (B.J6, B.J7, B.J8, C.J1, C.J2, originally Bugs 2-6)

**Verifier claim:** SigmaCalculator's first arg per truth table is
`sig1` (matrix), so passing a surfG object (`negf.g`) is wrong.
**User clarification:** "It's a real overload -- there are two
different ways to make a SigmaCalculator: with two energy independent
sigma matrices or with one surfG object."
**Action:** No fix applied. Truth table for SigmaCalculator needs an
overload note; addressed at Phase II truth-table refresh.

### calculate_transmission tuple-unpack (originally Bug 7)

**Verifier claim:** `T, Tspin = calculate_transmission(..., spin='u')`
is broken because truth table shows single-value return.
**Source check:** transport.py:507 returns `transmission, spin_trans`
when spin is specified; line 509 returns `transmission` otherwise.
The 2-tuple unpack with `spin='u'` is correct.
**Action:** No fix applied.

### calculate_dos tuple-unpack (originally Bug 8)

**Verifier claim:** `dos, dos_list = calculate_dos(...)` is broken
because truth table shows single-value return.
**Source check:** transport.py:631/633 returns `(dos_total,
dos_per_site)` always (sometimes with extra spin elements). The
2-tuple unpack is correct.
**Action:** No fix applied.

### NEGFE.g attribute on NEGFE (originally Bugs 3-6 partial)

**Verifier claim:** NEGFE has no `.g` attribute per truth table.
**Source check:** scfE.py:88, 162, 220 all set `self.g` after
contact-setup calls. The attribute IS set, just dynamically by
setContactBethe / setContact1D / setSigma.
**Action:** No fix applied. Truth table extraction needs a "dynamic
attributes" pass for Phase II.

### NEGF.sigma1 / NEGF.sigma2 attribute (originally Bug 1 partial)

**Verifier claim:** NEGF has no sigma1/sigma2 attributes.
**Source check:** scf.py:541-542 sets self.sigma1, self.sigma2 (full
names, not "sig1"/"sig2") after setSigma. The README's typo "sig1"
was the real issue (fixed above).
**Action:** Fixed via README typo correction; the attribute itself
is valid.

### B.J3 negf_dft.rst:182 comment clarification

**Issue:** Verifier flagged that "Basic NEGF Calculation" prose
ambiguously implied NEGFE.
**Action:** Skipped as too low-stakes; the original prose is fine
as-is.

---

## Deferred to Phase II / III

(None; all Phase I judgment items resolved.)

---

## Rejected (user override)

(None; the user did not reject any approved-by-default fixes.)
