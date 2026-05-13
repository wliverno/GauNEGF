# Phase I Code Bug Findings (Reclassified after user review)

Date: 2026-05-09 (initial), 2026-05-09 (post-user-review)

The Phase I verifiers flagged 8 findings as `code_bug=true`. After user
review during Task 9, the classification picture changed substantially:

| Original Bug # | Verifier claim | Actual status |
|----------------|----------------|---------------|
| Bug 1 | README sig1/sig2 nonexistent | DOC TYPO -- fixed (use getSigma()) |
| Bug 2 | SigmaCalculator(g) wrong | NOT A BUG -- valid overload |
| Bug 3, 4 | NEGFE.g attribute missing | NOT A BUG -- dynamic attribute |
| Bug 5, 6 | sigma1/sigma2/g missing | NOT A BUG -- valid attributes |
| Bug 7 | calculate_transmission tuple | NOT A BUG -- valid spin-mode return |
| Bug 8 | calculate_dos tuple | NOT A BUG -- valid 2-tuple return |

**Net result: zero genuine code bugs surfaced by Phase I.**

The verifier's misclassification stems from reading the truth table too
literally:
- Truth table extracts class-declaration line (e.g. `NEGFE(NEGF)`),
  not constructor signature, missing inheritance.
- Truth table extracts function-signature defaults but does not record
  variable-shape returns (single value vs tuple based on flags).
- Truth table extracts `__init__` parameters but does not record
  attributes set dynamically by other methods.
- Truth table does not record method overloads via duck typing
  (SigmaCalculator's surfG-vs-matrix dispatch).

These limitations are noted for the Phase II truth table refresh.

---

## Recommended truth table refresh additions for Phase II

When Phase II refreshes `api-truth.md`, add these conventions:

1. **Class inheritance constructors:** When a class has `class X(Y):`
   and no explicit `__init__`, document that X inherits Y's
   constructor. Currently shown as `Signature: NEGFE(NEGF)` which
   reads like "constructor takes a NEGF instance" -- misleading.

2. **Dynamic attributes:** Document attributes set by methods other
   than `__init__`. For example, NEGF.setSigma sets `self.sigma1`,
   `self.sigma2`, `self.sigma12`, `self.Gam1`, `self.Gam2`. NEGFE's
   `setContactBethe` / `setContact1D` / `setSigma` all set `self.g`.
   These need a per-class "Attributes set by methods" section.

3. **Variable-shape returns:** For functions with branching returns
   (single value vs tuple based on parameters), the truth table
   should note this. Examples: `calculate_transmission` returns
   tuple when spin-mode is on; `calculate_dos` always returns
   2+ tuple; `current` returns scalar; etc.

4. **Method overloads via duck typing:** Document classes/functions
   whose first-arg type changes behavior. SigmaCalculator accepts
   either two sigma matrices or a single surfG object.

5. **Nested-list parameter conventions:** Where parameters like
   `contactList` are documented as a list but actually expect a
   list-of-lists (one inner list per contact), note the convention.

---

## Truly open questions (none currently)

After user review, no findings remain in genuine "code bug requiring
investigation" state. If future audits find such issues, they go
here for triage.

---

End of bugs file.
