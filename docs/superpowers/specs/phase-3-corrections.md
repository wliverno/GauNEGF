# Phase III Verifier Corrections

Date: 2026-05-13
Source: phase-3-findings/guides-postwrite.json

## Summary

- Real mechanical fixes (applied automatically): 0
- Degrading-fix skips (per SP-6): 0
- Judgment fixes (require user approval): 0
- Code-bug flags (NOT auto-applied): 0

All 5 guide files verified clean against api-truth.md:
- contact_choice.rst: 0 findings
- contacts_1d.rst: 0 findings
- contacts_bethe.rst: 0 findings
- config_tuning.rst: 0 findings
- workflow_recipes.rst: 0 findings

Key validated patterns:
- NEGFE constructor inheritance (NEGFE(*args) is valid)
- setContactBethe nested list format ([[atoms...], [atoms...]])
- SigmaCalculator duck-typed overload with surfG object
- Variable-shape returns from calculate_transmission and calculate_dos
- Dynamic attributes negf.g and negf.fermi

## Author-review TODOs

1 section flagged for author review (from writer, not verifier):
- contacts_1d.rst Pattern B: synthesized custom-coupling example
  See docs/superpowers/specs/phase-3-todos.md for details.
