# Phase III Sections Flagged for Author Review

Date: 2026-05-13

---

## 1D Chain Contacts -- Pattern B: Custom Coupling with Auto Onsite

**Reason flagged:** Synthesized example; no real test file directly demonstrates this
pattern (tauList only, no alphas/betas) in the repository. The code is API-valid per
api-truth.md but was not lifted from a real test case.

**Location:** docs/source/guides/contacts_1d.rst -- "Three Usage Patterns" section,
sub-section on Pattern B.

**Suggested action:** User reviews the code block for Pattern B and either:
a) Confirms it is correct as written (leave the .. note:: directive), or
b) Replaces it with a real example from a production run file.

**Note directive inserted:** The section already has a ".. note:: This section needs
author review." marker in the RST output per SP-7 guidelines.
