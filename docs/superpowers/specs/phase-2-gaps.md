# Phase II Gap Inventory

Date: 2026-05-10
Source: gauNEGF/*.py + docs/superpowers/specs/api-truth.md
Method: AST-equivalent inspection by haiku agent

## Module summary

| Module | Module docstring | Style | API index | Public symbols (P/T/S/A) |
|--------|------------------|-------|-----------|--------------------------|
| scf | present | numpy | registered | 16 / 0 / 0 / 0 |
| scfE | present | numpy | registered | 10 / 0 / 0 / 0 |
| density | present | numpy | registered | 27 / 0 / 1 / 1 |
| transport | present | numpy | registered | 18 / 0 / 0 / 0 |
| surfG1D | absent | numpy | registered | 10 / 0 / 0 / 0 |
| surfG3D | present | numpy | missing | 30 / 0 / 0 / 0 |
| surfGBethe | present | numpy | registered | 6 / 0 / 0 / 0 |
| surfGTester | present | numpy | registered | 6 / 0 / 0 / 0 |
| matTools | present | numpy | registered | 5 / 0 / 0 / 0 |
| integrate | present | numpy | registered | 3 / 0 / 0 / 0 |
| spinTools | present | numpy | missing | 8 / 0 / 0 / 0 |
| utils | present | numpy | missing | 1 / 0 / 0 / 3 |
| protocols | present | google | missing | 5 / 0 / 0 / 0 |
| config | present | none | missing | 1 / 0 / 0 / 0 |
| fermiSearch | present | numpy | missing | 4 / 0 / 0 / 0 |

(P = present, T = thin, S = stub, A = absent. Total per row should
match truth-table public symbol count for that module.)

## Per-module detail

### gauNEGF.scf

No docstring gaps; style detected = numpy

### gauNEGF.scfE

No docstring gaps; style detected = numpy

### gauNEGF.density

- Module docstring: present
- Style detected: numpy
- API index: registered
- Symbol gaps:
  - `calcEmin`: absent

### gauNEGF.transport

No docstring gaps; style detected = numpy

### gauNEGF.surfG1D

- Module docstring: absent
- Style detected: numpy
- API index: registered
- Note: Module has no docstring; all public symbols present

### gauNEGF.surfG3D

- Module docstring: present
- Style detected: numpy
- API index: missing
- Note: No gaps detected in docstrings; module not registered in api/index.rst

### gauNEGF.surfGBethe

No docstring gaps; style detected = numpy

### gauNEGF.surfGTester

No docstring gaps; style detected = numpy

### gauNEGF.matTools

No docstring gaps; style detected = numpy

### gauNEGF.integrate

No docstring gaps; style detected = numpy

### gauNEGF.spinTools

- Module docstring: present
- Style detected: numpy
- API index: missing
- Note: No gaps detected in docstrings; module not registered in api/index.rst

### gauNEGF.utils

- Module docstring: present
- Style detected: none
- API index: missing
- Symbol gaps:
  - `inv`: absent
  - `eig`: absent
  - `eigh`: absent

### gauNEGF.protocols

- Module docstring: present
- Style detected: google
- API index: missing
- Note: No gaps detected in docstrings; module not registered in api/index.rst

### gauNEGF.config

- Module docstring: present
- Style detected: none
- API index: missing
- Note: No gaps detected in public symbols; module not registered in api/index.rst

### gauNEGF.fermiSearch

- Module docstring: present (deprecated marker present)
- Style detected: numpy
- API index: missing
- Note: No gaps detected in docstrings; module not registered in api/index.rst

## Aggregate gap counts

- Total absent docstrings across all modules: 4
  - calcEmin (density.py)
  - inv (utils.py)
  - eig (utils.py)
  - eigh (utils.py)
- Total stub docstrings: 0
- Total thin docstrings: 0
- Modules with absent module-level docstring: [surfG1D]
- Modules with mixed style: []

## API Index Registration Status

Registered modules (9):
- scf
- scfE
- density
- transport
- surfG1D
- surfGBethe
- surfGTester
- matTools
- integrate

Missing from api/index.rst (6):
- surfG3D
- spinTools
- utils
- protocols
- config
- fermiSearch

---

## Summary

Audited 15 gauNEGF modules. Total gap count: 5 (4 absent docstrings + 1 absent module-level docstring). The three utility functions in utils.py (inv, eig, eigh) all lack docstrings, which aligns with their minimal complexity as JAX wrappers around standard linalg operations. The calcEmin function in density.py is absent a docstring but is substantively implemented. surfG1D lacks a module-level docstring. Six modules are not yet registered in docs/source/api/index.rst: surfG3D, spinTools, utils, protocols, config, and fermiSearch. Docstring style across registered modules is predominantly numpy-style with one module (protocols) using google-style docstrings.
