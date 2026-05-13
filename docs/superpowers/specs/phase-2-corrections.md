# Phase II Verifier Corrections

Date: 2026-05-10
Source: postwrite verifier pass at phase-2-findings/group-D-postwrite.json

## Summary

- Real mechanical fixes (applied automatically): 3
- Degrading-fix skips (per SP-6): 0
- Judgment fixes (require user approval): 0
- Code-bug flags (NOT auto-applied): 0

All findings were in EXISTING docstrings, not the 5 newly-written ones.
The 5 new docstrings (density.calcEmin, surfG1D module-level,
utils.inv/eig/eigh) verified clean.

## Real mechanical fixes

### Fix 1: NEGF class docstring missing 'section' parameter

File: gauNEGF/scf.py
Symbol: NEGF (class docstring)
Lines: 90-111 (current_snippet area)
Status: APPLIED

The NEGF.__init__ signature at line 136 includes a `section` parameter
between `route` and `nPulay`:

    def __init__(self, fn, basis="chkbasis", func="hf", spin="r",
                 fullSCF=True, route=None, section=None,
                 nPulay=PULAY_MIXING_SIZE):

The class docstring's Parameters section was missing the `section`
entry. Inserted between `route` and `nPulay` documentation:

    section : str, optional
        Gaussian input section specification (default: None)

### Fix 2: densityReal docstring has extraneous 'maxN' parameter

File: gauNEGF/density.py
Symbol: densityReal
Lines: 486-487
Status: APPLIED

The densityReal signature at line 461 is:

    def densityReal(F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL,
                    T=TEMPERATURE, debug=False):

No `maxN` parameter exists. The docstring incorrectly listed:

    maxN : int, optional
        Maximum number of integration points (default: 1000)

Removed those two lines.

### Fix 3: currentF docstring missing 'T' parameter

File: gauNEGF/transport.py
Symbol: currentF
Lines: 877-882 (current_snippet area)
Status: APPLIED

The currentF signature at line 873 is:

    def currentF(fn, dE=ENERGY_STEP, T=TEMPERATURE):

The `T` parameter is used in the function body (line 901, passed to
current(...)). The docstring's Parameters section was missing T.
Inserted between `dE` and the Returns section:

    T : float, optional
        Temperature in Kelvin (default: TEMPERATURE from gauNEGF.config)

Incidental: stripped one trailing space from the function summary line
("Calculate current from saved SCF matrix file. " -> "Calculate current
from saved SCF matrix file."). Cosmetic.

## Degrading-fix skips

None.

## Judgment fixes

None.

## Code-bug flags

None. Per Phase I lessons (verifier guardrails, SP-5), the postwrite
verifier did not surface any false-positive code bug flags this round.
