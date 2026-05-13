# Design: nLower Integrity Check and Retry

Date: 2026-04-10
Branch: crossTermFermiSpec

## Problem

After computing `nLower = Tr(S @ P) + delta_N` from the lower density integral,
a value >= 1 indicates that the integration window [Eminf, Emin] accidentally
enclosed a real eigenvalue. This can happen if:

- The Fock matrix updated during SCF and a state dipped below the current Emin
- calcTSW overshot and set Eminf too low (below a state not in the original
  reference calculation)
- Numerical drift pushed an eigenvalue across the Emin boundary

A non-zero nLower silently corrupts the Fermi energy search and density
matrix for that SCF step, since nLower is subtracted from the total ne before
the complex contour integration.

## Why densityComplex for the Lower Integral

Both the main lower integral and the retry use densityComplex (not densityReal)
for the following reasons:

1. **No near-cancellation**: densityReal integrates Im(G^R) on the real axis.
   Even in a zero-DOS region, G^R and G^A individually can be large -- the result
   is computed as the difference of large terms with finite ETA, accumulating
   numerical noise. densityComplex evaluates G at complex energies (apex of
   semicircle is (Emin - Eminf)/2 eV into the upper half plane) where G is
   genuinely small and smooth.

2. **Fewer integration points**: the integrand is intrinsically close to zero
   away from the real axis, not approximately zero due to cancellation.

3. **Clean EEV detection**: if an eigenvalue IS present in [Eminf, Emin],
   densityComplex correctly encloses it via the residue theorem and gives
   nLower ~= 2 (a clean integer). densityReal gives a noisy fractional value
   that is harder to interpret.

## Detection

After computing nLower, check:

```python
if nLower >= 1 and self.N2 is None:
    # trigger retry
```

Threshold of 1: the lower window is a zero-DOS region so nLower should be
~0 numerically. Values in 0.1-0.9 are more likely numerical noise at the
band edge than a real EEV enclosure. A genuine EEV contributes ~1 or ~2
electrons (spin-paired), so >= 1 is the correct trigger. Because densityComplex
is used, a clean EEV enclosure gives an integer result (~2), making the
threshold reliable.

## Retry Strategy

When nLower >= 1 is detected, retry inline within the current FockToP
call. Do NOT null self.Emin mid-function -- self.Emin is used downstream by
compContourP2 as the lower bound of the complex contour. Nulling it would
crash that integration.

### Retry steps

1. Re-run calcEmin cold: `calcEmin(F, S, g, Emin=None)` -- no warm-start
2. Re-run calcTSW cold: `calcTSW(F, S, g, Eminf=self.Emin, TSW=None)` --
   TSW cold start, Eminf warm-started from the newly computed Emin
3. Re-run lower density integral with new self.Eminf and self.Emin using
   densityComplex
4. Recompute nLower and check >= 1 again
5. If still >= 1: print stronger warning, proceed with best available values
6. self.Emin, self.Eminf, self.TSW are updated to corrected values so that
   the downstream compContourP2(mu) uses the corrected self.Emin

### Key constraint

The retry must complete before compContourP2 is called, since compContourP2
uses self.Emin as its lower contour bound. The corrected self.Emin must be
consistent with self.Eminf (i.e., Eminf < Emin < mu).

## Location

`gauNEGF/scfE.py`, in `FockToP`, immediately after the nLower computation
at line 367.
