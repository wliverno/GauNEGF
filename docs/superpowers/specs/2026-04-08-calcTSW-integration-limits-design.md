# Design: calcTSW -- Spectral-Weight-Based Integration Limits

Date: 2026-04-08
Branch: crossTermFermiSpec

## Problem

The current approach to setting the lower integration bound (`calcEmin`) walks
down from the lowest eigenvalue until the DOS at that point falls below a
tolerance. The absolute lower floor `Eminf = -1e6 eV` is then used as the hard
lower bound when calling `densityComplex(F, S, g, Eminf, Emin, ...)` to count
electrons below the contour region (`nLower`). With overlap between unit cells,
the adaptive integrator called over the enormous `[Eminf, Emin]` window never
converges because the integrand has non-trivial structure across that span.

## Solution

Replace the DOS-threshold approach with a new function `calcTSW` that iteratively
expands both Emin and Emax until the Total Spectral Weight (TSW = Tr(rho @ S))
stabilizes. Once TSW stabilizes, Emin is guaranteed to be below all occupied
states, so the lower density integral starts at Emin directly -- no `Eminf` floor
and no `nLower` correction needed.

This approach is similar to the integration limit strategy used in ANT.Gaussian.

## New Function: calcTSW

### Location

`gauNEGF/density.py`, alongside the existing `calcEmin`.

### Signature

```python
def calcTSW(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES,
            Emin=None, Emax=None, TSW=None):
```

### Arguments

- `F`, `S`, `g`: Fock matrix, overlap matrix, surface Green's function object
- `tol`: convergence tolerance for TSW stabilization (default: FERMI_CALCULATION_TOL)
- `maxN`: maximum number of expansion iterations (default: MAX_CYCLES)
- `Emin`, `Emax`, `TSW`: optional warm-start values from a previous call (e.g.,
  prior SCF iteration); if None, initialized from eigenvalues

### Return

`(Emin, Emax, TSW)` -- converged bounds and the final spectral weight scalar

### Initialization

If `Emin` or `Emax` is None, diagonalize `S^-1 @ F` and set:
```
Emin = min(eigenvalues)
Emax = max(eigenvalues)
```
If `TSW` is None, set `TSW_prev = 0.0` to force at least one iteration.

### Expansion loop

`densityComplex` signature: `densityComplex(F, S, g, Emin, mu, tol, T)`.
For the TSW calculation, Emax is passed as `mu`. This causes the complex contour
to be a semicircle from Emin to Emax in the upper half-plane, enclosing all
poles of G^R between those bounds. When Emax is above all eigenvalues, this
yields the full spectral weight.

Each iteration:
1. Call `P, delta_N = densityComplex(F, S, g, Emin, Emax, tol, T=0)`
2. Compute `TSW_new = np.trace(P @ g.S).real + delta_N`
3. If `abs(TSW_new - TSW_prev) < tol`: converged, break
4. Else: `Emin -= 10`, `Emax += 10`, `TSW_prev = TSW_new`

Step size is 10 eV per iteration on both sides simultaneously. This was chosen
empirically -- large enough to converge in few iterations for typical Au/organic
systems (band widths ~20-40 eV), while staying coarse enough to avoid excessive
integration calls.

If `maxN` is reached without convergence, print a warning (using `print()`,
consistent with `calcEmin`).
Print final `Emin`, `Emax`, `TSW` on completion.

### Why T=0 for TSW

TSW measures the total spectral weight of all states, independent of temperature.
Using T=0 avoids smearing the upper bound and gives a clean count.

## Changes to scfE.py -- FockToP (active path only)

### Current code (lines ~356-361, the `N2 is None` branch)

```python
self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, Emin=self.Emin)
P, _delta_N_lower = densityComplex(self.F*har_to_eV, self.S, self.g,
                                   self.Eminf, self.Emin, self.tol, T=0)
nLower = np.trace(self.S@P).real + _delta_N_lower
```

### Replacement

```python
self.Emin, self.Emax, self.TSW = calcTSW(
    self.F*har_to_eV, self.S, self.g,
    Emin=self.Emin,
    Emax=getattr(self, 'Emax', None),
    TSW=getattr(self, 'TSW', None))
nLower = 0.0
```

`self.Emax` and `self.TSW` are stored on the object so subsequent SCF cycles
warm-start from the previous converged bounds, avoiding redundant expansion.

All downstream `ne - nLower` expressions remain syntactically unchanged (nLower
is 0.0, so they simplify arithmetically but no further edits are needed).

The `N2 is not None` branch (deprecated real-axis path) is left untouched for
back-compatibility.

`self.Eminf` is no longer passed to `densityComplex` calls in this path; Emin
from `calcTSW` is used as the lower bound directly.

### Note on bisectFermi / predict path

`bisectFermi` (the energy-independent method used in the `predict` Fermi update
path) still takes `Eminf` and calls the analytical `density()` formula. This
is a separate concern and is not changed.

## Changes to density.py -- getFermiContact

### Current code (lines ~1048-1056)

```python
if Emin is None:
    Emin = calcEmin(F, S, g, tol=conv, maxN=maxcycles)
P, _delta_N_lower = densityComplex(F, S, g, Eminf, Emin, tol, T=0)
nLower = np.trace(P@g.S).real + _delta_N_lower
assert nLower < ne, ...
ne -= nLower
```

### Replacement

```python
if Emin is None:
    Emin, _Emax, _TSW = calcTSW(F, S, g, tol=conv, maxN=maxcycles)
# nLower is 0 by construction: calcTSW guarantees Emin is below all occupied
# states, so the integral from Emin to mu captures all electrons.
# The assert nLower < ne and ne -= nLower lines are removed.
```

The `Eminf` parameter on `getFermiContact` can be left in the signature for
back-compatibility but is no longer used in the active code path.

## What is NOT changed

- `calcEmin` -- kept as-is for non-overlap cases and deprecated paths
- `integralFit`, `integralFitNEGF`, `setIntegrationLimits` -- deprecated,
  left untouched
- `calcFermiMuller`, `calcFermiSecant`, `calcFermiBisect`, `calcFermiPolyFit`
  -- take `Emin` as an argument, no internal use of `calcEmin` or `Eminf`
- `bisectFermi` -- energy-independent analytical method, separate concern

## Future work

`Emax` returned by `calcTSW` is a natural upper bound for the Fermi energy
bisection search (replaces the current eigenvalue-based `uBound`). This is
deferred to a later change.

## Testing

- Unit test: call `calcTSW` on a small system, verify TSW converges and that Emin < all eigenvalues and Emax > all eigenvalues
- Regression: run the existing Fermi energy and density tests -- results should
  be identical to the current `calcEmin` + `nLower` approach for non-overlap
  cases (where `nLower` was already ~0)
- Overlap case: verify adaptive integration converges where it previously failed, such as the system in AuNEGF.py
