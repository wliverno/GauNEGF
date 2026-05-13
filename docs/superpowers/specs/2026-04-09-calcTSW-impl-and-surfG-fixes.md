# Implementation: calcTSW Integration, densityReal Fix, and surfG.g() Warning

Date: 2026-04-09
Branch: crossTermFermiSpec

## Summary

This spec documents four changes implemented in this session:

1. `calcTSW` signature simplified to return `(Eminf, TSW)` only (dropped Emax)
2. `calcTSW` integrated into `scfE.py` to replace the hardcoded `-1e6 eV` floor
3. `densityReal.computePoint` fixed to track density matrix convergence (G^R - G^A)
4. `surfG.g()` now warns via `jax.debug.print` when max iterations is reached

---

## 1. calcTSW Signature

### Previous Behavior

`calcTSW` returned a 3-tuple `(Emin, Emax, TSW)` and accepted `Emin=`, `Emax=`
keyword arguments. Emax expansion was included but not needed -- all
`densityComplex` calls already use a hardcoded upper limit of `1e6 eV`, and no
caller cares about Emax.

### New Behavior

`calcTSW` returns a 2-tuple `(Eminf, TSW)` and accepts `Eminf=`, `TSW=`
keyword arguments. The upper bound is always `1e6 eV` internally.

### Signature

```python
def calcTSW(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES,
            Eminf=None, TSW=None) -> tuple[float, float]:
```

- `Eminf=None`: cold start initializes from `min(eigenvalues(S^-1 @ F))`
- `TSW=None`: cold start recomputes reference TSW via `densityComplex(-1e6, 1e6)`
- Warm start (both provided): skips the expensive reference calculation and
  checks immediately whether the current `Eminf` already matches TSW_ref

### Algorithm

1. If `TSW is None`, compute `TSW_ref = Tr(P @ S) + delta_N` from
   `densityComplex(F, S, g, -1e6, 1e6)`.
2. Loop: compute `TSW_new = Tr(P @ S) + delta_N` from
   `densityComplex(F, S, g, Eminf, 1e6)`.
3. If `|TSW_new - TSW_ref| < tol`, converged -- return `(Eminf, TSW_new)`.
4. Otherwise decrement `Eminf -= 50` and repeat.

### Files Changed

- `gauNEGF/density.py`: simplified `calcTSW` return and parameters, fixed
  print statements that referenced undefined `Emin`/`Emax` variables
- `tests/test_calcTSW.py`: updated to unpack 2-tuple, removed Emax assertions,
  updated warm-start test to use `Eminf=` keyword

---

## 2. calcTSW Integration into scfE.py

### Problem

`densityReal(F, S, g, Eminf, Emin, ...)` was called with `Eminf = -1e6 eV`
(the constant `ENERGY_MIN`). This creates a ~1e6 eV integration window. The
adaptive ANT quadrature distributes points across this window via Chebyshev
nodes -- at low point counts the near-band region is barely sampled, so the
integrator reports large error and keeps doubling N until it hits the cap (486
points) and prints "did not converge, error=1.813e+01".

### Fix

Use `calcTSW` to find a tight `Eminf` (typically 50-150 eV below the band
bottom, not 1e6 eV), so `densityReal` integrates over a small window where
convergence is trivial.

### setIntegralLimits (scfE.py)

When `Emin is None and tol is not None` (the adaptive path), after computing
`self.Emin = calcEmin(...)`, immediately call `calcTSW` to get a tight Eminf:

```python
if Emin is None and tol is not None:
    self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol=tol)
    self.Eminf, self.TSW = calcTSW(self.F*har_to_eV, self.S, self.g,
                                    Eminf=self.Emin, tol=tol)
    self.tol = tol
else:
    self.Emin = Emin
```

Passing `Eminf=self.Emin` seeds the search just below the band bottom so the
loop converges in 1-3 iterations rather than walking down from eigenvalues.

### FockToP (scfE.py)

Each SCF iteration warm-starts `calcTSW` so the Fock matrix update does not
silently invalidate the previous `Eminf`:

```python
if self.N2 is None:
    Eminf_ = self.Eminf if self.Eminf != -1e6 else self.Emin
    self.Eminf, self.TSW = calcTSW(self.F*har_to_eV, self.S, self.g,
                                    Eminf=Eminf_, TSW=self.TSW, tol=self.tol)
    self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, Emin=self.Emin)
    P, _delta_N_lower = densityReal(self.F*har_to_eV, self.S, self.g,
                                     self.Eminf, self.Emin, self.tol, T=0)
```

The `Eminf_ = ... if self.Eminf != -1e6 else self.Emin` guard handles the case
where `setIntegralLimits` was not called (e.g. object constructed with a fixed
Eminf) -- it falls back to `self.Emin` so `calcTSW` starts from a reasonable
point rather than -1e6.

### self.TSW Initialization

`self.TSW = None` is now set in `scf.py:__init__` alongside the existing
`self.Eminf = ENERGY_MIN`. This ensures `FockToP` can call `TSW=self.TSW`
safely before `setIntegralLimits` has been called (the `None` triggers a cold
start in `calcTSW`).

---

## 3. densityReal.computePoint Fix

### Problem

The adaptive integrator tracks convergence via `maxDP = max|new_P - old_P|`
where `P` accumulates the output of `computePoint`. Previously `computePoint`
returned the complex `G^R` contribution. The convergence criterion on `G^R`
is not meaningful for a density matrix -- we want convergence on
`(1/2pi) * (G^R - G^A)`, the anti-Hermitian part that contributes to the
physical density.

### Fix

`computePoint` now returns the density matrix contribution directly:

```python
def computePoint(x, w):
    E = mid * (x + 1) + Emin
    weights = mid * w * fermi(E, mu, T)
    mat, scl = GrIntCross(F, S, g, E, weights)
    return (1j / (2 * jnp.pi)) * (mat - mat.conj().T), scl
```

`mat - mat.conj().T` is the anti-Hermitian part of `G^R` (equivalent to
`G^R - G^A` since `G^A = (G^R)^H`). The integrator now tracks convergence on
the quantity that actually matters.

Note: `jnp.imag(mat)` (element-wise imaginary part) is NOT the same as
`(mat - mat.conj().T) / (2i)` when off-diagonal elements have non-zero real
parts. The correct expression is the adjoint-based one above.

---

## 4. surfG.g() Max Iterations Warning

### Problem

`surfG.g()` uses `lax.while_loop` with `MAX_ITER = 10000`. If an energy point
fails to converge, the loop exits silently. A plain `jax.debug.print` outside
any condition fires for every energy point.

### Fix

Use `lax.cond` with condition `count >= MAX_ITER` to fire `jax.debug.print`
only when the iteration limit is actually hit:

```python
lax.cond(count >= MAX_ITER,
         lambda _: jax.debug.print(
             "WARNING: surfG.g() hit MAX_ITER={n} at E={E:.4f} eV, diff={d:.2e}",
             n=MAX_ITER, E=E, d=diff),
         lambda _: None,
         None)
```

`jax.debug.print` is JIT-compatible and executes at runtime (not trace time),
so it fires once per energy point that genuinely hits the limit. Both lambda
branches return `None`, satisfying JAX's type-consistency requirement for
`lax.cond`. The `None` operand avoids closing over any traced value in the
false branch.

### File Changed

`gauNEGF/surfG1D.py`: replaced commented-out `lax.cond(diff > conv, ...)` with
the corrected `lax.cond(count >= MAX_ITER, ...)` form.
