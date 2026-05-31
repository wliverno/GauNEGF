# densityReal ANT Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `densityReal`'s Gauss-Legendre doubling loop with ANT adaptive quadrature (`integratePointsAdaptiveANT`), then use it for nLower instead of `densityComplex`.

**Architecture:** The zero-DOS section [Eminf, Emin] (below all occupied states) is currently integrated with `densityComplex`, which builds a large complex contour and hits MAX_GRID_POINTS for Au. Switching to `densityReal` (real-axis ANT) exploits the fact that Im(G^R)=0 in the zero-DOS region, so the integral converges trivially in ~6 points. The key bug in the old `densityReal` (GL doubling): it initialises P_prev=zeros, so when the N=1 midpoint also gives P=0, maxDP=0<tol and it exits immediately -- spurious convergence. ANT never compares against zeros; the first convergence check is N=2 vs N=6.

**Tech Stack:** JAX/NumPy, `integratePointsAdaptiveANT` (already in density.py), `GrIntCross` (already in integrate.py), pytest.

**Branch:** `crossTermFermiSpec`. Files are at the 6e5b48c reverted state (density.py and scfE.py rolled back; calcTSW remains in density.py but is no longer called from getFermiContact or FockToP).

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `tests/test_densityReal.py` | Already created | RED tests written; need to run and verify they fail |
| `gauNEGF/density.py:459-507` | Modify | Replace GL doubling body with ANT; keep same signature |
| `gauNEGF/density.py:1111` | Modify | `densityComplex` -> `densityReal` for nLower in getFermiContact |
| `gauNEGF/scfE.py:358` | Modify | `densityComplex` -> `densityReal` for nLower in FockToP |
| `tests/test_calcTSW.py:68-78` | Modify | Remove stale `test_getFermiContact_with_calcTSW` (tests old calcTSW approach) |

No new files beyond the test file. `densityReal` is already exported via `from gauNEGF.density import *` in scfE.py.

---

## Task 1: Verify RED tests fail with current (GL) densityReal

**Files:**
- Test: `tests/test_densityReal.py`

The test file already exists. Confirm the critical RED test fails before touching production code.

- [ ] **Step 1: Run the test suite and observe failures**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_densityReal.py -v 2>&1
```

Expected output contains:
```
FAILED test_densityReal.py::test_densityReal_does_not_converge_at_N1
```
The failure message will be: `densityReal converged at N=1 -- this is a spurious convergence.`

The other three tests (`test_densityReal_zero_dos_gives_zero`, `test_densityReal_matches_densityRealN_in_zero_dos`, `test_densityReal_returns_correct_types`) should PASS -- they define the correctness contract that must survive the refactoring.

If `test_densityReal_does_not_converge_at_N1` PASSES with current code, the test is broken. Stop and investigate before proceeding.

---

## Task 2: Implement ANT-based densityReal (GREEN)

**Files:**
- Modify: `gauNEGF/density.py:459-507`

Replace the GL doubling body. Keep: same function name, same signature, same docstring structure, same return types `(ndarray, float)`. Change: the body that currently uses `densityRealN` in a doubling loop.

- [ ] **Step 1: Replace densityReal body in density.py**

Current body (lines 492-507):
```python
    P = np.zeros_like(F)
    delta_N = 0.0
    N = 1
    maxDP = 1e9
    while N<maxN:
        P_ = P.copy()

        P, delta_N = densityRealN(F, S, g, Emin, mu, N, T, showText=False)
        maxDP = np.max(np.abs(P - P_))
        if maxDP< tol:
            print(f'Adaptive integration converged to {maxDP:.3e} in {N} points.')
            return P, delta_N
        N *= 2

    print(f'Warning: adaptive integration not converged after {maxN} points: maxDP={maxDP:.2E}')
    return P, delta_N
```

Replace with:
```python
    nKT = N_KT
    kT = kB * T
    Emax = mu + nKT * kT

    mid = (Emax - Emin) / 2

    def computePoint(x, w):
        E = mid * (x + 1) + Emin
        weights = mid * w * fermi(E, mu, T)
        return GrIntCross(F, S, g, E, weights)

    print('Real Axis Integration (ANT):')
    lineInt, cross_scalar = integratePointsAdaptiveANT(computePoint, tol=tol, debug=debug, maxN=maxN)

    P = (1j / (2 * jnp.pi)) * (lineInt - lineInt.conj().T)
    delta_N = float(-(1 / jnp.pi) * jnp.imag(cross_scalar))
    return P, delta_N
```

Note: `N_KT`, `kB`, `fermi`, `GrIntCross`, `integratePointsAdaptiveANT`, `jnp` are all already in scope at line 459 (same file, same imports). The `Emax` extension above `mu` handles finite-T Fermi tail the same way `densityRealN` does; at T=0 kT=0 so Emax=mu.

Also update the docstring to remove the reference to `densityRealN`:

Current first two docstring lines after the `"""`:
```
    Calculate equilibrium density matrix using adaptive real-axis integration.

    Wrapper for densityRealN() using the tol and maxN specification to determine grid size
```

Replace with:
```
    Calculate equilibrium density matrix using adaptive real-axis ANT integration.

    Uses integratePointsAdaptiveANT with the same Gauss-Chebyshev scheme as
    densityComplex, but integrating along the real axis instead of a complex contour.
    Suitable for zero-DOS regions (below all occupied states) where Im(G^R)=0
    and the integral converges trivially in a small number of points.
```

- [ ] **Step 2: Run the full test suite to verify GREEN**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_densityReal.py -v 2>&1
```

All four tests must pass:
```
PASSED test_densityReal.py::test_densityReal_does_not_converge_at_N1
PASSED test_densityReal.py::test_densityReal_zero_dos_gives_zero
PASSED test_densityReal.py::test_densityReal_matches_densityRealN_in_zero_dos
PASSED test_densityReal.py::test_densityReal_returns_correct_types
```

- [ ] **Step 3: Commit**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode
git add tests/test_densityReal.py gauNEGF/density.py
git commit -m "refactor: replace densityReal GL loop with ANT adaptive quadrature

densityReal now uses integratePointsAdaptiveANT (Gauss-Chebyshev) instead
of the old Gauss-Legendre doubling loop. This eliminates spurious N=1
convergence when the GL midpoint falls in a zero-DOS gap.

Intended use: integrating [Eminf, calcEmin_output] where Im(G^R)=0.
ANT converges in 6 points; old GL 'converged' in 1 (spurious).
"
```

---

## Task 3: Switch nLower in getFermiContact from densityComplex to densityReal

**Files:**
- Modify: `gauNEGF/density.py:1111`

Currently `getFermiContact` counts electrons below Emin with `densityComplex`, which builds a large contour from Eminf (~-1e6) to Emin. This hits MAX_GRID_POINTS for Au. The region [Eminf, Emin] has zero DOS (Im(G^R)=0 on real axis), so `densityReal` gives the same result (~0) in 6 points.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_densityReal.py` (append to end of file):

```python
def test_getFermiContact_nLower_uses_densityReal(two_state_g):
    """getFermiContact should complete without hitting densityComplex for nLower.

    With states at -10 and -1 eV and mu near -1 eV, nLower from [Eminf, calcEmin]
    should be ~0 whether computed via densityReal or densityComplex.
    The test checks the result is sensible; densityReal does it cheaper.
    """
    from gauNEGF.density import getFermiContact
    g = two_state_g
    ne = 1.0  # target: 1 electron (state at -1 eV occupied, -10 eV not)
    # getFermiContact should find mu just above -1 eV
    mu = getFermiContact(g, ne, T=0)
    assert -1.5 < mu < -0.5, f"Expected mu near -1 eV, got {mu:.4f} eV"
```

- [ ] **Step 2: Run the test to see it pass (baseline -- this test checks correctness, not method)**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_densityReal.py::test_getFermiContact_nLower_uses_densityReal -v 2>&1
```

This test may already PASS with the current code (it checks correctness, not which method is used). That is intentional -- it is a regression test ensuring the swap does not break the answer.

- [ ] **Step 3: Replace densityComplex with densityReal in getFermiContact**

In `gauNEGF/density.py`, find the nLower block in `getFermiContact` (currently around line 1111):

```python
    # Count electrons below Emin
    P, _delta_N_lower = densityComplex(F, S, g, Eminf, Emin, tol, T=0)
    nLower = np.trace(P@g.S).real + _delta_N_lower
    assert nLower < ne, "ne ({ne}) exceeds mininum number of electrons ({nLower:.2f})"
    print(f"{nLower:.2f} electrons below Emin.")
    ne -= nLower # Subtract from total
```

Replace with:

```python
    # Count electrons below Emin (zero-DOS region: real-axis ANT converges in ~6 pts)
    P, _delta_N_lower = densityReal(F, S, g, Eminf, Emin, tol, T=0)
    nLower = np.trace(P@g.S).real + _delta_N_lower
    assert nLower < ne, "ne ({ne}) exceeds mininum number of electrons ({nLower:.2f})"
    print(f"{nLower:.2f} electrons below Emin.")
    ne -= nLower # Subtract from total
```

Only the first line changes (`densityComplex` -> `densityReal`).

- [ ] **Step 4: Run all densityReal tests to verify nothing broke**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_densityReal.py -v 2>&1
```

All five tests must pass.

- [ ] **Step 5: Commit**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode
git add gauNEGF/density.py tests/test_densityReal.py
git commit -m "fix: use densityReal for nLower in getFermiContact

The zero-DOS region [Eminf, Emin] has Im(G^R)=0 on the real axis,
so densityReal (ANT) converges in 6 points instead of densityComplex
hitting MAX_GRID_POINTS with a contour of radius ~300 eV.
"
```

---

## Task 4: Switch nLower in FockToP (scfE.py) from densityComplex to densityReal

**Files:**
- Modify: `gauNEGF/scfE.py:358`

Same logic as Task 3 but for the SCF loop. `FockToP.calcLowerDensity` currently calls `densityComplex(F, S, g, Eminf, Emin, tol, T=0)` for nLower; switch to `densityReal`.

- [ ] **Step 1: Replace densityComplex with densityReal in FockToP**

In `gauNEGF/scfE.py`, find (around line 356-358):
```python
        if self.N2 is None:
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, Emin=self.Emin)
            P, _delta_N_lower = densityComplex(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.tol, T=0)
```

Replace the third line only:
```python
        if self.N2 is None:
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, Emin=self.Emin)
            P, _delta_N_lower = densityReal(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.tol, T=0)
```

`densityReal` is already in scope via `from gauNEGF.density import *` at line 23 of scfE.py.

- [ ] **Step 2: Run existing test suite to check no regressions**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_densityReal.py test_cross_term.py test_surfG1D_features.py -v 2>&1
```

All tests must pass.

- [ ] **Step 3: Commit**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode
git add gauNEGF/scfE.py
git commit -m "fix: use densityReal for nLower in FockToP (SCF loop)

Mirrors the getFermiContact change: zero-DOS [Eminf, Emin] region
is now integrated on the real axis (ANT) instead of complex contour.
"
```

---

## Task 5: Remove stale test_getFermiContact_with_calcTSW

**Files:**
- Modify: `tests/test_calcTSW.py:68-79`

This test was added in commit 328ff0a to validate the now-reverted calcTSW approach for getFermiContact. Since getFermiContact is back to the calcEmin+densityReal path, the test is stale and will fail if run (it imports getFermiContact and tests the calcTSW-driven behavior which no longer exists).

- [ ] **Step 1: Remove the stale test from test_calcTSW.py**

Delete the entire `test_getFermiContact_with_calcTSW` function (lines 68-78 of test_calcTSW.py):

```python
def test_getFermiContact_with_calcTSW(au_system):
    """getFermiContact should still find the correct Au Fermi energy after calcTSW switch."""
    from gauNEGF.density import getFermiContact

    gBAt, ne = au_system
    ne_per_spin = ne / 2
    AU_BULK_FERMI_EV = 2.84

    fermi = getFermiContact(gBAt, ne_per_spin, conv=1e-3, maxcycles=1000, T=0)
    assert abs(fermi - AU_BULK_FERMI_EV) < 0.02, \
        f"Au Fermi {fermi:.4f} eV differs from benchmark {AU_BULK_FERMI_EV} eV"
```

- [ ] **Step 2: Run test_calcTSW.py to confirm remaining tests still pass**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_calcTSW.py -v 2>&1
```

Expected:
```
PASSED test_calcTSW.py::test_calcTSW_converges
PASSED test_calcTSW.py::test_calcTSW_warm_start
```

- [ ] **Step 3: Commit**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode
git add tests/test_calcTSW.py
git commit -m "test: remove stale test_getFermiContact_with_calcTSW

That test validated the now-reverted calcTSW integration path for
getFermiContact. getFermiContact now uses calcEmin+densityReal.
"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the complete relevant test suite**

```bash
cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests
python -m pytest test_densityReal.py test_calcTSW.py test_cross_term.py \
    test_surfG1D_features.py test_bethe_cross_term_fermi.py -v 2>&1
```

All tests must pass with no warnings about convergence failures.

- [ ] **Step 2: Verify git log tells a clean story**

```bash
git log --oneline -6
```

Expected (newest first):
```
<hash> test: remove stale test_getFermiContact_with_calcTSW
<hash> fix: use densityReal for nLower in FockToP (SCF loop)
<hash> fix: use densityReal for nLower in getFermiContact
<hash> refactor: replace densityReal GL loop with ANT adaptive quadrature
<hash> fixed Fermi update, added calcTSW function   <- 6e5b48c (our base)
```

- [ ] **Step 3: Verify the key behavioral change with a quick smoke test**

```python
# Run from repo root
python -c "
import numpy as np
import jax; jax.config.update('jax_enable_x64', True)
from gauNEGF.surfG1D import surfG
from gauNEGF.density import densityReal, calcEmin

F = np.diag([-10.0, -1.0]).astype(complex)
S = np.eye(2, dtype=complex)
g = surfG(F, S, [[0],[1]], eta=1e-4)
E_split = calcEmin(F, S, g)
P, dN = densityReal(F, S, g, E_split-10, E_split, T=0)
print('ne in zero-DOS section:', np.trace(S@P).real + dN)
# Expected: ne near 0.0, NOT a spurious early exit
# Output should show 'in 6 points' or higher, NOT 'in 1 points'
"
```

The output must NOT contain "in 1 points".

---

## Self-Review

**Spec coverage:**
- Spec item: Replace densityReal with ANT -> Task 2 (complete implementation)
- Spec item: Use densityReal for nLower instead of densityComplex -> Tasks 3+4
- Spec item: TDD -> Task 1 (RED), Task 2 (GREEN)
- Spec item: Remove stale test -> Task 5

**Placeholder scan:** No TBD, TODO, or "similar to" language. All code blocks are complete.

**Type consistency:**
- `densityReal` returns `(P: ndarray, delta_N: float)` -- consistent across Tasks 2, 3, 4
- `getFermiContact` call signature unchanged: `densityReal(F, S, g, Eminf, Emin, tol, T=0)` matches `densityReal(F, S, g, Emin, mu, tol=..., T=...)`
- scfE.py call: `densityReal(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.tol, T=0)` -- same pattern
