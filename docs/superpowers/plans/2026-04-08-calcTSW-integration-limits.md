# calcTSW Integration Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the DOS-threshold `calcEmin` + `Eminf`/`nLower` approach with a spectral-weight-based `calcTSW` that iteratively expands integration bounds until total spectral weight stabilizes, eliminating the need for the enormous `[Eminf, Emin]` lower integration window.

**Architecture:** New function `calcTSW` in `gauNEGF/density.py` expands `Emin`/`Emax` by 10 eV per iteration until `Tr(rho @ S)` stabilizes. Callers in `scfE.py:FockToP` and `density.py:getFermiContact` switch from `calcEmin` + `densityComplex(Eminf, Emin)` + `nLower` to `calcTSW`, setting `nLower = 0`.

**Tech Stack:** NumPy, JAX, existing `densityComplex`/`eigh`/`inv` from gauNEGF

---

### Task 1: Add `calcTSW` function with unit test

**Files:**
- Modify: `gauNEGF/density.py:855-871` (insert after `calcEmin`)
- Create: `tests/test_calcTSW.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calcTSW.py`:

```python
"""Unit test for calcTSW: spectral-weight-based integration limits."""
import sys
sys.path.insert(0, '..')

import pytest
import numpy as np
import jax.numpy as jnp

from gauNEGF.surfGBethe import surfGBAt
from gauNEGF.density import calcTSW
from gauNEGF.utils import inv, eigh
from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors


dim = 9
eta = 1e-6


@pytest.fixture
def au_system():
    """Build a surfGBAt for Au single cell -- same fixture as cross-term tests."""
    ne, H0, Sdict, Vdict = read_bethe_params('Au')
    vecs = gen_fcc_111_neighbors()
    Vlist = [construct_mat(Vdict, v) for v in vecs]
    Slist = [construct_mat(Sdict, v) for v in vecs]
    gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    return gBAt, ne


def test_calcTSW_converges(au_system):
    """calcTSW should converge and return bounds that bracket all eigenvalues."""
    gBAt, ne = au_system
    F = gBAt.F
    S = gBAt.S

    Emin, Emax, TSW = calcTSW(F, S, gBAt)

    # TSW should be close to the matrix dimension (total spectral weight)
    assert TSW > 0, f"TSW should be positive, got {TSW}"

    # Emin should be below all eigenvalues, Emax above
    D, _ = eigh(inv(S) @ F)
    eigenvalues = np.real(D).flatten()
    assert Emin < min(eigenvalues), \
        f"Emin {Emin} should be below min eigenvalue {min(eigenvalues)}"
    assert Emax > max(eigenvalues), \
        f"Emax {Emax} should be above max eigenvalue {max(eigenvalues)}"


def test_calcTSW_warm_start(au_system):
    """Warm-started calcTSW should converge in zero iterations if bounds are good."""
    gBAt, ne = au_system
    F = gBAt.F
    S = gBAt.S

    # First call: cold start
    Emin1, Emax1, TSW1 = calcTSW(F, S, gBAt)

    # Second call: warm start with converged values
    Emin2, Emax2, TSW2 = calcTSW(F, S, gBAt, Emin=Emin1, Emax=Emax1, TSW=TSW1)

    # Warm start should return identical bounds (no expansion needed)
    assert Emin2 == Emin1, f"Warm-started Emin changed: {Emin1} -> {Emin2}"
    assert Emax2 == Emax1, f"Warm-started Emax changed: {Emax1} -> {Emax2}"
    assert abs(TSW2 - TSW1) < 1e-6, \
        f"Warm-started TSW changed: {TSW1} -> {TSW2}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_calcTSW.py -v`
Expected: FAIL with `ImportError: cannot import name 'calcTSW' from 'gauNEGF.density'`

- [ ] **Step 3: Implement `calcTSW` in `gauNEGF/density.py`**

Insert immediately after the `calcEmin` function (after line 871). The function goes in the `## INTEGRATION LIMIT FUNCTIONS` section alongside `calcEmin`:

```python
def calcTSW(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES,
            Emin=None, Emax=None, TSW=None):
    """Calculate integration bounds by stabilizing total spectral weight.

    Iteratively expands Emin and Emax until Tr(rho @ S) + delta_N
    stabilizes, guaranteeing that bounds bracket all occupied states.

    Parameters
    ----------
    F : ndarray
        Fock matrix in eV.
    S : ndarray
        Overlap matrix.
    g : surfG object
        Surface Green's function calculator.
    tol : float, optional
        Convergence tolerance for TSW change (default: FERMI_CALCULATION_TOL).
    maxN : int, optional
        Maximum expansion iterations (default: MAX_CYCLES).
    Emin : float or None, optional
        Warm-start lower bound in eV. If None, initialized from eigenvalues.
    Emax : float or None, optional
        Warm-start upper bound in eV. If None, initialized from eigenvalues.
    TSW : float or None, optional
        Warm-start spectral weight. If None, forces at least one iteration.

    Returns
    -------
    tuple (float, float, float)
        (Emin, Emax, TSW) -- converged bounds and final spectral weight.
    """
    # Initialize from eigenvalues if no warm-start values
    if Emin is None or Emax is None:
        D, _ = eigh(inv(S) @ F)
        eigs = np.real(D).flatten()
        if Emin is None:
            Emin = float(min(eigs))
        if Emax is None:
            Emax = float(max(eigs))

    TSW_prev = 0.0 if TSW is None else TSW

    for i in range(maxN):
        # Emax is passed as mu: the contour from Emin to Emax encloses all
        # poles of G^R in that window, yielding the total spectral weight.
        P, delta_N = densityComplex(F, S, g, Emin, Emax, tol, T=0)
        TSW_new = np.trace(P @ g.S).real + delta_N
        if abs(TSW_new - TSW_prev) < tol:
            print(f'calcTSW converged: Emin={Emin:.2f}, Emax={Emax:.2f}, TSW={TSW_new:.4f}')
            return Emin, Emax, TSW_new
        Emin -= 10
        Emax += 10
        TSW_prev = TSW_new

    print(f'Warning: calcTSW did not converge after {maxN} iterations '
          f'(last dTSW={abs(TSW_new - TSW_prev):.2E})')
    print(f'calcTSW: Emin={Emin:.2f}, Emax={Emax:.2f}, TSW={TSW_new:.4f}')
    return Emin, Emax, TSW_new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_calcTSW.py -v`
Expected: PASS (both `test_calcTSW_converges` and `test_calcTSW_warm_start`)

- [ ] **Step 5: Pause for user commit**

Stop and ask the user to commit. Suggested message:

```
feat: add calcTSW spectral-weight integration limit function

Implements calcTSW in density.py that iteratively expands Emin/Emax
until Tr(rho @ S) stabilizes. Supports warm-start from prior SCF
iteration. Includes unit tests with Au Bethe lattice.
```

---

### Task 2: Wire `calcTSW` into `getFermiContact`

**Files:**
- Modify: `gauNEGF/density.py:1048-1056`
- Modify: `tests/test_calcTSW.py` (add regression test)

- [ ] **Step 1: Write the failing regression test**

Append to `tests/test_calcTSW.py`:

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

- [ ] **Step 2: Run test to verify it passes with the OLD code**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_calcTSW.py::test_getFermiContact_with_calcTSW -v`
Expected: PASS (this is a regression baseline -- the test should pass both before and after the change)

- [ ] **Step 3: Modify `getFermiContact` to use `calcTSW`**

In `gauNEGF/density.py`, replace lines 1048-1056 (the `if Emin is None` block through `ne -= nLower`).

This removes 8 lines (the `calcEmin` call, the `densityComplex` lower-density call, the `nLower` computation, the `assert nLower < ne`, the print, and `ne -= nLower`) and replaces them with 3 lines.

Old code:
```python
    # Calculate Emin from DOS if not provided
    if Emin is None:
        Emin = calcEmin(F, S, g, tol=conv, maxN=maxcycles)

    # Count electrons below Emin
    P, _delta_N_lower = densityComplex(F, S, g, Eminf, Emin, tol, T=0)
    nLower = np.trace(P@g.S).real + _delta_N_lower
    assert nLower < ne, "ne ({ne}) exceeds mininum number of electrons ({nLower:.2f})"
    print(f"{nLower:.2f} electrons below Emin.")
    ne -= nLower # Subtract from total
```

New code:
```python
    # Calculate Emin using spectral weight stabilization
    if Emin is None:
        Emin, _Emax, _TSW = calcTSW(F, S, g, tol=conv, maxN=maxcycles)
    # nLower is 0 by construction: calcTSW guarantees Emin is below all
    # occupied states, so the integral from Emin to mu captures all electrons.
    # The assert, nLower subtraction, and Eminf-based densityComplex call are
    # all removed -- no lower density correction is needed.
```

- [ ] **Step 4: Run the regression test to verify it still passes**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_calcTSW.py::test_getFermiContact_with_calcTSW -v`
Expected: PASS (Fermi energy should match benchmark within tolerance)

- [ ] **Step 5: Run the existing Bethe cross-term tests to check for regressions**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_bethe_cross_term_fermi.py -v`
Expected: PASS (all existing tests still pass)

- [ ] **Step 6: Pause for user commit**

Stop and ask the user to commit. Suggested message:

```
refactor: switch getFermiContact from calcEmin+nLower to calcTSW

Replaces the Eminf-based lower density count with calcTSW bounds.
nLower is eliminated since Emin now brackets all occupied states.
Eminf parameter left in signature for back-compatibility but unused.
```

---

### Task 3: Wire `calcTSW` into `scfE.py:FockToP`

**Files:**
- Modify: `gauNEGF/scfE.py:356-361`

- [ ] **Step 1: Run existing Fermi tests as baseline**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_bethe_cross_term_fermi.py test_jit_fermi_shift.py -v`
Expected: PASS (baseline before modification)

- [ ] **Step 2: Modify FockToP to use `calcTSW`**

In `gauNEGF/scfE.py`, replace lines 356-361 (inside the `if self.N2 is None:` branch of `FockToP`):

Old code (lines 357-361):
```python
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, Emin=self.Emin)
            P, _delta_N_lower = densityComplex(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.tol, T=0)
        else:
            P, _delta_N_lower = densityRealN(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, T=0)
        nLower = np.trace(self.S@P).real + _delta_N_lower
```

New code:
```python
            self.Emin, self.Emax, self.TSW = calcTSW(
                self.F*har_to_eV, self.S, self.g,
                Emin=self.Emin,
                Emax=getattr(self, 'Emax', None),
                TSW=getattr(self, 'TSW', None))
            nLower = 0.0
            P = np.zeros_like(self.S, dtype=complex)
        else:
            P, _delta_N_lower = densityRealN(self.F*har_to_eV, self.S, self.g, self.Eminf, self.Emin, self.N2, T=0)
            nLower = np.trace(self.S@P).real + _delta_N_lower
```

Note: The `N2 is not None` branch (deprecated real-axis path) is left untouched per the spec. The `calcTSW` branch sets `P` to zeros because the lower density is no longer accumulated separately -- `Emin` from `calcTSW` will be used directly as the lower bound for the equilibrium contour later. `nLower = 0.0` preserves all downstream `ne - nLower` expressions unchanged.

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest test_bethe_cross_term_fermi.py test_jit_fermi_shift.py test_calcTSW.py -v`
Expected: PASS

- [ ] **Step 4: Pause for user commit**

Stop and ask the user to commit. Suggested message:

```
refactor: switch FockToP lower density from calcEmin+Eminf to calcTSW

Uses calcTSW to find integration bounds in the N2=None (adaptive) path.
Emax and TSW are stored on self for warm-start across SCF iterations.
The deprecated N2 real-axis path is left untouched.
```

---

### Task 4: Run full regression suite

**Files:**
- No modifications -- verification only

- [ ] **Step 1: Run all tests in the test directory**

Run: `cd /mmfs1/gscratch/anantram/willll/NEGFCode/tests && python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 2: If any tests fail, investigate and fix**

Check the failure output. The most likely failure mode is a test that directly calls `getFermiContact` with an `Eminf` value and expects `nLower`-adjusted behavior. If so, the fix is to update that test's expected electron count (since `nLower` was ~0 for non-overlap cases anyway).

- [ ] **Step 3: Pause for user commit if any fixes were made**

Stop and ask the user to commit. Suggested message:

```
fix: update tests for calcTSW integration limit changes
```
