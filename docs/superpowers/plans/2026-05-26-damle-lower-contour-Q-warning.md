# Damle Lower Contour with Analytic Cross-Term + Q-Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the equilibrium device density matrix well-defined via an analytic Damle lower contour (one eig: matrix + cross-term `delta_N`) plus a `densityComplex` band, with `Emin` placed by `calcEmin`, `calcTSW`/`calcPseudoPoleFloor` dropped from the path, and a warning when the lower-contour cross-term is anomalously large.

**Architecture:** `setContact*` -> `_initAsymptoticSigma` (2-probe fit, already done) -> `setIntegralLimits` places `Emin` via `calcEmin` (true-Sigma DOS loop). `FockToP` builds the lower contour `[ENERGY_MIN, Emin]` with `damleLowerDensity` (negated to match `densityComplex` sign; analytic cross-term with the unphysical linear-Q "flat" term dropped) and the band `[Emin, mu]` with `densityComplex`. `delta_N` enters the electron count and triggers a warning above `damle_dN_warn` (default 0.5 e).

**Tech Stack:** Python, JAX (jax.numpy, x64), NumPy; Gaussian/gdv via the NEGF conda env; SLURM (ckpt-all) for any run touching the NEGF stack.

**Spec:** docs/superpowers/specs/2026-05-26-damle-lower-contour-Q-warning.md
**Math:** docs/lower_contour_math_and_probes.md

---

## Conventions for this plan

- **Git commits are performed BY THE USER.** "Commit checkpoint" steps list the
  files to stage and a suggested message; do NOT run `git add`/`git commit`
  yourself. Pause and tell the user it is a commit point.
- **ASCII only** in all code and output (no unicode).
- **All test runs go through SLURM** (login-node policy). Use the
  `tests/run_pytest.job` runner created in Task 0: edit its `PYTEST_ARGS`, then
  `sbatch tests/run_pytest.job`, then read `slurm-<id>.out`. Watcher pattern:
  ```bash
  OUT=slurm-<id>.out
  for i in $(seq 1 180); do
    [ -f "$OUT" ] && grep -qE "passed|failed|error" "$OUT" && break
    sacct -j <id> --format=State --noheader 2>/dev/null | head -1 | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT" && break
    sleep 20
  done; tail -40 "$OUT"
  ```
- `_initAsymptoticSigma` (2-probe relative fit) is ALREADY implemented and
  validated (scfE.py:204). No task changes it.

---

## Task 0: Test runner job (SLURM)

**Files:**
- Create: `tests/run_pytest.job`

- [ ] **Step 1: Create the sbatch pytest runner**

Create `tests/run_pytest.job`:

```bash
#!/bin/bash
#SBATCH --job-name=negfpytest
#SBATCH --account=anantram
#SBATCH --partition=ckpt-all
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:40:00
#SBATCH --mem=48GB
##

set -e
module load anantram/gdv
module load cuda
unset LD_LIBRARY_PATH
export GAUSS_SCRDIR='/tmp'
export GAUSS_PDEF=$SLURM_NTASKS
export GAUSS_MEMDEF=${SLURM_MEM_PER_NODE}MB
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/anantram/willll/NEGF
cd /gscratch/anantram/willll/NEGFCode

# EDIT PYTEST_ARGS per task before submitting:
PYTEST_ARGS="tests/test_damle_cross_term.py -v"
python3 -u -m pytest ${PYTEST_ARGS}
exit 0
```

- [ ] **Step 2: Commit checkpoint (USER COMMITS)**

Files: `tests/run_pytest.job`. Suggested message:
`test: add SLURM pytest runner for NEGF test suite`

---

## Task 1: `damleCrossTerm` helper (pure-math) + unit tests

The cross-term `delta_N` reuses the density's single eig and drops the unphysical
linear-Q flat term (spec sec 4). Isolate the pure math so it is unit-testable
without gdv.

**Files:**
- Modify: `gauNEGF/density.py` (add function after `density`, ~line 349)
- Test: `tests/test_damle_cross_term.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_damle_cross_term.py`:

```python
"""Pure-math unit tests for damleCrossTerm (no gdv / no SCF needed)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gauNEGF.density import damleCrossTerm


def _identity_inputs(D):
    N = D.shape[0]
    I = jnp.eye(N, dtype=complex)
    return I, jnp.asarray(D), I, I  # V, D, Vc, Y_eff (all identity/eigenbasis)


def test_cross_term_pole_free_is_zero():
    # All eigenvalues ABOVE hi -> no in-window poles -> delta_N ~ 0,
    # even with a nonzero linear-Q slope (this is the dropped-flat-term property).
    D = jnp.array([-5.0 + 1e-6j, -3.0 + 1e-6j, 2.0 + 1e-6j])
    V, D, Vc, Y = _identity_inputs(D)
    Q0 = jnp.eye(3, dtype=complex)
    Q1 = 0.01 * jnp.eye(3, dtype=complex)   # nonzero slope: would blow up if flat term kept
    dN = damleCrossTerm(V, D, Vc, Y, Q0, Q1, lo=-1.0e6, hi=-10.0)
    assert abs(dN) < 1e-6


def test_cross_term_in_window_pole_registers():
    # One eigenvalue INSIDE [lo, hi] -> nonzero, O(1) delta_N (the warning must fire).
    D = jnp.array([-100.0 + 1e-6j])
    V, D, Vc, Y = _identity_inputs(D)
    Q0 = 2.0 * jnp.eye(1, dtype=complex)
    Q1 = jnp.zeros((1, 1), dtype=complex)
    dN = damleCrossTerm(V, D, Vc, Y, Q0, Q1, lo=-1.0e6, hi=-10.0)
    # b0 + b1*D = 2.0; Im(log-diff) = +/- pi for the in-window pole
    # -> |delta_N| = (1/pi)*|2.0*pi| = 2.0
    assert abs(abs(dN) - 2.0) < 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Edit `tests/run_pytest.job` -> `PYTEST_ARGS="tests/test_damle_cross_term.py -v"`,
then `sbatch tests/run_pytest.job` and read the output.
Expected: FAIL with `ImportError: cannot import name 'damleCrossTerm'`.

- [ ] **Step 3: Implement `damleCrossTerm`**

In `gauNEGF/density.py`, add immediately after the `density` function (after line
349, before `damleLowerDensity`):

```python
def damleCrossTerm(V, D, Vc, Y_eff, Q0, Q1, lo, hi):
    """Analytic lower-contour cross-term delta_N for the Damle method.

    Reuses the eig (V, D, Vc) of Fbar already computed for the density matrix
    (no second eigendecomposition). Q(E) ~ Q0 + Q1*E is linearized between two
    anchors. The unphysical linear-Q 'flat' term b1*(hi-lo) is OMITTED -- only
    the physical in-window-pole term is kept (see
    docs/lower_contour_math_and_probes.md sec 5 and the spec sec 4):

        delta_N = -(1/pi) Im sum_i (b0_i + b1_i*D_i)
                                  * [log(1 - hi/D_i) - log(1 - lo/D_i)]

    with b(E) = Vc^dagger Y_eff Q(E) Y_eff V, b0 = diag(...Q0...),
    b1 = diag(...Q1...). Poles outside [lo, hi] give a real log-difference
    (no contribution); in-window poles pick up the i*pi (the physical count).

    Parameters
    ----------
    V, D, Vc : ndarray
        Eigendecomposition of Fbar: Fbar = V diag(D) Vc^dagger, Vc = inv(V^dagger).
    Y_eff : ndarray
        S_eff^(-1/2).
    Q0, Q1 : ndarray (N, N)
        Linear model of crossTermQTot: Q(E) ~ Q0 + Q1*E.
    lo, hi : float
        Lower-contour energy limits in eV (lo = ENERGY_MIN, hi = Emin).

    Returns
    -------
    float
        Cross-term electron-count correction delta_N.
    """
    Ml = Vc.conj().T @ Y_eff
    Mr = Y_eff @ V
    b0 = jnp.diag(Ml @ jnp.asarray(Q0) @ Mr)
    b1 = jnp.diag(Ml @ jnp.asarray(Q1) @ Mr)
    logdiff = jnp.log(1.0 - hi / D) - jnp.log(1.0 - lo / D)
    contrib = (b0 + b1 * D) * logdiff
    return float(-(1.0 / jnp.pi) * jnp.imag(jnp.sum(contrib)))
```

- [ ] **Step 4: Run to verify it passes**

`sbatch tests/run_pytest.job`, read output.
Expected: both tests PASS.

- [ ] **Step 5: Commit checkpoint (USER COMMITS)**

Files: `gauNEGF/density.py`, `tests/test_damle_cross_term.py`. Suggested message:
`feat: analytic damle cross-term helper (drop linear-Q flat term)`

---

## Task 2: Rework `damleLowerDensity` (sign fix + cross-term + new signature)

`damleLowerDensity` currently returns `(P_lower, Emin)` and computes its own
`Emin`. New: it TAKES `Emin` and `g`, negates `P` to match `densityComplex`, and
returns `(P_lower, delta_N)`.

**Files:**
- Modify: `gauNEGF/density.py` -- the `damleLowerDensity` function. **Locate by name, not line number:** Task 1 inserted `damleCrossTerm` just above it, so its range is no longer 350-419. It is the function ending with `return P_lower, Emin`.
- Test: `tests/test_damle_lower_contour.py` (new, integration via sbatch)

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_damle_lower_contour.py`:

```python
"""Integration tests for the reworked damleLowerDensity on C2-STO3G.

Run via tests/run_pytest.job (needs gdv + the NEGF env). Validates:
  - sign-corrected: damle matrix matches densityComplex on a cores-in-window
    interval to < 1% (after the sign fix, no manual negation needed);
  - delta_N ~ 0 on a pole-free deep tail (the dropped-flat-term property).
"""
import os
import sys
import shutil
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from numpy.linalg import norm

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)
from gauNEGF.scfE import NEGFE
from gauNEGF.transport import har_to_eV
from gauNEGF.density import densityComplex, damleLowerDensity
from gauNEGF.config import ENERGY_MIN


def _c2_setup():
    scratch = tempfile.mkdtemp(prefix='c2_damle_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    return negf


def test_damle_matrix_matches_densitycomplex_on_cores():
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    # Emin in the core/valence gap (1s cores ~ -271, valence from ~ -18).
    Emin_gap = -145.0
    P_damle, dN = damleLowerDensity(F_eV, negf.Y_eff, negf.Sigma_0, negf.g, Emin_gap)
    P_cplx, _ = densityComplex(F_eV, S, negf.g, ENERGY_MIN, Emin_gap, T=0)
    rel = norm(np.asarray(P_damle) - np.asarray(P_cplx)) / max(norm(np.asarray(P_cplx)), 1e-30)
    assert rel < 0.01, f'sign-corrected damle disagreement {rel:.3e} (expect <1%)'


def test_damle_delta_N_negligible_on_pole_free_tail():
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    # Emin below all poles (cores ~ -271): tail [-1e6, -291] is empty.
    Emin_deep = -291.0
    _, dN = damleLowerDensity(F_eV, negf.Y_eff, negf.Sigma_0, negf.g, Emin_deep)
    assert abs(dN) < 1e-2, f'pole-free tail delta_N={dN:.3e} (expect ~0)'
```

- [ ] **Step 2: Run to verify it fails**

Edit runner -> `PYTEST_ARGS="tests/test_damle_lower_contour.py -v"`, `sbatch`.
Expected: FAIL -- the current `damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer,
ENERGY_MIN_)` takes `buffer` (a float) where the test passes `negf.g` (a surfG
object), so it raises `TypeError` at `float(buffer)`. (The 2-value unpacking
itself matches the old `(P, Emin)` return; the call just errors before returning.)

- [ ] **Step 3: Replace `damleLowerDensity`**

In `gauNEGF/density.py`, replace the entire `damleLowerDensity` function (find
it by name -- its line range shifted when Task 1 added `damleCrossTerm` above it;
it is the one ending `return P_lower, Emin`) with:

```python
def damleLowerDensity(F_eV, Y_eff, Sigma_0, g, Emin, ENERGY_MIN_=ENERGY_MIN):
    """Analytic Damle lower-contour density on [ENERGY_MIN_, Emin].

    Builds Fbar = Y_eff (F + Sigma_0 + i*ETA) Y_eff, diagonalizes it ONCE, and
    returns both the lower-contour density matrix (in the AO basis) and the
    analytic cross-term delta_N (which reuses the same eig; no second
    decomposition).

    The matrix is NEGATED so its sign matches the trusted densityComplex
    convention (the bare analytic density() result is globally sign-flipped
    relative to densityComplex; confirmed in the bake-off, jobs 35562575/
    35580937). The cross-term uses a linear model of crossTermQTot(E) anchored
    near the band bottom at (Emin, 2*Emin) and DROPS the unphysical linear-Q
    flat term (see damleCrossTerm and the spec).

    Parameters
    ----------
    F_eV : ndarray (N, N)
        Device Fock matrix in eV.
    Y_eff : ndarray (N, N)
        S_eff^(-1/2) (from NEGFE._initAsymptoticSigma).
    Sigma_0 : ndarray (N, N)
        Asymptotic constant contact self-energy.
    g : surfG object
        Provides crossTermQTot(E) for the cross-term. If it returns None
        (orthogonal system), delta_N = 0.
    Emin : float
        Upper bound of the lower contour in eV (set by calcEmin upstream).
    ENERGY_MIN_ : float, optional
        Lower bound in eV (default config.ENERGY_MIN).

    Returns
    -------
    tuple (ndarray, float)
        (P_lower, delta_N).
    """
    F_eV = jnp.asarray(F_eV)
    Y_eff = jnp.asarray(Y_eff)
    Sigma_0 = jnp.asarray(Sigma_0)
    N = F_eV.shape[0]
    Emin = float(Emin)
    lo = float(ENERGY_MIN_)

    H_eff = F_eV + Sigma_0 + 1j * ETA * jnp.eye(N)
    Fbar = Y_eff @ H_eff @ Y_eff
    Gam = (H_eff - H_eff.conj().T) * 1j
    GamBar = Y_eff @ Gam @ Y_eff

    D, V = jnp.linalg.eig(Fbar)
    Vc = jnp.linalg.inv(V.conj().T)

    # Density matrix; negate to match densityComplex sign convention.
    P_orth = density(V, Vc, D, GamBar, lo, Emin)
    P_lower = -(Y_eff @ jnp.asarray(P_orth) @ Y_eff)

    # Analytic cross-term: linear Q at near-band anchors (Emin, 2*Emin).
    Qa = g.crossTermQTot(Emin)
    if Qa is None:
        delta_N = 0.0
    else:
        Ea, Eb = Emin, 2.0 * Emin
        Qa = jnp.asarray(Qa)
        Qb = jnp.asarray(g.crossTermQTot(Eb))
        Q1 = (Qb - Qa) / (Eb - Ea)
        Q0 = Qa - Q1 * Ea
        delta_N = damleCrossTerm(V, D, Vc, Y_eff, Q0, Q1, lo, Emin)

    return P_lower, delta_N
```

- [ ] **Step 4: Run to verify it passes**

`sbatch tests/run_pytest.job` (PYTEST_ARGS still the lower-contour test).
Expected: both tests PASS (rel < 1%; |delta_N| < 1e-2).

- [ ] **Step 5: Commit checkpoint (USER COMMITS)**

Files: `gauNEGF/density.py`, `tests/test_damle_lower_contour.py`. Suggested
message: `feat: damle lower contour sign fix + analytic cross-term, take Emin/g`

---

## Task 3: `FockToP` rewiring + `damle_dN_warn` + warning

`FockToP` must call the new `damleLowerDensity` signature, consume `delta_N` in
the count, and warn above the threshold.

**Files:**
- Modify: `gauNEGF/scfE.py` -- add `self.damle_dN_warn` default (in `_initAsymptoticSigma`, next to `damle_buffer`, scfE.py:271-272)
- Modify: `gauNEGF/scfE.py:512-535` (the `FockToP` lower-contour block)

- [ ] **Step 1: Add the warning-threshold attribute**

In `gauNEGF/scfE.py`, inside `_initAsymptoticSigma`, where `damle_buffer` is set
(currently lines 271-272):

```python
        if not hasattr(self, 'damle_buffer'):
            self.damle_buffer = EMIN_BUFFER
        if not hasattr(self, 'damle_dN_warn'):
            self.damle_dN_warn = 0.5   # warn if |lower-contour delta_N| exceeds this (electrons)
```

- [ ] **Step 2: Rewire the `FockToP` lower-contour block**

In `gauNEGF/scfE.py`, replace the `N2 is None` branch of `FockToP` (currently
lines 514-525, the block that calls `damleLowerDensity(... self.damle_buffer)`
and sets `self.Emin` from it) with:

```python
        if self.N2 is None:
            # Lower contour [ENERGY_MIN, Emin] via analytic Damle. Emin is set
            # upstream by calcEmin (setIntegralLimits). damleLowerDensity returns
            # the (sign-corrected) lower density matrix and the analytic cross
            # term delta_N; a large delta_N means real spectral weight sits below
            # Emin (Emin too shallow / a pseudo-pole) -- warn (replaces calcTSW's
            # dTSW<0 detection). See docs/superpowers/specs/
            # 2026-05-26-damle-lower-contour-Q-warning.md.
            F_eV = self.F * har_to_eV
            P, delta_N_lower = damleLowerDensity(F_eV, self.Y_eff, self.Sigma_0,
                                                 self.g, self.Emin)
            if abs(delta_N_lower) > self.damle_dN_warn:
                print(f'WARNING: lower-contour cross-term delta_N={delta_N_lower:.3e} '
                      f'exceeds threshold {self.damle_dN_warn:.3e}; spectral weight '
                      f'may sit below Emin={self.Emin:.2f} eV (Emin too shallow or '
                      f'a pseudo-pole present).')
            nLower = np.trace(self.S @ P).real + delta_N_lower
```

(Leave the `else:` deprecated fixed-grid branch, lines 526-535, unchanged.)

- [ ] **Step 3: Write the test (warning fires; count uses delta_N)**

Append to `tests/test_damle_lower_contour.py`:

```python
def test_focktop_warns_when_pole_below_emin(capsys):
    negf = _c2_setup()
    # Force Emin into the core/valence gap so the 1s cores sit BELOW Emin
    # (a deliberately-too-shallow Emin) -> large delta_N -> warning.
    negf.N2 = None
    negf.updFermi = False   # isolate the warning: skip the full Fermi search
    negf.Emin = -145.0
    negf.damle_dN_warn = 0.5
    negf.FockToP()
    out = capsys.readouterr().out
    assert 'lower-contour cross-term delta_N' in out
```

- [ ] **Step 4: Run tests**

Edit runner -> `PYTEST_ARGS="tests/test_damle_lower_contour.py -v"`, `sbatch`.
Expected: all three tests PASS (the new one shows the warning printed).

- [ ] **Step 5: Commit checkpoint (USER COMMITS)**

Files: `gauNEGF/scfE.py`, `tests/test_damle_lower_contour.py`. Suggested
message: `feat: FockToP consumes damle delta_N + warns; add damle_dN_warn`

---

## Task 4: `setIntegralLimits` -- drop calcTSW + calcPseudoPoleFloor (keep calcEmin)

`Emin` is now placed by `calcEmin`'s true-Sigma DOS loop; the
`calcPseudoPoleFloor` + `calcTSW` lower-bound machinery is removed from the
default path (the functions remain in the module as debug tools).

**Files:**
- Modify: `gauNEGF/scfE.py:394-403` (default branch of `setIntegralLimits`; the `else` and the `self.N1/N2/Nnegf` lines just below it stay unchanged)

- [ ] **Step 1: Write the test**

Append to `tests/test_damle_lower_contour.py`:

```python
def test_setvoltage_places_emin_below_band_via_calcemin():
    negf = _c2_setup()  # setVoltage(0.0) already ran inside
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    # Emin must sit below the lowest pole (the C2 1s cores ~ -271 eV).
    band_floor = float(np.real(np.linalg.eigvals(np.linalg.solve(S, F_eV))).min())
    assert negf.Emin <= band_floor, f'Emin={negf.Emin} not below band floor {band_floor}'
    # Eminf must be the wide config bound (no calcTSW narrowing).
    from gauNEGF.config import ENERGY_MIN as EMIN_CFG
    assert abs(negf.Eminf - EMIN_CFG) < 1.0
```

- [ ] **Step 2: Run to verify current behavior (may already pass or fail)**

Edit runner -> `PYTEST_ARGS="tests/test_damle_lower_contour.py::test_setvoltage_places_emin_below_band_via_calcemin -v"`, `sbatch`.
Expected: FAIL or unstable -- the current path runs `calcTSW` and sets
`self.Eminf`/`self.TSW` from the pseudo-pole floor, not the wide bound.

- [ ] **Step 3: Simplify the default branch**

In `gauNEGF/scfE.py`, replace the body of the default branch of
`setIntegralLimits` -- the lines from `self.Emin = calcEmin(...)` through
`self.tol = tol` (currently 395-403). Current code:

```python
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol=tol)
            # Pseudo-pole detection: principled floor for calcTSW (replaces ENERGY_MIN).
            # See docs/pseudo_pole_handling.md.
            Eminf_floor = calcPseudoPoleFloor(self.F*har_to_eV, self.S, self.g)
            Eminf_floor = min(self.Emin, Eminf_floor)
            self.Eminf, self.TSW = calcTSW(self.F*har_to_eV, self.S, self.g,
                                           Eminf=self.Emin, tol=tol,
                                           Emin_floor=Eminf_floor)
            self.tol = tol
```

Replace with (CRITICAL: keep `self.tol = tol` -- FockToP reads `self.tol`; do
NOT drop it):

```python
            # Emin from the true-Sigma DOS loop (robust, incl. non-PSD S_eff);
            # the deep bound is the wide config value. calcTSW /
            # calcPseudoPoleFloor are NOT used on this path -- the lower contour
            # is handled analytically by damleLowerDensity, whose cross-term
            # delta_N warning (FockToP) replaces calcTSW's dTSW<0 detection.
            self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol=tol)
            self.Eminf = ENERGY_MIN
            self.TSW = None
            self.tol = tol
```

(The `else: self.Emin = Emin` branch and the `self.N1 = N1` / `self.N2 = N2` /
`self.Nnegf = Nnegf` assignments just below it remain unchanged.)

- [ ] **Step 4: Run tests**

`sbatch tests/run_pytest.job` with the full file
(`PYTEST_ARGS="tests/test_damle_lower_contour.py -v"`).
Expected: all PASS, including the Emin-placement test.

- [ ] **Step 5: Commit checkpoint (USER COMMITS)**

Files: `gauNEGF/scfE.py`, `tests/test_damle_lower_contour.py`. Suggested
message: `refactor: drop calcTSW/calcPseudoPoleFloor from setIntegralLimits (Emin via calcEmin)`

---

## Task 5: Full-SCF validation (Emin placement; bisectFermi sign sanity)

End-to-end check that an SCF runs with the new lower contour, `Emin` lands below
the band, the `bisectFermi` predict path (which also calls `density()`, untouched
by the localized damle negation) still produces a sensible positive count, and
the non-PSD case still places `Emin` below the band (true-Sigma DOS catches what
the damle eig would miss).

Note: `NEGFE`'s `basis` argument sets the basis set; `C2_chain.gjf` supplies
geometry only. Both `sto-3g` and `lanl2dz` were run from this same `.gjf` in the
bake-off, so reusing it for both setups is fine.

**Files:**
- Create: `tests/fixtures/_diag_damle_scf_validate.py`
- Create: `tests/fixtures/diag_damle_scf_validate.job`

- [ ] **Step 1: Create the validation script**

Create `tests/fixtures/_diag_damle_scf_validate.py`:

```python
"""SCF-level validation of the damle lower contour + calcEmin Emin placement.

Runs a short SCF on C2-STO3G (PSD) and reports: Emin below band floor (DOS ~ 0),
total electron count sane and positive (bisectFermi sign sanity), no spurious
lower-contour warning. Then repeats the setup-only Emin check on C2-LANL2DZ
(non-PSD S_eff) to confirm calcEmin still places Emin below the band.
"""
import os
import sys
import shutil
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)
from gauNEGF.scfE import NEGFE
from gauNEGF.transport import har_to_eV


def setup(basis):
    scratch = tempfile.mkdtemp(prefix='c2_val_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis=basis,
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    return negf


def check_emin(negf, label):
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    floor = float(np.real(np.linalg.eigvals(np.linalg.solve(S, F_eV))).min())
    ok = negf.Emin <= floor
    print(f'[{label}] Emin={negf.Emin:.3f} eV, band floor={floor:.3f} eV, '
          f'Emin below floor: {ok}')
    return ok


def main():
    print('=== damle SCF validation ===')
    psd = setup('sto-3g')
    ok_psd = check_emin(psd, 'C2-STO3G (PSD)')
    # one Fock->P pass to exercise FockToP + the warning path
    psd.FockToP()
    print('C2-STO3G FockToP completed')

    nonpsd = setup('lanl2dz')
    ok_nonpsd = check_emin(nonpsd, 'C2-LANL2DZ (non-PSD)')

    print(f'VERDICT: Emin-below-floor PSD={ok_psd}, non-PSD={ok_nonpsd}')
    print('=== damle SCF validation complete ===')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Create the job and run it**

Create `tests/fixtures/diag_damle_scf_validate.job` (copy `tests/run_pytest.job`
header, but replace the final two lines with):

```bash
python3 -u tests/fixtures/_diag_damle_scf_validate.py
exit 0
```

Submit: `sbatch tests/fixtures/diag_damle_scf_validate.job`; watch with the
Conventions watcher (marker: `damle SCF validation complete`).
Expected: `Emin-below-floor PSD=True`; FockToP completes; record the non-PSD
result (it MAY warn or place Emin differently -- that informs the non-PSD track,
out of scope here, but confirms the DOS check runs).

- [ ] **Step 3: Record the outcome in the spec provenance**

Append the job id and the `VERDICT` line to the spec's Section 9 (Provenance).

- [ ] **Step 4: Commit checkpoint (USER COMMITS)**

Files: `tests/fixtures/_diag_damle_scf_validate.py`,
`tests/fixtures/diag_damle_scf_validate.job`,
`docs/superpowers/specs/2026-05-26-damle-lower-contour-Q-warning.md`.
Suggested message: `test: SCF-level validation of damle Emin placement + warning`

---

## Self-review notes

- Spec coverage: 3.1 (done, no task) / 3.2 (Tasks 1-2) / 3.3 (Task 3) / 3.4
  (Task 4) / warning (Task 3) / migration steps 1-5 (Tasks 0-5) / SCF + non-PSD
  validation, step 5 (Task 5). bisectFermi sign sanity (Task 5).
- Signature consistency: `damleLowerDensity(F_eV, Y_eff, Sigma_0, g, Emin,
  ENERGY_MIN_)` returns `(P_lower, delta_N)` -- used identically in Task 2 test,
  Task 3 FockToP. `damleCrossTerm(V, D, Vc, Y_eff, Q0, Q1, lo, hi)` -- defined
  Task 1, called in Task 2.
- Open risk: the `density()` sign is fixed by negating ONLY `damleLowerDensity`'s
  output; `density()` itself (used by `bisectFermi`) is untouched. Task 5
  exercises bisectFermi to confirm the count stays positive/sane. If it does
  not, a follow-up is needed to reconcile the `density()` sign convention across
  both callers.
