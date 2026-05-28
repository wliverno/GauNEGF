# Damle Lower Contour: Analytic Deep-Tail Integration

Date: 2026-05-22
Status: Design (implemented Phases 0-3; lower-half Emin policy in debug)

## 1. Problem

The density integral is split into two contours that meet at `Emin`:

  - UPPER `[Emin, mu]`: `densityComplex` with the full energy-dependent
    `Sigma(E)`. This has historically worked and is NOT modified by this
    spec. Any bug is in the lower contour, never here.
  - LOWER `[ENERGY_MIN, Emin]`: the deep tail. This is what we replace.

The old lower-contour path, `densityComplex(F, S, g, Eminf, Emin)`,
misbehaves for E far below the band, where the non-orthogonal contact
self-energy grows linearly (`Sigma(E) ~ Sigma_0 + E*X`) and the device
Green function looks like a generalized eigenproblem in an indefinite
metric. Production stacked three workarounds to suppress the symptom:

  - `calcPseudoPoleFloor`: pick a floor and refuse to integrate past it.
  - `calcTSW`: iteratively search for the deepest Eminf at which Total
    Spectral Weight is stable.
  - `Eminf_floor` clamping in `FockToP`/`setIntegralLimits`.

Each layer has been patched repeatedly; the lower-contour math underneath
was never right. This spec replaces all three with a single principled
scheme: analytic Damle integration of the lower contour using the
effective overlap `S_eff = S - X` and the asymptotic constant `Sigma_0`.
The upper contour is left exactly as-is.

## 2. Math

### 2.1 Asymptotic Sigma

For E -> -inf along the real axis, the non-orthogonal contact self-energy
satisfies

    Sigma(E) = Sigma_0 + E * X + o(1)                            (1)

where Sigma_0 and X are E-independent NxN matrices determined entirely
by contact geometry, basis, and contact-Hamiltonian parameters. For
orthogonal contacts X = 0 and Sigma is genuinely bounded; for
non-orthogonal contacts X is nonzero and Sigma grows linearly.

### 2.2 Effective Green function

Substituting (1) into the device retarded Green function:

    G(E) = [E*S - F - Sigma(E)]^-1
         ~ [E*(S - X) - (F + Sigma_0)]^-1
         = [E*S_eff - H_eff]^-1                                  (2)

with

    H_eff = F + Sigma_0                                          (3)
    S_eff = S - X                                                (4)

This is the same algebraic form as a non-interacting Green function
with effective Hamiltonian H_eff and effective overlap S_eff. The
energy dependence of Sigma has been absorbed into the renormalized
overlap. The Damle analytic density formula, derived under the
assumption of energy-independent Sigma, becomes exact in this
representation in the asymptotic regime.

### 2.3 Complex-symmetric, non-Hermitian S_eff

S_eff is in general NEITHER Hermitian NOR positive definite. The retarded
contact self-energy has the form Sigma = A @ g_surf @ A^dagger with g_surf
complex-symmetric (the inverse of (E+i*eta)*S_c - H_c, both real-symmetric),
so Sigma -- and hence X_asymp and `S_eff = S - X_asymp` -- is
COMPLEX-SYMMETRIC (S_eff = S_eff^T) but NOT Hermitian: it carries broadening,
Gamma = i(Sigma - Sigma^dagger) != 0.

Empirically on C2 LANL2DZ: ||Im X||/||X|| = 0.51, ||X - X^dag||/||X|| = 1.03
(fully non-Hermitian), ||X - X^T||/||X|| = 3e-16 (exactly complex-symmetric).

Consequence for the effective inverse square root

    Y_eff = S_eff^(-1/2)                                         (5)

it MUST be computed with a general (non-symmetric) eigendecomposition,
`Y_eff = V @ diag(D^(-1/2)) @ V^(-1)` where `S_eff = V @ diag(D) @ V^(-1)`
(see `utils.inv_sqrt_general`). This satisfies `Y_eff @ S_eff @ Y_eff = I` to
machine precision for any diagonalizable S_eff. Using eigh (which assumes a
Hermitian matrix and uses one triangle) is WRONG here: on C2 LANL2DZ the
eigh-based Y_eff gives ||Y S_eff Y - I|| = 4.26 versus 9e-15 for general eig --
a complete failure, not a small approximation. The physical lower-contour
density is independent of the sqrt branch, since
G(E) = Y(EI - Fbar)^(-1)Y reduces to [E*S_eff - H_eff]^(-1) for any Y with
Y @ Y = S_eff^(-1).

### 2.4 Lower contour density

For the lower contour `[ENERGY_MIN, Emin]` where (1) holds to <1%
accuracy (verified empirically: ||Sigma(Emin) - (Sigma_0 + Emin*X)||_F
/ ||Sigma(Emin)||_F = 0.6% at Emin = -325 eV for C2 LANL2DZ):

    Fbar    = Y_eff @ H_eff @ Y_eff
    GamBar  = Y_eff @ (Sigma_0 - Sigma_0^H) * 1j @ Y_eff
    D, V    = eig(Fbar)
    Vc      = inv(V^H)
    P_orth  = density(V, Vc, D, GamBar, ENERGY_MIN, Emin)
    P_lower = Y_eff @ P_orth @ Y_eff                             (6)

The `density(...)` function is the existing Damle analytic-integral
implementation in `gauNEGF/density.py`. The test script already
exercises this function with complex Y_eff (indefinite S_eff case)
and produces finite output via `np.emath.sqrt` machinery, so no
modification is anticipated. Implementation must spot-check that
the function tolerates complex D (eigenvalues of Fbar with non-zero
imaginary part) without internal real-cast bugs.

### 2.5 Emin policy (split point) -- validation pending

The split point between Damle (lower) and densityComplex (upper) is

    Emin = min(D.real) - buffer                                  (7)

over ALL eigenvalues of Fbar (D from eig(Fbar)), real and complex. With a
CORRECT Y_eff (section 2.3), Fbar = Y_eff @ H_eff @ Y_eff has as its
eigenvalues the genuine generalized eigenvalues of (F + Sigma_0, S_eff) -- the
poles of the asymptotic Green's function [E*S_eff - H_eff]^(-1), i.e. the
physical pole locations in the deep tail. So (7) places Emin below the deepest
such pole:

  - `[ENERGY_MIN, Emin]` is the deep asymptotic tail; Damle handles it
    analytically (the asymptotic Sigma form (1) holds there).
  - `[Emin, mu]` is handled by standard `densityComplex` with the full
    energy-dependent Sigma(E). This upper contour has historically worked and
    is NOT modified here.

The total density is correct for any Emin that (a) lies where (1) holds and
(b) is stable across SCF cycles.

IMPORTANT (validation pending, see the lower-half debug plan): the earlier
C2 LANL2DZ divergence was diagnosed against an Emin computed from a GARBAGE
eigh-based Y_eff (section 2.3 -- the old Y_eff had ||Y S_eff Y - I|| = 4.26,
so Fbar and its eigenvalues were meaningless). The "Emin is the bug" diagnosis
therefore rests on invalid data. With the corrected Y_eff, policy (7) may be
fine as-is. Two residual risks to confirm by the DEFINITIVE C2 LANL2DZ SCF run
(no presumed Fermi value -- the contacted Fermi is unknown a priori; success =
the SCF converges to a stable self-consistent Fermi with a sensible electron
count and Hermitian P): (a) S_eff is indefinite/near-singular, so a near-zero
S_eff direction could throw a large-magnitude generalized eigenvalue and a
pathological Emin; (b) cycle-to-cycle Emin instability. If the SCF prints a
stable, sane Emin and converges, NO Emin change is made. Only if it does not
do we switch Emin to a stable integrability-based split (e.g. the (F,S) band
bottom) -- the single contingency change.

Buffer default: 20 eV. Exposed as `self.damle_buffer`, initialized from
`EMIN_BUFFER` in `gauNEGF/config.py`, tunable via config. Replaces the
hardcoded buffer that used to live in scf.py.

### 2.6 Cache freshness under mu changes

`surfG.setF(F, muL, muR)` applies a rigid-band Fermi shift per contact:
`dFermi_i = mu_i - fermi0_i`, and the surface Green's function is
evaluated at `g_surf(E - dFermi_i)` while the device-contact coupling
`t = E*stau - tau` stays at raw E. Asymptotically (g_surf ~ G_1/E +
O(1/E^2)) a Taylor expansion gives:

    sigma_i_after(E) = sigma_i_orig(E) + dFermi_i * X_i + O(1/E)     (8)

where X_i = stau_i @ G_1 @ stau_i.T is the per-contact slope of the
asymptotic expansion. Implications:

  - X_asymp_total = sum_i X_i is INVARIANT under any mu shift.
  - S_eff = S - X_asymp is INVARIANT.
  - Y_eff = S_eff^(-1/2) is INVARIANT.
  - Sigma_0_total shifts by sum_i X_i * dFermi_i. Per-contact X_i
    is nonzero only on the [inds_i, inds_i] block, so the shift is
    naturally block-diagonal.

Consequence: any code path that changes mu through setF
(notably `NEGFE.setVoltage`) MUST trigger a refit of `Sigma_0` before
the next `damleLowerDensity` call, or the lower contour integrates
with a stale Sigma_0 against the current contact state. The simplest
fix is to re-call `_initAsymptoticSigma` at the end of `setVoltage`;
cost is dominated by three sigmaTot probes which converge quickly
well below the band. See `tests/test_setF_mu_invariance.py` for the
empirical verification (linear theory matches at <1% rel err) and
the failing-then-passing TDD test that drives the setVoltage refit.

## 3. Components

### 3.1 New: `damleLowerDensity` in `gauNEGF/density.py`

Note: implementation adds the standard NEGF retarded-Green's-function
regularization `+i*ETA*I` to `H_eff` (where `ETA` is the canonical
broadening constant in `gauNEGF/config.py`). This guarantees a nonzero
anti-Hermitian piece so the analytic integrator's `1/(DD - DD^H)` factor
stays finite when `Sigma_0` happens to have negligible anti-Hermitian
piece (a degenerate case in production but trivial to handle once
flagged). It does NOT inject a magic constant -- it reuses an already-
present canonical parameter with documented physical meaning.

    damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer,
                      ENERGY_MIN_=ENERGY_MIN)
        -> (P_lower, Emin)

Single function. Builds `Fbar = Y_eff @ (F_eV + Sigma_0) @ Y_eff`,
runs `eig(Fbar)` once, chooses `Emin = min(D.real) - buffer` from
the same eigenvalues, builds `GamBar`, calls the existing
`density(V, Vc, D, GamBar, ENERGY_MIN_, Emin)`, back-transforms to
AO. Returns the AO density and the Emin it picked. Pure function:
no state, no cross-term return.

No internal split into private helpers. FockToP calls this once,
takes both return values, and uses `Emin` for the upper-contour
call that follows.

### 3.2 New: `_initAsymptoticSigma` on `NEGFE` in `gauNEGF/scfE.py`

Called from `setContact1D` / `setContactBethe` / `setSigma` after
`self.g` exists. ALSO called from `setVoltage` after `self.g.setF`
to refresh stale `Sigma_0` (see 2.6). Sets/refreshes:

    self.Sigma_0   # asymptotic constant from (1), complex-symmetric
    self.X_asymp   # asymptotic linear coefficient from (1), complex-symmetric
    self.S_eff     # S - X_asymp, complex-symmetric / non-Hermitian (NOT real)
    self.Y_eff     # S_eff^(-1/2) via general eig (inv_sqrt_general), complex
    self.damle_buffer  # initialized to EMIN_BUFFER on first call only,
                       # preserved on re-fit so user customization survives

Implementation: three deep probes (E1 = -1e3, E2 = -1e4, E3 = -1e5
defaults), fit a line through Sigma(E_i), report residual to caller,
warn if asymptotic linearity is poor (||residual|| / ||Sigma|| > 1%
at the deepest probe). X_asymp is kept FULLY COMPLEX (no Re(), no
symmetrization -- see 2.3), and `Y_eff = S_eff^(-1/2)` is computed with the
general-eig helper in 3.3. Also prints diagnostics: ||Im X||/||X||, the
non-Hermiticity ||X - X^dag||/||X||, the non-symmetry ||X - X^T||/||X||, and
the defining-property defect ||Y_eff @ S_eff @ Y_eff - I||.

On re-fit calls (from setVoltage), X_asymp/S_eff/Y_eff are recomputed
but mathematically guaranteed invariant (see 2.6); the redundant work
is accepted for code simplicity over per-component caching.

### 3.3 Extend `gauNEGF/utils.py`: general (non-Hermitian) inverse sqrt

S_eff is complex-symmetric and non-Hermitian (section 2.3), so a Hermitian
eigendecomposition (eigh) is the wrong tool. Add a JAXed helper that makes no
symmetry assumption:

    inv_sqrt_general(M)
        -> M^(-1/2) via general eig: V @ diag(D^(-1/2)) @ V^(-1)

Implementation: `D, V = jnp.linalg.eig(M)` (the same general-eig wrapper used
elsewhere in the module), `D_inv_sqrt = jnp.power(D.astype(complex128), -0.5)`,
return `V @ diag(D_inv_sqrt) @ jnp.linalg.inv(V)`. Satisfies `Y @ M @ Y = I` to
machine precision for any diagonalizable M (Hermitian, complex-symmetric, or
neither). `@jit`, requires `jax_enable_x64` (set at package import via
`density.py`). Lives alongside `fractional_matrix_power` and `inv`.

HISTORICAL NOTE: an earlier draft used an eigh-based
`fractional_matrix_power_signed`, which silently assumed a Hermitian S_eff.
That was a bug -- on C2 LANL2DZ it produced `||Y S_eff Y - I|| = 4.26`. It has
been removed (function + tests) and replaced by `inv_sqrt_general`.

The existing PSD-only `fractional_matrix_power` is kept unchanged
(faster real-only path for callers that know S is PSD).

### 3.4 FockToP replacement

Before (in `gauNEGF/scfE.py` near line 405):

    F_eV = self.F * har_to_eV
    Eminf_floor = calcPseudoPoleFloor(F_eV, self.S, self.g)
    self.Emin = calcEmin(F_eV, self.S, self.g, Emin=self.Emin)
    Emin_floor = min(self.Emin, Eminf_floor)
    Eminf_ = min(self.Emin, max(self.Eminf, Emin_floor))
    self.Eminf, self.TSW = calcTSW(F_eV, self.S, self.g,
                                   Eminf=Eminf_, tol=self.tol,
                                   Emin_floor=Emin_floor)
    P, _delta_N_lower = densityComplex(F_eV, self.S, self.g,
                                       self.Eminf, self.Emin,
                                       self.tol, T=0)
    nLower = np.trace(self.S @ P).real + _delta_N_lower

After:

    F_eV = self.F * har_to_eV
    P, self.Emin = damleLowerDensity(F_eV, self.Y_eff, self.Sigma_0,
                                     self.damle_buffer)
    nLower = np.trace(self.S @ P).real

The `compContourP2(mu)` helper for the upper piece is unchanged.

### 3.5 Removed / to-remove

Already removed:
  - `fractional_matrix_power_signed` (utils.py) + its tests -- replaced by
    `inv_sqrt_general` (the eigh assumption was wrong; see 2.3 / 3.3).

To remove once the Emin policy is validated (the definitive C2 LANL2DZ SCF run,
which still exercises the old setup path -- task: simplify `setIntegralLimits`):
  - `calcPseudoPoleFloor` (density.py)
  - `calcTSW` (density.py) and its `dTSW < 0` branch
  - pp-floor clamping in `FockToP` and `setIntegralLimits` (scfE.py)
  - `Emin_floor` / `Eminf_` derivations
  - `self.TSW` attribute and its warm-start logic
  - Corresponding test files (see Section 6)

NOT removed -- status revised:
  - `calcEmin` (density.py): RETAINED. The earlier "redundant once D supplies
    Emin" claim is withdrawn -- if the Fbar-based Emin (7) proves unstable on
    the definitive SCF run, `calcEmin`'s (F,S) band bottom is the contingency
    split. It is also still used at initial setup. Not pp machinery.

### 3.6 Kept, neutered

  - `self.Eminf`: pinned to `ENERGY_MIN` always, matching `scf.py`
    convention. Kept as an attribute for backward compatibility with
    any code that references it; no longer dynamically clamped.
  - `ENERGY_MIN` constant in `gauNEGF/config.py`: now serves as the
    Damle lower bound.

## 4. Pre-Implementation Gate

The design assumes the Mulliken cross-term `delta_N` returned by
`densityComplex` is subsumed into the S_eff framework and need not
appear separately in the Damle path. This claim MUST be verified
empirically before any production code is written.

### 4.1 Gate test

For C2 LANL2DZ at the converged isolated-molecule F (the test setup
already in `damle_lower_contour_test.py`):

  1. Old path:
        P_old, delta_N_old = densityComplex(F, S, g,
                                            Eminf=-deep, Emin)
        n_old_lower = trace(S @ P_old).real + delta_N_old

  2. New path:
        P_new_lower = damleLowerDensity(F, Y_eff, Sigma_0,
                                        Emin_from_D)
        n_new_lower = trace(S @ P_new_lower).real

  3. Add the upper piece to both (`compContourP2` with matched mu);
     compare the total `trace(S @ P_total)` between the two paths.

### 4.2 Pass/fail criterion

Total `trace(S @ P_total)` agrees between old and new paths to within
1e-3 electrons across at least two systems where the old path is
trustworthy: tight-binding (analytic ground truth available) and
C2 STO-3G (minimal basis, no pp pathology). C2 LANL2DZ is NOT a
valid gate system because the old path is itself broken there;
LANL2DZ comparisons belong in Section 5.3 validation, not the gate.

Pass: cross-term is subsumed into S_eff; proceed to implementation.
Fail: the Damle path must include an analytic cross-term computation
(separate sub-design needed; not covered by this spec).

## 5. Validation Plan

If the pre-implementation gate passes, validation proceeds in this
order:

### 5.1 Tight-binding Hamiltonians

1D chain with hand-built H, S, and coupling. Analytic ground truth
available; X can be computed in closed form; pp's absent or trivially
placed. The Damle integral should reproduce the analytic density to
machine precision. Catches math bugs in `damleLowerDensity`,
`inv_sqrt_signed`, and the asymptotic Sigma fit.

### 5.2 Single-zeta DFT (no pp's)

Minimal-basis DFT junctions such as C2 STO-3G. The old code path is
not pathological here, so old and new paths should agree on:

  - Converged Fermi level (within 1e-3 eV)
  - Total electron count (within 1e-3 electrons)
  - SCF energy (within 1e-6 Hartree)
  - Transmission shape (qualitative match)

Catches integration regressions vs trusted production behavior.

### 5.3 Double-zeta DFT with pp's

C2 LANL2DZ first, then CNT33 5-cell and Au10 PDT. Current production
is broken or shaky here, so direct comparison is not meaningful;
instead check that the new path produces:

  - Hermitian P (||P - P^H||_F / ||P||_F < 1e-3)
  - PSD P with natural-orbital occupations in [0, 2 + epsilon]
  - SCF convergence within iteration counts comparable to working
    systems
  - Transmission spectra qualitatively consistent with expected
    physics

## 6. Testing

### 6.1 New unit tests

  - `tests/test_damle_lower_density.py`: pure-math tests of
    `damleLowerDensity` on hand-constructed (F, Y_eff, Sigma_0).
    Cases: PSD S_eff (real Y_eff), indefinite S_eff (complex Y_eff),
    zero X (orthogonal contacts equivalent to standard Damle).

  - `tests/test_asymptotic_sigma_fit.py`: `_initAsymptoticSigma`
    returns sensible Sigma_0, X, S_eff for a tight-binding contact.
    Three-probe residual check warns when asymptotic linearity is
    poor.

  - `tests/test_fractional_matrix_power_signed.py`: helper
    round-trips correctly for PSD and indefinite inputs; output is
    symmetric to machine precision; matches the PSD-only
    `fractional_matrix_power` exactly when S is PSD.

  - `tests/test_setF_mu_invariance.py`: empirical verification of
    the rigid-band shift math from 2.6. Asserts X_asymp / S_eff /
    Y_eff invariant under setF mu changes (any bias), Sigma_0 shifts
    by predicted amount (per-contact, block-diagonal). Includes the
    TDD test that drives the setVoltage refit (failing before the
    fix, passing after).

### 6.2 Removed test files

  - `tests/test_pseudo_pole_detection.py` (4 tests added in the
    recent staged commits)
  - `tests/test_calcTSW_integration_limits.py` (or wherever the
    calcTSW tests live; to be located during implementation)

### 6.3 Tests to migrate

Any existing NEGF-SCF integration test that hard-codes `self.Eminf`
or expects the old pp behavior. Inventory to be produced during
implementation; tests that were validating the old workaround
machinery are removed, tests that validate physical density behavior
are migrated to the new path.

## 7. Open Questions and Risks

### 7.1 Pre-implementation gate may fail

If the cross-term `delta_N` does NOT subsume into S_eff, the design
needs an analytic cross-term computation. This is a non-trivial
sub-derivation and would extend the implementation scope. Probability
estimated: moderate. Mitigation: the gate test is fast (single C2
calculation) and can be run immediately.

### 7.2 Asymptotic linearity may not hold for some systems

The fit (1) assumes Sigma(E) is asymptotically linear in E. For
contact models that include non-polynomial energy dependence (e.g.,
square-root singularities at band edges), this could fail. The
three-probe residual check warns but does not refuse. Mitigation:
validation plan includes systems where asymptotic linearity is
expected; flag systems that fail the residual check for separate
treatment.

### 7.3 Complex Y_eff may not produce physical density

When S_eff is indefinite, Y_eff is complex and the resulting P from
Damle has nonzero imaginary part. Whether the imaginary part is
numerically negligible or load-bearing in the integral is not yet
proven. The C2 test showed `||Im(P_lower)||_F = 1.0e-2` vs
`||Re(P_lower)||_F = 7e0` -- small but not zero. Validation must
confirm this remains acceptable on production systems.

### 7.4 Buffer choice

`Emin = min(D.real) - buffer` with buffer = 20 eV is a heuristic.
If buffer is too small, broadening of states near Emin can push
density across the boundary; if too large, Damle is evaluated farther
into the asymptotic regime (which is fine but wastes precision).
Tunable via config; default validated on benchmark systems.

## 8. Non-Goals

This spec does NOT cover:

  - Modifying the `compContourP2(mu)` upper-contour path. The upper
    integral is correct as-is; pp's enclosed there are handled
    properly by the standard energy-dependent contour integration
    when the contour closes at the actual Fermi level.

  - Updating the Fermi-search `predict` method to use the
    S_eff / Sigma_0 framework consistently. The predict method
    currently uses `density(V, Vc, D, GamBar, Eminf, fermi)` with
    `X = S^(-1/2)` (standard Lowdin) and Sigma at the Fermi level.
    Bringing it into the same framework as the lower contour is a
    natural follow-up but kept out of scope here to keep this spec
    focused.

  - Refactoring the GHF / SOC code paths separately. The S_eff
    framework is basis-agnostic (real-symmetric and complex-Hermitian
    inputs are both handled by `eigh` + `np.emath.sqrt`), so the
    same `damleLowerDensity` should serve both. Confirmation is part
    of validation, not redesign.

  - Documentation overhaul. New API docstrings and theory-section
    updates are part of implementation; the existing
    `docs/pseudo_pole_handling.md` and `docs/tsw_convergence_notes.md`
    will be replaced with a single `damle_lower_contour.md` during
    implementation.

## 9. Implementation Sequencing

To be detailed in the implementation plan (next step). Outline:

  1. Run the pre-implementation gate test. If it fails, stop and
     revisit.
  2. Add `inv_sqrt_signed`, `damleLowerDensity`, `_initAsymptoticSigma`
     with full unit-test coverage.
  3. Wire into `setContact1D` / `setContacts` (compute Y_eff, Sigma_0
     once).
  4. Replace `FockToP` lower-contour block. The old code is preserved
     temporarily on a feature branch (NOT in main) so side-by-side
     runs are easy to set up by checking out the prior commit; the
     production file does not retain commented-out old code.
  5. Run tight-binding and single-zeta validation. With the old
     behavior reachable via git, comparison runs are straightforward.
  6. Once validation clears single-zeta DFT, delete the old code,
     tests, and `self.TSW` attribute; pin `self.Eminf = ENERGY_MIN`.
     This is the irreversible step.
  7. Run double-zeta DFT validation (C2 LANL2DZ, then CNT33, Au10).
  8. Update documentation: replace `pseudo_pole_handling.md` and
     `tsw_convergence_notes.md` with `damle_lower_contour.md`.
