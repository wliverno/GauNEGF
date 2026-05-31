# Contour Integration: Toy Model Findings and Path Forward

Date: 2026-04-09
Branch: crossTermFermiSpec

## Background

After switching from `calcEmin + nLower` to `calcTSW` (see
`2026-04-08-calcTSW-integration-limits-design.md`), the equilibrium density
integration (`densityComplex(Emin, mu, ...)`) began failing in the Au nanowire
SCF loop starting at iteration 1. `calcTSW` was converging correctly but
returning an Emin of -448 eV instead of the -428 eV it found at iteration 0,
and the equilibrium integration consistently hit MAX_GRID_POINTS (486) with
errors of ~1-3 in the density matrix.

## Toy Model Setup

Two-state orthonormal device (S = I):
- State 0: deep bound state at E_bound = -400 eV, left 1D chain contact
  (alpha=-400, beta=-0.5 eV, weak coupling tau=0.1 eV, stau_L=0.5)
- State 1: near-Fermi conducting state at E_cond = -5 eV, right 1D chain contact
  (alpha=-5, beta=-2 eV, strong coupling tau=1.0 eV, stau_R=0.8)
- Fermi energy mu = -4 eV, ne_expected = 1.063 (including large negative delta_N=-1.809
  from non-orthogonal coupling via GrIntCross)

All tests run with non-orthogonal coupling (stau_L=0.5, stau_R=0.8) to trigger
the cross-term path in GrIntCross. Results are in toy_contour_test.py.

## Findings

### 1. Im(z) SPACING between consecutive points near mu is the core problem

The ANT quadrature concentrates points at both endpoints (theta=0 near mu,
theta=pi near Emin). The number of Re(z) points within 0.5 eV of mu is constant
regardless of contour radius r:

```
         N=54   N=162   N=486
r=201:    12      36      108
r=223:    12      36      107
```

However, the IMAGINARY PART of consecutive points near mu grows linearly with r:

```
Im(z) spacing near mu ~= r * pi / N
```

For N=486 and r=222 eV (Emin=-448): Im(z) spacing ~= 1.44 eV between levels.
For N=486 and r=201 eV (Emin=-406): Im(z) spacing ~= 1.30 eV.

When Gamma (contact broadening near mu) is comparable to or smaller than this
spacing, G^R is not adequately resolved and the integral fails to converge.
The toy model has Gamma_R ~ 0.5 eV; the Au system likely has smaller effective
Gamma for some near-Fermi states, making the problem worse.

### 2. The toy model does NOT reproduce the Au convergence failure

The 2-state toy model converges at tol=1e-6 for all tested Emin values
(-402 to -600 eV), even at 486 points. The Au system failure requires the
full 166-orbital structure, likely because:
- Complex multi-orbital coupling creates richer integrand structure
- Some near-Fermi states have smaller effective Gamma than the toy model
- The Au 5d band and surface Green's function features interact

This means toy model results give mechanistic insight but do NOT prove that
any proposed fix will work in the Au system. Au testing is required.

### 3. TSW is stable; the warm-start bug causes unnecessary Emin expansion

TSW = Tr(P @ S) + delta_N remains constant as Emin varies (toy model verified
from -402 to -600 eV). The calcTSW convergence criterion is physically correct.

However, the warm-start bug (`TSW_prev = 0.0 if TSW is None else TSW`) uses
the old F's TSW as the baseline when F changes between SCF iterations. A
difference of ~0.001 exceeds FERMI_CALCULATION_TOL=1e-3, triggering two extra
-10 eV expansions: Emin goes from -428 to -448. This unnecessarily grows the
contour radius from r=212 to r=222 eV.

### 4. Emax for calcTSW has no bearing on the equilibrium density integration

calcTSW uses Emax as the upper bound for the TSW contour (to count all states).
The equilibrium density always uses mu (Fermi energy) as its upper bound.
Fixing Emax to a large constant (e.g. +900 eV) for calcTSW simplifies the
warm-start logic but does NOT affect the equilibrium contour radius.

### 5. Log-energy quadrature within the adaptive ANT framework does NOT work

A proper implementation of log-energy parameterization (uniform in
u = log(mu - Re(z)), using GrIntCross with full delta_N, inside
integratePointsAdaptiveANT) was tested. Results:

- Hits 486 point limit for all Emin tested (-406 to -450 eV)
- Gives systematically wrong answer (ne=1.10 vs reference 1.063, error ~4%)

Root cause: the ANT quadrature points are designed for smooth integrands in
x-space with the Chebyshev weight function. The log-v parameterization
produces a Jacobian dz/dx that varies by a factor of ~u_max/pi (~4x) across
the contour. The ANT points are not optimal for this transformed integrand,
and the nested error estimate misbehaves. A log-energy approach would require
a completely new adaptive integrator designed for log-v space.

### 6. densityReal (Gauss-Legendre) silently returns zero

densityReal starts at N=1 with a single GL point at the midpoint of [Emin, mu].
For Emin << mu, this midpoint falls in a spectral gap (Im(G^R) ~ 0), so P=0
and the convergence check |P - P_prev| = 0 < tol passes immediately. The
function returns zero with a spurious "converged in 1 points" message.
This is not a useful result and not a detectable failure.

### 7. The correct split: densityRealANT(Emin, E_split) + densityComplex(E_split, mu)

E_split is chosen as the output of calcEmin: the energy just below the lowest
Fock eigenvalue where DOS drops below threshold. This is NOT a spectral gap
within the spectrum -- it is below ALL states.

**Deep section: densityRealANT(Emin, E_split)**

The region [Emin, E_split] has zero DOS (Im(G^R) = 0 on the real axis).
Using adaptive ANT on the real axis:
- Converges correctly to zero (ne_deep = 0.000000)
- Requires only 18 points regardless of how deep Emin is
- Scales trivially with Emin depth (verified from Emin=-410 to -500 eV)

The reason densityReal (GL) fails but densityRealANT succeeds: GL starts
at N=1 with a point in the spectral gap and converges trivially; ANT also
reaches the same correct zero, but does so robustly with proper error control.

**Near-Fermi section: densityComplex(E_split, mu)**

The complex contour radius is now:
  r_near = (mu - E_split) / 2

This is determined by E_split (calcEmin output), NOT by Emin (calcTSW output).
The warm-start bug and conservative calcTSW expansion no longer affect
the complex contour radius.

For the toy model: r_near = (-4 - (-406)) / 2 = 201 eV regardless of whether
calcTSW gives Emin=-428 or -448. The complex contour converges in 162 points.

**Additivity verified (toy model):**
```
Emin    ne_deep_real   ne_near_cplx   ne_total    err
-410        0.000000       1.063473   1.063473   2.33e-10
-420        0.000000       1.063473   1.063473   7.97e-10
-440        0.000000       1.063473   1.063473   1.86e-09
-448        0.000000       1.063473   1.063473   2.26e-09
-500        0.000000       1.063473   1.063473   4.68e-09
```

delta_N from the cross-term is correctly zero for the deep section and
accumulates entirely in the near-Fermi section. Additivity holds.

## Root Cause (revised)

The equilibrium density integration fails because:

1. calcTSW warm-start bug causes Emin to expand to -448 eV instead of stopping
   at -428 eV (20 eV unnecessary expansion).
2. The equilibrium contour `densityComplex(Emin, mu)` uses radius r=(mu-Emin)/2,
   which grows with Emin depth. At r=222 eV and N=486, Im(z) spacing near mu
   is ~1.44 eV -- too coarse to resolve the near-Fermi G^R structure.
3. The old `nLower` from densityReal (GL) was always zero (E_split below all
   states, zero-DOS region), but returned zero silently via a spurious
   convergence. The failure was always in the full equilibrium contour.
4. The toy model cannot reproduce the Au failure -- actual Au testing required.

## Proposed Fix

Replace:
  `densityComplex(Emin_calcTSW, mu)` (single large contour)

With:
  `densityRealANT(Emin_calcTSW, E_split_calcEmin)` (trivially zero, 18 pts)
  `densityComplex(E_split_calcEmin, mu)`             (fixed radius, all physics)

Where E_split_calcEmin = calcEmin output (just below lowest eigenvalue, DOS < tol).

**Why this decouples the problem:**
- The complex contour radius r_near = (mu - E_split) / 2 is now fixed by
  the physics of the system (where the spectrum starts), not by how deep
  calcTSW pushes Emin.
- The warm-start bug no longer affects convergence (only affects whether
  densityRealANT needs 18 or 54 points in the zero-DOS section).
- If calcEmin produces a tighter E_split than calcTSW's Emin, the complex
  contour radius is strictly smaller.

**Open question:** Whether r_near with E_split from calcEmin is small enough
for the Au system to converge at 486 points. This depends on the effective
Gamma of the near-Fermi Au states. The toy model cannot answer this.

## Next Steps

1. Fix the calcTSW warm-start bug (compare TSW consecutive values with current
   F, not stored TSW from previous F). Fix Emax to a large constant.
2. Add `densityRealANT` to density.py (adaptive ANT on real axis, +1j sign).
3. Modify `FockToP` in scfE.py and `getFermiContact` in density.py to use the
   split: call calcEmin for E_split, then densityRealANT + densityComplex.
4. Test on the actual Au nanowire SCF loop to see if r_near is small enough.
5. If Au still fails, the contour radius is fundamentally too large and a
   custom log-v adaptive integrator (new integratePointsAdaptive_logV) would
   be required -- this is a significantly larger implementation effort.
