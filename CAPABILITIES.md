# gauNEGF Release Capabilities

Honest status of what works, what's limited, and what's not ready for
production use in this release. Three tiers:

- **Green** -- tested and fully working.
- **Yellow** -- works in limited cases; convergence may be finicky.
- **Red** -- needs testing or has a known limitation.

## Headline feature: non-orthogonal contact handling

The big change in this release is **proper treatment of non-orthogonal
device-contact coupling** in the NEGF Green's function integrals. The
derivation is in `docs/superpowers/specs/2026-03-18-cross-term-corrections-derivation.md`
and represents work going back to March -- it touches every contact
implementation and both the density and DOS integrals.

What was wrong before: when `stau != 0` (non-orthogonal coupling), the
contact self-energy `Sigma(E) = (E*stau - tau) @ g_surf(E) @ (E*stau - tau)^T`
has terms linear and quadratic in E. The previous code dropped these
cross-terms, silently treating non-orthogonal contacts as orthogonal.

What's correct now: `crossTermQ` and `crossTermQTot` methods on every
surfG implementation return the energy-dependent cross-term correction.
The density and DOS integrals (`GrInt` -> `GrIntCross`, `densityComplex`,
`damleLowerDensity`) accept and propagate the correction through the
contour. Result: correct charge / DOS / transmission on any non-orthogonal
contact setup.

**Finite-temperature equilibrium contour (corrected 2026-07):** the
arc and Fermi-window pieces are combined with consistent orientation
and the arc is integrated without the (analytically negligible) fermi
factor, so finite-T equilibrium densities converge cleanly at any
contact broadening. Covered by tests/test_arc_fermi_unity.py.
Finite-T results computed with earlier versions should be recomputed.

Supporting work shipped at the same time:

- **Damle analytic lower contour**: one eigendecomposition + analytic
  cross-term `delta_N` replaces the prior deep-tail densityComplex call.
  ~30-450x faster than adaptive contour; `damle_dN_warn` flags cases
  where the lower contour holds non-trivial weight.
- **2-probe asymptotic Sigma fit**: places the deep `Sigma_0` probes
  relative to the integration window instead of hardcoded
  `(-1e3, -1e4, -1e5)`.
- **`eigh(X @ F @ X)` correctness fix**: `eigh(inv(S) @ F)` was silently
  using the upper triangle of a non-Hermitian matrix and producing wrong
  eigenvalues whenever S was non-trivial. Replaced everywhere; was the
  root cause of Emin-seed failures on CNT- and Au10-class systems.
- **Production-readiness pass 1**: trimmed inaccurate / verbose
  docstrings and dead code across `gauNEGF/`.

## Green

- **Energy-independent NEGF** (`NEGF` base class with `setSigma`):
  constant complex self-energy + Pulay DIIS SCF. Base SCF I/O, voltage
  bias, spin treatments (`r` / `ro` / `u` / `g`).
- **Energy-dependent Bethe contacts** (`NEGFE` + `setContactBethe`):
  full Bethe lattice with the new non-orthogonal cross-term correction.
  Validated by `test_bethe_cross_term_fermi.py`,
  `test_cross_term.py`, `test_damle_*`.
- **Energy-dependent 1D contacts on minimal basis**
  (`NEGFE` + `setContact1D` with `contactFromFock=True`): C2-STO3G,
  Au3-CRENBS converge end-to-end through the Damle lower contour +
  asymptotic Sigma + cross-term path.
- **SOC for Bethe contacts**: `test_soc.py` (~20 tests) + `test_constructSOCterm.py`.
- **Transport**: `calculate_transmission`, `calculate_dos`,
  `calculate_current`. DOS with cross-term Q correction
  (`test_transport_dos_crossterm.py`). Checkpointing
  (`test_transport_checkpointing.py`).
- **`surfGAt3D` (atomic 3D Bethe contacts)**: per-atom 3D builder is
  fully tested (`test_surfGAt3D.py`).
- **Numerical infrastructure**: `inv_sqrt_general`,
  `fractional_matrix_power`, JIT'd integrals, adaptive complex contour,
  Fermi search (Muller / secant / poly / bisect / predict). Adaptive
  integration tolerance is validated end-to-end via the Damle SCF tests.

## Yellow

- **`setContact1D` with external alpha/beta matrices on minimal basis**:
  works, but convergence is finicky depending on system. Smaller test
  coverage than the `contactFromFock=True` path.

## Red

- **`surfG3` (full 3D-periodic contact wrapper)**: the math works
  independently -- DOS makes sense, band diagrams plot correctly -- but
  **contact generation needs work and end-to-end testing**. Use with
  care; do not assume converged transport results yet.
- **Non-minimal basis on 1D contacts**: LANL2DZ-class (double-zeta,
  diffuse, ECP-with-valence) breaks the S-PSD precondition of the
  asymptotic Sigma framework. The 2 LANL2DZ tests in
  `test_negfe_damle_integration.py` are deliberately skipped with this
  reason. Fix is k-resolved overlap regularization at the surfG
  construction step (next release).

## TODO for next release

- End-to-end validation of `surfG3` (3D-periodic contact transport on
  a real device, comparison to reference).
- K-resolved overlap regularization to lift the minimal-basis
  precondition on 1D contacts.
- Surface the green / yellow / red picture in the Sphinx user docs (it
  currently lives only here).
