# Changelog

All user-facing changes to gauNEGF. Commit hashes refer to the
[GauNEGF repository](https://github.com/wliverno/GauNEGF).

## Unreleased

Changes on top of the code merged to `main` in PR #3. The headline is a
complete electron count on non-orthogonal contacts: the device-contact
cross term now enters the bias window, the Fermi search and every count
diagnostic, and the Fermi search can be told which electrons to make
neutral.

### Migration

Code and results written against the previous `main` need these changes:

- **`setFock` needs `setVoltage` first and now builds the density.**
  `setFock(F)` (F in eV) sets the Fock matrix, computes the density it
  implies (`FockToP`, including the Fermi search if it is on) and hands that
  density to Gaussian. `NEGF.setFock` raises `AssertionError` and
  `NEGFE.setFock` raises `RuntimeError` when `setVoltage` has not run; the
  order is contacts, `setVoltage`, `setFock`, `SCF`. (935ea38)
- **`fermiCount` defaults.** The Fermi search counts the electrons chosen by
  `fermiCount`: `setContact1D` defaults to `'contact'` (contact-atom Mulliken
  count plus cross term, target = the contact atoms' nuclear charges),
  `setContactBethe` defaults to `'device_bare'` (whole-device `tr(PS)`
  without cross term, target = device electron count). `'device'` (whole
  device plus cross term) is rejected for Bethe contacts with `ValueError`;
  `'contact'` on Bethe contacts issues a `RuntimeWarning`. Two-cell 1D
  devices, whose contacts are the whole device, count as before.
  (9ff2565)
- **`crossTermQ` returns a tuple.** Every surface Green's function class
  returns `(Q_fwd, Q_rev, Q_sym)` (or `None` for an orthogonal contact);
  code that used the old single matrix should take index `[2]` (`Q_sym`).
  `crossTermQTot` still returns the summed `Q_sym` matrix. (4a3bdb1)
- **`densityGrid` / `densityGridN` return `(P, delta_N)` and need `ind`.**
  The bias-window integrators return the window density and its cross-term
  electron count, and the contact index `ind` is now a required argument
  of both `densityGrid` and `densityGridN`. (4559527)
- **Spin-resolved total transmission is the exact trace.** For `spin='g'`
  (and the `'u'`/`'ro'` spin-block path) the total returned by
  `calculate_transmission` and integrated by `calculate_current` is
  `Tr[Gamma_1 G^r Gamma_2 G^a]`. For spin-scalar leads it is bit-identical
  to the old sum of the four channels; for spin-mixing leads (e.g. the
  `AuSOC` Bethe lattice) total != sum of channels, and the difference is
  `T - Tspin.sum(axis=1)`. The four channels are unchanged. (05f72e2)
- **Configuration.** `config.CPU_THREADS` is removed. New constants:
  `HOST_BLAS_THREADS` (0afa2ef), `USE_INERTIA_EMIN` (9b726c0) and
  `CLEAR_JAX_CACHES_PER_CYCLE` (fe0091e).
- **New dependency: `threadpoolctl>=3.1`** (in `setup.py`; without it
  gauNEGF warns once and runs with uncapped BLAS threads). (0afa2ef)
- **Finite-temperature results computed before b963c99 must be
  recomputed** (sign of the finite-T Fermi-window term on the equilibrium
  contour). (b963c99)
- **Bethe-lattice and SOC numbers change.** The surface recursion no
  longer counts the back-branch twice (beccf9b), the out-of-plane cross-term
  kernel uses the bulk Green's function (a0a7b0e), and the on-site d block
  sits on each contact's cubic axes with the SOC `dxy` sign and p-orbital
  order corrected (2b99ade). Results from `setContactBethe` should be
  regenerated.
- **Biased runs on non-orthogonal contacts change.** The bias-window
  integral now carries the device-contact cross term (4559527), and the
  forward-real-axis cross-term count has the sign `+(1/pi) Im` (2ed4318).
- **`calculate_current` chemical-potential labels follow the SCF
  convention** (`mu1`, the left contact, is `fermi + qV/2`). I(V) is
  unchanged bit for bit. (7517007)
- **1D contacts.** `surfG1D` builds its device-contact couplings at
  `E - dFermi`, the exact shifted-lead algebra for non-orthogonal leads, and
  `density.calcFermi` gains `trackFermi` (default `True`; `getFermiContact`
  passes `False` so contact determination holds the leads fixed). (686e420)
- **`_symmetrize_F` defaults to `False` again**: only `setContact1D`
  averages the left/right contact Fock blocks; Bethe and constant-sigma
  contacts are no longer averaged, and contact on-site blocks follow the
  fresh Fock every cycle. (c425219)
- **`setSigma` takes full device-size matrices** and keeps their off-block
  support (needed for frozen orthogonalized-lattice self-energies);
  `spawnNEGF` passes them unsliced. (eccec68)
- **surfG protocol: `dFermi` / `dFermis` keywords.** `crossTermQ` and
  `crossTermQTot` accept `dFermi=None` on every class, and the Bethe
  `sigma` / `sigmaTot` accept `dFermi` / `dFermis`; contact Fermi shifts
  enter the compiled kernels as traced arguments. Custom surfG classes
  should accept these keywords. (fe0091e, cef10ff, 4a3bdb1)
- **`config` no longer sets `XLA_FLAGS` on import**; user-set XLA flags are
  kept and XLA sizes its own thread pool. (0afa2ef)

### Changed

- Fermi freeze guard and `dN_inclusive`. The per-cycle Fermi search is
  skipped (`Fermi frozen` in the log) while the previous cycle's
  cross-term-inclusive count error, stored as `negf.dN_inclusive`, is below
  the tolerance; it re-engages when the count moves. (0edfc84)
- New functions: `density.find_lower_bound` (9b726c0),
  `density.count_electrons` (9ff2565), `integrate.clear_kernel_cache`
  (fe0091e).
- Memory sizing. `integrate.MEMORY_PER_MATRIX_FACTOR` is 47 bytes per
  element per energy point (was 16) (b7c31fa); the bias-window kernel uses
  `GLESS_CROSS_MEMORY_FACTOR = 2 * MEMORY_PER_MATRIX_FACTOR` (7517007).
  Integration batches are smaller than before for the same matrix size.
- `NEGF.PToFock` refreshes `self.F` from Gaussian and `FockToP` uses
  `self.F` as given, so a Fock set with `setFock` is no longer overwritten
  by the next cycle. (935ea38)
- `NEGFE.PToFock` (and therefore `setDen` on a fresh `NEGFE`) raises an
  informative `RuntimeError` when called before `setVoltage`. (fe0091e)
- The default lower integration bound comes from `find_lower_bound`
  (inertia count of `E*S - F - Sigma(E)`) with a single contour; set
  `USE_INERTIA_EMIN = False` for the two-contour path. (9b726c0)
- Compiled integration kernels are cached per contact object and reused
  across SCF cycles; a Fermi update no longer recompiles them. (fe0091e,
  cef10ff)
- `FockToP` caps the host BLAS threads at `min(16, ncpu // 4)` (or
  `config.HOST_BLAS_THREADS`) for the density step; `utils.blas_limit` is a
  context manager that caps nothing until its `with` block is entered.
  `density()` is vectorized. (0afa2ef, 07c9937)
- A Fermi search left on under bias prints a warning: find the equilibrium
  Fermi energy first and pin it (`setVoltage(qV, fermi)`) for biased runs.
  The bias-window charge and the deviation from integer neutrality are
  reported as such, not as convergence errors. (723ab86, c56936e)
- Fermi search robustness: the bisection bracket is fixed, bisection keeps
  the failed method's bracket, the secant step is clamped to
  `[Emin, Emax]`, and a failed bisection keeps its starting Fermi energy
  and returns the density computed there. `fermiMethod='predict'` raises `ValueError` with
  `fermiCount='contact'`. (9ff2565)
- `spawnNEGF` refreshes a stale Fock (c128994) and reports a parent cross
  term: a WARNING when the spawned object searches its Fermi energy (its
  `tr(PS)`-only count shifts Ef), a NOTE when the Fermi energy is pinned.
  (ad4662f, bde7e68)
- `surfG1D` warns when a contact's infinite-chain Bloch overlap is not
  positive definite. (0edfc84)

### Fixed

- Finite-temperature equilibrium contour: Fermi-window sign, and the
  Fermi factor dropped on the arc (exact to `kT*exp(-N_KT)`), so adaptive
  refinement no longer exhausts the grid at T > 0. (b963c99)
- Stale JIT kernels after a contact Fermi update in Fermi scans and IV
  sweeps. (cef10ff)
- `_symmetrize_F` froze 1D contact on-site blocks at their initial values
  and L/R-averaged Bethe contact blocks. (c425219)
- Orthogonalized-lattice self-energies were truncated to the contact
  blocks in `spawnNEGF`. (eccec68)

### Known issues

- `surfG3D.surfGAt3D.crossTermQBulk` returns NaN on the toy fixture
  (pinned by `tests/test_crossterm_tuple_api.py`); the full 3D-periodic
  wrapper `surfG3` is not validated end to end. (4a3bdb1)
