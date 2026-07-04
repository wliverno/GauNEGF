# Inertia-based lower bound for equilibrium NEGF density integration

**Goal:** Replace the `calcEmin` / `calcTSW` lower-bound logic with a single,
robust finder for the lowest pole of the device Green's function, then integrate
from just below it with the existing `densityComplex`. This fixes the two failure
modes that produced the whole lower-contour / Damle saga, while reusing the
existing integration machinery and adding fewer grid points.

**Status:** Implemented and DEFAULT as of 2026-06-28 (`USE_INERTIA_EMIN = True`
in config; set `False` for the legacy `calcEmin`/`calcTSW` path). The new method
is JAX throughout (on-device `jnp.linalg.eigh` inertia count). Validated on the
synthetic 1D toy and real carbon nanowires (STO-3G), and the full test suite runs
with the new method as the default (fast pure-numpy 43/43; Gaussian SCF + scf_io
30 passed; Bethe/3D `getFermiContact` pass). Implementation plan:
docs/superpowers/plans/2026-06-24-inertia-lower-bound-implementation.md.
Validation fixtures: tests/fixtures/_diag_inertia_carbon.py and
tests/fixtures/_diag_inertia_singlecontour_scf.py. The legacy `calcEmin`/`calcTSW`
path remains behind the flag (deprecated, documentation only); three
Damle-FockToP-internal tests that assert legacy machinery (calcEmin call,
Sigma_0 refit, lower-contour warning) were removed as incompatible with the new
default.

**Scope decision (for review):** this proposal REPLACES both `calcEmin` and
`calcTSW` with one function `find_lowest_pole`, which sets the integration floor
`Eminf = (lowest pole) - buffer`. A SINGLE contour `densityComplex(F, S, g, Eminf,
mu)` then computes both the density and the cross-term. There is NO split into
lower/upper contours: an empirical comparison (Section 3.7) showed splitting is
both slower (more grid points) and incorrect at finite temperature, because
`densityComplex` treats its upper argument as the chemical potential. The Damle
analytic lower-contour path becomes unnecessary.

---

## 1. The problem

At equilibrium the device density matrix is (T = 0)

    P = -(1/pi) Im integral_{-inf}^{mu} G^R(E) dE ,
    G^R(E) = [ (E + i*eta) S - F - Sigma(E) ]^{-1}

The device electron count is `tr(P S)`, and the non-orthogonal device-lead
overlap correction is

    delta_N = -(1/pi) Im integral_{-inf}^{mu} Tr( G^R(E) Q(E) ) dE

In practice the lower limit `-inf` is replaced by a finite floor, and the integral
is evaluated on complex semicircular contours (`densityComplex`) for smoothness.
The floor must satisfy two competing requirements:

1. **Deep enough:** below EVERY pole of `G^R` (every occupied state), or the
   integral silently drops filled states and the count is wrong.
2. **Shallow enough:** the contour radius is `~(top - floor)/2`. If the floor is
   the wide config bound (`ENERGY_MIN = -1e6`), the contour spans six orders of
   magnitude and the fixed-grid quadrature cannot resolve it -- it hits the grid
   cap and returns over-tolerance (does not converge).

So the correct floor is `Eminf = (lowest pole of G^R) - buffer`, and the density is
ONE contour:

    P, delta_N = densityComplex(F, S, g, Eminf, mu)

`densityComplex` handles finite temperature itself (Fermi weighting at `mu`); no
split into lower/upper contours is needed or wanted (Section 3.7). The entire
difficulty is finding `Eminf` -- the lowest pole -- reliably and cheaply.

---

## 2. Background: why calcEmin and calcTSW both fail

Both current functions try to locate this bound and both have a known failure:

- **`calcEmin`** (`density.py:1018-1022`): seed `Emin` from `min eig(X F X)`,
  then `while DOS(Emin) > tol: Emin -= 10`. This is a threshold scan of the
  density of states. The DOS is **non-monotone** -- it dips toward zero between
  resonances and in off-resonant gaps -- so the threshold trips early and leaves
  real poles below the reported `Emin`. ("It finishes early and there are still
  poles far below.")

- **`calcTSW`** (`density.py:1103-1127`): widen `Eminf` (doubling) until the
  truncated total spectral weight matches a reference computed at
  `Emin_floor = ENERGY_MIN`. That deep reference contour straddles a spurious
  deep negative-weight feature, surfacing as the `dTSW < 0` case. (Note: that
  feature was attributed to a non-PSD effective overlap from asymptotically
  E-linear contact `Sigma`. Since the overlap is now kept PSD by construction,
  this is moot here -- see Non-goals -- but the doubling-against-a-deep-reference
  strategy is still fragile and slow.)

So one finder stops too shallow (misses poles) and the other reaches too deep
(slow, and historically tangled with the deep artifact). The target sits between
them and neither lands it reliably.

---

## 3. The math

### 3.1 Poles are where `M(E)` goes singular

Define the (energy-dependent) inverse Green's function

    M(E) = E S - F - Sigma(E)

A pole of `G^R = M(E)^{-1}` is exactly an energy `E` where `M(E)` is singular --
i.e. where `M(E)` has a zero eigenvalue. Physically a pole is an energy at which
the open device has a state. The lowest pole is the deepest such energy; below it
there are no occupied states and the integrand is zero.

Two regimes:
- **Below the lead band** `Sigma(E)` is real, so `M(E)` is Hermitian and its
  poles are real (these are bound states).
- **Inside the band** `Sigma(E)` is complex (the lead surface Green's function
  `g_s(E)` has a branch cut there); poles become complex resonances and the band
  edge is where `Im Sigma(E)` turns on.

### 3.2 Matrix inertia -- counting states without finding them

This is the key tool, and it is the same trick used inside bisection
eigensolvers (Sturm sequences), just applied to an open system.

**Definition.** The *inertia* of a Hermitian matrix `A` is the triple
`(n_+, n_-, n_0)` = the number of positive, negative, and zero eigenvalues.
Sylvester's law of inertia says this triple is unchanged by any congruence
`A -> X^dag A X` with `X` invertible. Practically, you get it from an
`LDL^dag` factorization by reading the signs of the pivots -- **no eigenvalues
need to be computed**, just one `O(N^3)` factorization.

**The counting fact you may already know.** For an ordinary generalized
eigenproblem `H v = lambda S v` with `S` positive-definite, the number of
eigenvalues `lambda` below a chosen energy `E` equals the number of NEGATIVE
eigenvalues of `(H - E S)`. Reason: write `H - E S = S^{1/2}(X H X - E I)S^{1/2}`
with `X = S^{-1/2}`; this is a congruence, so by Sylvester the sign pattern of
`(H - E S)` matches the sign pattern of `(lambda_n - E)`. Each `lambda_n < E`
contributes one negative eigenvalue. So "count negatives of `(H - E S)`" is a
cheap, exact "how many states are below `E`" -- and it ticks by exactly one each
time `E` crosses a state. That is how eigensolvers count eigenvalues in an
interval without solving for them.

**Our case.** The contacts make the effective Hamiltonian `F + Sigma(E)`
energy-dependent, but at each FIXED `E` (below the band, where `Sigma` is real)
`M(E) = E S - F - Sigma(E)` is just a fixed Hermitian matrix, and the same
counting applies. Note `M(E) = -((F + Sigma(E)) - E S)`, so its negative count
relates to states above/below `E` with a sign flip; the net result, taking the
deep limit as the reference, is:

    n_neg(E) = number of negative eigenvalues of M(E)
    (number of poles below E) = N - n_neg(E)

Concretely: as `E -> -inf`, `M ~ E S` is negative-definite, so `n_neg = N` (no
poles below). Each pole you pass moving up flips one eigenvalue from negative to
positive, decrementing `n_neg`. **Below the lowest pole, `n_neg = N`; the lowest
pole is the energy where `n_neg` first drops to `N-1`.**

Intuition in one line: `n_neg(E)` is a cheap "level counter" -- hand it an
energy and it tells you how many poles lie below, without ever solving for them.

### 3.3 Why it is robust: monotonicity from the PSD overlap

A DOS scan fails because the DOS is non-monotone. The inertia count is
**monotone**, and that is exactly what makes bisection safe. Monotonicity holds
because the eigenvalues of `M(E)` move monotonically with `E`:

    dM/dE = S - dSigma/dE

Below the band `Sigma(E)` is asymptotically linear, `dSigma/dE ~ X_asymp`, so
`dM/dE ~ S - X_asymp = S_eff`, the effective overlap. Because the overlap (and
`S_eff`) is kept positive semi-definite by construction, `dM/dE` is PSD, so every
eigenvalue of `M(E)` is non-decreasing in `E`. Each therefore crosses zero
exactly once, `n_neg(E)` is a monotone integer staircase, and a monotone integer
count cannot "finish early." This is the load-bearing role of the PSD-overlap
guarantee: it is not just "no pseudo-pole," it is what licenses the bisection.

This holds for well-conditioned bases (minimal / STO-3G), where `S_eff` is PSD and
the deep `Im Sigma` is tiny. A diffuse double-zeta basis (lanl2dz) drives `S_eff`
non-PSD and `Im Sigma` linearly divergent deep -- the separate negative/divergent
density-matrix pathology (Section 8), out of scope here and already broken for the
existing methods.

### 3.4 The lowest-pole finder

```
find_lowest_pole(F, S, g, mu, E_floor):
    N = dim(F)
    below(E):
        Mh = hermitian_part( E*S - F - Sigma(E) )    # one contact self-energy eval
        return count(eigvals(Mh) < 0) == N           # n_neg == N ; or LDL pivots
    assert below(E_floor) and not below(mu)
    bisect E in [E_floor, mu] on below(.) -> boundary
    return boundary
```

`below(E)` is true exactly when `E` is below every pole (all `N` eigenvalues of the
Hermitian part of `M(E)` still negative). Bisection on this monotone integer
predicate converges in `~log2((mu - E_floor)/tol)` steps, each a single `M(E)`
build plus its inertia. No integration, no doubling, no DOS threshold -- and, per
the carbon test (Section 5.2), no `Im Sigma` test (that clause was a toy artifact).

### 3.5 What the finder returns

`n_neg` ticks down at every real pole, so the finder returns the deepest one,
whatever it is. On the real carbon nanowires (Section 5.2) that is the C 1s core
near `-271 eV`; in other systems it can be a bound state pulled below the valence
band. Either way it is a real state caught by the inertia count -- no separate
band-edge or `Im Sigma` test is needed.

(An earlier draft also required `||Im Sigma|| < thresh` to catch a clean continuum
band-bottom. The carbon test killed that clause: on a diffuse basis `Im Sigma`
does not vanish deep -- it grows -- so it would never pass. `n_neg` alone is the
robust mechanism.)

### 3.6 The single contour and the grid-point payoff

With `Eminf = find_lowest_pole(...) - buffer`, one call does everything:

    P, delta_N = densityComplex(F, S, g, Eminf, mu)   # density AND cross-term, all T

The contour radius is now `~(mu - Eminf)/2` (tens of eV, not `1e6`), so the fixed
grid resolves it easily. Removing the dead deep stretch below `Eminf` is not just
a speed-up -- it is what lets the quadrature converge at all (Section 5.1). The deep
empty region that capped the old `[-1e6, mu]` contour is simply never integrated.

### 3.7 Why a single contour (do NOT split for temperature)

It is tempting to split into a deep `T=0` contour `[Eminf, Emin]` plus a near-`mu`
finite-`T` contour `[Emin, mu]`. We measured it (Section 5.1) and it is WORSE on both
axes:

- **Slower:** two contours each carry a minimum grid-point count. Toy numbers:
  one contour 162 pts (T=0) / 216 pts (T=300); split 216 pts / 288 pts.
- **Incorrect at finite `T`:** `densityComplex(F, S, g, Emin, mu)` treats its
  *upper argument as the chemical potential* and applies the Fermi broadening
  there. Splitting passes the split point to the lower contour as `mu`, putting a
  spurious Fermi edge at the split energy (where the occupation is really `1`).
  In the toy this shifted the count by `8.9e-4` at `T=300` (they agree at `T=0`,
  where there is no Fermi broadening). The single contour applies the Fermi
  function once, at the true `mu`, and is the correct value.

So `densityComplex` is built for one contour `[Emin, mu]` with `mu` the chemical
potential; the correct, cheaper use is one contour `[Eminf, mu]`. The inertia
finder concerns ONLY `Eminf` (a property of `G^R`, independent of `mu` and `T`);
the contour and its temperature handling are unchanged from the current code.

---

## 4. Proposed code changes

- **Add** `find_lowest_pole(F, S, g, mu, E_floor=ENERGY_MIN)`
  to `gauNEGF/density.py` (`mu` is the upper end of the bisection search).
  Uses only `g.sigmaTot(E)` and `F, S` (no lead
  `alpha`/`beta` required), and only the inertia count `n_neg` (no `Im Sigma`
  test). Inertia via `eigvalsh` initially; switch to an `LDL` pivot count for
  speed once correct. Returns the lowest pole; the integration floor is
  `Eminf = that - buffer`.
- **Replace** both `calcEmin` (`density.py:969`) and `calcTSW` (`density.py:1028`)
  at the call sites (`density.py:1329`, `:1461-1464`; `scfE.py:403`, `:524`) with
  `Eminf = find_lowest_pole(F, S, g, mu) - EMIN_BUFFER`. Reuse the existing config
  buffer (~20 eV): it puts the contour's lower endpoint well below the lowest
  pole's resonance, which keeps the integrand smooth there and should help the
  contour converge. (`calcEmin` already used this buffer; the finder just supplies
  a more robust lowest pole than its DOS scan.)
- **Reuse** `densityComplex(F, S, g, Eminf, mu)` -- a SINGLE contour -- for both
  `P` and `delta_N`, at whatever temperature. Do NOT split.
- **Deprecate** `calcEmin`, `calcTSW`, and the Damle analytic lower-contour path
  for this purpose (keep behind a flag for one release if a fallback is wanted).

---

## 5. Validation

### 5.1 Synthetic 1D toy

Benchmark `tests/fixtures/_diag_lower_contour_tb_benchmark.py`, 2-site dimer +
two 1D leads, fat lead band (`b_c = -40`), `mu = 0.101`:

```
                          lowest pole   tight contour            huge contour [-1e6, mu]
main toy (band bottom)    -50.0001      162 pts, err 1.8e-8      486 pts (cap), err 1.7e-3 (no converge)
planted bound state @-70  -70.0015      162 pts, err 1.2e-8      486 pts (cap), err 1.7e-3
```
- `tr(P S)` tight vs huge agree to `~1e-5`; the tight value is the converged
  (trustworthy) one, the huge contour is the failing baseline.
- The planted bound state at `-70` (a 3rd device orbital below the band) is
  caught by the inertia count and shows up correctly as the `+1` in `tr(P S)`
  (`2.066` vs `1.066`).
- Same `densityComplex` returns `delta_N` (`-0.0896`) in both cases.
- **Single vs split contour:** one contour `[Eminf, mu]` uses fewer grid points
  (162 vs 216 at T=0; 216 vs 288 at T=300) AND is correct at finite T. Splitting
  shifts the count by `8.9e-4` at T=300 by misplacing the Fermi edge at the split
  energy (Section 3.7). Single contour chosen.

So: exact lowest pole including bound states below the band, ~3x fewer grid
points, convergence where the wide contour caps out, and a single contour that is
both cheaper and finite-T-correct.

### 5.2 Real carbon nanowires (STO-3G)

Test `tests/fixtures/_diag_inertia_carbon.py` (SLURM job
`diag_inertia_carbon.job`), real Gaussian `b3lyp/sto-3g` devices built through
`NEGFE` + `setContact1D`. `mu` from `getFermiContact(g, ne)` with `ne = nae` (the
filled-orbital count). Two geometries:

```
case  N    ne   band floor   lowest pole   calcEmin    tr(P S)   delta_N   sum      |single - deeper|
C2    10   6    -271.32 eV   -271.34 eV    -291.32 eV  5.51988   0.48026   6.00000  1.16e-6
C3    15   9    -272.67 eV   -272.70 eV    -292.67 eV  8.51363   0.48581   9.00000  1.58e-6
```

(`C2` = `setContact1D([[1],[2]])`, the pathological 1D lead; `C3` =
`setContact1D([[1],[3]], tauList=[[2],[2]])`, leads through the middle atom.)

What this shows:

- **The finder finds the same floor `calcEmin` aims for, robustly.** The lowest
  pole is the C 1s core state, right at the band floor (`min eig(F, S)`). The
  `n_neg` finder lands it to `~0.02 eV`; `calcEmin` reports `~20 eV` deeper purely
  because it subtracts `EMIN_BUFFER`. So the finder is not "tighter" -- with the
  same `EMIN_BUFFER` it produces the SAME floor `calcEmin` would, but via a
  monotone count instead of a DOS scan that can finish early. The value is
  robustness, not a different bound.
- **The single contour is exact.** `tr(P S) + delta_N` recovers `ne` to five
  digits on both cases (`6.00000`, `9.00000`), and a 50-eV-deeper contour changes
  the count by `~1e-6` -- the floor is deep enough and the count is complete.
- **`n_neg` alone is the right predicate (no `Im Sigma` clause).** The `n_neg`-only
  finder and the older `n_neg + ||Im Sigma|| < thresh` finder agree to `~0.15 eV`;
  the difference is the Im-clause refusing to step the last fraction of an eV, not
  a different pole. On a clean STO-3G lead `||Im Sigma||` deep is `~6e-6` (flat),
  so the clause is harmless here -- but it is unnecessary, and on a diffuse basis
  it breaks (next point), so we drop it.
- **Diffuse basis (lanl2dz) is out of scope and breaks the same way for everyone.**
  Repeating `C2` in `lanl2dz` drives `||Im Sigma||` from `~4e2` near the floor to
  `~1.6e4` at `-1e4` (growing, not vanishing) and the SCF does not converge; the
  density matrix is negative/divergent. The finder correctly returns `None` (its
  deep-floor predicate fails because the effective overlap is non-PSD there). This
  is the known diffuse-double-zeta pathology (Section 8), which already breaks
  `calcEmin`/`calcTSW`; it is not a finder defect.

---

## 6. Scaling

The finder is one Hermitian inertia (an `LDL` factorization, `O(N_device^3)`) per
bisection step, `~20-60` steps. The integrator is hundreds of `O(N_device^3)`
solves. So the finder is a small fraction of the integration cost and grows the
same way -- it scales to large devices. It needs no lead `alpha`/`beta` and no
device-sized augmentation (contrast the shelved pole-fit, whose augmented matrix
is larger than the device).

---

## 7. Risks / open questions

1. **Diffuse / ill-conditioned bases:** the finder relies on `Re(S_eff)` being PD
   deep so `n_neg(E)` stays monotone. A diffuse double-zeta basis (lanl2dz) drives
   `S_eff` non-PSD and `Im Sigma` linearly divergent deep; there the finder
   returns `None` rather than a wrong floor (validated, Section 5.2). That is the
   correct behavior -- the density matrix is itself divergent there (Section 8) --
   but it means a caller on such a basis gets no bound and must fall back. Detect
   the `None` and surface it clearly. (The earlier `||Im Sigma|| < thresh`
   band-edge clause was removed: the carbon test showed `Im Sigma` does not vanish
   deep on a diffuse basis, so the clause would never pass; `n_neg` alone is the
   mechanism.)
2. **`buffer` choice:** how far below the lowest pole to start. Adopt the existing
   `EMIN_BUFFER` (~20 eV) -- the same buffer `calcEmin` used (Section 4). Still
   wants a documented default and a sanity floor for unusual lead bandwidths.
3. **Scaling proof:** validated on real carbon nanowires up to `N = 15` (STO-3G,
   Section 5.2) in addition to the toy; the finder cost is a small fraction of the
   integration. Confirm on a genuinely large device (hundreds of orbitals) before
   relying on it in production.
4. **Related but separate bug:** `densityReal` and `densityComplex` returned the
   cross-term `delta_N` with OPPOSITE signs on the truncated range. `densityComplex`
   matched the reference; `densityReal`'s sign needs auditing before either is
   trusted for `delta_N`. Tracked separately from this spec.

---

## 8. Non-goals

- **Non-PSD overlap / diffuse bases.** The overlap is kept PSD by construction;
  cases where it is not are out of scope. (This is why the historical
  "pseudo-pole" is a red herring here -- all poles of `[ES - F - Sigma(E)]^{-1}`
  are real states.) The concrete instance is a diffuse double-zeta basis
  (lanl2dz): `S_eff` goes non-PSD and `Im Sigma` diverges linearly deep, so the
  density matrix is itself negative/divergent (validated, Section 5.2). The finder
  returns `None` there by design. This is a pre-existing limitation that already
  breaks `calcEmin`/`calcTSW` -- fixing the divergent-density basis is a separate
  problem, not part of this spec.
- **A new integration method.** The pole-fit / auxiliary-mode expansion (which
  recovers the lower-contour count analytically from `Sigma` samples) is shelved:
  it is new math, its augmented eigenproblem is larger than the device, and it
  does not scale as well as reusing `densityComplex`. Revisit only if a
  no-integration route is ever specifically wanted.
