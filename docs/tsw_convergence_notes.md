# TSW Convergence: Deep Pseudo-Poles from Asymptotically E-linear Self-Energies

This note documents a mechanism by which the `calcTSW` lower-bound search
can converge to a physically wrong `Eminf` if not handled carefully, the
empirical evidence that identifies the mechanism for the gauNEGF Bethe-
contact path, and the heuristic choices in the current implementation
that produce SCF-usable density matrices.

It is written for anyone reading `calcTSW`, looking at `dTSW < 0` debug
output, or wondering why the convergence test is one-sided rather than
two-sided. It is also relevant background for anyone modifying any
non-orthogonal contact surface Green's function (`surfGBethe`,
`surfG1D`, `surfG3D`), because the mechanism originates in those
classes' surface GF iteration, not in `calcTSW` itself.

---

## 1. What `calcTSW` is computing

`calcTSW` returns a lower integration bound `Eminf` such that the complex-
contour integral

```
P  =  -(1/pi) Im integral_{Eminf}^{Emax} G^R(z) dz   (along semicircle in UHP)
TSW = Tr(P @ S)
```

is stable against further lowering of `Eminf`. The reference is the same
quantity evaluated at the configured `Emin_floor` (effectively the
"infinite" contour). The goal is to find the *largest* `Eminf` whose
contour still captures all the physically meaningful integrated spectral
weight, so subsequent SCF iterations integrate over a tight contour.

The function doubles `|Eminf|` each iteration. The one-sided convergence
test is

```
dTSW = TSW_ref - TSW_new
if dTSW / |TSW_ref| < tol: converged
```

A truly converged search has `dTSW` small and positive (the truncated
contour captured everything the reference did). `dTSW < 0` means the
truncated contour is *larger* than the reference - the reference contour
enclosed something the truncated one excluded. The test accepts that
case as converged, because the truncated value is the physically correct
one. The rest of this note is why.

---

## 2. The orthogonal-basis, well-behaved case

For PSD `S` and a self-energy `Sigma(E)` that decays at large `|E|` (e.g.
constant `Sigma` on a few contact orbitals), the spectral function
`A(E) = -(1/pi) Im G^R(E)` is PSD at every real `E`. The real-axis
integral

```
Tr(P @ S) = integral_{Eminf}^{Emax} Tr[A(E) S] dE
```

has a non-negative integrand, so `TSW` is monotonically non-increasing
in `Eminf`. The one-sided test converges with `dTSW = 0+` at the Eminf
where all the contact-broadened tails of physical states are captured.

Empirically: we verified this directly for AuBetheFerroceneHSE by
swapping `surfGBethe` for `surfGTest` with `sig = -0.05j` on the contact
atoms (`AuStudies/test_constSigma_TSW.py`). Sweeping `Eminf` from -1e6
to -50 gave `TSW` rising monotonically from 1247.97 to 1155.72, no sign
flip, no anomaly. The mechanism described in section 3 is therefore
*not* about `(F, S)` or the resolvent construction in general; it is
specific to self-energies with a particular asymptotic structure.

---

## 3. The mechanism: asymptotically E-linear Sigma -> non-PSD S_eff

The Bethe surface GF in `surfGBethe.surfGB` / `surfGBAt` solves a
self-consistent Dyson equation for the surface self-energy. The relevant
body of the iteration (`sigmaK`, `sigmaSurf` in `surfGBethe.py`) builds

```
B     = E_eff * S_k - V_k
B_bar = E_eff * S_k^dag - V_k^dag
sigmaK[k] = B @ g_K @ B_bar
```

where `S_k` is the inter-cell overlap to the k-th lattice neighbor and
`g_K = inv(E_eff*I - H0 - sigTot + sigmaK[pair_k])`. For the Bethe
lattice with non-orthogonal overlap (`S_k != 0`), this iteration has an
asymptotic form at large `|E|`:

```
B @ g_K @ B_bar  ~  (E*S_k) @ (1/E * I) @ (E*S_k^dag)  =  E * S_k @ S_k^dag
```

so `Sigma(E)` is asymptotically **linear in E**, with a matrix
coefficient determined by the contact overlap structure:

```
Sigma_L(E)  ->  X_L * E   as E -> -infinity
Sigma_R(E)  ->  X_R * E
```

This is the regime where the model breaks. The deep-E resolvent becomes

```
G^R(E)  ~  (E * (S - X_L - X_R) - F)^-1
        =  (E * S_eff - F)^-1
```

The *effective overlap* `S_eff = S - X_L - X_R` can be non-PSD even when
the device `S` itself is PSD, because subtracting `X_L + X_R` can push
some eigendirections below zero. The generalized eigenproblem
`(F, S_eff)` then has eigenvalues at deep negative real energies whose
right eigenvectors `v_n` satisfy `<v_n, S_eff v_n> < 0`. These are the
**pseudo-poles**.

For each pseudo-pole, the contour integral picks up a residue weight

```
Tr(residue_n @ S)  =  <v_n, S v_n> / <v_n, S_eff v_n>
```

The numerator `<v_n, S v_n>` is positive (since `S` is PSD), the
denominator `<v_n, S_eff v_n>` is negative (by construction), so the
contribution to `TSW` is **negative**. A contour wide enough to include
the pseudo-pole locations subtracts this weight; a contour that excludes
them does not.

---

## 4. Direct empirical confirmation (AuBetheFerroceneHSE, HSE/Au-SOC)

> Note: the E-linear Sigma asymptotic is set by the contact `S_k` overlaps
> alone, so localization and mechanism conclusions here are independent of
> the contact on-site Hamiltonian. Absolute Fermi-level, charge, and weight
> numbers should not be used for physics conclusions without re-running with
> a current checkout; the ~0.3255 asymptote per direction and the -315 dTSW
> magnitude are expected to be stable to small corrections.

Three diagnostics in `AuStudies/` establish this concretely.

**`diag_bethe_deepE.py`** - Sigma scaling at deep E:

```
E (eV)    ||Sig_L||    ||Sig_L||/|E|   max eig anti-H (Gamma/2)
-1e+04    3.241e+03    3.241e-01        4.344e-05
-1e+05    3.253e+04    3.253e-01        4.392e-05
-1e+06    3.255e+05    3.255e-01        4.397e-05

||Sigma_L(-1e6)/(-1e6) - Sigma_L(-1e5)/(-1e5)|| = 1.304e-04
```

`||Sigma_L||/|E|` converges to a constant ~0.3255, and successive
`Sigma_L(E)/E` agree to four significant figures. The Bethe Sigma is
demonstrably linear in E at deep E. The anti-Hermitian part of Sigma
stays non-negative everywhere (decimation converges to a properly
retarded Sigma), so the mechanism is *not* a decimation-convergence
failure.

`S_eff = S - Re(X_L) - Re(X_R)`:

```
S       eigenvalues: min=+1.076e-03, max=8.592   #neg=0
S_eff   eigenvalues: min=-4.947e-04, max=8.588   #neg=8
```

S itself is PSD; S_eff has 8 negative eigenvalues. The mechanism is
*not* about non-PSD device S - it is about non-PSD S_eff induced by the
contact Sigma.

**`diag_bethe_pseudopoles.py`** - localizing the pseudo-poles:

```
Top deep generalized eigenvalues of (F, S_eff):
  Re lambda     Im lambda      <v,S v>   <v,S_eff v>
  -3.813e+03   -8.2e-10        46.5       -1.0
  -3.813e+03   -2.1e-10        46.5       -1.0
  -3.330e+03    5.7e-10        40.7       -1.0
  -3.330e+03    3.9e-11        40.7       -1.0
  -3.200e+03    4.1e-10        39.2       -1.0
  -3.200e+03    1.8e-10        39.2       -1.0
  -2.751e+03   -3.2e-10        33.7       -1.0
  -2.751e+03   -1.2e-10        33.7       -1.0
```

8 eigenvalues at real energies -2751 to -3813 eV, all with
`<v, S_eff v> = -1` (normalized) and `<v, S v> = +33.7 to +46.5`. Each
contributes weight `<v,S v> / <v,S_eff v> = -33.7 to -46.5` to TSW when
included. Sum: `-(2*46.5 + 2*40.7 + 2*39.2 + 2*33.7) = -320`.

The production calcTSW output for this system reports
`dTSW = -3.15e+02` stable across iterations - **matching the
predicted -320 within scipy.eig accuracy**. The pseudo-poles are
identified.

---

## 5. Localizing the source within surfGBethe (Tests A+B+C)

If Sigma(E) is asymptotically E-linear, which part of the surfGBethe
construction produces that? Three probes in
`AuStudies/diag_bethe_localize_ABC.py` isolate the source without
modifying surfGBethe.

**Test A: atomic surface GF (`surfGBAt.sigmaSurf`).** The 9 surface
directions' self-energies in the atomic 18x18 SOC basis, probed at
deep E:

```
E (eV)    | d00   d01   d02   d03   d04   d05   d06   d07   d08
-1.0e+02  | 0.017 0.017 0.017 0.017 0.017 0.017 0.017 0.017 0.017
-1.0e+03  | 0.025 0.025 0.025 0.026 0.026 0.026 0.025 0.025 0.025
-1.0e+04  | 0.027 0.027 0.027 0.027 0.027 0.027 0.027 0.027 0.027
-1.0e+05  | 0.027 0.027 0.027 0.027 0.027 0.027 0.027 0.027 0.027
-1.0e+06  | 0.027 0.027 0.027 0.027 0.027 0.027 0.027 0.027 0.027
```

All 9 directions converge to ~0.027 by |E|=1e4. The linearity is in
the surface Dyson iteration itself, not introduced downstream.

**Test B: bulk Bethe Dyson (`surfGBAt.sigmaK`, 12 directions).** Same
scaling check on the bulk iteration produces the same ~0.027 across
all 12 directions. The linearity is bulk-iteration-intrinsic, not
specific to the surface termination step.

**Test C: de-orthonormalization.** The `surfGB.sigma` SOC branch has

```
sig = lax.cond(self.Sdict['sss'] == 0,
              lambda s: kron(self.Xi, eye(2)) @ s @ kron(self.Xi, eye(2)),
              lambda s: s,
              sig)
```

For AuSOC, `Sdict['sss'] = 0.107 != 0`, so de-orth is NOT on the
active production path. Confirmed by `||sig_prod - sig_manual_no_deorth||`
matching to ~1 part in 1e5 at deep E. Applying the alternate
(de-orth-ON) path manually gives `||sig||/|E| = 0.3864` vs `0.3255`
production -- de-orth amplifies the asymptotic by ~19%, but does NOT
eliminate it. **De-orth is innocent of being the source.**

**Leading-order analytical match.** The Dyson iteration body
`Sigma_k = (E S_k - V_k) g_K (E S_k - V_k)^H` admits a leading-order
fixed point `Sigma_k -> E * S_k S_k^H` at large |E|. Comparing the
observed asymptotic coefficient `X_k = Sigma_k(-1e6)/(-1e6)` to the
prediction:

```
k    ||X_k||_F   ||S_k S_k^H||_F   ratio
0    0.0266     0.0243            1.096
1    0.0266     0.0243            1.096
2    0.0266     0.0243            1.096
3    0.0270     0.0243            1.113
4    0.0270     0.0243            1.113
5    0.0270     0.0243            1.113
6    0.0266     0.0243            1.096
7    0.0266     0.0243            1.096
8    0.0266     0.0243            1.096
```

Match within ~10%. The residual comes from sub-leading corrections in
the resolvent expansion `g_K ~ (1/E)(I - (H_0 + Sigma_tot)/E)^-1`
picking up H_0 and self-coupling at higher order. The iteration is
doing what theory predicts; the asymptotic is the correct fixed point.

---

## 6. Generalization to all non-orthogonal NEGF contacts (Test D)

The mechanism is NOT Bethe-specific. It appears in any non-orthogonal
NEGF surface Green's function iteration of the form

```
g(E) = inv(A - B g B_bar),   A = E * S_alpha - alpha,
                              B = E * S_beta  - beta
```

whenever the inter-cell overlap `S_beta` is non-zero.

Verified directly with `surfG1D` on a toy 1D chain
(`AuStudies/diag_1D_TestD.py`): 3-orbital device with single-orbital
semi-infinite contacts at each end. Per-contact 1x1 matrices
`alpha = [[0]]`, `beta = [[-1]]`, `aS = [[1]]`, with variable
inter-cell overlap `Sbeta = stau` in {0, 0.05, 0.10, 0.20, 0.40}.
Probe `sigma(E, 0)` at `E` in {-1e2, -1e3, -1e4, -1e5, -1e6}.

For the 1x1 case the fixed-point equation has a closed-form solution:

```
g -> X/E   with   X = (1 - sqrt(1 - 4*Sb^2)) / (2*Sb^2)   (retarded branch)
sigma -> E * stau^2 * X
```

Results:

```
stau    numerical ||sigma||/|E| at -1e6    analytical stau^2 * X
0.00    0.000000 (decays as 1/E^2)         0
0.05    0.002506                            0.00251
0.10    0.010101                            0.01010
0.20    0.041741                            0.04174
0.40    0.199985                            0.20000
```

Match to 4+ significant figures across the range. **Orthogonal
coupling (stau = 0) produces NO E-linear asymptotic** -- the iteration
returns `g(E) ~ 1/E`, `sigma ~ 1/E`, and `||sigma||/|E|` decays
like `1/|E|^2`. **Non-orthogonal coupling produces exactly the linear
asymptotic predicted by the fixed-point equation.**

The Total Spectral Weight reported by surfG1D's internal contour
integration at init grows with stau: 3.0 at `stau=0` (exact, 3-orbital
device) to 3.50 at `stau=0.4` (~17% excess weight). Same pseudo-pole
mechanism as the Bethe case, just at small enough scale that it is
tractable analytically.

**Implication:** any non-orthogonal contact in this codebase (surfG1D,
surfG3D, surfGBethe) can in principle trigger the same pseudo-pole
mechanism in calcTSW. The Bethe case was the first failure encountered
because Bethe contacts dominate AuSOC SCF runs; the failure mode is
general to non-orthogonal NEGF formulations.

---

## 7. The heuristic: exclude pseudo-poles by truncating the contour

The mathematics says these pseudo-poles are genuine poles of `(z*S_eff -
F)^-1`, and the contour integral correctly picks up their residues.
Including them in `P` reproduces `Tr(P @ S) = 1049` (the deep-contour
reference for this system). Excluding them gives `Tr(P @ S) = 1365`. The
total device charge is 370.

Both numbers are far from 370 because `Tr(P @ S)` from the contour
integration is *not* the device electron count; it includes contact-tail
contributions that have to be subtracted (typically via the cross-term Q
correction). The difference between the two contours - 315 electrons -
is the pseudo-pole contribution specifically.

**Empirical observation:** the density matrix produced by the
pseudo-pole-INCLUDED contour has eigenvalues that Gaussian rejects
("bad density matrix values"); the SCF fails on the first iteration and
electrons "fly off." The density matrix produced by the
pseudo-pole-EXCLUDED contour is well-conditioned and SCF converges
normally.

**This is the heuristic.** There is no rigorous derivation in this
codebase that says "deep negative-weight residues of the generalized
eigenproblem `(F, S_eff)` should be omitted from the density matrix."
Mathematically they belong, and the contour integral correctly captures
them. Physically they are artifacts of the non-orthogonal Bethe contact's
asymptotic representation - the X_L * E and X_R * E terms in Sigma do
not correspond to physical contact density at energies thousands of eV
below the band; they are the price of representing a semi-infinite
contact with a self-energy that does not vanish at large `|E|`.

The current `calcTSW` excludes them by stopping `Eminf` before it widens
past the pseudo-pole locations. The dropped weight is not bounded
analytically by anything in the current implementation; we simply trust
that "real" core states are accounted for by the time the truncated
contour reaches their location, and that pseudo-poles sit deeper than
those.

---

## 8. Why one-sided convergence is correct

```
dTSW = TSW_ref - TSW_new
if dTSW / |TSW_ref| < tol: converged   # accepts dTSW <= 0 too
```

- `dTSW > tol`: truncated contour missed real spectral weight. Widen.
- `dTSW ~ 0`: truncated contour caught what the reference caught.
  Converged.
- `dTSW < 0`: truncated contour has *more* weight than reference. This
  only happens when the reference enclosed a pseudo-pole the truncated
  contour excluded. Accept the truncated contour - it is the SCF-usable
  one.

A two-sided test `|dTSW|/|TSW_ref| < tol` would detect the `dTSW < 0`
case but then keep widening `Eminf` in an attempt to make the truncated
contour also enclose the pseudo-pole. That produces an SCF-failing density
matrix. The one-sided test is correct: accept `dTSW <= 0` and stop.

When `dTSW < 0` is hit, `calcTSW` prints a `FERMI_DEBUG`-gated note
pointing here. That is informational only - the truncation is the
intended outcome.

---

## 9. Hard floor (`Emin_floor`)

`Emin_floor` (default `ENERGY_MIN = -1e6` eV from `config.py`) serves two
roles:

- It is the contour bound used for the reference `TSW` when no warm-start
  is provided.
- It is a hard limit on the doubling loop. If the next `2 * Eminf` would
  pass `Emin_floor`, `calcTSW` warns and returns. With the one-sided
  test, hitting this warning means either (a) the reference contour does
  not capture all real spectral weight (Emin_floor needs to be lowered)
  or (b) the system has unusually deep core states pushing the doubling
  to its limit before convergence. Case (a) is exotic on the eV scale we
  use; case (b) is the normal failure mode.

---

## 10. Diagnosing a suspected pseudo-pole

If `dTSW < 0` shows up in the debug output for a new system, or
`calcTSW` triggers the floor warning unexpectedly, replicate the deep-E
diagnostic for that system. The script template lives at
`AuStudies/diag_bethe_deepE.py` and `AuStudies/diag_bethe_pseudopoles.py`.
Key things to check:

1. **`||Sigma_L(E)||/|E|` at deep `|E|`.** If it approaches a constant,
   Sigma is asymptotically E-linear and the rest of this story applies.
2. **`S_eff = S - Re(X_L) - Re(X_R)` eigenvalues.** Negative eigenvalues
   are the predictor; their count and magnitude set the number and
   strength of pseudo-poles.
3. **Generalized eigenproblem `(F, S_eff)`.** Deep real eigenvalues with
   `<v, S_eff v> < 0` are the pseudo-poles. Their `<v, S v> / <v, S_eff v>`
   weights, summed, should match the observed `dTSW`.

If the diagnostic shows linear-in-E Sigma but `<v, S_eff v>` is positive
everywhere (no pseudo-poles), the system is in the well-behaved regime
and any `dTSW < 0` is numerical noise.

If Sigma is *not* asymptotically linear in E, the mechanism here does
not apply and the source of `dTSW < 0` is something else - decimation
divergence at deep E (check anti-Hermitian eigenvalues of Sigma stay
non-negative), contour-quadrature failure, or a real issue with the
contact construction.

---

## 11. Fixing it upstream of `calcTSW` (open work)

The real fix is in the non-orthogonal NEGF surface GF construction
itself. Section 6 (Test D) showed the mechanism is general to any
non-orth contact: `surfGBethe`, `surfG1D`, `surfG3D` are all
susceptible whenever inter-cell overlap is non-zero. None of the
following are implemented:

- **Renormalize the asymptotic.** Subtract the asymptotic `X * E` part
  from Sigma before passing it to the resolvent (and add the
  corresponding correction elsewhere). `S_eff` reduces to `S` and the
  pseudo-poles disappear. Applicable to all non-orth surface GFs
  uniformly. `X` is extractable cheaply from a single Sigma evaluation
  at a deep `E` (Tests A+B+C confirmed `||X_k||` converges by
  `|E| = 1e4`).
- **Detect and warn from the surface GF directly.** During or after
  surface GF construction, check whether `S_eff = S - Re(Sigma(E_deep))/E_deep`
  has negative eigenvalues. If non-PSD, emit a warning naming the
  affected contact and the predicted number of pseudo-poles. A single
  Sigma evaluation at `E = -1e6` suffices for both Bethe and 1D
  contacts.
- **Cross-check via real-axis LDOS.** `-Im Tr[G^R(E+i*eta) S]/pi` is
  the LDOS and must be non-negative for any retarded `G^R`. On the
  real axis this stays positive even with pseudo-poles (the pseudo-pole
  residues only "show up" through the contour deformation), but a
  pseudo-pole near `E < mu` should appear as a sharp finite spike in
  LDOS at its location. A quick LDOS sweep over `[Emin_floor, mu]` at
  contact setup time could surface them before SCF.
- **Analytical fixed point as initial guess.** For 1x1 channels the
  fixed point has a closed form `X = (1 - sqrt(1 - 4 Sb^2))/(2 Sb^2)`
  (see section 6). For matrix S_beta the corresponding generalized
  problem `X = (S_alpha - S_beta X S_beta.H)^{-1}` could be solved
  once at contact setup and used as the analytical asymptotic in the
  renormalization above.

The current state of the code is: `calcTSW`'s one-sided test excludes
the pseudo-poles from `P`, which is sufficient to make SCF converge,
but the underlying issue in the non-orthogonal surface GF iteration
is not fixed and could in principle cause harder-to-detect distortions
to `P` even in cases where dTSW happens to look fine.

---

## 12. Configuration

The relevant config knobs in `gauNEGF/config.py`:

```
ENERGY_MIN              # eV - default Emin_floor for calcTSW (MUST BE NEGATIVE)
FERMI_SEARCH_CYCLES     # max doubling iterations
FERMI_CALCULATION_TOL   # convergence tolerance for relative TSW change
FERMI_DEBUG             # if True, prints per-iteration Eminf and dTSW,
                        # and the dTSW<0 pseudo-pole notice
```

`Emin_floor` is also a kwarg on `calcTSW` for per-call override.

---

## 13. Appendix: SOC implementation bug discovered during this work

During the investigation of the E-linear Sigma asymptotic, an
independent basis-convention bug was found in
`gauNEGF/spinTools.py:constructSOCterm`. It is described here because
it affected all AuSOC SCF runs that contributed to the diagnostic
results in section 4, and any historical AuSOC physics results
predating the fix should be re-validated.

**What was wrong (two bugs that needed to be fixed together):**

1. `genLSMatrix` transformed L_x, L_y, L_z from spherical to real
   orbital basis with the wrong direction:
   `V.conj().T @ L @ V` instead of `V.conj() @ L @ V.T`. This
   produced a matrix that was NOT L in the real orbital basis -- in
   particular it had nonzero diagonal entries, violating
   `<real_i | L_alpha | real_i> = 0` for the standard real orbital
   conventions.
2. The L.S matrix was assembled via
   `np.block([[Lz, L-], [L+, -Lz]])`, which produces a *spin-major*
   layout (first all-orbitals-spin-up, then all-orbitals-spin-dn). But
   downstream it was added to `kron(H0, eye(2))`, which is
   *interleaved* (orbital o -> indices [2o, 2o+1]). The two layouts
   differ on the p and d blocks, so SOC matrix elements landed on
   wrong orbital pairs.

**Why it wasn't caught:** the existing `test_constructSOCterm.py`
contained a *local copy* of the same buggy `genLSMatrix` and tested
`constructSOCterm` against that local reference -- circular
validation that passed regardless of correctness. The eigenvalue tests
(`test_eigenvalues_p_orbital`, `test_eigenvalues_d_orbital`) checked
the j-j coupling spectrum (`4x(+0.5), 2x(-1.0)` for p,
`6x(+1.0), 4x(-1.5)` for d) which is invariant under any unitary
basis transform -- so a wrong-basis L.S still passes the spectrum
check. None of the tests in `tests/test_soc.py` tested basis-
specific structure either.

**Why this is independent of the pseudo-pole mechanism:** the bug
affected only the on-site Hamiltonian `H_0` (via `kron(H0, eye(2)) +
Hsoc`). The E-linear Sigma asymptotic is set by the contact inter-cell
overlap `S_k`, which is built via `constructMat(..., SOC=True) =
kron(M_rot, eye(2))` -- correctly interleaved. At deep `|E|`, `H_0`
becomes negligible relative to `E*S_k` in the Dyson body, so the
leading-order `||Sigma||/|E|` value is independent of the bug. The
diagnostic conclusions in sections 4-6 hold post-fix.

**Effect on physics:** SOC matrix elements were being added to the
wrong (orbital, spin) pairs of `H_0`, contaminating the SOC physics
of the contact onsite Hamiltonian. Fermi level, DOS, magnetic
moments, and charge densities computed under SOC are all affected.
For Au with `soc_d = 2.7e-2 Ha = 0.735 eV`, the d-block SOC
contributions to `H_0` were sizeable. The plausibility of SCF
fragility we have seen with AuSOC systems (where pure-Au runs with
`Au.bethe` were stable but AuSOC runs needed careful starting points
and sometimes wandered) is consistent with this bug introducing
spurious orbital mixing into the contact onsite Hamiltonian.

**The fix:**

```python
def genLSMatrix(l):
    Lx, Ly, Lz = LOps(l)
    V = genOrbList(l)
    Lx = V.conj() @ Lx @ V.T     # FIX 1: corrected transform direction
    Ly = V.conj() @ Ly @ V.T
    Lz = V.conj() @ Lz @ V.T
    # FIX 2: interleaved (orbital-major) ordering via kron
    return 0.5 * (np.kron(Lx, sigx)
                 + np.kron(Ly, sigy)
                 + np.kron(Lz, sigz))
```

**Verification:** the fixed L.S matrix satisfies (in `tests/test_constructSOCterm.py`):

- Hermitian, traceless, s-block zero (basis-invariant; passed before
  and after fix)
- p-block eigenvalues `4x(+0.5), 2x(-1.0)`; d-block `6x(+1.0), 4x(-1.5)`
  (basis-invariant; passed before and after)
- *NEW* basis-convention-sensitive tests, all of which would have
  caught the bug:
  - `test_diagonal_zero_in_real_basis`: `H[i,i] = 0` for all i
    (catches Bug 1)
  - `test_interleaved_intra_orbital_spin_flip_zero`: `H[2i, 2i+1] = 0`
    for all i (catches Bug 2)
  - `test_interleaved_consistent_with_kron_h0`: `diag(kron(H0, I2) + Hsoc)`
    matches `kron(diag(H0), [1,1])` (catches both together)
  - `test_d_block_no_spin_block_segregation`: 5x5 d-sub-block norms
    are comparable, not segregated (catches Bug 2)

All 12 tests in `test_constructSOCterm.py` and all 25 (non-slow) in
`test_soc.py` pass with the fix.

**Recommended follow-up:** re-run any AuSOC SCF results currently
relied on for physics conclusions, and compare to the buggy outputs
to assess sensitivity. The fix changes the on-site H, so the
absolute Fermi level and density values shift; the E-linear Sigma
story documented above does not.

---

## 14. Production detection function

The investigation in sections 1-13 motivated `calcPseudoPoleFloor` in
`gauNEGF/density.py`. See `docs/pseudo_pole_handling.md` for the math and
AuBetheFerrocene empirical verification.

Signature: `calcPseudoPoleFloor(F, S, g, alpha=0.1, min_buffer=50.0, E1=-1e3, E2=-1e4)`.
It does a two-point asymptotic probe of `g.sigmaTot` to extract X and Sigma_0,
forms `(H_eff = F + Sigma_0, S_eff = S - X)`, reduces to the standard
non-Hermitian eigproblem `T = S_eff^{-1} H_eff` (so indefinite S_eff is
handled correctly), classifies each eigenvector by Hermitian S_eff-norm
`v^H S_eff v < 0`, and returns a buffered floor above the shallowest
negative-energy pseudo-pole. All matrices are treated as complex Hermitian,
so this works for GHF/SOC as well as the real-symmetric case.

The production SCF path (FockToP / damleLowerDensity) does not call
`calcPseudoPoleFloor` or `calcTSW`; instead FockToP warns when the lower
contour holds significant weight, which is the corresponding signal that
pseudo-pole content has entered the integration window. `calcPseudoPoleFloor`
is available for direct use in fixed-grid or custom workflows.

Tests: `tests/test_pseudo_pole_detection.py` (unit tests including a
general-coupling test that catches the failure mode of a whitening-only
approach, and a complex-Hermitian test that catches a `.real`-only
approach) and `tests/test_calcTSW.py::test_calcTSW_with_pseudo_pole_floor`
(regression test on the Au Bethe-lattice fixture).
