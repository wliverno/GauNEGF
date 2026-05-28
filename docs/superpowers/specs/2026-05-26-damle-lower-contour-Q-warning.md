# Spec: Damle lower contour with analytic cross-term + Q-warning (no calcTSW)

Date: 2026-05-26. Status: finalized (user-approved).
Math reference: docs/lower_contour_math_and_probes.md (derivations, eqs cited here).

## 1. Goal

Construct a well-defined equilibrium device density matrix with a clean,
self-contained split:

  - lower contour [ENERGY_MIN, Emin]: analytic Damle (one eig), builds the
    matrix AND an analytic cross-term delta_N;
  - upper contour [Emin, mu]: densityComplex (always);
  - Emin is placed by calcEmin's true-Sigma DOS loop (cheap, robust); calcTSW
    and calcPseudoPoleFloor are dropped from the path;
  - a WARNING fires when the lower-contour cross-term delta_N is anomalously
    large -- the diagnostic that replaces calcTSW's dTSW<0 detection.

Rationale (empirically established, Section 8): when Emin sits below all poles
the lower contour holds negligible charge (DOS ~ 1/E^2, doc eqs 4-6), so the
lower contour is normally ~0; a large delta_N means real weight slipped below
Emin, which the warning surfaces. Analytic Damle costs one eigendecomposition
and -- unlike densityReal, which silently returns zero on sharp sub-Emin poles
-- captures any sub-Emin spectral weight exactly.

Why calcEmin (true-Sigma DOS) places Emin, not the damle eig: the damle
estimate min Re eig(Fbar) - buffer uses the CONSTANT Sigma_0, but the true G^R
poles solve det(E*S - F - Sigma(E)) = 0 with energy-dependent Sigma(E); near
the band Sigma(E) != Sigma_0, so Fbar's lowest eigenvalue is not the true
lowest pole, and a fixed buffer only guesses at the gap. This is worst exactly
when S_eff = S - X is non-PSD (the diffuse-basis case): Y_eff is then
ill-conditioned and Fbar's eigenvalues are unreliable. calcEmin evaluates the
true Sigma(E) via g.sigmaTot, sees the real poles, and is robust to all of
this; it is cheap (a handful of G^R inversions) and converges in ~1 step when
seeded near the device-spectrum floor.

## 2. Architecture

```
setContact1D / setContactBethe / setSigma
    -> _initAsymptoticSigma()                    # 2-probe fit -> Sigma_0, X_asymp, S_eff, Y_eff
    -> setIntegralLimits(): Emin = calcEmin(...)  # true-Sigma DOS loop (NO calcPseudoPoleFloor / calcTSW)

FockToP()
    lower [ENERGY_MIN, Emin]:  P_lo, delta_N = damleLowerDensity(F, Y_eff, Sigma_0, Emin, ...)
                               warn if |delta_N| > damle_dN_warn   (default 0.5 e)
    band  [Emin, mu]:          P_hi = densityComplex(F, S, g, Emin, mu)
    P = P_lo + P_hi
    electron count uses tr(P S) + delta_N
```

One eig for the lower contour; one smooth densityComplex contour for the band
(accurate even when wide -- the contour bulges far from the poles, which makes
the integrand smoother, not harder).

## 3. Components and changes

### 3.1 `_initAsymptoticSigma` -- 2-probe relative fit  [DONE]

Implemented and validated. Fits Sigma(E) = X_asymp*E + Sigma_0 from two DEEP
probes placed RELATIVE to the integration window, dropping the hardcoded
(-1e3, -1e4, -1e5):

  E1 = ENERGY_MIN
  E2 = (Emin_est + ENERGY_MIN) / 2          # Emin_est = min eig(inv(S)F) - EMIN_BUFFER (jax/GPU)

Two points fit the two-parameter line exactly; the third probe was only a
linearity residual and is dropped (doc sec 7, eqs 12-13). Validated:
||X_asymp|| identical to the old 3-probe fit, ||Im X||/||X|| and the
Y_eff S_eff Y_eff = I defect 3-4 orders CLEANER (deeper probes are more
asymptotic). See Section 8.

### 3.2 `damleLowerDensity` -- sign fix + analytic cross-term + warning

New signature: `damleLowerDensity(F_eV, Y_eff, Sigma_0, Emin, ...)` taking Emin
as an INPUT (from calcEmin) and returning `(P_lower, delta_N)` -- it no longer
computes or returns Emin.

(a) SIGN FIX. The analytic density() result is globally sign-flipped relative
    to the trusted densityComplex (confirmed: sign-corrected matrix agreement
    0.76% on the C2 cores; identical for the -1e6-bound and finite-bound
    intervals, so it is a sign convention, not a log-branch artifact). Fix by
    negating the Damle lower-contour density so it matches densityComplex.
    LOCALIZE the negation to the damle path (not a global edit to density()),
    and verify bisectFermi -- which also calls density() in the predict-Fermi
    path -- is unaffected (it uses density() in a self-consistent count ratio;
    confirm the sign cancels or correct it there too).

(b) ANALYTIC CROSS-TERM delta_N. Reuse the SINGLE eig already computed for the
    density matrix (Fbar = V D Vc^H). No second eigendecomposition (doc sec 5).
    Linearize Q(E) ~ Q0 + Q1*E from crossTermQTot at two anchors NEAR THE BAND
    BOTTOM, (Emin, 2*Emin); form b(E) = Vc^H Y_eff Q(E) Y_eff V = B0 + B1*E with
    Ml = Vc^H Y_eff, Mr = Y_eff V precomputed. Then:

      delta_N = -(1/pi) * Im( sum_i (B0_ii + B1_ii*D_i) *
                              [log(1 - hi/D_i) - log(1 - lo/D_i)] )

    The b1*(hi-lo) "flat" term from the partial-fraction split is OMITTED: it is
    the unphysical background of the linear-Q model (real Q does not grow
    linearly into the deep tail) and is the source of the cross-term artifact (a
    spurious +8.9e-3 on a pole-free tail). Dropping it makes delta_N
    window-independent and exact for the physical in-window-pole contribution:
    pole-free tail -> delta_N ~ 0; sub-Emin pole -> correct nonzero value.

    Anchor choice (Emin, 2*Emin) is near-band because, with the flat term gone,
    the cross-term evaluates Q effectively at the in-window pole energies, and a
    pole that slips below Emin (the anomaly) sits just below Emin -- so near-band
    anchors interpolate it rather than extrapolating from the deep Sigma_0
    probes. Low-stakes: in the normal case there are no in-window poles and
    delta_N ~ 0 regardless of anchors.

    This SUPERSEDES the prior claim that the cross-term is "subsumed into the
    S_eff framework" (test_damle_seff_gate). It is not subsumed; it is ~0 only
    because a correctly-placed Emin leaves the lower contour empty. It must be
    computed.

(c) WARNING. The warning keys on the TOTAL lower count, not |delta_N| alone.
    Because calcEmin places Emin BELOW the whole band, the lower contour should
    hold ~zero charge (Damle uses the constant asymptotic Sigma_0, valid only in
    the deep tail). After computing the lower density, if ANY of the three terms
    -- the bulk tr(P@S), the cross-term delta_N, or their sum tr(P@S)+delta_N
    (the lower electron count) -- exceeds self.damle_dN_warn in absolute value
    (new instance attribute, default 0.5 electrons; configurable), print a
    warning that significant spectral weight lies in the lower contour (Emin too
    shallow, or a pseudo-pole present), where Damle's constant-Sigma_0
    approximation is suspect. An empty lower contour gives ~0 on all three; a
    trapped real/core pole gives a large tr(P@S), and a trapped lead-coupled
    pseudo-pole drives delta_N and the sum large (and the sum negative). This is
    the diagnostic replacing calcTSW's dTSW<0. (Empirically on C2-STO3G: Emin
    below the cores gives all terms ~1e-6; Emin at -145 traps the 2 core
    electrons -> tr(P@S)~2.0 fires; Emin in the valence band -> sum goes
    negative. See tests/fixtures/_diag_deltaN_vs_emin.py.)

    Note: the attribute name damle_dN_warn is retained for continuity, but it
    now thresholds all three terms, not delta_N alone.

### 3.3 `FockToP` -- wiring

PER-CYCLE RECOMPUTE (critical). self.F changes every SCF cycle, and the
asymptotic fit (Sigma_0, S_eff, Y_eff) AND Emin all depend on it. So at the top
of the default (N2=None) path FockToP MUST recompute them on the current Fock:
  self._initAsymptoticSigma()            # refit Sigma_0 / S_eff / Y_eff
  self.Emin = calcEmin(self.F*har_to_eV, self.S, self.g, tol=self.tol)
This restores the long-standing pre-Damle behavior (FockToP re-placed Emin every
call). Freezing them at setup is a bug: a stale Y_eff/Sigma_0 applied to an
evolved F yields a garbage Fbar with spurious sub-Emin eigenvalues, which Damle
counts as huge spurious lower charge -> nLower wrong -> the band Fermi search
asserts (observed: Au3-CRENBS SCF crash, job 35623534). C2 happened to survive
the frozen bug only because its Emin sits ~270 eV below the band (huge margin).

Then the lower contour: call damleLowerDensity(F, Y_eff, Sigma_0, g, self.Emin)
to get (P_lo, delta_N); compute trLower = tr(S P_lo) and nLower = trLower +
delta_N; emit the warning if any of |trLower|, |delta_N|, |nLower| >
self.damle_dN_warn; set the band via densityComplex[self.Emin, mu]; total
density P = P_lo + P_hi; lower electron count uses nLower = tr(P_lo S) + delta_N
(delta_N now real, not discarded). The deprecated fixed-grid (N2 not None) path
is untouched.

Emin uses calcEmin as-is (true-Sigma DOS, fresh each cycle). The "use the Damle
estimate (Fbar floor) as the Emin guess" refinement was investigated and found
UNNECESSARY: with a fresh fit, the eig(Fbar) floor ~= the true band floor
(within ~0.5 eV on both C2 and Au3), so calcEmin already lands below the Fbar
band. The deep-Fbar-mismatch this would have addressed does not occur for PSD
S_eff. See Section 9.

### 3.4 `setIntegralLimits` -- drop calcTSW + calcPseudoPoleFloor (keep calcEmin)

In the default (Emin is None, tol given) path, remove the calcPseudoPoleFloor +
calcTSW steps. KEEP calcEmin: self.Emin = calcEmin(F_eV, S, g, tol) places Emin
below the band using the true energy-dependent Sigma (the robust DOS loop). The
explicit-Emin and fixed-grid paths are unchanged.

calcTSW, calcPseudoPoleFloor (and the unused Eminf/TSW machinery) REMAIN in the
module as debugging tools for future tests; they are noted as needing fixing and
are not called by the default path. (calcEmin stays live as the Emin source.)

## 4. Cross-term math (the Q improvement), condensed

G^R(E) = Y_eff (E I - Fbar)^{-1} Y_eff, Fbar = V D Vc^H (the density's eig).
Tr(G^R Q) = sum_i b_i(E)/(E - D_i), b(E) = Vc^H Y_eff Q(E) Y_eff V. With
Q(E) ~ Q0 + Q1 E and the split b0 + b1 E = b1*(E-D) + (b0 + b1 D):

  integral_lo^hi (b0 + b1 E)/(E-D) dE
      = b1*(hi-lo)                                  <- OMIT (linear-Q artifact)
      + (b0 + b1 D)*[log(1-hi/D) - log(1-lo/D)]     <- KEEP (physical pole term)

Im[log(1-hi/D) - log(1-lo/D)] = pi if Re(D) in (lo,hi), else 0 (doc eq 11), so
the kept term counts exactly the spectral weight inside the window.

## 5. Migration sequence (working system at every step)

1. [DONE] _initAsymptoticSigma 2-probe relative fit + jax eig. Validated
   (Section 8).
2. damleLowerDensity: take Emin as input; add sign fix + analytic cross-term
   (drop-flat-term, near-band anchors) + return delta_N + warning. FockToP can
   ignore the new delta_N initially (still works). Unit test: sign-corrected
   agreement < 1% vs densityComplex on a cores-in-window fixture; delta_N ~ 0 on
   a pole-free tail.
3. setIntegralLimits: drop calcPseudoPoleFloor + calcTSW from the default path
   (keep calcEmin). System works (Emin still defined by calcEmin).
4. FockToP: consume delta_N in the count and emit the warning. System works.
5. SCF validation: run an SCF and confirm DOS(Emin) < tol (Emin below the band,
   does not skip up near a pole), on a PSD case and a non-PSD case -- the
   non-PSD case confirms the true-Sigma DOS check catches what the damle eig
   would miss.

## 6. Decisions (resolved)

1. Emin source: calcEmin's true-Sigma DOS loop (NOT the damle eig). Robust,
   cheap, and correct under non-PSD S_eff. [confirmed]
2. damle_dN_warn default: 0.5 electrons, configurable per instance. [confirmed]
3. Q linear-fit anchors: (Emin, 2*Emin), near the band bottom (distinct from the
   deep Sigma_0 probes; near-band is correct for the cross-term, low-stakes).
   [confirmed]
4. calcTSW / calcPseudoPoleFloor / calcEmin: left in the module. calcEmin stays
   live (Emin source); calcTSW + calcPseudoPoleFloor are retained as debugging
   tools, flagged as needing fixing, unused by the default path. [confirmed]
5. Sigma_0/Y_eff and Emin are recomputed EVERY FockToP call on the current Fock,
   not frozen at setup (they all depend on self.F, which changes each SCF cycle).
   This is the pre-Damle behavior; the Damle rewiring had regressed it. [confirmed]
6. Emin guess: plain calcEmin (true-Sigma DOS), NOT the Fbar-floor estimate.
   For PSD S_eff the Fbar floor ~= the true floor, so calcEmin already lands
   below the Damle band; the Fbar-guess refinement is unnecessary. [confirmed]

## 7. Non-goals

- The non-PSD diffuse-basis contact-overlap problem (k-resolved regularization)
  is OUT of scope; tracked separately. This spec targets the minimal-basis /
  ECP regime where S(k) is PSD. (The non-PSD SCF test in step 5 is only to
  confirm Emin placement is robust, not to fix non-PSD transport.)
- Corollary (learned during validation): a non-PSD S_eff makes Y_eff complex and
  the Fbar spectrum meaningless, so the Damle lower contour is garbage. There is
  therefore no point trying to make the Damle contour "work" on a non-PSD S_eff
  -- any future non-PSD correction MUST restore S_eff PSD as a precondition.
  Both in-scope systems (C2-STO3G, Au3-CRENBS) have PSD S_eff (Section 9).
- No change to the band integrator (densityComplex stays).
- densityReal is not used for the lower contour (it silently zeros on sub-Emin
  poles; Section 8).
- No fix to calcTSW / calcPseudoPoleFloor (retained as-is for debugging).

## 8. Validation evidence (bake-off, jobs 35562575 / 35580937)

Fixture tests/fixtures/_diag_lower_tail_methods.py on C2-STO3G (1D contact,
1s cores) and Au3-CRENBS (Bethe, ECP, g-spin):

- 2-probe fit: ||X_asymp|| identical to 3-probe (slope robust, eq 12);
  ||Im X||/||X|| 3.0e-12 -> 9.3e-16 (C2), 8.5e-9 -> 2.7e-12 (Au3);
  Y_eff S_eff Y_eff = I defect at machine precision. Probe change is safe and
  cleaner.
- Deep tail negligible: ||P_lower||/||P_band|| = 3.7e-5 (C2), 6.4e-5 (Au3) when
  Emin is below all poles; all three methods agree at ~0.
- damle matrix vs densityComplex on the cores (C2 placement B and C): sign-
  corrected ||(-P)-P_complex||/||P_complex|| = 0.76%, identical for -1e6 and
  finite lower bounds -> global sign convention, not a log-branch artifact.
- densityReal returns ne ~ 0 on a cores-in-window interval (silent miss) ->
  unfit for any pole-containing lower contour.
- Cross-term artifact: with the b1*(hi-lo) flat term included, damle delta_N =
  +8.9e-3 on a pole-free tail (should be ~0); dropping the flat term (Section 4)
  removes it.
- Timing (synchronous median): damle ~0.03 s vs densityComplex ~1-15 s; damle
  is 30-450x faster (one eig vs an adaptive contour loop).

## 9. Provenance

- Math: docs/lower_contour_math_and_probes.md
- Prior design: docs/superpowers/specs/2026-05-22-damle-lower-contour-design.md
- Bake-off fixture: tests/fixtures/_diag_lower_tail_methods.py (jobs 35562575,
  35580937); slurm-35580937.out
- 2-probe implementation: gauNEGF/scfE.py:_initAsymptoticSigma (committed work
  in progress)

### Implementation run (2026-05-27, subagent-driven)

- Task 1 damleCrossTerm unit tests: jobs 35593602 (fail, ImportError), 35593629
  (2 passed). gauNEGF/density.py, tests/test_damle_cross_term.py.
- Task 2 damleLowerDensity rework (sign fix + cross-term + take Emin/g): jobs
  35594112 (fail), 35594768 (2 passed). Also fixed a latent bug in commit
  63f7f54: _initAsymptoticSigma had a local `import jax.numpy as jnp` AFTER an
  earlier jnp use, raising UnboundLocalError (the committed milestone could not
  run _initAsymptoticSigma). Removed the local import (module-level jnp exists).
- delta_N vs Emin sweep (C2-STO3G): job 35614159,
  tests/fixtures/_diag_deltaN_vs_emin.py. Showed |delta_N| alone is a poor signal
  (cores below Emin give tr(P@S)~2.0 but delta_N~5e-6). Drove the warning
  criterion to the TOTAL count: warn if any of |tr(P@S)|, |delta_N|, |sum| > 0.5.
- Task 3 FockToP warning (total-count criterion) + damle_dN_warn: jobs 35614033
  (1 failed, old delta_N-only premise), 35615592 (4 passed, new criterion +
  no-false-positive test). gauNEGF/scfE.py, tests/test_damle_lower_contour.py.
- Task 4 drop calcTSW/calcPseudoPoleFloor (keep calcEmin; Eminf=ENERGY_MIN,
  TSW=None, self.tol kept): jobs 35615712 (fail, Eminf assertion), 35615848
  (5 passed). gauNEGF/scfE.py.
- Task 5 SCF validation: job 35615896,
  tests/fixtures/_diag_damle_scf_validate.py.
  VERDICT: Emin-below-floor PSD=True, non-PSD=True.
    [C2-STO3G (PSD)]      Emin=-281.414 eV, band floor=-271.317 eV, below: True
    [C2-LANL2DZ (non-PSD)] Emin=-324.974 eV, band floor=-280.971 eV, below: True
  C2-STO3G FockToP completed; electron count 12.000 (bisectFermi sign sane);
  NO spurious lower-contour warning fired in either run (warning silent in
  normal operation, as designed). Non-PSD: X_asymp Im/||X||~0.51 (flagged
  non-PSD) but calcEmin still placed Emin below the band.

### Full-SCF convergence + the staleness root cause (2026-05-27)

- Full SCF (negf.SCF), job 35623534, tests/fixtures/_diag_minimal_basis_scf.py:
  C2-STO3G CONVERGED in 10 iterations (convLevel 5.9e-5). Au3-CRENBS CRASHED at
  iter ~6: calcFermiMuller asserts ne < len(g.F) because the lower contour
  reported tr(P@S)=30.5, delta_N=-97.7 (nLower=-67) -> ne-nLower ~ ne+67.
- Spectrum comparison (setup, true vs Damle), job 35626565,
  tests/fixtures/_diag_spectrum_compare.py:
    C2-STO3G : true floor -271.32, Fbar floor -270.14, Emin -281.41; 0 poles
               below Emin (both); S_eff 0/10 neg eig (PSD); ||Im Y_eff||/||Y|| 1e-16.
    Au3-CRENBS: true floor -11.88, Fbar floor -11.36, Emin -16.89; 0 poles below
               Emin (both); S_eff 0/54 neg eig (PSD); ||Im Y_eff||/||Y|| 1e-13.
  => At setup BOTH systems are consistent (Emin below an empty contour), S_eff is
  PSD, the two spectra agree to ~0.5 eV. This REFUTED the earlier "Sigma_0 shifts
  Fbar deep" hypothesis.
- Corrected root cause: the Au3 crash is STALENESS. FockToP had frozen
  Sigma_0/Y_eff/Emin at setup; as self.F evolved over SCF cycles, the stale
  transform Y_eff_0 (F_cycle + Sigma_0_0) Y_eff_0 produced a garbage Fbar with
  spurious eigenvalues below the frozen Emin -> spurious lower charge -> crash.
  Fix: recompute Sigma_0/Y_eff + Emin every FockToP (Section 3.3). C2 survived the
  frozen bug only via its ~270 eV Emin margin.
