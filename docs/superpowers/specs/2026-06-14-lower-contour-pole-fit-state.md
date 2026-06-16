# Lower-contour analytic method -- working state (2026-06-14)

Scratch notes so the context survives a /compact. NOT a final spec.

## Goal
Analytic (no-integration) lower-contour **density + cross-term** for equilibrium
NEGF that captures the in-band continuum tail, using ONLY `Sigma(E)` / `Q(E)`
samples + device `F, S`. Lead `alpha`/`beta` matrices are NOT accessible for all
surfG objects. Constraints: run via sbatch only (login node), no commits, ASCII.

## Toy (validation system)
2-site device + two semi-infinite 1D leads, built directly as a surfG (pattern in
`tests/fixtures/_diag_lower_contour_tb_benchmark.py`). Fat-tail params: `b_c=-40`.
- lead band bottom = `-50.0` eV  (= `(a_c + 2 b_c)/(1 + 2 s_c)`)
- gate `Emin = -20.909` eV
- TRUE lower-contour count (converged truncated contour): bulk `tr(P S) = 0.0634`,
  cross `delta_N = -0.0748`, net `-0.0114`.

## Findings (established by runs)
1. Direct integration over `[-1e6, Emin]` does NOT converge (6 orders of magnitude;
   oscillates ~1e-3). Truncating to `[band_bottom - buffer, Emin]` DOES converge
   (densityComplex 54 pts, densityReal 162-486 pts) and recovers `0.0634`.
   (densityReal cross-term came out sign-flipped vs densityComplex -- unresolved.)
2. Damle const/linear with the asymptotic REAL `Sigma_0` returns `~0`: the model has
   discrete poles, no continuum. `density()` returns P proportional to pole
   OCCUPATIONS (the `invmat` 1/(D_i - D_j*) cancels the Gam factor) -- NOT prop. to
   Gam (earlier claim was wrong).
3. `Im Sigma`, `Im Q` turn on at the band edge (`-50`) and drop ~linearly toward
   `Emin` (lin-fit resid 2.6% / 5.1% on the drop; the rise from edge to peak is a
   non-linear van Hove). Only the imaginary parts contribute to the count/cross-term.
4. Band edge = generalized eig of the lead Bloch Hamiltonian `(alpha + 2 beta,
   aOv + 2 bOv)` = `-50`. NOT a device-`Fbar` eigenvalue (`eig(Fbar) = +3.16, +6.81`
   asymptotic; `-1.25, +0.70` in-band). `Fbar` carries the contact SELF-ENERGY
   (frozen), not the contact DISPERSION.
5. `Q(E)` and `Sigma(E)` share `g_s(E)` (the lead surface GF); their imaginary parts
   on the real axis are the lead spectral density (branch cut along the band).
6. Device states / bound states come from `eig(Fbar)` (uses current `F`, SCF-safe).
   Lower bound for any truncation = `min(band_edge, min eig Fbar) - buffer`.

## The method being proven: pole-fit / auxiliary-mode expansion
Generalize Damle: instead of freezing `Sigma` to a constant (-> no continuum), fit
the SAMPLED `Sigma(E)` to poles and embed them as auxiliary modes.

    Sigma(E) ~ A + sum_k V_k V_k^dag / (E - p_k)      [rational fit of Sigma samples]

Exact Schur complement of an augmented Hamiltonian:

    H_aug = [[F + A,   V       ],     S_aug = blockdiag(S, I)
             [V^dag,   diag(p_k)]]

so device `G^R` = top-left block of `[(E)S_aug - H_aug]^{-1}` (no approx beyond fit).
Then a closed-form residue sum (no integration):

    lam, R = gen-eig(H_aug, S_aug);  M = (S_aug R)^{-1};  Sbar = blockdiag(S, 0)
    c_j = diag(M Sbar R)
    count = -(1/pi) Im sum_j c_j [ log(Emin - lam_j) - log(ENERGY_MIN - lam_j) ]

The fitted `Sigma` poles `p_k` (complex) carry the band -> captures the continuum the
bare device poles couldn't. Uses only `Sigma(E)` + `F, S`. Cross-term: fit `Q(E)` the
same way; `Tr(G^R Q)` is again a residue sum.

## Validation status -- DENSITY PROOF PASSED (2026-06-14)
Prototype `diagnose_pole_fit` in the benchmark: fit `Sigma[0,0]` to `A + K` poles
(fixed poles, lstsq residues), embed, residue-sum over [band_bottom, Emin], compare
to REF bulk `0.063425`. Result:

    K=4:  count 0.063135  (REF - 2.9e-4)   sum(c_j)=2.0000
    K=8:  count 0.063056  (REF - 3.7e-4)   sum(c_j)=2.0000
    K=16: count 0.063008  (REF - 4.2e-4)   sum(c_j)=2.0000
    K=32: count 0.063064  (REF - 3.6e-4)   sum(c_j)=2.0000

Recovers the continuum count to ~0.5% (REF-noise level) with as few as K=4 poles.
sum(c_j)=2 confirms the embedding (total device weight = device dim). Same 0.063 the
discrete-pole Damle returned ~0 for. Uses ONLY Sigma samples + F, S (no alpha/beta,
no integration).

KEY BUG FOUND + FIXED: residue sum must integrate [band_bottom, Emin], NOT
[ENERGY_MIN, Emin] -- the fitted poles' tails extend (spuriously) below the band
edge; integrating them subtracts ~0.04 and gives a wrong ~0.02. The true
contribution below band_bottom is zero (Im Sigma = 0), so restricting the integral
is both correct and necessary.

## SUPERSEDED DIRECTION -- the pole-fit is shelved (new math, doesn't scale)
The augmented pole-fit eig is bigger-than-device and is a NEW integration method.
User reset: fix the lower bound for the EXISTING integrator, no new math. Overlap is
ALWAYS PSD (enforced) -> the "pseudo-pole" is a red herring; every pole of
[ES - F - Sigma(E)]^{-1} is a real state.

## SOLUTION (validated 2026-06-14): inertia finder + tight existing contour
find_lowest_pole(F, S, g, mu): bisection on the INERTIA of M(E) = E S - F - Sigma(E).
'below the lowest pole' <=> n_neg(M) == N AND ||Im Sigma|| < thresh. n_neg is MONOTONE
in E because the overlap (and S_eff) is PSD -> cannot finish early like calcEmin's DOS
scan. Single M(E) calls, no integration. Then densityComplex([lowest_pole - buffer, mu])
-> BOTH density and cross-term, existing machinery.

Validated on the toy:
  main toy:    lowest pole = -50.0001 (band bottom, exact)
               TIGHT [-55, mu]:  converged 1.8e-8 in 162 pts
               HUGE  [-1e6, mu]: grid cap 486 pts, error 1.7e-3 (NOT converged)
               tight tr(PS)=1.066140, dN=-0.089569 ; matches huge to 1.1e-5
  planted -70: lowest pole = -70.0015 (CAUGHT the bound state below the band)
               tr(PS)=2.066 (the +1 bound state correctly included)

So: robust lower bound (catches bound states calcEmin missed; monotone, no early-finish),
existing integrator, density + cross-term together, ~3x fewer grid points AND it
converges where [-1e6, mu] caps out. No pseudo-pole (never integrate to ENERGY_MIN),
no TSW doubling, no new math.

calcEmin (density.py:1018-1022) = non-monotone DOS scan -> finishes early.
calcTSW (density.py:1103-1127) = TSW doubling w/ reference at ENERGY_MIN -> straddles
the deep artifact. The inertia finder replaces both.

## Remaining
- Scaling check: n_neg is one N x N inertia (LDL) per bisection step (~20-60 steps),
  cheaper than the integrator's hundreds of solves -- verify on a large system.
- Band bottom for complex/multi-band leads: thresh on ||Im Sigma|| is the continuum
  onset; confirm robust when the lead has structure.
- densityReal vs densityComplex cross-term SIGN discrepancy still unresolved.
- Pole-fit (shelved): only revisit if a no-integration method is ever wanted.
