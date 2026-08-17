# Non-equilibrium Mulliken cross term: design spec

Date: 2026-08-13
Branch: fix/noneq-crossterm (worktree ../NEGFCode-crossterm, base b7c31fa)
Source of truth for the math: the final boxed equations of
  paper/derivations/derivation.tex
Supersedes: docs/superpowers/plans/2026-08-05-negf-nonequilibrium-fixes.md
  (archived at paper/planning/archive-negfcode-fix/; salvage tasks from it
  only where this spec says so)

## Goal

STATUS QUO, stated precisely: the EQUILIBRIUM cross-term count is
implemented and provably working (contour F(z) path with Q_sym; in
production on every converged run). The NON-EQUILIBRIUM cross term
does not exist in the code at all - the bias window today integrates
only the device block. This work ADDS the missing non-equilibrium
count; the equilibrium path is protected, not repaired.

Implement the derived non-equilibrium electron count for non-orthogonal
bases: the bias-window integral gains the cross-term kernel pieces
that are currently absent (T13), with the dFermi shift threaded as a traced JAX
argument so kernels compile once (T12), the forward-axis sign errors and
their hazardous documentation corrected together (T15), and the
transport.py bias convention aligned with the SCF (T14, one task).

Verification is by parameter-free identities. No
test may contain a fitted sign, tolerance-tuned constant, or
"measured then pinned" value.

## Decisions (settled 2026-08-13)

- Single reference decomposition. The two decompositions
  are analytically identical, so only one is implemented;
  the consistency identity is a TEST, not a production path.
- crossTermQ returns all three matrices (for readability):
  (Q_fwd, Q_rev, Q_sym) or None. crossTermQTot keeps returning summed
  Q_sym only (equilibrium-contour object; 6 density.py call sites
  unchanged in meaning).
- protocols.py is updated in lockstep (unused today, kept
  consistent for future work).
- Each tranche ends as a single reviewed commit.

## Architecture

    surfG1D.crossTermQ / surfGBethe.crossTermQ   (Q_fwd, Q_rev, Q_sym)
            |                                     dFermi as explicit arg
    integrate.py            ALL integration assembly:
       equilibrium contour: Q_sym (F(z) path; math unchanged)
       window integral:     new W_beta cross pieces from Q_fwd, Q_rev
       GrLessInt:           gains lesser cross-term scalar accumulation
            |
    density.py              T15 forward-axis sign fixes
    scfE.py                 count consumer (see "Count consumer")
    transport.py            T14 convention flip
    protocols.py            signatures + corrected sign documentation

Rule: surfG classes produce matrices; integrate.py owns all integration
math; scfE only consumes counts.

## The equations being implemented

All from the final boxed results of paper/derivations/derivation.tex;
no derivation detail is needed to implement or review this.

Coupling matrices, per contact a (built from the surface GF g_r, the
coupling tau_Da = E*S_Da - H_Da, and the overlap blocks):

    Q_fwd_a = tau_Da  g_r_a  S_aD
    Q_rev_a = S_Da    g_r_a  taubar_aD      taubar_aD = E*S_aD - H_aD
    Q_sym_a = (Q_fwd_a + Q_rev_a) / 2

Count under bias, reference mu_1:

    N_D = N_eq(mu_1) + (1/2pi) INT dE (f_2 - f_1) W_2(E)

where N_eq is the existing equilibrium contour of
F(z) = Tr[Gr_D (S_D - Q_sym_L - Q_sym_R)] (already implemented,
protected), and the window kernel is

    W_b(E) = Tr[Gr Gamma_b Ga S_D]                        (device term)
           + Im Tr[Gr Q_fwd_b] + Im Tr[Ga Q_rev_b]        (own-tail)
           - (1/2) sum_a Tr[(Gr Gamma_b Ga)(Q_rev_a + Q_rev_a^dag)]
                                                          (device-tail)

The device term is already captured through tr(P_window S); the
own-tail and device-tail pieces are the MISSING content (T13).

Implementation mirrors the existing contour pattern: the equilibrium
kernel (integrate.py weighted_combined, def :288, return :298) already
returns (w*Gr, w*Tr[Gr Q_tot]) per point; the window kernel
(weighted_func_GrLess) gains the same scalar second slot. Zero extra
linear solves: _gless_matrix_ops returns bare Gr @ Gamma @ Ga (no i, no
fermi factor - the (f2-f1) weight is already in the quadrature weights,
density.py:675), so the device-tail matrix is literally in hand.
HARD REQUIREMENT: the cross accumulation RAISES on ind=None - the
ind=None default (density.py:628,746) makes sigma = sigTot so the
"kernel" would be built from Gamma_L + Gamma_R, the exact mispairing
the pairing rule forbids. Cross accumulation is per-contact or an
error, never a silent total.

Plumbing facts, budgeted in tranche 4:
- GrLessInt runs through _GInt (integrate.py:143-228), shared with
  GrInt and hard-returning a single matrix; the tuple-returning window
  integrator is a _GLessIntCross twin of _GIntCross (integrate.py:
  265-356, ~90 lines incl. vmap/scan split and padding), leaving GrInt
  and the orthogonal fast path (integrate.py:376) untouched.
- integratePointsAdaptiveANT already supports tuple returns
  (density.py:243-282, used by densityReal:623); densityGrid mirrors
  that. BUT the adaptive stop test is maxDP on the MATRIX only
  (density.py:267); extend the stop test to the co-accumulated scalar
  so the cross count cannot ride along under-converged.
- MEMORY_PER_MATRIX_FACTOR = 47 was measured for the current kernel
  body; per-contact Q_fwd/Q_rev/Q_rev^dag live in the vmap raise the
  per-point footprint. Re-measure on GPU as part of tranche 4's gate,
  before the production-shape receipt.
- Mixed orthogonal/non-orthogonal contacts: if the KERNEL contact b is
  orthogonal (crossTermQ None), the own-tail vanishes but the
  device-tail does NOT (its alpha sum still sees the other contact's
  Q_rev). None handling is per-term - own-tail keyed on b, device-tail
  keyed on the alpha loop. Copying the skip-everything pattern of
  integrate.py:374-377 silently drops a real term; tested explicitly.
- integralFitNEGF (density.py:1499-1500) uses a different, two-window
  multi-contact decomposition for grid sizing only; it deliberately
  gets NO cross accumulation. Tuple-breakage call sites listed for
  tranche 4: density.py:1499-1500, scfE.py:734, scfE.py:737.

DOS: NO production change. DOS is occupation-independent; the existing
Q_sym form -1/pi Im F stays exact under bias because
W_L + W_R = 2 pi DOS_D identically. A standalone W_b(E) assembly
function is implemented for tests only, so that identity compares two
different code paths.

Binding implementation facts:
- taubar_aD carries UNCONJUGATED z off the real axis; it is NOT
  tau_Da^dagger except on the real axis. Matters for complex
  Hermitian H (SOC): build it from S_aD and H_aD directly.
- Every piece of W_b is manifestly real by construction; the
  implementation takes Re[] of the assembled scalar as numerical
  hygiene only and ASSERTS |Im| < tol rather than silently discarding
  (for complex H a large Im means an assembly bug, not physics).
- Q_rev^dag in the device-tail is formed by adjointing the ASSEMBLED
  Q_rev matrix - never by rebuilding with a conjugated energy (the
  existing code already builds taubar with unconjugated z at
  surfG1D.py:572 and surfGBethe.py:1195,1219; keep it that way).
- Pairing rule: reference mu_1 windows W_2 with weight (f_2 - f_1);
  kernel label and weight sign flip together. Windowing W_1 with
  (f_2 - f_1) is a different quantity, not a sign choice.
- Cross-count sign on the forward real axis is +(1/pi) Im. The contour
  form carries orientation in the traversal and extracts no Im at all.

## API

crossTermQ(E, i, conv=..., dFermi=None) -> (Q_fwd, Q_rev, Q_sym) | None

FIVE implementers, all updated in tranche 2 (protocols.py:6 lists them):
  surfG1D.py:550 (surfG), surfGBethe.py:610 (surfGB),
  surfGBethe.py:1250 (surfGBAt), surfG3D.py:471, surfGTester.py:158.
integrate.py:295 calls it polymorphically; a partial update breaks the
others silently, so tranche 2 gates on all five.

- surfG1D: already computes Q_fwd, Q_rev internally (surfG1D.py:574-576)
  and discards them after averaging; return all three.
- surfGBethe: Q_fwd and Q_rev are already formed per direction k at
  surfGBethe.py:1196-1198 and 1220-1222 with a hardcoded /2 average
  (the mix kwarg is surface-GF iteration damping, NOT the fwd/rev
  weight). Return them alongside the average; one pass per atom, no
  extra surface-GF solves.
- None contract: the "is not None" trace-time check survives a tuple,
  but the ACCUMULATION sites do not and are explicit edits:
  integrate.py:297 (Q_tot += Q_i), surfG1D.py:585-587,
  surfGBethe.py:661 (sum(qs)). Note surfGB/surfG3D never return None.
- crossTermQTot(E, conv=..., dFermi=None) -> summed Q_sym | None
  (unchanged semantics; 6 density.py call sites untouched in meaning).
- protocols.py: new signatures; docstring sign corrected from
  -(1/pi) Im to the forward-axis +(1/pi) Im with a note
  distinguishing the contour form.

## Count consumer (the T13 number must land somewhere conservative)

The window density is added to P AFTER the fermi search (scfE.py:
731-738) and the search target is contour-referenced (ne - nLower,
scfE.py:604,640,650); nothing today consumes tr(P_window S) in the
balance. Rule: the window enters the count diagnostics COMPLETELY or
not at all - adding its cross charge to a balance that omits its
device charge is non-conservative. Decision:
- dN_inclusive and every reported electron count include BOTH window
  pieces: tr(P_window S) + the window cross scalar.
- The fermi-search target itself stays contour-referenced in this
  work: bias production runs pin the fermi (contFermi), and a
  window-aware search is a behavior change out of scope here. If a
  future unpinned-bias search is wanted, it consumes the same complete
  pair, never one piece.

## T12: dFermi threading

Purpose: prevent JIT recompilation, not merely fix
staleness. The blunt alternative (version-bump in updateFermi) is
rejected because it forfeits the kernel cache.

- integrate.py:412: g.sigma(E, ind) -> thread dfs[ind] (surfGBethe.sigma
  already has the dFermi kwarg).
- integrate.py:295: g.crossTermQ(E, i) -> thread dfs[i] via the new
  dFermi parameter.
- surfGBethe.crossTermQ: stop reading self.gList[i].dFermi as a
  trace-time concrete float; use the traced argument (same mechanism as
  the existing _stot(E, dfs)).
- updateFermi's version-bump skip becomes correct for all branches.

## T15: forward-axis signs + documentation hazard

Flip -(1/pi) Im -> +(1/pi) Im at density.py densityRealN:571,
densityReal:624, damleCrossTerm:388. In the SAME pass, correct the
2026-03-18 derivation document (docs/superpowers/specs/
2026-03-18-cross-term-corrections-derivation.md) whose Section 7 error
currently cancels against the backward contour arc in densityComplex/N
- anyone "fixing" the working contour path from that document breaks
it. densityComplex/N themselves are correct and MUST NOT change.
Production note: with USE_INERTIA_EMIN=True (default), the forward-axis
routines are off the production path; no banked number moves.

## T14: transport.py bias convention

transport.py:690-691 muL = fermi - qV/2 contradicts scf.py:394-395
mu1 = fermi + qV/2 with mu1 = LEFT (scfE.py:477). Flip transport to
match scf. Numerically inert for |I|; fixes rectification-direction
readings. Edge: naive flip makes np.arange(muL, muR, dE) empty for
qV > 0 - iterate min->max and apply the sign explicitly.

## Test plan

Family A - parameter-free identities (gates for T13):
  A1 dual-split: N_eq(mu1) + N_win(W2, f2-f1) == N_eq(mu2) +
     N_win(W1, f1-f2) on toy junctions, near machine precision on
     dense grids. KNOWN LIMIT: A1's residual reduces to the same
     kernel-sum identity as A2, so a full L<->R kernel mislabel or an
     invalid folded own-tail (2 Im Tr[Gr Q_sym]) passes BOTH. A1/A2
     are necessary, not sufficient - A5 is the split-sensitive gate.
  A2 kernel sum: W_L(E) + W_R(E) == 2 pi DOS_D(E) pointwise, the two
     sides from DIFFERENT code paths, with eta pinned EXPLICITLY and
     identically on both (integrators use max(g.eta, ETA);
     _compute_dos_at_energy defaults to bare ETA - an eta mismatch
     fails the identity at O(eta) for a non-bug reason).
  A5 per-kernel split gate (the one A1/A2 cannot cover): on an
     ASYMMETRIC toy junction at nonzero bias, W_beta assembled by the
     production path vs an independent single-contact construction
     (exact diagonalization / generalized-eigh count of the biased
     junction). Sensitive to kernel mislabels and to any equilibrium
     folding smuggled into the window.
  A4 orthogonal limit (all Q terms vanish) + sign arbiter: window
     cross count equals the exact generalized-eigh Mulliken count to
     the QUADRATURE-DERIVED tolerance (tolerance from the grid, never
     a pinned observed digit - the no-fitted-values rule applies to
     our own tests too).
  A3 equilibrium collapse (mu1 == mu2 equals pure contour): kept as a
     NO-REGRESSION check only - the window weight is identically zero
     there, so it gates nothing about the kernel.

Family B - JAX/JIT proof (gates for T12):
  B5 single-trace: instrument the kernel body with a trace counter;
     run dFermi = a, b, a. Assert compiled EXACTLY once and results
     shift correctly (dFermi traced, not baked). Runs on GPU backend.
  B6 stale-bake regression: updateFermi then integrate without cache
     clear equals a fresh-process reference (unit form of the
     job-38165220 control).
  B7 per-contact index gate (B5/B6 are blind to a dfs[0]/dfs[1]
     swap): two contacts with deliberately DIFFERENT dFermi; compare
     crossTermQ(E, i, dFermi=d_i) against a baked-shift reference
     per contact.

Family C - no-regression and edge cases:
  C7 Q_sym bit-identity from the three-return crossTermQ across ALL
     FIVE implementers, spin r/u/g including a complex-H (SOC) toy
     with a known analytic answer.
  C8 full suite green: the 232 passed / 2 skipped baseline all still
     pass, plus every new-family test added so far.
  C9 T14 directional toy: muL > muR iff qV > 0 in both conventions;
     arange empty-window edge covered.
  C10 edge cases, one test each: (a) mixed orthogonal/non-orthogonal
     contacts - device-tail survives when the kernel contact is the
     orthogonal one; (b) ind=None raises in the cross accumulation;
     (c) spin factor - the window delta_N enters the count on the
     same per-spin footing as the contour count (scfE halves ne for
     spin='r'); (d) T=0 hard-step window - quadrature nodes land on
     the step edges, count still passes A4's tolerance.

Test environment: DEFAULT IS A GPU NODE (ckpt-all, --gpus=1; no
dedicated nodes exist and a ckpt job without a GPU is strictly
dominated). Every tranche gate - full suite, Family B, anything Bethe-
or production-shaped - runs there via sbatch. Family B in particular
MUST pass on the GPU backend: it certifies compile-once behavior on
the jax_cuda12 path production actually uses, and a CPU-only pass does
not certify that. The ONLY carve-out is the TDD inner loop on tiny
dense toys (<= ~20 orbitals, seconds), which may iterate login-node
with JAX_PLATFORMS=cpu before its gate run on GPU. Tests that depend
on MAX_GRID_POINTS set it explicitly (the uncommitted local config
mod is uncommitted; the worktree default is 500).

## Tranches (each leaves a working system, one commit each)

  1 Scaffold: worktree (done) + C7 baseline harness pinned BEFORE any
    change (record current Q_sym outputs).
  2 API: three-return crossTermQ + protocols.py + call sites.
    Gate: C7, C8.
  3 T12: dFermi threading. Gate: B5, B6, B7, C8.
  4 T13: window cross accumulation + W_beta assembly + scfE sum.
    Gate: A1, A2, A5, A4, C10, C8.
  5 T15: sign flips + 2026-03-18 doc + protocols docstring.
    Gate: A4, C8.
  6 T14: transport flip. Gate: C9, C8.

## Validation beyond unit tests

One production-shape receipt: AuTipPDT-class Bethe junction at modest
bias, fixed converged density; compare dN_inclusive old vs new and the
full T(E) overlay (spectrum-comparison rule: full curves, judged
by trend not point values). Small compute job (8 tasks / 32G class). This measures the
SIZE of the omitted window cross term on a real junction - the number
that decides how loudly the paper's bias sections need correcting.

## Non-goals

- The B1 +/-V resweep and any campaign re-banking: AFTER this lands.
- Bound states outside both contact bands: out of scope.
- More-than-two-contact generalization (the math supports it; nothing
  in gauNEGF needs it yet).

## Risks

- If the fixed code still shows the B1 asymmetry, that is a physics
  result (T12 was the last live suspect) - a finding, not a failure of
  this work.
- Bethe SOC assembly has no orthogonalized twin to check against; C7's
  complex-H toy is the only guard and gets written first.
- The archived 2026-08-05 plan is salvage material only; its
  <MEASURED> sign-pinning tasks are explicitly replaced by Family A.

## Open questions

None. All decision points above are settled.
