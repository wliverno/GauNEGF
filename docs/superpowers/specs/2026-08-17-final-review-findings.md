# Final whole-branch review: findings and merge gate

Branch fix/noneq-crossterm, commits b7c31fa..3b4639f (5 commits).
Verdict: DO NOT MERGE until the blockers below are closed.
The physics is correct; the packaging, one shared-helper change, one
missed branch, one sizing constant, and the strength of the test-suite
claims are not.

## What the review independently confirmed (no findings)

The reviewer re-derived the window kernel from scattering states rather
than reading the spec back, and confirmed term by term:
- device-tail -(1/2) sum_a Tr[K_b (Q_rev_a + Q_rev_a^dag)], K_b = Gr Gam_b Ga
- own-tail +Im Tr[Gr Q_fwd_b] + Im Tr[Ga Q_rev_b]
- regulator residual -2 eta Re Tr[Gr S Ga (S - sum_a Herm(Q_rev_a))]
- forward-axis count +(1/pi) Im INT f Tr[Gr Q_sym], contour keeps its
  minus (backward arc), protocols.py and the corrected 2026-03-18 doc
  both consistent
- pairing: ind=-1 -> contact 1 = mu2, weight f2 - f1 for both polarities
Also verified clean: every tuple-return call site, spin r/u/g/go
footing, SOC unconjugated-energy coupling, mixed orthogonal contacts,
multi-contact generality, and that the transport change is numerically
inert (I(V) unchanged in sign and magnitude).

## Blockers

B1. THE WINDOW TEST SUITE IS NOT IN THE BRANCH.
    tests/test_window_cross_term.py is untracked - it holds all four
    identity gates, the mixed-contact test, the ind=None raise test,
    the count-audit test and the transport-window test. Also untracked:
    the design spec and the implementation plan. tests/wbeta_reference.py
    IS committed and no tracked file imports it, so the branch looks
    covered and is not. Every gate run (231/236/248/249 passed) executed
    against a working tree richer than the commits.

B2. THE EQUILIBRIUM CONTOUR IS NO LONGER BIT-IDENTICAL.
    density.py:266-268 puts the co-accumulated scalar into the stop test
    of integratePointsAdaptiveANT, which is shared. densityComplex drives
    that helper with a tuple-returning computePoint, so the equilibrium
    ladder is now gated by a count-scaled criterion: maxDP is an
    element-wise density delta, accum is a trace of order 0.1-1 electrons,
    both compared against 1e-4. Expect the N *= 3 ladder to deepen 1-3
    levels, changing P and the SCF trajectory. The spec asked for this on
    the window integrator only.

B3. THE BISECT BRANCH DROPS THE NEW AUDIT ENTIRELY.
    scfE.py:745 still does compContourP2(self.mu1)[0], so _eqN is never
    set and the post-window refresh is skipped whenever bisect runs -
    and bisect is the mandatory fallback for poly/muller/secant. The
    earlier fix converted three of the four sites.

B4. GLESS_CROSS_MEMORY_FACTOR = 97 IS KNOWN WRONG AND SHIPPED.
    Labeled "(measured)" while the measurement itself showed 40 at the
    small config and 97 at the large one, i.e. an unmodeled size
    dependence. Counting live NxN arrays in weighted_gless_cross gives
    ~190-210 at two contacts, and the per-contact loop makes it grow
    with contact count; the formula has no num_contacts term. This is an
    out-of-memory failure on the first production-scale junction.

B5. slurm/test_gate.sbatch HARDCODES THE WORKTREE PATH
    (--output, PYTHONPATH, cd). After merge it tests the wrong tree or
    fails outright.

## Test-suite strength: the claim is stronger than the tests

A2 (sum_b W_b == 2 pi DOS) is an ALGEBRAIC IDENTITY independent of Q:
it follows from Hermiticity of sum_a Herm(Q_rev_a) and the resolvent
relation, and holds for arbitrary matrices in the Q slots. A Q_fwd /
Q_rev exchange passes it unchanged. A1's residual reduces to A2. A4
exercises the equilibrium Q_sym DOS only. A5 is therefore the sole
load-bearing gate, and its scope is surfG1D, two contacts, real spin,
vmap path, CList identity - and it compares against a test helper
(W_beta_dense) rather than against GrLessIntCross itself.

Consequence for the record: the commit message of f491c93 says
correctness is "pinned by parameter-free identities". That overstates
what three of the four gates establish and should be reworded.

### Real-symmetric toys cannot host a Q_fwd/Q_rev mutation test

For real H and S, Q_rev = Q_fwd^T and the window scalar is invariant
under exchanging them (verified numerically: Q_rev == Q_fwd^T to 3.4e-17
on the real-symmetric 1D chain toy, and Tr[Gr Q_fwd] == Tr[Gr Q_rev] to
8e-17, because Gr is complex symmetric there). No real-symmetric fixture
can therefore detect a fwd/rev swap, no matter how the test is written.
The distinction is physical only for complex Hermitian H (spin-orbit
coupling, magnetic field), which is where the mutation test now lives.

Other test findings:
- The forward-axis sign flip has NO test in either direction
  (test_damle_cross_term asserts abs(abs(dN) - 2.0); test_densityReal
  asserts a near-zero count in a zero-DOS region).
- test_bethe_direction_exclusion.py (430 lines) skips unconditionally -
  its geometry path lives under a gitignored directory.
- test_blas_in_multiprocessing.py (198 lines) asserts nothing, returns
  values from test functions (an error in pytest 8.4+), and runs heavy
  matmuls plus subprocesses at collection.
- test_crossterm_tuple_api.py:38-45 asserts that surfGAt3D's bulk path
  is still NaN, so it fails when someone fixes the defect. Make it xfail.
- Q_sym symmetrization asserts restate the implementation line; the
  stored baseline holds only Q_sym, which is swap-invariant, so a
  fwd/rev exchange is invisible to that whole file.
- Fitted tolerances to remove: the 0.6 and 2 factors in the arbiter
  envelope, the 10 in the dual-split test.
- Never executed: the scan/batched path of _GLessIntCross, and the
  threads_shifts (Bethe) branch of GrLessIntCross - which is the live
  production path.

## Important (not merge-blocking, but decide before paper runs)

- dN_inclusive can no longer reach zero under bias: the search targets
  the equilibrium count, so at convergence the reported mismatch tends
  to |windowN|. The freeze guard (dN_inclusive < conv) then never fires,
  every biased cycle pays a full Fermi search, and a converged run reads
  as unconverged. Either report the complete balance under a separate
  name or feed the previous cycle's windowN into the search target.
- The damleCrossTerm sign flip sits on the default SCF path via nLower,
  so every converged Fermi level moves: no pre-branch checkpoint may be
  resumed for publication runs, and the damle dN warning will begin
  firing where it never did.
- densityGrid/densityGridN docstrings say "ind: -1 for total". False:
  -1 selects the last contact and None now raises.
- threads_shifts keys on the presence of gList, which is true for
  surfG3 whose sigma rejects the argument. Gate on capability.
- Kernel cache retains g strongly, so every spawnNEGF / bias point leaks
  its object and executables; this branch adds per-contact kinds.
- _GLessIntCross has one call site (the same pattern rejected in
  transport.py). Defensible at ~65 lines as a readability exception -
  state it explicitly or inline it.

## Production smoke tests required before any paper number

1. Zero-bias non-orthogonal Bethe run, adaptive default: log the
   converged grid N and delta_N pre- and post-branch. If the ladder
   deepened, B2 is not fixed.
2. Biased AuTipPDT-class run with the Fermi search on, 10+ cycles:
   does the freeze guard ever fire, does |dN| plateau at |windowN|,
   what is the wall-clock per cycle vs pre-branch.
3. The same run forced onto bisect: confirm the audit is present.
4. Memory profile of GrLessIntCross at n >= 1500 on a real junction,
   two contacts, then a three-contact toy for the num_contacts slope.
5. Bethe window path with a live updateFermi between cycles (the
   untested production branch).
6. On one converged pre-branch system: old vs new nLower, lower-contour
   delta_N and converged fermi, to size how much of the existing bank
   the sign fix invalidates. Fresh rundirs, no seeding.
7. Full suite from a CLEAN CHECKOUT of the merge commit, not the
   worktree - this is what would have caught B1.
