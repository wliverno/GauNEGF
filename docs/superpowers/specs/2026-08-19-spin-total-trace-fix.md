# Spin-path total transmission: report the exact trace, keep the
# 4-channel projection

Date: 2026-08-19. Status: DESIGN SPEC, not implemented. Owner review
required before any code change.

Verdict up front: the spin-resolved transmission kernel's "total" is
not the total. _transmission_kernel_spin_block (transport.py:198-220)
sums four channel terms built from only the SPIN-DIAGONAL blocks of
gamma1/gamma2; when the contact self-energy carries spin structure
(AuSOC Bethe lattice), gamma has off-diagonal spin blocks and the sum
differs from the exact Tr[Gamma1 Gr Gamma2 Ga] by ~1.4% at the
ferrocene Ef -- and the difference is BASIS-DEPENDENT, which is what
tripped the frame-invariance receipt. The fix: the same kernel also
computes the exact full trace and returns THAT as the total; the four
channels stay exactly as they are (a well-defined projection); the
remainder is implicit (total - sum(channels)), the (T, Tspin) return
shape is unchanged. calculate_current's spin path integrates the same
truncated sum for its total current, so banked spin='g' IV totals on
AuSOC systems are in the blast radius too; calculate_dos is verified
exact and untouched.

ASCII conventions: ^dag = conjugate transpose, Tr[.] = trace,
Gamma_i = 1j*(sigma_i - sigma_i^dag), Ga = Gr^dag, kron(A,B) =
Kronecker product, R = 2x2 SU(2) spin rotation. Spin block indices
a,b in {u,d}.

---

## 1. Problem statement

### 1.1 What the code computes

_transmission_kernel_spin_block (transport.py:198-220), shared by the
'u'/'ro' path (transport.py:291-295) and the 'g' path after the
spinor-to-block shuffle (transport.py:297-313):

    T_ab = Tr[ gamma1_aa @ Gr_ab @ gamma2_bb @ (Gr_ab)^dag ]   (4 terms)
    total = sum_ab T_ab                                        transport.py:220

Only the aa/bb (spin-diagonal) blocks of gamma1/gamma2 enter
(transport.py:210-211). The exact Landauer total is

    T_exact = Tr[ Gamma1 @ Gr @ Gamma2 @ Ga ]

over the full 2Nx2N matrices -- the same math the restricted kernel
already does (_transmission_kernel_restricted, transport.py:189-195).
Expanding T_exact in spin blocks gives 16 terms; the code's four are
the ones with both gamma factors spin-diagonal. Every dropped term
contains at least one OFF-DIAGONAL spin block of gamma1 or gamma2.

### 1.2 When it matters and when it does not

Spin-scalar contacts (plain Au lattice, constant sigma): the gamma
off-diagonal spin blocks are identically zero, so the 4-term sum IS
the exact trace. Control receipt: spin-only rotation norms on the
plain-Au sigma are machine zero, 1.6e-16
(paper/studies/09_bias_physchem/s09_sigma_rotation_test.out; hoppings
are kron(M, I2), surfGBethe.py:381).

AuSOC Bethe contacts: sigma is NOT spin-rotation invariant --
||U^dag Sig U - Sig||/||Sig|| ~ 1.0e-1 at Ef for sigmaTot,
1.1e-1 for Gamma_L(Ef) (same receipt file, job 38643829). The onsite
L.S term survives Bethe assembly at full strength
(surfGBethe.py:366-369). Gamma carries off-diagonal spin blocks and
the 4-term sum is wrong as a TOTAL.

### 1.3 Receipts (AuBetheFerrocene junction, spin='g', AuSOC)

Leak diagnostic paper/studies/09_bias_physchem/txyz_leak_diag.py,
output s09_txyz_leak_diag.out (job 38644973): dense-numpy full trace
vs the production spin path, sigmaTot == sig1+sig2 to 0 exactly (not
a wrapper leak):

    E-Ef=+0.0  frame z: T_numpy=5.3884893885e-03  T_prod=5.4651185777e-03
               frame x: T_numpy=5.3884893885e-03  T_prod=5.4467979888e-03
    E-Ef=-1.0  frame z: T_numpy=5.8352645498e-04  T_prod=5.8463583557e-04
    E-Ef=+1.0  frame z: T_numpy=2.8186385185e-02  T_prod=2.9406022398e-02

  - Magnitude at Ef: |deviation| = 1.42% of the exact trace. NOTE ON
    DIRECTION: the ledger entry (2026-08-19 early) says the 4-term sum
    "sits ~1.4% below the exact trace"; the receipt above shows it
    1.42% ABOVE at this Ef and frame. The remainder is a real number
    with NO fixed sign (it is not a trace of a PSD product), so only
    the magnitude is a stable statement. The ledger wording should be
    corrected when this spec lands.
  - Basis dependence: the exact trace is frame-invariant to 10 digits
    under a spin-frame rotation U = kron(I, R) applied to F and sigma,
    while the production "total" moves (z vs x above; the original
    ferrocene Tx/Ty/Tz run s09_ferrocene_txyz.out fired the invariance
    receipt at ~3% drift). A basis-dependent "total transmission" is
    not a physical observable.

### 1.4 What is a defect vs what was a documented limitation

Documented and STILL TRUE: the 4-channel decomposition is incomplete
under spin-mixing leads. The S4 campaign banked this as the "0.69%
gamma lead-mixing floor at Ef" for CHANNEL claims
(paper/studies/reports/2026-07-20-recipe-ledger.md, search
"lead-mixing"): channel-anisotropy statements must be made relative
to the per-frame decomposition remainder. Nothing in this spec
changes that; the channels remain a projection.

The DEFECT: the TOTAL was believed exact and is not. The sum of an
incomplete decomposition was reported as "total transmission" and
integrated into "total current". That is the only thing this spec
fixes.

---

## 2. Proposed fix design

### 2.1 Kernel

_transmission_kernel_spin_block computes the exact total alongside
the existing four channel terms. Split each gamma into spin-block-
diagonal and off-diagonal parts, Gamma_i = D_i + O_i:

    remainder = Tr[ O1 @ Gr @ Gamma2 @ Ga ] + Tr[ D1 @ Gr @ O2 @ Ga ]
    total     = sum_ab T_ab + Re(remainder)

This covers the 12 dropped terms exactly once (O1*D2, O1*O2, D1*O2),
and Tr[D1 Gr D2 Ga] is by construction the existing 4-term sum. The
channels T_ab are computed exactly as today (transport.py:209-219,
unchanged).

Why the D/O split instead of one dense Tr[Gamma1 Gr Gamma2 Ga]:
when the leads are spin-scalar, O1 = O2 = 0 EXACTLY, both remainder
traces are exactly 0.0 in floating point (products with a zero
matrix), and total = old 4-term sum + 0.0 -- BIT-IDENTICAL to the old
path. A single dense trace would change the floating-point summation
order and perturb every banked spin-scalar total in the last bits,
which the reproducibility discipline here forbids for no benefit.
Cost: two additional 2Nx2N triple products per energy, comparable to
the existing four block products; T(E) evaluation stays cheap
relative to sigma construction.

Ga is formed once from the already-computed Gr (jnp.conj(Gr).T); no
second inverse.

### 2.2 API shape: keep (T, Tspin), remainder implicit. DECIDED.

transmission_single_energy keeps its exact return contract for
'u'/'ro'/'g': (total_transmission, [T_uu, T_ud, T_du, T_dd]).
total_transmission becomes the exact trace; the list stays 4 channel
terms. The remainder is NOT returned as a fifth component; it is
recoverable exactly as total - sum(channels), and the docstrings of
transmission_single_energy, calculate_transmission, cohTransSpin, and
cohTransSpinE state explicitly: under spin-mixing leads,
total != sum(channels); the difference is the off-diagonal-gamma
remainder and is basis-dependent.

Rejected alternative -- 5-column Tspin: every existing consumer
unpacks (T, Tspin) with Tspin of width 4: the checkpoint format
('spin_transmission', (N,4), transport.py:460, 470, 492-493), .mat /
.npz files already banked and read by the S1/S4 figure scripts, and
calculate_current's `for i in range(4)` integration loops
(transport.py:714, 730). A width change silently breaks or corrupts
all of them for zero information gain (the remainder is a subtraction
away). If the owner later wants the remainder surfaced in checkpoint
files, an OPTIONAL extra npz key is additive and non-breaking -- open
question 3.

Backward-compatibility consequence, stated plainly: after the fix,
sum(Tspin) == T no longer holds on spin-mixing leads. Any consumer
that ASSERTS that identity (none found in gauNEGF/ or paper/lib/) is
asserting the bug.

Checkpoint resume hazard: .npz transmission checkpoints carry no code
version. A partial pre-fix checkpoint resumed under the fixed code
would mix truncated and exact totals in one array. AuSOC re-banks
must start from fresh checkpoint files (spin-scalar resumes are safe;
the values are bit-identical).

### 2.3 calculate_current spin path

calculate_current (transport.py:713-733) currently builds
current_spin by integrating each channel column and then sets
current_total = sum(current_spin) -- the SAME truncation, integrated.
Fix: current_total integrates the exact total array
(`transmissions`, which after 2.1 holds the exact trace), in both the
T == 0 and finite-T branches:

    current_total = grid_sign * eoverh * trapezoid(transmissions [* dfermi],
                                                   integration_energies)

current_spin stays as-is (channel currents are the projection,
unchanged), and the docstring states current_total !=
sum(current_spin) under spin-mixing leads. For spin-scalar leads the
new total current agrees with the old to floating-point roundoff
(same numbers integrated, different summation order); bit-identity is
guaranteed for T(E) values but NOT promised for the integrated
current -- the regression test bounds it at rtol 1e-13.

### 2.4 calculate_dos: verified exact, NO CHANGE

The 'u'/'ro' and 'g' DOS paths (transport.py:361-396) take the
diagonal of the FULL product Gr @ (S - Q); the two spin-block
diagonals are together the full diagonal, so total_dos is the
complete trace. Nothing is dropped; nothing changes.

---

## 3. Blast radius

Affected: every banked spin='u'/'ro'/'g' TOTAL (transmission and
current) computed through calculate_transmission /
calculate_current / cohTransSpin(E) on a system whose CONTACTS carry
spin structure -- in this campaign, AuSOC Bethe lattices. Order ~1% at
Ef (1.42% measured on ferrocene; energy-dependent, 4.3% at Ef+1).
Channel columns are unchanged everywhere.

Not affected: all spin='r' results (restricted kernel already exact);
all spin-path results on plain-Au or constant-sigma contacts
(bit-identical T(E) by the 2.1 design; currents to roundoff); all DOS
results; SCF/density results (density.py does not use this kernel).

Manuscript exhibits citing affected totals (paper/manuscript/
outline.md):
  - 4.5 Fig S1 (GSO size series, all four members AuSOC exact): the
    black "total" curve in every panel; channel curves unchanged.
  - 4.4 Fig R3x GSO legs (Au2AntMin GSO + AuSOC, AuTip GSO): total
    T(E) and the Ef provenance numbers (Min GSO T(Ef)=0.182, flip
    fraction 34%) -- flip FRACTIONS have the total in the denominator.
  - 6.1 Fig S2 (ferrocene equilibrium): total T(E).
  - 6.2 Fig S4 (ferrocene IV): total currents in the hysteresis sweep;
    the 6.0-6.3% flip-channel share and +0.264 pp directional
    efficiency have total-current denominators, so they shift at the
    ~1% RELATIVE level (6.3% -> ~6.2%) -- the claims survive but the
    numbers must be re-derived, not restyled.
  - 4.6 Fig S3 (AuTip GSO IV): affected only if its contacts are
    AuSOC (S1 lists AuTip GSO under the AuSOC constraint) -- verify
    the banked run's lattice tag before re-banking; if AuSOC, totals
    move, if plain Au, bit-identical.
  - Channel-level statements (Iuu vs Idd, Iud vs Idu comparisons)
    are unaffected -- the spin-transmission channels themselves do
    not change, only the reported total.

Re-bank cost: T(E)/IV re-evaluation from the banked converged .mat
checkpoints -- transmission passes only, NO SCF. Fresh checkpoint
files and fresh tags per the reproducibility rule (no seeding from
pre-fix transmission checkpoints; the converged F/S matrices
themselves are untouched by this fix and remain valid inputs).
Ballpark: the original ferrocene sweep was 8h wall for 20 bias points
INCLUDING SCF; transmission-only re-banks are a small fraction of
that per system.

---

## 4. Test plan (TDD; tests fit the code, no production hooks)

Tests live in tests/. Both build small dense fixtures and compare
against straight numpy computed inside the test -- no new accessors,
no kernel changes for testability. Red first: test 1 fails against
the current kernel by construction (it asserts the exact trace).

Test 1 -- spin-mixing exactness and frame invariance
(tests/test_spin_total_trace.py):
  - Fixture: N=3 orbitals (6x6 spin space), complex Hermitian F,
    S = I + small Hermitian perturbation (non-orthogonal), and two
    energy-independent 6x6 sigmas with NONZERO off-diagonal spin
    blocks and negative-definite anti-Hermitian parts (physical
    gammas). Fed through SigmaCalculator(sig1, sig2,
    energy_dependent=False) with spin='u' (matrix already 2Nx2N, no
    expansion path ambiguity); a second case runs spin='g' with the
    same matrices permuted to spinor ordering to exercise the shuffle
    (transport.py:301-308).
  - Assert 1 (exactness): total from transmission_single_energy
    equals dense numpy Tr[Gamma1 Gr Gamma2 Ga] (numpy inv, explicit
    ETA) within 1e-11 relative, at 3 energies.
  - Assert 2 (frame invariance): for U = kron(I2, R) with R a
    non-trivial SU(2) rotation (block order matching the path under
    test), transform F, S, sig1, sig2 by U^dag . U and assert the
    total is invariant within 1e-11 relative. The CURRENT kernel
    fails this by ~percent on such fixtures; the fixed kernel passes.
  - Assert 3 (remainder bookkeeping): total - sum(channels) equals
    the numpy-computed off-diagonal remainder within 1e-11.

Test 2 -- spin-scalar regression (same file):
  - Fixture: same F/S, sigmas built as kron with IDENTICALLY ZERO
    off-diagonal spin blocks.
  - Assert: fixed total == sum of the returned four channels,
    BIT-IDENTICAL (== on the floats, no tolerance) -- this pins the
    2.1 design guarantee that spin-scalar banked totals do not move.
  - Assert: the four channel values equal the 4-term sums computed
    in-test by numpy from the gamma blocks (1e-11), pinning that the
    channel definition did not drift.

Test 3 -- current path (same file): tiny grid, spin='u',
spin-mixing sigmas: calculate_current total equals
grid_sign*eoverh*trapezoid of the exact-total array (1e-12), and !=
sum(current_spin) by the integrated remainder; spin-scalar twin
asserts current_total == sum(current_spin) within rtol 1e-13.

---

## 5. Out of scope

  - Changing the channel DEFINITION (the spin-diagonal projection) or
    adding alternative decompositions that close the remainder. The
    channels are a documented projection; their incompleteness under
    spin-mixing leads is a property, not a bug.
  - Rotating-frame / Tx-Ty-Tz analysis tooling: lives in
    paper/studies (ferrocene_txyz.py already reports exact totals and
    per-frame remainders as analysis-side receipts) and stays there.
  - The sigma spin-rotation rule itself (rotate F AND sigma; settled
    2026-08-18, ledger) -- restated here only as context.
  - DOS and density/SCF code paths (verified unaffected, section 2.4).

---

## 6. Open questions for the owner

  1. Re-bank scope and order: S1 (4 members), S2, R3x GSO legs are
     transmission-only re-banks from banked checkpoints; S4 (and S3
     if its lattice is AuSOC) need IV re-integration. Approve the
     list, and which re-banked totals gate manuscript prose edits
     (the 6.3% flip share and 0.182/34% R3x numbers will move at the
     ~1% relative level).
  2. Ledger correction: the 2026-08-19 entry's "sits ~1.4% below the
     exact trace" contradicts the s09_txyz_leak_diag.out receipt
     (4-term sum 1.42% ABOVE at Ef, frame z). Amend the entry, or is
     there a second receipt behind the "below" wording?
  3. Should calculate_transmission's checkpoint optionally store the
     remainder column (additive npz key, loaders unaffected), or is
     total - sum(channels) at analysis time sufficient?
  4. Spin-scalar current totals: the fix preserves T(E) bit-identity
     but current integration order changes (agreement to ~1e-13
     relative). Acceptable, or must the current path also preserve
     bit-identity (sum(current_spin) + integrated-remainder form)?
