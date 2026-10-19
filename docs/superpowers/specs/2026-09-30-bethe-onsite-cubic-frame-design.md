# Bethe-lattice on-site block in the right frame and basis: design (rev 4, 2026-09-30)

Status: REVIEWED
- rev 1 was reviewed APPROVED WITH FIXES: 0 Critical, 4 Important, 13 Minor, all folded in here
  (see section 8).
- Approved 2026-09-30: fix it, test-first.
- Defect 2 (section 7) is IN SCOPE, measured first with a red test.
- rev 3 (ruling 2026-09-30): constructMat builds matrices from physical directions, so it may also
  build the on-site matrix. So the on-site block is built by constructMat when it is given all 12 neighbour
  directions; there is no separate method. The validation is trimmed to the two real guards.
- rev 4 (ruling 2026-09-30, after review of the merged code): the on-site build belongs in a separate
  method. This REVERSES rev 3: the on-site block is built by a separate method, `surfGB.constructOnsite(dirList)`,
  and constructMat returns to its HEAD form (single bond direction only). The math and the tests are
  unchanged; only the call sites change.
- Branch `testing`. No commits by the agent; the owner commits.

## 1. Defect 1: e_g/t2g split on the wrong axes

`surfGB` (gauNEGF/surfGBethe.py) builds the on-site Hamiltonian of the Bethe-lattice atom once, from
the lattice parameter file:
- surfGBethe.py:360-363:
  `hdiag = [es] + [ep]*3 + [edd] + [edt]*2 + [edd, edt]`, `H0 = diag(hdiag)`.
- The d order is the one constructMat uses (surfGBethe.py:457): [d3z2-r2, dxz, dyz, dx2-y2, dxy].
- So `edd` (e_g) sits on {z2, x2-y2} and `edt` (t2g) on {xz, yz, xy} of the device's Cartesian axes.
- surfGBethe.py:204 hands this one H0 to every contact:
  `surfGBAt(self.H0.copy(), Slist, Vlist, ...)`.

The neighbour hoppings and overlaps are rotated correctly:
- Each contact's 12 nearest-neighbour directions come from its own geometry (genNeighbors,
  surfGBethe.py:148-150, 226-301).
- constructMat rotates every two-centre block onto its bond (surfGBethe.py:374-497,
  `M_rot = tr @ M @ tr.T`). The reviewer checked all 25 d_block entries by hand: the d basis rotates
  like the tensors T1..T5 of section 3.1, signs included.

The e_g/t2g split is different. It is the cubic crystal field of the fcc environment, and the
Slater-Koster sets define it on the crystal's cubic axes:
- Palacios PRB 66, 035322 (2002), Table I "Au-b";
- paper/studies/reports/2026-09-29-lattice-parameter-provenance.md:88: edt = t2g, edd = eg.

The on-site block has no bond vector, so it is never rotated. Every PDT device in
paper/studies/02_pdt_bethe/systems.py has its (111) contact normal along Cartesian +/-x (review I3,
paper/studies/reports/2026-09-29-lattice-dos-review.md). The ferrocene junction was not checked, but
it is rerun anyway.

**Which lattices are affected.** Only those with edd != edt, i.e. Au2.bethe (edd 0.12771,
edt 0.137645 Ha):
- Au.bethe and AuSOC.bethe have edd == edt. The audit confirms that the 483 copies of the three
  .bethe files in the repo are byte-identical.
- The diagonal on-site convention itself is inherited from ANT.Gaussian's InitBetheLattice
  (BetheLattice.F90:241-265, cited at provenance report lines 243-244).

**Measured size on the orthogonalized lattice** (paper/studies/reports/2026-09-29-lattice-dos.md,
"Findings beyond the brief" item 1, reviewer-converged numbers):
- fcc Gamma_25' is split into 2.53/2.61/2.80 eV instead of 2.80 eV x 3.
- The Bethe lattice E_F moves by 19 meV and the fcc E_F by 25 meV.
- For the AuTipPDT contact, Gamma(E_F) changes by 0.8-0.9 %. It changes by at most 1.8 % within
  +/-1 eV of E_F, and by up to 10 % across the d band.

**Ruling (2026-09-30):** the defect is fixed even though the change is small, with code review.
Design A was chosen on the same day.

## 2. Goal and success criteria

**Goal.** Each contact's Bethe atom carries its on-site d block with the e_g/t2g split on that
contact's own cubic axes. Afterwards every paper result built on Au2.bethe is recomputed from
scratch.

**Success criteria**
1. For Au2.bethe, the fcc Gamma-point spectrum built from a contact's on-site block and its 12
   hoppings has the cubic multiplicities. That means one singlet, one doublet and two triplets.
   - This must hold for EVERY contact of a two-contact junction whose contacts have different cubic
     frames.
   - The eigenvalues of each contact's on-site d block are {edd x2, edt x3}.
2. Rotating the whole junction leaves the following unchanged, to numerical precision:
   - the lattice's Gamma-point levels;
   - the lattice Fermi level;
   - the eigenvalues of the contact's surface Sigma(E).
3. For every lattice file with edd == edt (Au.bethe, AuSOC.bethe), each contact's on-site block is
   array_equal to today's H0 in any orientation. No exact-overlap result moves.
4. Given the 12 cubic <110> vectors (any permutation or sign pattern), `constructOnsite` returns
   today's diagonal H0 to allclose 1e-14.
5. The full test suite passes, apart from failures documented as pre-existing in the baseline
   run.
6. Every affected paper result is recomputed in a fresh run dir under a new tag. The owner has
   read the old-vs-new comparison for each (section 3.4).

## 3. Design (approach A: e_g projector from the contact's cubic axes)

### 3.1 Math

**Recovering the cubic axes.** Take a contact's 12 unit neighbour vectors {n_k} (its stored
`dirList`). In an fcc crystal each n_k is (+/-1, +/-1, 0)/sqrt2 up to permutation, in cubic
coordinates.
- Take n_0 and find a neighbour n_j with |n_0 . n_j| < 1e-6. The NN dot products are 0, +/-1/2
  and +/-1, so any tolerance in (1e-12, 0.25) separates them.
- Exactly two such neighbours exist (+/- the same vector), and either may be used. If there is no
  perpendicular partner, raise ValueError.
- Then c1 = (n_0 + n_j)/|n_0 + n_j|, c2 = (n_0 - n_j)/|n_0 - n_j|, c3 = c1 x c2.

**Validation (raise ValueError otherwise)**
- for every n_k, the sorted absolute components (|n_k . c1|, |n_k . c2|, |n_k . c3|) equal
  (0, 1/sqrt2, 1/sqrt2) within 1e-8.

**The e_g projector.** Map each real d orbital to its unit-Frobenius-norm traceless symmetric
tensor, in constructMat's d order. With e_x, e_y, e_z the Cartesian unit vectors and ab^T an outer
product:

    T1 (z2)    = (2 e_z e_z^T - e_x e_x^T - e_y e_y^T)/sqrt6
    T2 (xz)    = (e_x e_z^T + e_z e_x^T)/sqrt2
    T3 (yz)    = (e_y e_z^T + e_z e_y^T)/sqrt2
    T4 (x2-y2) = (e_x e_x^T - e_y e_y^T)/sqrt2
    T5 (xy)    = (e_x e_y^T + e_y e_x^T)/sqrt2

The contact's e_g subspace is spanned by the traceless tensors that are diagonal on its cubic axes:

    A1 = (2 c3 c3^T - c1 c1^T - c2 c2^T)/sqrt6,   A2 = (c1 c1^T - c2 c2^T)/sqrt2

    P_eg[m, n] = sum_a <T_m, A_a>_F <A_a, T_n>_F        (5x5, rank 2, P^2 = P)

**Properties**
- P_eg is invariant under any permutation or sign flip of (c1, c2, c3).
- For c = Cartesian axes, P_eg = diag(1, 0, 0, 1, 0) up to round-off.
- The reviewer verified by hand that constructMat's d basis rotates exactly like T1..T5, signs
  included.

**The d block.**

    H0_d = edt * I_5 + (edd - edt) * P_eg

- Cubic axes on Cartesian give diag(edd, edt, edt, edd, edt), to allclose 1e-14.
- If edd == edt, then (edd - edt) is exactly 0.0, so H0_d == edt * I_5. It is array_equal up to
  signed zeros, which array_equal treats as equal.
- The s and p on-site entries are unchanged.

**SOC.** With SOC the per-contact on-site is `kron(H0_9x9_rotated, I_2) + Hsoc`, using today's Hsoc
(surfGBethe.py:364-369). A correct atomic L.S is isotropic and needs no rotation. Whether today's
Hsoc IS a correct L.S in constructMat's basis is defect 2 (section 7).

### 3.2 Code changes: gauNEGF/surfGBethe.py only

1. **readBetheParams (304-372).**
   - `self.H0` keeps its current value and meaning: the on-site block in the lattice's own cubic
     frame.
   - Keep the SOC term available as an attribute so the per-contact build can reuse it.
   - A one-line docstring note says that the contacts use `gList[i].H0`, and that `surfGB.H0` is
     the cubic-frame reference.
2. **New surfGB method `constructOnsite(dirList)`** (rev 4; package style: short docstring, 1-2 line
   comments), placed directly before constructMat.
   - It returns the contact's on-site matrix, 9x9 or 18x18 with SOC (it reads `self.Edict` and
     `self.SOC` itself), built as in 3.1.
   - The axis recovery and its ValueError validation live in it.
   - constructMat is unchanged from HEAD (single bond direction only).
   - It must be called with the STORED `self.dirLists[c]`, never with directions re-derived from
     genNeighbors.
3. **Line 204.** Each contact's surfGBAt gets its own `constructOnsite(dirList)`, inside the per-contact
   loop over `zip(self.dirLists, self.Slists, self.Vlists)`.
4. **Fermi level (206-213).** No change. It is computed from gList[0] and copied to all contacts.
   After the fix every contact is the same crystal rotated, so the copy agrees with each contact's
   own value to solver resolution (T1 checks this).

Not changed:
- constructMat, genNeighbors, surfGBAt;
- the integration code;
- spawnNEGF.

surfGBethe.py:204 is the only place contacts are built for surfGB. Nothing in gauNEGF/ reads
surfGB.H0 (reviewer, check 5).

### 3.3 Tests, written first (TDD)

**File and setup**
- New file `tests/test_bethe_onsite_frame.py`.
- It builds surfGB through a MagicMock `bar` (coordinates in bohr, `ibfatm`, `ibftyp`), as
  tests/test_bethe_direction_exclusion.py:169-203 (`_construct_surfGB`) does. It runs from a tmp dir.
- **Precondition, from review M7:** `_construct_surfGB` always copies Au.bethe. So the Au2 tests
  copy ./Au2.bethe explicitly and assert `Edict['dd'] != Edict['dt']` before any physics check.
- AuSOC tests use spin='g' and S = eye(2N).

| id | test | today | after |
|---|---|---|---|
| T1 | Au2.bethe, two-contact junction, normals +x and -x (as in the paper devices), in-plane first neighbours not related by a cube operation. For EACH contact: eig(H0_c + sum_k V_k) splits into one singlet, one doublet and two triplets, each degenerate within 1e-10 eV; eig(d block of H0_c) = {edd x2, edt x3} within 1e-12; gList[0].H0 != gList[1].H0 | FAIL | PASS |
| T2 | Au2.bethe, a patch and the same patch rigidly rotated by a generic R. First assert R @ dirLists[c] (unrotated) == dirLists[c] (rotated) elementwise, so the labels match. Then compare: Gamma levels (1e-10 eV); lattice fermi within 2 meV, both recomputed with calcFermi(tol=1e-6) (solver resolution: the Dyson and quadrature stopping tests are not rotation invariant, plan review I1; 2 meV is 1/10 of the 19 meV frame effect; a miss while the Gamma and Sigma checks pass is reported, never widened); eigenvalues of the surface Sigma from `sigma(E, i, conv=1e-12)` at 3 complex energies (1e-8 relative) | FAIL | PASS |
| T3 | `constructOnsite(dirs)` called directly on the 12 cubic <110> vectors, and on a permuted, sign-flipped copy: returns today's diag H0 (allclose 1e-14). Only T3 can see a swapped e_g/t2g, a wrong scale or a dropped split, because those errors still have O_h symmetry and T1/T2 cannot see them | ERROR (no method) | PASS |
| T4 | Au.bethe (spin 'r') and AuSOC.bethe (spin 'g', 18x18), paper-device orientation: each gList[c].H0 is array_equal to readBetheParams' H0 | PASS (guards bit-identity) | PASS |
| T5 | `constructOnsite` raises ValueError on a non-fcc direction list | ERROR (no method) | PASS |
| T6 | Au2.bethe patch with an out-of-plane atom that forces the twin (nVecs2) stacking (surfGBethe.py:164-171): T1's multiplicity check holds for that contact. This catches an implementation that re-derives the axes from genNeighbors instead of using the stored dirList | FAIL | PASS |

**Tolerance note on T2 (review I1).** The lattice self-energy iteration updates in place
(surfGBethe.py:1072-1077, 1147-1154). At the default 1e-5 convergence the result therefore depends
on the direction-label order. T2 uses conv=1e-12 and asserts label correspondence first. The
Gamma-level comparison is a sum over all 12 V_k, so it is label-free.

**Where tests run.** On a compute node, never on the login node: all jax numerics go through
srun/sbatch on an allocation. Each surfGB construction runs calcFermi (about 30-60 s), so fixtures
are module-scoped where possible. The full suite runs before any change, to record the baseline of
pre-existing failures, and again after.

### 3.4 Rerun campaign (after the code change is reviewed and green)

**Redo list.** paper/studies/reports/2026-09-30-au2-bethe-usage-audit.md, six paper items:
- Table 1: two rows (Au10-PDT and Au19-PDT, both on the orthogonalized lattice);
- Fig 1: its orthogonal legs;
- Fig 1b: see section 6;
- Fig 5: its T2 leg;
- Fig 6: all three panels;
- SI-F3: its T2 leg.

The audit estimates about 14 h of compute before queue overhead.

**Rules**
- No compute before the owner's go.
- Every affected result is recomputed from scratch in a fresh run dir under a new, distinct tag.
  There is no seeding from pre-fix matrices. The manifest records the gauNEGF HEAD and diff.
- Every data file on the redo list is regenerated (Table 1, Fig 1/5/6, SI-F3). Numbers in
  paper/assembly/* are updated only as the owner directs, and the owner writes the prose.

**How each item is compared, old vs new**
- T(E) legs: a full T(E) overlay (paper/lib/tcompare.py), sent to the owner as a picture. The owner
  reads the trends.
- Table 1 rows: old and new rows side by side (bare, cross, inclusive).
- Fig 6c timing bars: old and new bars side by side.
- The pinned-E_F real-part arm: its T(E) overlay, plus the old and new pinned E_F values.

## 4. Non-goals

- **surfG3D.py.** It carries the same diagonal at surfG3D.py:272-273, but it is parked under
  docs/superpowers/specs/2026-09-29-surfG3D-embedding-fix-design.md. A carry-over note was added to
  that spec's header on 2026-09-30. No surfG3D code changes here.
- No parameter value or file changes, and no ANT.Gaussian changes.
- **Scripts that bypass surfGB.__init__** keep the old pairing on Au2. They are all Au-only or
  non-paper today:
  - paper/studies/11_bethe_vs_fcc/kscan_surfdos.py:297;
  - paper/studies/12_lattice_params/lattice_bethe_dos.py:164-203;
  - paper/studies/12_lattice_params/lattice_fcc_ksum.py:137;
  - paper/studies/11_bethe_vs_fcc/t_bethe_vs_fcc.py:209 (`surfGAt3D(geo.H0, ...)` with a surfGB, 'kstar_bethegeom' arm; added by the final review);
  - tests/au_parameters_benchmarking_report.py:221-234.

  From now on, any such script uses `gList[i].H0` or `constructOnsite(dirList)`.
- **Pre-existing, owner awareness only (review M10).** One contact can mix the two stacking lists.
  The stacking is chosen per atom, and dirLists stores the last atom's list (surfGBethe.py:155-186).
  Design A follows the stored list, as the hoppings do.
- No Au10 runs beyond the redo list, and those only with the owner's go.

## 5. Risks

1. **Sign convention of T2..T5 against constructMat.** Verified by hand in review. It is also pinned
   by T1/T2 (a wrong subspace) and T3 (scale and assignment).
2. **A non-fcc direction list.** It cannot occur by construction (genNeighbors builds ideal
   directions). The ValueError validation turns it into a loud failure.
3. **Label-dependent convergence** in the lattice iteration. T2 handles it (see the tolerance note).
4. **Tests that read an H0 directly** (tests/test_soc.py:363, 439-463; tests/test_dfermi_threading.py:53;
   tests/test_gless_structure.py:226, 400, which read surfGBAt.H0 of a directly built Au atom). All
   use edd == edt files or check only shapes. The implementer confirms this by running them.

## 6. Owner calls

The code change is designed; these decisions come at the rerun phase, and are carried from the
audit's section 6:
1. **Fig 1b.** It is rendered but not placed in the paper. Rerun it or not?
2. **e1_timing matrix** (16 dirs, Au2). It backs only a prose timing sentence in the Fig 6 caption.
   Rerun it, or reword the sentence?
3. **Fig 6 real-part arm.** It is affected only indirectly, through the pinned E_F inherited from
   the B3/B6 legs. It is rerun with them.
4. **The 57 non-paper Au2 run dirs.** They are not rerun unless the owner asks.
5. **SOC reruns** (defect 2): unconditional (owner 2026-09-30). The list comes from the AuSOC audit.

## 7. Defect 2, found in review: the spin-orbit term uses -dxy (IN SCOPE, owner 2026-09-30)

**The finding** (review I4, re-derived independently by the controller):
- constructSOCterm (gauNEGF/spinTools.py) builds real orbitals from spherical harmonics through
  genOrbList.
- For m = -2 (d orbital 4) it puts -i/sqrt2 on Y_2^-2 and +i/sqrt2 on Y_2^+2, which gives -xy. The
  standard convention (+i/sqrt2 on Y^-|m| and -(-1)^|m| i/sqrt2 on Y^+|m|) gives +xy.
- The two agree for |m| = 1 (xz, yz are fine) and for m >= 0.
- constructMat's orbital 4 is +xy (surfGBethe.py:485).
- So Hsoc_d = D (L.S) D in constructMat's basis, with D = diag(1, 1, 1, 1, -1) (x I_2):
  - every L.S element that couples dxy to another d orbital has the wrong sign;
  - no rotation maps D (L.S) D onto L.S, so the AuSOC lattice is not a physical crystal and depends
    on how the junction is oriented;
  - the atomic SOC spectrum is unchanged, because D is unitary. That is why the existing SOC tests
    (tests/test_soc.py: shape, hermiticity, trace, s block) cannot see it.

**Latent and inactive** (soc_p = 0 in AuSOC.bethe): the p block comes out in the order [pz, px, py]
(m = 0, +1, -1), while the lattice p order is [px, py, pz].

**What it would touch.** Every AuSOC-lattice result:
- Table 1 rows Au4-PDT (g) and Au10-ferrocene (g) (table1.csv:4, 7);
- the Fig 4 ferrocene B1 leg;
- SI-SOC and SI-S1, per the audit.

A dedicated AuSOC usage audit would produce the full list.

**Sequence (owner-approved 2026-09-30)**
- **Red tests first, on a compute node:**
  - (a) In Hsoc's up-up d block, [d4, d3] / [d2, d1] = +2 (standard basis). Today it should be -2.
  - (b) T2-style rigid-rotation Gamma-level invariance on an AuSOC patch, with spin='g'.
- **Fix:**
  - In genOrbList, use the standard negative-m coefficients.
  - Order the l = 1 block as m = (+1, -1, 0), so that p = [px, py, pz].
  - Update the constructSOCterm docstring if needed.
- **Reruns are unconditional.** Ruling 2026-09-30: every listed result is rerun; the question is
  only how much it changes numerically.
  - Every AuSOC-lattice paper result (list: paper/studies/reports/2026-09-30-ausoc-bethe-usage-audit.md)
    is rerun under the same rules as 3.4.
  - The old-vs-new overlays from those reruns give the size of the change. There is no separate
    impact-check step.
- **Why the band structures never showed it:**
  - The flip is unitary on the atom, so the atomic splitting is unchanged.
  - Inside the t2g set it equals a 180-degree spin rotation about z, which leaves bulk bands
    unchanged.
  - The wrong t2g-e_g part vanishes wherever dxy does not mix with other orbitals. In the cubic
    frame that includes Gamma and the Gamma-X [001] line, where the bands are identical.
  - Elsewhere it enters mostly at second order.
  - Orientation invariance does catch it, which is test (b).
- **If the red tests do not fail,** defect 2 is dropped and recorded as a false alarm.

## 8. Review record

- **rev 1 spec review:** paper/planning/2026-09-30-bethe-onsite-frame-reviews/spec-review-r1.md.
  APPROVED WITH FIXES (0 Critical / 4 Important / 13 Minor). All math was hand-derived, on the login
  node with no numerics.
- **Folded into rev 2**
  - I1: T2 tolerance and label check.
  - I2: T3 is a direct call on the cubic <110> set; its role is made explicit.
  - I3: T1 covers a two-contact junction and checks each contact's own block.
  - I4: section 7.
  - M1: citations fixed.
  - M2: "PDT devices".
  - M3: "convention inherited".
  - M4: allclose wording.
  - M5: tolerance values, no-partner and duplicate errors.
  - M6: method named consistently. In rev 3 the method was merged into constructMat by owner ruling;
    in rev 4 the owner reversed that (separate method `constructOnsite`).
  - M7: test precondition, and AuSOC with spin='g'.
  - M8: d-block eigenvalue check in T1.
  - M9: T6, the twin stacking.
  - M10 and M11: section 4.
  - M12: per-item comparison.
  - M13: section 6.
