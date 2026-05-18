# NEGF I/O Test Suite Design

Date: 2026-05-15
Author: collaborative (willll + Claude)

## Goal

Comprehensive pytest test suite covering input/output public methods of the
`NEGF` class in `gauNEGF/scf.py`, exercised against ethane (small, fast, real
molecule, already in `examples/`). Doubles as:

- Regression harness for the recently added `updatePulay()` method (untested
  by the user to date).
- TDD harness for `writeChk()`, which is currently broken. Tests are written
  first (RED), the fix follows after the actual failure mode is observed.

## Non-goals

- Full SCF convergence testing (one short maxcycles=3 run only, to exercise
  PMix + updatePulay in-loop).
- Energy-dependent NEGFE subclass (separate effort).
- Spin-polarized / non-collinear paths beyond constructor sanity (separate
  effort).

## Layout

```
tests/
  test_scf_io.py              # new
  scratch_scf_io/             # working dir for tests, gitignored
    ethane.gjf                # copied from examples/
    ethane.chk                # cached across runs
    ...
```

The session-scoped fixture builds the NEGF object once per pytest invocation.
The `.chk` cache means the second-and-later pytest runs skip the expensive
Gaussian SCF.

## Fixtures

- `ethane_negf` (scope=session): copies `examples/ethane.gjf` to
  `tests/scratch_scf_io/`, calls
  `NEGF(fn="ethane", basis="lanl2dz", func="b3lyp", spin="r")` from that cwd.
  Returns the object.
- `configured_negf` (scope=function): deep copies state from `ethane_negf`
  after calling `setContacts(lContact=[1], rContact=[2])`,
  `setSigma(sig=-0.1j)`, `setVoltage(0.0)`. Each test starts from a known
  fully-wired state. Note: deep copy is shallow at the `bar` level (qcb.BinAr
  is not copy-friendly); tests that mutate `bar` must order themselves last
  or use the unconfigured fixture and re-wire.

## Tests

1. `test_init_state` -- F/P/S shapes consistent; `nelec ~= 18.0` (ethane,
   neutral, Z = 2*6 + 6*1 = 18); `locs` length == nsto; `S` symmetric.
2. `test_setFock_roundtrip` -- call `setFock(F * har_to_eV)`, assert
   `self.F` matches original Hartree-unit F within 1e-12.
3. `test_setDen_roundtrip` -- call `setDen(P)`; nelec preserved; `getDen(bar)`
   returns same P within tol.
4. `test_getHOMOLUMO` -- returns length-2 array, HOMO < LUMO, both finite.
5. `test_setContacts` -- pass `lContact=[1]`, `rContact=[2]` (the two C
   atoms); resulting lInd/rInd index into the F matrix correctly via locs;
   `nelecContacts` matches sum of contact-atom atomic numbers (12 for two C).
6. `test_setSigma_scalar` -- scalar `sig=-0.1j`; assert `sigma1.shape ==
   F.shape`; `Gam1` and `Gam2` are Hermitian within tol; eigenvalues of
   Gam matrices are >= -1e-10 (positive semi-definite).
7. `test_setSigma_dim_mismatch` -- bad-shape vector raises Exception.
8. `test_setVoltage` -- `setVoltage(qV=0.1)`; assert `mu1 - mu2 == 0.1` (or
   matches the +/-qV/2 convention); fermi set when entered as None; the
   three X-/Y-/Z-EFIELD scalars are present on `bar`.
9. `test_getSigma` -- returns a 2-tuple matching `(sigma1, sigma2)`.
10. `test_updatePulay` -- start from default size N; call `updatePulay(3)`
    and assert pList shape `(3, nsto, nsto)`, DPList shape `(3, nsto, nsto)`,
    pMat shape `(4, 4)`, pB shape `(4,)`, pMat[-1,-1]==0, pB[-1]==-1; then
    `updatePulay(8)` and reverify; `updatePulay(0)` raises ValueError;
    pList is seeded with current `self.P`.
11. `test_saveMAT` -- save to `tmp_path/out.mat`; load with `scipy.io.loadmat`;
    assert keys `F, sig1, sig2, S, fermi, qV, spin, den, conv` present with
    correct shapes/types.
12. `test_short_SCF_and_updatePulay_inloop` -- run
    `SCF(conv=1e10, maxcycles=3, pulay=True)` (huge conv -> 3-cycle exit);
    assert three lists returned, all length 3; checkpoint `.mat` file created;
    then call `updatePulay(2)` and run another `SCF(conv=1e10, maxcycles=2)`
    to confirm the resized buffers survive a real SCF loop.

### writeChk (TDD, expected to fail initially)

13. `test_writeChk_creates_file` -- after `writeChk()`, `ethane.chk` exists
    and is > 0 bytes.
14. `test_writeChk_roundtrip` -- open the written `.chk` via
    `qcb.BinAr(inputfile=<chkfile>)` (or whatever the supported gauopen
    read path is); verify Fock and overlap matrices match the originals
    within tol.
15. `test_writeChk_cleans_fchk` -- intermediate `.fchk` is removed on
    success.
16. `test_writeChk_unfchk_failure` -- monkeypatch `os.system` to return
    non-zero; verify `.fchk` is kept (current fallback contract from the
    method's docstring).

## Subagent plan

Per user direction, all work is dispatched to haiku subagents.

1. **Agent A -- Scout.** Read `gauNEGF/density.py`, `gauNEGF/matTools.py`,
   and one existing test (`tests/test_densityReal.py`) to confirm function
   signatures (`getDen`, `storeDen`, `getFock`, `formSigma`), `bar.matlist`
   keys, and existing pytest style. Returns notes only, no code.
2. **Agent B -- Writer.** Receives this spec + Agent A notes; writes
   `tests/test_scf_io.py`. Must NOT run the tests (separate agent).
3. **Agent C -- Runner.** Runs `pytest tests/test_scf_io.py -v` on this
   compute node (P100, conda env loaded), returns full output. Expected:
   tests 1-12 PASS, 13-16 FAIL (writeChk is broken).
4. **Diagnosis (interactive with user).** User and Claude examine actual
   writeChk failure output together. No fixer agent until diagnosis is
   agreed.
5. **Agent D -- Fixer.** Implements the writeChk fix based on the
   diagnosis; runs the tests again to confirm GREEN.

## Risks / open questions

- `qcb.BinAr` may not deep-copy cleanly. If `configured_negf` blows up,
  fall back to re-wiring contacts/sigma/voltage at the top of each test
  that needs them.
- ethane DFT under b3lyp/lanl2dz at the chosen geometry may take 30-90s
  on first run; `.chk` reuse on later runs should drop this to seconds.
- writeChk roundtrip test depends on `qcb.BinAr` accepting `.chk` directly
  as input. If it doesn't (gauopen quirk), substitute a check that loads
  via `formchk + reading the .fchk back`.
- E-field scalar storage uses `round(field[N])` in scf.py, so for very
  small qV the rounded values may be zero -- the test must pick a qV
  large enough to produce a non-zero rounded field, or just check that
  the scalars exist on `bar`.

## Not committing

Per user's standing instruction (no commits by Claude), the test file is
written but not committed. User handles git.
