"""Characterization tests for NEGFE.spawnNEGF (gauNEGF/scfE.py:479).

spawnNEGF converts an energy-DEPENDENT NEGFE calculation (Bethe contacts)
into an energy-INDEPENDENT constant-sigma NEGF by freezing sigma at mu1/mu2.
These tests characterize its behavior on the smallest real Bethe system,
AuTipPDT (280 orbitals, restricted, T=0), reconstructed from the banked
converged state in
paper/studies/02_pdt_bethe/runs/AuTipPDT_b3lyp_r_Au_ESCF_v2fix/.

FINDINGS (characterization runs, jobs 37900813 + 37900979, 2026-07-29):
  - The SUSPECTED setDen-before-setVoltage order bug does NOT crash:
    the spawned object is a plain NEGF, whose PToFock (scf.py:703) never
    touches mu1/mu2 -- the RuntimeError trap is NEGFE.PToFock-specific
    (scfE.py:764).
  - REAL BUG (test_03c, strict xfail): spawnNEGF leaves spawn.F at the
    HARRIS-GUESS Fock. negf.setDen(P) -> NEGF.PToFock runs Gaussian
    dofock=DENSITY, which updates only bar's internal Fock; base
    NEGF.PToFock never refreshes self.F (only FockToP does, scf.py:567;
    NEGFE.PToFock does refresh, scfE.py:771). Measured: bar-internal
    Fock matches parent.F to 3.2e-14 hartree while spawn.F is off by
    0.925 hartree; T(Ef) from spawn.F = 0.979 vs parent 2.39e-4
    (4000x). spawn.SCF() self-heals on cycle 0 (FockToP pulls bar's
    Fock), but any DIRECT use of spawn.F after spawnNEGF -- transmission,
    getHOMOLUMO, and the updFermi branch's HOMO/LUMO fermi guess inside
    spawnNEGF itself -- sees the Harris Fock.
    PROPOSED PATCH (scfE.py, spawnNEGF, after negf.setDen(self.P)):
        negf.F, negf.locs = getFock(negf.bar, negf.spin)
    (getFock already in scope via matTools import; mirrors NEGFE.PToFock.)
  - With the Fock corrected, spawn T(Ef) matches parent to 1.04e-8
    absolute; the residual is formSigma's -1e-9j*S off-block seed
    (stripping it gives machine-level agreement, test_03b receipt).

Covers:
  01  spawnNEGF() does not raise (setDen-order adjudication)
  01b manual workaround (setSigma -> setVoltage -> setDen) if 01 fails
  02  frozen sigma blocks equal parent.getSigma(fermi) blocks
  03a spawn's bar-internal Fock == parent.F (isolates the stale-self.F bug)
  03b T(Ef) parent == spawn using bar Fock (proves the proposed patch)
  03c T(Ef) using spawn.F as a user would -- strict xfail on the known bug
  04  T(Ef +/- 2 eV) parent vs spawn diverge (frozen sigma is E-independent)
  05  spawn.SCF(1e-3, 1e-3, 5): no crash/NaN; cycles + convLevel REPORTED
  06  wall time: one warm spawned-NEGF cycle vs one NEGFE cycle (receipt)

Requires gdv (constructing NEGF/NEGFE runs Gaussian immediately) -- run via
sbatch on a compute node (see tests/run_pytest_slow.job pattern). Marked
slow so the default suite (-m "not slow") never launches Gaussian jobs
this heavy; run explicitly with -m slow.
"""
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import pytest
from scipy import io as sio

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which('gdv') is None,
                       reason='gdv not in PATH (run via sbatch with '
                              'module load anantram/gdv)'),
]

SRC = ('/mmfs1/gscratch/anantram/willll/NEGFCode/paper/studies/02_pdt_bethe/'
       'runs/AuTipPDT_b3lyp_r_Au_ESCF_v2fix')
MAT = os.path.join(SRC, 'AuTipPDT_b3lyp_r_Au_ESCF.mat')
TMAT = os.path.join(SRC, 'T.AuTipPDT_b3lyp_r_Au_ESCF.mat')
LCONTACT = [1, 2, 3]
RCONTACT = [20, 21, 22]
TAG = '[SPAWN-CHAR]'


def _bar_fock(obj):
    """Fock currently held by the Gaussian bar (not obj.F, which can lag)."""
    from gauNEGF.matTools import getFock
    F, _ = getFock(obj.bar, obj.spin)
    return np.asarray(F)


@pytest.fixture(scope='module')
def results():
    """Cross-test scratchboard for the characterization report."""
    return {}


@pytest.fixture(scope='module')
def parent_ctx(tmp_path_factory):
    """Reconstruct the converged AuTipPDT NEGFE parent in a private dir."""
    scratch = tmp_path_factory.mktemp('spawn_negf_parent')
    for f in ('AuTipPDT.gjf', 'AuTipPDT.chk', 'Au.bethe'):
        shutil.copy(os.path.join(SRC, f), os.path.join(scratch, f))
    saved = sio.loadmat(MAT)
    fermi = float(saved['fermi'][0][0])
    den = saved['den']

    cwd = os.getcwd()
    os.chdir(scratch)
    from gauNEGF.scfE import NEGFE
    negf = NEGFE(fn='AuTipPDT', func='b3lyp', basis='chkbasis', spin='r',
                 route='integral=grid=superfine')
    negf.setContactBethe([LCONTACT, RCONTACT], 'Au', T=0.0)
    negf.setVoltage(0.0, fermi)  # setVoltage BEFORE setDen (documented trap)
    negf.setDen(den)
    try:
        yield {'negf': negf, 'scratch': str(scratch), 'fermi': fermi}
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _in_scratch(parent_ctx):
    """Override conftest's repo-root chdir: Gaussian files live in scratch."""
    os.chdir(parent_ctx['scratch'])
    yield


@pytest.fixture(scope='module')
def spawn_ctx(parent_ctx, results):
    """Attempt spawnNEGF once; capture outcome (success or exact traceback)."""
    os.chdir(parent_ctx['scratch'])
    negf = parent_ctx['negf']
    try:
        t0 = time.perf_counter()
        spawn = negf.spawnNEGF()
        results['spawn_ok'] = True
        results['spawn_tb'] = None
        results['spawn_obj'] = spawn
        results['spawn_source'] = 'spawnNEGF'
        results['spawn_wall_s'] = time.perf_counter() - t0
    except Exception:
        results['spawn_ok'] = False
        results['spawn_tb'] = traceback.format_exc()
        results['spawn_obj'] = None
        results['spawn_source'] = None
    return results


@pytest.fixture()
def spawn(spawn_ctx):
    """Spawned object (or manual-workaround object) for downstream tests."""
    if spawn_ctx.get('spawn_obj') is None:
        pytest.skip('no spawned object available (spawnNEGF failed and '
                    'workaround not built or also failed)')
    return spawn_ctx['spawn_obj']


def test_01_spawn_does_not_raise(spawn_ctx):
    """spawnNEGF() must not raise. If it does, the traceback IS the finding."""
    if not spawn_ctx['spawn_ok']:
        print(f'{TAG} spawnNEGF RAISED. Exact traceback follows:')
        print(spawn_ctx['spawn_tb'])
        pytest.fail('spawnNEGF raised (setDen-order verdict: see traceback '
                    'above):\n' + spawn_ctx['spawn_tb'])
    print(f"{TAG} spawnNEGF succeeded in {spawn_ctx['spawn_wall_s']:.1f} s")
    print(f'{TAG} setDen-order verdict: setDen-before-setVoltage did NOT '
          f'crash the spawned plain NEGF (NEGF.PToFock does not need mu1/mu2; '
          f'the RuntimeError trap is NEGFE.PToFock-specific).')


def test_01b_manual_workaround_order(parent_ctx, spawn_ctx, results):
    """Only if spawnNEGF crashed: prove setSigma->setVoltage->setDen works."""
    if spawn_ctx['spawn_ok']:
        pytest.skip('spawnNEGF succeeded; workaround not needed')
    negf = parent_ctx['negf']
    from gauNEGF.scf import NEGF
    sig1, sig2 = negf.getSigma(negf.mu1, negf.mu2)
    manual = NEGF('AuTipPDT', negf.basis, negf.func, negf.spin, False,
                  negf.otherRoute, negf.section)
    manual.setSigma(LCONTACT, RCONTACT,
                    np.asarray(sig1)[np.ix_(negf.lInd, negf.lInd)],
                    np.asarray(sig2)[np.ix_(negf.rInd, negf.rInd)])
    manual.setVoltage(0.0, parent_ctx['fermi'])  # voltage BEFORE density
    manual.setDen(negf.P)
    results['spawn_obj'] = manual
    results['spawn_source'] = 'manual-workaround'
    print(f'{TAG} manual workaround (setSigma -> setVoltage -> setDen) '
          f'succeeded; downstream tests use it.')


def test_02_frozen_sigma_matches_parent(parent_ctx, spawn):
    """Spawn's stored sigma blocks == parent.getSigma(fermi) blocks."""
    negf = parent_ctx['negf']
    fermi = parent_ctx['fermi']
    assert negf.mu1 == negf.mu2 == pytest.approx(fermi), \
        'qV=0 reconstruction should give mu1 == mu2 == fermi'
    sig1_ref = np.asarray(negf.getSigma(fermi)[0])
    sig2_ref = np.asarray(negf.getSigma(fermi)[1])
    lIx = np.ix_(negf.lInd, negf.lInd)
    rIx = np.ix_(negf.rInd, negf.rInd)
    s1 = np.asarray(spawn.sigma1)
    s2 = np.asarray(spawn.sigma2)
    d1 = np.max(np.abs(s1[lIx] - sig1_ref[lIx]))
    d2 = np.max(np.abs(s2[rIx] - sig2_ref[rIx]))
    print(f'{TAG} max|sigma1 block diff| = {d1:.3e}, '
          f'max|sigma2 block diff| = {d2:.3e}')
    # Off-block characterization via masks (no in-place writes: jax buffers
    # can surface as read-only numpy views).
    mask1 = np.zeros(s1.shape, dtype=bool)
    mask1[lIx] = True
    bg = -1e-9j * np.asarray(spawn.S)  # formSigma's seed background
    off_dev = np.max(np.abs((s1 - bg)[~mask1]))
    par_off = np.max(np.abs(sig1_ref[~mask1]))
    print(f'{TAG} spawn sigma1 off-block deviation from -1e-9j*S = '
          f'{off_dev:.3e}; parent sigma1 off-block max = {par_off:.3e} '
          f'(freezing block-only loses this much)')
    # spawnNEGF passes full-support sigma; setSigma's full-size path adds
    # the -1e-9j*S background additively, perturbing blocks at 1e-9 scale.
    assert np.allclose(s1[lIx], sig1_ref[lIx], rtol=0.0, atol=1e-8)
    assert np.allclose(s2[rIx], sig2_ref[rIx], rtol=0.0, atol=1e-8)


def test_03a_bar_fock_matches_parent(parent_ctx, spawn, results):
    """Diagnosis: bar holds Fock(P_parent); spawn.F lags at the Harris guess."""
    negf = parent_ctx['negf']
    Fbar = _bar_fock(spawn)
    dbar = np.max(np.abs(Fbar - np.asarray(negf.F)))
    dself = np.max(np.abs(np.asarray(spawn.F) - np.asarray(negf.F)))
    results['dF_bar'] = dbar
    results['dF_self'] = dself
    print(f'{TAG} max|barFock(spawn) - parent.F| = {dbar:.3e} hartree '
          f'(same density through the same Gaussian dofock=DENSITY)')
    print(f'{TAG} max|spawn.F - parent.F| = {dself:.3e} hartree '
          f'(self.F never refreshed after setDen -> stale Harris Fock)')
    assert dbar < 1e-8, ('bar Fock should be the parent Fock rebuilt from '
                         'the same density')


def test_03b_transmission_at_fermi_with_bar_fock(parent_ctx, spawn, results):
    """T(Ef): parent == spawn ONCE the bar Fock is used (the proposed patch)."""
    from gauNEGF.transport import (har_to_eV, SigmaCalculator,
                                   calculate_transmission)
    negf = parent_ctx['negf']
    fermi = parent_ctx['fermi']
    Tpar = calculate_transmission(np.asarray(negf.F) * har_to_eV, negf.S,
                                  SigmaCalculator(negf.g), [fermi])[0]
    Tsp = calculate_transmission(_bar_fock(spawn) * har_to_eV, spawn.S,
                                 SigmaCalculator(spawn.sigma1, spawn.sigma2),
                                 [fermi])[0]
    rel = abs(Tpar - Tsp) / max(abs(Tpar), 1e-30)
    print(f'{TAG} T(Ef={fermi:.4f} eV): parent = {Tpar:.8e}, '
          f'spawn(barF) = {Tsp:.8e}, rel diff = {rel:.3e}, '
          f'abs diff = {abs(Tpar - Tsp):.3e}')
    # External receipt: banked T curve (fermi-relative grid) near E-Ef = 0.
    banked = sio.loadmat(TMAT)
    Eb = np.ravel(banked['Elist'])
    Tb = np.ravel(banked['T'])
    Tb0 = float(Tb[np.argmin(np.abs(Eb))])
    print(f'{TAG} banked T at E-Ef ~ 0 = {Tb0:.8e} '
          f'(reconstruction receipt, report-only)')
    # Attribution receipt: strip formSigma's -1e-9j*S off-block seed; the
    # remaining diff is then only the 3e-14-hartree Fock rebuild noise.
    negf_lIx = np.ix_(negf.lInd, negf.lInd)
    negf_rIx = np.ix_(negf.rInd, negf.rInd)
    s1c = np.zeros_like(np.asarray(spawn.sigma1))
    s2c = np.zeros_like(np.asarray(spawn.sigma2))
    s1c[negf_lIx] = np.asarray(spawn.sigma1)[negf_lIx]
    s2c[negf_rIx] = np.asarray(spawn.sigma2)[negf_rIx]
    Tclean = calculate_transmission(_bar_fock(spawn) * har_to_eV, spawn.S,
                                    SigmaCalculator(s1c, s2c), [fermi])[0]
    rel_clean = abs(Tpar - Tclean) / max(abs(Tpar), 1e-30)
    print(f'{TAG} spawn(barF, background stripped) = {Tclean:.8e}, '
          f'rel diff = {rel_clean:.3e} (residual in the line above is the '
          f'-1e-9j*S seed, not the frozen blocks)')
    results['T_ef'] = (Tpar, Tsp, rel, Tb0)
    assert np.isfinite(Tpar) and np.isfinite(Tsp)
    # T(Ef) ~ 2.4e-4 here, so gate on ABSOLUTE agreement at the eta-seed
    # scale (measured 1.04e-8) with margin, plus a loose relative guard.
    assert abs(Tpar - Tsp) < 1e-7, (
        f'T(Ef) abs mismatch beyond the -1e-9j*S background scale: parent '
        f'{Tpar!r} vs spawn {Tsp!r}')
    assert rel < 1e-3
    assert rel_clean < 1e-6, (
        f'background-stripped spawn T(Ef) should match parent to machine '
        f'level: rel {rel_clean:.3e}')


# xfail REMOVED 2026-07-29: the stale-F patch landed (scfE.py spawnNEGF
# now refreshes negf.F via getFock after setDen); this test asserts the
# fixed contract -- spawn.F is transmission-ready immediately.
def test_03c_transmission_from_self_F_contract(parent_ctx, spawn):
    """The natural user path (spawn.F straight after spawnNEGF) -- xfail."""
    from gauNEGF.transport import (har_to_eV, SigmaCalculator,
                                   calculate_transmission)
    negf = parent_ctx['negf']
    fermi = parent_ctx['fermi']
    Tpar = calculate_transmission(np.asarray(negf.F) * har_to_eV, negf.S,
                                  SigmaCalculator(negf.g), [fermi])[0]
    Tsp = calculate_transmission(np.asarray(spawn.F) * har_to_eV, spawn.S,
                                 SigmaCalculator(spawn.sigma1, spawn.sigma2),
                                 [fermi])[0]
    rel = abs(Tpar - Tsp) / max(abs(Tpar), 1e-30)
    print(f'{TAG} T(Ef) via spawn.F: parent = {Tpar:.8e}, '
          f'spawn(self.F) = {Tsp:.8e}, rel diff = {rel:.3e}')
    assert rel < 1e-5, ('spawn.F is stale (Harris guess); see xfail reason '
                        f'-- rel diff {rel:.3e}')


def test_04_transmission_diverges_off_fermi(parent_ctx, spawn, results):
    """T at Ef +/- 2 eV should DIFFER (frozen sigma is energy-independent)."""
    from gauNEGF.transport import (har_to_eV, SigmaCalculator,
                                   calculate_transmission)
    negf = parent_ctx['negf']
    fermi = parent_ctx['fermi']
    Elist = [fermi - 2.0, fermi + 2.0]
    Tpar = calculate_transmission(np.asarray(negf.F) * har_to_eV, negf.S,
                                  SigmaCalculator(negf.g), Elist)
    Tsp = calculate_transmission(_bar_fock(spawn) * har_to_eV, spawn.S,
                                 SigmaCalculator(spawn.sigma1, spawn.sigma2),
                                 Elist)
    rels = []
    for E, tp, ts in zip(Elist, Tpar, Tsp):
        r = abs(tp - ts) / max(abs(tp), 1e-30)
        rels.append(r)
        print(f'{TAG} T(E={E:.4f} eV, Ef{E - fermi:+.1f}): '
              f'parent = {tp:.6e}, spawn = {ts:.6e}, rel diff = {r:.3e}')
    results['T_off'] = list(zip(Elist, Tpar, Tsp, rels))
    assert np.all(np.isfinite(Tpar)) and np.all(np.isfinite(Tsp))
    assert max(rels) > 1e-4, (
        'parent and spawn UNEXPECTEDLY AGREE away from Ef '
        f'(rel diffs {rels}); frozen sigma should diverge at Ef +/- 2 eV')


def test_05_spawn_scf_stability(spawn, results):
    """spawn.SCF(1e-3, 1e-3, 5): report cycles + convLevel; assert no NaN."""
    t0 = time.perf_counter()
    count, PP, TotalE = spawn.SCF(1e-3, 1e-3, 5, checkpoint=False)
    wall = time.perf_counter() - t0
    ncyc = len(TotalE)
    results['scf_cycles'] = ncyc
    results['scf_convLevel'] = spawn.convLevel
    results['scf_wall_s'] = wall
    print(f'{TAG} spawn.SCF ran {ncyc} cycles in {wall:.1f} s '
          f'({wall / max(ncyc, 1):.1f} s/cycle incl. first-cycle JIT)')
    print(f'{TAG} final convLevel = {spawn.convLevel:.3e} '
          f'(conv target 1e-3; convergence NOT asserted -- the spawned map '
          f'need not share the parent fixed point)')
    print(f'{TAG} nelec trajectory: {[f"{p:.4f}" for p in PP]}')
    assert spawn.convLevel != 9999, \
        'convLevel sentinel 9999: a Gaussian cycle errored during spawn.SCF'
    assert np.isfinite(spawn.convLevel)
    assert not np.any(np.isnan(np.asarray(spawn.P))), 'NaN in spawned P'
    assert not np.any(np.isnan(np.asarray(spawn.F))), 'NaN in spawned F'
    assert np.all(np.isfinite(np.asarray(PP, dtype=float))), 'NaN/inf nelec'


def test_06_cycle_timing_receipt(parent_ctx, spawn, results):
    """Warm one-cycle wall time: spawned NEGF vs parent NEGFE (report-only)."""
    negf = parent_ctx['negf']

    def one_cycle(obj):
        t0 = time.perf_counter()
        obj.FockToP()
        t1 = time.perf_counter()
        obj.PMix(1e-3, False)
        dE = obj.PToFock()
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, t2 - t0

    # Spawn is warm (test_05 ran 5+ cycles).
    sp_fock, sp_rest, sp_tot = one_cycle(spawn)
    # Parent cycle 1 is cold (first FockToP contour = JIT compile); cycle 2 warm.
    pa1_fock, pa1_rest, pa1_tot = one_cycle(negf)
    pa2_fock, pa2_rest, pa2_tot = one_cycle(negf)
    ratio = pa2_tot / sp_tot
    results['timing'] = {'spawn': sp_tot, 'parent_cold': pa1_tot,
                         'parent_warm': pa2_tot, 'ratio_warm': ratio}
    print(f'{TAG} spawn warm cycle:  FockToP {sp_fock:8.2f} s + '
          f'mix/PToFock {sp_rest:8.2f} s = {sp_tot:8.2f} s')
    print(f'{TAG} parent cold cycle: FockToP {pa1_fock:8.2f} s + '
          f'mix/PToFock {pa1_rest:8.2f} s = {pa1_tot:8.2f} s (incl. JIT)')
    print(f'{TAG} parent warm cycle: FockToP {pa2_fock:8.2f} s + '
          f'mix/PToFock {pa2_rest:8.2f} s = {pa2_tot:8.2f} s')
    print(f'{TAG} TIMING RECEIPT: warm NEGFE cycle / warm spawned cycle = '
          f'{ratio:.1f}x (FockToP-only ratio '
          f'{pa2_fock / max(sp_fock, 1e-12):.1f}x; the shared Gaussian '
          f'dofock=DENSITY dominates the spawned cycle)')
    assert np.isfinite(ratio) and sp_tot > 0 and pa2_tot > 0
