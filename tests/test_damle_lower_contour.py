"""Integration tests for the reworked damleLowerDensity on C2-STO3G.

Run via tests/run_pytest.job (needs gdv + the NEGF env). Validates:
  - sign-corrected: damle matrix matches densityComplex on a cores-in-window
    interval to < 1% (after the sign fix, no manual negation needed);
  - delta_N ~ 0 on a pole-free deep tail (the dropped-flat-term property).
"""
import os
import sys
import shutil
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from numpy.linalg import norm

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)
from gauNEGF.scfE import NEGFE
from gauNEGF.transport import har_to_eV
from gauNEGF.density import densityComplex, damleLowerDensity
from gauNEGF.config import ENERGY_MIN


def _c2_setup():
    scratch = tempfile.mkdtemp(prefix='c2_damle_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    return negf


def test_damle_matrix_matches_densitycomplex_on_cores():
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    # Emin in the core/valence gap (1s cores ~ -271, valence from ~ -18).
    Emin_gap = -145.0
    P_damle, dN = damleLowerDensity(F_eV, negf.Y_eff, negf.Sigma_0, negf.g, Emin_gap)
    P_cplx, _ = densityComplex(F_eV, S, negf.g, ENERGY_MIN, Emin_gap, T=0)
    rel = norm(np.asarray(P_damle) - np.asarray(P_cplx)) / max(norm(np.asarray(P_cplx)), 1e-30)
    assert rel < 0.01, f'sign-corrected damle disagreement {rel:.3e} (expect <1%)'


def test_damle_delta_N_negligible_on_pole_free_tail():
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    # Emin below all poles (cores ~ -271): tail [-1e6, -291] is empty.
    Emin_deep = -291.0
    _, dN = damleLowerDensity(F_eV, negf.Y_eff, negf.Sigma_0, negf.g, Emin_deep)
    assert abs(dN) < 1e-2, f'pole-free tail delta_N={dN:.3e} (expect ~0)'


def test_focktop_warns_when_pole_below_emin(capsys, monkeypatch):
    # When Emin lands too shallow and the lower contour traps real charge, the
    # warning must fire. FockToP now recomputes Emin every cycle via calcEmin
    # (per-cycle update), so we cannot force a shallow Emin by setting the
    # attribute -- it gets overwritten. Monkeypatch calcEmin to force Emin=-145
    # (in the C2 core/valence gap) instead, so the 1s cores sit BELOW Emin and
    # the lower contour traps ~2 electrons -> tr(P@S) > 0.5 -> warning fires.
    import gauNEGF.scfE as scfE
    monkeypatch.setattr(scfE, 'calcEmin', lambda *a, **kw: -145.0)
    negf = _c2_setup()
    negf.N2 = None
    negf.updFermi = False   # isolate the warning: skip the full Fermi search
    negf.damle_dN_warn = 0.5
    negf.FockToP()
    out = capsys.readouterr().out
    assert 'lower-contour holds significant weight' in out


def test_focktop_no_warning_in_normal_operation(capsys):
    # In normal SCF operation FockToP places Emin below the whole band each
    # cycle (calcEmin true-Sigma DOS), so the lower contour is empty (~1e-7
    # weight) and the warning must STAY SILENT (no false positive).
    negf = _c2_setup()
    negf.N2 = None
    negf.updFermi = False
    negf.damle_dN_warn = 0.5
    negf.FockToP()
    out = capsys.readouterr().out
    assert 'lower-contour holds significant weight' not in out


def test_setvoltage_places_emin_below_band_via_calcemin():
    negf = _c2_setup()  # setVoltage(0.0) already ran inside
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    # Emin must sit below the lowest pole (the C2 1s cores ~ -271 eV).
    band_floor = float(np.real(np.linalg.eigvals(np.linalg.solve(S, F_eV))).min())
    assert negf.Emin <= band_floor, f'Emin={negf.Emin} not below band floor {band_floor}'
    # Eminf must be the wide config bound (no calcTSW narrowing).
    from gauNEGF.config import ENERGY_MIN as EMIN_CFG
    assert abs(negf.Eminf - EMIN_CFG) < 1.0


def test_focktop_replaces_stale_emin_for_current_fock():
    # Emin, Sigma_0, Y_eff all depend on self.F, which changes every SCF cycle.
    # FockToP must re-place Emin on the CURRENT Fock, not reuse a frozen value.
    negf = _c2_setup()
    negf.updFermi = False   # isolate: skip the Fermi search
    # Stale Emin carried from a hypothetical prior cycle: absurdly shallow,
    # above the C2 1s cores (~ -271 eV). A frozen-Emin FockToP keeps this;
    # the per-cycle recompute re-places it below the band (calcEmin ~ -281).
    negf.Emin = -30.0
    negf.FockToP()
    assert negf.Emin < -260.0, f'FockToP did not re-place stale Emin: {negf.Emin}'


def test_focktop_refits_stale_sigma0_for_current_fock():
    # FockToP must refit the asymptotic fit (Sigma_0/Y_eff) on the current Fock.
    # A stale Sigma_0/Y_eff applied to an evolved F yields a garbage Fbar (the
    # Au3 crash). Here F is unchanged, so a correct refit restores the setup value.
    negf = _c2_setup()
    negf.updFermi = False
    sigma0_setup = np.asarray(negf.Sigma_0).copy()
    # Corrupt the cached fit (simulate staleness w.r.t. the current Fock).
    negf.Sigma_0 = np.zeros_like(negf.Sigma_0)
    negf.FockToP()
    rel = norm(np.asarray(negf.Sigma_0) - sigma0_setup) / max(norm(sigma0_setup), 1e-30)
    assert rel < 1e-6, f'FockToP did not refit Sigma_0 (rel diff {rel:.3e})'


def test_init_asymptotic_sigma_sets_damle_emin_from_fbar_floor():
    # damleEmin is the Damle-anchored Emin guess: band floor of the operator
    # Damle integrates, Fbar = Y_eff (F + Sigma_0) Y_eff, minus damle_buffer.
    # For large-Sigma_0 systems Fbar's floor sits well below the bare-Fock floor,
    # so this is the right anchor (CNT33_5cell mismatch case).
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    Y = np.asarray(negf.Y_eff)
    S0 = np.asarray(negf.Sigma_0)
    Fbar = Y @ (F_eV + S0) @ Y
    fbar_floor = float(np.real(np.linalg.eigvals(Fbar)).min())
    expected = fbar_floor - negf.damle_buffer
    assert hasattr(negf, 'damleEmin'), 'damleEmin not set by _initAsymptoticSigma'
    assert abs(negf.damleEmin - expected) < 1e-6, \
        f'damleEmin={negf.damleEmin}, expected={expected} (Fbar floor {fbar_floor} - buffer {negf.damle_buffer})'


def test_focktop_seeds_calcemin_with_damle_emin_guess(monkeypatch):
    # FockToP recomputes Emin via calcEmin every cycle (per-cycle update). The
    # seed handed to calcEmin must be self.damleEmin (the Fbar-anchored guess),
    # not None. This keeps the DOS loop confirming/deepening from where Damle
    # actually has its band floor, fixing the true-vs-Damle spectrum mismatch
    # that CNT33_5cell hits.
    negf = _c2_setup()
    negf.updFermi = False
    captured = {}
    def fake_calcEmin(F, S, g, tol=None, maxN=None, Emin=None):
        captured['Emin'] = Emin
        return -281.0  # plausible deep value so FockToP downstream runs cleanly
    import gauNEGF.scfE as scfE
    monkeypatch.setattr(scfE, 'calcEmin', fake_calcEmin)
    negf.FockToP()
    assert 'Emin' in captured, 'FockToP did not call calcEmin'
    assert captured['Emin'] is not None, \
        'FockToP called calcEmin without an Emin seed (the Fbar-anchored guess)'
    assert abs(captured['Emin'] - negf.damleEmin) < 1e-9, \
        f'FockToP seeded calcEmin with {captured["Emin"]}, expected damleEmin={negf.damleEmin}'


def test_calcemin_default_offset_is_EMIN_BUFFER():
    # calcEmin without an Emin seed used to subtract a hardcoded 5 eV from the
    # band-floor eigenvalue; it now subtracts EMIN_BUFFER (20 eV by default).
    # For C2-STO3G this puts the no-seed start ~15 eV deeper than before.
    from gauNEGF.density import calcEmin
    from gauNEGF.config import EMIN_BUFFER
    assert EMIN_BUFFER > 10.0, f'EMIN_BUFFER unexpectedly small: {EMIN_BUFFER}'
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    emin = float(calcEmin(F_eV, S, negf.g, tol=1e-3))
    # Old behavior (5 eV offset): emin ~ -281 (C2). New (EMIN_BUFFER=20):
    # emin ~ -296. Anything <= -290 confirms the new offset is in effect.
    assert emin <= -290.0, \
        f'calcEmin returned {emin}; expected <= -290 with EMIN_BUFFER={EMIN_BUFFER}'


def test_calctsw_floors_eminf_to_warmstart_when_shallower(capsys):
    # calcTSW's postcondition: never return Eminf shallower than the warm-start
    # (would invert the caller's lower-contour integration in getFermiContact's
    # densityComplex(F, S, g, Eminf, Emin, ...)). The loop structure preserves
    # this for valid negative warm-starts (doubling deepens), but the guard
    # catches invalid input -- here a positive warm-start, which the doubling
    # loop would naturally drift more positive (= shallower in lower-contour
    # terms). The postcondition floors to the warm-start and warns.
    from gauNEGF.density import calcTSW
    negf = _c2_setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    bad_warm = 5.0   # invalid positive warm-start
    Eminf_out, _ = calcTSW(F_eV, S, negf.g, tol=1e-3, maxN=2, Eminf=bad_warm)
    out = capsys.readouterr().out
    assert Eminf_out == bad_warm, \
        f'calcTSW returned Eminf={Eminf_out}, expected floor to warm-start {bad_warm}'
    assert 'shallower' in out and 'flooring' in out, \
        f'calcTSW did not warn about flooring; captured stdout:\n{out[:400]}'
