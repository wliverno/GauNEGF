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
