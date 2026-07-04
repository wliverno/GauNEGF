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
