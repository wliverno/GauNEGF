"""Contact-restricted Fermi search (spec 2026-10-12-contact-neutral-fermi-spec.md). Login-safe 4-site synthetic device
(same surfG as tests/test_fermi_freeze_guard.py). Contact count on sites [0,3] is flat ~1.0 per spin across the gap;
its 0.5-per-spin crossing is at -0.937 eV; the device-neutral Ef (2.0 per spin) is near +0.25 eV."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from gauNEGF.surfG1D import surfG
from gauNEGF.density import (count_electrons, densityComplex, calcFermiBisect, calcFermiMuller,
                             calcFermiSecant, calcFermiPolyFit)

CONTACT = np.array([0, 3])


def _g():
    N = 4
    F = np.diag([-1.0, 0.0, 1.0, 2.0]).astype(float)
    S = np.eye(N)
    g = surfG(F, S, [np.array([0]), np.array([N - 1])],
              taus=[np.array([[-0.3]]), np.array([[-0.3]])], staus=[np.array([[0.05]]), np.array([[0.05]])],
              alphas=[np.array([[0.0]]), np.array([[0.0]])], aOverlaps=[np.array([[1.0]]), np.array([[1.0]])],
              betas=[np.array([[-0.5]]), np.array([[-0.5]])], bOverlaps=[np.array([[0.05]]), np.array([[0.05]])],
              eta=1e-5, spin='r')
    g.setF(F, mu1=0.5, mu2=0.5)
    return g


def _count_at(g, Ef, inds, cross=True):
    g.setF(g.F, Ef, Ef)
    P, dN = densityComplex(g.F, g.S, g, -30.0, Ef, 1e-4, 0.0)
    return count_electrons(P, g.S, dN, inds, cross)


def test_count_device_is_todays_expression_bit_for_bit():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)); P = A + A.T
    B = rng.normal(size=(4, 4)); S = np.eye(4) + 0.05 * (B + B.T)
    assert count_electrons(P, S, 0.37) == np.trace(P @ S).real + 0.37


def test_count_subset_and_cross_flag():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4)); P = A + A.T
    S = np.eye(4)
    assert np.isclose(count_electrons(P, S, 0.2, CONTACT), P[0, 0] + P[3, 3] + 0.2)
    assert np.isclose(count_electrons(P, S, 0.2, CONTACT, cross=False), P[0, 0] + P[3, 3])
    assert np.isclose(count_electrons(P, S, 0.2, None, cross=False), np.trace(P))


def test_bisect_bracket_regression_reaches_far_target():
    # today's bracketing loop leaves Ef at the bound while Ncurr is at the stepped-past E -> uBound == lBound -> revert
    g = _g()
    Ef, dE, P = calcFermiBisect(g, 2.0, -30.0, -0.9, None, tol=1e-4, conv=1e-4, maxcycles=20)
    assert Ef != -0.9 and abs(_count_at(g, Ef, None) - 2.0) < 1e-3


def test_bisect_meets_contact_target():
    g = _g()
    Ef, dE, P = calcFermiBisect(g, 0.5, -30.0, -0.9, None, tol=1e-4, conv=1e-4, maxcycles=20, inds=CONTACT)
    assert abs(_count_at(g, Ef, CONTACT) - 0.5) < 1e-3 and abs(Ef - (-0.937)) < 0.02


def test_muller_secant_poly_meet_contact_target():
    for fn in (calcFermiMuller, calcFermiSecant, calcFermiPolyFit):
        g = _g()
        out = fn(g, 0.5, -30.0, -0.9, None, tol=1e-4, conv=1e-4, inds=CONTACT)
        assert abs(_count_at(g, out[0], CONTACT) - 0.5) < 1e-3, fn.__name__


def test_secant_stays_in_spectrum():
    g = _g()          # from the gap the contact count is flat: unbounded secant ran to 3.9e20 eV in review
    out = calcFermiSecant(g, 0.5, -30.0, 0.5, None, tol=1e-4, conv=1e-4, inds=CONTACT, lBound=-30.0, uBound=5.0)
    assert -30.0 <= out[0] <= 5.0


def test_bisect_revert_density_at_restored_ef():
    g = _g()
    Ef0 = 0.5
    Ef, dE, P = calcFermiBisect(g, 1.2, -30.0, Ef0, None, tol=1e-4, conv=1e-4, maxcycles=8,
                                inds=np.array([3]))        # site 3 holds at most 1 per spin: never reached
    assert Ef == Ef0
    g.setF(g.F, Ef0, Ef0)
    P0, _ = densityComplex(g.F, g.S, g, -30.0, Ef0, 1e-4, 0.0)
    assert np.allclose(np.asarray(P), np.asarray(P0))


import inspect
import pytest
from gauNEGF.scfE import NEGFE
from test_fermi_freeze_guard import _build_focktop_shell


def _shell(mode=None, n_contact=2.0):
    obj = _build_focktop_shell()
    obj.nelecContacts = n_contact
    if mode is not None:
        obj._setFermiCount(mode)
    return obj


def test_contact_mode_sets_inds_and_total_target():
    obj = _shell('contact')
    assert list(obj.fermiCountInds) == [0, 3] and obj.fermiTarget == 2.0 and obj.fermiCross is True


def test_device_bare_drops_cross_and_has_no_inds():
    obj = _shell('device_bare')
    assert obj.fermiCountInds is None and obj.fermiTarget is None and obj.fermiCross is False


def test_contacts_covering_device_collapse_to_device():
    obj = _shell()
    obj.lInd, obj.rInd = np.array([0, 1]), np.array([2, 3])
    obj._setFermiCount('contact')
    assert obj.fermiCountInds is None and obj.fermiTarget is None and obj.fermiCross is True


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        _shell('contacts')
    with pytest.raises(ValueError):
        _shell('contact_bare')


def test_bethe_rejects_device_count():
    with pytest.raises(ValueError, match='device_bare'):
        NEGFE.setContactBethe(object(), [[1], [2]], fermiCount='device')


def test_bethe_contact_count_warns():
    with pytest.warns(RuntimeWarning, match='same atoms as the device basis'):
        with pytest.raises((AttributeError, TypeError)):   # the bare object fails right after the warning
            NEGFE.setContactBethe(object(), [[1], [2]], fermiCount='contact')


def test_new_keywords_are_last():
    for fn, default in ((NEGFE.setContact1D, 'contact'), (NEGFE.setContactBethe, 'device_bare')):
        params = inspect.signature(fn).parameters
        assert list(params)[-1] == 'fermiCount' and params['fermiCount'].default == default


import gauNEGF.scfE as scfE_mod
import gauNEGF.density as dens_mod


def _contact_shell(method='muller', target_total=1.0, inds=(0, 3), start=-0.9):
    # start below the gap: the contact count is flat across it; FockToP's crossing is at -1.002 eV
    obj = _build_focktop_shell()
    obj.fermiCross = True
    obj.fermiCountInds, obj.fermiTarget = np.array(inds), target_total
    obj.fermiMethod = method
    obj.fermi = obj.mu1 = obj.mu2 = start
    obj.Emax = 2.0                       # the shell has no __init__; top of its spectrum
    return obj


def test_contact_mode_meets_target_and_differs_from_device():
    obj = _contact_shell(); obj.FockToP()
    assert obj.dN_inclusive is not None and obj.dN_inclusive < 1e-3
    assert abs(obj.fermi - (-1.002)) < 0.02
    assert abs(obj.fermi - 0.25) > 0.5     # device-neutral Ef is near +0.25 eV


def test_frozen_guard_round_trip_no_thrash(capsys):
    obj = _contact_shell()
    obj.FockToP(); capsys.readouterr()
    obj.FockToP(); assert 'Fermi frozen' in capsys.readouterr().out
    obj.FockToP(); assert 'Fermi frozen' in capsys.readouterr().out


def test_predict_raises_in_contact_mode():
    with pytest.raises(ValueError):
        _contact_shell(method='predict').FockToP()


def test_every_count_honours_inds_and_cross(monkeypatch):
    calls = []
    real = dens_mod.count_electrons
    def spy(P, S, dN, inds=None, cross=True):
        calls.append((None if inds is None else tuple(np.asarray(inds)), cross))
        return real(P, S, dN, inds, cross)
    monkeypatch.setattr(dens_mod, 'count_electrons', spy)
    monkeypatch.setattr(scfE_mod, 'count_electrons', spy)
    obj = _contact_shell(); obj.fermiCross = False
    obj.FockToP()                                          # search cycle
    obj.updFermi = False
    obj.mu1, obj.mu2, obj.qV = obj.fermi + 0.1, obj.fermi - 0.1, 0.2
    obj.FockToP()                                          # pinned-bias cycle: nLower, _eqN, windowN
    assert len(calls) >= 5 and all(c == ((0, 3), False) for c in calls), calls


def test_T300_meets_target():
    obj = _contact_shell(); obj.T = 300.0
    obj.FockToP()
    assert obj.dN_inclusive < 1e-3


def test_bisect_in_contact_mode_converges():
    # unbracketed bisect walks out from its start; -1.0 converges in 4 cycles (from -0.9 it needs 11 > 10)
    obj = _contact_shell(method='bisect', start=-1.0); obj.FockToP()
    assert abs(obj.fermi - (-1.002)) < 0.02


def test_failed_search_keeps_start_ef():
    obj = _contact_shell(method='bisect', target_total=2.4, inds=(3,), start=0.5)   # never reachable
    obj.FockToP()
    assert obj.fermi == 0.5


def test_secant_runaway_is_not_kept():
    obj = _contact_shell(method='secant', start=0.5)
    obj.FockToP()
    assert np.isfinite(obj.fermi) and obj.Emin <= obj.fermi <= obj.Emax


def test_device_shell_without_attributes_unchanged():
    obj = _build_focktop_shell(); obj.FockToP()
    assert getattr(obj, 'fermiCountInds', None) is None


def test_integralCheck_routes_contact_target(monkeypatch):
    obj = _contact_shell()
    seen = {}
    monkeypatch.setattr(scfE_mod, 'integralFit', lambda *a, **k: (-30.0, None, None))
    monkeypatch.setattr(scfE_mod, 'densityRealN', lambda *a, **k: (np.zeros((4, 4)), 0.0))
    def secant(g, ne, Emin, Ef, N, **kw):
        seen.update(ne=ne, inds=kw.get('inds'), cross=kw.get('cross'))
        return Ef, 0.0, np.zeros((4, 4)), 0.0
    monkeypatch.setattr(scfE_mod, 'calcFermiSecant', secant)
    monkeypatch.setattr(obj, 'setVoltage', lambda *a, **k: None)
    obj.integralCheck(cycles=0)
    assert seen['ne'] == 0.5 and list(seen['inds']) == [0, 3] and seen['cross'] is True    # 'r': total 1.0 halved
    obj.spin = 'g'
    obj.integralCheck(cycles=0)
    assert seen['ne'] == 1.0                                                                 # 'g': not halved
