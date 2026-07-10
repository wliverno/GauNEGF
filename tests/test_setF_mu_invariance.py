"""Test how surfG.setF(F, muL, muR) mu changes propagate to cached Sigma_0 / Y_eff.

Background (UPDATED 2026-07-08, shift-algebra regression fix): surfG1D
applies the rigid-band Fermi shift to the surface Green's function AND to
the E-dependent coupling -- the exact shifted-lead identity. surfG1D.sigma():

    E_shifted = E - dFermiList[i]
    t = (-tau) if stau is None else (E_shifted*stau - tau)   # shifted
    sig = t_reg @ g_surf(E_shifted) @ bar_t_reg              # shifted

so sigma_new(E) = sigma_old(E - dF) exactly (every S-weighted block of the
lead shifts together, matching surfGB.updateH). For the asymptotically
E-linear tail sigma(E) ~ Sigma_0 + X*E this gives:

    sigma_new(E) ~ Sigma_0 + X*(E - dF) = (Sigma_0 - X*dF) + X*E

    X_asymp invariant under mu shift.
    S_eff   invariant (only depends on X_asymp).
    Y_eff   invariant.
    Sigma_0 shifts by -sum_i X_i * dFermi_i  (NEGATIVE sign).

If both contacts shift by the same dFermi:
    delta Sigma_0 = -X_asymp_total * dFermi

(The pre-fix code shifted only g_surf, leaving the coupling at raw E; that
inconsistent algebra produced a +X*dF shift, which these tests originally
encoded. The sign flip here is the regression test FOR the fix.)

This test runs setF -> _initAsymptoticSigma -> shift mu via setF ->
_initAsymptoticSigma again, and checks the predicted invariance/shift.
The test passing means the cached Sigma_0 in NEGFE goes STALE after any
voltage change unless _initAsymptoticSigma is re-invoked. The fix is to
add self._initAsymptoticSigma() at the end of NEGFE.setVoltage().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from gauNEGF.surfG1D import surfG
from gauNEGF.scfE import NEGFE
from gauNEGF.utils import fractional_matrix_power


def _build_minimal_negfe_with_surfg(non_orth_coupling=True):
    """Build NEGFE-like shell wrapping a hand-crafted surfG (no DFT).

    A 4-site 1D-chain device with single-orbital contacts on each end. To get
    asymptotically LINEAR Sigma(E) (Sigma_0 + X*E + O(1/E)) the device-contact
    coupling stau and the contact-contact overlap bOverlaps must both be
    nonzero. With staus=None (orthogonal device-contact coupling), t in
    surfG.sigma() collapses to -tau (E-independent), giving Sigma ~ 1/E with
    no linear term -- the Damle framework does not apply. With non_orth=True
    we pass nonzero staus/bOverlaps so X_asymp != 0.
    """
    N = 4
    F = np.diag([-1.0, 0.0, 1.0, 2.0]).astype(float)
    S = np.eye(N)
    inds = [np.array([0]), np.array([N - 1])]
    # 1x1 contact alphas and betas
    alphas = [np.array([[0.0]]), np.array([[0.0]])]
    betas = [np.array([[-0.5]]), np.array([[-0.5]])]
    aOverlaps = [np.array([[1.0]]), np.array([[1.0]])]
    # tauList in matrix form so contactFromFock=False (exercises the rigid-band shift)
    taus = [np.array([[-0.3]]), np.array([[-0.3]])]  # 1x1 coupling device->contact
    if non_orth_coupling:
        # Non-orthogonal device-contact AND contact-contact overlap so
        # sigma(E) ~ Sigma_0 + X*E asymptotically with X != 0.
        staus = [np.array([[0.05]]), np.array([[0.05]])]
        bOverlaps = [np.array([[0.05]]), np.array([[0.05]])]
    else:
        # Orthogonal: Sigma will decay as 1/E -> X_asymp ~ 0.
        staus = [None, None]
        bOverlaps = [np.array([[0.0]]), np.array([[0.0]])]
    g = surfG(F, S, inds, taus=taus, staus=staus,
              alphas=alphas, aOverlaps=aOverlaps,
              betas=betas, bOverlaps=bOverlaps,
              eta=1e-5, spin='r')
    assert g.contactFromFock is False, 'Test requires contactFromFock=False path'
    # Capture fermi0 with initial mu=0.0
    g.setF(F, mu1=0.0, mu2=0.0)
    # Build NEGFE shell
    obj = NEGFE.__new__(NEGFE)
    obj.F = F
    obj.S = S
    obj.g = g
    # _initAsymptoticSigma uses self.X (Lowdin orthogonalizer S^(-1/2)) for
    # its band-floor estimate; supply it directly since we bypassed __init__.
    obj.X = np.asarray(fractional_matrix_power(np.asarray(S), -0.5))
    return obj


def test_x_asymp_invariant_under_setF_mu_change():
    """X_asymp must NOT change when mu is shifted via setF."""
    obj = _build_minimal_negfe_with_surfg(non_orth_coupling=True)
    obj._initAsymptoticSigma()
    X0 = obj.X_asymp.copy()

    # Sanity: X_asymp is nontrivially nonzero, else the test is vacuous.
    assert np.linalg.norm(X0) > 1e-6, (
        f'Test fixture produced X_asymp ~ 0 ({np.linalg.norm(X0):.3e}); cannot '
        'distinguish invariance from triviality. Increase bOverlaps magnitude.'
    )

    dFermi = 0.5
    obj.g.setF(obj.F, mu1=dFermi, mu2=dFermi)
    obj._initAsymptoticSigma()
    X1 = obj.X_asymp.copy()

    rel_change = np.linalg.norm(X1 - X0) / max(np.linalg.norm(X0), 1e-30)
    assert rel_change < 1e-4, (
        f'X_asymp shifted by relative ||delta||/||X|| = {rel_change:.3e} '
        f'after setF mu change (dFermi={dFermi}). Expected invariance.'
    )


def test_y_eff_invariant_under_setF_mu_change():
    """Y_eff = S_eff^(-1/2) must NOT change when mu is shifted via setF."""
    obj = _build_minimal_negfe_with_surfg(non_orth_coupling=True)
    obj._initAsymptoticSigma()
    Y0 = obj.Y_eff.copy()
    Seff0 = obj.S_eff.copy()

    dFermi = 0.5
    obj.g.setF(obj.F, mu1=dFermi, mu2=dFermi)
    obj._initAsymptoticSigma()
    Y1 = obj.Y_eff.copy()
    Seff1 = obj.S_eff.copy()

    rel_seff = np.linalg.norm(Seff1 - Seff0) / max(np.linalg.norm(Seff0), 1e-30)
    rel_yeff = np.linalg.norm(Y1 - Y0) / max(np.linalg.norm(Y0), 1e-30)
    assert rel_seff < 1e-4, f'S_eff shifted: rel ||delta|| = {rel_seff:.3e}'
    assert rel_yeff < 1e-4, f'Y_eff shifted: rel ||delta|| = {rel_yeff:.3e}'


def test_sigma_0_shifts_by_predicted_amount():
    """Sigma_0 must shift by -X_asymp_total * dFermi when both mu's shift equally.

    Derivation (see module docstring): the exact shifted-lead algebra gives
    sigma_new(E) = sigma_old(E - dF), so the E-linear tail's intercept moves
    by -X * dF at leading order.

    If THIS test passes, it confirms cached self.Sigma_0 in NEGFE goes stale
    after any voltage change. The fix is to re-call self._initAsymptoticSigma()
    after self.g.setF in NEGFE.setVoltage.
    """
    obj = _build_minimal_negfe_with_surfg(non_orth_coupling=True)
    obj._initAsymptoticSigma()
    Sigma_0_before = obj.Sigma_0.copy()
    X_asymp = obj.X_asymp.copy()

    dFermi = 0.5
    obj.g.setF(obj.F, mu1=dFermi, mu2=dFermi)
    obj._initAsymptoticSigma()
    Sigma_0_after = obj.Sigma_0.copy()

    actual_shift = Sigma_0_after - Sigma_0_before
    predicted_shift = -X_asymp * dFermi
    rel_err = (np.linalg.norm(actual_shift - predicted_shift) /
               max(np.linalg.norm(predicted_shift), 1e-30))

    print(f'  ||Sigma_0 before||  = {np.linalg.norm(Sigma_0_before):.3e}')
    print(f'  ||Sigma_0 after||   = {np.linalg.norm(Sigma_0_after):.3e}')
    print(f'  ||actual shift||    = {np.linalg.norm(actual_shift):.3e}')
    print(f'  ||predicted shift|| = {np.linalg.norm(predicted_shift):.3e}')
    print(f'  rel err             = {rel_err:.3e}')

    # The 1/E sub-leading terms in surfG are O(1/E_deep) ~ O(1e-5) at our
    # deepest probe; relative error to the leading-order prediction should be
    # below a few percent.
    assert rel_err < 5e-2, (
        f'Sigma_0 shift does not match rigid-band prediction: rel err = {rel_err:.3e}'
    )


def test_asymmetric_bias_sigma_0_shift():
    """Per-contact rigid shifts add correctly under asymmetric bias dFermi_L != dFermi_R.

    Total delta Sigma_0 is element-wise (exact shifted-lead algebra):
        delta Sigma_0[L block] = -X_L * dFermi_L
        delta Sigma_0[R block] = -X_R * dFermi_R
        delta Sigma_0 elsewhere = 0

    For a contact that only touches device orbitals in indsList[i], the
    asymptotic X_i is nonzero only on the [inds_i, inds_i] block, so X_total
    is block-diagonal (no L-R cross terms). Reconstruct the predicted shift
    by slicing X_total per-contact and scaling by the corresponding dFermi.
    """
    obj = _build_minimal_negfe_with_surfg(non_orth_coupling=True)
    obj._initAsymptoticSigma()
    Sigma_0_before = obj.Sigma_0.copy()
    X_total = obj.X_asymp.copy()

    dFermi_L = 0.3
    dFermi_R = -0.2
    obj.g.setF(obj.F, mu1=dFermi_L, mu2=dFermi_R)
    obj._initAsymptoticSigma()
    Sigma_0_after = obj.Sigma_0.copy()

    actual = Sigma_0_after - Sigma_0_before
    # Per-contact predicted shift (slice X_total on the contact block).
    lInd = np.asarray(obj.g.indsList[0])
    rInd = np.asarray(obj.g.indsList[-1])
    predicted = np.zeros_like(X_total)
    predicted[np.ix_(lInd, lInd)] = -X_total[np.ix_(lInd, lInd)] * dFermi_L
    predicted[np.ix_(rInd, rInd)] = -X_total[np.ix_(rInd, rInd)] * dFermi_R
    rel_err = (np.linalg.norm(actual - predicted) /
               max(np.linalg.norm(predicted), 1e-30))
    print(f'  ||actual||    = {np.linalg.norm(actual):.3e}')
    print(f'  ||predicted|| = {np.linalg.norm(predicted):.3e}')
    print(f'  rel err       = {rel_err:.3e}')
    assert rel_err < 5e-2, f'Asymmetric bias prediction off: rel_err = {rel_err:.3e}'


def test_setVoltage_refreshes_cached_sigma_0():
    """NEGFE.setVoltage must refresh cached self.Sigma_0 to reflect new mu1/mu2.

    Failure mode under test: setVoltage calls self.g.setF(F, mu1, mu2), which
    updates dFermiList in surfG so subsequent sigmaTot(E) probes return shifted
    values. But cached self.Sigma_0 from earlier _initAsymptoticSigma is stale
    -- still represents the OLD mu. damleLowerDensity would then integrate
    with a Sigma_0 that doesn't match the current contact state.

    Drives the simple shift via qV=0, fermi=0.5 (uniform shift, both contacts
    move to mu=0.5). For our symmetric fixture this gives the largest signal:
    delta Sigma_0_total = -X_total * 0.5.
    """
    from unittest.mock import MagicMock

    obj = _build_minimal_negfe_with_surfg(non_orth_coupling=True)
    obj._initAsymptoticSigma()
    Sigma_0_initial = obj.Sigma_0.copy()
    X_asymp = obj.X_asymp.copy()

    # State for setVoltage / NEGF.setVoltage to run
    obj.fermi = 0.0
    obj.lContact = [1]
    obj.rContact = [4]
    obj.lInd = np.array([0])
    obj.rInd = np.array([3])
    obj.updFermi = False
    obj.N1 = None
    obj.N2 = None
    # Mock bar with the minimum interface used by NEGF.setVoltage's field calc:
    # bar.c is a flat array of atomic coordinates, bar.scalar() sets fields.
    obj.bar = MagicMock()
    obj.bar.c = np.array([0., 0., 0., 1., 0., 0., 2., 0., 0., 3., 0., 0.])

    # Uniform shift: qV=0, fermi=0.5 -> mu1 = mu2 = 0.5 (dFermi = 0.5 for both)
    obj.setVoltage(qV=0.0, fermi=0.5)

    actual_shift = obj.Sigma_0 - Sigma_0_initial
    predicted_shift = -X_asymp * 0.5
    rel_err = (np.linalg.norm(actual_shift - predicted_shift) /
               max(np.linalg.norm(predicted_shift), 1e-30))
    print(f'  ||actual shift||    = {np.linalg.norm(actual_shift):.3e}')
    print(f'  ||predicted shift|| = {np.linalg.norm(predicted_shift):.3e}')
    print(f'  rel err             = {rel_err:.3e}')
    assert rel_err < 5e-2, (
        f'setVoltage did not refresh Sigma_0 (rel err = {rel_err:.3e}). '
        'Cached Sigma_0 is stale w.r.t. the new mu.'
    )


def test_orthogonal_contact_gives_zero_x_asymp():
    """Sanity: with bOverlaps=0 (orthogonal contact), X_asymp ~ 0 and Sigma_0 ~ invariant."""
    obj = _build_minimal_negfe_with_surfg(non_orth_coupling=False)
    obj._initAsymptoticSigma()
    X_asymp = obj.X_asymp.copy()
    Sigma_0_before = obj.Sigma_0.copy()

    # X_asymp should be essentially zero for orthogonal contacts
    assert np.linalg.norm(X_asymp) < 1e-3, (
        f'Orthogonal contact gave nonzero X_asymp: ||X|| = {np.linalg.norm(X_asymp):.3e}'
    )

    dFermi = 0.5
    obj.g.setF(obj.F, mu1=dFermi, mu2=dFermi)
    obj._initAsymptoticSigma()
    Sigma_0_after = obj.Sigma_0.copy()

    # With X_asymp ~ 0, predicted shift is ~ 0, so Sigma_0 should be ~ invariant.
    rel_shift = (np.linalg.norm(Sigma_0_after - Sigma_0_before) /
                 max(np.linalg.norm(Sigma_0_before), 1e-30))
    assert rel_shift < 1e-3, (
        f'Sigma_0 shifted for orthogonal contact: rel shift = {rel_shift:.3e}'
    )
