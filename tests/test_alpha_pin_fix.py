"""Regression tests for the two _symmetrize_F bugs planted by commit
826ca2a (both found 2026-07-17).

Bug 1 (alpha pin): _symmetrize_F used to write self.g.aList[0]/[-1]
back into self.F's contact on-site blocks whenever contactFromFock is
True, BEFORE self.g.setF() re-derives aList from the (now-corrupted) F.
Since g.setF()'s _setContacts() extracts aList straight from the F it is
given, this created a one-cycle-stale fixed point: aList never advanced
past whatever it was at the first setF() call, discarding every fresh
Gaussian-computed contact block PToFock produces (verified dynamically
at up to 250 eV/cycle on a real Au nanowire run, see
paper/studies/06_1d_aunanowire/verify_alpha_pin.py/.out in the main
tree). The fix deletes the two-line injection; the L/R
_symmetrize_contacts average now acts on the fresh block that getFock()
just wrote into self.F.

Bug 2 (default flip): the same commit flipped the averaging gate's
fallback from getattr(self, '_symmetrize_contacts', False) -- the
f51a5f9 design, symmetrization off unless requested -- to default True.
Only setContact1D ever SETS the attribute; setContactBethe and setSigma
never do, so every Bethe-lattice and constant-sigma NEGFE SCF run
silently L/R-averaged its contact Fock blocks every cycle (banked Bethe
.mats show max|Fl-Fr| = 0.0 exactly; see
paper/studies/reports/2026-07-17-bethe-averaging-bug.md in the main
tree). The fix restores the False fallback; setContact1D still sets the
flag explicitly, so 1D/contactFromFock behavior is unchanged.

Login-safe (no Gaussian): builds surfG/NEGFE shells directly, following
the MagicMock-shell pattern in test_fermi_freeze_guard.py -- no bar/DFT
object is touched, only _symmetrize_F is exercised.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from gauNEGF.scfE import NEGFE
from gauNEGF.surfG1D import surfG

har_to_eV = 27.211386


def _build_contactFromFock_true_shell():
    """NEGFE shell wrapping a contactFromFock=True (alphas=None) surfG.

    6-site chain, 2-orbital contacts at each end (fully-automatic
    extraction pattern, same as test_surfG1D_features.test_xi_attribute_
    exists_after_init). aList[0]/aList[-1] are captured once at surfG
    construction time from F_eV's initial contact blocks.
    """
    F_eV = np.diag([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    S = np.eye(6)
    inds = [np.array([0, 1]), np.array([4, 5])]
    g = surfG(F_eV, S, inds, eta=1e-5, spin='r')
    assert g.contactFromFock is True, 'fixture requires contactFromFock=True'

    obj = NEGFE.__new__(NEGFE)
    obj.F = F_eV.copy() / har_to_eV   # NEGFE.F is stored in Hartree
    obj.S = S
    obj.g = g
    obj.lInd = inds[0]
    obj.rInd = inds[-1]
    obj._symmetrize_contacts = True
    return obj, g


def test_symmetrize_F_uses_fresh_blocks_not_stale_aList():
    """After _symmetrize_F, F's contact blocks must equal the L/R average
    of the FRESH blocks just written into F -- not the stale g.aList
    captured at construction. This test FAILS on the pre-fix code (the
    injection overwrites the fresh X_L/X_R with aList before averaging)
    and PASSES post-fix.
    """
    obj, g = _build_contactFromFock_true_shell()

    # Fresh "this-cycle Gaussian" contact blocks: symmetric, distinct from
    # each other AND from the aList captured at g's construction.
    X_L_eV = np.array([[5.0, 0.3], [0.3, 5.2]])
    X_R_eV = np.array([[9.0, -0.1], [-0.1, 9.4]])
    obj.F[np.ix_(obj.lInd, obj.lInd)] = X_L_eV / har_to_eV
    obj.F[np.ix_(obj.rInd, obj.rInd)] = X_R_eV / har_to_eV

    obj._symmetrize_F()

    FL = np.asarray(obj.F[np.ix_(obj.lInd, obj.lInd)])
    FR = np.asarray(obj.F[np.ix_(obj.rInd, obj.rInd)])
    avg_expected = (X_L_eV + X_R_eV) / 2 / har_to_eV

    assert np.allclose(FL, avg_expected), (
        f'F left contact block after _symmetrize_F = {FL}, expected '
        f'L/R average of the FRESH blocks = {avg_expected}. The fresh '
        'block was discarded (alpha-pin bug present).'
    )
    assert np.allclose(FR, avg_expected), (
        f'F right contact block after _symmetrize_F = {FR}, expected '
        f'L/R average of the FRESH blocks = {avg_expected}.'
    )

    # And explicitly: must NOT equal the stale stored aList (scaled to
    # Hartree), which is what the pre-fix injection produced instead.
    aList_L_as_F = np.asarray(g.aList[0]) / har_to_eV
    aList_R_as_F = np.asarray(g.aList[-1]) / har_to_eV
    assert not np.allclose(FL, aList_L_as_F), (
        'F left contact block equals the stale g.aList -- fresh Fock '
        'block was pinned/discarded.'
    )
    assert not np.allclose(FR, aList_R_as_F), (
        'F right contact block equals the stale g.aList -- fresh Fock '
        'block was pinned/discarded.'
    )


def test_symmetrize_contacts_false_leaves_F_untouched():
    """With _symmetrize_contacts=False, _symmetrize_F must not modify F
    at all (no injection, no averaging)."""
    obj, g = _build_contactFromFock_true_shell()
    obj._symmetrize_contacts = False

    X_L_eV = np.array([[7.0, 0.0], [0.0, 7.5]])
    X_R_eV = np.array([[-2.0, 0.2], [0.2, -1.5]])
    obj.F[np.ix_(obj.lInd, obj.lInd)] = X_L_eV / har_to_eV
    obj.F[np.ix_(obj.rInd, obj.rInd)] = X_R_eV / har_to_eV
    F_before = obj.F.copy()

    obj._symmetrize_F()

    assert np.allclose(np.asarray(obj.F), np.asarray(F_before)), (
        '_symmetrize_F modified F while _symmetrize_contacts=False; '
        'F must be left exactly as the caller set it.'
    )


def _build_contactFromFock_false_shell():
    """NEGFE shell wrapping a contactFromFock=False (explicit alphas)
    surfG, matching tests/test_setF_mu_invariance.py's fixture pattern.
    """
    N = 4
    F = np.diag([-1.0, 0.0, 1.0, 2.0]).astype(float)
    S = np.eye(N)
    inds = [np.array([0]), np.array([N - 1])]
    alphas = [np.array([[0.0]]), np.array([[0.0]])]
    betas = [np.array([[-0.5]]), np.array([[-0.5]])]
    aOverlaps = [np.array([[1.0]]), np.array([[1.0]])]
    taus = [np.array([[-0.3]]), np.array([[-0.3]])]
    g = surfG(F, S, inds, taus=taus, alphas=alphas, aOverlaps=aOverlaps,
              betas=betas, eta=1e-5, spin='r')
    assert g.contactFromFock is False, 'fixture requires contactFromFock=False'

    obj = NEGFE.__new__(NEGFE)
    obj.F = F.copy() / har_to_eV
    obj.S = S
    obj.g = g
    obj.lInd = inds[0]
    obj.rInd = inds[-1]
    obj._symmetrize_contacts = True
    return obj, g


def test_contactFromFock_false_only_average_applies():
    """For contactFromFock=False, _symmetrize_F must only ever apply the
    enabled L/R average -- the aList-injection branch never fired for
    this case even pre-fix (gated on contactFromFock), so this is a
    non-regression check that removing the injection block left this
    path untouched.
    """
    obj, g = _build_contactFromFock_false_shell()
    aList_before = [np.asarray(a).copy() for a in g.aList]

    X_L_eV = np.array([[3.0]])
    X_R_eV = np.array([[-6.0]])
    obj.F[np.ix_(obj.lInd, obj.lInd)] = X_L_eV / har_to_eV
    obj.F[np.ix_(obj.rInd, obj.rInd)] = X_R_eV / har_to_eV

    obj._symmetrize_F()

    avg_expected = (X_L_eV + X_R_eV) / 2 / har_to_eV
    FL = np.asarray(obj.F[np.ix_(obj.lInd, obj.lInd)])
    FR = np.asarray(obj.F[np.ix_(obj.rInd, obj.rInd)])
    assert np.allclose(FL, avg_expected)
    assert np.allclose(FR, avg_expected)

    # g.aList must be completely untouched by _symmetrize_F in either case.
    for before, after in zip(aList_before, g.aList):
        assert np.allclose(before, np.asarray(after)), (
            '_symmetrize_F must never write to g.aList for '
            'contactFromFock=False.'
        )


class _BetheLikeG:
    """Stand-in for surfGB/surfGTest: no contactFromFock attribute at all.

    A plain class (NOT MagicMock) so getattr(g, 'contactFromFock', False)
    genuinely falls through to False on pre-fix code -- a MagicMock would
    fabricate a truthy attribute and fire the wrong branch.
    """
    pass


def _build_bethe_like_shell():
    """NEGFE shell mimicking a setContactBethe/setSigma setup: g has no
    contactFromFock attribute and -- critically -- the shell has NO
    _symmetrize_contacts attribute, exactly like real setContactBethe/
    setSigma objects (only setContact1D ever sets it)."""
    F_eV = np.diag([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    obj = NEGFE.__new__(NEGFE)
    obj.F = F_eV.copy() / har_to_eV
    obj.S = np.eye(6)
    obj.g = _BetheLikeG()
    obj.lInd = np.array([0, 1])
    obj.rInd = np.array([4, 5])
    assert not hasattr(obj, '_symmetrize_contacts'), (
        'fixture must not carry _symmetrize_contacts -- the unset-attribute '
        'fallback is exactly what is under test'
    )
    return obj


def test_bethe_like_default_leaves_F_untouched():
    """With _symmetrize_contacts UNSET (Bethe/constant-sigma setup),
    _symmetrize_F must leave F byte-identical. FAILS on pre-fix code,
    where 826ca2a's default-True fallback averaged the L/R contact
    blocks of every Bethe/constant-sigma run.
    """
    obj = _build_bethe_like_shell()

    # Distinct, asymmetric contact blocks so any averaging is detectable.
    X_L_eV = np.array([[4.0, 0.7], [0.7, 4.4]])
    X_R_eV = np.array([[-3.0, 0.1], [0.1, -2.2]])
    obj.F[np.ix_(obj.lInd, obj.lInd)] = X_L_eV / har_to_eV
    obj.F[np.ix_(obj.rInd, obj.rInd)] = X_R_eV / har_to_eV
    F_before = obj.F.copy()

    obj._symmetrize_F()

    assert np.array_equal(np.asarray(obj.F), np.asarray(F_before)), (
        '_symmetrize_F modified F although _symmetrize_contacts was never '
        'set (Bethe/constant-sigma contact). The default-True fallback '
        'averaging bug is present.'
    )


def test_explicit_true_still_averages_bethe_like():
    """Sanity for the 1D path: an EXPLICIT _symmetrize_contacts = True
    (as setContact1D sets it) must still trigger the L/R average even on
    a g without contactFromFock -- the flag alone controls averaging.
    """
    obj = _build_bethe_like_shell()
    obj._symmetrize_contacts = True

    X_L_eV = np.array([[4.0, 0.7], [0.7, 4.4]])
    X_R_eV = np.array([[-3.0, 0.1], [0.1, -2.2]])
    obj.F[np.ix_(obj.lInd, obj.lInd)] = X_L_eV / har_to_eV
    obj.F[np.ix_(obj.rInd, obj.rInd)] = X_R_eV / har_to_eV

    obj._symmetrize_F()

    avg_expected = (X_L_eV + X_R_eV) / 2 / har_to_eV
    FL = np.asarray(obj.F[np.ix_(obj.lInd, obj.lInd)])
    FR = np.asarray(obj.F[np.ix_(obj.rInd, obj.rInd)])
    assert np.allclose(FL, avg_expected), (
        'explicit _symmetrize_contacts=True no longer averages -- the '
        '1D two-cell seam averaging path regressed.'
    )
    assert np.allclose(FR, avg_expected)
