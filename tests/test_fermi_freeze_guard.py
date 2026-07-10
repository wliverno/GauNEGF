"""FockToP freeze guard: skip the per-cycle Fermi search when the
previous cycle's CROSS-TERM-INCLUSIVE electron-count mismatch
(self.dN_inclusive) is already below conv; re-engage when it is not.

Login-safe (no Gaussian): extends the MagicMock NEGFE shell pattern from
test_setF_mu_invariance.py per the 2026-07-09 review recipe. The shell
wires exactly the attributes FockToP's updFermi block consumes and runs
the real FockToP end-to-end on a 4-site synthetic device.

Also asserts the review-caught invariant directly: the gate must never
read the bare-trace count (self.nelec) -- an earlier revision did and
was rejected, since updateN() omits delta_N (O(1) e on Au junctions).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock

import numpy as np
from scipy.linalg import fractional_matrix_power

from gauNEGF.scfE import NEGFE
from gauNEGF.surfG1D import surfG


def _build_focktop_shell():
    """NEGFE shell able to run FockToP()'s updFermi path without DFT."""
    N = 4
    F = np.diag([-1.0, 0.0, 1.0, 2.0]).astype(float)
    S = np.eye(N)
    inds = [np.array([0]), np.array([N - 1])]
    alphas = [np.array([[0.0]]), np.array([[0.0]])]
    betas = [np.array([[-0.5]]), np.array([[-0.5]])]
    aOverlaps = [np.array([[1.0]]), np.array([[1.0]])]
    taus = [np.array([[-0.3]]), np.array([[-0.3]])]
    staus = [np.array([[0.05]]), np.array([[0.05]])]
    bOverlaps = [np.array([[0.05]]), np.array([[0.05]])]
    g = surfG(F, S, inds, taus=taus, staus=staus,
              alphas=alphas, aOverlaps=aOverlaps,
              betas=betas, bOverlaps=bOverlaps,
              eta=1e-5, spin='r')
    g.setF(F, mu1=0.0, mu2=0.0)

    obj = NEGFE.__new__(NEGFE)
    obj.F = F / 27.211386        # FockToP multiplies by har_to_eV
    obj.S = S
    obj.g = g
    obj.X = np.asarray(fractional_matrix_power(np.asarray(S), -0.5))
    obj.spin = 'r'
    obj.fermi = 0.5              # mid-gap of the synthetic spectrum
    obj.mu1 = 0.5
    obj.mu2 = 0.5
    obj.qV = 0.0
    obj.updFermi = True
    obj.fermiMethod = 'muller'
    obj.convLevel = 9999
    obj.nelec = 0.0              # bare trace; the gate must NOT read it
    obj.N1 = None
    obj.N2 = None
    obj.Nnegf = None
    obj.tol = 1e-4
    obj.T = 0.0
    obj.Emin = -30.0
    obj.Eminf = -1e6
    obj.damle_dN_warn = 0.5
    obj.lContact = [1]
    obj.rContact = [4]
    obj.lInd = np.array([0])
    obj.rInd = np.array([3])
    obj.bar = MagicMock()
    obj.bar.ne = 4.0             # 2 filled orbitals per spin
    obj.bar.c = np.array([0., 0., 0., 1., 0., 0., 2., 0., 0., 3., 0., 0.])
    obj._initAsymptoticSigma()
    return obj


def test_first_cycle_always_searches(capsys):
    obj = _build_focktop_shell()
    assert getattr(obj, 'dN_inclusive', None) is None
    obj.FockToP()
    out = capsys.readouterr().out
    assert 'Fermi frozen' not in out
    assert 'METHOD' in out, 'a search method should have run'
    # the cycle must leave an inclusive mismatch stash (or None after
    # a bisect fallback) for the next cycle's gate
    assert hasattr(obj, 'dN_inclusive')


def test_primed_below_conv_freezes_and_keeps_fermi(capsys):
    obj = _build_focktop_shell()
    obj.dN_inclusive = 1e-9
    fermi_before = obj.fermi
    obj.FockToP()
    out = capsys.readouterr().out
    assert 'Fermi frozen' in out
    assert 'METHOD' not in out, 'no search may run while frozen'
    assert obj.fermi == fermi_before, 'frozen cycle must not move fermi'
    # the frozen path re-audits its own equilibrium piece
    assert obj.dN_inclusive is not None


def test_primed_above_conv_searches(capsys):
    obj = _build_focktop_shell()
    obj.dN_inclusive = 10.0
    obj.FockToP()
    out = capsys.readouterr().out
    assert 'Fermi frozen' not in out
    assert 'METHOD' in out


def test_gate_ignores_bare_trace(capsys):
    """Regression for the review-caught bug: a wildly wrong bare-trace
    nelec must not affect the gate when the inclusive mismatch is
    primed below conv."""
    obj = _build_focktop_shell()
    obj.dN_inclusive = 1e-9
    obj.nelec = 1e6              # absurd bare trace
    obj.FockToP()
    out = capsys.readouterr().out
    assert 'Fermi frozen' in out, 'gate consulted the bare trace'
