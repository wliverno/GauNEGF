"""Unit tests for find_lower_bound (integration lower bound via inertia step-down).

find_lower_bound returns an Eminf FLOOR below the lowest pole (n_neg == N there),
not the pole itself. Energy-INDEPENDENT mocks first (analytic), then the
pure-numpy surfG1D toy. No Gaussian, no GPU -- runs on a login node.
"""
import numpy as np
from scipy.linalg import eigh as eigh_gen

from gauNEGF.surfG1D import surfG
from gauNEGF.density import find_lower_bound


class ConstSigmaG:
    """Mock surfG with an energy-INDEPENDENT (constant) self-energy."""

    def __init__(self, F, S, Sig):
        self.F = np.asarray(F)
        self.S = np.asarray(S)
        self._Sig = np.asarray(Sig)
        self.eta = 1e-6

    def sigmaTot(self, E):
        return self._Sig

    def crossTermQTot(self, E):
        return None

    def setF(self, F, mu1, mu2):
        pass


class LinearSigmaG:
    """Mock with an E-LINEAR self-energy Sigma(E) = E * X_asymp, chosen so the
    effective overlap S - X_asymp is NON-PSD -- the divergent-density regime where
    the inertia count never reaches n_neg == N and find_lower_bound returns None."""

    def __init__(self, F, S, X_asymp):
        self.F = np.asarray(F)
        self.S = np.asarray(S)
        self._X = np.asarray(X_asymp)
        self.eta = 1e-6

    def sigmaTot(self, E):
        return float(E) * self._X

    def crossTermQTot(self, E):
        return None

    def setF(self, F, mu1, mu2):
        pass


def _nneg(F, S, g, E):
    M = E * np.asarray(S) - np.asarray(F) - np.asarray(g.sigmaTot(E))
    Mh = 0.5 * (M + M.conj().T)
    return int(np.sum(np.real(np.linalg.eigvalsh(Mh)) < 0.0))


def test_find_lower_bound_below_pole_constant_sigma():
    # Real constant self-energy: the lowest pole is the lowest generalized
    # eigenvalue of (F + Sig, S) = -12.5. The lower bound must sit BELOW it with
    # all N eigenvalues of M negative (n_neg == N).
    F = np.diag([-12.0, -3.0]).astype(complex)
    S = np.eye(2, dtype=complex)
    Sig = np.diag([-0.5, -0.5]).astype(complex)
    g = ConstSigmaG(F, S, Sig)
    lowest_pole = float(np.sort(eigh_gen((F + Sig).real, S.real, eigvals_only=True))[0])

    lb = find_lower_bound(F, S, g)
    assert lb is not None
    assert lb < lowest_pole, (lb, lowest_pole)
    assert _nneg(F, S, g, lb) == F.shape[0]


def test_find_lower_bound_nonpsd_overlap_uses_spectral_weight():
    # E-linear Sigma with X_asymp = diag(2, 0): one direction of S - X_asymp is
    # negative -- a finite-overlap artifact, NOT a pole. The total pole count is
    # n_neg at the deep cutoff (< N), so find_lower_bound returns a real floor
    # (not None) by referencing that count rather than the matrix dimension.
    F = np.diag([-5.0, -5.0]).astype(complex)
    S = np.eye(2, dtype=complex)
    X_asymp = np.diag([2.0, 0.0]).astype(complex)
    g = LinearSigmaG(F, S, X_asymp)
    n_ref = _nneg(F, S, g, -1.0e6)
    assert n_ref < F.shape[0]            # finite overlap reduces the count below N
    lb = find_lower_bound(F, S, g)
    assert lb is not None
    assert _nneg(F, S, g, lb) == n_ref   # floor sits below all n_ref poles


def _build_lead_toy():
    # 2-site device (eigenvalues a_d +/- b_d = {-30, 0}) + two 1D leads, band
    # ~[-10, -5]. staus=None -> energy-INdependent hop, but Sigma(E)=t g_s(E) t is
    # genuinely energy-dependent. The deep -30 device state is the lowest pole.
    a_d, b_d = -15.0, -15.0
    F = np.array([[a_d, b_d], [b_d, a_d]], dtype=complex)
    S = np.eye(2, dtype=complex)
    inds = [[0], [1]]
    taus = [np.array([[-0.5]]), np.array([[-0.5]])]
    staus = [None, None]
    alphas = [np.array([[-8.0]]), np.array([[-8.0]])]
    aOv = [np.array([[1.0]]), np.array([[1.0]])]
    betas = [np.array([[-2.0]]), np.array([[-2.0]])]
    bOv = [np.array([[0.1]]), np.array([[0.1]])]
    g = surfG(F, S, inds, taus, staus, alphas, aOv, betas, bOv, spin='r')
    return F, S, g


def test_find_lower_bound_nneg_at_floor_toy():
    # Genuinely energy-dependent Sigma(E) (real 1D lead). The lower bound must be
    # below the deep -30 device state with n_neg == N.
    F, S, g = _build_lead_toy()
    lb = find_lower_bound(F, S, g)
    assert lb is not None
    assert lb < -30.0, lb
    assert _nneg(F, S, g, lb) == F.shape[0]
