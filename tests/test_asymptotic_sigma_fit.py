"""Unit tests for NEGFE._initAsymptoticSigma (asymptotic Sigma fit setup)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from gauNEGF.utils import fractional_matrix_power


class _LinearSigmaG:
    """Mimics a surfG with sigmaTot(E) = Sigma_0_true + E * X_true exactly."""
    def __init__(self, F, S, X_true, Sigma_0_true):
        self.F = np.asarray(F, dtype=complex)
        self.S = np.asarray(S, dtype=complex)
        self._X = np.asarray(X_true, dtype=complex)
        self._Sigma_0 = np.asarray(Sigma_0_true, dtype=complex)

    def sigmaTot(self, E, conv=None):
        return E * self._X + self._Sigma_0


def _make_minimal_negfe(F, S, g):
    """Build a NEGFE-like shell with only the attributes _initAsymptoticSigma needs."""
    from gauNEGF.scfE import NEGFE
    # Construct as a bare object; bypass __init__ which requires Gaussian files.
    # _initAsymptoticSigma reads self.S, self.g.sigmaTot, and self.X (Lowdin
    # orthogonalizer S^(-1/2) -- used for the cheap band-floor estimate that
    # places the deep probes).
    obj = NEGFE.__new__(NEGFE)
    obj.F = np.asarray(F)
    obj.S = np.asarray(S)
    obj.g = g
    obj.X = np.asarray(fractional_matrix_power(np.asarray(S), -0.5))
    return obj


def test_exact_linear_recovers_sigma_0_and_x():
    """When Sigma(E) is exactly linear, _initAsymptoticSigma must recover X and Sigma_0."""
    N = 3
    F = np.diag([-1.0, 0.0, 1.0])  # eV; not actually used by _initAsymptoticSigma
    S = np.eye(N)
    X_true = 0.1 * np.eye(N)
    Sigma_0_true = -0.05j * np.eye(N)
    g = _LinearSigmaG(F, S, X_true, Sigma_0_true)
    obj = _make_minimal_negfe(F, S, g)
    obj._initAsymptoticSigma()
    assert np.allclose(obj.X_asymp, X_true, atol=1e-8)
    assert np.allclose(obj.Sigma_0, Sigma_0_true, atol=1e-8)


def test_s_eff_constructed_correctly():
    """S_eff = S - X_asymp (with symmetrized real part of X)."""
    N = 3
    F = np.zeros((N, N))
    S = np.eye(N) * 2.0
    X_true = 0.3 * np.eye(N)
    Sigma_0_true = np.zeros((N, N), dtype=complex)
    g = _LinearSigmaG(F, S, X_true, Sigma_0_true)
    obj = _make_minimal_negfe(F, S, g)
    obj._initAsymptoticSigma()
    expected_S_eff = np.real(S) - 0.5 * (np.real(X_true) + np.real(X_true).T)
    assert np.allclose(obj.S_eff, expected_S_eff, atol=1e-10)


def test_y_eff_is_inverse_sqrt_of_s_eff_psd():
    """For PSD S_eff, Y_eff @ S_eff @ Y_eff == I."""
    N = 3
    F = np.zeros((N, N))
    S = np.eye(N)
    X_true = 0.1 * np.eye(N)  # S_eff = 0.9 I -> PSD
    Sigma_0_true = np.zeros((N, N), dtype=complex)
    g = _LinearSigmaG(F, S, X_true, Sigma_0_true)
    obj = _make_minimal_negfe(F, S, g)
    obj._initAsymptoticSigma()
    Y = np.asarray(obj.Y_eff)
    Seff = np.asarray(obj.S_eff)
    assert np.allclose(Y @ Seff @ Y, np.eye(N), atol=1e-8)


def test_y_eff_complex_when_s_eff_indefinite():
    """When X_asymp pushes S_eff indefinite, Y_eff must be complex."""
    N = 3
    F = np.zeros((N, N))
    S = np.eye(N)
    X_true = np.diag([0.1, 0.1, 1.5])  # S_eff = diag(0.9, 0.9, -0.5) -> indefinite
    Sigma_0_true = np.zeros((N, N), dtype=complex)
    g = _LinearSigmaG(F, S, X_true, Sigma_0_true)
    obj = _make_minimal_negfe(F, S, g)
    obj._initAsymptoticSigma()
    Y = np.asarray(obj.Y_eff)
    assert not np.allclose(Y.imag, 0.0, atol=1e-10), 'Indefinite S_eff -> complex Y_eff expected'


