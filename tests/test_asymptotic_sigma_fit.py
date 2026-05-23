"""Unit tests for NEGFE._initAsymptoticSigma (asymptotic Sigma fit setup)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


class _LinearSigmaG:
    """Mimics a surfG with sigmaTot(E) = Sigma_0_true + E * X_true exactly."""
    def __init__(self, F, S, X_true, Sigma_0_true):
        self.F = np.asarray(F, dtype=complex)
        self.S = np.asarray(S, dtype=complex)
        self._X = np.asarray(X_true, dtype=complex)
        self._Sigma_0 = np.asarray(Sigma_0_true, dtype=complex)

    def sigmaTot(self, E, conv=None):
        return E * self._X + self._Sigma_0


class _NoisyLinearSigmaG(_LinearSigmaG):
    """Adds 1/E correction so the linear fit has nonzero residual."""
    def __init__(self, F, S, X_true, Sigma_0_true, Sigma_m1):
        super().__init__(F, S, X_true, Sigma_0_true)
        self._Sigma_m1 = np.asarray(Sigma_m1, dtype=complex)

    def sigmaTot(self, E, conv=None):
        return E * self._X + self._Sigma_0 + self._Sigma_m1 / E


def _make_minimal_negfe(F, S, g):
    """Build a NEGFE-like shell with only the attributes _initAsymptoticSigma needs."""
    from gauNEGF.scfE import NEGFE
    # Construct as a bare object; bypass __init__ which requires Gaussian files.
    # _initAsymptoticSigma only reads self.S and self.g.sigmaTot (not self.F),
    # so F is stored only because other NEGFE methods might touch it later.
    obj = NEGFE.__new__(NEGFE)
    obj.F = np.asarray(F)
    obj.S = np.asarray(S)
    obj.g = g
    obj.damle_buffer = 20.0
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


def test_residual_warning_on_nonlinear_sigma(capsys):
    """When Sigma has a 1/E term that exceeds the linearity tolerance, warn.

    Sizing: at E_deep = -1e5, the 1/E perturbation contributes Sigma_m1/E_deep.
    For this to dominate the linear-fit residual at the >1% level vs the
    leading X*E_deep term (= -1e4), we need |Sigma_m1/E_deep| / 1e4 > 0.01,
    i.e. |Sigma_m1| > 1e7. Use 1e8 to be comfortably above threshold.
    """
    N = 3
    F = np.zeros((N, N))
    S = np.eye(N)
    X_true = 0.1 * np.eye(N)
    Sigma_0_true = -0.01j * np.eye(N)
    Sigma_m1 = 1e8 * np.eye(N, dtype=complex)
    g = _NoisyLinearSigmaG(F, S, X_true, Sigma_0_true, Sigma_m1)
    obj = _make_minimal_negfe(F, S, g)
    obj._initAsymptoticSigma()
    captured = capsys.readouterr()
    assert 'WARNING' in captured.out or 'warning' in captured.out, (
        f'Expected linearity warning in output, got: {captured.out!r}'
    )
