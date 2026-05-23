"""Unit tests for damleLowerDensity (Damle analytic lower-contour density)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from gauNEGF.density import damleLowerDensity
from gauNEGF.config import ENERGY_MIN


def _inv_sqrt_signed_numpy(M):
    """Reference implementation (numpy) for cross-check."""
    D, U = np.linalg.eigh(M)
    return U @ np.diag(1.0 / np.emath.sqrt(D)) @ U.T


def test_psd_seff_real_output():
    """PSD S_eff (no pp's) -> real Y_eff -> real P_lower with finite values."""
    rng = np.random.default_rng(2)
    N = 4
    A = rng.normal(size=(N, N))
    F_eV = (A + A.T)
    # PSD S_eff
    B = rng.normal(size=(N, N))
    S_eff = B @ B.T + np.eye(N)
    Y_eff = _inv_sqrt_signed_numpy(S_eff)
    Sigma_0 = np.zeros((N, N), dtype=complex)  # X=0 case
    P_lower, Emin = damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer=10.0)
    P_lower = np.asarray(P_lower)
    assert np.isfinite(P_lower).all()
    assert np.allclose(P_lower.imag, 0.0, atol=1e-8)
    assert Emin < 0.0


def test_indefinite_seff_complex_output_small_imag():
    """Indefinite S_eff -> complex Y_eff -> P_lower has small imaginary part."""
    N = 4
    F_eV = np.diag([-5.0, -3.0, 0.0, 2.0])
    # Indefinite S_eff (1 negative eigenvalue)
    S_eff = np.diag([1.0, 1.0, 1.0, -0.5])
    Y_eff = _inv_sqrt_signed_numpy(S_eff)
    Sigma_0 = -0.01j * np.eye(N)  # Hermitian-anti-Hermitian piece for nonzero Gam
    P_lower, Emin = damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer=10.0)
    P_lower = np.asarray(P_lower)
    assert np.isfinite(P_lower).all()
    # Imag part may be nonzero (Y_eff is complex) but should be small relative to real part.
    rel_imag = np.linalg.norm(P_lower.imag) / max(np.linalg.norm(P_lower.real), 1e-30)
    assert rel_imag < 0.5, f'Imag/real ratio = {rel_imag:.3e} too large'


def test_emin_below_eigvals():
    """The returned Emin must be below min(D.real) of Fbar by buffer."""
    N = 3
    F_eV = np.diag([-2.0, 0.0, 5.0])
    S_eff = np.eye(N)
    Y_eff = np.eye(N)
    Sigma_0 = np.zeros((N, N), dtype=complex)
    buffer = 7.0
    P_lower, Emin = damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer=buffer)
    # Fbar = F_eV, eigvals = [-2, 0, 5], expected Emin = -2 - 7 = -9
    assert np.isclose(Emin, -9.0, atol=1e-8)


def test_zero_x_matches_standard_damle():
    """When S_eff = S = I, damleLowerDensity matches direct density() call (modulo ETA).

    damleLowerDensity adds 1j*ETA*I to H_eff for retarded-GF regularization
    (standard NEGF convention). The reference computation here mirrors that
    so the comparison stays bit-for-bit at atol=1e-8.
    """
    from gauNEGF.density import density  # standard analytic integrator (Damle Eq. 27)
    from gauNEGF.config import ETA
    N = 4
    F_eV = np.diag([-3.0, -1.0, 1.0, 3.0])
    Y_eff = np.eye(N)
    Sigma_0 = 1e-3j * np.eye(N)  # tiny broadening for nonzero Gam
    P_lower, Emin = damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer=5.0)
    # Reference: call density directly with same Fbar/GamBar (including ETA shift).
    H_eff_ref = F_eV + Sigma_0 + 1j * ETA * np.eye(N)
    Fbar = H_eff_ref
    Gam = (H_eff_ref - H_eff_ref.conj().T) * 1j
    D, V = np.linalg.eig(Fbar)
    Vc = np.linalg.inv(V.conj().T)
    P_ref = density(V, Vc, D, Gam, float(ENERGY_MIN), Emin)
    P_ref = np.asarray(P_ref)
    assert np.allclose(np.asarray(P_lower), P_ref, atol=1e-8)


def test_pure_function_no_state():
    """damleLowerDensity must be a pure function (calling twice gives same result)."""
    N = 3
    F_eV = np.diag([-1.0, 0.0, 1.0])
    Y_eff = np.eye(N)
    Sigma_0 = -0.01j * np.eye(N)
    P1, E1 = damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer=5.0)
    P2, E2 = damleLowerDensity(F_eV, Y_eff, Sigma_0, buffer=5.0)
    assert np.allclose(np.asarray(P1), np.asarray(P2))
    assert E1 == E2
