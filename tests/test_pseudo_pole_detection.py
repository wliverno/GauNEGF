"""Unit tests for calcPseudoPoleFloor (pseudo-pole detection for NEGF Eminf).

See docs/pseudo_pole_handling.md for the math.

These tests use a small _SyntheticG fixture that mimics a surfG-like object
with a programmable asymptotic Sigma(E) = E*X + Sigma_0 (+ optional 1/E
correction). For any target effective generalized eigenproblem (H_eff, S_eff)
we construct _SyntheticG with X = S - S_eff and Sigma_0 = H_eff - F, so the
two-point asymptotic probe inside calcPseudoPoleFloor recovers (H_eff, S_eff)
exactly. This lets us test the indefinite-metric solver and the floor
selection through the production API without any real surfG.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import pytest

from gauNEGF.density import calcPseudoPoleFloor
from gauNEGF.config import ENERGY_MIN


class _SyntheticG:
    """Mimics a surfG with F, S, sigmaTot for top-level testing.

    Sigma(E) = E * X + Sigma_0 + Sigma_minus1 / E

    The optional 1/E correction lets us probe the accuracy of the two-point
    linear extrapolation in calcPseudoPoleFloor.
    """
    def __init__(self, F, S, X, Sigma_0, Sigma_minus1=None):
        self.F = np.asarray(F, dtype=complex)
        self.S = np.asarray(S, dtype=complex)
        self._X = np.asarray(X, dtype=complex)
        self._Sigma_0 = np.asarray(Sigma_0, dtype=complex)
        self._Sigma_minus1 = (np.zeros_like(self._X)
                              if Sigma_minus1 is None
                              else np.asarray(Sigma_minus1, dtype=complex))

    def sigmaTot(self, E, conv=None):
        return E * self._X + self._Sigma_0 + self._Sigma_minus1 / E


def _make_g(F, S, H_eff_target, S_eff_target, Sigma_minus1=None):
    """Build a _SyntheticG whose asymptotic probe recovers (H_eff_target, S_eff_target)."""
    X = np.asarray(S, dtype=complex) - np.asarray(S_eff_target, dtype=complex)
    Sigma_0 = np.asarray(H_eff_target, dtype=complex) - np.asarray(F, dtype=complex)
    return _SyntheticG(F, S, X, Sigma_0, Sigma_minus1=Sigma_minus1)


# ---------- Fixture sanity ----------

def test_synthetic_g_sigmaTot_matches_expected():
    """Sanity: _SyntheticG.sigmaTot reproduces E*X + Sigma_0 (+ 1/E)."""
    X = np.diag([0.1, 0.2]).astype(complex)
    Sigma_0 = np.diag([1.0, -1.0]).astype(complex)
    g = _SyntheticG(np.zeros_like(X), np.eye(2, dtype=complex), X, Sigma_0)
    expected = -1000.0 * X + Sigma_0
    np.testing.assert_allclose(g.sigmaTot(-1000.0), expected, rtol=1e-12)


# ---------- No pseudo-poles cases ----------

def test_orth_contact_returns_energy_min():
    """Orthogonal contact (X = 0): S_eff = S stays PSD, no pseudo-poles,
    floor falls back to ENERGY_MIN."""
    N = 3
    F = np.diag([-2.0, 0.0, 2.0]).astype(complex)
    S = np.eye(N, dtype=complex)
    g = _SyntheticG(F, S, X=np.zeros((N, N)), Sigma_0=np.zeros((N, N)))

    floor = calcPseudoPoleFloor(F, S, g)

    assert floor == ENERGY_MIN, \
        f"Orth case should return ENERGY_MIN ({ENERGY_MIN}), got {floor}"


def test_PSD_S_eff_returns_energy_min():
    """S_eff = S - X is positive-definite (even though X is non-zero): no
    pseudo-poles -> ENERGY_MIN."""
    N = 3
    F = np.diag([-2.0, 0.0, 2.0]).astype(complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = F
    S_eff_target = np.diag([0.9, 0.9, 0.9]).astype(complex)   # PD
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g)

    assert floor == ENERGY_MIN, f"PSD S_eff should give ENERGY_MIN, got {floor}"


def test_positive_pseudopole_returns_energy_min():
    """A pseudo-pole at POSITIVE E cannot be excluded by Eminf (which is a
    lower bound on the contour). calcPseudoPoleFloor should filter it out
    and, if it's the only pseudo-pole, return ENERGY_MIN.

    Construction: H_eff = diag(-5, -3), S_eff = diag(1, -1).
      gen eigval 1: -5 / 1  = -5,  v=[1,0],  v^H S v = +1  (physical)
      gen eigval 2: -3 / -1 = +3,  v=[0,1],  v^H S v = -1  (pseudo, positive)
    Filtered out by pp_energies < 0; pp_neg is empty -> ENERGY_MIN.
    """
    N = 2
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = np.diag([-5.0, -3.0]).astype(complex)
    S_eff_target = np.diag([1.0, -1.0]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g)

    assert floor == ENERGY_MIN, \
        f"Positive pseudo-pole should be filtered; got floor={floor}, expected ENERGY_MIN"


# ---------- Single / multiple pseudo-poles ----------

def test_single_pseudopole_with_buffer():
    """S_eff has one negative eigenvalue -> one pseudo-pole.
    Construct H_eff = diag(1, 1, 10), S_eff = diag(1, 1, -0.01) so the
    pseudo-pole is at 10 / (-0.01) = -1000 eV.
    """
    N = 3
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = np.diag([1.0, 1.0, 10.0]).astype(complex)
    S_eff_target = np.diag([1.0, 1.0, -0.01]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=100.0)

    # Highest pseudo-pole at -1000 eV + 100 buffer = -900 eV
    np.testing.assert_allclose(floor, -900.0, rtol=1e-3)


def test_multiple_pseudopoles_use_max():
    """When multiple pseudo-poles exist, the floor is set by the HIGHEST
    (least negative) one. Pseudo-poles at -50 and -100; max is -50."""
    N = 4
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = np.diag([1.0, 5.0, 1.0, -3.0]).astype(complex)
    # Generalized eigvals = (1/1, 5/-0.1, 1/-0.01, -3/1) = (1, -50, -100, -3)
    # Pseudo-poles: where S_eff diagonal is negative -> -50 and -100
    S_eff_target = np.diag([1.0, -0.1, -0.01, 1.0]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=10.0)

    # Max pseudo-pole = -50, floor = -50 + 10 = -40
    np.testing.assert_allclose(floor, -40.0, rtol=1e-3)


def test_fractional_buffer_at_deep_pseudopole():
    """At deep pseudo-poles, the buffer scales fractionally with |E_pp|.
    pp at -3000 with alpha=0.1 -> buffer = 300 -> floor = -2700.
    min_buffer=50 is the irreducible floor; fractional dominates here.
    """
    N = 3
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = np.diag([1.0, 1.0, 30.0]).astype(complex)
    # gen eigvals: 1/1, 1/1, 30/-0.01 = -3000 (pseudo)
    S_eff_target = np.diag([1.0, 1.0, -0.01]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.1, min_buffer=50.0)

    # buffer = max(0.1 * 3000, 50) = 300; floor = -3000 + 300 = -2700
    np.testing.assert_allclose(floor, -2700.0, rtol=1e-3)


def test_min_buffer_dominates_at_moderate_pseudopole():
    """At moderate pseudo-pole depth, min_buffer overrides the fractional
    formula. pp at -100 with alpha=0.1, min_buffer=50: fractional gives 10,
    but min_buffer=50 wins -> floor = -100 + 50 = -50.
    """
    N = 3
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = np.diag([1.0, 1.0, 10.0]).astype(complex)
    # gen eigvals: 1/1, 1/1, 10/-0.1 = -100 (pseudo)
    S_eff_target = np.diag([1.0, 1.0, -0.1]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.1, min_buffer=50.0)

    # buffer = max(0.1 * 100, 50) = 50; floor = -100 + 50 = -50
    np.testing.assert_allclose(floor, -50.0, rtol=1e-3)


def test_shallow_pseudopole_buffered_positive_returns_energy_min():
    """If a pseudo-pole is so shallow that even the buffered floor is
    non-negative, there is no safe Eminf -> return ENERGY_MIN so calcTSW's
    dTSW<0 path handles the system.

    pp at -7, alpha=0.1, min_buffer=50: buffer = max(0.7, 50) = 50,
    floor candidate = -7 + 50 = +43 (>= 0). Should return ENERGY_MIN.

    (This is the C2_chain LANL2DZ case: shallow pseudo-pole near the HOMO,
    no safe Eminf exists.)
    """
    N = 2
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    # gen eigvals: 1/1 = 1 (physical), 7/-1 = -7 (pseudo)
    H_eff_target = np.diag([1.0, 7.0]).astype(complex)
    S_eff_target = np.diag([1.0, -1.0]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.1, min_buffer=50.0)

    assert floor == ENERGY_MIN, \
        f"Shallow pp with positive buffered floor should give ENERGY_MIN, got {floor}"


def test_multiple_pseudopoles_shallowest_buffered_wins():
    """When multiple deep pseudo-poles exist, the floor is set by whichever
    has the shallowest BUFFERED floor (not necessarily the shallowest raw pp).

    pp at -3000, -800, -200 with alpha=0.1, min_buffer=50:
      buffers = [300, 80, 50] (fractional dominates first two, min_buffer
                               dominates third)
      floors  = [-2700, -720, -150]
      shallowest = -150 (from pp=-200).
    """
    N = 4
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    # gen eigvals: 1/1 = 1 (physical),
    #              3000 / -1   = -3000 (pseudo)
    #              80 / -0.1   = -800  (pseudo)
    #              2 / -0.01   = -200  (pseudo)
    H_eff_target = np.diag([1.0, 3000.0, 80.0, 2.0]).astype(complex)
    S_eff_target = np.diag([1.0, -1.0, -0.1, -0.01]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.1, min_buffer=50.0)

    np.testing.assert_allclose(floor, -150.0, rtol=1e-3)


def test_offdiagonal_S_eff():
    """Rotated (H_eff, S_eff) -- same physics as the diagonal case, just
    expressed in a non-aligned basis. Tests that the implementation is
    basis-independent."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    H_eff_diag = np.diag([1.0, 1.0, 10.0])
    S_eff_diag = np.diag([1.0, 1.0, -0.01])
    H_eff_target = (Q @ H_eff_diag @ Q.T).astype(complex)
    S_eff_target = (Q @ S_eff_diag @ Q.T).astype(complex)

    N = 3
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=0.0)
    # Same pseudo-pole as the diagonal case: -1000 eV
    np.testing.assert_allclose(floor, -1000.0, rtol=1e-4)


# ---------- Truly general case (positive/negative-norm subspaces COUPLED) ----------

def test_general_coupling_indefinite_metric():
    """H_eff couples the positive-norm and negative-norm directions of S_eff.

    This is the case where a 'whitening' approach (jit-eigh on the
    S_eff^{-1/2} H S_eff^{-1/2} standard form) gives the wrong answer; the
    correct generalized solver must reduce to a non-Hermitian eigproblem of
    T = S_eff^{-1} H_eff.

    Closed form for H_eff = [[2, 1], [1, 1]], S_eff = diag(1, -1):
        det(H_eff - E S_eff) = (2-E)(1+E) - 1 = -E^2 + E + 1
        E = (1 +- sqrt(5)) / 2  =>  E_pseudo = (1 - sqrt(5))/2 ~ -0.618
    """
    H_eff_target = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=complex)
    S_eff_target = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    N = 2
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=0.0)

    expected = (1.0 - np.sqrt(5.0)) / 2.0   # ~ -0.618
    np.testing.assert_allclose(floor, expected, rtol=1e-8)


# ---------- Complex Hermitian (GHF / SOC regime) ----------

def test_complex_hermitian_F():
    """GHF/SOC: F is complex Hermitian (not real symmetric). A naive .real
    on F would solve a DIFFERENT eigenproblem -- this test catches that bug.

    Pick H_eff = [[1, 1+i], [1-i, 2]] (Hermitian), S_eff = diag(1, -1).
        det(H_eff - E S_eff) = (1-E)(2+E) - (1+i)(1-i)
                             = (1-E)(2+E) - 2 = -E^2 - E + 0 = -E(E+1)
        Eigvals: E = 0 and E = -1.
        E=-1 eigenvector has v^H S v = -1 -> pseudo-pole at -1
        E= 0 eigenvector has v^H S v = +1 -> physical (filtered out)
    Expected floor with alpha=0, min_buffer=0: -1.0

    If the implementation wrongly takes .real of H_eff, the real-only
    answer differs (golden-ratio-ish), so floor != -1 if .real is used.
    """
    H_eff_target = np.array([[1.0,        1.0 + 1.0j],
                             [1.0 - 1.0j, 2.0      ]], dtype=complex)
    S_eff_target = np.array([[1.0, 0.0],
                             [0.0, -1.0]], dtype=complex)

    N = 2
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    g = _make_g(F, S, H_eff_target, S_eff_target)

    floor = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=0.0)

    np.testing.assert_allclose(floor, -1.0, atol=1e-8)

    # Wrong-real guardrail: solving (1, 1) instead of (1, 1+i) gives
    # eigvals of det((1-E)(2+E) - 1) = -E^2 - E + 1 = 0 -> E = (-1 +- sqrt(5))/2
    # negative root ~ -1.618. Make sure we're not getting that.
    wrong_real_pp = (-1.0 - np.sqrt(5.0)) / 2.0   # ~ -1.618
    assert abs(floor - wrong_real_pp) > 0.3, \
        f"Floor {floor} matches the real-only answer; complex support is broken"


# ---------- Asymptotic probe accuracy ----------

def test_deeper_probes_suppress_1overE_correction():
    """Adding a 1/E piece to Sigma contaminates the two-point asymptotic
    probe. Deeper probes (larger |E|) suppress this contamination."""
    # Single pseudo-pole at -1000 eV, but Sigma has a 1/E correction term.
    N = 3
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    H_eff_target = np.diag([1.0, 1.0, 10.0]).astype(complex)
    S_eff_target = np.diag([1.0, 1.0, -0.01]).astype(complex)
    # Strong 1/E perturbation on the pseudo-pole channel
    Sigma_m1 = np.diag([0.0, 0.0, 50.0]).astype(complex)
    g = _make_g(F, S, H_eff_target, S_eff_target, Sigma_minus1=Sigma_m1)

    # Shallow probe: contamination dominates
    floor_shallow = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=0.0,
                                         E1=-1e2, E2=-1e3)
    err_shallow = abs(floor_shallow - (-1000.0))

    # Deep probe: contamination suppressed
    floor_deep = calcPseudoPoleFloor(F, S, g, alpha=0.0, min_buffer=0.0,
                                      E1=-1e4, E2=-1e5)
    err_deep = abs(floor_deep - (-1000.0))

    assert err_deep < err_shallow, \
        f"Deeper probe should be more accurate: shallow err={err_shallow:.3e}, deep err={err_deep:.3e}"
    assert err_deep < 1.0, \
        f"Deep-probe error {err_deep:.3e} should be < 1 eV"


