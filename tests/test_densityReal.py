"""Tests for densityReal using ANT quadrature.

Behavioral contract:
  - densityReal(Eminf, E_split) in zero-DOS region returns (P~0, delta_N~0)
  - densityReal must NOT spuriously converge at N=1 (the GL bug)
  - Results match densityRealN with large N for smooth integrands
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import io
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest

from gauNEGF.surfG1D import surfG
from gauNEGF.density import densityReal, densityRealN, calcEmin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_state_g():
    """2-state device: deep state at -10 eV, near-Fermi state at -1 eV.

    Zero-DOS region is [Eminf, calcEmin_output] (below -10 eV eigenvalue).
    Midpoint of the zero-DOS section falls nowhere near any state, so the
    old GL densityReal converges spuriously at N=1.
    """
    F = np.diag([-10.0, -1.0]).astype(complex)
    S = np.eye(2, dtype=complex)
    # indsList [[0],[1]]: each state connects to one contact via tauFromFock
    # With off-diagonal F=0 (diagonal F), tau=0 -> no self-energy broadening
    # eta provides only the retarded +i*eta shift
    return surfG(F, S, [[0], [1]], eta=1e-4)


@pytest.fixture
def zero_dos_bounds(two_state_g):
    """Return (Eminf, E_split) for the zero-DOS region of two_state_g."""
    g = two_state_g
    F = np.array(g.F)
    S = np.array(g.S)
    E_split = calcEmin(F, S, g)   # just below -10 eV eigenvalue
    Eminf = E_split - 10.0        # deeper, still zero-DOS
    return Eminf, E_split


# ---------------------------------------------------------------------------
# RED TEST: must fail with old GL densityReal, pass with new ANT densityReal
# ---------------------------------------------------------------------------

def test_densityReal_does_not_converge_at_N1(two_state_g, zero_dos_bounds):
    """densityReal must not spuriously converge at N=1.

    Old GL densityReal compares N=1 result to zeros_like initial P.  In a
    zero-DOS region the N=1 midpoint also gives P=0, so maxDP=0 < tol and
    it prints 'in 1 points' -- spurious convergence.

    New ANT densityReal first checks convergence at N=6 (N=2 has no prior
    to compare against), so it can never print 'in 1 points'.
    """
    g = two_state_g
    F = np.array(g.F)
    S = np.array(g.S)
    Eminf, E_split = zero_dos_bounds

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        densityReal(F, S, g, Eminf, E_split, T=0)
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert ' in 1 points' not in output, (
        "densityReal converged at N=1 -- this is a spurious convergence. "
        "ANT quadrature must perform at least one convergence check (N=2 vs N=6)."
    )


# ---------------------------------------------------------------------------
# Correctness tests (pass with both old and new, define contract)
# ---------------------------------------------------------------------------

def test_densityReal_zero_dos_gives_zero(two_state_g, zero_dos_bounds):
    """densityReal in zero-DOS region returns near-zero P and delta_N."""
    g = two_state_g
    F = np.array(g.F)
    S = np.array(g.S)
    Eminf, E_split = zero_dos_bounds

    P, delta_N = densityReal(F, S, g, Eminf, E_split, T=0)
    ne = np.trace(S @ P).real + delta_N

    # Threshold accommodates Lorentzian-tail contribution from finite eta=1e-4:
    # tail ~ eta * range / dE^2 ~ 1e-4 * 10 / 15^2 ~ 5e-6 (matches observed value)
    assert abs(ne) < 1e-4, f"Expected ne~0 in zero-DOS region, got ne={ne:.3e}"
    assert np.max(np.abs(P)) < 1e-4, f"Expected P~0, got max|P|={np.max(np.abs(P)):.3e}"


def test_densityReal_matches_densityRealN_in_zero_dos(two_state_g, zero_dos_bounds):
    """densityReal agrees with densityRealN(N=100) for the zero-DOS region."""
    g = two_state_g
    F = np.array(g.F)
    S = np.array(g.S)
    Eminf, E_split = zero_dos_bounds

    P_adaptive, delta_N_adaptive = densityReal(F, S, g, Eminf, E_split, T=0)
    P_fixed, delta_N_fixed = densityRealN(F, S, g, Eminf, E_split, N=100, T=0)

    np.testing.assert_allclose(
        np.array(P_adaptive), np.array(P_fixed),
        atol=1e-8,
        err_msg="densityReal and densityRealN(N=100) disagree in zero-DOS region"
    )
    assert abs(delta_N_adaptive - delta_N_fixed) < 1e-8, (
        f"delta_N mismatch: {delta_N_adaptive:.3e} vs {delta_N_fixed:.3e}"
    )


def test_densityReal_returns_correct_types(two_state_g, zero_dos_bounds):
    """densityReal returns (ndarray, float) tuple."""
    g = two_state_g
    F = np.array(g.F)
    S = np.array(g.S)
    Eminf, E_split = zero_dos_bounds

    result = densityReal(F, S, g, Eminf, E_split, T=0)
    assert isinstance(result, tuple) and len(result) == 2, "Should return (P, delta_N)"
    P, delta_N = result
    assert hasattr(P, 'shape'), "P should be array-like"
    assert P.shape == F.shape, f"P shape {P.shape} != F shape {F.shape}"
    assert isinstance(float(delta_N), float), "delta_N should be scalar"


def test_getFermiContact_nLower_uses_densityReal(two_state_g):
    """getFermiContact should complete without hitting densityComplex for nLower.

    With states at -10 and -1 eV and mu near -1 eV, nLower from [Eminf, calcEmin]
    should be ~0 whether computed via densityReal or densityComplex.
    The test checks the result is sensible; densityReal does it cheaper.
    """
    from gauNEGF.density import getFermiContact
    g = two_state_g
    ne = 1.0  # target: 1 electron -- state at -10 eV occupied, -1 eV empty
    # mu should fall in the gap between the two orbital energies
    mu = getFermiContact(g, ne, T=0)
    assert -10.0 < mu < -1.0, f"Expected mu in gap (-10, -1) eV, got {mu:.4f} eV"
