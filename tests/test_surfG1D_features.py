"""
Tests for surfG1D features ported from surfGBethe:
  - Xi (S^+0.5) attribute at init
  - None stau -> orthonormal coupling, no crash
  - None aOverlaps/bOverlaps defaults in setContacts
  - De-orthonormalization in sigma() when staus=None
  - setF() Fermi update bug fix (pattern c)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from gauNEGF.surfG1D import surfG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chain(N, t=-1.0, eps=0.0, S_offdiag=0.0):
    """Return F, S for an N-site 1D tight-binding chain.

    S_offdiag != 0 gives a non-identity overlap matrix (nearest-neighbor).
    """
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N):
        F[i, i] = eps
    for i in range(N - 1):
        F[i, i + 1] = t
        F[i + 1, i] = t
        S[i, i + 1] = S_offdiag
        S[i + 1, i] = S_offdiag
    return F, S


def make_pattern_c_contact(n=3):
    """Return simple (alpha, beta) contact matrices for pattern (c) testing."""
    alpha = np.diag(np.array([0.0] * n, dtype=complex))
    beta = np.diag(np.array([-1.0] * n, dtype=complex))
    return alpha, beta


# ---------------------------------------------------------------------------
# Test 1: Xi attribute exists after init
# ---------------------------------------------------------------------------

def test_xi_attribute_exists_after_init():
    """surfG should store Xi = S^+0.5 after init (like surfGBethe)."""
    N = 6
    F, S = make_chain(N)
    g = surfG(F, S, [[0, 1], [4, 5]])
    assert hasattr(g, 'Xi'), "surfG should have Xi = S^(+0.5) attribute"
    assert g.Xi.shape == (N, N), f"Xi should be {N}x{N}, got {g.Xi.shape}"


# ---------------------------------------------------------------------------
# Test 2: staus=None does not crash (orthonormal coupling)
# ---------------------------------------------------------------------------

def test_staus_none_does_not_crash():
    """Providing tau matrices but staus=None should not crash init."""
    N = 6
    F, S = make_chain(N)
    # Pattern (b): tau matrices provided, staus=None -> orthonormal coupling
    tau_L = jnp.array(F[0:1, 1:3])   # 1x2 coupling block
    tau_R = jnp.array(F[5:6, 3:5])
    # staus=None (default) with matrix taus -> should work now
    g = surfG(F, S, [[1, 2], [3, 4]], taus=[tau_L, tau_R], staus=None)
    assert g is not None
    assert g.stauList[0] is None
    assert g.stauList[1] is None


# ---------------------------------------------------------------------------
# Test 3: None aOverlaps defaults to identity in setContacts
# ---------------------------------------------------------------------------

def test_aoverlaps_none_defaults_to_identity():
    """When contactFromFock=False and aOverlaps=None, aSList should be identity."""
    N = 6
    n_contact = 2
    F, S = make_chain(N, S_offdiag=0.05)
    alpha, beta = make_pattern_c_contact(n_contact)
    tau = np.array(F[0:n_contact, n_contact:n_contact + 2])
    stau = np.array(S[0:n_contact, n_contact:n_contact + 2])

    # Pattern (c) with aOverlaps=None -> should default to identity
    g = surfG(F, S,
              [[n_contact, n_contact + 1], [N - n_contact - 2, N - n_contact - 1]],
              taus=[tau, tau],
              staus=[stau, stau],
              alphas=[alpha, alpha],
              aOverlaps=None,
              betas=[beta, beta],
              bOverlaps=None)
    # aSList should be identity matrices
    for aS in g.aSList:
        np.testing.assert_allclose(
            np.array(aS), np.eye(n_contact), atol=1e-12,
            err_msg="aSList[i] should be identity when aOverlaps=None")
    # bSList should be zeros
    for bS in g.bSList:
        np.testing.assert_allclose(
            np.array(bS), np.zeros((n_contact, n_contact)), atol=1e-12,
            err_msg="bSList[i] should be zeros when bOverlaps=None")


# ---------------------------------------------------------------------------
# Test 4: setF Fermi update bug fix (pattern c)
# ---------------------------------------------------------------------------

def test_setf_fermi_update_pattern_c_no_crash():
    """Second call to setF() with changed mu should not crash for pattern (c)."""
    N = 6
    n_contact = 2
    F, S = make_chain(N)
    alpha, beta = make_pattern_c_contact(n_contact)
    tau = np.array(F[0:n_contact, n_contact:n_contact + 2])

    g = surfG(F, S,
              [[n_contact, n_contact + 1], [N - n_contact - 2, N - n_contact - 1]],
              taus=[tau, tau],
              staus=None,
              alphas=[alpha, alpha],
              betas=[beta, beta])

    # First setF call: initializes fermiList
    g.setF(F, mu1=0.0, mu2=0.0)

    # Second setF call with different mu -> previously crashed with AttributeError
    g.setF(F, mu1=0.1, mu2=-0.1)  # should not raise

    # Verify alpha was shifted correctly (aList[0] should be alpha + 0.1*I)
    expected = np.array(alpha) + 0.1 * np.eye(n_contact)
    np.testing.assert_allclose(
        np.array(g.aList[0]), expected, atol=1e-12,
        err_msg="aList[0] should be shifted by dFermi after setF")


# ---------------------------------------------------------------------------
# Test 5: De-orthonormalization applied in sigma() when staus=None
# ---------------------------------------------------------------------------

def test_sigma_deortho_applied_when_staus_none():
    """When staus=None and S != I, sigma with de-ortho should differ from without."""
    # Use a non-trivial S (nearest-neighbor overlap) so Xi != I
    N = 6
    n_contact = 1
    S_offdiag = 0.2
    F, S = make_chain(N, S_offdiag=S_offdiag)

    # Contact indices (pattern b with staus=None for orthonormal coupling)
    tau_ortho = jnp.array([[-1.0 + 0j]])  # 1x1 hopping (orthonormal, no overlap)

    # surfG with staus=None -> de-ortho will be applied
    g_deortho = surfG(F, S, [[1], [4]], taus=[tau_ortho, tau_ortho], staus=None)

    # surfG with staus=zeros matrix -> de-ortho NOT applied (stau is not None)
    stau_zero = jnp.zeros((1, 1), dtype=complex)
    g_no_deortho = surfG(F, S, [[1], [4]], taus=[tau_ortho, tau_ortho],
                         staus=[stau_zero, stau_zero])

    E = 0.5
    sig_deortho = np.array(g_deortho.sigma(E, 0))
    sig_no_deortho = np.array(g_no_deortho.sigma(E, 0))

    # With S_offdiag != 0, Xi != I, so the de-orthonormalized sigma must differ
    # from the raw sigma (unless Xi happens to be exactly I which it won't be)
    max_diff = np.max(np.abs(sig_deortho - sig_no_deortho))
    assert max_diff > 1e-6, (
        f"sigma with de-ortho should differ from sigma without de-ortho "
        f"when S has off-diagonal elements, but max diff = {max_diff:.2e}")
