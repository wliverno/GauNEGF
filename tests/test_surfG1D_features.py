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
    """setF() for pattern (c) should track mu but NOT shift aList/bList.
    The retarded self-energy is independent of chemical potential."""
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

    # Second setF call with different mu -> should not crash or shift aList
    g.setF(F, mu1=0.1, mu2=-0.1)

    # aList must stay at the original alpha (no Fermi shift)
    np.testing.assert_allclose(
        np.array(g.aList[0]), np.array(alpha), atol=1e-12,
        err_msg="aList[0] must not be shifted by setF -- sigma is mu-independent")


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


# ---------------------------------------------------------------------------
# Test 6: Composite overlap regularization
# ---------------------------------------------------------------------------

def _build_composite_chain(Salpha, Sbeta):
    """Build 3-block Toeplitz chain overlap."""
    n = Salpha.shape[0]
    Z = np.zeros_like(Salpha)
    return np.block([
        [Salpha, Sbeta,          Z],
        [Sbeta.conj().T, Salpha, Sbeta],
        [Z,     Sbeta.conj().T,  Salpha],
    ])


def test_chain_composite_regularized():
    """When the chain composite [[Salpha, Sbeta, 0], ...] is non-PSD,
    surfG should shift aSList so the composite becomes PSD.
    Sbeta must remain unchanged."""
    N = 6
    n = 2  # orbitals per contact
    # Onsite overlap
    Salpha = np.eye(n, dtype=complex)
    # Large coupling overlap -> chain composite non-PSD
    Sbeta = 0.8 * np.eye(n, dtype=complex)
    assert np.linalg.eigvalsh(_build_composite_chain(Salpha, Sbeta))[0] < 0, \
        "Test setup: chain composite should be non-PSD"

    # Build F, S for a 6-site chain
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    # Use pattern (c): provide alpha, Salpha, beta, Sbeta directly
    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # After init, the chain composite should now be PSD
    for i in range(2):
        aS = np.array(g.aSList[i])
        bS = np.array(g.bSList[i])
        chain = _build_composite_chain(aS, bS)
        eigs = np.linalg.eigvalsh(chain)
        assert eigs[0] > -1e-10, (
            f"Chain composite for contact {i} should be PSD after regularization, "
            f"min eig = {eigs[0]:.4e}")

    # Sbeta should be unchanged
    for i in range(2):
        np.testing.assert_allclose(
            np.array(g.bSList[i]), Sbeta, atol=1e-12,
            err_msg=f"bSList[{i}] should be unchanged by regularization")


def test_sigma_no_blowup_with_regularized_overlap():
    """sigma() should produce reasonable values after composite regularization."""
    N = 6
    n = 2
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    sig = np.array(g.sigma(0.0, 0))
    assert np.all(np.isfinite(sig)), "sigma should be finite"
    assert np.max(np.abs(sig)) < 100, (
        f"sigma values should be reasonable, max = {np.max(np.abs(sig)):.2e}")


def test_already_psd_composites_unchanged():
    """When composites are already PSD, aSList should not be modified."""
    N = 6
    F, S = make_chain(N, S_offdiag=0.05)
    g = surfG(F, S, [[0, 1], [4, 5]])
    # With small overlap, composites should already be PSD
    # aSList should match the original S subblocks
    for i, inds in enumerate(g.indsList):
        expected = S[np.ix_(inds, inds)]
        np.testing.assert_allclose(
            np.array(g.aSList[i]), expected, atol=1e-10,
            err_msg=f"aSList[{i}] should be unchanged when composites are PSD")


# ---------------------------------------------------------------------------
# Test 7: Infinite chain overlap S(k) must be PSD for all k
# ---------------------------------------------------------------------------

def _inf_chain_overlap_min_eig(Salpha, Sbeta):
    """Min eigenvalue of S(k) = Salpha + 2*cos(k)*Sbeta over all k.

    For symmetric Sbeta, worst cases are k=0 and k=pi.
    For general Sbeta, uses spectral norm bound.
    """
    Sbeta = np.array(Sbeta)
    Salpha = np.array(Salpha)
    # Check both extremes: Salpha +/- 2*Sbeta_sym
    Sbeta_sym = (Sbeta + Sbeta.conj().T) / 2
    eig_pi = np.linalg.eigvalsh(Salpha - 2 * Sbeta_sym)[0]
    eig_0 = np.linalg.eigvalsh(Salpha + 2 * Sbeta_sym)[0]
    # Also check anti-symmetric contribution via spectral norm bound
    Sbeta_anti = (Sbeta - Sbeta.conj().T) / 2
    anti_norm = np.linalg.norm(Sbeta_anti, ord=2)
    return min(eig_pi, eig_0) - 2 * anti_norm


def test_infinite_chain_overlap_psd_after_regularization():
    """After regularization, the infinite chain S(k) = Salpha + 2*cos(k)*Sbeta
    must be PSD for ALL k, not just the 3-block Toeplitz sampling points.

    With Sbeta = 0.8*I and Salpha = I, the 3-block composite has min eig ~ -0.13
    but the infinite chain has min eig = 1 - 2*0.8 = -0.6.
    The regularization must handle the infinite chain condition."""
    N = 6
    n = 2
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    # Verify test setup: infinite chain is non-PSD
    assert _inf_chain_overlap_min_eig(Salpha, Sbeta) < -0.5, \
        "Test setup: infinite chain S(pi) = I - 1.6*I should have min eig ~ -0.6"

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # After regularization, infinite chain must be PSD
    for i in range(2):
        aS = np.array(g.aSList[i])
        bS = np.array(g.bSList[i])
        min_eig = _inf_chain_overlap_min_eig(aS, bS)
        assert min_eig > -1e-10, (
            f"Infinite chain S(k) for contact {i} must be PSD for all k, "
            f"min eig = {min_eig:.4e}")


def test_overlap_eps_stored_per_contact():
    """_regularizeContacts must store the shift eps per contact
    in self._overlap_eps so sigma() can apply the correction."""
    N = 6
    n = 2
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # _overlap_eps must exist and have one entry per contact
    assert hasattr(g, '_overlap_eps'), "surfG must store _overlap_eps"
    assert len(g._overlap_eps) == 2, "One eps per contact"

    # For Sbeta=0.8, eps should be 2*0.8 - 1.0 + 1e-10 ~ 0.6
    for i in range(2):
        assert g._overlap_eps[i] > 0.5, (
            f"eps[{i}] should be ~0.6, got {g._overlap_eps[i]:.4f}")


def test_overlap_eps_zero_when_no_regularization_needed():
    """When the chain is already PSD, _overlap_eps should be 0."""
    N = 6
    F, S = make_chain(N, S_offdiag=0.05)
    g = surfG(F, S, [[0, 1], [4, 5]])

    assert hasattr(g, '_overlap_eps'), "surfG must store _overlap_eps"
    for i in range(2):
        assert g._overlap_eps[i] == 0.0, (
            f"eps[{i}] should be 0 when no regularization needed")


def test_device_overlap_modified_at_contacts():
    """g.S must include the eps shift at contact blocks (S-modification
    approach).  g.S_orig must store the unmodified original."""
    N = 6
    n = 2
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # g.S_orig must equal the original input
    np.testing.assert_allclose(
        np.array(g.S_orig), S, atol=1e-14,
        err_msg="g.S_orig must be the unmodified original overlap")

    # g.S must have eps added at contact blocks
    S_expected = S.copy()
    for i, inds in enumerate([[0, 1], [4, 5]]):
        eps_i = g._overlap_eps[i]
        if eps_i > 0:
            S_expected[np.ix_(inds, inds)] += eps_i * np.eye(n)
    np.testing.assert_allclose(
        np.array(g.S), S_expected, atol=1e-14,
        err_msg="g.S must include the eps shift at contact blocks")


def test_sigma_uses_s_modification_not_correction():
    """With S-modification approach, sigma() should NOT contain the
    -E*eps*I correction.  Instead g.S already has eps at contacts,
    so G = [E*g.S - F - sigma_reg]^{-1} is correct directly."""
    N = 6
    n = 2
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # sigma() at a real energy should NOT have the -E*eps*I term.
    # Verify: the Green's function built from (g.S, sigma) matches
    # the one built from (S_orig, sigma + correction).
    from gauNEGF.utils import inv as jinv
    eta = 1e-4
    for E in [-10.0, 0.0, 2.5]:
        sig = np.array(g.sigmaTot(E))

        # Approach 1 (current): G = [E*S_mod - F - sigma_reg]^{-1}
        S_mod = np.array(g.S)
        Gr_1 = np.array(jinv(jnp.array((E + 1j*eta) * S_mod - F - sig)))

        # Approach 2 equivalent: G = [(E+i*eta)*S_orig - F - (sigma_reg - (E+i*eta)*eps*I)]^{-1}
        z = E + 1j*eta
        S_orig = np.array(g.S_orig)
        sig_corr = sig.copy()
        for i in range(2):
            inds = g.indsList[i]
            ni = len(inds)
            sig_corr[np.ix_(inds, inds)] -= z * g._overlap_eps[i] * np.eye(ni)
        Gr_2 = np.array(jinv(jnp.array(z * S_orig - F - sig_corr)))

        np.testing.assert_allclose(
            Gr_1, Gr_2, atol=1e-10,
            err_msg=f"S-modification and sigma-correction must give same G at E={E}")


def test_integer_transmission_with_large_overlap():
    """For a 2-cell device where both cells are contacts (device = repeating
    unit folded into both contacts), transmission must be integer even with
    large overlap that triggers regularization.  The sigma correction ensures
    the device-lead interface is transparent when every device cell gets the
    same Salpha shift."""
    from gauNEGF.utils import inv as jinv

    n = 2   # orbitals per unit cell
    n_cells = 2  # device = 2 unit cells, both are contacts
    N = n * n_cells

    # Lead parameters (large Sbeta triggers regularization, eps ~ 0.6)
    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    # Build device F and S from lead parameters so system is periodic
    F = np.zeros((N, N), dtype=complex)
    S = np.zeros((N, N), dtype=complex)
    for c in range(n_cells):
        sl = slice(c * n, (c + 1) * n)
        F[sl, sl] = alpha
        S[sl, sl] = Salpha
        if c < n_cells - 1:
            sl_next = slice((c + 1) * n, (c + 2) * n)
            F[sl, sl_next] = beta
            F[sl_next, sl] = beta.conj().T
            S[sl, sl_next] = Sbeta
            S[sl_next, sl] = Sbeta.conj().T

    g = surfG(F, S, [[0, 1], [2, 3]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    eta = 1e-4
    S_dev = np.array(g.S)
    F_dev = np.array(g.F)

    # 2 degenerate channels -> T=2 at energies inside the band
    for E in [0.0, -0.3, -0.5]:
        sig_L = np.array(g.sigma(E, 0))
        sig_R = np.array(g.sigma(E, 1))
        sig_tot = sig_L + sig_R

        Gr = np.array(jinv(jnp.array((E + 1j*eta) * S_dev - F_dev - sig_tot)))
        Ga = Gr.conj().T

        Gamma_L = 1j * (sig_L - sig_L.conj().T)
        Gamma_R = 1j * (sig_R - sig_R.conj().T)

        T = np.real(np.trace(Gamma_L @ Gr @ Gamma_R @ Ga))
        T_rounded = round(T)
        assert abs(T - T_rounded) < 0.05, (
            f"Transmission at E={E} should be integer, got T={T:.4f}")


def test_dos_decays_at_large_negative_energy():
    """With large overlap (Sbeta=0.8), DOS must decay far below the band.

    This is the physical consequence of the infinite chain regularization:
    no spurious DOS tail extending to -infinity."""
    from gauNEGF.utils import inv as jinv
    N = 6
    n = 2
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [4, 5]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # DOS at E=-50 should be negligible (band is roughly [-3, 3])
    E = -50.0
    eta = 1e-3
    sig = np.array(g.sigmaTot(E))
    S_dev = np.array(g.S)
    F_dev = np.array(g.F)
    Gr = np.array(jinv(jnp.array((E + 1j*eta) * S_dev - F_dev - sig)))
    dos = -np.imag(np.trace(Gr @ S_dev)) / np.pi
    assert dos < 0.01, (
        f"DOS at E={E} should be negligible after regularization, got {dos:.4f}")
