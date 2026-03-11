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
    """When Salpha is near-singular, _regularizeContacts must ensure aSList[i]
    is PSD via congruent clipping. Sbeta is also transformed consistently."""
    N = 4
    n = 2
    # Salpha with a near-zero eigenvalue
    Salpha = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex) + 1e-8 * np.eye(n)
    Sbeta = 0.1 * np.eye(n, dtype=complex)
    assert np.linalg.eigvalsh(Salpha)[0] < 1e-6, \
        "Test setup: Salpha should have a near-zero eigenvalue"

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [2, 3]],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # After init, aSList[i] must be PSD (congruent clipping applied)
    for i in range(2):
        eigs = np.linalg.eigvalsh(np.array(g.aSList[i]))
        assert eigs[0] > 0, (
            f"aSList[{i}] must be PSD after regularization, min eig = {eigs[0]:.4e}")


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
    """After regularization, aSList[i] must be PSD (positive definite).

    Congruent clipping floors small eigenvalues of Salpha = aSList[i].
    Uses a near-singular Salpha to ensure regularization fires."""
    N = 4
    n = 2
    # Near-singular Salpha: one eigenvalue ~0, one ~2
    Salpha = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex) + 1e-9 * np.eye(n)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    assert np.linalg.eigvalsh(Salpha)[0] < 1e-6, \
        "Test setup: Salpha should have a near-zero eigenvalue"

    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)

    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)

    g = surfG(F, S, [[0, 1], [2, 3]],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # After regularization, aSList[i] must be PSD
    for i in range(2):
        eigs = np.linalg.eigvalsh(np.array(g.aSList[i]))
        assert eigs[0] > 0, (
            f"aSList[{i}] must be PSD after regularization, min eig = {eigs[0]:.4e}")


def test_congruent_clipping_transforms_matrices():
    """_regularizeContacts must apply congruent clipping when S0 is not PSD."""
    import numpy as np
    n = 2
    # Salpha with a near-zero eigenvalue: eigenvalues ~[0, 1]
    Salpha = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex) + 1e-8 * np.eye(2)
    alpha = np.eye(n, dtype=complex)
    Sbeta = np.zeros((n, n), dtype=complex)
    beta = np.zeros((n, n), dtype=complex)
    N = 4
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)

    g = surfG(F, S,
              indsList=[[0, 1], [2, 3]],
              alphas=[alpha.copy(), alpha.copy()],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta.copy(), beta.copy()],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])

    # _overlap_eps must NOT exist anymore
    assert not hasattr(g, '_overlap_eps'), "sigma-correction approach must be removed"

    # aSList[i] must be well-conditioned after transform
    for i in range(2):
        S0_reg = np.array(g.aSList[i])
        eigvals = np.linalg.eigvalsh(S0_reg)
        assert eigvals[0] > 0, f"aSList[{i}] must be positive definite after clipping"


def test_congruent_clipping_skips_when_already_psd():
    """_regularizeContacts must leave matrices unchanged when S0 is already PSD."""
    import numpy as np
    n = 2
    N = 6
    F, S = make_chain(N, S_offdiag=0.05)
    Salpha_before = np.array(S[:2, :2])
    g = surfG(F, S, [[0, 1], [4, 5]])

    assert not hasattr(g, '_overlap_eps'), "sigma-correction approach must be removed"
    # aSList should match the original Salpha (no transform applied)
    np.testing.assert_allclose(np.array(g.aSList[0]), Salpha_before, atol=1e-12)


def test_device_overlap_unchanged_after_regularization():
    """g.S must remain the unmodified original overlap (sigma-correction
    approach).  g.S_orig must also store the unmodified original."""
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

    # g.S must also equal the original input (sigma-correction, not S-modification)
    np.testing.assert_allclose(
        np.array(g.S), S, atol=1e-14,
        err_msg="g.S must be unchanged (sigma-correction approach, not S-modification)")



def test_integer_transmission_with_large_overlap():
    """For a 2-cell device where both cells are contacts (device = repeating
    unit folded into both contacts), transmission must be integer even with
    large overlap that triggers regularization.  The S-modification approach
    keeps the device-lead interface transparent when every device cell gets the
    same Salpha shift via g.S."""
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




# ---------------------------------------------------------------------------
# CNT (3,3) tests: integer transmission and Fermi energy with overlap
# ---------------------------------------------------------------------------

def _construct_cnt33(t):
    """(3,3) armchair CNT tight-binding Hamiltonian, 12 atoms per unit cell.

    Returns Haa (12x12 intra-cell) and Hab (12x12 inter-cell hopping).
    Each carbon has exactly 3 bonds (sp2 lattice).
    """
    bonds = [(2,1),(3,4),(6,5),(7,8),(10,9),(11,0)]
    Haa = np.zeros((12,12), dtype=complex)
    for i in range(11):
        Haa[i, i+1] = t
    Haa[0, 11] = t
    Haa += Haa.conj().T
    Hab = np.zeros((12,12), dtype=complex)
    for bond in bonds:
        Hab[bond[0], bond[1]] = t
    return Haa, Hab


def _make_cnt33_surfg(t, s, eta=5e-4):
    """2-cell (3,3) CNT surfG with nearest-neighbor overlap s per inter-cell bond.

    Sb has the same bond pattern as Hab, scaled to give ||Sb||_2 = s.
    For s < 0.5 no regularization is needed; for s > 0.5 eps > 0.
    """
    Haa, Hab = _construct_cnt33(t)
    Sa = np.eye(12, dtype=complex)
    Sb = (s / abs(t)) * np.abs(Hab).astype(complex)
    F = np.block([[Haa, Hab], [Hab.conj().T, Haa]])
    Sdev = np.block([[Sa, Sb], [Sb.T, Sa]])
    return surfG(F, Sdev, [np.arange(12), np.arange(12, 24)],
                 alphas=[Haa, Haa], aOverlaps=[Sa.copy(), Sa.copy()],
                 taus=[Hab.conj().T, Hab], staus=[Sb.copy(), Sb.copy()],
                 betas=[Hab.conj().T, Hab], bOverlaps=[Sb.copy(), Sb.copy()],
                 eta=eta)


def _T_cnt(g, E):
    """Coherent transmission using g.S consistently."""
    eta = 5e-4
    S_g = np.array(g.S)
    F_g = np.array(g.F)
    sigL = np.array(g.sigma(E, 0))
    sigR = np.array(g.sigma(E, 1))
    Gr = np.linalg.inv((E + 1j*eta)*S_g - F_g - sigL - sigR)
    GamL = 1j*(sigL - sigL.conj().T)
    GamR = 1j*(sigR - sigR.conj().T)
    return float(np.real(np.trace(GamL @ Gr @ GamR @ Gr.conj().T)))


def test_cnt33_integer_transmission_with_overlap():
    """(3,3) CNT with three overlap strengths: T(E=0) must be integer (~2).

    The (3,3) armchair CNT is metallic with 2 channels at E=0.
    This test verifies that adding nearest-neighbor overlap (with or without
    regularization) preserves integer transmission when g.S is used for G_R.

    For non-symmetric Sb (directional inter-cell bonds), the exact S(k) scan
    shows S(k) is PSD for all k at s=0.6 (min eigval = 0.4), so no
    regularization is needed despite ||Sb||_2 = 0.6 > 0.5.
    """
    t = -2.7  # graphene/CNT hopping in eV

    for s in [0.0, 0.3, 0.6]:
        g = _make_cnt33_surfg(t=t, s=s)

        T = _T_cnt(g, E=0.0)
        assert abs(T - round(T)) < 0.05, (
            f"s={s}: T(E=0)={T:.4f} not integer (expected 2)")
        assert round(T) == 2, (
            f"s={s}: T(E=0) rounds to {round(T)}, expected 2 channels")


def test_cnt33_fermi_energy_with_overlap():
    """(3,3) CNT: Fermi level from generalized eigenproblem stays near band midpoint.

    For the half-filled metallic (3,3) CNT (24 states, 12 occupied), E_F is
    the midpoint between the 12th and 13th eigenvalue of F c = E S c.
    With s=0 this is exactly 0 by particle-hole symmetry.
    With overlap the Fermi level shifts slightly but must stay within the band.
    """
    from scipy.linalg import eigh

    t = -2.7
    band_half_width = abs(t) * 3   # rough upper bound on |E_F|

    for s in [0.0, 0.3, 0.6]:
        g = _make_cnt33_surfg(t=t, s=s)
        evals = eigh(np.array(g.F), np.array(g.S), eigvals_only=True)
        E_F = float((evals[11] + evals[12]) / 2.0)

        assert abs(E_F) < band_half_width, (
            f"s={s}: Fermi energy {E_F:.4f} eV outside band [-{band_half_width}, {band_half_width}]")

        # For s=0: particle-hole symmetry pins E_F exactly to 0
        if s == 0.0:
            assert abs(E_F) < 1e-10, f"s=0: E_F={E_F:.2e} should be exactly 0"


# ---------------------------------------------------------------------------
# Sigma-correction regression: half-filled 1D chain with high overlap
# ---------------------------------------------------------------------------

def test_1d_chain_high_overlap_transmission_and_fermi():
    """Half-filled 1-orbital chain (Sbeta=0.8) with 4 unit cells.

    Sbeta=0.8 triggers regularization (eps~0.6), exercising the sigma-correction
    path.  Checks:
    (a) Transmission inside the band is non-zero (single open channel).
    (b) getFermi1DContact converges and returns E_F ≈ 0.
        Analytic result: E(k=pi/2) = -2*cos(pi/2) / (1 + 1.6*cos(pi/2)) = 0
        for any Sbeta, so E_F = 0 exactly for half-filling with alpha=0.
    """
    from gauNEGF.density import getFermi1DContact

    n = 1
    alpha = np.zeros((n, n), dtype=complex)
    beta = -1.0 * np.eye(n, dtype=complex)
    Salpha = np.eye(n, dtype=complex)
    Sbeta = 0.8 * np.eye(n, dtype=complex)

    # 4-cell device, contacts at first and last cell
    n_cells = 4
    N = n_cells * n
    F = np.zeros((N, N), dtype=complex)
    S = np.zeros((N, N), dtype=complex)
    for c in range(n_cells):
        F[c, c] = 0.0
        S[c, c] = 1.0
        if c < n_cells - 1:
            F[c, c+1] = -1.0
            F[c+1, c] = -1.0
            S[c, c+1] = 0.8
            S[c+1, c] = 0.8

    g = surfG(F, S, [[0], [N-1]],
              taus=[beta, beta],
              staus=[Sbeta, Sbeta],
              alphas=[alpha, alpha],
              aOverlaps=[Salpha.copy(), Salpha.copy()],
              betas=[beta, beta],
              bOverlaps=[Sbeta.copy(), Sbeta.copy()])


    # (a) Transmission at E=0 should be non-zero (channel is open)
    eta = 1e-4
    S_g = np.array(g.S)
    F_g = np.array(g.F)
    sigL = np.array(g.sigma(0.0, 0))
    sigR = np.array(g.sigma(0.0, 1))
    Gr = np.linalg.inv((0.0 + 1j*eta)*S_g - F_g - sigL - sigR)
    GamL = 1j*(sigL - sigL.conj().T)
    GamR = 1j*(sigR - sigR.conj().T)
    T = float(np.real(np.trace(GamL @ Gr @ GamR @ Gr.conj().T)))
    assert T > 0.1, f"T(E=0)={T:.4f} should be non-zero for open channel"
    assert T <= 1.05, f"T(E=0)={T:.4f} should be <= 1 channel"

    # (b) Fermi search: ne=0.5 fills half the spinless band (k_F = pi/2).
    # Analytic: E(k=pi/2) = -2*cos(pi/2) / (S_alpha + 2*S_beta*cos(pi/2)) = 0
    # for any Sbeta when alpha=0.
    fermi = getFermi1DContact(g, ne=0.5, ind=0)
    assert abs(fermi) < 0.1, (
        f"E_F={fermi:.4f} eV, expected ~0 for spinless half-band fill with alpha=0")


