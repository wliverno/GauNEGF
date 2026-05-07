"""Tests for DOS cross-term Q correction in transport.dos_single_energy.

The correct DOS formula is:
    DOS = -1/pi * Im[Tr(G^R @ (S - Q_tot))]

The current transport._dos_kernel uses:
    DOS = -1/pi * Im[Tr(G^R)]   <-- wrong: missing S weighting and Q correction

These tests verify the fix matches density._compute_dos_at_energy (the reference).
"""
import numpy as np
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Test system helpers (reuse pattern from test_cross_term.py)
# ---------------------------------------------------------------------------

def make_nonortho_chain(N=4, nc=2):
    """Non-orthogonal 1D chain with nonzero stauList (S_DL != 0)."""
    from gauNEGF.surfG1D import surfG
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    for i in range(N - 1):
        F = F.at[i, i+1].set(-0.3)
        F = F.at[i+1, i].set(-0.3)
    S = jnp.eye(N) + 0.0j
    for i in range(N - 1):
        S = S.at[i, i+1].set(0.1)
        S = S.at[i+1, i].set(0.1)

    indsList = [jnp.array(list(range(nc))), jnp.array(list(range(N - nc, N)))]
    taus = [jnp.ones((nc, nc), dtype=complex) * -0.3,
            jnp.ones((nc, nc), dtype=complex) * -0.3]
    staus = [jnp.ones((nc, nc), dtype=complex) * 0.1,
             jnp.ones((nc, nc), dtype=complex) * 0.1]
    alphas = [jnp.diag(jnp.array([-0.5] * nc, dtype=complex)),
              jnp.diag(jnp.array([0.5] * nc, dtype=complex))]
    betas = [jnp.ones((nc, nc), dtype=complex) * -0.3,
             jnp.ones((nc, nc), dtype=complex) * -0.3]
    aOverlaps = [jnp.eye(nc, dtype=complex) * 1.05,
                 jnp.eye(nc, dtype=complex) * 1.05]
    bOverlaps = [jnp.ones((nc, nc), dtype=complex) * 0.1,
                 jnp.ones((nc, nc), dtype=complex) * 0.1]
    return surfG(F, S, indsList, taus=taus, staus=staus,
                 alphas=alphas, betas=betas,
                 aOverlaps=aOverlaps, bOverlaps=bOverlaps, eta=1e-3)


def make_ortho_chain(N=4, nc=2):
    """Orthogonal 1D chain with identity overlap and no stau."""
    from gauNEGF.surfG1D import surfG
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    for i in range(N - 1):
        F = F.at[i, i+1].set(-0.3)
        F = F.at[i+1, i].set(-0.3)
    S = jnp.eye(N)
    indsList = [jnp.array(list(range(nc))), jnp.array(list(range(N - nc, N)))]
    return surfG(F, S, indsList, eta=1e-3)


def make_sigma_calc(g):
    """Wrap a surfG into a SigmaCalculator."""
    from gauNEGF.transport import SigmaCalculator
    return SigmaCalculator(g)


def density_dos_reference(E, g):
    """Compute DOS using the reference formula from density._compute_dos_at_energy."""
    from gauNEGF.density import _compute_dos_at_energy
    sigma_tot = jnp.asarray(g.sigmaTot(E))
    Q_tot = g.crossTermQTot(E)
    Q_jax = jnp.asarray(Q_tot) if Q_tot is not None else None
    return float(_compute_dos_at_energy(E, g.F, g.S, sigma_tot, Q_jax))


# ---------------------------------------------------------------------------
# RED tests: these should FAIL before the fix
# ---------------------------------------------------------------------------

def test_dos_nonortho_matches_density_reference():
    """dos_single_energy with non-orthogonal contacts must match density reference.

    Currently fails because _dos_kernel uses Tr(G^R) instead of
    Tr(G^R @ (S - Q_tot)).
    """
    g = make_nonortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    total_dos, _ = dos_single_energy_r(E, g, sigma_calc)
    ref_dos = density_dos_reference(E, g)

    assert abs(total_dos - ref_dos) < 1e-8, (
        f"transport DOS {total_dos:.8f} does not match density reference {ref_dos:.8f} "
        f"(diff={abs(total_dos - ref_dos):.2e}). "
        f"Missing S-weighting or Q cross-term correction."
    )


def test_dos_nonortho_q_correction_changes_result():
    """DOS with Q correction must differ from naive Tr(G^R) for non-orthogonal system.

    Currently fails because transport.py uses the naive formula and the test
    asserts the corrected result differs from it.
    """
    from gauNEGF.utils import inv
    from gauNEGF.config import ETA
    g = make_nonortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    # Naive (wrong) DOS: Tr(G^R) / (-pi)
    sigma_tot = jnp.asarray(g.sigmaTot(E))
    mat = (E + 1j * ETA) * g.S - g.F - sigma_tot
    Gr = inv(mat)
    naive_dos = float(-jnp.imag(jnp.trace(Gr)) / jnp.pi)

    # Corrected DOS via reference
    ref_dos = density_dos_reference(E, g)

    assert abs(ref_dos - naive_dos) > 1e-6, (
        f"Q correction had no effect: ref_dos={ref_dos:.8f}, naive_dos={naive_dos:.8f}. "
        f"The non-orthogonal system should show a measurable DOS difference."
    )

    # transport dos_single_energy should match reference, not naive
    total_dos, _ = dos_single_energy_r(E, g, sigma_calc)
    assert abs(total_dos - ref_dos) < abs(total_dos - naive_dos), (
        f"transport DOS {total_dos:.8f} is closer to naive {naive_dos:.8f} than "
        f"to reference {ref_dos:.8f}. Q correction is not applied."
    )


# ---------------------------------------------------------------------------
# GREEN guard tests: must pass before and after the fix
# ---------------------------------------------------------------------------

def test_dos_ortho_matches_density_reference():
    """dos_single_energy with orthogonal contacts matches density reference.

    For orthogonal system: S=I, Q=None, so both formulas give Tr(G^R).
    This should pass before and after the fix (regression guard).
    """
    g = make_ortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    total_dos, _ = dos_single_energy_r(E, g, sigma_calc)
    ref_dos = density_dos_reference(E, g)

    assert abs(total_dos - ref_dos) < 1e-8, (
        f"transport DOS {total_dos:.8f} does not match density reference {ref_dos:.8f} "
        f"for orthogonal system (regression)."
    )


def test_dos_ortho_q_is_zero():
    """For orthogonal system, Q cross-term is None -- no correction applied."""
    g = make_ortho_chain()
    assert g.crossTermQTot(-1.0) is None, (
        "Orthogonal system should have crossTermQTot = None"
    )


def test_dos_nonortho_q_is_nonzero():
    """For non-orthogonal system, Q cross-term is nonzero."""
    g = make_nonortho_chain()
    Q = g.crossTermQTot(-1.0)
    assert Q is not None, "Non-orthogonal system should have non-None crossTermQTot"
    assert np.any(np.abs(np.array(Q)) > 1e-10), (
        "crossTermQTot should be nonzero for non-orthogonal contacts"
    )


# ---------------------------------------------------------------------------
# Spin='u' DOS tests
# ---------------------------------------------------------------------------

def make_nonortho_unrestricted(N=4, nc=2):
    """Non-orthogonal chain with F/S expanded to 2N x 2N for spin='u'."""
    g = make_nonortho_chain(N=N, nc=nc)
    F2 = np.kron(np.eye(2), np.array(g.F))
    S2 = np.kron(np.eye(2), np.array(g.S))
    return g, jnp.array(F2), jnp.array(S2)


def test_dos_unrestricted_orthogonal_total_equals_2x_restricted():
    """For orthogonal spin='u', total DOS == 2 * spin='r' DOS (both spins equal)."""
    from gauNEGF.transport import dos_single_energy
    g = make_ortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    total_r, _ = dos_single_energy(E, g.F, g.S, sigma_calc, spin='r')
    g2, F2, S2 = make_ortho_unrestricted()
    total_u, _, up, dn = dos_single_energy(E, F2, S2, sigma_calc, spin='u')

    assert abs(total_u - 2 * total_r) < 1e-6, (
        f"Unrestricted total DOS {total_u:.6f} should equal 2x restricted {2*total_r:.6f}"
    )
    np.testing.assert_allclose(up, dn, atol=1e-10,
        err_msg="Up and down DOS should be equal for symmetric system")


def test_dos_unrestricted_nonortho_matches_2x_restricted():
    """For non-orthogonal spin='u', total DOS == 2 * spin='r' DOS (degenerate spins)."""
    from gauNEGF.transport import dos_single_energy
    g = make_nonortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    total_r, _ = dos_single_energy(E, g.F, g.S, sigma_calc, spin='r')
    g, F2, S2 = make_nonortho_unrestricted()
    total_u, _, up, dn = dos_single_energy(E, F2, S2, sigma_calc, spin='u')

    assert abs(total_u - 2 * total_r) < 1e-6, (
        f"Unrestricted total DOS {total_u:.6f} should equal 2x restricted {2*total_r:.6f}"
    )


def make_ortho_unrestricted(N=4, nc=2):
    """Orthogonal chain with F/S expanded to 2N x 2N for spin='u'."""
    g = make_ortho_chain(N=N, nc=nc)
    F2 = np.kron(np.eye(2), np.array(g.F))
    S2 = np.kron(np.eye(2), np.array(g.S))
    return g, jnp.array(F2), jnp.array(S2)


# ---------------------------------------------------------------------------
# Spin='g' (generalized open shell) DOS tests
# ---------------------------------------------------------------------------

def make_ortho_generalized(N=4, nc=2):
    """Orthogonal chain in spinor form for spin='g'.

    Spinor expansion: F_g = kron(F, I2), S_g = kron(S, I2).
    Alpha at even indices, beta at odd indices.
    """
    g = make_ortho_chain(N=N, nc=nc)
    F_g = np.kron(np.array(g.F), np.eye(2))
    S_g = np.kron(np.array(g.S), np.eye(2))
    return g, jnp.array(F_g), jnp.array(S_g)


def make_nonortho_generalized(N=4, nc=2):
    """Non-orthogonal chain in spinor form for spin='g'."""
    g = make_nonortho_chain(N=N, nc=nc)
    F_g = np.kron(np.array(g.F), np.eye(2))
    S_g = np.kron(np.array(g.S), np.eye(2))
    return g, jnp.array(F_g), jnp.array(S_g)


def test_dos_generalized_orthogonal_total_equals_2x_restricted():
    """For orthogonal spin='g', total DOS == 2 * spin='r' DOS (degenerate spinors)."""
    from gauNEGF.transport import dos_single_energy
    g = make_ortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    total_r, _ = dos_single_energy(E, g.F, g.S, sigma_calc, spin='r')
    _, F_g, S_g = make_ortho_generalized()
    total_g, _, alpha, beta = dos_single_energy(E, F_g, S_g, sigma_calc, spin='g')

    assert abs(total_g - 2 * total_r) < 1e-6, (
        f"Generalized total DOS {total_g:.6f} should equal 2x restricted {2*total_r:.6f}"
    )
    np.testing.assert_allclose(alpha, beta, atol=1e-10,
        err_msg="Alpha and beta DOS should be equal for spin-degenerate system")


def test_dos_generalized_nonortho_matches_2x_restricted():
    """For non-orthogonal spin='g', total DOS == 2 * spin='r' DOS (Q correction included).

    The spinor expansion kron(Q_orbital, I2) must be applied correctly so that
    the generalized case gives exactly twice the restricted result for a spin-degenerate
    (no SOC) Hamiltonian.
    """
    from gauNEGF.transport import dos_single_energy
    g = make_nonortho_chain()
    sigma_calc = make_sigma_calc(g)
    E = -1.0

    total_r, _ = dos_single_energy(E, g.F, g.S, sigma_calc, spin='r')
    _, F_g, S_g = make_nonortho_generalized()
    total_g, _, alpha, beta = dos_single_energy(E, F_g, S_g, sigma_calc, spin='g')

    assert abs(total_g - 2 * total_r) < 1e-6, (
        f"Generalized total DOS {total_g:.6f} should equal 2x restricted {2*total_r:.6f}"
    )


def test_dos_generalized_q_expansion_is_spinor_form():
    """get_Q_tot with spin='g' returns kron(Q, I2) expansion.

    The spinor-form Q must match kron(Q_orbital, I2), not kron(I2, Q_orbital)
    (that would be the 'u' expansion).
    """
    from gauNEGF.transport import SigmaCalculator
    g = make_nonortho_chain()
    sigma_calc = SigmaCalculator(g)
    E = -1.0
    matrix_size = 2 * g.F.shape[0]

    Q_g = sigma_calc.get_Q_tot(E, spin='g', matrix_size=matrix_size)
    Q_r = sigma_calc.get_Q_tot(E, spin='r', matrix_size=g.F.shape[0])

    assert Q_g is not None, "Non-orthogonal system should have non-None Q for spin='g'"
    assert Q_g.shape == (matrix_size, matrix_size), \
        f"Q_g shape {Q_g.shape} should be ({matrix_size}, {matrix_size})"

    expected = np.kron(np.array(Q_r), np.eye(2))
    np.testing.assert_allclose(np.array(Q_g), expected, atol=1e-14,
        err_msg="spin='g' Q should be kron(Q_orbital, I2)")


# ---------------------------------------------------------------------------
# Helper: call dos_single_energy for spin='r', return (total, per_site)
# ---------------------------------------------------------------------------

def dos_single_energy_r(E, g, sigma_calc):
    from gauNEGF.transport import dos_single_energy
    return dos_single_energy(E, g.F, g.S, sigma_calc, spin='r')
