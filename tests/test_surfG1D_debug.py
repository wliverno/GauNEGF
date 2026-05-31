"""
Test surfG1D with a simple 1D tight-binding chain.

Model: 4-site chain with 1 orbital per site, nearest-neighbor hopping + overlap.
    [C1] - [D1] - [D2] - [C2]
      0      1      2      3

Parameters:
    eps = -0.5   (onsite energy)
    t   = -1.0   (hopping integral)
    s   =  0.1   (hopping overlap)
"""

import jax.numpy as jnp
import numpy as np

# Model parameters
eps = -0.5
t = -1.0
s = 0.1

# Build 4x4 Fock and Overlap matrices
F = jnp.array([
    [eps,  t,  0,  0],
    [ t, eps,  t,  0],
    [ 0,   t, eps,  t],
    [ 0,   0,  t, eps]
])

S = jnp.array([
    [1,  s,  0,  0],
    [ s, 1,  s,  0],
    [ 0,  s, 1,  s],
    [ 0,  0,  s, 1]
])

# Contact indices
indsList = [[0], [3]]
# Connection indices (device atoms connecting to contacts)
tauInds = [[1], [2]]

def analytic_g(E, eps, t, s, eta=1e-9):
    """Analytic surface Green's function for 1-orbital 1D chain."""
    z = E + 1j * eta
    a = z * 1.0 - eps  # z*S_onsite - H_onsite (S_onsite=1)
    b = z * s - t       # z*S_hop - H_hop

    # Quadratic: b^2 * g^2 - a*g + 1 = 0
    # g = (a +/- sqrt(a^2 - 4*b^2)) / (2*b^2)
    disc = a**2 - 4 * b**2
    sqrt_disc = np.sqrt(disc + 0j)

    g1 = (a + sqrt_disc) / (2 * b**2)
    g2 = (a - sqrt_disc) / (2 * b**2)

    # Retarded: pick root with Im(g) < 0 (for broadening eta > 0)
    if np.imag(g1) < np.imag(g2):
        return g1
    return g2


def test_pattern_a():
    """Pattern (a): fully automatic extraction from Fock matrix."""
    from gauNEGF.surfG1D import surfG

    sg = surfG(F, S, indsList, tauInds)

    # Test at band center
    E = 0.0
    g0 = sg.g(E, 0)
    g1 = sg.g(E, 1)

    # g should be 1x1 complex matrix
    assert g0.shape == (1, 1), f"Expected (1,1), got {g0.shape}"
    assert jnp.isfinite(g0).all(), "g contains non-finite values"

    # Imaginary part should be negative (retarded)
    assert float(jnp.imag(g0[0, 0])) < 0, f"Im(g) should be < 0, got {float(jnp.imag(g0[0,0]))}"

    # Compare against analytic solution
    g_exact = analytic_g(E, eps, t, s)
    np.testing.assert_allclose(complex(g0[0, 0]), g_exact, rtol=1e-4,
                               err_msg="g(E=0) doesn't match analytic solution")

    # Test sigma
    sig0 = sg.sigma(E, 0)
    sig1 = sg.sigma(E, 1)

    assert sig0.shape == F.shape, f"Sigma shape mismatch: {sig0.shape} vs {F.shape}"

    # Sigma should be nonzero only at contact indices
    sig0_block = sig0[0, 0]
    assert abs(complex(sig0_block)) > 1e-12, "Sigma should be nonzero at contact index"

    # Off-contact entries should be zero
    assert float(jnp.abs(sig0[1, 1])) < 1e-12, "Sigma should be zero outside contact"
    assert float(jnp.abs(sig0[2, 2])) < 1e-12, "Sigma should be zero outside contact"

    # Im(sigma) < 0 for retarded self-energy in band
    assert float(jnp.imag(sig0_block)) < 0, f"Im(Sigma) should be < 0, got {float(jnp.imag(sig0_block))}"

    # sigmaTot should be sum of both contacts
    sigTot = sg.sigmaTot(E)
    np.testing.assert_allclose(np.array(sigTot), np.array(sig0 + sig1), atol=1e-12,
                               err_msg="sigmaTot != sigma(0) + sigma(1)")

    print("Pattern (a) PASSED")
    return sg


def test_pattern_b():
    """Pattern (b): Fock matrix with custom coupling, staus=None (orthonormal coupling)."""
    from gauNEGF.surfG1D import surfG

    # Manual coupling matrices (no overlap)
    tau0 = jnp.array([[t]])  # coupling from device to contact 0
    tau1 = jnp.array([[t]])  # coupling from device to contact 1

    sg = surfG(F, S, indsList, [tau0, tau1], staus=None)

    # Verify stauList contains None (sentinel for de-orthonormalization)
    assert sg.stauList[0] is None, f"stauList[0] should be None, got {sg.stauList[0]}"
    assert sg.stauList[1] is None, f"stauList[1] should be None, got {sg.stauList[1]}"

    E = 0.0
    g0 = sg.g(E, 0)

    assert g0.shape == (1, 1), f"Expected (1,1), got {g0.shape}"
    assert float(jnp.imag(g0[0, 0])) < 0, f"Im(g) should be < 0, got {float(jnp.imag(g0[0,0]))}"

    sig0 = sg.sigma(E, 0)
    sig0_block = sig0[0, 0]
    assert abs(complex(sig0_block)) > 1e-12, "Sigma should be nonzero"
    assert float(jnp.imag(sig0_block)) < 0, f"Im(Sigma) should be < 0, got {float(jnp.imag(sig0_block))}"

    print("Pattern (b) PASSED")
    return sg


def test_pattern_c():
    """Pattern (c): fully specified contacts."""
    from gauNEGF.surfG1D import surfG

    # Manual everything
    alpha0 = jnp.array([[eps]])
    alpha1 = jnp.array([[eps]])
    aOverlap0 = jnp.array([[1.0]])
    aOverlap1 = jnp.array([[1.0]])
    beta0 = jnp.array([[t]])
    beta1 = jnp.array([[t]])
    bOverlap0 = jnp.array([[s]])
    bOverlap1 = jnp.array([[s]])
    tau0 = jnp.array([[t]])
    tau1 = jnp.array([[t]])
    stau0 = jnp.array([[s]])
    stau1 = jnp.array([[s]])

    sg = surfG(F, S, indsList,
               [tau0, tau1], [stau0, stau1],
               [alpha0, alpha1], [aOverlap0, aOverlap1],
               [beta0, beta1], [bOverlap0, bOverlap1])

    E = 0.0
    g0 = sg.g(E, 0)

    # Compare against analytic
    g_exact = analytic_g(E, eps, t, s)
    np.testing.assert_allclose(complex(g0[0, 0]), g_exact, rtol=1e-4,
                               err_msg="Pattern (c) g doesn't match analytic")

    sig0 = sg.sigma(E, 0)
    sig0_block = sig0[0, 0]
    assert float(jnp.imag(sig0_block)) < 0, f"Im(Sigma) should be < 0, got {float(jnp.imag(sig0_block))}"

    print("Pattern (c) PASSED")
    return sg


def test_pattern_a_vs_c_consistency():
    """Patterns (a) and (c) with same parameters should give same results."""
    from gauNEGF.surfG1D import surfG

    # Pattern (a)
    sg_a = surfG(F, S, indsList, tauInds)

    # Pattern (c) with same parameters as what (a) extracts
    tau0 = F[jnp.ix_(jnp.array(tauInds[0]), jnp.array(indsList[0]))]
    tau1 = F[jnp.ix_(jnp.array(tauInds[1]), jnp.array(indsList[1]))]
    stau0 = S[jnp.ix_(jnp.array(tauInds[0]), jnp.array(indsList[0]))]
    stau1 = S[jnp.ix_(jnp.array(tauInds[1]), jnp.array(indsList[1]))]
    alpha0 = F[jnp.ix_(jnp.array(indsList[0]), jnp.array(indsList[0]))]
    alpha1 = F[jnp.ix_(jnp.array(indsList[1]), jnp.array(indsList[1]))]
    aOverlap0 = S[jnp.ix_(jnp.array(indsList[0]), jnp.array(indsList[0]))]
    aOverlap1 = S[jnp.ix_(jnp.array(indsList[1]), jnp.array(indsList[1]))]

    sg_c = surfG(F, S, indsList,
                 [tau0, tau1], [stau0, stau1],
                 [alpha0, alpha1], [aOverlap0, aOverlap1],
                 [tau0, tau1], [stau0, stau1])

    E = 0.0
    g_a = sg_a.g(E, 0)
    g_c = sg_c.g(E, 0)

    np.testing.assert_allclose(np.array(g_a), np.array(g_c), rtol=1e-6,
                               err_msg="Pattern (a) and (c) give different g")

    sig_a = sg_a.sigma(E, 0)
    sig_c = sg_c.sigma(E, 0)

    np.testing.assert_allclose(np.array(sig_a), np.array(sig_c), rtol=1e-6,
                               err_msg="Pattern (a) and (c) give different sigma")

    print("Pattern (a) vs (c) consistency PASSED")


def test_energy_sweep():
    """Test g and sigma across an energy range spanning the band."""
    from gauNEGF.surfG1D import surfG

    sg = surfG(F, S, indsList, tauInds)

    energies = np.linspace(-4, 4, 50)

    for E in energies:
        g0 = sg.g(float(E), 0)
        sig0 = sg.sigma(float(E), 0)

        # g should always be finite
        assert jnp.isfinite(g0).all(), f"g not finite at E={E}"

        # Im(g) <= 0 for retarded (can be ~0 outside band)
        assert float(jnp.imag(g0[0, 0])) <= 1e-10, f"Im(g) > 0 at E={E}"

        # Gamma = -2*Im(Sigma) should be >= 0 (broadening)
        gamma = -2 * float(jnp.imag(sig0[0, 0]))
        assert gamma >= -1e-10, f"Gamma < 0 at E={E}: {gamma}"

    print("Energy sweep PASSED")


def test_setF():
    """Test that setF correctly updates the Fock matrix."""
    from gauNEGF.surfG1D import surfG

    E = 0.0
    F_new = F + 0.5 * jnp.eye(4)

    sg = surfG(F, S, indsList, tauInds)
    sig_before = sg.sigma(E, 0)

    sg.setF(F_new)
    sig_after = sg.sigma(E, 0)

    # Sigma should change
    diff = float(jnp.max(jnp.abs(sig_after - sig_before)))
    assert diff > 1e-6, f"Sigma should change after setF, diff={diff}"

    # Result should match a fresh surfG built with F_new
    sg_fresh = surfG(F_new, S, indsList, tauInds)
    sig_fresh = sg_fresh.sigma(E, 0)
    np.testing.assert_allclose(np.array(sig_after), np.array(sig_fresh), rtol=1e-4,
                               err_msg="setF result doesn't match fresh surfG(F_new)")

    print("setF PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing surfG1D with tight-binding model")
    print("=" * 60)

    test_pattern_a()
    test_pattern_b()
    test_pattern_c()
    test_pattern_a_vs_c_consistency()
    test_energy_sweep()
    test_setF()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
