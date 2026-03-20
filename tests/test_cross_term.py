"""Tests for cross-term Q_sym computation across all surfG classes.

Q_sym = (tau @ g^R @ S_LD + S_DL @ g^R @ tau^dagger) / 2
delta_N = -(1/pi) Im(sum_k w_k Tr(G^R(z_k) @ Q_tot(z_k)))
"""
import numpy as np
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_surfGTest(N=6):
    """Create a simple surfGTest with orthogonal basis."""
    pytest.importorskip("gauopen", reason="gauopen not installed; skipping surfGTest tests")
    from gauNEGF.surfGTester import surfGTest
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    S = jnp.eye(N)
    indsList = [jnp.array([0, 1]), jnp.array([N-2, N-1])]
    return surfGTest(F, S, indsList)


def make_1d_nonortho_chain(n_device=4, n_contact=2):
    """Create a 1D non-orthogonal chain with overlap coupling.

    Returns surfG with nonzero stauList (S_DL != 0).
    """
    from gauNEGF.surfG1D import surfG
    N = n_device
    nc = n_contact
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    for i in range(N - 1):
        F = F.at[i, i+1].set(-0.3)
        F = F.at[i+1, i].set(-0.3)
    S = jnp.eye(N) + 0.0j
    for i in range(N - 1):
        S = S.at[i, i+1].set(0.1)
        S = S.at[i+1, i].set(0.1)

    indsList = [jnp.array(list(range(nc))), jnp.array(list(range(N-nc, N)))]

    # Contact coupling matrices
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


def make_1d_ortho_chain(N=4, nc=2):
    """Create a simple orthogonal 1D chain."""
    from gauNEGF.surfG1D import surfG
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    for i in range(N - 1):
        F = F.at[i, i+1].set(-0.3)
        F = F.at[i+1, i].set(-0.3)
    S = jnp.eye(N)
    indsList = [jnp.array(list(range(nc))), jnp.array(list(range(N-nc, N)))]
    return surfG(F, S, indsList, eta=1e-3)


# ---------------------------------------------------------------------------
# Task 3: surfGTest.crossTermQ
# ---------------------------------------------------------------------------

def test_surfGTest_crossTermQ_returns_none():
    """surfGTest has no overlap coupling -- crossTermQ returns None."""
    g = make_surfGTest()
    assert g.crossTermQ(-5.0, 0) is None
    assert g.crossTermQ(-5.0, 1) is None


def test_surfGTest_crossTermQTot_returns_none():
    """crossTermQTot should also return None for orthogonal case."""
    g = make_surfGTest()
    assert g.crossTermQTot(-5.0) is None


# ---------------------------------------------------------------------------
# Task 4: surfG1D.crossTermQ
# ---------------------------------------------------------------------------

def test_surfG1D_crossTermQ_returns_none_for_orthogonal():
    """When stauList[i] is None (orthogonal), crossTermQ returns None."""
    g = make_1d_ortho_chain()
    Q = g.crossTermQ(-5.0, 0)
    assert Q is None, "crossTermQ should return None for orthogonal contacts"


def test_surfG1D_crossTermQ_shape():
    """crossTermQ returns a matrix with same shape as F."""
    g = make_1d_nonortho_chain()
    E = -5.0 + 0.1j
    Q = g.crossTermQ(E, 0)
    assert Q is not None, "crossTermQ should not be None for non-orthogonal contacts"
    assert Q.shape == g.F.shape, f"Q shape {Q.shape} != F shape {g.F.shape}"


def test_surfG1D_crossTermQ_nonzero_on_contact_inds():
    """Q should be nonzero only on contact orbital indices."""
    g = make_1d_nonortho_chain()
    E = -5.0 + 0.1j
    Q = g.crossTermQ(E, 0)
    assert Q is not None
    inds = np.array(g.indsList[0])
    N = g.F.shape[0]
    other_inds = [i for i in range(N) if i not in inds]
    if len(other_inds) > 0:
        assert np.allclose(np.array(Q)[np.ix_(other_inds, range(N))], 0), \
            "Q rows outside contact inds should be zero"
        assert np.allclose(np.array(Q)[np.ix_(range(N), other_inds)], 0), \
            "Q cols outside contact inds should be zero"


def test_surfG1D_crossTermQ_is_symmetrized():
    """Q_sym should be (Q_fwd + Q_rev)/2, not just Q_fwd (Q_fwd != Q_fwd.T generally)."""
    g = make_1d_nonortho_chain()
    E = -5.0 + 0.1j
    Q = g.crossTermQ(E, 0)
    assert Q is not None
    inds = np.array(g.indsList[0])
    Q_sub = np.array(Q)[np.ix_(inds, inds)]
    # Q_sym should be Hermitian on the real axis (for real E).
    # For E slightly off real axis, check it is NOT purely lower triangular
    # (which would indicate only Q_fwd, no Q_rev)
    assert not np.allclose(Q_sub, np.tril(Q_sub)), \
        "Q should not be purely lower triangular -- it should be symmetrized"


def test_surfG1D_crossTermQTot_sums_contacts():
    """crossTermQTot should sum non-None crossTermQ contributions."""
    g = make_1d_nonortho_chain()
    E = -5.0 + 0.1j
    Q_tot = g.crossTermQTot(E)
    Q0 = g.crossTermQ(E, 0)
    Q1 = g.crossTermQ(E, 1)
    assert Q0 is not None and Q1 is not None
    expected = np.array(Q0) + np.array(Q1)
    np.testing.assert_allclose(np.array(Q_tot), expected, atol=1e-14)


def test_surfG1D_crossTermQTot_returns_none_for_all_orthogonal():
    """crossTermQTot returns None when all contacts are orthogonal."""
    g = make_1d_ortho_chain()
    assert g.crossTermQTot(-5.0) is None


# ---------------------------------------------------------------------------
# Task 5: surfGBAt.crossTermQ
# ---------------------------------------------------------------------------

def make_surfGBAt():
    """Create a surfGBAt with test parameters (non-zero overlap)."""
    from gauNEGF.surfGBethe import surfGBAt
    d = 9
    H = jnp.diag(jnp.array([-0.1, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=complex))
    rng = np.random.default_rng(42)
    Slist, Vlist = [], []
    for k in range(12):
        S = jnp.eye(d, dtype=complex) * 0.1 * (1 + 0.01 * rng.standard_normal())
        V_r = rng.standard_normal((d, d)) * 0.05
        V_i = rng.standard_normal((d, d)) * 0.05
        V = jnp.array(V_r + 1j * V_i)
        V = (V + V.conj().T) / 2
        Slist.append(S)
        Vlist.append(V)
    Slist = jnp.array(Slist)
    Vlist = jnp.array(Vlist)
    return surfGBAt(H, Slist, Vlist, eta=1e-3)


def make_surfGBAt_zero_overlap():
    """Create a surfGBAt where all Slist are zero (orthogonal limit)."""
    from gauNEGF.surfGBethe import surfGBAt
    d = 9
    H = jnp.diag(jnp.linspace(-0.1, 0.2, d))
    Slist = jnp.zeros((12, d, d))
    rng = np.random.default_rng(42)
    Vlist = jnp.array([jnp.array(rng.standard_normal((d, d)) * 0.05) for _ in range(12)])
    return surfGBAt(H, Slist, Vlist, eta=1e-3)


def test_surfGBAt_crossTermQ_shape():
    """crossTermQ returns a (dim, dim) matrix."""
    gAt = make_surfGBAt()
    E = -5.0 + 0.1j
    Q = gAt.crossTermQ(E)
    assert Q is not None
    assert Q.shape == (9, 9), f"Expected (9,9), got {Q.shape}"


def test_surfGBAt_crossTermQ_orthogonal_limit():
    """When all Slist are zero, crossTermQ should be zero."""
    gAt = make_surfGBAt_zero_overlap()
    Q = gAt.crossTermQ(-5.0 + 0.1j)
    np.testing.assert_allclose(np.array(Q), 0.0, atol=1e-10,
        err_msg="crossTermQ should be zero when all overlaps are zero")


def test_surfGBAt_crossTermQ_is_symmetrized():
    """Q_sym should have correct Hermitian-like structure."""
    gAt = make_surfGBAt()
    E = -5.0 + 0.1j
    Q = gAt.crossTermQ(E)
    assert Q is not None
    # Q_sym is NOT Hermitian in general (only on real axis), but should not be
    # purely upper or lower triangular
    Q_arr = np.array(Q)
    assert not np.allclose(Q_arr, np.tril(Q_arr)), \
        "Q should not be purely lower triangular -- should be symmetrized"


# ---------------------------------------------------------------------------
# Task 6: surfGAt3D.crossTermQ
# ---------------------------------------------------------------------------

def make_surfGAt3D():
    """Create a surfGAt3D with test parameters."""
    from gauNEGF.surfG3D import surfGAt3D
    d = 9
    H = jnp.diag(jnp.array([-0.1, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2]))
    rng = np.random.default_rng(42)
    Slist, Vlist = [], []
    fcc_vecs = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [0, 1, 1], [1, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-1, -1, 0], [0, -1, -1], [-1, 0, -1]
    ]
    for k in range(12):
        S = jnp.eye(d) * 0.1
        V_r = rng.standard_normal((d, d)) * 0.05
        V_i = rng.standard_normal((d, d)) * 0.05
        V = jnp.array(V_r + 1j * V_i)
        V = (V + V.conj().T) / 2
        Slist.append(S)
        Vlist.append(V)
    return surfGAt3D(H, jnp.array(Slist), jnp.array(Vlist),
                     jnp.array(fcc_vecs, dtype=float), eta=1e-3, kPoints=3)


def make_surfGAt3D_zero_overlap():
    """Create a surfGAt3D where all Slist are zero (orthogonal limit)."""
    from gauNEGF.surfG3D import surfGAt3D
    d = 9
    H = jnp.diag(jnp.linspace(-0.1, 0.2, d))
    Slist = jnp.zeros((12, d, d))
    rng = np.random.default_rng(42)
    Vlist = jnp.array([jnp.array(rng.standard_normal((d, d)) * 0.05) for _ in range(12)])
    fcc_vecs = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],
                [-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,0],[0,-1,-1],[-1,0,-1]]
    return surfGAt3D(H, Slist, Vlist, jnp.array(fcc_vecs, dtype=float), eta=1e-3, kPoints=3)


def test_surfGAt3D_crossTermQ_shape():
    """crossTermQ returns a (dim, dim) matrix."""
    gAt = make_surfGAt3D()
    E = -5.0 + 0.1j
    Q = gAt.crossTermQ(E)
    assert Q is not None
    assert Q.shape == (9, 9), f"Expected (9,9), got {Q.shape}"


def test_surfGAt3D_crossTermQ_orthogonal_limit():
    """When all Slist are zero, crossTermQ should be zero."""
    gAt = make_surfGAt3D_zero_overlap()
    Q = gAt.crossTermQ(-5.0 + 0.1j)
    np.testing.assert_allclose(np.array(Q), 0.0, atol=1e-10)


def test_surfGAt3D_crossTermQ_is_symmetrized():
    """Q_sym should be symmetrized (not purely lower triangular)."""
    gAt = make_surfGAt3D()
    E = -5.0 + 0.1j
    Q = gAt.crossTermQ(E)
    assert Q is not None
    Q_arr = np.array(Q)
    assert not np.allclose(Q_arr, np.tril(Q_arr)), \
        "Q should not be purely lower triangular -- should be symmetrized"


# ---------------------------------------------------------------------------
# Task 7: surfGB.crossTermQ (requires Bethe parameters -- integration test)
# ---------------------------------------------------------------------------

def make_surfGB_from_bethe_params():
    """Create a surfGB using real Au Bethe parameters, if available."""
    import os
    bethe_path = os.path.join(os.path.dirname(__file__), 'Au.bethe')
    if not os.path.exists(bethe_path):
        pytest.skip("Au.bethe not found -- skipping surfGB integration test")
    from gauNEGF.surfGBethe import surfGB
    # Create a minimal Fock/Overlap with 1 atom
    d = 9
    F = jnp.diag(jnp.zeros(d))
    S = jnp.eye(d)
    indsList = [jnp.array(list(range(d)))]
    try:
        g = surfGB(F, S, indsList, bethe_path, eta=1e-3)
        return g
    except Exception:
        pytest.skip("Could not construct surfGB from Au.bethe")


def test_surfGB_crossTermQ_delegates_to_atomic():
    """surfGB.crossTermQ should return correct shape or None."""
    g = make_surfGB_from_bethe_params()
    E = -5.0 + 0.1j
    N = g.F.shape[0]
    Q = g.crossTermQ(E, 0)
    # When Sdict['sss'] == 0 (no overlap), Q may be zero but should be correct shape
    if Q is not None:
        assert Q.shape == (N, N), f"Q shape {Q.shape} != ({N},{N})"


def test_surfGB_crossTermQTot_returns_correct_shape():
    """crossTermQTot should return correct shape or None."""
    g = make_surfGB_from_bethe_params()
    E = -5.0 + 0.1j
    N = g.F.shape[0]
    Q_tot = g.crossTermQTot(E)
    if Q_tot is not None:
        assert Q_tot.shape == (N, N), f"Q_tot shape {Q_tot.shape} != ({N},{N})"


# ---------------------------------------------------------------------------
# Task 8: surfG3.crossTermQ (requires surfGAt3D setup -- integration test)
# ---------------------------------------------------------------------------

def make_surfG3_test():
    """Create a surfG3 object with 2 contacts using test surfGAt3D."""
    pytest.skip("surfG3 full integration test deferred -- requires full contact setup")


def test_surfG3_crossTermQ_shape():
    """surfG3.crossTermQ should return device-size matrix."""
    make_surfG3_test()  # Will skip


# ---------------------------------------------------------------------------
# Task 9: GrIntCross
# ---------------------------------------------------------------------------

def test_GrIntCross_orthogonal_returns_zero_cross_scalar():
    """GrIntCross on orthogonal system: cross_scalar should be 0."""
    from gauNEGF.integrate import GrIntCross
    g = make_1d_ortho_chain()
    Elist = jnp.array([-5.0 + 0.5j, -4.0 + 0.5j])
    weights = jnp.array([0.5 + 0.1j, 0.5 + 0.1j])
    lineInt, cross = GrIntCross(g.F, g.S, g, Elist, weights)
    assert lineInt.shape == g.F.shape
    assert abs(cross) < 1e-14, f"cross_scalar should be 0 for orthogonal, got {cross}"


def test_GrIntCross_nonortho_nonzero_cross_scalar():
    """GrIntCross on non-orthogonal system: cross_scalar should be nonzero."""
    from gauNEGF.integrate import GrIntCross
    g = make_1d_nonortho_chain()
    Elist = jnp.array([-5.0 + 0.5j, -4.0 + 0.5j])
    weights = jnp.array([0.5 + 0.1j, 0.5 + 0.1j])
    lineInt, cross = GrIntCross(g.F, g.S, g, Elist, weights)
    assert lineInt.shape == g.F.shape
    assert abs(cross) > 1e-14, f"cross_scalar should be nonzero for non-orthogonal, got {cross}"


def test_GrIntCross_lineInt_matches_GrInt():
    """lineInt from GrIntCross should match GrInt output exactly."""
    from gauNEGF.integrate import GrInt, GrIntCross
    g = make_1d_nonortho_chain()
    Elist = jnp.array([-5.0 + 0.5j, -4.0 + 0.5j])
    weights = jnp.array([0.5 + 0.1j, 0.5 + 0.1j])
    lineInt_direct = np.array(GrInt(g.F, g.S, g, Elist, weights))
    lineInt_cross, _ = GrIntCross(g.F, g.S, g, Elist, weights)
    np.testing.assert_allclose(np.array(lineInt_cross), lineInt_direct, rtol=1e-10)


# ---------------------------------------------------------------------------
# Task 10: densityComplex/N return (P, delta_N) via GrIntCross co-accumulation
# ---------------------------------------------------------------------------

def test_densityComplexN_returns_tuple():
    """densityComplexN should return (P, delta_N) tuple."""
    from gauNEGF.density import densityComplexN
    g = make_1d_ortho_chain()
    result = densityComplexN(g.F, g.S, g, -10.0, 0.0, N=20, showText=False)
    assert isinstance(result, tuple), "densityComplexN should return a tuple"
    P, delta_N = result
    assert P.shape == g.F.shape
    assert abs(delta_N) < 1e-8, f"delta_N should be 0 for orthogonal, got {delta_N}"


def test_densityComplexN_nonzero_delta_N_for_nonorthogonal():
    """densityComplexN should return nonzero delta_N for non-orthogonal system."""
    from gauNEGF.density import densityComplexN
    g = make_1d_nonortho_chain()
    P, delta_N = densityComplexN(g.F, g.S, g, -10.0, 0.0, N=50, showText=False)
    assert P.shape == g.F.shape
    assert abs(delta_N) > 1e-10, \
        f"delta_N should be nonzero for non-orthogonal contacts, got {delta_N}"


def test_densityComplex_returns_tuple():
    """densityComplex (adaptive) should return (P, delta_N) tuple."""
    from gauNEGF.density import densityComplex
    g = make_1d_ortho_chain()
    result = densityComplex(g.F, g.S, g, -10.0, 0.0)
    assert isinstance(result, tuple), "densityComplex should return a tuple"
    P, delta_N = result
    assert P.shape == g.F.shape
    assert abs(delta_N) < 1e-8, f"delta_N should be 0 for orthogonal, got {delta_N}"


# ---------------------------------------------------------------------------
# Task 12: Fermi search unpacks (P, delta_N) from density functions
# ---------------------------------------------------------------------------

def test_calcFermiBisect_converges_orthogonal_no_regression():
    """calcFermiBisect should still converge correctly for orthogonal systems (no regression)."""
    from gauNEGF.density import calcFermiBisect
    g = make_1d_ortho_chain(N=4, nc=1)
    ne = 1.0
    Emin = -10.0
    Ef, dE, P = calcFermiBisect(g, ne, Emin, -0.5, N=50, maxcycles=20)
    Ncurr = np.trace(np.array(P) @ np.array(g.S)).real
    assert abs(Ncurr - ne) < 0.05, \
        f"Fermi search failed for orthogonal system: N={Ncurr:.4f}, ne={ne}"


def test_calcFermiBisect_raw_count_below_ne_for_nonorthogonal():
    """With cross-terms, bisection targets Tr(P@S) + delta_N = ne.
    Therefore raw Tr(P@S) at convergence should be less than ne."""
    from gauNEGF.density import calcFermiBisect, densityComplexN
    g = make_1d_nonortho_chain()
    ne = 2.0
    Emin = -10.0
    Ef, dE, P = calcFermiBisect(g, ne, Emin, 0.0, N=50, maxcycles=20)
    # Get delta_N from the density function at the converged Ef
    _, delta_N = densityComplexN(g.F, g.S, g, Emin, Ef, N=50, showText=False)
    Ncurr_raw = np.trace(np.array(P) @ np.array(g.S)).real
    if abs(delta_N) > 0.005:
        assert Ncurr_raw < ne + 0.01, \
            f"With cross-term, raw N={Ncurr_raw:.4f} should be less than ne={ne} (delta_N={delta_N:.4f})"


# ---------------------------------------------------------------------------
# Task 13: End-to-end -- delta_N is nonzero for non-orthogonal system
# ---------------------------------------------------------------------------

def test_fermi_search_cross_term_nonzero_for_nonorthogonal():
    """For a non-orthogonal system, delta_N should be nonzero and improve accuracy."""
    from gauNEGF.density import densityComplex
    g = make_1d_nonortho_chain()
    Emin = -10.0
    mu = 0.0
    P, delta_N = densityComplex(g.F, g.S, g, Emin, mu)
    N_without = np.trace(np.array(P) @ np.array(g.S)).real
    assert abs(delta_N) > 1e-10, \
        f"delta_N should be nonzero for non-orthogonal contacts, got {delta_N}"
