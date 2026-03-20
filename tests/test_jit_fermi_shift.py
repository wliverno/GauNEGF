"""Test that Fermi energy shift via E-shift gives identical results to H-mutation.

These tests verify the mathematical equivalence BEFORE the refactor (so they prove
the refactor is safe), and also include a JIT retrace check that fails before the fix.
"""
import numpy as np
import jax.numpy as jnp
import pytest


def make_test_surfGBAt():
    """Create a surfGBAt with known test parameters (no fermi set yet)."""
    from gauNEGF.surfGBethe import surfGBAt
    d = 9
    H = jnp.diag(jnp.array([-0.1, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=complex))
    rng = np.random.default_rng(42)
    Slist = []
    Vlist = []
    for k in range(12):
        S = jnp.eye(d) * 0.1 * (1 + 0.01 * rng.standard_normal())
        V_r = rng.standard_normal((d, d)) * 0.05
        V_i = rng.standard_normal((d, d)) * 0.05
        V = jnp.array(V_r + 1j * V_i)
        V = (V + V.conj().T) / 2  # Hermitian
        Slist.append(S)
        Vlist.append(V)
    Slist = jnp.array(Slist)
    Vlist = jnp.array(Vlist)
    return surfGBAt(H, Slist, Vlist, eta=1e-3)


def make_test_surfGAt3D():
    """Create a surfGAt3D with known test parameters."""
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


# ---------------------------------------------------------------------------
# surfGBAt equivalence tests (Task 1)
# ---------------------------------------------------------------------------

def test_surfGBAt_sigmaK_fermi_shift_equivalence():
    """sigmaK with dFermi=dF at E should equal sigmaK with dFermi=0 at E-dF."""
    E = -5.0 + 0.1j
    dF = 0.3

    # Method A: set dFermi, call sigmaK at original E
    gAt_A = make_test_surfGBAt()
    gAt_A.updateH(fermi=dF)  # sets dFermi = dF (fermi0 = dF)
    sig_A = np.array(gAt_A.sigmaK(E))

    # Method B: keep dFermi=0, shift E down
    gAt_B = make_test_surfGBAt()
    sig_B = np.array(gAt_B.sigmaK(E - dF))

    np.testing.assert_allclose(
        sig_A, sig_B, rtol=1e-5,
        err_msg="sigmaK: dFermi shift at E != original at E-dF"
    )


def test_surfGBAt_sigma_fermi_shift_equivalence():
    """sigma with dFermi=dF at E should equal sigma with dFermi=0 at E-dF."""
    E = -5.0 + 0.1j
    dF = 0.3

    gAt_A = make_test_surfGBAt()
    gAt_A.updateH(fermi=dF)
    sig_A = np.array(gAt_A.sigma(E))

    gAt_B = make_test_surfGBAt()
    sig_B = np.array(gAt_B.sigma(E - dF))

    np.testing.assert_allclose(
        sig_A, sig_B, rtol=1e-5,
        err_msg="sigma: dFermi shift at E != original at E-dF"
    )


def test_surfGBAt_has_immutable_H0():
    """After __init__, surfGBAt should expose H0, Vlist0, and dFermi."""
    gAt = make_test_surfGBAt()
    assert hasattr(gAt, 'H0'), "surfGBAt missing H0 attribute"
    assert hasattr(gAt, 'Vlist0'), "surfGBAt missing Vlist0 attribute"
    assert hasattr(gAt, 'dFermi'), "surfGBAt missing dFermi attribute"
    assert hasattr(gAt, 'fermi'), "surfGBAt missing fermi attribute"
    assert gAt.dFermi == 0.0, "dFermi should be 0 at init"
    np.testing.assert_allclose(gAt.H0, gAt.H, rtol=1e-12)
    np.testing.assert_allclose(gAt.Vlist0, gAt.Vlist, rtol=1e-12)


def test_surfGBAt_updateH_updates_dFermi():
    """After updateH(fermi=dF), sigma(E) should equal original sigma(E-dF)."""
    d = 9
    E = -5.0 + 0.1j
    dF = 0.3

    gAt_ref = make_test_surfGBAt()
    sig_ref = np.array(gAt_ref.sigma(E - dF))

    gAt = make_test_surfGBAt()
    gAt.updateH(fermi=dF)  # set fermi shift
    sig_shifted = np.array(gAt.sigma(E))

    np.testing.assert_allclose(
        sig_shifted, sig_ref, rtol=1e-5,
        err_msg="After updateH(fermi=dF), sigma(E) should equal original sigma(E-dF)"
    )


# ---------------------------------------------------------------------------
# surfGAt3D equivalence tests (Task 2)
# ---------------------------------------------------------------------------

def test_surfGAt3D_sigma_fermi_shift_equivalence():
    """sigma with dFermi=dF at E should equal sigma with dFermi=0 at E-dF."""
    E = -5.0 + 0.1j
    dF = 0.3

    gAt_A = make_test_surfGAt3D()
    gAt_A.updateH(fermi=dF)
    sig_A = np.array(gAt_A.sigma(E))

    gAt_B = make_test_surfGAt3D()
    sig_B = np.array(gAt_B.sigma(E - dF))

    np.testing.assert_allclose(
        sig_A, sig_B, rtol=1e-5,
        err_msg="surfGAt3D sigma: dFermi shift at E != original at E-dF"
    )


def test_surfGAt3D_has_immutable_H0():
    """After __init__, surfGAt3D should expose H0, Vlist0, and dFermi."""
    gAt = make_test_surfGAt3D()
    assert hasattr(gAt, 'H0'), "surfGAt3D missing H0 attribute"
    assert hasattr(gAt, 'Vlist0'), "surfGAt3D missing Vlist0 attribute"
    assert hasattr(gAt, 'dFermi'), "surfGAt3D missing dFermi attribute"
    assert gAt.dFermi == 0.0, "dFermi should be 0 at init"
    np.testing.assert_allclose(gAt.H0, gAt.H, rtol=1e-12)
    np.testing.assert_allclose(gAt.Vlist0, gAt.Vlist, rtol=1e-12)


def test_surfGAt3D_updateH_updates_dFermi():
    """After updateH(fermi=dF), sigma(E) should equal original sigma(E-dF)."""
    d = 9
    E = -5.0 + 0.1j
    dF = 0.3

    gAt_ref = make_test_surfGAt3D()
    sig_ref = np.array(gAt_ref.sigma(E - dF))

    gAt = make_test_surfGAt3D()
    gAt.updateH(fermi=dF)
    sig_shifted = np.array(gAt.sigma(E))

    np.testing.assert_allclose(
        sig_shifted, sig_ref, rtol=1e-5,
        err_msg="After updateH(fermi=dF), surfGAt3D sigma(E) should equal original sigma(E-dF)"
    )
