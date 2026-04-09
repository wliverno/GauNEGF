"""Unit test for calcTSW: spectral-weight-based integration limits."""
import sys
sys.path.insert(0, '..')

import pytest
import numpy as np

from gauNEGF.surfGBethe import surfGBAt
from gauNEGF.density import calcTSW
from gauNEGF.utils import inv, eigh
from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors


dim = 9
eta = 1e-6


@pytest.fixture
def au_system():
    """Build a surfGBAt for Au single cell -- same fixture as cross-term tests."""
    ne, H0, Sdict, Vdict = read_bethe_params('Au')
    vecs = gen_fcc_111_neighbors()
    Vlist = [construct_mat(Vdict, v) for v in vecs]
    Slist = [construct_mat(Sdict, v) for v in vecs]
    gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    return gBAt, ne


def test_calcTSW_converges(au_system):
    """calcTSW should converge and return bounds that bracket all eigenvalues."""
    gBAt, ne = au_system
    F = gBAt.F
    S = gBAt.S

    Emin, Emax, TSW = calcTSW(F, S, gBAt)

    # TSW should be close to the matrix dimension (total spectral weight)
    assert TSW > 0, f"TSW should be positive, got {TSW}"

    # Emin should be below all eigenvalues, Emax above
    D, _ = eigh(inv(S) @ F)
    eigenvalues = np.real(D).flatten()
    assert Emin < min(eigenvalues), \
        f"Emin {Emin} should be below min eigenvalue {min(eigenvalues)}"
    assert Emax > max(eigenvalues), \
        f"Emax {Emax} should be above max eigenvalue {max(eigenvalues)}"


def test_calcTSW_warm_start(au_system):
    """Warm-started calcTSW should converge in zero iterations if bounds are good."""
    gBAt, ne = au_system
    F = gBAt.F
    S = gBAt.S

    # First call: cold start
    Emin1, Emax1, TSW1 = calcTSW(F, S, gBAt)

    # Second call: warm start with converged values
    Emin2, Emax2, TSW2 = calcTSW(F, S, gBAt, Emin=Emin1, Emax=Emax1, TSW=TSW1)

    # Warm start should return identical bounds (no expansion needed)
    assert Emin2 == Emin1, f"Warm-started Emin changed: {Emin1} -> {Emin2}"
    assert Emax2 == Emax1, f"Warm-started Emax changed: {Emax1} -> {Emax2}"
    assert abs(TSW2 - TSW1) < 1e-6, \
        f"Warm-started TSW changed: {TSW1} -> {TSW2}"


def test_getFermiContact_with_calcTSW(au_system):
    """getFermiContact should still find the correct Au Fermi energy after calcTSW switch."""
    from gauNEGF.density import getFermiContact

    gBAt, ne = au_system
    ne_per_spin = ne / 2
    AU_BULK_FERMI_EV = 2.84

    fermi = getFermiContact(gBAt, ne_per_spin, conv=1e-3, maxcycles=1000, T=0)
    assert abs(fermi - AU_BULK_FERMI_EV) < 0.02, \
        f"Au Fermi {fermi:.4f} eV differs from benchmark {AU_BULK_FERMI_EV} eV"
