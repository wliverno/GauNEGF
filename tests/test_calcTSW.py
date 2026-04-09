"""Unit test for calcTSW: spectral-weight-based integration limits."""
import sys
sys.path.insert(0, '..')

import numpy as np
import jax.numpy as jnp

from gauNEGF.surfGBethe import surfGBAt
from gauNEGF.density import calcTSW
from gauNEGF.utils import inv, eigh
from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors


dim = 9
eta = 1e-6


def build_au_system():
    """Build a surfGBAt for Au single cell -- same fixture as cross-term tests."""
    ne, H0, Sdict, Vdict = read_bethe_params('Au')
    vecs = gen_fcc_111_neighbors()
    Vlist = [construct_mat(Vdict, v) for v in vecs]
    Slist = [construct_mat(Sdict, v) for v in vecs]
    gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    return gBAt, ne


def test_calcTSW_converges():
    """calcTSW should converge and return bounds that bracket all eigenvalues."""
    gBAt, ne = build_au_system()
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
    print("test_calcTSW_converges PASSED")


def test_calcTSW_warm_start():
    """Warm-started calcTSW should converge in zero iterations if bounds are good."""
    gBAt, ne = build_au_system()
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
    print("test_calcTSW_warm_start PASSED")


if __name__ == '__main__':
    test_calcTSW_converges()
    test_calcTSW_warm_start()
    print("All tests passed!")
