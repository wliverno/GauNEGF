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


@pytest.mark.slow
def test_calcTSW_converges(au_system):
    """calcTSW should converge and return Eminf below all eigenvalues.

    Marked slow: calcTSW search takes ~5 min on P100.
    """
    gBAt, ne = au_system
    F = gBAt.F
    S = gBAt.S

    Eminf, TSW = calcTSW(F, S, gBAt)

    # TSW should be positive (total spectral weight)
    assert TSW > 0, f"TSW should be positive, got {TSW}"

    # Eminf should be below all eigenvalues
    D, _ = eigh(inv(S) @ F)
    eigenvalues = np.real(D).flatten()
    assert Eminf < min(eigenvalues), \
        f"Eminf {Eminf} should be below min eigenvalue {min(eigenvalues)}"


@pytest.mark.slow
def test_calcTSW_warm_start(au_system):
    """Warm-started calcTSW should converge in zero iterations if bounds are good.

    Marked slow: cold-start calcTSW dominates (~7 min total on P100).
    """
    gBAt, ne = au_system
    F = gBAt.F
    S = gBAt.S

    # First call: cold start
    Eminf1, TSW1 = calcTSW(F, S, gBAt)

    # Second call: warm start with converged values
    Eminf2, TSW2 = calcTSW(F, S, gBAt, Eminf=Eminf1, TSW=TSW1)

    # Warm start should return identical bounds (no expansion needed)
    assert Eminf2 == Eminf1, f"Warm-started Eminf changed: {Eminf1} -> {Eminf2}"
    assert abs(TSW2 - TSW1) < 1e-6, \
        f"Warm-started TSW changed: {TSW1} -> {TSW2}"
