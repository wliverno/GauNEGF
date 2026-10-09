"""Vectorized density() must equal the original row-by-row implementation."""
import numpy as np
import jax.numpy as jnp
import pytest

from gauNEGF.density import density


def _density_rowwise(V, Vc, D, Gam, Emin, mu):
    # verbatim copy of density.py:332-352 at 2b99ade (pre-vectorization reference)
    Nd = len(V)
    DD = jnp.array([D for i in range(Nd)]).T
    logmat = jnp.array([np.emath.log(1-(mu/D)) for i in range(Nd)]).T
    logmat2 = jnp.array([np.emath.log(1-(Emin/D)) for i in range(Nd)]).T
    invmat = 1/(2*np.pi*(DD-DD.conj().T))
    pref2 = logmat - logmat.conj().T
    pref3 = logmat2-logmat2.conj().T
    prefactor = jnp.multiply(invmat,(pref2-pref3))
    Gammam = Vc.conj().T@Gam@Vc
    prefactor = jnp.multiply(prefactor,Gammam)
    return V@ prefactor @ V.conj().T


def _case(n, seed, real_d=False):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, n)); H = (H + H.T) / 2
    A = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    Gam = A @ A.conj().T
    Fbar = H - 0.5j * Gam
    D, V = np.linalg.eig(Fbar)
    if real_d:
        D = D.real + 0.0
    Vc = np.linalg.inv(V.conj().T)
    return V, Vc, D, Gam


@pytest.mark.parametrize("n,seed", [(1, 0), (5, 1), (30, 2), (64, 3)])
def test_matches_rowwise(n, seed):
    V, Vc, D, Gam = _case(n, seed)
    for Emin, mu in [(-50.0, -4.4), (-2468.0, 0.3)]:
        ref = np.asarray(_density_rowwise(V, Vc, D, Gam, Emin, mu))
        new = np.asarray(density(V, Vc, D, Gam, Emin, mu))
        assert np.max(np.abs(new - ref)) <= 1e-12 * max(1.0, np.max(np.abs(ref)))


def test_real_dtype_eigenvalues():
    V, Vc, D, Gam = _case(6, 4, real_d=True)
    Dr = D.real.astype(complex) - 1e-3j   # small Im keeps DD - DD^H nonzero on the diagonal, as Im(D) < 0 does
    ref = np.asarray(_density_rowwise(V, Vc, Dr, Gam, -50.0, -4.4))
    new = np.asarray(density(V, Vc, Dr, Gam, -50.0, -4.4))
    assert np.max(np.abs(new - ref)) <= 1e-12 * max(1.0, np.max(np.abs(ref)))
