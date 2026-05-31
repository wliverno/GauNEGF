"""Pre-implementation gate for the Damle / S_eff lower-contour replacement.

Tests that on systems where the OLD lower-contour path (densityComplex over
a deep Eminf) is trustworthy, the NEW path (Damle on S_eff over the same
range) recovers the same total trace(S @ P) to within 1e-3 electrons.

If this passes, the spec's claim that delta_N is subsumed into S_eff is
empirically supported, and implementation can proceed. If it fails, the
Damle path needs an explicit cross-term term and the spec must be revised
before any production code is written.

Two fixtures: tight-binding 1D chain (no pp's, analytic ground truth) and
C2 STO-3G (single-zeta DFT, no pp pathology).
"""
import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy.linalg import eig as scipy_eig

from gauNEGF.scfE import NEGFE
from gauNEGF.density import density, densityComplex
from gauNEGF.transport import har_to_eV
from gauNEGF.config import ENERGY_MIN


def _inv_sqrt_signed(M):
    """M^(-1/2) for real-symmetric possibly-indefinite M (complex output if indef)."""
    D, U = np.linalg.eigh(M)
    return U @ np.diag(1.0 / np.emath.sqrt(D)) @ U.T


def _damle_P(Y_eff, F_eV, Sigma_0, Elow, Eup):
    """One-shot Damle density on [Elow, Eup] in S_eff framework."""
    Fbar = Y_eff @ (F_eV + Sigma_0) @ Y_eff
    Gam = (Sigma_0 - Sigma_0.conj().T) * 1j
    GamBar = Y_eff @ Gam @ Y_eff
    D, V = scipy_eig(Fbar)
    Vc = np.linalg.inv(V.conj().T)
    P_orth = density(V, Vc, D, GamBar, Elow, Eup)
    return Y_eff @ np.asarray(P_orth) @ Y_eff


def _asymptotic_sigma(g, probes=(-1e3, -1e4, -1e5)):
    """Fit Sigma(E) ~ Sigma_0 + E*X over three probes. Returns (Sigma_0, X)."""
    Es = np.asarray(probes, dtype=float)
    Sigs = np.stack([np.asarray(g.sigmaTot(E)) for E in Es], axis=0)
    # Least-squares: Sig = X * E + Sigma_0, treating each matrix element independently.
    A = np.stack([Es, np.ones_like(Es)], axis=1)  # (3, 2)
    sol, *_ = np.linalg.lstsq(A, Sigs.reshape(len(Es), -1), rcond=None)
    X = sol[0].reshape(Sigs.shape[1:])
    Sigma_0 = sol[1].reshape(Sigs.shape[1:])
    return Sigma_0, X


def _compare_paths(negf, deep_Eminf=-500.0, label=''):
    """Run old path and new path on the same converged-isolated F. Return diff."""
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    Emin_old = float(negf.Emin)

    # OLD: densityComplex over [Eminf, Emin] (lower) + densityComplex over [Emin, mu] (upper)
    mu = float(negf.fermi)
    P_old_lower, dN_old = densityComplex(F_eV, S, negf.g, deep_Eminf, Emin_old, tol=1e-4, T=0)
    n_old_lower = float(np.real(np.trace(S @ np.asarray(P_old_lower)))) + dN_old
    P_old_upper, _ = densityComplex(F_eV, S, negf.g, Emin_old, mu, tol=1e-4, T=0)
    n_old_upper = float(np.real(np.trace(S @ np.asarray(P_old_upper))))
    n_old_total = n_old_lower + n_old_upper

    # NEW: Damle over [ENERGY_MIN, Emin_old] + densityComplex over [Emin_old, mu]
    Sigma_0, X = _asymptotic_sigma(negf.g)
    X_sym = 0.5 * (np.real(X) + np.real(X).T)
    S_eff = np.real(S) - X_sym
    Y_eff = _inv_sqrt_signed(S_eff)
    P_new_lower = _damle_P(Y_eff, F_eV, Sigma_0, float(ENERGY_MIN), Emin_old)
    n_new_lower = float(np.real(np.trace(S @ P_new_lower)))
    # Upper piece is the same densityComplex call, same Emin -> reuse.
    n_new_total = n_new_lower + n_old_upper

    print(f'  [{label}] n_old_total = {n_old_total:+10.6f}  '
          f'(lower={n_old_lower:+.6f}, upper={n_old_upper:+.6f}, delta_N={dN_old:+.3e})')
    print(f'  [{label}] n_new_total = {n_new_total:+10.6f}  '
          f'(lower={n_new_lower:+.6f}, upper={n_old_upper:+.6f})')
    print(f'  [{label}] diff        = {n_new_total - n_old_total:+10.6f}')
    return abs(n_new_total - n_old_total)


@pytest.fixture(scope='module')
def c2_sto3g_negf(tmp_path_factory):
    """C2 chain at STO-3G (minimal basis, no pp pathology)."""
    scratch = tmp_path_factory.mktemp('damle_gate_sto3g')
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    gjf_src = os.path.join(repo_root, 'examples', 'C2_chain.gjf')
    gjf_dst = os.path.join(scratch, 'C2_chain.gjf')
    shutil.copy(gjf_src, gjf_dst)
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                     route='integral=grid=superfine')
        negf.setContact1D([[1], [2]], symmetrize_contacts=True)
        negf.setVoltage(0.0)
        yield negf
    finally:
        os.chdir(cwd)


def test_gate_c2_sto3g(c2_sto3g_negf):
    """Gate: C2 STO-3G total electron count matches between old and new paths."""
    diff = _compare_paths(c2_sto3g_negf, deep_Eminf=-500.0, label='C2 STO-3G')
    assert diff < 1e-3, (
        f'GATE FAIL: new path differs from old by {diff:+.3e} electrons. '
        'delta_N is NOT subsumed into S_eff. Spec must be revised before '
        'implementation can proceed.'
    )


def test_gate_tight_binding():
    """Gate: tight-binding 1D chain total electron count matches between old and new paths."""
    # Build a simple TB chain device + 1D contacts. Use NEGFE.setContact1D with
    # alphas/betas so no Gaussian run is needed.
    pytest.importorskip('gauNEGF')
    # Device: 4-site chain, on-site = 0, hopping = -1.0, orthogonal basis (S=I).
    # Contacts: same on-site/hopping. Orthogonal -> X=0 -> S_eff = S, no pp's.
    N_dev = 4
    F_dev = np.zeros((N_dev, N_dev))
    for i in range(N_dev - 1):
        F_dev[i, i+1] = -1.0
        F_dev[i+1, i] = -1.0
    S_dev = np.eye(N_dev)
    # Run only the lower-contour piece via NEGFE wrappers is overkill here;
    # do the comparison directly with helper functions and a surfGTest contact.
    from gauNEGF.surfGTester import surfGTest
    inds = ([0], [N_dev - 1])
    g = surfGTest(F_dev, S_dev, inds, sig1=-0.1j, sig2=-0.1j, spin='r')
    # Both paths need a mu and Emin. Pick mu in the band, Emin below band.
    mu = -0.5
    Emin = -3.0
    P_old, dN_old = densityComplex(F_dev, S_dev, g, -500.0, Emin, tol=1e-4, T=0)
    n_old = float(np.real(np.trace(S_dev @ np.asarray(P_old)))) + dN_old
    Sigma_0, X = _asymptotic_sigma(g)
    X_sym = 0.5 * (np.real(X) + np.real(X).T)
    S_eff = np.real(S_dev) - X_sym
    Y_eff = _inv_sqrt_signed(S_eff)
    P_new = _damle_P(Y_eff, F_dev, Sigma_0, float(ENERGY_MIN), Emin)
    n_new = float(np.real(np.trace(S_dev @ P_new)))
    diff = abs(n_new - n_old)
    print(f'  [TB chain] n_old={n_old:+.6f}, n_new={n_new:+.6f}, diff={diff:+.3e}')
    assert diff < 1e-3, (
        f'GATE FAIL on tight-binding fixture: diff = {diff:+.3e}. '
        'delta_N is NOT subsumed into S_eff (even for orthogonal contacts).'
    )
