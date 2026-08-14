"""Unit tests for inv_sqrt_general (general, non-Hermitian inverse square root).

The effective overlap S_eff = S - X_asymp is built from the asymptotic slope of
the retarded contact self-energy. That self-energy is Sigma = A @ g_surf @ A^dag
with g_surf complex-symmetric, so it carries broadening Gamma = i(Sigma-Sigma^dag)
and is NOT Hermitian. eigh assumes Hermitian and silently uses one triangle, so it
computes the wrong S_eff^(-1/2). The correct construction is a general
eigendecomposition: Y = V D^(-1/2) V^(-1), which satisfies Y @ M @ Y = I for any
diagonalizable M (Hermitian, complex-symmetric, or neither).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)  # match production (density.py enables x64)

import numpy as np
import pytest

from gauNEGF.utils import inv_sqrt_general


def _defect(Y, M):
    """||Y @ M @ Y - I|| -- zero iff Y is a valid M^(-1/2)."""
    Y = np.asarray(Y)
    M = np.asarray(M)
    return float(np.linalg.norm(Y @ M @ Y - np.eye(M.shape[0])))


def _eigh_inv_sqrt(M):
    """The OLD Hermitian-assuming inverse sqrt (the WRONG tool for non-Hermitian
    M): eigh uses one triangle and assumes M = M^dagger."""
    M = np.asarray(M)
    w, U = np.linalg.eigh(M)
    return U @ np.diag(1.0 / np.emath.sqrt(w)) @ U.conj().T


def test_defining_property_spd():
    """Sanity: still correct on a symmetric-positive-definite (Hermitian) M."""
    rng = np.random.default_rng(0)
    N = 5
    B = rng.normal(size=(N, N))
    M = B @ B.T + N * np.eye(N)
    assert _defect(inv_sqrt_general(M), M) < 1e-8


def test_defining_property_complex_symmetric_nonhermitian():
    """The real S_eff structure: complex-symmetric (M = M^T), NOT Hermitian."""
    rng = np.random.default_rng(1)
    N = 6
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    M = A + A.T  # complex symmetric, non-Hermitian
    assert _defect(inv_sqrt_general(M), M) < 1e-7


def test_defining_property_fully_general():
    """Even a fully general (non-symmetric, non-Hermitian) M works."""
    rng = np.random.default_rng(7)
    N = 5
    M = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    assert _defect(inv_sqrt_general(M), M) < 1e-7


def test_eigh_path_demonstrably_wrong_on_nonhermitian():
    """Direct comparison: the eigh-based inverse sqrt FAILS Y @ M @ Y = I on a
    non-Hermitian M, while the general eig path passes. This is the quantitative
    reason the Re-part + symmetrize shortcut is not justified."""
    rng = np.random.default_rng(2)
    N = 6
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    M = A + A.T  # complex symmetric, non-Hermitian
    err_gen = _defect(inv_sqrt_general(M), M)
    err_eigh = _defect(_eigh_inv_sqrt(M), M)
    assert err_gen < 1e-7
    assert err_eigh > 1e-2, (
        f'eigh defect = {err_eigh:.3e}; expected it to be materially wrong on a '
        f'non-Hermitian matrix (general-eig defect = {err_gen:.3e}).')


def test_output_is_numpy_complex():
    """Returns a concrete numpy array (callers store it on the NEGFE object)."""
    M = np.diag([1.0, 2.0, 3.0])
    Y = inv_sqrt_general(M)
    assert isinstance(np.asarray(Y), np.ndarray)
