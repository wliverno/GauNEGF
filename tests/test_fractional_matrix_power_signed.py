"""Unit tests for fractional_matrix_power_signed (indefinite-tolerant matrix power)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
import pytest

from gauNEGF.utils import fractional_matrix_power, fractional_matrix_power_signed


def test_psd_matches_existing_helper():
    """On a positive-definite input the signed helper must match the PSD-only helper."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    S = A @ A.T + np.eye(5)  # PSD by construction
    S = jnp.asarray(S)
    out_signed = np.asarray(fractional_matrix_power_signed(S, -0.5))
    out_existing = np.asarray(fractional_matrix_power(S, -0.5))
    assert np.allclose(out_signed.imag, 0.0, atol=1e-10), 'PSD input should give real output'
    assert np.allclose(out_signed.real, out_existing, atol=1e-8)


def test_indefinite_input_returns_complex():
    """On an indefinite input the signed helper must return complex output without NaN."""
    # Symmetric matrix with one negative eigenvalue.
    M = np.array([[1.0, 0.5], [0.5, -2.0]])
    M = jnp.asarray(M)
    out = np.asarray(fractional_matrix_power_signed(M, -0.5))
    assert np.isfinite(out).all(), 'Output must not contain NaN or Inf'
    assert not np.allclose(out.imag, 0.0, atol=1e-10), 'Indefinite input should give complex output'


def test_round_trip_identity_psd():
    """S^(1/2) @ S^(1/2) == S for PSD S."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    S = A @ A.T + np.eye(4)
    S = jnp.asarray(S)
    half = np.asarray(fractional_matrix_power_signed(S, 0.5))
    assert np.allclose(half @ half, np.asarray(S), atol=1e-6)


def test_round_trip_identity_indefinite():
    """S^(1/2) @ S^(1/2) == S for indefinite S (complex arithmetic)."""
    M = np.array([[1.0, 0.5], [0.5, -2.0]])
    Mj = jnp.asarray(M)
    half = np.asarray(fractional_matrix_power_signed(Mj, 0.5))
    assert np.allclose(half @ half, M, atol=1e-6)


def test_output_symmetry():
    """Output of fractional_matrix_power_signed must be (complex-)symmetric for symmetric input."""
    M = np.array([[1.0, 0.5], [0.5, -2.0]])
    Mj = jnp.asarray(M)
    out = np.asarray(fractional_matrix_power_signed(Mj, -0.5))
    assert np.allclose(out, out.T, atol=1e-10)
