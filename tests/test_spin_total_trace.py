"""Spin-path total transmission: exact trace, 4-channel projection kept.

Spec: docs/superpowers/specs/2026-08-19-spin-total-trace-fix.md.
T1 exactness vs dense numpy, T2 spin-frame invariance, T3 spin-scalar
bit-identity regression, T4 current path integrates the exact total.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from gauNEGF.config import ETA
from gauNEGF.transport import (SigmaCalculator, transmission_single_energy,
                               calculate_transmission, calculate_current,
                               eoverh)
import jax.numpy as jnp

N = 3          # orbitals; spin space is 2N x 2N
TWO_N = 2 * N
ENERGIES = [-0.7, 0.1, 1.3]


def _hermitian(rng, n, scale=1.0):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return scale * (A + A.conj().T) / 2


def _psd(rng, n, scale=1.0):
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return scale * (B @ B.conj().T)


def make_spin_mixing_system(seed=7):
    """Small dense system with NONZERO off-diagonal spin blocks in gamma.

    Block ordering [up_0..up_N-1, dn_0..dn_N-1] (the 'u' path layout).
    sigma_i = H_i - 0.5j*gamma_i so gamma_i = 1j*(sig - sig^dag) exactly.
    """
    rng = np.random.default_rng(seed)
    F = _hermitian(rng, TWO_N)
    S = np.eye(TWO_N) + 0.05 * _hermitian(rng, TWO_N)
    gamma1 = _psd(rng, TWO_N, 0.3)
    gamma2 = _psd(rng, TWO_N, 0.3)
    sig1 = _hermitian(rng, TWO_N, 0.1) - 0.5j * gamma1
    sig2 = _hermitian(rng, TWO_N, 0.1) - 0.5j * gamma2
    return F, S, sig1, sig2


def make_spin_scalar_system(seed=11):
    """Same layout, sigmas kron(I2, sig_N): off-diagonal spin blocks are
    EXACTLY zero (spin-scalar leads)."""
    rng = np.random.default_rng(seed)
    F = _hermitian(rng, TWO_N)
    S = np.eye(TWO_N) + 0.05 * _hermitian(rng, TWO_N)
    sig1_N = _hermitian(rng, N, 0.1) - 0.5j * _psd(rng, N, 0.3)
    sig2_N = _hermitian(rng, N, 0.1) - 0.5j * _psd(rng, N, 0.3)
    sig1 = np.kron(np.eye(2), sig1_N)
    sig2 = np.kron(np.eye(2), sig2_N)
    return F, S, sig1, sig2


def dense_reference(E, F, S, sig1, sig2):
    """Straight numpy: exact trace, 4 channel terms, off-diagonal remainder."""
    gamma1 = 1j * (sig1 - sig1.conj().T)
    gamma2 = 1j * (sig2 - sig2.conj().T)
    Gr = np.linalg.inv((E + 1j * ETA) * S - F - sig1 - sig2)
    Ga = Gr.conj().T
    T_exact = np.real(np.trace(gamma1 @ Gr @ gamma2 @ Ga))

    channels = []
    blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (a,b) for uu, ud, du, dd
    sl = [slice(0, N), slice(N, TWO_N)]
    for a, b in blocks:
        g1 = gamma1[sl[a], sl[a]]
        g2 = gamma2[sl[b], sl[b]]
        Gab = Gr[sl[a], sl[b]]
        channels.append(np.real(np.trace(g1 @ Gab @ g2 @ Gab.conj().T)))

    remainder = T_exact - sum(channels)
    return T_exact, channels, remainder


def block_to_spinor(M):
    """Permute a block-ordered 2Nx2N matrix into spinor ordering
    [up_0, dn_0, up_1, dn_1, ...] (inverse of the transport.py shuffle)."""
    perm = np.concatenate([np.arange(0, TWO_N, 2), np.arange(1, TWO_N, 2)])
    out = np.empty_like(M)
    out[np.ix_(perm, perm)] = M
    return out


def su2_rotation():
    """Non-trivial SU(2) rotation."""
    t, phi = 0.6, 0.9
    return np.array([[np.cos(t), -np.exp(1j * phi) * np.sin(t)],
                     [np.exp(-1j * phi) * np.sin(t), np.cos(t)]])


# ---------------------------------------------------------------- T1

@pytest.mark.parametrize("E", ENERGIES)
def test_T1_total_matches_dense_full_trace_u(E):
    F, S, sig1, sig2 = make_spin_mixing_system()
    calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    total, channels = transmission_single_energy(
        E, jnp.asarray(F), jnp.asarray(S), calc, spin='u')
    T_exact, ref_channels, remainder = dense_reference(E, F, S, sig1, sig2)
    assert remainder != 0.0  # fixture really mixes spin
    np.testing.assert_allclose(channels, ref_channels, rtol=1e-11)
    np.testing.assert_allclose(total, T_exact, rtol=1e-11)
    # remainder bookkeeping: total - sum(channels) is the off-diag remainder
    np.testing.assert_allclose(total - sum(channels), remainder,
                               rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("E", ENERGIES)
def test_T1_total_matches_dense_full_trace_g(E):
    F, S, sig1, sig2 = make_spin_mixing_system()
    Fp, Sp = block_to_spinor(F), block_to_spinor(S)
    s1p, s2p = block_to_spinor(sig1), block_to_spinor(sig2)
    calc = SigmaCalculator(s1p, s2p, energy_dependent=False)
    total, channels = transmission_single_energy(
        E, jnp.asarray(Fp), jnp.asarray(Sp), calc, spin='g')
    T_exact, ref_channels, _ = dense_reference(E, F, S, sig1, sig2)
    np.testing.assert_allclose(channels, ref_channels, rtol=1e-11)
    np.testing.assert_allclose(total, T_exact, rtol=1e-11)


# ---------------------------------------------------------------- T2

@pytest.mark.parametrize("E", ENERGIES)
def test_T2_total_frame_invariant_under_spin_rotation(E):
    F, S, sig1, sig2 = make_spin_mixing_system()
    calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    total, _ = transmission_single_energy(
        E, jnp.asarray(F), jnp.asarray(S), calc, spin='u')

    # Block ordering puts spin on the OUTER index: U = kron(R, I_N)
    U = np.kron(su2_rotation(), np.eye(N))
    Ud = U.conj().T
    calc_rot = SigmaCalculator(Ud @ sig1 @ U, Ud @ sig2 @ U,
                               energy_dependent=False)
    total_rot, _ = transmission_single_energy(
        E, jnp.asarray(Ud @ F @ U), jnp.asarray(Ud @ S @ U), calc_rot,
        spin='u')
    np.testing.assert_allclose(total_rot, total, rtol=1e-12)


# ---------------------------------------------------------------- T3

@pytest.mark.parametrize("E", ENERGIES)
def test_T3_spin_scalar_total_bit_identical_to_old_4term_sum(E):
    F, S, sig1, sig2 = make_spin_scalar_system()
    calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    total, channels = transmission_single_energy(
        E, jnp.asarray(F), jnp.asarray(S), calc, spin='u')

    # Old kernel total was float(jnp.sum(T_spin)); with zero off-blocks
    # the D+O remainder is exactly 0.0 so the new total must be bitwise
    # equal to that same reduction over the returned channels.
    old_total = float(jnp.sum(jnp.asarray(channels, dtype=jnp.float64)))
    assert total == old_total

    # channel definition did not drift
    _, ref_channels, remainder = dense_reference(E, F, S, sig1, sig2)
    assert remainder == pytest.approx(0.0, abs=1e-13)
    np.testing.assert_allclose(channels, ref_channels, rtol=1e-11)


# ---------------------------------------------------------------- T4

FERMI, QV, DE = 0.1, 0.5, 0.05


def _grid():
    muL, muR = FERMI + QV / 2.0, FERMI - QV / 2.0
    lo, hi = min(muL, muR), max(muL, muR)
    return np.arange(lo, hi, DE), np.sign(muL - muR)


def test_T4_current_total_integrates_exact_total_spin_mixing():
    F, S, sig1, sig2 = make_spin_mixing_system()
    calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    I_total, I_spin = calculate_current(F, S, calc, fermi=FERMI, qV=QV,
                                        T=0, spin='u', dE=DE)
    energies, grid_sign = _grid()
    exact = np.array([dense_reference(E, F, S, sig1, sig2)[0]
                      for E in energies])
    I_exact = grid_sign * eoverh * trapezoid(exact, energies)
    np.testing.assert_allclose(I_total, I_exact, rtol=1e-11)

    # channel currents unchanged bitwise vs direct trapezoid of channels
    _, Tspin = calculate_transmission(F, S, calc, energies, spin='u')
    for i in range(4):
        assert I_spin[i] == grid_sign * eoverh * trapezoid(
            Tspin[:, i], energies)

    # remainder integrates to a nonzero difference: total != sum(channels)
    I_rem = I_total - sum(I_spin)
    rem = np.array([dense_reference(E, F, S, sig1, sig2)[2]
                    for E in energies])
    np.testing.assert_allclose(
        I_rem, grid_sign * eoverh * trapezoid(rem, energies), rtol=1e-9)
    assert abs(I_rem) > 1e-12 * abs(I_total)


def test_T4_current_spin_scalar_total_matches_channel_sum():
    F, S, sig1, sig2 = make_spin_scalar_system()
    calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    I_total, I_spin = calculate_current(F, S, calc, fermi=FERMI, qV=QV,
                                        T=0, spin='u', dE=DE)
    np.testing.assert_allclose(I_total, sum(I_spin), rtol=1e-13)
