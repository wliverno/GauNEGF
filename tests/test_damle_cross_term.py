"""Pure-math unit tests for damleCrossTerm (no gdv / no SCF needed)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gauNEGF.density import damleCrossTerm


def _identity_inputs(D):
    N = D.shape[0]
    I = jnp.eye(N, dtype=complex)
    return I, jnp.asarray(D), I, I  # V, D, Vc, Y_eff (all identity/eigenbasis)


def test_cross_term_pole_free_is_zero():
    # All eigenvalues ABOVE hi -> no in-window poles -> delta_N ~ 0,
    # even with a nonzero linear-Q slope (this is the dropped-flat-term property).
    D = jnp.array([-5.0 + 1e-6j, -3.0 + 1e-6j, 2.0 + 1e-6j])
    V, D, Vc, Y = _identity_inputs(D)
    Q0 = jnp.eye(3, dtype=complex)
    Q1 = 0.01 * jnp.eye(3, dtype=complex)   # nonzero slope: would blow up if flat term kept
    dN = damleCrossTerm(V, D, Vc, Y, Q0, Q1, lo=-1.0e6, hi=-10.0)
    assert abs(dN) < 1e-6


def test_cross_term_in_window_pole_registers():
    # One eigenvalue INSIDE [lo, hi] -> nonzero, O(1) delta_N (the warning must fire).
    D = jnp.array([-100.0 + 1e-6j])
    V, D, Vc, Y = _identity_inputs(D)
    Q0 = 2.0 * jnp.eye(1, dtype=complex)
    Q1 = jnp.zeros((1, 1), dtype=complex)
    dN = damleCrossTerm(V, D, Vc, Y, Q0, Q1, lo=-1.0e6, hi=-10.0)
    # b0 + b1*D = 2.0; Im(log-diff) = +/- pi for the in-window pole
    # -> |delta_N| = (1/pi)*|2.0*pi| = 2.0
    assert abs(abs(dN) - 2.0) < 1e-3
