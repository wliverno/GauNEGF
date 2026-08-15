"""Test-only W_beta and DOS assemblies. Two INDEPENDENT code paths:
W_beta_dense builds the per-contact kernel from sigma/crossTermQ blocks;
dos_qsym builds DOS from the production Q_sym form. Their agreement is
the parameter-free implementation check."""
import numpy as np
import jax

_JIT = {}


def _prod(g, E):
    """Production sigma/crossTermQ blocks at E, as numpy.

    Jitted per object: sigma's eager lax.cond recompiles on every call
    outside jit (80x slower); the returned values agree to machine epsilon.
    """
    key = (id(g), getattr(g, '_gauNEGF_version', 0))
    if key not in _JIT:
        _JIT[key] = (g, jax.jit(g.sigma, static_argnums=(1,)),
                     jax.jit(g.crossTermQ, static_argnums=(1,)))
    # the stored g pins id(g) against CPython id reuse -- dropping it from
    # the tuple would let a freed object's address rebind to a stale cache
    _, sigma, crossQ = _JIT[key]
    sigs, qs = [], []
    for i in range(g.num_contacts):
        sigs.append(np.array(sigma(E, i)))
        q = crossQ(E, i)
        qs.append(None if q is None else tuple(np.array(x) for x in q))
    return sigs, qs


def _gr(g, E, eta):
    F = np.array(g.F); S = np.array(g.S)
    sigs, _ = _prod(g, E)
    Gr = np.linalg.inv((E + 1j * eta) * S - F - sum(sigs))
    return Gr, sigs, S


def W_beta_dense(g, E, b, eta):
    Gr, sigs, S = _gr(g, E, eta)
    _, qs = _prod(g, E)
    Ga = Gr.conj().T
    Gam_b = 1j * (sigs[b] - sigs[b].conj().T)
    dev = np.real(np.trace(Gr @ Gam_b @ Ga @ S))
    own = 0.0
    if qs[b] is not None:
        own = (np.imag(np.trace(Gr @ qs[b][0]))
               + np.imag(np.trace(Ga @ qs[b][1])))
    tail = 0.0
    for a in range(g.num_contacts):
        if qs[a] is not None:
            Qr = qs[a][1]
            tail -= 0.5 * np.real(np.trace(Gr @ Gam_b @ Ga
                                           @ (Qr + Qr.conj().T)))
    return dev + own + tail


def dos_qsym(g, E, eta):
    Gr, _, S = _gr(g, E, eta)
    _, qs = _prod(g, E)
    Q = sum(q[2] for q in qs if q is not None)
    return (-1.0 / np.pi) * np.imag(np.trace(Gr @ (S - Q)))
