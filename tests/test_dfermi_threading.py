"""Tests for the dFermi threading dispatch in integrate.py.

crossTermQ/sigma take a per-contact dFermi shift kwarg. Threading it into a
JIT trace needs a build-time choice: gList-carrying (Bethe-protocol) objects
get dfs[i] threaded as a traced runtime argument; everything else keeps its
own (E, i)-only call so a stored per-contact shift (e.g. surfG1D's
dFermiList) is read directly instead of being clamped/zeroed by an
out-of-range dfs[i] index.

make_bethe_device() builds a minimal two-contact surfGB-shaped fixture
directly from tests/Au.bethe (bypassing surfGB.__init__'s geometry
auto-detection, which needs a real multi-atom coordinate file not present in
this worktree): two isolated Au atoms (9 orbitals each, spin='r'), each its
own contact, using real surfGBAt hopping/overlap matrices so sigma()/
crossTermQ() are the genuine Bethe-lattice code paths -- only the
geometry-detection step (nIndLists/indsLists/gList assembly) is done by
hand.

Run (GPU gate; Bethe fixtures need the surface-GF solve):
    JAX_PLATFORMS=cpu python3 -m pytest test_dfermi_threading.py -q
test_dispatch_leaves_surfG1D_shifts_alone alone is login-safe.
"""
import os
import types

import numpy as np
import jax.numpy as jnp

from gauNEGF import integrate
from gauNEGF.surfGBethe import surfGB, surfGBAt
from test_cross_term import make_1d_nonortho_chain


# ---------------------------------------------------------------------------
# Bethe device fixture
# ---------------------------------------------------------------------------

def _build_atomic_bethe(spin='r'):
    """Build onsite H0, the Sdict overlap dict, and the 12 FCC[111]
    hopping/overlap matrices shared by both contact atoms below, using the
    real surfGB parameter-construction code so the physics matches production."""
    class _Params:
        pass
    params = _Params()
    params.spin = spin
    bethe_path = os.path.join(os.path.dirname(__file__), 'Au')
    surfGB.readBetheParams(params, bethe_path)
    plane_normal = jnp.array([0., 0., 1.])
    first_neighbor = jnp.array([1., 0., 0.])
    dirList = surfGB.genNeighbors(params, plane_normal, first_neighbor)
    Slist = [surfGB.constructMat(params, params.Sdict, d, False) for d in dirList]
    Vlist = [surfGB.constructMat(params, params.Vdict, d, False) for d in dirList]
    return params.H0, Slist, Vlist, params.Sdict


def make_bethe_device():
    """Minimal two-contact device: two isolated Au atoms (9 orbitals each,
    spin='r'), each its own contact. g.gList is non-None, g.num_contacts ==
    2, g.crossTermQ(0.3, 0) returns a (Q_fwd, Q_rev, Q_sym) triple.

    Each atom's fermi/fermi0/dFermi are set to 0.0 directly, skipping the
    calcFermi bisection since these tests only exercise dispatch of a shift
    away from whatever baseline fermi0 is."""
    H0, Slist, Vlist, Sdict = _build_atomic_bethe()
    eta = 1e-3
    atoms = []
    for _ in range(2):
        at = surfGBAt(H0, Slist, Vlist, eta, T=0, SOC=False)
        at.fermi = 0.0
        at.fermi0 = 0.0
        at.dFermi = 0.0
        atoms.append(at)

    class MockSurfGB:
        pass
    g = MockSurfGB()
    g.N = 18
    g.SOC = False
    g.spin = 'r'
    g.gList = atoms
    g.num_contacts = 2
    g.indsLists = [[jnp.arange(0, 9)], [jnp.arange(9, 18)]]
    g.nIndLists = [[[]], [[]]]  # no excluded directions -> full 9-direction sigma sum
    g.Sdict = Sdict
    g.Xi = jnp.eye(18)
    g.eta = eta

    # Small nonzero device Fock so G^R/crossterm are non-trivial. Overlap
    # is identity (device-level S; unrelated to the atomic Sdict['sss']
    # used internally by sigma/crossTermQ's de-orthonormalization switch).
    rng = np.random.default_rng(0)
    offdiag = 0.05 * rng.standard_normal((9, 9))
    F = np.zeros((18, 18))
    F[:9, :9] = np.diag(np.full(9, -5.0))
    F[9:, 9:] = np.diag(np.full(9, -5.0))
    F[:9, 9:] = offdiag
    F[9:, :9] = offdiag.T
    g.F = F
    g.S = np.eye(18)

    # Bind the real (unbound) surfGB methods; they only touch attributes
    # set above, so this is the genuine Bethe-lattice code path.
    g.sigma = types.MethodType(surfGB.sigma, g)
    g.crossTermQ = types.MethodType(surfGB.crossTermQ, g)
    g.sigmaTot = types.MethodType(surfGB.sigmaTot, g)
    g.crossTermQTot = types.MethodType(surfGB.crossTermQTot, g)
    g.updateFermi = types.MethodType(surfGB.updateFermi, g)
    return g


E_LIST = np.linspace(-1.0, 1.0, 7)
W = np.ones_like(E_LIST)


def _run_cross(g):
    return integrate.GrIntCross(np.array(g.F), np.array(g.S), g, E_LIST, W)


def test_kernel_compiles_once_across_shift_changes():
    g = make_bethe_device()
    calls = {'n': 0}
    orig = integrate._cached_kernel
    def counting(key, gg, build):
        def build2():
            calls['n'] += 1
            return build()
        return orig(key, gg, build2)
    integrate._cached_kernel = counting
    try:
        integrate.clear_kernel_cache()
        m_a, s_a = _run_cross(g)      # shifts (0, 0)
        g.updateFermi(0, 0.3)         # move contact-0 shift
        m_b, s_b = _run_cross(g)
        g.updateFermi(0, 0.0)         # back
        m_c, s_c = _run_cross(g)
    finally:
        integrate._cached_kernel = orig
    assert calls['n'] == 1, f"kernel built {calls['n']} times, want 1"
    assert not np.allclose(np.array(s_a), np.array(s_b))
    assert np.allclose(np.array(s_a), np.array(s_c), atol=1e-12)


def test_shift_update_matches_fresh_compile():
    g1 = make_bethe_device(); g1.updateFermi(0, 0.3)
    m1, s1 = _run_cross(g1)               # may reuse a cached kernel
    g2 = make_bethe_device(); g2.updateFermi(0, 0.3)
    integrate.clear_kernel_cache()        # force a fresh compile
    m2, s2 = _run_cross(g2)
    assert np.allclose(np.array(m1), np.array(m2), atol=1e-10)
    assert np.allclose(np.array(s1), np.array(s2), atol=1e-10)


def test_crossterm_shift_uses_own_contact_index():
    # catches a dfs[0]/dfs[1] swap: two different shifts, per-contact
    # comparison against stored-shift baking
    g = make_bethe_device()
    g.updateFermi(0, 0.2); g.updateFermi(1, -0.5)
    E = 0.37 + 0.0j
    for i, d in [(0, 0.2), (1, -0.5)]:
        q_threaded = g.crossTermQ(E, i, dFermi=d)
        q_stored = g.crossTermQ(E, i)          # uses stored shift d
        for a, b in zip(q_threaded, q_stored):
            assert np.allclose(np.array(a), np.array(b), atol=1e-12), i


def test_dispatch_leaves_surfG1D_shifts_alone():
    # the dispatcher must NOT thread dfs into non-gList objects (zeros
    # would overwrite the stored per-contact dFermiList)
    g = make_1d_nonortho_chain()
    assert getattr(g, 'gList', None) is None
    g.dFermiList = [0.3, -0.2]
    F_np, S_np = np.array(g.F), np.array(g.S)
    _, s_kernel = integrate.GrIntCross(F_np, S_np, g, E_LIST, W)

    ref = 0.0 + 0j
    eta = max(g.eta, integrate.ETA)
    for E, w in zip(E_LIST, W):
        sigTot = np.array(g.sigmaTot(E))
        Gr = np.linalg.solve((E + 1j * eta) * S_np - F_np - sigTot, np.eye(F_np.shape[0]))
        Q_tot = np.zeros_like(F_np, dtype=complex)
        for i in range(g.num_contacts):
            Q_i = g.crossTermQ(E, i)  # stored dFermiList, not zeroed by dfs
            if Q_i is not None:
                Q_tot = Q_tot + np.array(Q_i[2])
        ref += w * np.trace(Gr @ Q_tot)
    assert np.allclose(np.array(s_kernel), ref, atol=1e-8)


def test_sigma_shift_tracks_and_uses_own_contact_index():
    """GrLessInt's per-contact sigma call (weighted_func_GrLess in
    integrate.py) is the other dFermi threading site besides crossTermQ.
    Covers it two ways: (a) shift/revert tracking through a full
    GrLessInt(ind=0) integration, and (b) a per-contact index check
    comparing threaded vs stored shifts directly via g.sigma."""
    # (a) shift / revert tracking through GrLessInt(ind=0)
    g = make_bethe_device()
    F_np, S_np = np.array(g.F), np.array(g.S)
    m_a = integrate.GrLessInt(F_np, S_np, g, E_LIST, W, ind=0)
    g.updateFermi(0, 0.3)
    m_b = integrate.GrLessInt(F_np, S_np, g, E_LIST, W, ind=0)
    g.updateFermi(0, 0.0)
    m_c = integrate.GrLessInt(F_np, S_np, g, E_LIST, W, ind=0)
    assert not np.allclose(np.array(m_a), np.array(m_b))
    assert np.allclose(np.array(m_a), np.array(m_c), atol=1e-12)

    # (b) per-contact index check: two different stored shifts, threaded
    # dfs[i] vs stored-shift sigma per contact -- a dfs[0]/dfs[1] swap
    # would make this fail
    g2 = make_bethe_device()
    g2.updateFermi(0, 0.2)
    g2.updateFermi(1, -0.5)
    dfs = integrate._current_dfermis(g2)
    E = 0.37 + 0.0j
    for i, d in [(0, 0.2), (1, -0.5)]:
        sig_threaded = g2.sigma(E, i, dFermi=dfs[i])
        sig_stored = g2.sigma(E, i)  # uses stored shift d
        assert np.allclose(np.array(sig_threaded), np.array(sig_stored),
                            atol=1e-12), i
