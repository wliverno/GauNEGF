"""Tests for GrLessIntCross: the window integrator that carries the
non-equilibrium Mulliken cross term (own-tail + device-tail per contact)."""
import numpy as np
from gauNEGF import integrate
from test_cross_term import make_1d_nonortho_chain, make_1d_ortho_chain
from wbeta_reference import W_beta_dense, dos_qsym, _gr, _prod

ETA = 1e-6


def dense_reference(g, E, ind):
    """Brute-force W-cross pieces + Gless at one energy, numpy only."""
    F = np.array(g.F); S = np.array(g.S)
    nc = g.num_contacts
    sigs = [np.array(g.sigma(E, i)) for i in range(nc)]
    eta = max(g.eta, ETA)
    Gr = np.linalg.inv((E + 1j * eta) * S - F - sum(sigs))
    Ga = Gr.conj().T
    sb = sigs[ind]
    Gam = 1j * (sb - sb.conj().T)
    Gless = Gr @ Gam @ Ga
    own = 0.0
    q_b = g.crossTermQ(E, ind)
    if q_b is not None:
        own = (np.imag(np.trace(Gr @ np.array(q_b[0])))
               + np.imag(np.trace(Ga @ np.array(q_b[1]))))
    tail = 0.0
    for a in range(nc):
        q_a = g.crossTermQ(E, a)
        if q_a is not None:
            Qr = np.array(q_a[1])
            tail -= 0.5 * np.real(np.trace(Gless @ (Qr + Qr.conj().T)))
    return Gless, own + tail


def test_grlessintcross_matches_dense():
    g = make_1d_nonortho_chain()
    Elist = np.linspace(-0.8, 0.9, 5); w = np.linspace(0.5, 1.5, 5)
    mat, scl = integrate.GrLessIntCross(np.array(g.F), np.array(g.S),
                                        g, Elist, w, ind=1)
    ref_m = np.zeros_like(np.array(g.F), dtype=complex); ref_s = 0.0
    for E, wi in zip(Elist, w):
        m, s = dense_reference(g, E, 1)
        ref_m += wi * m; ref_s += wi * s
    assert np.allclose(np.array(mat), ref_m, atol=1e-8)
    assert abs(complex(scl).imag) < 1e-8 * max(1.0, abs(complex(scl).real))
    assert np.isclose(complex(scl).real, ref_s, atol=1e-8)


def test_grlessintcross_raises_on_ind_none():
    g = make_1d_nonortho_chain()
    import pytest
    with pytest.raises(ValueError):
        integrate.GrLessIntCross(np.array(g.F), np.array(g.S), g,
                                 np.array([0.1]), np.array([1.0]), ind=None)


def test_mixed_ortho_kernel_contact_keeps_device_tail():
    """C10(a): kernel contact orthogonal, other contact non-orthogonal:
    own-tail = 0 but device-tail != 0. Build a fresh mixed toy (contact 0
    orthogonal, contact 1 not) mirroring make_1d_nonortho_chain, rather
    than mutating a live object's stauList."""
    from gauNEGF.surfG1D import surfG
    import jax.numpy as jnp
    N, nc = 4, 2
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    for i in range(N - 1):
        F = F.at[i, i+1].set(-0.3)
        F = F.at[i+1, i].set(-0.3)
    S = jnp.eye(N) + 0.0j
    for i in range(N - 1):
        S = S.at[i, i+1].set(0.1)
        S = S.at[i+1, i].set(0.1)
    indsList = [jnp.array(list(range(nc))), jnp.array(list(range(N-nc, N)))]
    taus = [jnp.ones((nc, nc), dtype=complex) * -0.3,
            jnp.ones((nc, nc), dtype=complex) * -0.3]
    staus = [None, jnp.ones((nc, nc), dtype=complex) * 0.1]
    alphas = [jnp.diag(jnp.array([-0.5] * nc, dtype=complex)),
              jnp.diag(jnp.array([0.5] * nc, dtype=complex))]
    betas = [jnp.ones((nc, nc), dtype=complex) * -0.3,
             jnp.ones((nc, nc), dtype=complex) * -0.3]
    aOverlaps = [jnp.eye(nc, dtype=complex) * 1.05,
                 jnp.eye(nc, dtype=complex) * 1.05]
    bOverlaps = [jnp.ones((nc, nc), dtype=complex) * 0.1,
                 jnp.ones((nc, nc), dtype=complex) * 0.1]
    g = surfG(F, S, indsList, taus=taus, staus=staus,
              alphas=alphas, betas=betas,
              aOverlaps=aOverlaps, bOverlaps=bOverlaps, eta=1e-3)

    assert g.crossTermQ(0.3, 0) is None
    assert g.crossTermQ(0.3, 1) is not None
    mat, scl = integrate.GrLessIntCross(np.array(g.F), np.array(g.S), g,
                                        np.array([0.3]), np.array([1.0]),
                                        ind=0)
    ref_m, ref_s = dense_reference(g, 0.3, 0)
    assert abs(complex(scl).real) > 0.0   # device-tail survives via contact 1
    assert np.allclose(np.array(mat), ref_m, atol=1e-8)
    assert np.isclose(complex(scl).real, ref_s, atol=1e-8)


def test_grlessintcross_fully_orthogonal_scalar_is_zero():
    """All contacts orthogonal: cross scalar is exactly zero, Gless is finite."""
    g = make_1d_ortho_chain()
    mat, scl = integrate.GrLessIntCross(np.array(g.F), np.array(g.S), g,
                                        np.array([0.3]), np.array([1.0]),
                                        ind=0)
    assert complex(scl) == 0.0
    assert np.all(np.isfinite(np.array(mat)))


def test_zero_bias_window_vanishes():
    """Equal contact potentials collapse the window to an empty integral."""
    from gauNEGF.density import densityGrid
    g = make_1d_nonortho_chain()
    P, dN = densityGrid(np.array(g.F), np.array(g.S), g, 0.2, 0.2,
                        ind=-1, T=0.0)
    assert np.allclose(np.array(P), 0.0, atol=1e-14) and dN == 0.0


def test_zero_temperature_hard_step_window_quadratures_agree():
    """Adaptive and fixed-grid quadrature agree on a T=0 hard-step window."""
    from gauNEGF.density import densityGrid, densityGridN
    g = make_1d_nonortho_chain()
    Pa, dNa = densityGrid(np.array(g.F), np.array(g.S), g, -0.3, 0.4,
                          ind=-1, T=0.0, tol=1e-8)
    Pn, dNn = densityGridN(np.array(g.F), np.array(g.S), g, -0.3, 0.4,
                           ind=-1, N=600, T=0.0)
    assert np.isfinite(dNa) and np.isfinite(dNn)
    assert np.isclose(dNa, dNn, rtol=1e-3)   # two quadratures agree


def test_window_count_enters_balance_unhalved():
    """windowN enters the count balance on the same per-spin footing as eqN, unhalved."""
    from gauNEGF.scfE import count_audit
    ne_total, eqN, windowN, nLower = 8.0, 3.6, 0.4, 0.0
    assert np.isclose(count_audit(ne_total / 2, nLower, eqN, windowN), 0.0, atol=1e-12)
    assert np.isclose(count_audit(ne_total, nLower, eqN, windowN), 4.0)


# ---------------------------------------------------------------------------
# Asymmetric two-lead model junction: unequal lead onsites/hoppings, complex
# Hermitian blocks, non-orthogonal intra-lead and device-lead overlap.
# ---------------------------------------------------------------------------

_NL, _ND = 2, 5

_ALPHA = [np.array([[0.10, -0.9 + 0.15j], [-0.9 - 0.15j, -0.05]]),
          np.array([[0.45, -0.60], [-0.60, 0.20]], dtype=complex)]
_BETA = [np.array([[-1.0, 0.2 + 0.1j], [0.15 - 0.05j, -0.8]]),
         np.array([[-0.70, 0.05], [0.10, -0.55]], dtype=complex)]
_SALPHA = [np.array([[1.0, 0.12 + 0.03j], [0.12 - 0.03j, 1.0]]),
           np.array([[1.0, 0.07], [0.07, 1.0]], dtype=complex)]
_SBETA = [np.array([[0.20 + 0.02j, 0.05 - 0.02j], [0.04 + 0.01j, 0.15 + 0.006j]]),
          np.array([[0.12, 0.03], [0.02, 0.09]], dtype=complex)]
_TAU = [np.array([[-0.80, 0.10], [0.05, -0.30]], dtype=complex),
        np.array([[-0.20, 0.10], [-0.60, -0.25]], dtype=complex)]
_STAU = [np.array([[0.15, 0.02], [0.01, 0.05]], dtype=complex),
         np.array([[0.03, 0.02], [0.25, 0.08]], dtype=complex)]
_INDS = [np.array([0, 1]), np.array([3, 4])]

_FDEV = np.array([[0.30, -0.45, 0.10, 0.0, 0.0],
                  [-0.45, -0.10, -0.50 + 0.05j, 0.08, 0.0],
                  [0.10, -0.50 - 0.05j, 0.20, -0.40, 0.05],
                  [0.0, 0.08, -0.40, 0.05, -0.35],
                  [0.0, 0.0, 0.05, -0.35, -0.25]])
_SDEV = np.array([[1.0, 0.12, 0.04, 0.0, 0.0],
                  [0.12, 1.0, 0.11, 0.02, 0.0],
                  [0.04, 0.11, 1.0, 0.09, 0.01],
                  [0.0, 0.02, 0.09, 1.0, 0.07],
                  [0.0, 0.0, 0.01, 0.07, 1.0]], dtype=complex)


def _model_surfG(eta):
    """Production surfG1D on the model blocks (explicit tauList/stauList)."""
    from gauNEGF.surfG1D import surfG
    import jax.numpy as jnp
    return surfG(jnp.array(_FDEV), jnp.array(_SDEV),
                 [jnp.array(i) for i in _INDS],
                 taus=[jnp.array(t) for t in _TAU],
                 staus=[jnp.array(t) for t in _STAU],
                 alphas=[jnp.array(a) for a in _ALPHA],
                 betas=[jnp.array(b) for b in _BETA],
                 aOverlaps=[jnp.array(a) for a in _SALPHA],
                 bOverlaps=[jnp.array(b) for b in _SBETA], eta=eta)


def _model_gsurf(E, eta, a):
    """Sancho-Rubio decimation for the semi-infinite lead, same EOM
    (z*Salpha - alpha, z*Sbeta - beta) that surfG1D.g iterates."""
    z = E + 1j * eta
    fwd = z * _SBETA[a] - _BETA[a]
    rev = z * _SBETA[a].conj().T - _BETA[a].conj().T
    e = z * _SALPHA[a] - _ALPHA[a]
    es = e.copy()
    for _ in range(200):
        gg = np.linalg.inv(e)
        fgr = fwd @ gg @ rev
        rgf = rev @ gg @ fwd
        es = es - fgr
        e = e - fgr - rgf
        fwd = fwd @ gg @ fwd
        rev = rev @ gg @ rev
        if np.linalg.norm(fwd) < 1e-14 and np.linalg.norm(rev) < 1e-14:
            break
    return np.linalg.inv(es)


def _model_kernel(E, eta, b):
    """Per-contact window kernel W_b assembled from the decimation surface
    GFs: sigma_a = t_a g_a t_a^dag, Q_fwd = t g stau^dag, Q_rev = stau g t^dag.
    Assumes CList identity (contact regularization disabled) -- re-enabling it
    in production must fail here rather than pass silently."""
    sig = [np.zeros((_ND, _ND), complex) for _ in range(2)]
    Qf = [np.zeros((_ND, _ND), complex) for _ in range(2)]
    Qr = [np.zeros((_ND, _ND), complex) for _ in range(2)]
    for a in range(2):
        ga = _model_gsurf(E, eta, a)
        t = E * _STAU[a] - _TAU[a]
        ix = np.ix_(_INDS[a], _INDS[a])
        sig[a][ix] = t @ ga @ t.conj().T
        Qf[a][ix] = t @ ga @ _STAU[a].conj().T
        Qr[a][ix] = _STAU[a] @ ga @ t.conj().T
    Gr = np.linalg.inv((E + 1j * eta) * _SDEV - _FDEV - sig[0] - sig[1])
    Ga = Gr.conj().T
    K = Gr @ (1j * (sig[b] - sig[b].conj().T)) @ Ga
    w = (np.real(np.trace(K @ _SDEV)) + np.imag(np.trace(Gr @ Qf[b]))
         + np.imag(np.trace(Ga @ Qr[b])))
    for a in range(2):
        w -= 0.5 * np.real(np.trace(K @ (Qr[a] + Qr[a].conj().T)))
    return w


def _model_finite_chain(M):
    """Device plus M explicit lead cells per contact, same blocks as
    _model_surfG. Cell 0 touches the device; hopping runs cell i -> i+1."""
    n = _ND + 2 * M * _NL
    H = np.zeros((n, n), dtype=complex)
    S = np.zeros((n, n), dtype=complex)
    H[:_ND, :_ND] = _FDEV
    S[:_ND, :_ND] = _SDEV
    for a in range(2):
        off = _ND + a * M * _NL
        for i in range(M):
            r = off + i * _NL
            H[r:r+_NL, r:r+_NL] = _ALPHA[a]
            S[r:r+_NL, r:r+_NL] = _SALPHA[a]
            if i < M - 1:
                H[r:r+_NL, r+_NL:r+2*_NL] = _BETA[a]
                H[r+_NL:r+2*_NL, r:r+_NL] = _BETA[a].conj().T
                S[r:r+_NL, r+_NL:r+2*_NL] = _SBETA[a]
                S[r+_NL:r+2*_NL, r:r+_NL] = _SBETA[a].conj().T
        cell0 = np.arange(off, off + _NL)
        H[np.ix_(_INDS[a], cell0)] = _TAU[a]
        H[np.ix_(cell0, _INDS[a])] = _TAU[a].conj().T
        S[np.ix_(_INDS[a], cell0)] = _STAU[a]
        S[np.ix_(cell0, _INDS[a])] = _STAU[a].conj().T
    return H, S


def _fermi(E, mu, kT):
    return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -500.0, 500.0)))


def _model_count_negf(g, eta, mu, kT, dE, stride=1):
    """Equilibrium device Mulliken count from the Q_sym DOS on a real grid.
    The -7.0 floor truncates the eta-Lorentzian tail below the band; that
    piece scales with eta, so it sits inside the broadening budget."""
    grid = np.arange(-7.0, mu + 20 * kT, dE)[::stride]
    dos = np.array([dos_qsym(g, E, eta) for E in grid])
    return np.trapz(dos * _fermi(grid, mu, kT), grid)


def _model_count_exact(M, mu, kT):
    """Device Mulliken count Re Tr_D[rho S] by generalized eigh on the chain."""
    import scipy.linalg as sla
    H, S = _model_finite_chain(M)
    evals, evecs = sla.eigh(H, S)
    rho = (evecs * _fermi(evals, mu, kT)) @ evecs.conj().T
    return float(np.real(np.trace((rho @ S)[:_ND, :_ND])))


def _eta_residual(g, E, eta):
    """Analytic O(eta) regulator term carried by the kernel sum:
    sum_b W_b - 2 pi DOS = -2 eta Re Tr[Gr S Ga (S - sum_a Herm(Q_rev_a))]."""
    Gr, _, S = _gr(g, E, eta)
    _, qs = _prod(g, E)
    tail = sum(0.5 * (q[1] + q[1].conj().T) for q in qs if q is not None)
    return -2 * eta * np.real(np.trace(Gr @ S @ Gr.conj().T @ (S - tail)))


def test_kernel_sum_equals_dos_with_eta_residual():
    """The +i*eta*S regulator sits on the whole matrix, so the kernel sum
    matches 2 pi DOS only up to that analytic term - exact at every eta."""
    g = make_1d_nonortho_chain()
    for eta in (1e-3, 1e-4, 1e-5):
        for E in np.linspace(-1.2, 1.4, 9):
            Wsum = sum(W_beta_dense(g, E, b, eta)
                       for b in range(g.num_contacts))
            assert np.isclose(Wsum, 2 * np.pi * dos_qsym(g, E, eta)
                              + _eta_residual(g, E, eta),
                              rtol=1e-8, atol=1e-10), (eta, E)


def test_per_contact_kernel_matches_independent_construction():
    """Split-sensitive gate: production kernel vs a numpy-only assembly from
    decimation surface GFs, with the contact-label swap as negative control."""
    from gauNEGF.config import SURFACE_GREEN_CONVERGENCE as CONV
    for eta in (1e-3, 1e-4, 1e-5):
        g = _model_surfG(eta)
        for E in np.linspace(-1.2, 1.4, 7):
            for b in range(2):
                w_prod = W_beta_dense(g, E, b, eta)
                w_ref = _model_kernel(E, eta, b)
                # fixed-point error is linear in conv and flat in eta; the
                # 100x covers its propagation through the kernel assembly
                assert np.isclose(w_ref, w_prod, rtol=100 * CONV), (eta, E, b)
                assert not np.isclose(_model_kernel(E, eta, 1 - b), w_prod,
                                      rtol=100 * CONV), (eta, E, b)
        # tightening conv drives production onto the decimation result at
        # machine level: same function, so the 100x above is solver slack only
        for b in range(2):
            g_tight = np.array(g.g(0.13, b, 1e-12))
            assert np.allclose(g_tight, _model_gsurf(0.13, eta, b), atol=1e-8)


def test_count_matches_exact_diagonalization():
    """Arbiter on the EQUILIBRIUM count: the Q_sym device count converges to
    the generalized-eigh Mulliken count of an explicit finite chain as the
    broadening shrinks. The biased window is anchored indirectly - the kernel
    sum fixes sum_b W_b and the per-contact gate fixes its split - since a
    finite-chain eigh arbiter is not defined out of equilibrium."""
    mu, kT, dE = -0.20, 0.05, 2e-3
    exact = _model_count_exact(200, mu, kT)
    finite_size = abs(exact - _model_count_exact(100, mu, kT))
    coarse = _model_count_negf(_model_surfG(2e-3), 2e-3, mu, kT, dE)
    g = _model_surfG(1e-3)
    negf = _model_count_negf(g, 1e-3, mu, kT, dE)
    quad = abs(negf - _model_count_negf(g, 1e-3, mu, kT, dE, stride=2))
    # broadening covers both the eta-linear error and the truncated tail
    broadening = abs(negf - coarse)
    err = abs(negf - exact)
    assert err <= 2 * (broadening + quad + finite_size), (err, broadening,
                                                          quad, finite_size)
    assert err < 0.6 * abs(coarse - exact), (err, abs(coarse - exact))


def test_orthogonal_kernel_reduces_to_device_term():
    g = make_1d_ortho_chain()
    eta = 1e-5
    for b in range(g.num_contacts):
        Gr, sigs, S = _gr(g, 0.3, eta)
        Gam = 1j * (sigs[b] - sigs[b].conj().T)
        dev = np.real(np.trace(Gr @ Gam @ Gr.conj().T @ S))
        assert np.isclose(W_beta_dense(g, 0.3, b, eta), dev, atol=1e-12)


def test_reference_choice_invariance():
    """The reference-equilibrium plus window split must not depend on which
    contact is the reference, once the kernel sum's O(eta) term is included.
    Reference and window share the integrator's broadening; the grid resolves it."""
    from gauNEGF.density import densityGrid
    from gauNEGF.integrate import ETA as INT_ETA
    g = make_1d_nonortho_chain()
    eta = max(g.eta, INT_ETA)                  # match the integrator
    mu1, mu2 = -0.25, 0.4
    dE = eta / 10.0
    grid = np.arange(-3.0, mu2 + dE / 2, dE)
    dos_vals = np.array([dos_qsym(g, E, eta) for E in grid])

    def neq(mu, stride=1):
        # half-spacing slack keeps mu itself on the node set at either stride
        gg = grid[::stride]
        m = gg <= mu + dE * stride / 2
        return np.trapz(dos_vals[::stride][m], gg[m])

    win = (grid >= mu1 - dE / 2)
    rvals = np.array([_eta_residual(g, E, eta) for E in grid[win]])

    def i_resid(stride=1):
        return np.trapz(rvals[::stride], grid[win][::stride]) / (2 * np.pi)

    quad_err = (abs((neq(mu1) - neq(mu2)) - (neq(mu1, 2) - neq(mu2, 2)))
                + abs(i_resid() - i_resid(2)) + 1e-12)
    # reference mu1, kernel W_2; then reference mu2, kernel W_1 with the mus
    # swapped so the (f - f_ref) weight flips with the kernel label
    Pw, dNw = densityGrid(np.array(g.F), np.array(g.S), g, mu1, mu2,
                          ind=-1, T=0.0, tol=1e-8)
    n_ref1 = neq(mu1) + np.trace(Pw @ np.array(g.S)).real + dNw
    Pw2, dNw2 = densityGrid(np.array(g.F), np.array(g.S), g, mu2, mu1,
                            ind=0, T=0.0, tol=1e-8)
    n_ref2 = neq(mu2) + np.trace(Pw2 @ np.array(g.S)).real + dNw2
    assert abs(n_ref1 - n_ref2 - i_resid()) <= 10 * quad_err, (
        n_ref1, n_ref2, i_resid(), quad_err)


def test_calculate_current_grid_covers_both_bias_signs():
    """The L/R labeling is numerically inert for |I| -- this pins the
    grid construction, which is where a naive convention flip breaks."""
    from gauNEGF.transport import calculate_current
    from test_transport_checkpointing import setup_energy_independent_test

    F, S, sigma_calc = setup_energy_independent_test('small_nanowire')
    fermi, qV = 0.0, 0.4

    for T in (0, 300):
        i_pos = calculate_current(F, S, sigma_calc, fermi=fermi, qV=qV,
                                   T=T, spin='r')
        i_neg = calculate_current(F, S, sigma_calc, fermi=fermi, qV=-qV,
                                   T=T, spin='r')
        assert np.isfinite(i_pos) and np.isfinite(i_neg)
        assert np.isclose(i_pos, -i_neg, rtol=1e-6)
