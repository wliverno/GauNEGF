"""Structural invariants of the lesser Green's function G^< (2026-09-01).

Encodes two boxed statements of paper/derivations/derivation.tex, on the
two block families the Mulliken count uses.

DEVICE BLOCK -- eqn "density" (derivation.tex:291-296):
    G^<_D = i sum_beta f_beta G^r_D Gamma_beta G^a_D
Gamma_beta = i(Sigma_beta - Sigma_beta^dag) is Hermitian by construction
and PSD for a genuine retarded self-energy, and G^a_D = (G^r_D)^dag, so
-i G^<_D is HERMITIAN and POSITIVE SEMI-DEFINITE.  Hermiticity forces the
diagonal of G^<_D to be purely imaginary -- the "no spurious real part"
statement, on the block where it is actually true.  It holds with or
without SOC: a complex Hermitian F changes nothing in the argument.

CROSS BLOCK -- eqn "GlessDa" (derivation.tex:318-322):
    G^<_{D alpha} = -[ G^r_D tau_{D alpha} g^<_alpha
                     + G^<_D tau_{D alpha} g^a_alpha ]
This block is a d_D x (contact) RECTANGLE, so Hermiticity does not apply,
and elementwise "purely imaginary" is false as soon as H is complex
Hermitian (derivation.tex:70 -- which is exactly why the Mulliken partial
trace keeps Re).  Its structural statement is the EQUILIBRIUM COLLAPSE,
eqn "collapse" (derivation.tex:327-333):
    f_L = f_R = f  =>  G^<_{XY} = i f A_{XY}
That is what is asserted here, together with the finite-bias case where it
correctly does NOT hold, so the equilibrium assertion is not vacuous.

Production paths covered on the device block (both, deliberately):
  * integrate._gless_cross_matrix_ops (integrate.py:366-373), the window
    kernel: builds Gless = Gr @ gamma @ Gr.conj().T from ONE solve, so it
    is Hermitian by construction and PSD is the only content there.
  * integrate._gless_matrix_ops (integrate.py:136-145), the density
    kernel: builds G^r and G^a from TWO INDEPENDENT linear solves, so its
    anti-Hermiticity is only NUMERICAL, not structural.  That is the path
    where a real defect could appear, and where density.py:1519's
    np.real(Pw) would silently discard it.

Cross blocks are never materialized in production (they are contracted
into Q_fwd/Q_rev/Q_sym before leaving surfG), so the (D,L)/(D,R) leg uses
a reference construction built HERE from the same surfG object's own
tau / g_surf, gated by a "bridge" assertion that the reconstruction
reproduces production's own Sigma before any structural claim is made on
it.

Systems, none of which need gdv/gauopen:
  (a) toy 1D chain via surfG1D, sharing the toy construction of
      test_cross_term.py / test_window_cross_term.py;
  (b) the BETHE lattice via surfGBethe.surfGBAt, built from the repo's
      Au.bethe parameter file;
  (c) the Bethe lattice in the DEVICE basis via surfGBethe.surfGB, built
      from a MagicMock 'bar' over real .gjf coordinates (the recipe of
      test_bethe_direction_exclusion.py:169-207), which exercises the
      per-atom vacant-direction assembly of surfGBethe.py:499-582 that the
      surfGBAt-level test skips.

RED evidence lives in this file as explicit negative controls (the suite's
existing style, cf. test_window_cross_term.py:392's contact-label swap):
every criterion used by a green test is re-run against a deliberately
perturbed input and asserted to FLAG it.  No production file was modified
to make any of this testable.

Tolerances below were calibrated against measured defects (job 39428673),
not guessed; each states what sets it.
"""
import contextlib
import io
import os
import shutil
import sys

import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gauNEGF import integrate
from gauNEGF.surfGBethe import surfGB, surfGBAt

from test_cross_term import make_1d_nonortho_chain
from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GEOM_PATH = os.path.join(REPO, 'paper', 'geometries', 'Au20AntPDT.gjf')
BETHE_SRC_PATH = os.path.join(REPO, 'Au.bethe')

# --- criteria tolerances (all RELATIVE; see the three helpers below) -------
# HERM_TOL: max|M - M^dag| / max(1,max|M|).  The window kernel is exactly
# Hermitian (single solve).  The density kernel's defect is pure float64
# roundoff from two independent solves: measured worst case 2.8e-16 on the
# 1D chain and 1.5e-16 on the Bethe surface atom, so 1e-11 leaves five
# decades of headroom and is still far tighter than any physical defect.
HERM_TOL = 1e-11
# PSD_TOL: most negative eigenvalue of Herm(M) / largest |eigenvalue|.
# Gamma_beta is routinely rank deficient (a single Bethe branch broadens
# only part of the 9-orbital block), so its zero modes sit at +-roundoff:
# measured worst case -2.3e-16 (1D) and exactly 0 (Bethe).
PSD_TOL = 1e-10
# EXACT_TOL: identities that are exact float algebra given identical
# inputs -- the collapse, the spectral rewrite, the eta sum rule.  All
# three measured at or below 1.1e-16 relative.
EXACT_TOL = 1e-11
# Convergence for the Bethe surface fixed point.  Tight because the bridge
# assertion compares a test-side reconstruction of g_surf against
# production's Sigma and must not be limited by the iteration.
TIGHT_CONV = 1e-12
# Bridge tolerance for that Bethe reconstruction: set by the residual of
# the mixed fixed point at TIGHT_CONV, not by float precision (measured
# 5.3e-13, flat in eta).
BETHE_BRIDGE_TOL = 1e-10
# Broadening used for both Bethe systems.  Large enough that the
# surface recursion and the 54x54 device-basis solves stay well
# conditioned, small enough to be a physically usable value.
BETHE_ETA = 1e-4
# Negative controls must be flagged by a wide margin, not marginally.
CONTROL_MARGIN = 1e-3


# ---------------------------------------------------------------------------
# Criteria.  The green tests and the negative controls call the SAME
# functions -- that sharing is what makes a control evidence about the
# criterion rather than about a parallel implementation of it.
# ---------------------------------------------------------------------------

def herm_defect(M):
    """max|M - M^dag| / max(1, max|M|).  Zero iff M is Hermitian."""
    M = np.asarray(M)
    return float(np.max(np.abs(M - M.conj().T))) / max(1.0, float(np.max(np.abs(M))))


def psd_defect(M):
    """Most negative eigenvalue of Herm(M), over the largest |eigenvalue|.
    Non-negative up to roundoff iff M is positive semi-definite."""
    M = np.asarray(M)
    H = 0.5 * (M + M.conj().T)
    w = np.linalg.eigvalsh(H)
    return float(w[0]) / max(1.0, float(np.max(np.abs(w))))


def rel_defect(A, B):
    """max|A - B| / max(1, max|B|)."""
    A = np.asarray(A)
    B = np.asarray(B)
    return float(np.max(np.abs(A - B))) / max(1.0, float(np.max(np.abs(B))))


def check_device_block(minus_i_gless, tag, herm_tol=HERM_TOL, psd_tol=PSD_TOL):
    """The eqn-'density' criterion on -i G^<_D: Hermitian, PSD, purely
    imaginary on the diagonal of G^<_D itself, and not identically zero."""
    M = np.asarray(minus_i_gless)
    assert float(np.max(np.abs(M))) > 0.0, \
        f'{tag}: -i G^<_D is identically zero (vacuous)'
    hd = herm_defect(M)
    assert hd < herm_tol, f'{tag}: -i G^<_D not Hermitian, defect {hd:.3e}'
    pd = psd_defect(M)
    assert pd > -psd_tol, f'{tag}: -i G^<_D not PSD, min eig {pd:.3e}'
    # Hermiticity => real diagonal of -i G^<_D => purely imaginary diagonal
    # of G^<_D.  Asserted separately: it is the "no spurious real part"
    # question in its own right, and it is what density.py:1519 discards.
    d = np.diag(M)
    assert np.max(np.abs(d.imag)) <= herm_tol * max(1.0, float(np.max(np.abs(d)))), \
        f'{tag}: G^<_D carries a real part on the diagonal'


# ---------------------------------------------------------------------------
# Device block, through both production kernels
# ---------------------------------------------------------------------------

KERNELS = (('window(_gless_cross_matrix_ops)', integrate._gless_cross_matrix_ops),
           ('density(_gless_matrix_ops)', integrate._gless_matrix_ops))


def minus_i_gless_device(kernel, sig_list, E, F, S, eta, occs):
    """-i G^<_D = sum_beta f_beta (G^r Gamma_beta G^a), assembled from the
    PRODUCTION per-contact kernel, which returns exactly G^r Gamma_b G^a
    (the leading i and the f_beta are applied by the kernels' callers)."""
    sigTot = sum(sig_list)
    out = None
    for f, sb in zip(occs, sig_list):
        res = kernel(jnp.array(sb), jnp.array(sigTot), float(E),
                     jnp.array(F), jnp.array(S), float(eta))
        m = np.array(res[0] if isinstance(res, tuple) else res)
        out = f * m if out is None else out + f * m
    return out


def _device_block_sweep(sig_of, F, S, eta, energies, occs, tag):
    for E in energies:
        sig_list = [np.array(s) for s in sig_of(E)]
        for kname, kern in KERNELS:
            M = minus_i_gless_device(kern, sig_list, E, F, S, eta, occs)
            check_device_block(M, f'{tag} E={E:+.4f} {kname}')


def test_device_block_hermitian_psd_1d_chain():
    """eqn 'density' (derivation.tex:291-296) on the toy 1D chain, through
    BOTH production kernels.  The window kernel is Hermitian by
    construction; the density kernel's two independent solves are the case
    with real content."""
    g = make_1d_nonortho_chain()
    _device_block_sweep(lambda E: [g.sigma(E, i) for i in range(g.num_contacts)],
                        np.array(g.F), np.array(g.S), g.eta,
                        np.linspace(-1.2, 1.4, 7), occs=[0.83, 0.41],
                        tag='1d-chain')


def test_device_block_hermitian_psd_1d_complex_hamiltonian():
    """Same statement with a complex Hermitian F and complex tau/stau (the
    SOC-shaped toy of test_window_cross_term.py:53-95).  derivation.tex:70:
    complex H kills elementwise imaginarity of the CROSS block, but the
    device-block Hermiticity of eqn 'density' is untouched by it."""
    g = soc_shaped_chain()
    F = np.array(g.F)
    assert np.allclose(F, F.conj().T), 'toy F must be Hermitian'
    assert not np.allclose(F, F.real), 'toy F must be genuinely complex'
    _device_block_sweep(lambda E: [g.sigma(E, i) for i in range(g.num_contacts)],
                        F, np.array(g.S), g.eta,
                        np.linspace(-0.8, 0.9, 5), occs=[0.7, 0.2],
                        tag='1d-soc')


def test_device_block_hermitian_psd_bethe_surface_atom(bethe_atom):
    """eqn 'density' on the BETHE lattice: the surface atom (F = H0, S = I)
    with its nine semi-infinite Bethe branches as the contacts beta, each
    branch self-energy taken straight from production's sigmaSurf."""
    gB = bethe_atom
    H0 = np.array(gB.H0)
    I = np.eye(gB.dim)
    occs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for E in BETHE_ENERGIES:
        sigS = np.array(gB.sigmaSurf(E, TIGHT_CONV))
        assert np.max(np.abs(sigS)) > 0.0, f'Bethe Sigma vanishes at E={E}'
        for kname, kern in KERNELS:
            M = minus_i_gless_device(kern, list(sigS), E, H0, I, gB.eta, occs)
            check_device_block(M, f'bethe-atom E={E:+.2f} {kname}')


def test_device_block_hermitian_psd_bethe_device_basis(bethe_device):
    """eqn 'density' on the Bethe lattice AFTER the device-basis assembly
    of surfGB.sigma (surfGBethe.py:499-582): the per-atom vacant-direction
    sum scattered into the full basis.  F is a random Hermitian matrix --
    the kernels take F as an argument, so no production state is touched."""
    g = bethe_device
    N = g.F.shape[0]
    rng = np.random.default_rng(20260901)
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    F = 0.5 * (A + A.conj().T)
    S = np.eye(N)
    for E in (0.5, 2.84):
        sig_list = [np.array(g.sigma(E, i)) for i in range(g.num_contacts)]
        assert np.max(np.abs(sig_list[0])) > 0.0
        occs = [0.75] * len(sig_list)
        for kname, kern in KERNELS:
            M = minus_i_gless_device(kern, sig_list, E, F, S, g.eta, occs)
            check_device_block(M, f'bethe-device E={E:+.2f} {kname}')


# ---------------------------------------------------------------------------
# Cross block: reference construction from the surfG object's own tau/g_surf
# ---------------------------------------------------------------------------

def cross_reference_1d(g, E, ind, occs, bad_conj=False):
    """Assemble every object eqn 'GlessDa' (derivation.tex:318-322) needs
    from surfG1D's OWN tau/stau/CList/g_surf, using the same expressions
    surfG1D.sigma (surfG1D.py:500-548) uses, at the same default conv.

    'bridge' is the gate: tau_D a_alpha tau_D^dag must reproduce
    production's Gamma_ind before anything else here is trusted.
    bad_conj=True is the negative control -- g^r in place of g^a in the
    second term of eqn 'GlessDa'.
    """
    N = g.F.shape[0]
    F = np.array(g.F)
    S = np.array(g.S)
    eta = float(g.eta)
    sigs = [np.array(g.sigma(E, i)) for i in range(g.num_contacts)]
    gams = [1j * (s - s.conj().T) for s in sigs]
    Gr = np.linalg.inv((E + 1j * eta) * S - F - sum(sigs))
    Ga = Gr.conj().T

    # tau_{D alpha}: the EOM coupling block (device rows x contact-cell
    # columns), exactly surfG1D.sigma's t_reg scattered into device rows.
    inds = np.array(g.indsList[ind])
    tau = np.array(g.tauList[ind])
    stau = np.array(g.stauList[ind])
    n = len(g.aList[ind])
    C_mid = np.array(g.CList[ind])[:n, :n]
    t_reg = (E * stau - tau) @ C_mid
    tauD = np.zeros((N, t_reg.shape[1]), dtype=complex)
    tauD[inds, :] = t_reg

    g_r = np.array(g.g(E, ind))
    g_a = g_r.conj().T
    a_al = 1j * (g_r - g_a)                        # eqn 'abetalimit'

    A_D = sum(Gr @ gm @ Ga for gm in gams)         # eqn 'AD'
    Gless_D = 1j * sum(f * (Gr @ gm @ Ga) for f, gm in zip(occs, gams))
    g_less = 1j * occs[ind] * a_al                 # eqn 'gless'

    second = g_r if bad_conj else g_a
    Gless_Da = -(Gr @ tauD @ g_less + Gless_D @ tauD @ second)
    A_Da = -(Gr @ tauD @ a_al + A_D @ tauD @ g_a)  # beta-sum of eqn 'ADalphabeta'

    return dict(S=S, eta=eta, Gr=Gr, Ga=Ga, tauD=tauD, g_r=g_r, g_a=g_a,
                a=a_al, A_D=A_D, A_Da=A_Da, Gless_D=Gless_D, Gless_Da=Gless_Da,
                bridge=rel_defect(tauD @ a_al @ tauD.conj().T, gams[ind]))


def test_cross_block_equilibrium_collapse_1d():
    """eqn 'collapse' (derivation.tex:327-333) on the (D,L) and (D,R)
    rectangles: with f_L = f_R = f, G^<_{D alpha} = i f A_{D alpha}."""
    g = make_1d_nonortho_chain()
    f = 0.37
    for E in np.linspace(-1.2, 1.4, 7):
        for ind in range(g.num_contacts):
            r = cross_reference_1d(g, E, ind, [f] * g.num_contacts)
            assert r['bridge'] < EXACT_TOL, \
                f'reconstruction misses production Sigma: {r["bridge"]:.3e}'
            assert np.max(np.abs(r['A_Da'])) > 0.0, 'A_{D alpha} vanishes'
            d = rel_defect(r['Gless_Da'], 1j * f * r['A_Da'])
            assert d < EXACT_TOL, f'collapse fails E={E:+.3f} ind={ind}: {d:.3e}'


def test_cross_block_collapse_fails_under_bias_1d():
    """Non-vacuity gate for the test above: with f_L != f_R the collapse
    identity must NOT hold, for any single f.  derivation.tex:'master' --
    the decomposition stops being irrelevant exactly when f_L != f_R."""
    g = make_1d_nonortho_chain()
    fL, fR = 0.9, 0.1
    worst = 0.0
    for E in np.linspace(-1.2, 1.4, 7):
        for ind in range(g.num_contacts):
            r = cross_reference_1d(g, E, ind, [fL, fR])
            for f in (fL, fR, 0.5 * (fL + fR)):
                worst = max(worst, rel_defect(r['Gless_Da'], 1j * f * r['A_Da']))
    assert worst > CONTROL_MARGIN, \
        f'finite bias did not break the collapse ({worst:.3e}); the ' \
        'equilibrium assertion would be vacuous'


def test_cross_block_spectral_rewrite_1d():
    """The one-line check derivation.tex makes on eqn 'ADalphabeta': taking
    A_D as i(G^r_D - G^a_D), the beta-sum A_{D alpha} equals
    i(G^r_{D alpha} - G^a_{D alpha}) with G^r_{D alpha} = -G^r_D tau g^r
    (eqn 'elim') and G^a_{D alpha} = -G^a_D tau g^a (eqn 'elimadv')."""
    g = make_1d_nonortho_chain()
    for E in np.linspace(-1.2, 1.4, 7):
        for ind in range(g.num_contacts):
            r = cross_reference_1d(g, E, ind, [0.5, 0.5])
            A_D_x = 1j * (r['Gr'] - r['Ga'])
            A_Da_x = -(r['Gr'] @ r['tauD'] @ r['a'] + A_D_x @ r['tauD'] @ r['g_a'])
            Gr_Da = -r['Gr'] @ r['tauD'] @ r['g_r']
            Ga_Da = -r['Ga'] @ r['tauD'] @ r['g_a']
            d = rel_defect(A_Da_x, 1j * (Gr_Da - Ga_Da))
            assert d < EXACT_TOL, f'spectral rewrite fails E={E:+.3f}: {d:.3e}'


def test_device_spectral_sum_rule_carries_exactly_one_eta_term_1d():
    """Why the collapse test uses A_D = G^r Gamma G^a and not i(G^r - G^a):
    at finite eta the two differ by EXACTLY 2 eta G^r S G^a, because the
    +i eta S regulator sits on the whole matrix (cf. the _eta_residual of
    test_window_cross_term.py).  Pinning it closes the only gap between the
    two A_D conventions used above."""
    g = make_1d_nonortho_chain()
    for E in np.linspace(-1.2, 1.4, 7):
        r = cross_reference_1d(g, E, 0, [0.5, 0.5])
        lhs = 1j * (r['Gr'] - r['Ga'])
        rhs = r['A_D'] + 2 * r['eta'] * (r['Gr'] @ r['S'] @ r['Ga'])
        assert rel_defect(lhs, rhs) < EXACT_TOL


# --- Bethe cross block -----------------------------------------------------

PLANE_DIRS = (0, 1, 2, 6, 7, 8)   # surfGBethe.py:1141 planeVec


def cross_reference_bethe(gB, E, k, occs, bad_conj=False):
    """Same assembly for the BETHE lattice, one branch direction k as the
    contact alpha.  'Device' = the surface atom (F = H0, S = I);
    tau_{D alpha} = B_k = E_eff S_k - V_k and g^r_alpha = the neighbour
    Green's function, both rebuilt here from the object's OWN Slist/Vlist0
    and its returned sigmaSurf, with the identical expressions of
    surfGBethe.py:1136-1148 and :1180-1196.

    Restricted to PLANE_DIRS: those are the only directions sigmaSurf's
    fixed point updates, hence the only ones for which
    sigSurf[k] = B_k g_k B_bar_k holds -- which 'bridge' asserts.

    PRODUCTION NOTE: B_bar_k carries the COMPLEX E_eff on both sides
    (surfGBethe.py:1148), so B_bar_k = B_k^dag + 2i eta S_k^dag.  That is
    the analytic continuation of Sigma^r to complex z, not a defect, but it
    means production's Gamma_k = i(Sigma_k - Sigma_k^dag) differs from the
    derivation's tau a tau^dag at O(eta).  This reference uses the
    derivation's tau^dag so the collapse stays an exact identity;
    'gamma_gap' measures the difference and has its own test below.
    """
    assert (k + 6) % 12 < 9, 'only PLANE_DIRS have a surface fixed point'
    dim = gB.dim
    eta = float(gB.eta)
    E_eff = E + 1j * eta
    H0 = np.array(gB.H0)
    sigS = np.array(gB.sigmaSurf(E, TIGHT_CONV))
    sigTot = sigS.sum(axis=0)
    A = E_eff * np.eye(dim) - H0
    Gr = np.linalg.inv(A - sigTot)
    Ga = Gr.conj().T
    gams = [1j * (s - s.conj().T) for s in sigS]

    g_r = np.linalg.inv(A - sigTot + sigS[(k + 6) % 12])
    g_a = g_r.conj().T
    a_al = 1j * (g_r - g_a)
    B_k = E_eff * np.array(gB.Slist[k]) - np.array(gB.Vlist0[k])
    B_bar = (E_eff * np.array(gB.Slist[k]).conj().T
             - np.array(gB.Vlist0[k]).conj().T)

    A_D = sum(Gr @ gm @ Ga for gm in gams)
    Gless_D = 1j * sum(f * (Gr @ gm @ Ga) for f, gm in zip(occs, gams))
    g_less = 1j * occs[k] * a_al
    second = g_r if bad_conj else g_a
    Gless_Da = -(Gr @ B_k @ g_less + Gless_D @ B_k @ second)
    A_Da = -(Gr @ B_k @ a_al + A_D @ B_k @ g_a)

    return dict(Gr=Gr, Ga=Ga, A_D=A_D, A_Da=A_Da, Gless_Da=Gless_Da,
                bridge=rel_defect(B_k @ g_r @ B_bar, sigS[k]),
                gamma_gap=rel_defect(gams[k], B_k @ a_al @ B_k.conj().T))


def test_cross_block_equilibrium_collapse_bethe(bethe_atom):
    """eqn 'collapse' on the Bethe lattice, one semi-infinite branch as the
    contact alpha, gated by the bridge assertion that the test-side g_surf
    reconstruction reproduces production's own sigmaSurf[k]."""
    f = 0.44
    for E in BETHE_ENERGIES:
        for k in PLANE_DIRS:
            r = cross_reference_bethe(bethe_atom, E, k, [f] * 9)
            assert r['bridge'] < BETHE_BRIDGE_TOL, \
                f'sigmaSurf[{k}] != B g B_bar at E={E}: {r["bridge"]:.3e}'
            assert np.max(np.abs(r['A_Da'])) > 0.0
            d = rel_defect(r['Gless_Da'], 1j * f * r['A_Da'])
            assert d < EXACT_TOL, f'Bethe collapse fails E={E} k={k}: {d:.3e}'


def test_cross_block_collapse_fails_under_bias_bethe(bethe_atom):
    """Non-vacuity gate for the Bethe collapse: unequal branch occupations
    must break it."""
    occs = [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6]
    worst = 0.0
    for E in BETHE_ENERGIES:
        for k in (0, 6):
            r = cross_reference_bethe(bethe_atom, E, k, occs)
            for f in (0.05, 0.5, 0.95):
                worst = max(worst, rel_defect(r['Gless_Da'], 1j * f * r['A_Da']))
    assert worst > CONTROL_MARGIN, \
        f'unequal occupations did not break the Bethe collapse ({worst:.3e})'


def test_bethe_gamma_matches_dagger_form_to_order_eta(bethe_atom_factory):
    """PRODUCTION OBSERVATION, pinned.  surfGBethe.py:1147-1148 builds
    B_bar_k with the COMPLEX E_eff on both sides, so B_bar_k != B_k^dag and
    production's Gamma_k differs from the derivation's tau a tau^dag at
    O(eta).  Gamma stays Hermitian either way (i(X - X^dag) always is), so
    the device-block test is untouched; this is the analytic continuation
    of Sigma^r to complex z, not a defect.  Asserted as: the gap shrinks
    at least proportionally to eta."""
    gaps = {}
    for eta in (1e-3, 1e-4, 1e-5):
        gB = bethe_atom_factory(eta)
        gaps[eta] = max(cross_reference_bethe(gB, 2.84, k, [0.5] * 9)['gamma_gap']
                        for k in PLANE_DIRS)
    assert gaps[1e-3] > 0.0, gaps
    assert gaps[1e-4] < 0.2 * gaps[1e-3], gaps
    assert gaps[1e-5] < 0.2 * gaps[1e-4], gaps


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS -- the RED evidence.  Each re-runs a criterion used by a
# green test against a deliberately perturbed input and asserts the
# criterion FLAGS it.  No production code is modified anywhere.
# ---------------------------------------------------------------------------

def test_control_nonhermitian_gamma_is_flagged():
    """RED for check_device_block's Hermiticity leg.  The production
    kernels build gamma = i(sig - sig^dag), which is Hermitian for ANY sig,
    so a non-Hermitian Gamma cannot be pushed through them; it is injected
    into a numpy replica of integrate.py:136-145 instead.  A small REAL,
    unmirrored asymmetry on Gamma is exactly the defect that would leave a
    spurious real part in G^<_D."""
    g = make_1d_nonortho_chain()
    F = np.array(g.F)
    S = np.array(g.S)
    eta = float(g.eta)
    E = 0.31
    sigs = [np.array(g.sigma(E, i)) for i in range(g.num_contacts)]
    Gr = np.linalg.inv((E + 1j * eta) * S - F - sum(sigs))
    Ga = Gr.conj().T
    gam = 1j * (sigs[0] - sigs[0].conj().T)
    # baseline: the unperturbed replica passes the green test's criterion
    check_device_block(Gr @ gam @ Ga, 'control-baseline')
    bad = gam.copy()
    bad[0, gam.shape[0] - 1] += 1e-2 * np.max(np.abs(gam))
    assert herm_defect(bad) > CONTROL_MARGIN
    with pytest.raises(AssertionError, match='not Hermitian'):
        check_device_block(Gr @ bad @ Ga, 'control-nonhermitian-gamma')


def test_control_advanced_self_energy_is_flagged_as_non_psd():
    """RED for check_device_block's PSD leg, driven THROUGH the production
    window kernel: handing integrate._gless_cross_matrix_ops the ADVANCED
    self-energy sig^dag in place of sig flips the sign of gamma, so
    G^r gamma G^r-dag turns negative semi-definite.  Hermiticity survives
    (structural in that kernel) and only the PSD leg fires -- which is why
    both legs are asserted separately."""
    g = make_1d_nonortho_chain()
    F = np.array(g.F)
    S = np.array(g.S)
    eta = float(g.eta)
    E = 0.31
    sigs = [np.array(g.sigma(E, i)) for i in range(g.num_contacts)]
    sigTot = sum(sigs)
    good, _ = integrate._gless_cross_matrix_ops(
        jnp.array(sigs[0]), jnp.array(sigTot), E, jnp.array(F), jnp.array(S), eta)
    check_device_block(np.array(good), 'control-baseline-window')
    bad, _ = integrate._gless_cross_matrix_ops(
        jnp.array(sigs[0].conj().T), jnp.array(sigTot), E,
        jnp.array(F), jnp.array(S), eta)
    bad = np.array(bad)
    assert herm_defect(bad) < HERM_TOL, 'window kernel stays Hermitian by design'
    assert psd_defect(bad) < -CONTROL_MARGIN, \
        'advanced self-energy did not produce a negative semi-definite block'
    with pytest.raises(AssertionError, match='not PSD'):
        check_device_block(bad, 'control-advanced-sigma')


def test_control_wrong_conjugation_in_cross_block_is_flagged_1d():
    """RED for the collapse criterion.  eqn 'GlessDa' carries g^a_alpha in
    its second term (the advanced contact propagator, derivation.tex:319);
    substituting g^r_alpha is the one-character error the collapse test
    exists to catch."""
    g = make_1d_nonortho_chain()
    f = 0.37
    worst = 0.0
    for E in np.linspace(-1.2, 1.4, 5):
        for ind in range(g.num_contacts):
            good = cross_reference_1d(g, E, ind, [f] * g.num_contacts)
            bad = cross_reference_1d(g, E, ind, [f] * g.num_contacts, bad_conj=True)
            assert rel_defect(good['Gless_Da'], 1j * f * good['A_Da']) < EXACT_TOL
            worst = max(worst, rel_defect(bad['Gless_Da'], 1j * f * good['A_Da']))
    assert worst > CONTROL_MARGIN, f'wrong conjugation not flagged ({worst:.3e})'


def test_control_wrong_conjugation_in_cross_block_is_flagged_bethe(bethe_atom):
    """The same negative control on the Bethe lattice."""
    f = 0.44
    worst = 0.0
    for k in (0, 6):
        good = cross_reference_bethe(bethe_atom, 2.84, k, [f] * 9)
        bad = cross_reference_bethe(bethe_atom, 2.84, k, [f] * 9, bad_conj=True)
        assert rel_defect(good['Gless_Da'], 1j * f * good['A_Da']) < EXACT_TOL
        worst = max(worst, rel_defect(bad['Gless_Da'], 1j * f * good['A_Da']))
    assert worst > CONTROL_MARGIN, f'wrong conjugation not flagged ({worst:.3e})'


# ---------------------------------------------------------------------------
# Toy / fixture construction
# ---------------------------------------------------------------------------

# Au d-band sits near 0 eV and the bulk Fermi benchmark is 2.84 eV
# (test_bethe_cross_term_fermi.py); these span occupied and empty states.
BETHE_ENERGIES = (-1.0, 0.5, 2.84, 5.0)


def soc_shaped_chain():
    """The complex-Hermitian toy of test_window_cross_term.py:53-95."""
    from gauNEGF.surfG1D import surfG
    N, nc = 4, 2
    F = jnp.diag(jnp.linspace(-1.0, 1.0, N)) + 0.0j
    for i in range(N - 1):
        F = F.at[i, i + 1].set(-0.3)
        F = F.at[i + 1, i].set(-0.3)
    A = np.array([[0.0, 0.05, -0.02, 0.01],
                  [-0.05, 0.0, 0.03, -0.01],
                  [0.02, -0.03, 0.0, 0.04],
                  [-0.01, 0.01, -0.04, 0.0]])
    F = F + 1j * jnp.array(A)
    S = jnp.eye(N) + 0.0j
    for i in range(N - 1):
        S = S.at[i, i + 1].set(0.1)
        S = S.at[i + 1, i].set(0.1)
    indsList = [jnp.array(list(range(nc))), jnp.array(list(range(N - nc, N)))]
    dtau = jnp.array([[0.05j, -0.03j], [0.02j, -0.04j]])
    dstau = jnp.array([[0.02j, -0.01j], [0.015j, -0.02j]])
    taus = [jnp.ones((nc, nc), dtype=complex) * -0.3 + dtau] * 2
    staus = [jnp.ones((nc, nc), dtype=complex) * 0.1 + dstau] * 2
    alphas = [jnp.diag(jnp.array([-0.5] * nc, dtype=complex)),
              jnp.diag(jnp.array([0.5] * nc, dtype=complex))]
    betas = [jnp.ones((nc, nc), dtype=complex) * -0.3] * 2
    aOverlaps = [jnp.eye(nc, dtype=complex) * 1.05] * 2
    bOverlaps = [jnp.ones((nc, nc), dtype=complex) * 0.1] * 2
    return surfG(F, S, indsList, taus=taus, staus=staus, alphas=alphas,
                 betas=betas, aOverlaps=aOverlaps, bOverlaps=bOverlaps, eta=1e-3)


@pytest.fixture(scope='module')
def bethe_atom_factory():
    """surfGBAt for Au at a requested eta, from the repo's Au.bethe.  No
    gdv: read_bethe_params/construct_mat/gen_fcc_111_neighbors are the same
    pure-numpy helpers test_bethe_cross_term_fermi.py uses.  calcFermi is
    deliberately NOT run -- these are structural identities and the Fermi
    level only shifts E rigidly."""
    _, H0, Sdict, Vdict = read_bethe_params('Au')
    vecs = gen_fcc_111_neighbors()
    Vlist = [construct_mat(Vdict, v) for v in vecs]
    Slist = [construct_mat(Sdict, v) for v in vecs]
    cache = {}

    def make(eta):
        if eta not in cache:
            cache[eta] = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
        return cache[eta]
    return make


@pytest.fixture(scope='module')
def bethe_atom(bethe_atom_factory):
    return bethe_atom_factory(BETHE_ETA)


@pytest.fixture(scope='module')
def bethe_device(tmp_path_factory):
    """surfGB in the DEVICE basis, over the six outer contact atoms of
    paper/geometries/Au20AntPDT.gjf, built from a MagicMock 'bar' -- the
    recipe of test_bethe_direction_exclusion.py:169-207, needing neither
    Gaussian nor gauopen.  Construction runs surfGBAt.calcFermi (a ~30-60 s
    bisection), hence module scope."""
    from unittest.mock import MagicMock
    from test_bethe_direction_exclusion import parse_gjf_coords, PLANE1_ATOMS
    if not os.path.exists(GEOM_PATH):
        pytest.skip(f'geometry file not present: {GEOM_PATH}')
    coords, _ = parse_gjf_coords(GEOM_PATH)
    sub = np.asarray([coords[a - 1] for a in PLANE1_ATOMS], dtype=float)
    natoms = len(sub)
    N = natoms * 9
    bar = MagicMock()
    bar.c = (sub / 0.52917721).flatten()
    bar.ibfatm = np.repeat(np.arange(1, natoms + 1), 9)
    bar.ibftyp = np.zeros(N, dtype=int)
    tmp_dir = str(tmp_path_factory.mktemp('gless_bethe'))
    shutil.copy(BETHE_SRC_PATH, os.path.join(tmp_dir, 'Au.bethe'))
    old = os.getcwd()
    os.chdir(tmp_dir)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = surfGB(np.zeros((N, N)), np.eye(N),
                       [list(range(1, natoms + 1))], bar,
                       latFile='Au', eta=BETHE_ETA)
    finally:
        os.chdir(old)
    return g
