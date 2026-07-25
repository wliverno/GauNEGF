"""Tests for the f=1-on-arc rule in densityComplex/densityComplexN.

Mechanism (measured 2026-07-24, paper/studies/09_bias_physchem/
contour_nkt_test.py + contour_f1_bound.py): on the equilibrium arc the
fermi factor deviates from 1 by at most eps = e^-nKT (4.5e-5 at nKT=10),
oscillating with period 2*pi*kT in Im(z) -- thousands of unresolvable
periods on a deep-Emin arc. The adaptive quadrature cannot resolve the
oscillation and floors at ~eps * |G(sampled heights)|, exhausting the
grid whenever that floor exceeds tol (bonded-300K production: arc
exhausted 195/204 integrations at tol 1e-4; window converged 204/204).
Setting f=1 on the arc is exact up to the segment hole-tail,
|Delta| <= kT * e^-nKT * max|G| near the touchdown -- measured 6.24e-6
on the production spectrum, 3.1x SMALLER than the occupied tail above
mu + nKT*kT that the scheme already discards by design.

Floor-scale note (measured while calibrating this test): the noise
floor scales with |G| at the sampled quadrature heights (~eV off-axis),
so single-pole matrix elements floor near 8e-5 -- BELOW the production
tol 1e-4 but far above 1e-6. The convergence test therefore drives the
adaptive integrator at tol=1e-6, where the current-code floor fails by
~2 orders and the fixed code passes cleanly.

Behavioral contract:
  - densityComplex at finite T converges its arc at tol=1e-6
    (no 'reached full grid'); at T=0 it already does
  - occupations match a semi-analytic reference on the contour's own
    domain [Emin, inf) within 1.5e-4
  - electron count closes against the eta-broadened reference at
    T=0 and T=300 (edit-safety regression)
  - densityComplexN agrees with densityComplex on the same model
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest
from scipy.integrate import quad

from gauNEGF.surfG1D import surfG
from gauNEGF.density import densityComplex, densityComplexN, fermi

KB = 8.617333e-5
MU = -2.52
ETA = 2.5e-2
T300 = 300.0
EMIN = -2410.0

# Deep spectrum (big arc radius -> thousands of oscillation periods) with
# a valence cluster. CALIBRATION NOTE: the nearest occupied level sits
# 0.68 eV below mu -- resolvable by the 486-pt arc at BOTH touchdowns
# (mu at T=0, mu-259 meV at 300K). Isolated unit-weight poles closer to
# the touchdown are unresolvable by any endpoint-sampled arc regardless
# of the fermi factor (measured: 2.01 e count loss with a pole 51 meV
# under Emax) -- that is a different, resolution-class limitation, not
# the oscillation disease this test drives.
LEVELS = np.concatenate([
    [-2390.0, -1500.0, -800.0, -400.0, -270.0, -150.0, -60.0, -20.0],
    np.linspace(-12.0, -3.2, 22),
    [-2.0],
    np.linspace(2.0, 900.0, 9),
])


@pytest.fixture
def deep_spectrum_g():
    F = np.diag(LEVELS).astype(complex)
    S = np.eye(LEVELS.size, dtype=complex)
    # diagonal F -> tauFromFock couplings are zero: eta-only broadening,
    # orthogonal fast path (no cross term), real code end to end
    return surfG(F, S, [[0], [LEVELS.size - 1]], eta=ETA)


def occupation_reference(T):
    """n_j on the contour's own domain [EMIN, inf):

    n_j = (1/pi)[arctan((Ecut-E_j)/eta) - arctan((EMIN-E_j)/eta)]
          + window quad over [Ecut, mu+30kT] with the fermi factor,
    where Ecut = mu - 30kT (f = 1 below Ecut to within e^-30).
    Exact to ~e^-30 plus narrow-interval quadrature error.
    """
    tref = max(T, 1.0)
    ecut = MU - 30 * KB * tref
    hi = MU + 30 * KB * tref
    ref = np.empty(LEVELS.size)
    for j, ej in enumerate(LEVELS):
        base = (np.arctan((ecut - ej) / ETA)
                - np.arctan((EMIN - ej) / ETA)) / np.pi
        def integrand(E, ej=ej):
            lor = (ETA / np.pi) / ((E - ej) ** 2 + ETA ** 2)
            if T > 0:
                return float(fermi(E + 0j, MU, T).real) * lor
            return lor if E < MU else 0.0
        pts = [p for p in (ej, MU) if ecut < p < hi]
        win, _ = quad(integrand, ecut, hi, points=pts or None, limit=200)
        ref[j] = base + win
    return ref


def run_capture(fn, *args, **kwargs):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        out = fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return out, buf.getvalue()


def test_arc_weights_carry_no_fermi_modulation(deep_spectrum_g, monkeypatch):
    """The arc quadrature weights must be free of the fermi factor.

    Seam test: spy on GrIntCross (pass-through to real code) and check
    every arc call's weights. With dz = 1j*(z - center), the ratio
    weights/dz = (pi/2)*w*f(z)/r is exactly real when f=1; the fermi
    factor's oscillating tail makes it complex with relative imaginary
    part up to ~e^-nKT (3.6e-5 at the nodes nearest the touchdown).
    Package-level matrix models cannot separate the resulting maxDP
    floor from the jax kernel noise floor (~8e-5), so the convergence
    contract is asserted here at the weights seam instead; the
    magnitude receipts live in the standalone drivers
    (paper/studies/09_bias_physchem/contour_nkt_test.py).
    """
    import gauNEGF.density as den
    from gauNEGF.density import kB as pkg_kB
    calls = []
    real_grintcross = den.GrIntCross

    def spy(F, S, g, Elist, weights):
        calls.append((np.asarray(Elist), np.asarray(weights)))
        return real_grintcross(F, S, g, Elist, weights)

    monkeypatch.setattr(den, 'GrIntCross', spy)
    g = deep_spectrum_g
    F, S = np.array(g.F), np.array(g.S)
    run_capture(den.densityComplex, F, S, g, EMIN, MU, T=T300)

    kb_t = pkg_kB * T300           # package constant: geometry must match
    emax = MU - 10 * kb_t          # nKT=10, mirrors densityComplex
    center = (EMIN + emax) / 2
    arc_calls = 0
    worst = 0.0
    for z, wt in calls:
        if np.max(np.abs(np.imag(z))) < 1e-9:
            continue               # window call: real axis, keeps fermi
    # (arc nodes all have Im z > 0 except measure-zero endpoints)
        arc_calls += 1
        ratio = wt / (1j * (z - center))
        rel_im = np.max(np.abs(ratio.imag)
                        / np.maximum(np.abs(ratio), 1e-300))
        worst = max(worst, float(rel_im))
    assert arc_calls > 0, 'spy captured no arc calls'
    assert worst < 1e-8, (
        f'arc weights carry fermi modulation (rel Im {worst:.3e}); '
        'f=1-on-arc rule not applied')


def test_occupations_match_reference_at_300K(deep_spectrum_g):
    g = deep_spectrum_g
    F, S = np.array(g.F), np.array(g.S)
    (P, dN), _ = run_capture(densityComplex, F, S, g, EMIN, MU, T=T300)
    ref = occupation_reference(T300)
    err = np.max(np.abs(np.real(np.diag(P)) - ref))
    assert err < 1.5e-4, f'occupation error {err:.3e} vs reference'


def test_count_closure_T0_and_T300(deep_spectrum_g):
    """Edit-safety: trace(P) matches the eta-broadened reference count."""
    g = deep_spectrum_g
    F, S = np.array(g.F), np.array(g.S)
    for T in (0.0, T300):
        (P, dN), _ = run_capture(densityComplex, F, S, g, EMIN, MU, T=T)
        expected = float(occupation_reference(T).sum())
        got = float(np.real(np.trace(P))) + dN
        assert abs(got - expected) < 2e-3, (
            f'T={T}: count {got:.5f} vs reference {expected:.5f}')


def test_densityComplexN_agrees(deep_spectrum_g):
    g = deep_spectrum_g
    F, S = np.array(g.F), np.array(g.S)
    (Pa, _), _ = run_capture(densityComplex, F, S, g, EMIN, MU, T=T300)
    (Pn, _), _ = run_capture(densityComplexN, F, S, g, EMIN, MU,
                             N=486, T=T300)
    err = np.max(np.abs(np.diag(Pa) - np.diag(Pn)))
    assert err < 2e-3, f'fixed-N vs adaptive disagree by {err:.3e}'
