"""Lower-contour method bake-off: densityReal vs densityComplex vs analytic Damle.

Focus (per user): METHOD AGREEMENT, not absolute electron counts (these are
initial-Fock, pre-NEGF-SCF, so absolute ne is not physical). The band [Emin, mu]
is ALWAYS densityComplex and is not a candidate; only the lower region is compared.

Bookkeeping fix: the true count from densityReal/densityComplex is
  tr(P @ S) + delta_N   (delta_N = Mulliken device-lead cross term).
analytic Damle returns NO cross term (subsumed into S_eff), so its count is
tr(P @ S) alone. This run reports tr(P@S), delta_N, and the total for every
method so the cross-term tracking is visible.

damle sign: placement-B numbers suggested P_damle = -P_complex. We test that two
ways: report ||(-P_damle) - P_complex||, and run a FINITE-bound core interval to
tell whether the flip is a global density() convention bug or a log-branch artifact
of the -1e6 lower limit.

Intervals:
  A  pole-free deep tail   [ENERGY_MIN, Emin_deep]   -> expect ~0 (all agree)
  band  finite, poles      [Emin_deep, mu]           -> damle sign check on poles
  B  cores, -1e6 bound     [ENERGY_MIN, Emin_gap]    -> C2 only (1s cores inside)
  C  cores, finite bound   [Emin_deep, Emin_gap]     -> C2 only (isolate sign cause)
"""
import os
import sys
import time
import shutil
import tempfile

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from numpy.linalg import norm, eigvals, eig, inv, solve

from gauNEGF.scfE import NEGFE
from gauNEGF.transport import har_to_eV
from gauNEGF.density import densityReal, densityComplex, density
from gauNEGF.config import ENERGY_MIN, ETA


def time_sync(fn, repeats=3):
    """Warm up once (compile + first surface-GF eval), then median wall time,
    blocking on the result each call so async dispatch cannot lie."""
    out = fn()
    jax.block_until_ready(out)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def damle_full(F_eV, S, Sigma_0, Y_eff, g, lo, hi):
    """Analytic Damle on [lo, hi]: density matrix P (via density()) AND the
    cross-term delta_N (analytic, with Q linearized between two anchors).

    G^R(E) = Y_eff (E I - Fbar)^-1 Y_eff (the linear-Sigma model, since S_eff=S-X
    is baked into Y_eff). With Q(E) ~ Q0 + Q1 E anchored at (hi, 2*hi),
      Tr(G^R Q) = sum_i b_i(E)/(E - D_i),  b(E) = Vc^H Y_eff Q(E) Y_eff V
      delta_N = -(1/pi) Im integral_lo^hi.
    Returns (P_AO, delta_N). Anchors picked near the band bottom where the
    integrand lives; validated against densityComplex's delta_N."""
    N = F_eV.shape[0]
    H_eff = F_eV + Sigma_0 + 1j * ETA * np.eye(N)
    Fbar = Y_eff @ H_eff @ Y_eff
    Gam = (H_eff - H_eff.conj().T) * 1j
    GamBar = Y_eff @ Gam @ Y_eff
    D, V = eig(np.asarray(Fbar))
    Vc = inv(V.conj().T)

    # Density matrix (same path as damleLowerDensity).
    P_orth = np.asarray(density(V, Vc, D, GamBar, float(lo), float(hi)))
    P = np.asarray(Y_eff @ P_orth @ Y_eff)

    # Cross-term: linear Q fit at two anchors near the active region.
    Ea, Eb = float(hi), float(2.0 * hi)
    Qa = g.crossTermQTot(Ea)
    if Qa is None:                      # orthogonal system -> no cross term
        return P, 0.0
    Qa = np.asarray(Qa)
    Qb = np.asarray(g.crossTermQTot(Eb))
    Q1 = (Qb - Qa) / (Eb - Ea)
    Q0 = Qa - Q1 * Ea

    Ml = Vc.conj().T @ Y_eff            # Vc^H Y_eff
    Mr = Y_eff @ V                      # Y_eff V
    b0 = np.diag(Ml @ Q0 @ Mr)
    b1 = np.diag(Ml @ Q1 @ Mr)
    logdiff = np.emath.log(1.0 - hi / D) - np.emath.log(1.0 - lo / D)
    I = np.sum(b1 * (hi - lo) + (b0 + b1 * D) * logdiff)
    delta_N = float(-(1.0 / np.pi) * np.imag(I))
    return P, delta_N


def run_methods(F_eV, S, g, Y_eff, Sigma_0, lo, hi, label, band_norm=None):
    print('  %s : [%.3e, %.3f] eV' % (label, lo, hi))
    specs = [
        ('densityComplex', lambda: densityComplex(F_eV, S, g, lo, hi, T=0)),
        ('densityReal',    lambda: densityReal(F_eV, S, g, lo, hi, T=0)),
        ('damle',          lambda: damle_full(F_eV, S, Sigma_0, Y_eff, g, lo, hi)),
    ]
    res = {}
    print('    %-14s %11s %11s %11s %12s %8s'
          % ('method', 'tr(P S)', 'delta_N', 'total', '||P||', 't(s)'))
    for name, fn in specs:
        try:
            out, t = time_sync(fn)
            P, dN = out
            P = np.asarray(P)
            dN = float(dN)
            trps = float(np.real(np.trace(P @ S)))
            res[name] = P
            print('    %-14s %+11.5f %+11.5f %+11.5f %12.4e %8.4f'
                  % (name, trps, dN, trps + dN, norm(P), t))
        except Exception as e:
            res[name] = None
            print('    %-14s FAILED: %s' % (name, repr(e)[:70]))
    Pc = res.get('densityComplex')
    if Pc is not None:
        nPc = max(norm(Pc), 1e-300)
        for name in ('densityReal', 'damle'):
            P = res.get(name)
            if P is None:
                continue
            d_as = norm(P - Pc)
            line = '    agree %-14s ||P-complex||=%.4e (rel %.3e)' % (name, d_as, d_as / nPc)
            if name == 'damle':
                line += '   ||(-P)-complex||=%.4e (rel %.3e)' % (norm(-P - Pc), norm(-P - Pc) / nPc)
            print(line)
        if band_norm is not None:
            print('    ratio ||P_complex||/||band|| = %.3e' % (norm(Pc) / max(band_norm, 1e-300)))
    return res


def analyze(negf, label):
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    g = negf.g
    Y_eff = np.asarray(negf.Y_eff)
    Sigma_0 = np.asarray(negf.Sigma_0)
    buffer = getattr(negf, 'damle_buffer', None)
    buffer = 20.0 if buffer is None else float(buffer)
    mu = float(negf.fermi)

    ev = np.sort(eigvals(solve(S, F_eV)).real)
    Emin_deep = float(ev[0]) - buffer

    print('\n=== SYSTEM: %s ===' % label)
    print('N=%d, fermi=%.4f eV, ENERGY_MIN=%.3e, buffer=%.2f' % (F_eV.shape[0], mu, ENERGY_MIN, buffer))
    print('eig(inv(S)F): first 6 = %s' % np.array2string(ev[:6], precision=2))
    print('              last 4  = %s' % np.array2string(ev[-4:], precision=2))

    P_band, dN_band = densityComplex(F_eV, S, g, Emin_deep, mu, T=0)
    P_band = np.asarray(P_band)
    band_norm = norm(P_band)
    print('band densityComplex[%.3f, %.3f]: tr(P S)=%.4f, delta_N=%.4f, total=%.4f, ||P||=%.4e'
          % (Emin_deep, mu, float(np.real(np.trace(P_band @ S))), float(dN_band),
             float(np.real(np.trace(P_band @ S))) + float(dN_band), band_norm))

    # A: pole-free deep tail
    run_methods(F_eV, S, g, Y_eff, Sigma_0, ENERGY_MIN, Emin_deep, 'A pole-free tail', band_norm)
    # band, finite, contains poles -> damle sign check on poles
    run_methods(F_eV, S, g, Y_eff, Sigma_0, Emin_deep, mu, 'band finite (poles)', band_norm)

    # B / C: cores inside the tail (only if a deep core/valence gap exists)
    below = ev[ev < mu - 2.0]
    if len(below) >= 2:
        gaps = np.diff(below)
        gi = int(np.argmax(gaps))
        if gaps[gi] > 30.0:
            Emin_gap = 0.5 * (below[gi] + below[gi + 1])
            print('  deep gap %.1f eV between %.1f and %.1f -> Emin_gap=%.2f'
                  % (gaps[gi], below[gi], below[gi + 1], Emin_gap))
            run_methods(F_eV, S, g, Y_eff, Sigma_0, ENERGY_MIN, float(Emin_gap),
                        'B cores, -1e6 bound', band_norm)
            run_methods(F_eV, S, g, Y_eff, Sigma_0, Emin_deep, float(Emin_gap),
                        'C cores, finite bound', band_norm)
        else:
            print('  no deep gap: B/C skipped.')


def setup_c2():
    scratch = tempfile.mkdtemp(prefix='c2_tail_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    return negf


def setup_au3():
    os.chdir(REPO)
    negf = NEGFE(fn='Au3CRENBS', func='b3lyp', basis='chkbasis',
                 route='integral=grid=superfine', spin='g')
    negf.setContactBethe([[1, 2, 3], [1, 2, 3]], 'AuSOC', T=0)
    negf.setVoltage(0.0)
    return negf


def main():
    print('LOWER-CONTOUR METHOD BAKE-OFF v2 (cross-terms + damle sign)')
    analyze(setup_c2(), 'C2-STO3G')
    analyze(setup_au3(), 'Au3-CRENBS-Bethe')
    print('\n=== bake-off v2 complete ===')


if __name__ == '__main__':
    main()
