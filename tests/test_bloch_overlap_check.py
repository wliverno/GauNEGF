"""surfG._checkBlochOverlap: warn iff the infinite-chain Bloch overlap
S(k) = Sa + Sb e^{ik} + Sb^H e^{-ik} is non-positive-definite.

Motivation (2026-07-09): the CRENBS Au chain at 2.68 A has min_k
eig(S(k)) = -0.28 even though the 2-cell overlap block is PD -- the
indefiniteness only appears in the Bloch/long-supercell limit. In that
regime the semi-infinite lead is not passive: Gamma can be indefinite
and transmission unbounded (observed: T ~ 1.8e3 vs a 36-channel
ceiling with SO-ECPs), even though the surface-GF solve converges
exactly. Minimal contact basis is necessary but NOT sufficient; this
check is the sufficient-side diagnostic.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from gauNEGF.surfG1D import surfG


def _build(alpha, beta, salpha, sbeta, eta=1e-6):
    n = len(alpha)
    Fdev = np.block([[alpha, beta], [beta.conj().T, alpha]])
    Sdev = np.block([[salpha, sbeta], [sbeta.conj().T, salpha]])
    return surfG(Fdev, Sdev, [np.arange(n), np.arange(n, 2 * n)],
                 taus=[beta.conj().T, beta],
                 staus=[sbeta.conj().T, sbeta],
                 alphas=[alpha, alpha], aOverlaps=[salpha, salpha],
                 betas=[beta.conj().T, beta],
                 bOverlaps=[sbeta.conj().T, sbeta], eta=eta)


def test_pd_bloch_overlap_is_silent(capsys):
    alpha = np.array([[0.0, 0.05], [0.05, 6.0]])
    beta = np.array([[-1.0, 0.6], [-0.35, -0.3]])
    salpha = np.eye(2)
    sbeta = np.array([[0.12, 0.05], [-0.03, 0.08]])
    # sanity: this toy's S(k) is PD across the BZ
    for k in np.linspace(0, np.pi, 73):
        Sk = salpha + sbeta * np.exp(1j * k) + sbeta.T * np.exp(-1j * k)
        assert np.linalg.eigvalsh(Sk).min() > 0
    _build(alpha, beta, salpha, sbeta)
    out = capsys.readouterr().out
    assert 'non-positive-definite' not in out


def test_non_pd_bloch_overlap_warns(capsys):
    # sbeta large enough that S(k) dips negative at k=0 even though the
    # 2-cell block [[Sa, Sb], [Sb^T, Sa]] stays PD -- the Au-chain
    # signature (block-level check passes, Bloch-level fails).
    alpha = np.array([[0.0, 0.05], [0.05, 6.0]])
    beta = np.array([[-1.0, 0.6], [-0.35, -0.3]])
    salpha = np.eye(2)
    sbeta = np.diag([0.45, 0.05])
    S2 = np.block([[salpha, sbeta], [sbeta.T, salpha]])
    assert np.linalg.eigvalsh(S2).min() > 0, 'fixture must keep 2-cell PD'
    Sk0 = salpha + 2 * sbeta          # S(k=0) has eig 1 - 2*0.45 < 0? no: 1+0.9
    Skpi = salpha - 2 * sbeta         # S(k=pi): 1 - 0.9 = 0.1 > 0 -- adjust
    # use k where it actually dips: for diagonal sbeta the extrema are at
    # 0/pi; make one diagonal entry 0.55 so S(pi) goes negative while the
    # 2-cell block eigenvalues (1 +- 0.55 = 0.45/1.55) stay positive.
    sbeta = np.diag([0.55, 0.05])
    S2 = np.block([[salpha, sbeta], [sbeta.T, salpha]])
    assert np.linalg.eigvalsh(S2).min() > 0, 'fixture must keep 2-cell PD'
    assert np.linalg.eigvalsh(salpha - 2 * sbeta).min() < 0, \
        'fixture must break Bloch PD at k=pi'
    _build(alpha, beta, salpha, sbeta)
    out = capsys.readouterr().out
    assert 'non-positive-definite' in out
    assert 'not a passive system' in out


def test_orthogonal_coupling_is_skipped(capsys):
    alpha = np.array([[0.0, 0.05], [0.05, 6.0]])
    beta = np.array([[-1.0, 0.6], [-0.35, -0.3]])
    # no overlaps supplied at all: bSList is all-zero, check must skip
    n = len(alpha)
    Fdev = np.block([[alpha, beta], [beta.conj().T, alpha]])
    g = surfG(Fdev, np.eye(2 * n), [np.arange(n), np.arange(n, 2 * n)],
              taus=[beta.conj().T, beta],
              alphas=[alpha, alpha], betas=[beta.conj().T, beta])
    out = capsys.readouterr().out
    assert 'non-positive-definite' not in out


def test_check_runs_once_not_per_setF(capsys):
    """Reviewer finding (2026-07-09): the check sits on the _setContacts
    path, which setF() re-invokes every SCF cycle for contactFromFock
    contacts -- without a once-guard it re-warned every cycle. The
    overlap blocks derive from S, which never mutates post-init, so one
    check per object is complete."""
    alpha = np.array([[0.0, 0.05], [0.05, 6.0]])
    beta = np.array([[-1.0, 0.6], [-0.35, -0.3]])
    n = len(alpha)
    # contactFromFock=True path with a non-PD Bloch overlap
    salpha = np.eye(2)
    sbeta = np.diag([0.55, 0.05])
    Fdev = np.block([[alpha, beta], [beta.conj().T, alpha]])
    Sdev = np.block([[salpha, sbeta], [sbeta.T, salpha]])
    g = surfG(Fdev, Sdev, [np.arange(n), np.arange(n, 2 * n)])
    n_construct = capsys.readouterr().out.count('non-positive-definite')
    assert n_construct >= 1, 'check must fire at construction'
    for _ in range(3):
        g.setF(Fdev, 0.0, 0.0)
    n_cycles = capsys.readouterr().out.count('non-positive-definite')
    assert n_cycles == 0, f'check re-fired {n_cycles} times on setF'


def test_complex_blocks_scan_full_bz(capsys):
    """Reviewer finding (2026-07-09): eig(S(-k)) == eig(S(k)) only holds
    for REAL overlap blocks; a complex-block lead whose non-PD region
    lives entirely in (-pi, 0) must still be caught."""
    # complex Sb = c e^{i ph} D (D real diagonal): S(k) is diagonal with
    # entries 1 + 2|c| d_i cos(k + ph), dipping at k* = pi - ph (mod 2pi).
    # ph = 4.0 puts k* = -0.86 -- non-PD only in (-pi, 0), PD on [0, pi],
    # which is exactly the region the old [0, pi]-only scan missed.
    Sa = np.eye(2).astype(complex)
    Sb = 0.55 * np.exp(1j * 4.0) * np.diag([1.0, 0.1])
    def minEig(lo, hi):
        return min(np.linalg.eigvalsh(Sa + Sb*np.exp(1j*k)
                   + Sb.conj().T*np.exp(-1j*k)).min()
                   for k in np.linspace(lo, hi, 361))
    assert minEig(-np.pi, 0) < 0, 'fixture must break PD in (-pi, 0)'
    assert minEig(0, np.pi) > 0, 'fixture must stay PD on [0, pi]'
    alpha = np.array([[0.0, 0.05], [0.05, 6.0]]).astype(complex)
    beta = np.array([[-1.0, 0.6], [-0.35, -0.3]]).astype(complex)
    _build(alpha, beta, Sa, Sb)
    out = capsys.readouterr().out
    # _build mirrors the blocks: contact 0 gets Sb.conj().T, whose dip
    # mirrors into [0, pi] and is caught even by the old half-BZ scan --
    # so asserting on contact 1 (dip ONLY in (-pi, 0)) is what actually
    # pins this regression (re-review mutation test: old scan warns once,
    # fixed scan warns twice).
    assert 'contact 1 Bloch' in out, \
        'complex-block non-PD region in (-pi,0) was missed'
    assert out.count('non-positive-definite') == 2
