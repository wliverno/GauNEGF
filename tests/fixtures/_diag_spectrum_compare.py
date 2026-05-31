"""Phase-1 evidence for the Au3 SCF crash (systematic-debugging).

Compares the TWO spectra in play, for C2-STO3G (works) and Au3-CRENBS (crashes):

  TRUE spectrum  -- what calcEmin's DOS loop and densityComplex see, via the
                    full energy-dependent Sigma(E). Poles of [E S - F - Sigma(E)].
  DAMLE spectrum -- eig(Fbar), Fbar = Y_eff (F + Sigma_0) Y_eff, what
                    damleLowerDensity integrates with the CONSTANT Sigma_0.

Read-only. No SCF, no FockToP, no fixes. Just instrumentation to show WHERE the
two spectra disagree and WHY Emin=-16.89 was placed where it was for Au3.
"""
import os
import sys
import shutil
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)
from gauNEGF.scfE import NEGFE
from gauNEGF.transport import har_to_eV
from gauNEGF.density import _compute_dos_at_energy
from gauNEGF.config import ETA, FERMI_CALCULATION_TOL


def analyze(negf, label, dos_energies):
    print(f'\n===== {label} =====', flush=True)
    F = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    N = F.shape[0]
    print(f'N basis = {N}, mu = {float(negf.fermi):.3f} eV, '
          f'setup Emin = {float(negf.Emin):.3f} eV, '
          f'damle_buffer = {float(getattr(negf, "damle_buffer", 20.0)):.1f}')

    # TRUE Fock band (bare; what calcEmin's start estimate and the physical
    # states track).
    tv = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(S, F))))
    print(f'true eig(inv(S)F) eV: min={tv[0]:.2f}, '
          f'lowest6={np.array2string(tv[:6], precision=2)}')

    # Asymptotic fit magnitudes.
    Sig0 = np.asarray(negf.Sigma_0)
    X = np.asarray(negf.X_asymp)
    Yeff = np.asarray(negf.Y_eff)
    Seff = np.asarray(negf.S_eff)
    Xn = max(np.linalg.norm(X), 1e-30)
    Yn = max(np.linalg.norm(Yeff), 1e-30)
    print(f'||Sigma_0||={np.linalg.norm(Sig0):.3e}, ||X_asymp||={np.linalg.norm(X):.3e}, '
          f'||Im X||/||X||={np.linalg.norm(np.imag(X))/Xn:.3e}')

    # S_eff conditioning / "PSD" proxy (S_eff is complex non-Hermitian).
    se = np.linalg.eigvals(Seff)
    print(f'eig(S_eff): Re in [{np.min(se.real):.3e}, {np.max(se.real):.3e}], '
          f'min|eig|={np.min(np.abs(se)):.3e}, count Re<0: {int(np.sum(se.real < 0))}/{N}')
    print(f'||Im Y_eff||/||Y_eff|| = {np.linalg.norm(np.imag(Yeff))/Yn:.3e}')

    # DAMLE band: eig of Fbar = Y_eff (F + Sigma_0 + i*ETA) Y_eff.
    Fbar = Yeff @ (F + Sig0 + 1j * ETA * np.eye(N)) @ Yeff
    dv = np.sort(np.real(np.linalg.eigvals(Fbar)))
    buf = float(getattr(negf, 'damle_buffer', 20.0))
    print(f'Re eig(Fbar) eV: min={dv[0]:.2f}, '
          f'lowest6={np.array2string(dv[:6], precision=2)}')
    print(f'Fbar floor - buffer = {dv[0] - buf:.2f} eV   '
          f'(would-be Damle Emin guess)')

    # Disagreement: poles below the setup Emin in each spectrum.
    e0 = float(negf.Emin)
    print(f'poles below setup Emin={e0:.2f} eV:  true={int(np.sum(tv < e0))},  '
          f'Damle(Fbar)={int(np.sum(dv < e0))}   <-- mismatch if these differ')

    # TRUE-Sigma DOS sweep (exactly what calcEmin's loop tests).
    print(f'true-Sigma DOS sweep (calcEmin tol = {FERMI_CALCULATION_TOL:.0e}):')
    for E in dos_energies:
        Q = negf.g.crossTermQTot(E)
        dos = _compute_dos_at_energy(E, F, S, negf.g.sigmaTot(E), Q)
        print(f'    E={E:+8.1f} eV -> DOS={float(dos):.3E}')


def setup_c2():
    sc = tempfile.mkdtemp(prefix='c2_sp_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(sc, 'C2_chain.gjf'))
    os.chdir(sc)
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
    print('=== spectrum comparison: TRUE vs DAMLE ===', flush=True)
    analyze(setup_c2(), 'C2-STO3G (works)',
            [-300, -285, -280, -150, -50, -20])
    analyze(setup_au3(), 'Au3-CRENBS (crashes)',
            [-300, -100, -50, -30, -20, -17, -15, -10, -5])
    print('\n=== spectrum comparison complete ===', flush=True)


if __name__ == '__main__':
    main()
