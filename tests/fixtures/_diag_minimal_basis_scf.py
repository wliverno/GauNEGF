"""Full-SCF convergence check for the minimal-basis systems with the new Damle
lower contour: C2 (STO-3G, 1D contact) and Au3 (CRENBS, Bethe contact, SOC).

Per system, runs negf.SCF() to convergence (or maxcycles) and reports:
  - SCF converged?  (negf.convLevel < SCF_CONVERGENCE_TOL)
  - final convLevel, trace(S@P), expected electron count, Emin, fermi
  - whether a lower-contour weight warning fired

C2 results print (unbuffered) before Au3 starts, so the fast PSD answer is
visible early even while the heavier Au3-SOC run continues.
"""
import os
import sys
import shutil
import tempfile
import traceback

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)
from gauNEGF.scfE import NEGFE
from gauNEGF.config import SCF_CONVERGENCE_TOL

MAXCYC = 60


def report(negf, label):
    S = np.asarray(negf.S)
    P = np.asarray(negf.P)
    tr = float(np.real(np.trace(S @ P)))
    ne = negf.bar.ne
    conv = float(negf.convLevel)
    converged = conv < SCF_CONVERGENCE_TOL
    print(f'  [{label}] SCF CONVERGED: {converged}  (convLevel={conv:.3e}, '
          f'tol={SCF_CONVERGENCE_TOL:.0e}, maxcycles={MAXCYC})')
    print(f'  [{label}] trace(S@P)={tr:.4f}, expected ne={ne}, '
          f'Emin={negf.Emin:.2f} eV, fermi={negf.fermi:.3f} eV', flush=True)


def run_c2():
    print('--- C2 (STO-3G, 1D contact) ---', flush=True)
    scratch = tempfile.mkdtemp(prefix='c2_scf_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    negf.SCF(maxcycles=MAXCYC, checkpoint=False, pulay=True)
    report(negf, 'C2-STO3G')


def run_au3():
    print('--- Au3 (CRENBS, Bethe contact, SOC) ---', flush=True)
    os.chdir(REPO)
    negf = NEGFE(fn='Au3CRENBS', func='b3lyp', basis='chkbasis',
                 route='integral=grid=superfine', spin='g')
    negf.setContactBethe([[1, 2, 3], [1, 2, 3]], 'AuSOC', T=0)
    negf.setVoltage(0.0)
    negf.SCF(maxcycles=MAXCYC, checkpoint=False, pulay=True)
    report(negf, 'Au3-CRENBS')


def main():
    print('=== minimal-basis SCF convergence ===', flush=True)
    try:
        run_c2()
    except Exception:
        print('C2 FAILED:')
        traceback.print_exc()
    print('=== C2 done ===', flush=True)
    try:
        run_au3()
    except Exception:
        print('Au3 FAILED:')
        traceback.print_exc()
    print('=== minimal-basis SCF convergence complete ===', flush=True)


if __name__ == '__main__':
    main()
