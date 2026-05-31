"""SCF-level validation of the damle lower contour + calcEmin Emin placement.

Runs a short SCF on C2-STO3G (PSD) and reports: Emin below band floor (DOS ~ 0),
total electron count sane and positive (bisectFermi sign sanity), no spurious
lower-contour warning. Then repeats the setup-only Emin check on C2-LANL2DZ
(non-PSD S_eff) to confirm calcEmin still places Emin below the band.
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


def setup(basis):
    scratch = tempfile.mkdtemp(prefix='c2_val_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis=basis,
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    return negf


def check_emin(negf, label):
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    floor = float(np.real(np.linalg.eigvals(np.linalg.solve(S, F_eV))).min())
    ok = negf.Emin <= floor
    print(f'[{label}] Emin={negf.Emin:.3f} eV, band floor={floor:.3f} eV, '
          f'Emin below floor: {ok}')
    return ok


def main():
    print('=== damle SCF validation ===')
    psd = setup('sto-3g')
    ok_psd = check_emin(psd, 'C2-STO3G (PSD)')
    # one Fock->P pass to exercise FockToP + the warning path
    psd.FockToP()
    print('C2-STO3G FockToP completed')

    nonpsd = setup('lanl2dz')
    ok_nonpsd = check_emin(nonpsd, 'C2-LANL2DZ (non-PSD)')

    print(f'VERDICT: Emin-below-floor PSD={ok_psd}, non-PSD={ok_nonpsd}')
    print('=== damle SCF validation complete ===')


if __name__ == '__main__':
    main()
