"""Diagnostic: measure the Damle lower-contour cross-term delta_N vs Emin on
C2-STO3G, to understand when (if ever) the FockToP warning should fire.

delta_N is the device-lead Mulliken CROSS term, not the total charge below Emin.
This script reports tr(P@S) (the bulk lower charge) AND delta_N (the cross term)
at a sweep of Emin values from below-all-poles up into the valence region. Its
finding: |delta_N| alone is a poor warning signal (cores below Emin give
tr(P@S)~2 but delta_N~1e-6), which is why FockToP keys the warning on the TOTAL
count -- it fires when any of |tr(P@S)|, |delta_N|, or their sum exceeds
damle_dN_warn (default 0.5 e).
"""
import os
import sys
import shutil
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from numpy.linalg import norm

REPO = '/mmfs1/gscratch/anantram/willll/NEGFCode'
sys.path.insert(0, REPO)
from gauNEGF.scfE import NEGFE
from gauNEGF.transport import har_to_eV
from gauNEGF.density import damleLowerDensity


def setup():
    scratch = tempfile.mkdtemp(prefix='c2_dN_')
    shutil.copy(os.path.join(REPO, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    os.chdir(scratch)
    negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                 route='integral=grid=superfine')
    negf.setContact1D([[1], [2]], symmetrize_contacts=True)
    negf.setVoltage(0.0)
    return negf


def main():
    print('=== delta_N vs Emin (C2-STO3G) ===')
    negf = setup()
    F_eV = np.asarray(negf.F) * har_to_eV
    S = np.asarray(negf.S)
    poles = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(S, F_eV))))
    print('poles eig(inv(S)F) eV:', np.array2string(poles, precision=2))
    print('fermi (mu) eV:', float(negf.fermi))
    print()
    print('%12s %14s %14s' % ('Emin', 'tr(P@S)', 'delta_N'))
    for Emin in [-291.0, -281.0, -200.0, -145.0, -50.0, -30.0, -20.0, -15.0]:
        P, dN = damleLowerDensity(F_eV, negf.Y_eff, negf.Sigma_0, negf.g, Emin)
        trps = float(np.real(np.trace(np.asarray(P) @ S)))
        print('%12.2f %14.6f %14.6e' % (Emin, trps, dN))
    print('=== diag complete ===')


if __name__ == '__main__':
    main()
