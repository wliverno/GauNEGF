"""Standalone runner for the 3D single-cell Fermi check.

Bypasses pytest so output streams live (with `python -u`). Prints wall-clock
markers around setup and calcFermi, then asserts the same condition as
test_3d_single_cell_fermi.
"""
import sys
import time
sys.path.insert(0, '/gscratch/anantram/willll/NEGFCode')
sys.path.insert(0, '/gscratch/anantram/willll/NEGFCode/tests')

import jax
from gauNEGF.surfG3D import surfGAt3D
from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors

dim = 9
eta = 1e-3  # bumped from 1e-6 to smooth resonances for adaptive integration
AU_BULK_FERMI_EV = 2.84


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


stamp(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

stamp("Reading Au.bethe params + neighbors")
ne, H0, Sdict, Vdict = read_bethe_params('Au')
vecs = gen_fcc_111_neighbors()
Slist = [construct_mat(Sdict, v) for v in vecs]
Vlist = [construct_mat(Vdict, v) for v in vecs]
stamp(f"  ne={ne}, dim={H0.shape}, n_neighbors={len(vecs)}")

stamp("Constructing surfGAt3D (kPoints=3)")
t0 = time.time()
gAt = surfGAt3D(H0, Slist, Vlist, vecs, eta=eta, T=0, kPoints=3)
stamp(f"  ctor done in {time.time()-t0:.2f}s; num_contacts={gAt.num_contacts}")

stamp(f"F.shape={gAt.F.shape}, S.shape={gAt.S.shape}")
assert gAt.F.shape == (dim, dim)
assert gAt.S.shape == (dim, dim)
assert gAt.num_contacts == 1

stamp(f"Calling calcFermi(ne/2={ne/2}) ...")
t0 = time.time()
fermi = gAt.calcFermi(ne / 2)
elapsed = time.time() - t0
stamp(f"calcFermi DONE in {elapsed:.1f}s => fermi={fermi:.4f} eV (benchmark={AU_BULK_FERMI_EV})")

diff = abs(fermi - AU_BULK_FERMI_EV)
stamp(f"|fermi - benchmark| = {diff:.4f} eV (tolerance 1.5)")

if diff < 1.5:
    stamp("PASS")
    sys.exit(0)
else:
    stamp("FAIL: outside tolerance")
    sys.exit(1)
