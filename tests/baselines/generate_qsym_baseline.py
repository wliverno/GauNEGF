"""Run ONCE on pre-change code. Stores Q_sym for fixed toy inputs."""
import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test_cross_term import (make_1d_nonortho_chain, make_surfGBAt,
                             make_surfGAt3D)

out = {}

# surfG1D: use 0.37 (already finite)
g1 = make_1d_nonortho_chain()
E1 = 0.37 + 0.0j
Q1 = np.array(g1.crossTermQ(E1, 0))
assert np.all(np.isfinite(Q1)), "surfG1D Q_sym contains NaN/Inf"
out['surfG1D_c0'] = Q1
out['surfG1D_c0_E'] = np.array([E1])

# surfGBAt: use 0.37 (already finite)
gb = make_surfGBAt()
Eb = 0.37 + 0.0j
Qb = np.array(gb.crossTermQ(Eb, 0))
assert np.all(np.isfinite(Qb)), "surfGBAt Q_sym contains NaN/Inf"
out['surfGBAt_c0'] = Qb
out['surfGBAt_c0_E'] = np.array([Eb])

# surfGAt3D: use crossTermQSurf surface path (bulk path NaNs, known defect)
# NOTE: crossTermQ delegates to crossTermQBulk (12-direction) -> NaNs at every energy.
# crossTermQSurf (9-direction surface) is FINITE at -5.0+0.1j (existing test energy).
g3 = make_surfGAt3D()
E3 = -5.0 + 0.1j
Q3 = np.array(g3.crossTermQSurf(E3))
assert np.all(np.isfinite(Q3)), "surfGAt3D_surf Q_sym contains NaN/Inf"
out['surfGAt3D_surf'] = Q3
out['surfGAt3D_surf_E'] = np.array([E3], dtype=complex)

np.savez(os.path.join(os.path.dirname(__file__), 'qsym_baseline.npz'), **out)
print('baseline written:', {k: v.shape for k, v in out.items() if not k.endswith('_E')})
print('energies used:', {'surfG1D': E1, 'surfGBAt': Eb, 'surfGAt3D_surf': E3})
