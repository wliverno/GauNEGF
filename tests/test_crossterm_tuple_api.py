import numpy as np, os, pytest
from test_cross_term import (make_1d_nonortho_chain, make_1d_ortho_chain,
                             make_surfGBAt, make_surfGAt3D, make_surfGTest)
E = 0.37 + 0.0j
BASE = np.load(os.path.join(os.path.dirname(__file__),
                            'baselines', 'qsym_baseline.npz'))

def _tuple_case(g, key, i=0, use_surf=False):
    # Read per-class energy from baseline (key_E holds the energy used)
    E_probe = complex(BASE[key + '_E'][0])
    if use_surf:
        out = g.crossTermQSurf(E_probe)
    else:
        out = g.crossTermQ(E_probe, i)
    assert isinstance(out, tuple) and len(out) == 3
    qf, qr, qs = out
    # Q_sym is the average of the halves
    assert np.allclose(np.array(qs), (np.array(qf) + np.array(qr)) / 2,
                       rtol=0, atol=1e-14)
    # bit-identity on CPU (baseline provenance); GPU backends cannot
    # reproduce CPU floats bitwise, so tight allclose there (B6)
    import jax
    if jax.default_backend() == 'cpu':
        assert np.array_equal(np.array(qs), BASE[key])
    else:
        assert np.allclose(np.array(qs), BASE[key], rtol=1e-13, atol=0)

def test_surfG1D_tuple_and_baseline():
    _tuple_case(make_1d_nonortho_chain(), 'surfG1D_c0')

def test_surfGBAt_tuple_and_baseline():
    _tuple_case(make_surfGBAt(), 'surfGBAt_c0')

def test_surfGAt3D_crossTermQSurf_tuple_and_baseline():
    # surfGAt3D.crossTermQSurf (surface path, 9 directions) is FINITE at -5.0+0.1j.
    # Baseline captures this energy. Expects tuple return once Task 2 implements API.
    _tuple_case(make_surfGAt3D(), 'surfGAt3D_surf', use_surf=True)

def test_surfGAt3D_bulk_nan_known_defect():
    # Documents pre-existing defect: crossTermQ delegates to crossTermQBulk
    # (12-direction bulk iteration), which produces NaN systematically on this toy.
    # surfG3D is red-status; this test ensures we track if the bulk path ever gets fixed.
    # If this test fails, the defect was fixed and baseline should be extended.
    g = make_surfGAt3D()
    Q_bulk = np.array(g.crossTermQ(-5.0 + 0.1j, 0)[2])
    assert not np.all(np.isfinite(Q_bulk)), \
        "crossTermQBulk NaN defect was fixed; update baseline to include bulk entry"

def test_orthogonal_still_none_surfG1D():
    assert make_1d_ortho_chain().crossTermQ(E, 0) is None

def test_orthogonal_still_none_surfGTest():
    # separate test: make_surfGTest importorskips gauopen and must not
    # mask the surfG1D assertion when gauopen is absent
    assert make_surfGTest().crossTermQ(E, 0) is None

def test_crossTermQTot_still_matrix():
    g = make_1d_nonortho_chain()
    tot = g.crossTermQTot(E)
    assert not isinstance(tot, tuple)
    q0 = g.crossTermQ(E, 0); q1 = g.crossTermQ(E, 1)
    assert np.allclose(np.array(tot),
                       np.array(q0[2]) + np.array(q1[2]), atol=1e-14)
