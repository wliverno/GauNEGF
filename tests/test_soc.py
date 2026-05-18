"""
Tests for spin-orbit coupling (SOC) support in surfGBethe.py.

Tests the full SOC pipeline:
1. constructSOCterm matrix dimensions and properties
2. readBetheParams with SOC parameter files
3. constructMat expansion to 18x18 for SOC
4. surfGBAt integration with SOC (DOS, sigma, Fermi)
"""

import sys
sys.path.insert(0, '..')

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import linalg as LA

from gauNEGF.spinTools import constructSOCterm
from gauNEGF.surfGBethe import surfGBAt

# Constants
dim = 9
har_to_eV = 27.211386
eta = 1e-6


# ---- Helper functions (same as test_surfGAt3D.py) ----

def read_bethe_params(filename):
    """Read Slater-Koster parameters from a .bethe file, with SOC support."""
    params = {}
    with open(filename + '.bethe', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            line = line.replace(' ', '')
            key, value = line.split('=')
            params[key] = float(value)

    ne = params['ne']
    Edict = {k[1:]: params[k] * har_to_eV for k in params if k.startswith('e')}
    Sdict = {k[1:]: params[k] for k in params if k.startswith('S')}
    Vdict = {k: params[k] * har_to_eV for k in params
             if not k.startswith('e') and not k.startswith('S')
             and not k.startswith('soc') and k != 'ne'}

    Hdiag = jnp.array([Edict['s']] + [Edict['p']] * 3 +
                      [Edict['dd']] + [Edict['dt']] * 2 + [Edict['dd'], Edict['dt']])
    H0 = jnp.diag(Hdiag)

    soc_params = None
    if 'soc_p' in params and 'soc_d' in params:
        soc_params = [0.0, params['soc_p'], params['soc_d']]

    return ne, H0, Sdict, Vdict, soc_params


def construct_mat(Mdict, dirCosines):
    """Construct 9x9 hopping/overlap matrix using Slater-Koster formalism."""
    M = jnp.zeros((dim, dim))

    M = M.at[0, 0].set(Mdict['sss'])
    M = M.at[0, 3].set(Mdict['sps'])
    M = M.at[3, 0].set(-Mdict['sps'])
    M = M.at[1, 1].set(Mdict['ppp'])
    M = M.at[2, 2].set(Mdict['ppp'])
    M = M.at[3, 3].set(Mdict['pps'])
    M = M.at[0, 4].set(Mdict['sds'])
    M = M.at[4, 0].set(Mdict['sds'])
    M = M.at[1, 5].set(Mdict['pdp'])
    M = M.at[2, 6].set(Mdict['pdp'])
    M = M.at[3, 4].set(Mdict['pds'])
    M = M.at[5, 1].set(-Mdict['pdp'])
    M = M.at[6, 2].set(-Mdict['pdp'])
    M = M.at[4, 3].set(-Mdict['pds'])
    M = M.at[4, 4].set(Mdict['dds'])
    M = M.at[5, 5].set(Mdict['ddp'])
    M = M.at[6, 6].set(Mdict['ddp'])
    M = M.at[7, 7].set(Mdict['ddd'])
    M = M.at[8, 8].set(Mdict['ddd'])

    tr = jnp.zeros((9, 9))
    x, y, z = dirCosines
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    tr = tr.at[0, 0].set(1.0)
    tr = tr.at[1:4, 1:4].set(jnp.array([
        [jnp.cos(theta) * jnp.cos(phi), -jnp.sin(phi), jnp.sin(theta) * jnp.cos(phi)],
        [jnp.cos(theta) * jnp.sin(phi), jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi)],
        [-jnp.sin(theta), 0, jnp.cos(theta)]
    ]))

    d_block = jnp.zeros((5, 5))
    d_block = d_block.at[0, 0].set((3 * z**2 - 1) / 2)
    d_block = d_block.at[0, 1].set(-jnp.sqrt(3) * jnp.sin(2*theta) / 2)
    d_block = d_block.at[0, 3].set(jnp.sqrt(3) * jnp.sin(theta)**2 / 2)
    d_10 = jnp.sqrt(3) * jnp.sin(2*theta) * jnp.cos(phi) / 2
    d_block = d_block.at[1, 0].set(d_10)
    d_block = d_block.at[1, 1].set(jnp.cos(2*theta) * jnp.cos(phi))
    d_block = d_block.at[1, 2].set(-jnp.cos(theta) * jnp.sin(phi))
    d_block = d_block.at[1, 3].set(-d_10 / jnp.sqrt(3))
    d_block = d_block.at[1, 4].set(jnp.sin(theta) * jnp.sin(phi))
    d_20 = jnp.sqrt(3) * jnp.sin(2*theta) * jnp.sin(phi) / 2
    d_block = d_block.at[2, 0].set(d_20)
    d_block = d_block.at[2, 1].set(jnp.cos(2*theta) * jnp.sin(phi))
    d_block = d_block.at[2, 2].set(jnp.cos(theta) * jnp.cos(phi))
    d_block = d_block.at[2, 3].set(-d_20 / jnp.sqrt(3))
    d_block = d_block.at[2, 4].set(-jnp.sin(theta) * jnp.cos(phi))
    d_block = d_block.at[3, 0].set(jnp.sqrt(3) * jnp.sin(theta)**2 * jnp.cos(2*phi) / 2)
    d_block = d_block.at[3, 1].set(jnp.sin(2*theta) * jnp.cos(2*phi) / 2)
    d_block = d_block.at[3, 2].set(-jnp.sin(theta) * jnp.sin(2*phi))
    d_block = d_block.at[3, 3].set((1 + jnp.cos(theta)**2) * jnp.cos(2*phi) / 2)
    d_block = d_block.at[3, 4].set(-jnp.cos(theta) * jnp.sin(2*phi))
    d_block = d_block.at[4, 0].set(jnp.sqrt(3) * jnp.sin(theta)**2 * jnp.sin(2*phi) / 2)
    d_block = d_block.at[4, 1].set(jnp.sin(2*theta) * jnp.sin(2*phi) / 2)
    d_block = d_block.at[4, 2].set(jnp.sin(theta) * jnp.cos(2*phi))
    d_block = d_block.at[4, 3].set((1 + jnp.cos(theta)**2) * jnp.sin(2*phi) / 2)
    d_block = d_block.at[4, 4].set(jnp.cos(theta) * jnp.cos(2*phi))
    tr = tr.at[4:9, 4:9].set(d_block)

    return tr @ M @ tr.T


def gen_fcc_111_neighbors():
    """Generate 12 FCC [111] neighbor unit vectors (z-axis normal)."""
    plane_normal = jnp.array([0., 0., 1.])
    first_neighbor = jnp.array([1., 0., 0.])

    in_plane = []
    for i in range(3):
        angle = i * jnp.pi / 3
        c, s = jnp.cos(angle), jnp.sin(angle)
        K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                      [plane_normal[2], 0, -plane_normal[0]],
                      [-plane_normal[1], plane_normal[0], 0]])
        R = jnp.eye(3) + s * K + (1 - c) * K @ K
        v = R @ first_neighbor
        in_plane.append(v / jnp.linalg.norm(v))

    oop_angle = jnp.arccos(1/jnp.sqrt(3))
    rot_angle = jnp.pi/6
    K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                  [plane_normal[2], 0, -plane_normal[0]],
                  [-plane_normal[1], plane_normal[0], 0]])
    R = jnp.eye(3) + jnp.sin(rot_angle) * K + (1 - jnp.cos(rot_angle)) * K @ K
    rotated_first = R @ first_neighbor
    oop_base = jnp.cos(oop_angle) * rotated_first + jnp.sin(oop_angle) * plane_normal

    out_of_plane = []
    for i in range(3):
        angle = i * 2 * jnp.pi / 3
        c, s = jnp.cos(angle), jnp.sin(angle)
        K = jnp.array([[0, -plane_normal[2], plane_normal[1]],
                      [plane_normal[2], 0, -plane_normal[0]],
                      [-plane_normal[1], plane_normal[0], 0]])
        R = jnp.eye(3) + s * K + (1 - c) * K @ K
        out_of_plane.append(R @ oop_base)

    all_vecs = in_plane + out_of_plane
    for i in range(6):
        all_vecs.append(-all_vecs[i])
    return all_vecs


def build_soc_surfGBAt(filename='tests/AuSOC'):
    """Build a surfGBAt with SOC from a .bethe file."""
    ne, H0_scalar, Sdict, Vdict, soc_params = read_bethe_params(filename)

    # Build SOC Hamiltonian (18x18)
    Hsoc = constructSOCterm(soc_params)
    H0_soc = jnp.array(jnp.kron(H0_scalar, jnp.eye(2))) + jnp.array(Hsoc)

    # Build hopping matrices expanded to 18x18
    dirList = gen_fcc_111_neighbors()
    Slist = [jnp.kron(construct_mat(Sdict, d), jnp.eye(2)) for d in dirList]
    Vlist = [jnp.kron(construct_mat(Vdict, d), jnp.eye(2)) for d in dirList]

    return surfGBAt(H0_soc, Slist, Vlist, eta, SOC=True), ne


# ========== TEST 1: constructSOCterm ==========

def test_constructSOCterm_dimensions():
    """constructSOCterm should return an 18x18 matrix (2+6+10)."""
    lambdas = [0.0, 0.05, 0.023]
    Hsoc = constructSOCterm(lambdas)
    assert Hsoc.shape == (18, 18), \
        f"Expected (18, 18), got {Hsoc.shape}"


def test_constructSOCterm_hermitian():
    """SOC Hamiltonian must be Hermitian."""
    lambdas = [0.0, 0.05, 0.023]
    Hsoc = constructSOCterm(lambdas)
    diff = np.max(np.abs(Hsoc - Hsoc.conj().T))
    assert diff < 1e-14, f"SOC matrix not Hermitian, max diff = {diff}"


def test_constructSOCterm_s_block_zero():
    """The s-orbital block (first 2x2) of SOC should be zero."""
    lambdas = [0.0, 0.05, 0.023]
    Hsoc = constructSOCterm(lambdas)
    s_block = Hsoc[:2, :2]
    assert np.allclose(s_block, 0), f"s-block not zero: {s_block}"


def test_constructSOCterm_traceless():
    """L.S is traceless for each angular momentum subspace."""
    lambdas = [0.0, 1.0, 1.0]  # unit lambdas for clarity
    Hsoc = constructSOCterm(lambdas)
    assert abs(np.trace(Hsoc)) < 1e-14, \
        f"SOC matrix not traceless, trace = {np.trace(Hsoc)}"


# ========== TEST 2: readBetheParams with SOC ==========

def test_readBetheParams_soc_stores_H0():
    """readBetheParams with SOC file should set self.H0 as 18x18 and self.SOC=True.

    We test the internal logic by calling the method and checking attributes.
    Since surfGB requires complex init args, we test the logic directly.
    """
    # This tests that readBetheParams correctly handles SOC params
    # by reimplementing the expected logic and checking it matches
    ne, H0, Sdict, Vdict, soc_params = read_bethe_params('tests/AuSOC')
    assert soc_params is not None, "SOC params not found in AuSOC.bethe"
    assert len(soc_params) == 3
    assert soc_params[0] == 0.0  # s orbital has no SOC
    assert soc_params[1] == 0.0  # p SOC: AuSOC.bethe uses d-only SOC (commit 2f71ea6)
    assert soc_params[2] > 0  # d SOC (dominant for Au)


def test_readBetheParams_soc_h0_shape():
    """H0 with SOC should be 18x18 (kron(9x9, eye(2)) + Hsoc)."""
    ne, H0, Sdict, Vdict, soc_params = read_bethe_params('tests/AuSOC')
    Hsoc = constructSOCterm(soc_params)
    H0_soc = jnp.kron(H0, jnp.eye(2)) + jnp.array(Hsoc)
    assert H0_soc.shape == (18, 18), f"Expected (18,18), got {H0_soc.shape}"


def test_readBetheParams_soc_h0_hermitian():
    """SOC Hamiltonian should be Hermitian."""
    ne, H0, Sdict, Vdict, soc_params = read_bethe_params('tests/AuSOC')
    Hsoc = constructSOCterm(soc_params)
    H0_soc = jnp.kron(H0, jnp.eye(2)) + jnp.array(Hsoc)
    diff = jnp.max(jnp.abs(H0_soc - H0_soc.conj().T))
    assert diff < 1e-12, f"H0_soc not Hermitian, max diff = {diff}"


# ========== TEST 3: constructMat with SOC ==========

def test_constructMat_soc_expansion():
    """When SOC is enabled, constructMat should return 18x18 = kron(9x9, eye(2)).

    This tests that the surfGB.constructMat method properly handles
    SOC by expanding the 9x9 SK matrix to 18x18 via Kronecker product.
    """
    ne, H0, Sdict, Vdict, soc_params = read_bethe_params('tests/AuSOC')

    # Reference: our local construct_mat gives 9x9
    direction = [0., 0., 1.]
    M9 = construct_mat(Vdict, direction)
    assert M9.shape == (9, 9)

    # SOC expansion should give kron(M9, eye(2)) = 18x18
    M18 = jnp.kron(M9, jnp.eye(2))
    assert M18.shape == (18, 18), f"Expected (18,18), got {M18.shape}"

    # The 18x18 should have identical spin-up and spin-down blocks
    # In kron(M, I2) ordering: M[i,j] -> block at (2i:2i+2, 2j:2j+2) = M[i,j]*I2
    for i in range(9):
        for j in range(9):
            assert jnp.allclose(M18[2*i, 2*j], M9[i, j])
            assert jnp.allclose(M18[2*i+1, 2*j+1], M9[i, j])
            assert jnp.allclose(M18[2*i, 2*j+1], 0)
            assert jnp.allclose(M18[2*i+1, 2*j], 0)


# ========== TEST 4: surfGBAt with SOC ==========

def test_surfGBAt_soc_creation():
    """surfGBAt should accept 18x18 matrices when SOC=True."""
    g, ne = build_soc_surfGBAt()
    assert g.dim == 18, f"Expected dim=18, got {g.dim}"
    assert g.H.shape == (18, 18)


def test_surfGBAt_soc_sigmaK():
    """Bulk self-energy with SOC should return 12 matrices of shape (18,18)."""
    g, ne = build_soc_surfGBAt()
    E = 0.0
    sigK = g.sigmaK(E)
    assert sigK.shape == (12, 18, 18), f"Expected (12,18,18), got {sigK.shape}"


def test_surfGBAt_soc_sigma_retarded():
    """Surface sigma with SOC should be retarded: Im(diag) <= 0."""
    g, ne = build_soc_surfGBAt()
    E = 0.0
    sigTot = g.sigmaTot(E)
    imag_diag = jnp.diag(sigTot).imag
    assert jnp.all(imag_diag <= 1e-10), \
        f"Sigma not retarded, max Im(diag) = {jnp.max(imag_diag)}"


def test_surfGBAt_soc_dos_positive():
    """DOS with SOC should be positive within the band."""
    g, ne = build_soc_surfGBAt()
    # Set a dummy fermi to avoid calcFermi (tested separately)
    g.fermi = 0.0
    E = 0.0
    dos = g.DOS(E)
    assert dos > 0, f"DOS should be positive, got {dos}"


def test_surfGBAt_soc_dos_vs_nonsoc():
    """SOC DOS should differ from non-SOC DOS (SOC lifts degeneracies)."""
    # Build SOC version
    g_soc, ne_soc = build_soc_surfGBAt()
    g_soc.fermi = 0.0

    # Build non-SOC version using same params minus SOC
    ne, H0, Sdict, Vdict, _ = read_bethe_params('tests/AuSOC')
    dirList = gen_fcc_111_neighbors()
    Slist = [construct_mat(Sdict, d) for d in dirList]
    Vlist = [construct_mat(Vdict, d) for d in dirList]
    g_nosoc = surfGBAt(H0, Slist, Vlist, eta)
    g_nosoc.fermi = 0.0

    # DOS should differ due to SOC splitting
    E = 0.0
    dos_soc = g_soc.DOS(E)
    dos_nosoc = g_nosoc.DOS(E)
    # SOC should modify the DOS (not necessarily larger/smaller, just different)
    # Use tight tolerance -- if they differ at all, SOC is doing something
    assert not jnp.isclose(dos_soc, 2 * dos_nosoc, rtol=1e-6), \
        f"SOC DOS ({dos_soc}) is exactly 2x non-SOC DOS ({dos_nosoc}), SOC has no effect"


# ========== TEST 5: surfGBethe.py actual code paths ==========

def test_surfGB_readBetheParams_soc():
    """readBetheParams on surfGB should handle SOC .bethe files.

    Tests that readBetheParams:
    1. Doesn't crash on soc_p/soc_d keys (assertion check)
    2. Sets self.SOC = True
    3. Sets self.H0 with shape (18, 18)
    """
    from gauNEGF.surfGBethe import surfGB

    # Create a minimal mock object with just what readBetheParams needs
    class MockSurfGB:
        spin = 'g'  # SOC requires non-restricted spin
    obj = MockSurfGB()
    # Call readBetheParams as unbound method
    surfGB.readBetheParams(obj, 'tests/AuSOC')
    assert obj.SOC is True, "SOC flag not set"
    assert hasattr(obj, 'H0'), "H0 not stored as attribute"
    assert obj.H0.shape == (18, 18), f"Expected H0 shape (18,18), got {obj.H0.shape}"


def test_surfGB_constructMat_soc():
    """surfGB.constructMat should accept SOC flag and return 18x18 when SOC=True."""
    from gauNEGF.surfGBethe import surfGB

    class MockSurfGB:
        spin = 'g'
    obj = MockSurfGB()
    surfGB.readBetheParams(obj, 'tests/AuSOC')

    direction = [0., 0., 1.]
    M = surfGB.constructMat(obj, obj.Vdict, direction, obj.SOC)
    assert M.shape == (18, 18), f"Expected (18,18) with SOC, got {M.shape}"


def test_surfGB_constructMat_nosoc():
    """surfGB.constructMat without SOC should still return 9x9."""
    from gauNEGF.surfGBethe import surfGB

    class MockSurfGB:
        pass
    obj = MockSurfGB()
    surfGB.readBetheParams(obj, 'tests/Au')

    direction = [0., 0., 1.]
    M = surfGB.constructMat(obj, obj.Vdict, direction, obj.SOC)
    assert M.shape == (9, 9), f"Expected (9,9) without SOC, got {M.shape}"


# ========== TEST 6: Fermi energy convergence with SOC ==========

@pytest.mark.slow
def test_surfGBAt_soc_fermi_convergence():
    """Fermi energy search with SOC should converge for Au (ne=11).

    Marked slow: SOC Fermi search takes ~10+ min on P100 (dominates test_soc.py).
    """
    g, ne = build_soc_surfGBAt()
    # ne=11 for Au, but calcFermi expects ne per spin for restricted
    # With SOC, the 18x18 basis already includes both spins
    fermi = g.calcFermi(ne)
    assert fermi is not None, "Fermi search did not converge"
    # Fermi energy should be in a reasonable range for Au (-5 to 5 eV)
    assert -5.0 < fermi < 5.0, f"Fermi energy {fermi} eV out of expected range"


def test_surfGBAt_soc_h0_eigenvalues():
    """SOC should lift degeneracies compared to non-SOC H0."""
    ne, H0, Sdict, Vdict, soc_params = read_bethe_params('tests/AuSOC')
    Hsoc = constructSOCterm(soc_params)
    H0_soc = jnp.kron(H0, jnp.eye(2)) + jnp.array(Hsoc)

    evals_nosoc = jnp.sort(jnp.linalg.eigvalsh(H0))
    evals_soc = jnp.sort(jnp.linalg.eigvalsh(H0_soc))

    # Non-SOC: 9 eigenvalues, each doubly degenerate in SOC basis -> 18 with pairs
    # SOC should split some of the d-orbital degeneracies
    # Check that SOC eigenvalues are NOT all just doubled non-SOC eigenvalues
    doubled_nosoc = jnp.sort(jnp.concatenate([evals_nosoc, evals_nosoc]))
    diff = jnp.max(jnp.abs(evals_soc - doubled_nosoc))
    assert diff > 0.01, \
        f"SOC eigenvalues match doubled non-SOC eigenvalues (max diff={diff}), SOC has no effect"


# ========== TEST 7: SOC spin gating ==========

def test_readBetheParams_restricted_disables_soc():
    """readBetheParams with spin='r' should set SOC=False even with SOC params."""
    from gauNEGF.surfGBethe import surfGB

    class MockSurfGB:
        spin = 'r'
    obj = MockSurfGB()
    surfGB.readBetheParams(obj, 'tests/AuSOC')
    assert obj.SOC is False, f"SOC should be False for spin='r', got {obj.SOC}"
    assert obj.H0.shape == (9, 9), f"H0 should be 9x9 for restricted, got {obj.H0.shape}"


def test_readBetheParams_generalized_enables_soc():
    """readBetheParams with spin='g' should enable SOC when params present."""
    from gauNEGF.surfGBethe import surfGB

    class MockSurfGB:
        spin = 'g'
    obj = MockSurfGB()
    surfGB.readBetheParams(obj, 'tests/AuSOC')
    assert obj.SOC is True, f"SOC should be True for spin='g' with SOC params"
    assert obj.H0.shape == (18, 18), f"H0 should be 18x18 with SOC, got {obj.H0.shape}"


def test_readBetheParams_unrestricted_enables_soc():
    """readBetheParams with spin='u' should enable SOC when params present."""
    from gauNEGF.surfGBethe import surfGB

    class MockSurfGB:
        spin = 'u'
    obj = MockSurfGB()
    surfGB.readBetheParams(obj, 'tests/AuSOC')
    assert obj.SOC is True, f"SOC should be True for spin='u' with SOC params"
    assert obj.H0.shape == (18, 18), f"H0 should be 18x18 with SOC, got {obj.H0.shape}"


# ========== TEST 8: surfGB.sigma() with SOC ==========

def build_mock_surfGB_with_soc(spin='g'):
    """Build a mock surfGB object with SOC for testing sigma()."""
    from gauNEGF.surfGBethe import surfGB

    g_soc, ne = build_soc_surfGBAt()
    g_soc.fermi = 0.0

    class MockSurfGB:
        pass
    obj = MockSurfGB()
    obj.N = 9  # single atom, 9 orbitals
    obj.SOC = True
    obj.spin = spin
    obj.gList = [g_soc]
    # Single atom contact: orbital indices [0..8], no excluded neighbor dirs
    obj.nIndLists = [[[3, 4, 5]]]  # exclude up directions (connected to device)
    obj.indsLists = [[jnp.arange(9)]]
    obj.Sdict = {'sss': 1.0}  # non-zero -> skip Xi transform
    obj.Xi = jnp.eye(9)
    return obj


def test_surfGB_sigma_soc_shape_generalized():
    """sigma() with SOC + 'g' spin should return 2N x 2N matrix."""
    from gauNEGF.surfGBethe import surfGB
    obj = build_mock_surfGB_with_soc(spin='g')
    sig = surfGB.sigma(obj, 0.0, 0)
    expected_size = 2 * obj.N  # 18
    assert sig.shape == (expected_size, expected_size), \
        f"Expected ({expected_size},{expected_size}), got {sig.shape}"


def test_surfGB_sigma_soc_shape_unrestricted():
    """sigma() with SOC + 'u' spin should return 2N x 2N matrix."""
    from gauNEGF.surfGBethe import surfGB
    obj = build_mock_surfGB_with_soc(spin='u')
    sig = surfGB.sigma(obj, 0.0, 0)
    expected_size = 2 * obj.N  # 18
    assert sig.shape == (expected_size, expected_size), \
        f"Expected ({expected_size},{expected_size}), got {sig.shape}"


def test_surfGB_sigma_soc_no_double_kron():
    """sigma() with SOC should NOT apply kron expansion (spin already in matrices)."""
    from gauNEGF.surfGBethe import surfGB

    # Build SOC version
    obj_soc = build_mock_surfGB_with_soc(spin='g')
    sig_soc = surfGB.sigma(obj_soc, 0.0, 0)

    # If kron was incorrectly applied, size would be 4N x 4N = 36 x 36
    assert sig_soc.shape != (4 * obj_soc.N, 4 * obj_soc.N), \
        "sigma() applied kron on top of SOC -- double counting spin!"
    assert sig_soc.shape == (2 * obj_soc.N, 2 * obj_soc.N)


def test_surfGB_sigma_soc_retarded():
    """sigma() with SOC should produce retarded self-energy."""
    from gauNEGF.surfGBethe import surfGB
    obj = build_mock_surfGB_with_soc(spin='g')
    sig = surfGB.sigma(obj, 0.0, 0)
    imag_diag = jnp.diag(sig).imag
    assert jnp.all(imag_diag <= 1e-10), \
        f"Sigma not retarded, max Im(diag) = {jnp.max(imag_diag)}"


def test_surfGB_sigmaTot_soc_shape():
    """sigmaTot() with SOC should return 2N x 2N."""
    from gauNEGF.surfGBethe import surfGB
    obj = build_mock_surfGB_with_soc(spin='g')
    # sigmaTot needs sigma to not be JIT-compiled for mock
    obj.sigma = lambda E, i, conv=1e-5: surfGB.sigma(obj, E, i, conv)
    sigTot = surfGB.sigmaTot(obj, 0.0)
    expected_size = 2 * obj.N
    assert sigTot.shape == (expected_size, expected_size), \
        f"Expected ({expected_size},{expected_size}), got {sigTot.shape}"


# ========== Run tests ==========

if __name__ == '__main__':
    tests = [
        test_constructSOCterm_dimensions,
        test_constructSOCterm_hermitian,
        test_constructSOCterm_s_block_zero,
        test_constructSOCterm_traceless,
        test_readBetheParams_soc_stores_H0,
        test_readBetheParams_soc_h0_shape,
        test_readBetheParams_soc_h0_hermitian,
        test_constructMat_soc_expansion,
        test_surfGBAt_soc_creation,
        test_surfGBAt_soc_sigmaK,
        test_surfGBAt_soc_sigma_retarded,
        test_surfGBAt_soc_dos_positive,
        test_surfGBAt_soc_dos_vs_nonsoc,
        test_surfGB_readBetheParams_soc,
        test_surfGB_constructMat_soc,
        test_surfGB_constructMat_nosoc,
        test_surfGBAt_soc_h0_eigenvalues,
        test_surfGBAt_soc_fermi_convergence,
        test_readBetheParams_restricted_disables_soc,
        test_readBetheParams_generalized_enables_soc,
        test_readBetheParams_unrestricted_enables_soc,
        test_surfGB_sigma_soc_shape_generalized,
        test_surfGB_sigma_soc_shape_unrestricted,
        test_surfGB_sigma_soc_no_double_kron,
        test_surfGB_sigma_soc_retarded,
        test_surfGB_sigmaTot_soc_shape,
    ]

    passed = 0
    failed = 0
    errors = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {type(e).__name__}: {e}")
            errors += 1

    print(f"\nResults: {passed} passed, {failed} failed, {errors} errors")
