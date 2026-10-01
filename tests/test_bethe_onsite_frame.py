"""Bethe-lattice on-site block (spec docs/superpowers/specs/2026-09-30-bethe-onsite-cubic-frame-design.md).
Defect 1: the e_g/t2g split sits on each contact's cubic axes. Defect 2 (Task 2): SOC in the lattice basis."""
import contextlib
import io
import os
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.linalg as sla

from gauNEGF.surfGBethe import surfGB

REPO = os.path.join(os.path.dirname(__file__), '..')
LATTICE_FILES = {'Au': os.path.join(REPO, 'Au.bethe'),
                 'Au2': os.path.join(REPO, 'Au2.bethe'),
                 'AuSOC': os.path.join(REPO, 'tests', 'AuSOC.bethe')}
DNN = 4.078 / np.sqrt(2)     # Au nearest-neighbour distance, angstrom
ETA = 0.1                    # large eta: fast lattice convergence; invariances hold for any eta
CUBIC_DIRS = [np.array(v, dtype=float) / np.sqrt(2) for v in
              [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
               (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
               (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]]


def rotation(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


R_GENERIC = rotation((0.3, -0.5, 0.8), 0.7)


def build(coords, contacts, latFile, tmp_path_factory, spin='r'):
    """surfGB from a MagicMock bar (recipe of test_bethe_direction_exclusion._construct_surfGB),
    copying the REQUESTED lattice file (that helper always copies Au.bethe)."""
    coords = np.asarray(coords, dtype=float)
    natoms = len(coords)
    N = 9 * natoms
    bar = MagicMock()
    bar.c = (coords / 0.52917721).flatten()
    bar.ibfatm = np.repeat(np.arange(1, natoms + 1), 9)
    bar.ibftyp = np.zeros(N, dtype=int)
    tmp = str(tmp_path_factory.mktemp('onsite'))
    shutil.copy(LATTICE_FILES[latFile], os.path.join(tmp, f'{latFile}.bethe'))
    Nm = N if spin == 'r' else 2 * N
    old = os.getcwd()
    os.chdir(tmp)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = surfGB(np.zeros((Nm, Nm)), np.eye(Nm), contacts, bar,
                       latFile=latFile, spin=spin, eta=ETA)
    finally:
        os.chdir(old)
    return g


def hexagon(center, normal, u, pull=0.002):
    """7-atom fcc(111) patch, center first. Ring atom 0 sits `pull` angstrom closer to the
    center, so the first-neighbour choice (surfGBethe.py:143) is unique and survives rotation."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.asarray(u, float)
    u = u - np.dot(u, n) * n
    u = u / np.linalg.norm(u)
    w = np.cross(n, u)
    c = np.asarray(center, float)
    ring = [c + DNN * (np.cos(k * np.pi / 3) * u + np.sin(k * np.pi / 3) * w) for k in range(6)]
    ring[0] = ring[0] - pull * u
    return [c] + ring


def junction_coords():
    """Two contacts, normals -x (left) and +x (right); right in-plane axis turned by 17 deg,
    so the two cubic frames are not related by a cube operation."""
    left = hexagon((0, 0, 0), (1, 0, 0), (0, 1, 0))
    u_r = np.array([0.0, np.cos(np.radians(17)), np.sin(np.radians(17))])
    right = hexagon((3 * DNN, 0, 0), (1, 0, 0), u_r)
    return left + right, [list(range(1, 8)), list(range(8, 15))]


def gamma_levels(g, c):
    """fcc Gamma-point levels of contact c's lattice: H0_c + sum_k V_k over S = I + sum_k S_k."""
    H = np.asarray(g.gList[c].H0) + sum(np.asarray(V) for V in g.Vlists[c])
    S = np.eye(H.shape[0]) + sum(np.asarray(Sk) for Sk in g.Slists[c])
    return np.sort(sla.eigh(H, S, eigvals_only=True))


def multiplicities(levels, tol):
    groups, run = [], 1
    for a, b in zip(levels[:-1], levels[1:]):
        if b - a < tol:
            run += 1
        else:
            groups.append(run)
            run = 1
    groups.append(run)
    return sorted(groups)


def assert_labels_match(gA, gB, R, c=0):
    """Precondition: patch B's direction labels are patch A's, rotated (review I1)."""
    for dA, dB in zip(gA.dirLists[c], gB.dirLists[c]):
        np.testing.assert_allclose(R @ np.asarray(dA), np.asarray(dB), atol=1e-10, rtol=0)


@pytest.fixture(scope='module')
def au2_pair(tmp_path_factory):
    A = hexagon((0, 0, 0), (1, 0, 0), (0, 1, 0)) + [np.array([-5.0, 0.0, 0.0])]
    B = [R_GENERIC @ p for p in A]
    gA = build(A, [list(range(1, 8))], 'Au2', tmp_path_factory)
    gB = build(B, [list(range(1, 8))], 'Au2', tmp_path_factory)
    return gA, gB


@pytest.fixture(scope='module')
def au2_junction(tmp_path_factory):
    coords, contacts = junction_coords()
    return build(coords, contacts, 'Au2', tmp_path_factory)


def test_T1_each_contact_has_cubic_gamma_levels(au2_junction):
    g = au2_junction
    assert g.Edict['dd'] != g.Edict['dt']                       # really the Au2 lattice
    for c in range(2):
        assert multiplicities(gamma_levels(g, c), 1e-10) == [1, 2, 3, 3]
        d_eigs = np.sort(np.linalg.eigvalsh(np.asarray(g.gList[c].H0)[4:9, 4:9]))
        np.testing.assert_allclose(
            d_eigs, np.sort([g.Edict['dd']] * 2 + [g.Edict['dt']] * 3), atol=1e-12, rtol=0)
    assert not np.allclose(np.asarray(g.gList[0].H0), np.asarray(g.gList[1].H0))
    # Review Focus 2: the fermi copied from contact 0 (surfGBethe.py:206-213) is contact 1's own, to
    # solver resolution (plan review I1: 2 meV = 1/10 of the 19 meV frame effect; tight bisection tol)
    ef0 = g.gList[0].calcFermi(g.ne / 2, tol=1e-6)
    ef1 = g.gList[1].calcFermi(g.ne / 2, tol=1e-6)
    assert abs(ef1 - ef0) < 2e-3, f'dEf = {ef1 - ef0:.3e} eV'


def test_T2_rigid_rotation_invariance(au2_pair):
    gA, gB = au2_pair
    assert gA.Edict['dd'] != gA.Edict['dt']
    assert_labels_match(gA, gB, R_GENERIC)
    np.testing.assert_allclose(gamma_levels(gA, 0), gamma_levels(gB, 0), atol=1e-10, rtol=0)
    for dE in (-3.0, 0.0, 1.0):
        E = float(gA.gList[0].fermi) + dE
        sA = np.asarray(gA.sigma(E, 0, conv=1e-12))
        sB = np.asarray(gB.sigma(E, 0, conv=1e-12))
        for part in (lambda s: (s + s.conj().T) / 2, lambda s: 1j * (s - s.conj().T)):
            eA = np.linalg.eigvalsh(part(sA))
            eB = np.linalg.eigvalsh(part(sB))
            np.testing.assert_allclose(eA, eB, atol=1e-8 * max(np.max(np.abs(eA)), 1e-12), rtol=0)
    # last, so a fermi miss cannot hide the Gamma/Sigma verdicts (plan review I1)
    efA = gA.gList[0].calcFermi(gA.ne / 2, tol=1e-6)
    efB = gB.gList[0].calcFermi(gB.ne / 2, tol=1e-6)
    assert abs(efA - efB) < 2e-3, f'dEf = {efA - efB:.3e} eV'


def test_T3_cubic_axes_reproduce_parameter_file_diagonal(au2_pair):
    g = au2_pair[0]
    ref = np.asarray(g.H0)                                      # readBetheParams' cubic-frame diagonal
    np.testing.assert_allclose(np.asarray(g.constructOnsite(CUBIC_DIRS)), ref, atol=1e-14, rtol=0)
    M = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)   # cube operation: permute + flip axes
    shuffled = [M @ CUBIC_DIRS[k] for k in (5, 11, 0, 7, 3, 9, 1, 10, 4, 8, 2, 6)]
    np.testing.assert_allclose(np.asarray(g.constructOnsite(shuffled)), ref, atol=1e-14, rtol=0)


@pytest.mark.parametrize('latFile,spin', [('Au', 'r'), ('AuSOC', 'g')])
def test_T4_isotropic_d_block_is_bit_identical(latFile, spin, tmp_path_factory):
    coords, contacts = junction_coords()
    g = build(coords, contacts, latFile, tmp_path_factory, spin=spin)
    assert g.Edict['dd'] == g.Edict['dt']
    for c in range(2):
        assert np.array_equal(np.asarray(g.gList[c].H0), np.asarray(g.H0))
    if spin == 'g':                                             # Review Focus 3
        assert g.SOC
        assert np.asarray(g.sigma(float(g.gList[0].fermi), 0)).shape == (2 * 9 * 14, 2 * 9 * 14)


def test_T5_rejects_non_fcc_direction_lists(au2_pair):
    g = au2_pair[0]
    rng = np.random.default_rng(7)
    rand = [v / np.linalg.norm(v) for v in rng.normal(size=(12, 3))]
    with pytest.raises(ValueError):
        g.constructOnsite(rand)


def test_T6_twin_stacking_uses_the_stored_directions(tmp_path_factory):
    patch = hexagon((0, 0, 0), (1, 0, 0), (0, 1, 0))
    n, u = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    twin = [np.asarray(v) for v in surfGB.genNeighbors(None, n, -u)]   # nVecs2 (surfGBethe.py:150)
    device = patch[-1] + DNN * twin[9]          # out-of-plane neighbour of the LAST contact atom, -x side
    g = build(patch + [device], [list(range(1, 8))], 'Au2', tmp_path_factory)
    for dS, dT in zip(g.dirLists[0], twin):     # precondition: the twin list is the one stored
        np.testing.assert_allclose(np.asarray(dS), dT, atol=1e-10, rtol=0)
    assert multiplicities(gamma_levels(g, 0), 1e-10) == [1, 2, 3, 3]


def test_T7_relaxed_device_contact(tmp_path_factory):            # Review Focus 1
    from test_bethe_direction_exclusion import parse_gjf_coords, PLANE1_ATOMS
    geom = os.path.join(REPO, 'paper', 'geometries', 'Au20AntPDT.gjf')
    if not os.path.exists(geom):
        pytest.skip('geometry file not present')
    coords, _ = parse_gjf_coords(geom)
    sub = [np.asarray(coords[a - 1], dtype=float) for a in PLANE1_ATOMS]
    g = build(sub, [list(range(1, len(sub) + 1))], 'Au2', tmp_path_factory)
    assert multiplicities(gamma_levels(g, 0), 1e-10) == [1, 2, 3, 3]


@pytest.fixture(scope='module')
def ausoc_pair(tmp_path_factory):
    A = hexagon((0, 0, 0), (1, 0, 0), (0, 1, 0)) + [np.array([-5.0, 0.0, 0.0])]
    B = [R_GENERIC @ p for p in A]
    gA = build(A, [list(range(1, 8))], 'AuSOC', tmp_path_factory, spin='g')
    gB = build(B, [list(range(1, 8))], 'AuSOC', tmp_path_factory, spin='g')
    return gA, gB


def test_soc_lattice_rigid_rotation_invariance(ausoc_pair):
    """A correct atomic L.S is rotation invariant, so the SOC lattice's Gamma levels cannot
    depend on how the junction is oriented (spec section 7, test (b))."""
    gA, gB = ausoc_pair
    assert gA.SOC and gB.SOC
    assert_labels_match(gA, gB, R_GENERIC)
    np.testing.assert_allclose(gamma_levels(gA, 0), gamma_levels(gB, 0), atol=1e-9, rtol=0)
