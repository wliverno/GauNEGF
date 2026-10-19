"""Bethe-lattice attachment audit invariants (2026-08-20 evidence pass).

Cross-checks surfGB's per-atom direction assignments (gauNEGF/surfGBethe.py
__init__ :123-186, sigma assembly :563-567) against INDEPENDENT geometry
truth computed here from the .gjf coordinates, for a small-tip control
(AuTipPDT) and the wide/extended contacts (AuWidePDT, Au20AntPDT).

Invariants, evaluated in the frame of the STORED contact dirList (the frame
in which the Slist/Vlist sigma matrices are actually constructed):
  1. NO DOUBLE COUNTING: no attached sigma direction (index in
     set(range(9)) - nInds) points at an explicit atom of the device.
  2. NO MISSING DIRECTIONS: every vacant direction among slots 0-8 carries
     a sigma.
  3. Every real nearest neighbor of a contact atom maps onto one of the 12
     lattice directions (dot > 0.9) -- guards against a per-atom
     orientation flip (nVecs1 vs nVecs2, :163-171) that disagrees with the
     stored dirList (:186 keeps only the LAST atom's orientation).

Truth occupancy uses a fixed 2.5-3.2 A window (Au NN ~2.84 A here), NOT
the code's own latDist window, so it is an independent check.

Construction reuses the mock-bar recipe of test_bethe_direction_exclusion
but patches out surfGBAt.calcFermi (a ~30-60 s bisection irrelevant to the
geometry bookkeeping under test) to keep the suite fast.

Audit provenance: paper/studies/02_pdt_bethe/scratch_bethe_audit/
bethe_attach_audit.{py,out} (job 38698855) -- all campaign systems came
out clean; these tests are the green regression guard for that finding.
"""
import contextlib
import io
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gauNEGF.surfGBethe import surfGB, surfGBAt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GEOM_DIR = os.path.join(REPO, 'paper', 'geometries')
BETHE_SRC = os.path.join(REPO, 'Au.bethe')

# Contact lists from paper/studies/02_pdt_bethe/systems.py (owner-confirmed).
SYSTEMS = {
    'AuTipPDT':   [[1, 2, 3], [20, 21, 22]],
    'AuWidePDT':  [list(range(23, 38)), list(range(38, 53))],
    'Au20AntPDT': [[1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                   [37, 38, 39, 40, 41, 42, 43, 44, 45, 49, 50, 51, 52, 53, 54]],
}

NN_MIN, NN_MAX = 2.5, 3.2  # A; independent truth window around Au NN 2.84 A


def parse_gjf_coords(path):
    """Read Cartesian coords from a .gjf (see test_bethe_direction_exclusion)."""
    with open(path) as f:
        lines = f.readlines()
    start = None
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2:
            try:
                int(parts[0]); int(parts[1])
            except ValueError:
                continue
            start = idx
            break
    assert start is not None, f'no charge/multiplicity line in {path}'
    coords = []
    for line in lines[start + 1:]:
        parts = line.split()
        if len(parts) != 4:
            break
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float)


def _construct_surfGB(coords, contacts, tmp_path_factory):
    """Mock-bar surfGB construction; calcFermi patched (geometry-independent)."""
    coords = np.asarray(coords, dtype=float)
    natoms = len(coords)
    N = natoms * 9

    bar = MagicMock()
    bar.c = (coords / 0.52917721).flatten()  # surfGB scales bar.c by bohr_to_ang
    bar.ibfatm = np.repeat(np.arange(1, natoms + 1), 9)
    bar.ibftyp = np.zeros(N, dtype=int)

    tmp_dir = str(tmp_path_factory.mktemp('bethe'))
    shutil.copy(BETHE_SRC, os.path.join(tmp_dir, 'Au.bethe'))
    old_cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(surfGBAt, 'calcFermi', lambda self, ne, tol=1e-5: 0.0):
            g = surfGB(np.zeros((N, N)), np.eye(N), contacts, bar, latFile='Au')
    finally:
        os.chdir(old_cwd)
    return g


@pytest.fixture(scope='module', params=list(SYSTEMS))
def system_case(request, tmp_path_factory):
    name = request.param
    path = os.path.join(GEOM_DIR, f'{name}.gjf')
    if not os.path.exists(path):
        pytest.skip(f'geometry file not present: {path}')
    coords = parse_gjf_coords(path)
    g = _construct_surfGB(coords, SYSTEMS[name], tmp_path_factory)
    return name, coords, SYSTEMS[name], g


def _occupied_slots(coords, atom_idx0, dirList):
    """Truth: slots j whose stored direction points (dot>0.9) at an explicit
    atom within the NN window. Also returns the neighbor unit vectors."""
    c = coords[atom_idx0]
    units = []
    for c2 in coords:
        d = np.linalg.norm(c2 - c)
        if NN_MIN < d < NN_MAX:
            units.append((c2 - c) / d)
    occ = set()
    for j, dvec in enumerate(dirList):
        if any(np.dot(u, dvec) > 0.9 for u in units):
            occ.add(j)
    return occ, units


def test_no_sigma_points_at_explicit_atom(system_case):
    """Q2 (double counting): attached sigma directions must all be vacant."""
    name, coords, contacts, g = system_case
    for i, contact in enumerate(contacts):
        dirList = [np.asarray(v, dtype=float) for v in g.dirLists[i]]
        for k, atom in enumerate(contact):
            occ, _ = _occupied_slots(coords, atom - 1, dirList)
            attached = set(range(9)) - {int(x) for x in g.nIndLists[i][k]}
            dbl = attached & occ
            assert not dbl, (
                f'{name} contact {i} atom {atom}: sigma attached on slots '
                f'{sorted(dbl)} that point at explicit device atoms '
                f'(occupied={sorted(occ)}, attached={sorted(attached)})')


def test_no_missing_sigma_directions(system_case):
    """Q3 (missing): every vacant slot 0-8 must carry a sigma."""
    name, coords, contacts, g = system_case
    for i, contact in enumerate(contacts):
        dirList = [np.asarray(v, dtype=float) for v in g.dirLists[i]]
        for k, atom in enumerate(contact):
            occ, _ = _occupied_slots(coords, atom - 1, dirList)
            attached = set(range(9)) - {int(x) for x in g.nIndLists[i][k]}
            missing = (set(range(9)) - occ) - attached
            assert not missing, (
                f'{name} contact {i} atom {atom}: vacant slots '
                f'{sorted(missing)} carry no sigma '
                f'(occupied={sorted(occ)}, attached={sorted(attached)})')


def test_every_neighbor_maps_to_a_stored_direction(system_case):
    """Q1/Q4 frame guard: each real neighbor must align (dot>0.9) with a
    stored direction; a miss means a per-atom orientation flip disagreed
    with the contact's stored dirList (surfGBethe.py:186)."""
    name, coords, contacts, g = system_case
    for i, contact in enumerate(contacts):
        dirList = [np.asarray(v, dtype=float) for v in g.dirLists[i]]
        for k, atom in enumerate(contact):
            _, units = _occupied_slots(coords, atom - 1, dirList)
            for u in units:
                best = max(float(np.dot(u, d)) for d in dirList)
                assert best > 0.9, (
                    f'{name} contact {i} atom {atom}: real neighbor along '
                    f'{np.round(u, 3)} matches no stored direction '
                    f'(best dot {best:.3f}) -- orientation frame mismatch')


def test_below_plane_layer_carries_no_sigma(system_case):
    """Q4: slots 9-11 (device-side/out-of-plane inward, where the second
    layer sits below the surface plane) are structurally outside the sigma
    sum (set(range(9)), surfGBethe.py:565) -- verify occupied inward
    neighbors were never recorded as slots 3-5 by mistake."""
    name, coords, contacts, g = system_case
    for i, contact in enumerate(contacts):
        dirList = [np.asarray(v, dtype=float) for v in g.dirLists[i]]
        for k, atom in enumerate(contact):
            occ, _ = _occupied_slots(coords, atom - 1, dirList)
            recorded = {int(x) for x in g.nIndLists[i][k]}
            # recorded slots must agree with truth occupancy exactly
            assert recorded == occ, (
                f'{name} contact {i} atom {atom}: recorded neighbor slots '
                f'{sorted(recorded)} != geometry-truth occupied {sorted(occ)}')
