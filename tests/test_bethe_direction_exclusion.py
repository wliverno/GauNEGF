"""Regression test pinning the multi-layer Bethe-lattice exclusion invariant.

INVARIANT (verbatim intent): multi-layer Bethe contacts are valid "as long
as the out of plane atoms that actually correspond to the atoms of the
bottom layer are NOT included in the sigma."

Mechanism (gauNEGF/surfGBethe.py):
  - surfGB.__init__ (:104-186) detects each contact atom's REAL geometric
    neighbors (any other atom in the full coordinate set within
    0.8x-1.2x the nearest-neighbor distance), maps each neighbor unit
    vector onto the 12 FCC[111] lattice directions via a dot-product
    argmax (threshold > 0.9), and records the per-atom list of OCCUPIED
    direction indices in self.nIndLists.
  - surfGB.sigma (:499-582) attaches a semi-infinite Bethe branch only on
    the VACANT directions of the 9 "surface" directions:
        sigInds = set(range(9)) - {occupied directions for that atom}
        sigAtom = sum(sigSurf[j] for j in sigInds)
    (surfGBethe.py:550/565 for SOC/non-SOC respectively; the assembly used
    here mirrors :563-567). The remaining "inward" triple of direction
    indices (9, 10, 11) is structurally never summed at all (sigInds only
    ever draws from range(9)), regardless of whether it is recorded as
    occupied.

Direction index layout produced by genNeighbors (surfGBethe.py:226-301),
given a per-contact plane_normal that points AWAY from the device/molecule
into the bulk contact:
    0, 1, 2   in-plane vectors (60 degrees apart)
    3, 4, 5   OUTWARD out-of-plane vectors (further from the molecule,
              i.e. towards vacuum/bulk beyond the explicit cluster)
    6, 7, 8   = -(0, 1, 2)  (in-plane, opposite sense)
    9, 10, 11 = -(3, 4, 5)  (INWARD out-of-plane, i.e. towards the molecule)

For a 2-layer contact (an outer "plane 1" that is the true surface facing
the implicit bulk, and an inner "plane 2" between plane 1 and the
molecule): plane 2's real neighbor toward plane 1 must show up as one
OCCUPIED outward direction (one of 3, 4, 5) so that surfGB.sigma does NOT
also attach a redundant Bethe branch on top of that real, explicit bond
(which would double-count it). Plane 1's real neighbors toward plane 2 show
up in the inward triple (9, 10, 11), which never enters the sigma sum
regardless (a structural, not bookkeeping-dependent, exclusion).

Login-node constraints (see task brief): no Gaussian, no NEGF/NEGFE
construction from .gjf files, JAX_PLATFORMS=cpu only. This test builds
surfGB directly from a MagicMock 'bar' object carrying only the few
attributes surfGB.__init__ actually reads (bar.c, bar.ibfatm, bar.ibftyp),
using real coordinates parsed out of paper/geometries/Au20AntPDT.gjf, so it
needs neither Gaussian nor a real Gaussian interface object.

Run with (from repo root):
    JAX_PLATFORMS=cpu python3 -m pytest tests/test_bethe_direction_exclusion.py -q
Construction is not free: each surfGB(...) call runs a small bisection to
find the intrinsic Bethe-lattice bulk Fermi level (surfGBAt.calcFermi),
independent of device size, taking on the order of 30-60 seconds. This
file builds two independent surfGB objects (module-scoped fixtures, each
built once and shared across the tests that need it), so the whole file
takes on the order of a couple of minutes -- expected, not a regression.

Do NOT mark these tests @pytest.mark.slow: pytest.ini sets
`addopts = -m "not slow"`, which would silently deselect them from a plain
`pytest` invocation (0 tests "passing" is not the same as this invariant
being checked).
"""
import contextlib
import io
import os
import shutil
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gauNEGF.surfGBethe import surfGB
from gauNEGF.config import SURFACE_GREEN_CONVERGENCE

# ---------------------------------------------------------------------------
# Geometry / contact bookkeeping (ground truth from the task brief, verified
# 2026-07-11 on klone-login01 against paper/geometries/Au20AntPDT.gjf)
# ---------------------------------------------------------------------------

GEOM_PATH = os.path.join(os.path.dirname(__file__), '..', 'paper',
                          'geometries', 'Au20AntPDT.gjf')
BETHE_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'Au.bethe')

# Left contact = 6-atom outer ("plane 1") layer + 9-of-12 second ("plane 2")
# layer atoms (the other 3 of that 12-atom layer are close enough to the
# molecule to be treated as explicit device atoms instead of contact atoms).
LEFT_CONTACT = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18]
RIGHT_CONTACT = [37, 38, 39, 40, 41, 42, 43, 44, 45, 49, 50, 51, 52, 53, 54]
PLANE1_ATOMS = [1, 2, 3, 4, 5, 6]
PLANE2_ATOMS = [10, 11, 12, 13, 14, 15, 16, 17, 18]

# Right contact: same physical two-layer design, but the FILE ORDERING is
# REVERSED relative to the left (inner plane-2 atoms 37-45 listed BEFORE
# the outer plane-1 atoms 49-54). This ordering is exactly the layout
# under which an ordering-sensitive regression would hide, so the right
# side gets its own mirrored structural assertions (review finding 1,
# 2026-07-11; invariant field-verified on this geometry the same day).
RIGHT_PLANE1_ATOMS = [49, 50, 51, 52, 53, 54]
RIGHT_PLANE2_ATOMS = [37, 38, 39, 40, 41, 42, 43, 44, 45]

OUTWARD_OOP = {3, 4, 5}    # further from the molecule (vacuum/bulk side)
INWARD_OOP = {9, 10, 11}   # towards the molecule

# Environment-pinned exact occupied-direction lists (see
# test_exact_occupied_directions_environment_pinned docstring for caveats).
EXPECTED_OCCUPIED = {
    1: [1, 2, 6, 7, 9, 10, 11],
    2: [0, 1, 2, 8, 9, 10, 11],
    3: [0, 6, 7, 8, 9, 10, 11],
    4: [6, 7, 9, 10, 11],
    5: [1, 2, 9, 10, 11],
    6: [0, 8, 9, 10, 11],
    10: [0, 1, 2, 3],
    11: [1, 2, 4, 6],
    12: [0, 1, 2, 3, 8],
    13: [1, 2, 4, 6, 7],
    14: [5, 6, 7, 8],
    15: [0, 5, 6, 7, 8],
    16: [2, 4, 6, 7],
    17: [0, 5, 7, 8],
    18: [0, 1, 3, 8],
}


def parse_gjf_coords(path):
    """Minimal Gaussian .gjf Cartesian-coordinate reader.

    Finds the charge/multiplicity line (the first line with exactly two
    whitespace-separated integer tokens), then reads "Element x y z" lines
    until a blank line. Returns (coords (N,3) float ndarray, elements list).
    """
    with open(path) as f:
        lines = f.readlines()

    charge_mult_idx = None
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2:
            try:
                int(parts[0])
                int(parts[1])
            except ValueError:
                continue
            charge_mult_idx = idx
            break
    assert charge_mult_idx is not None, f'no charge/multiplicity line found in {path}'

    coords = []
    elements = []
    for line in lines[charge_mult_idx + 1:]:
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 4:
            break
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            break
        elements.append(parts[0])
        coords.append(xyz)

    return np.array(coords, dtype=float), elements


def _construct_surfGB(coords, contacts, tmp_path_factory, latFile='Au'):
    """Build a surfGB directly (no Gaussian, no NEGF/NEGFE) via a MagicMock
    'bar' carrying only the attributes surfGB.__init__ reads.

    Returns (g, stdout_text): stdout_text is everything printed during
    construction, captured manually (not via pytest's capsys) because this
    helper backs module-scoped fixtures and capsys is function-scoped only.

    Handles the CWD dependency of surfGB.readBetheParams (opens
    ./{latFile}.bethe relative to the current directory) by copying the
    repo's {latFile}.bethe into a fresh tmp dir and chdir'ing there for the
    duration of construction, restoring the original cwd afterward.
    """
    coords = np.asarray(coords, dtype=float)
    natoms = len(coords)
    N = natoms * 9

    bar = MagicMock()
    bar.c = (coords / 0.52917721).flatten()  # surfGB multiplies bar.c by bohr_to_ang
    bar.ibfatm = np.repeat(np.arange(1, natoms + 1), 9)
    bar.ibftyp = np.zeros(natoms * 9, dtype=int)

    tmp_dir = str(tmp_path_factory.mktemp('bethe'))
    shutil.copy(BETHE_SRC_PATH, os.path.join(tmp_dir, f'{latFile}.bethe'))

    old_cwd = os.getcwd()
    os.chdir(tmp_dir)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            g = surfGB(np.zeros((N, N)), np.eye(N), contacts, bar, latFile=latFile)
    finally:
        os.chdir(old_cwd)
    return g, buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def au20_geometry():
    if not os.path.exists(GEOM_PATH):
        pytest.skip(f'geometry file not present: {GEOM_PATH}')
    return parse_gjf_coords(GEOM_PATH)


@pytest.fixture(scope='module')
def au20_bethe(au20_geometry, tmp_path_factory):
    """The real, verified-working recipe: full 54-atom Au20AntPDT geometry,
    both contacts (left + right), two explicit atomic layers per side."""
    coords, _elements = au20_geometry
    return _construct_surfGB(coords, [LEFT_CONTACT, RIGHT_CONTACT], tmp_path_factory)


@pytest.fixture(scope='module')
def plane1_only_bethe(au20_geometry, tmp_path_factory):
    """Control: ONLY the 6 plane-1 (outer) atoms exist in the whole
    coordinate set (nothing physically present beyond them), and they are
    the sole contact -- the single-explicit-layer-per-side case used by
    every other campaign system."""
    coords, _elements = au20_geometry
    plane1_coords = coords[:6]
    return _construct_surfGB(plane1_coords, [[1, 2, 3, 4, 5, 6]], tmp_path_factory)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_construction_has_no_lattice_vec_warning(au20_bethe):
    """Pins: a correct 2-layer, 2-contact construction must not trigger the
    geometric neighbor-classification fallback warning ('Warning: Lattice
    Vec #... mismatch, neighbor not recorded', surfGBethe.py:181). Scope
    (review 2026-07-11, mutation D2): this catches CLASSIFICATION-rejection
    failures (a detected neighbor that maps onto no lattice direction).
    A neighbor silently dropped earlier, at the distance filter
    (surfGBethe.py:160), never reaches the classify-or-warn path and is
    caught by the plane1/plane2 structural tests instead.
    """
    _g, stdout_text = au20_bethe
    assert 'Lattice Vec' not in stdout_text, (
        'unexpected neighbor-classification warning during construction:\n'
        + stdout_text)


def test_plane2_atoms_occupy_exactly_one_outward_direction(au20_bethe):
    """Core invariant, plane-2 side: a second-layer contact atom's real
    neighbor toward the first (outer) layer must be recorded as an
    OCCUPIED outward direction (one of indices 3, 4, 5 -- surfGBethe.py:165
    outOfPlane list / :169 dirInd threshold), so that surfGB.sigma's
    vacant-direction sum (surfGBethe.py:550/565) excludes it.

    If this regresses to zero occupied outward directions, the Bethe
    lattice would ALSO attach a semi-infinite branch on top of the real,
    explicit plane-1 neighbor -- double-counting that bond. If it
    regresses to more than one, real in-plane bonds would be wrongly
    excluded from sigma instead.
    """
    g, _stdout = au20_bethe
    for atom in PLANE2_ATOMS:
        k = LEFT_CONTACT.index(atom)
        occupied = {int(x) for x in g.nIndLists[0][k]}
        outward_hits = occupied & OUTWARD_OOP
        assert len(outward_hits) == 1, (
            f'atom {atom}: expected exactly one of {OUTWARD_OOP} occupied, '
            f'got {outward_hits} (full occupied set {sorted(occupied)})')


def test_plane1_atoms_occupy_all_inward_directions(au20_bethe):
    """Core invariant, plane-1 side: outer-layer contact atoms have REAL
    neighbors in the plane-2 direction, so all three inward directions
    {9, 10, 11} must show up as occupied. These indices are structurally
    never summed in surfGB.sigma regardless (sigInds only ever draws from
    set(range(9)) -- surfGBethe.py:550/565 -- indices 9-11 are outside that
    range), so plane 1's exclusion is structural rather than
    bookkeeping-driven, unlike plane 2's (which depends on nIndLists
    actually recording the right outward index).

    Also checks the outward directions {3, 4, 5} stay VACANT for plane-1
    atoms -- that vacancy is exactly what attaches the true semi-infinite
    Bethe branch representing bulk beyond the explicit cluster.
    """
    g, _stdout = au20_bethe
    for atom in PLANE1_ATOMS:
        k = LEFT_CONTACT.index(atom)
        occupied = {int(x) for x in g.nIndLists[0][k]}
        assert INWARD_OOP.issubset(occupied), (
            f'atom {atom}: expected inward directions {INWARD_OOP} all '
            f'occupied (real plane-2 neighbors), got {sorted(occupied)}')
        assert occupied.isdisjoint(OUTWARD_OOP), (
            f'atom {atom}: outward directions {OUTWARD_OOP} must stay '
            f'vacant (that vacancy is where the Bethe branch attaches), '
            f'found overlap {occupied & OUTWARD_OOP}')


def test_right_contact_plane2_atoms_occupy_exactly_one_outward_direction(au20_bethe):
    """Right-contact mirror of the plane-2 core invariant (review finding 1,
    2026-07-11). The right contact's FILE ORDERING is reversed relative to
    the left (inner plane-2 atoms 37-45 listed before outer plane-1 atoms
    49-54) -- exactly the layout an ordering-sensitive regression would
    hide in, so it gets its own assertions rather than trusting symmetry.
    """
    g, _stdout = au20_bethe
    for atom in RIGHT_PLANE2_ATOMS:
        k = RIGHT_CONTACT.index(atom)
        occupied = {int(x) for x in g.nIndLists[1][k]}
        outward_hits = occupied & OUTWARD_OOP
        assert len(outward_hits) == 1, (
            f'right-contact atom {atom}: expected exactly one of '
            f'{OUTWARD_OOP} occupied, got {outward_hits} '
            f'(full occupied set {sorted(occupied)})')


def test_right_contact_plane1_atoms_occupy_all_inward_directions(au20_bethe):
    """Right-contact mirror of the plane-1 core invariant (review finding 1,
    2026-07-11): all three inward directions occupied, all three outward
    directions vacant, despite the reversed atom-listing order.
    """
    g, _stdout = au20_bethe
    for atom in RIGHT_PLANE1_ATOMS:
        k = RIGHT_CONTACT.index(atom)
        occupied = {int(x) for x in g.nIndLists[1][k]}
        assert INWARD_OOP.issubset(occupied), (
            f'right-contact atom {atom}: expected inward {INWARD_OOP} all '
            f'occupied, got {sorted(occupied)}')
        assert occupied.isdisjoint(OUTWARD_OOP), (
            f'right-contact atom {atom}: outward {OUTWARD_OOP} must stay '
            f'vacant, found overlap {occupied & OUTWARD_OOP}')


def test_sigma_actually_excludes_the_recorded_directions(au20_bethe):
    """Sigma-side verification, not just nIndLists bookkeeping: for every
    atom in the left contact, pull its 9x9 diagonal block out of the
    assembled contact sigma and check it against summing sigmaSurf over the
    complement of its recorded occupied set, mirroring surfGB.sigma's own
    assembly by hand (surfGBethe.py:563-567):
        sigInds = set(range(9)) - occupied
        sigAtom = sum(sigSurf[j] for j in sigInds)

    This is what actually protects the physics: nIndLists could be
    bookkept correctly while a refactor of sigma() itself breaks the
    exclusion (e.g. summing over the wrong index set, or over all 9
    unconditionally). Scope (review 2026-07-11): this test validates
    CONSISTENCY between sigma() and nIndLists -- expected values derive
    from nIndLists itself, so it offers zero protection when nIndLists is
    wrong; the plane1/plane2/exact-pinned tests own that failure family. All atoms in one contact share the same underlying
    sigmaSurf (it depends only on E, conv, and the contact's Bethe-lattice
    parameters, not on which atom), so checking every atom here costs one
    extra sigmaSurf evaluation, not fifteen.
    """
    g, _stdout = au20_bethe
    E = -5.0

    sig_full = np.asarray(g.sigma(E, 0, conv=SURFACE_GREEN_CONVERGENCE))
    dF = g.gList[0].dFermi
    sigSurf = np.asarray(g.gList[0].sigmaSurf(E - dF, SURFACE_GREEN_CONVERGENCE))

    for k, atom in enumerate(LEFT_CONTACT):
        occupied = {int(x) for x in g.nIndLists[0][k]}
        sigInds = sorted(set(range(9)) - occupied)
        expected = sum(sigSurf[j] for j in sigInds)

        Finds = np.asarray(g.indsLists[0][k], dtype=int)
        block = sig_full[np.ix_(Finds, Finds)]

        assert np.allclose(block, expected, atol=1e-8), (
            f'sigma block for atom {atom} does not match the hand-'
            f'assembled sum over vacant directions {sigInds} '
            f'(occupied={sorted(occupied)}); max abs diff '
            f'{np.max(np.abs(block - expected)):.3e}')


def test_single_layer_contact_has_no_outward_occupied_directions(plane1_only_bethe):
    """Guards the common (single explicit atomic layer per side) case used
    by every other campaign system: with ONLY the six plane-1 atoms present
    in the whole coordinate set (nothing geometrically beyond them, in
    either out-of-plane sense), no out-of-plane direction -- neither
    outward {3, 4, 5} nor inward {9, 10, 11} -- can ever be a REAL
    neighbor, so surfGB.sigma sums over all 9 surface directions for every
    atom: nothing gets excluded, and no Bethe branch is missing either.
    """
    g, _stdout = plane1_only_bethe
    for k in range(6):
        occupied = {int(x) for x in g.nIndLists[0][k]}
        oop_hits = occupied & (OUTWARD_OOP | INWARD_OOP)
        assert not oop_hits, (
            f'atom index {k} (1-indexed atom {k + 1}): unexpected out-of-'
            f'plane neighbor direction(s) {oop_hits} in a single-layer '
            f'contact where no atoms exist beyond this plane')


def test_exact_occupied_directions_environment_pinned(au20_bethe):
    """Secondary, more sensitive tripwire: exact occupied-direction lists
    captured from a real run on klone-login01 (2026-07-11) building the
    Au20AntPDT left contact. In-plane direction indices (0, 1, 2, and their
    mirror 6, 7, 8) are only fixed up to the arbitrary in-plane rotation
    picked by genNeighbors' first-neighbor-vector convention in
    surfGB.__init__, so this could legitimately shift under a different
    JAX/XLA version or platform even with the mechanism fully intact.

    If ONLY this test fails while the four structural tests above still
    pass, treat that as a likely environment permutation, not a
    regression, and re-capture EXPECTED_OCCUPIED rather than "fixing" the
    code. If a structural test above ALSO fails, that is a real
    regression.
    
    Triage note (review 2026-07-11): before re-capturing after a
    failure here, confirm the shift is a symmetry-consistent index
    RELABELING (same occupied-set sizes, same outward/inward split,
    plane1/plane2 tests still green) -- this is the only test that
    inspects the in-plane indices at all, so an arbitrary in-plane
    misassignment would surface only here and deserves a real look.
    """
    g, _stdout = au20_bethe
    for atom, expected in EXPECTED_OCCUPIED.items():
        k = LEFT_CONTACT.index(atom)
        occupied = sorted(int(x) for x in g.nIndLists[0][k])
        assert occupied == expected, (
            f'atom {atom}: occupied {occupied} != environment-pinned '
            f'{expected} (see docstring: check structural tests before '
            f'treating this alone as a regression)')
