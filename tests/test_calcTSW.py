"""Unit tests for calcTSW: spectral-weight-based integration limits.

Uses a 2-atom C chain (LANL2DZ double-zeta) extended as an infinite 1D chain
via setContact1D. Replaces the previous Au-Bethe-lattice fixture which took
~2 minutes per run; this C2 fixture runs in seconds because the device has
only 18 orbitals and the contact tau/alpha/beta blocks are small.

The double-zeta LANL2DZ basis on a tightly-bonded C-C dimer creates the
near-linear-dependence in S - X that calcPseudoPoleFloor is designed to
detect, so test_calcTSW_with_pseudo_pole_floor exercises the production
wire-up of calcPseudoPoleFloor + calcTSW on a system that may genuinely
have pseudo-poles (depending on the loaded F).
"""
import os
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from gauNEGF.scfE import NEGFE
from gauNEGF.density import calcTSW, calcPseudoPoleFloor
from gauNEGF.transport import har_to_eV
from gauNEGF.config import ENERGY_MIN


# ===========================================================================
# SESSION-SCOPED FIXTURE: c2_chain_negf
# ===========================================================================

@pytest.fixture(scope="session")
def c2_chain_negf(tmp_path_factory):
    """Build a NEGFE for a C2 1D infinite chain (LANL2DZ).

    Copies examples/C2_chain.gjf into a per-session tmp dir, instantiates
    NEGFE there (runs Gaussian for the initial guess), and wires the 1D
    contacts (atom 1 left, atom 2 right). The fixture yields the negf
    object plus its Fock (in eV) and overlap for direct calcTSW calls.

    NEGFE.setContact1D internally calls the deprecated setIntegralLimits,
    which now also calls calcPseudoPoleFloor + calcTSW. The fixture
    therefore captures self.Eminf / self.TSW for tests that want to
    inspect the initial bounds without redoing the search.
    """
    scratch_dir = tmp_path_factory.mktemp("scratch_calcTSW")
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    gjf_src = os.path.join(repo_root, 'examples', 'C2_chain.gjf')
    gjf_dst = os.path.join(scratch_dir, 'C2_chain.gjf')
    shutil.copy(gjf_src, gjf_dst)

    original_cwd = os.getcwd()
    os.chdir(scratch_dir)
    try:
        negf = NEGFE(fn='C2_chain', func='b3lyp', basis='lanl2dz',
                     route='integral=grid=superfine')
        negf.setContact1D([[1], [2]], symmetrize_contacts=True)
        negf.setVoltage(0.0)
        yield negf, negf.F * har_to_eV, negf.S
    finally:
        os.chdir(original_cwd)


# ===========================================================================
# TESTS
# ===========================================================================

def test_calcTSW_converges(c2_chain_negf):
    """calcTSW should converge to a finite Eminf above the hard floor.

    For non-orthogonal contact systems with near-singular S (double-zeta on a
    short C-C bond), the generalized eigenvalues of inv(S)@F include spurious
    near-infinite values from pseudo-poles, so the Bethe-Au "Eminf below all
    eigenvalues" criterion does not apply. The meaningful check is that
    calcTSW returned a finite Eminf (not capped at the hard ENERGY_MIN floor)
    and a positive TSW.
    """
    negf, F, S = c2_chain_negf

    Eminf, TSW = calcTSW(F, S, negf.g)

    assert TSW > 0, f"TSW should be positive, got {TSW}"
    assert Eminf > ENERGY_MIN, \
        f"Eminf hit hard floor ({Eminf} <= {ENERGY_MIN}); calcTSW did not converge"


def test_calcTSW_warm_start(c2_chain_negf):
    """Warm-started calcTSW should return identical bounds (zero expansion)."""
    negf, F, S = c2_chain_negf

    # First call: cold start
    Eminf1, TSW1 = calcTSW(F, S, negf.g)

    # Second call: warm start with converged values
    Eminf2, TSW2 = calcTSW(F, S, negf.g, Eminf=Eminf1, TSW=TSW1)

    assert Eminf2 == Eminf1, f"Warm-started Eminf changed: {Eminf1} -> {Eminf2}"
    assert abs(TSW2 - TSW1) < 1e-6, \
        f"Warm-started TSW changed: {TSW1} -> {TSW2}"


def test_calcTSW_with_pseudo_pole_floor(c2_chain_negf):
    """calcTSW with Emin_floor from calcPseudoPoleFloor should converge
    to the same TSW as the default-floor call.

    Regression: verifies the production wire-up (FockToP path) is
    consistent with calcTSW's standalone behavior. When the floor is
    tighter than ENERGY_MIN, calcTSW only changes its lower cap on
    Eminf doubling; the upper integration bound stays at -ENERGY_MIN
    so the captured physical spectral weight is unchanged.
    """
    negf, F, S = c2_chain_negf

    # Old behavior: default Emin_floor = ENERGY_MIN (from config)
    Eminf_old, TSW_old = calcTSW(F, S, negf.g)

    # New behavior: principled Emin_floor from pseudo-pole detection
    floor = calcPseudoPoleFloor(F, S, negf.g)
    Eminf_new, TSW_new = calcTSW(F, S, negf.g, Emin_floor=floor)

    print(f'Old: Eminf={Eminf_old:.2f}, TSW={TSW_old:.4f}')
    print(f'New: Eminf={Eminf_new:.2f}, TSW={TSW_new:.4f}, floor={floor:.2f}')

    # Physical TSW should match within tolerance (same upper bound, same
    # physical states captured; only the lower cap can shift).
    assert abs(TSW_new - TSW_old) / abs(TSW_old) < 1e-2, \
        f"TSW changed beyond tolerance: {TSW_old} -> {TSW_new}"
