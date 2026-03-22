"""Regression test: Bethe lattice cross-term Fermi energy consistency.

Validates that surfGBAt.crossTermQBulk produces the correct Mulliken
electron count by comparing two independent Fermi calculations:

  Method 1 (truth): 117x117 extended system, Tr((P@S)[-9:,-9:])
  Method 2 (test):  9x9 single atom + bulk self-energy + cross-terms

Both must agree, proving the cross-term correctly accounts for
center-neighbor overlap in the Mulliken population.
"""
import sys
sys.path.insert(0, '..')

import pytest
import jax.numpy as jnp

from gauNEGF.surfGBethe import surfGBAt
from gauNEGF.density import getFermiContact
from gauNEGF.config import SURFACE_GREEN_CONVERGENCE
from test_surfGAt3D import read_bethe_params, construct_mat, gen_fcc_111_neighbors

dim = 9
eta = 1e-6

# Known benchmark: Au bulk Fermi energy from extended system (117x117)
AU_BULK_FERMI_EV = 2.84


# ---------------------------------------------------------------------------
# Wrapper: presents surfGBAt as a single-atom system for getFermiContact
# ---------------------------------------------------------------------------

class _SingleAtomBulkWrapper:
    """Adapts surfGBAt to a 9x9 system with bulk self-energies + cross-terms.

    sigmaTot returns sum of all 12 bulk sigmaK.
    crossTermQ delegates to surfGBAt.crossTermQBulk (production code).
    """
    def __init__(self, gBAt):
        self.gBAt = gBAt
        self.F = gBAt.H.copy()
        self.S = jnp.eye(gBAt.dim)
        self.num_contacts = 1

    def sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE):
        return jnp.sum(self.gBAt.sigmaK(E, conv), axis=0)

    def crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE):
        return self.gBAt.crossTermQBulk(E, conv)

    def crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE):
        return self.crossTermQ(E, 0, conv)

    def setF(self, F, mu1, mu2):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def au_params():
    """Read Au Bethe parameters and build hopping/overlap matrices."""
    ne, H0, Sdict, Vdict = read_bethe_params('Au')
    vecs = gen_fcc_111_neighbors()
    Vlist = [construct_mat(Vdict, v) for v in vecs]
    Slist = [construct_mat(Sdict, v) for v in vecs]
    return ne, H0, Slist, Vlist


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bethe_fermi_benchmark(au_params):
    """Au bulk Fermi energy should match known benchmark (~2.84 eV)."""
    ne, H0, Slist, Vlist = au_params
    gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    fermi = gBAt.calcFermi(ne / 2)
    assert abs(fermi - AU_BULK_FERMI_EV) < 0.02, \
        f"Au Fermi {fermi:.4f} eV differs from benchmark {AU_BULK_FERMI_EV} eV"


def test_bethe_cross_term_fermi_consistency(au_params):
    """Extended system Fermi must match single-atom + crossTermQBulk Fermi.

    This validates that crossTermQBulk correctly uses the neighbor's
    Green's function g_k = inv(A - sigTot + sigK[pair_k]) for each
    direction, rather than the center atom's own Green's function.
    """
    ne, H0, Slist, Vlist = au_params
    ne_per_spin = ne / 2

    # Method 1: Extended system (117x117) -- analytical truth
    gBAt1 = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    fermi_ext = gBAt1.calcFermi(ne_per_spin)

    # Method 2: Single atom (9x9) + cross-terms via production crossTermQBulk
    gBAt2 = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    wrapper = _SingleAtomBulkWrapper(gBAt2)
    fermi_cross = getFermiContact(wrapper, ne_per_spin, maxcycles=1000, nOrbs=0)

    assert abs(fermi_ext - fermi_cross) < 0.01, \
        (f"Extended ({fermi_ext:.4f} eV) vs cross-term ({fermi_cross:.4f} eV) "
         f"differ by {abs(fermi_ext - fermi_cross):.4f} eV")
