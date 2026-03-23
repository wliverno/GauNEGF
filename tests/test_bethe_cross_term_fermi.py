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
import numpy as np
import jax.numpy as jnp

from gauNEGF.surfGBethe import surfGBAt
from gauNEGF.density import getFermiContact, densityComplexN
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


def test_cross_term_symmetrization_consistency(au_params):
    """Symmetrized Q_sym formula must match unsymmetrized (c_Q, c_Q_rev).

    The cross-term electron count is:
        delta_N = Re[-i/(2pi) * (-c_Q + c_Q_rev*)]

    where c_Q = sum w_k Tr(G^R @ Q_fwd) from lineInt_DL
    and c_Q_rev = sum w_k Tr(G^R @ Q_rev) from lineInt_LD.

    The symmetrized formula delta_N = -(1/pi) Im(c_sym) with
    c_sym = (c_Q + c_Q_rev)/2 must give the same result.

    Re(cross_scalar) != 0 is expected (same as Re[Tr(lineInt_DD @ S)]
    being nonzero). It does not contribute to delta_N.
    """
    import jax.numpy.linalg as LA
    from gauNEGF.density import calcEmin, getANTPoints, fermi as fermi_func
    from gauNEGF.integrate import GrIntCross

    ne, H0, Slist, Vlist = au_params
    gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    ne_per_spin = ne / 2
    fermi_E = gBAt.calcFermi(ne_per_spin)

    # Build wrapper for integration
    gBAt2 = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
    wrapper = _SingleAtomBulkWrapper(gBAt2)
    F = wrapper.F
    S = wrapper.S
    Emin = calcEmin(F, S, wrapper)

    # Contour
    N_pts = 100
    x, w = getANTPoints(N_pts)
    center = (Emin + fermi_E) / 2
    r = (fermi_E - Emin) / 2
    theta = np.pi / 2 * (x + 1)
    Elist = center + r * np.exp(1j * theta)
    dz = 1j * r * np.exp(1j * theta)
    weights = (np.pi / 2) * w * fermi_func(Elist, fermi_E, 0) * dz

    # Method 1: symmetrized (production code path)
    _, cross_scalar = GrIntCross(F, S, wrapper, Elist, weights)
    delta_N_sym = -(1 / np.pi) * float(jnp.imag(cross_scalar))

    # Method 2: unsymmetrized (c_Q and c_Q_rev separately)
    c_fwd = 0.0 + 0j
    c_rev = 0.0 + 0j
    for idx in range(len(Elist)):
        E = Elist[idx]
        wt = weights[idx]
        sigTot = wrapper.sigmaTot(E)
        Gr = jnp.linalg.solve((E + 1j * eta) * S - F - sigTot, jnp.eye(dim))

        sigK = gBAt2.sigmaK(E)
        E_eff = E - gBAt2.dFermi + gBAt2.eta * 1j
        A = E_eff * jnp.eye(dim) - gBAt2.H0
        sigTot_bulk = jnp.sum(sigK, axis=0)

        Q_fwd_tot = jnp.zeros((dim, dim), dtype=complex)
        Q_rev_tot = jnp.zeros((dim, dim), dtype=complex)
        for k in range(12):
            pair_k = (k + 6) % 12
            g_k = LA.inv(A - sigTot_bulk + sigK[pair_k])
            B_k = E_eff * gBAt2.Slist[k] - gBAt2.Vlist0[k]
            B_k_bar = E_eff * gBAt2.Slist[k].conj().T - gBAt2.Vlist0[k].conj().T
            Q_fwd_tot += B_k @ g_k @ gBAt2.Slist[k].conj().T
            Q_rev_tot += gBAt2.Slist[k] @ g_k @ B_k_bar

        c_fwd += wt * jnp.trace(Gr @ Q_fwd_tot)
        c_rev += wt * jnp.trace(Gr @ Q_rev_tot)

    # Full formula: delta_N = Re[-i/(2pi) * (-c_Q + c_Q_rev*)]
    full_expr = (-1j / (2 * np.pi)) * (-c_fwd + np.conj(c_rev))
    delta_N_full = float(np.real(full_expr))
    delta_N_imag = float(np.imag(full_expr))

    print(f"\nc_Q     = {float(jnp.real(c_fwd)):.6e} + {float(jnp.imag(c_fwd)):.6e}j")
    print(f"c_Q_rev = {float(jnp.real(c_rev)):.6e} + {float(jnp.imag(c_rev)):.6e}j")
    print(f"delta_N (symmetrized) = {delta_N_sym:.6f}")
    print(f"delta_N (full formula) = {delta_N_full:.6f}")
    print(f"Im(delta_N_full) = {delta_N_imag:.6e} (should be ~0)")

    # Symmetrized and full formulas must agree
    assert abs(delta_N_sym - delta_N_full) < 1e-10, \
        (f"Symmetrized ({delta_N_sym:.6f}) vs full ({delta_N_full:.6f}) "
         f"differ by {abs(delta_N_sym - delta_N_full):.6e}")

    # delta_N must be purely real (Im part of full expression ~ 0)
    assert abs(delta_N_imag) < 1e-10, \
        f"delta_N has imaginary part {delta_N_imag:.6e} (should be zero)"

    # delta_N must be nonzero (cross-terms contribute for non-orthogonal basis)
    assert abs(delta_N_sym) > 0.01, \
        f"delta_N = {delta_N_sym:.6f} is unexpectedly small for Au"
