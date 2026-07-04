"""getFermiContact single-contour (inertia) path tests.

Energy-INDEPENDENT comparison vs the analytic Damle density() first, then the
pure-numpy surfG1D toy. No Gaussian, no GPU -- login node."""
import numpy as np

from gauNEGF.surfG1D import surfG
from gauNEGF.density import (find_lower_bound, densityComplex, density,
                             getFermiContact)


class ConstSigmaG:
    """Mock surfG with an energy-INDEPENDENT (constant) self-energy. Enough for
    find_lower_bound (sigmaTot) and densityComplex (sigmaTot + crossTermQTot)."""

    def __init__(self, F, S, Sig):
        self.F = np.asarray(F)
        self.S = np.asarray(S)
        self._Sig = np.asarray(Sig)
        self.eta = 1e-6

    def sigmaTot(self, E):
        return self._Sig

    def crossTermQTot(self, E):
        # Orthogonal contact: return None so GrIntCross takes its fast path. A
        # zero array would route to _GIntCross, which needs num_contacts /
        # crossTermQ (integrate.py:203,212) that this mock does not implement.
        return None

    def setF(self, F, mu1, mu2):
        pass


def test_single_contour_matches_analytic_density_const_sigma():
    # Energy-INDEPENDENT constant COMPLEX self-energy (broadening). Compare the
    # numeric single-contour densityComplex count against the analytic Damle
    # density() (density.py:295, the routine scf.py:588 uses).
    F = np.diag([-12.0, -3.0]).astype(complex)
    S = np.eye(2, dtype=complex)
    eta_b = 0.1
    Sig = np.diag([-1j * eta_b, -1j * eta_b])  # constant, complex
    g = ConstSigmaG(F, S, Sig)
    mu = 0.0

    Eminf = find_lower_bound(F, S, g)
    assert Eminf is not None

    # Numeric single contour [Eminf, mu].
    Pn, _dNn = densityComplex(F, S, g, Eminf, mu, T=0)
    count_num = float(np.real(np.trace(np.asarray(Pn) @ S)))

    # Analytic Damle reference with the SAME constant Sigma (S = I -> X = I).
    # Eig of F + Sig (NOT F.real): a complex Sig gives complex D, so the
    # diagonal of (D - D.conj().T) in density() is nonzero. Real D would make
    # that diagonal 0 -> 1/0 = inf -> NaN in the density (density.py:336).
    D, V = np.linalg.eig(F + Sig)
    Vc = np.linalg.inv(V.conj().T)
    Gam = 1j * (Sig - Sig.conj().T)
    Pa = np.asarray(density(V, Vc, D, Gam, Eminf, mu))
    count_ana = float(np.real(np.trace(Pa @ S)))

    # Both contacts pull both levels (-12, -3) below mu=0: count ~ 2.
    assert abs(count_num - count_ana) < 5e-3, (count_num, count_ana)
    assert abs(count_num - 2.0) < 5e-2, count_num


def test_getfermi_single_vs_legacy_toy():
    # Pure-numpy surfG1D dimer. The single-contour Fermi energy must agree with
    # the legacy two-contour Fermi energy.
    F = np.diag([-10.0, -1.0]).astype(complex)
    S = np.eye(2, dtype=complex)
    g = surfG(F, S, [[0], [1]], eta=1e-4)
    ne = 1.0  # one filled orbital

    mu_legacy = float(getFermiContact(g, ne, T=0, useInertiaEmin=False))
    mu_single = float(getFermiContact(g, ne, T=0, useInertiaEmin=True))
    # 5e-2 eV: both paths target the same Fermi; they differ only by toy
    # numerical-integration noise (single vs two-contour, T=0), not physics.
    assert abs(mu_single - mu_legacy) < 5e-2, (mu_single, mu_legacy)
