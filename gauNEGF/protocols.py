"""Protocol definitions for the surfG interface hierarchy.

Two-level design:
- SurfGAtomicProtocol: atomic-level calculators (surfGAt3D, surfGBAt)
- SurfGProtocol: wrapper-level calculators (surfG, surfG3, surfGB, surfGTest)

These are structural (duck-typed) protocols -- classes do NOT need to
inherit from them. They exist for documentation and optional type-checking
with isinstance() via @runtime_checkable.

Limitation: @runtime_checkable only verifies that methods and attributes
exist on the instance. It does not check signatures, return types, or
attribute types. For full correctness, see test_cross_term.py and
test_jit_fermi_shift.py which test functional behavior.
"""

from typing import Protocol, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class SurfGAtomicProtocol(Protocol):
    """Protocol for atomic-level surface Green's function calculators.

    Implemented by: surfGAt3D, surfGBAt

    These classes compute self-energy and cross-term Q for a single atom's
    orbital space. The wrapper classes (surfG3, surfGB) delegate to these
    and map results into the full device basis.

    Callers are responsible for shifting E by dFermi before calling
    sigma() or crossTermQ().
    """

    H0: np.ndarray          # reference onsite Hamiltonian (immutable)
    dFermi: float            # current Fermi shift from reference

    def updateH(self, fermi: float = None) -> None:
        """Update Fermi shift. Does NOT mutate H0 or Vlist0."""
        ...

    def sigma(self, E: complex, *args, **kwargs) -> np.ndarray:
        """Self-energy at energy E."""
        ...

    def crossTermQ(self, E: complex, *args, **kwargs) -> Optional[np.ndarray]:
        """Cross-term Q_sym at energy E.

        Returns None if the contact basis is orthogonal (S_DL = 0).
        """
        ...


@runtime_checkable
class SurfGProtocol(Protocol):
    """Protocol for wrapper-level surface Green's function calculators.

    This is the interface that density.py and integrate.py interact with.

    Implemented by: surfG, surfG3, surfGB, surfGTest

    Note: Parameter names for setF() vary across implementations
    (mu1/mu2 vs muL/muR). This does not affect structural typing.
    """

    F: np.ndarray            # Fock matrix (N x N)
    S: np.ndarray            # Overlap matrix (N x N)
    num_contacts: int        # Number of contacts

    def sigma(self, E: complex, i: int, conv: float = ...) -> np.ndarray:
        """Self-energy for contact i in full device basis (N x N)."""
        ...

    def sigmaTot(self, E: complex, conv: float = ...) -> np.ndarray:
        """Total self-energy from all contacts (N x N)."""
        ...

    def setF(self, F: np.ndarray, mu1: float, mu2: float) -> None:
        """Update Fock matrix and contact chemical potentials."""
        ...

    def crossTermQ(self, E: complex, i: int, conv: float = ...) -> Optional[np.ndarray]:
        """Symmetrized cross-term overlap matrix Q_sym_i in full device basis.

        Returns None if contact i has orthogonal coupling (S_DL = 0).

        The cross-term electron count correction is:
            delta_N_i = -(1/pi) * Im(sum_k w_k * Tr(G_DD^R(z_k) @ Q_sym_i(z_k)))
        """
        ...

    def crossTermQTot(self, E: complex, conv: float = ...) -> Optional[np.ndarray]:
        """Sum of Q_sym over all contacts, or None if all orthogonal."""
        ...
