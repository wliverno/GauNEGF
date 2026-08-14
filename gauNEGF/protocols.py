"""Protocol definitions for the surfG interface.

SurfGProtocol: the single interface that density.py and integrate.py
interact with.

Implemented by: surfG, surfG3, surfGB, surfGTest, surfGAt3D, surfGBAt

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
class SurfGProtocol(Protocol):
    """Protocol for surface Green's function calculators.

    This is the interface that density.py and integrate.py interact with.

    Implemented by: surfG, surfG3, surfGB, surfGTest, surfGAt3D, surfGBAt

    Note: Parameter names for setF() vary across implementations
    (mu1/mu2 vs muL/muR). This does not affect structural typing.
    """

    F: np.ndarray            # Fock matrix (N x N)
    S: np.ndarray            # Overlap matrix (N x N)
    eta: float               # broadening
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

    def crossTermQ(self, E: complex, i: int, conv: float = ...,
                   dFermi: Optional[float] = None
                   ) -> Optional[tuple]:
        """Cross-term matrices (Q_fwd, Q_rev, Q_sym) for contact i,
        each in the full device basis; None if contact i is orthogonal.

        Q_fwd = tau_Di g_i S_iD ; Q_rev = S_Di g_i taubar_iD ;
        Q_sym = (Q_fwd + Q_rev) / 2.

        Sign conventions for the cross-term count:
          forward real axis:  delta_N = +(1/pi) * Im INT f Tr[Gr Q_sym_tot] dE
          closed contour:     delta_N = contour integral of Tr[Gr Q_sym_tot];
                              orientation carried by traversal, no Im taken.
        """
        ...

    def crossTermQTot(self, E: complex, conv: float = ...,
                      dFermi: Optional[float] = None
                      ) -> Optional[np.ndarray]:
        """Sum of Q_sym over all contacts, or None if all orthogonal."""
        ...
