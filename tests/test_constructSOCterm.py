"""
Test-driven development for constructSOCterm function.

This test verifies that constructSOCterm produces the correct L.S matrices
for p and d orbitals, using the verified math from calcSOCOrbs.py.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from gauNEGF.spinTools import constructSOCterm


def LOps(l: int):
    """Generate angular momentum operators in spherical harmonic basis."""
    m = np.arange(-l, l+1)
    Lz = np.diag(m) + 0j
    Lp = np.zeros((2*l+1, 2*l+1), dtype=complex)
    for i, mi in enumerate(m):
        for j, mj in enumerate(m):
            if mj == mi+1:
                Lp[j, i] = np.sqrt((l-mi)*(l+mi+1))
    Lm = Lp.T
    Lx = 0.5*(Lp+Lm)
    Ly = -0.5j*(Lp-Lm)
    return Lx, Ly, Lz


def genOrbList(l: int):
    """Generate transformation matrix from spherical to real orbitals."""
    m = np.zeros(2*l+1, dtype=int)
    for i in range(2*l+1):
        if i == 0:
            m[i] = 0
        elif i % 2 == 1:
            m[i] = (i + 1) // 2
        else:
            m[i] = -((i + 1) // 2)
    V = np.zeros((2*l+1, 2*l+1), dtype=complex)
    for i, mi in enumerate(m):
        if mi == 0:
            V[i, l] = 1.0
        elif mi > 0:
            V[i, l+mi] = (-1.0**mi)/np.sqrt(2)
            V[i, l-mi] = 1.0/np.sqrt(2)
        else:
            V[i, l+mi] = -(-1.0**mi)*1j/np.sqrt(2)
            V[i, l-mi] = 1j/np.sqrt(2)
    return V


def genLSMatrix(l: int):
    """Generate L.S matrix for angular momentum l (verified reference)."""
    Lx, Ly, Lz = LOps(l)
    orbs = genOrbList(l)
    # Use Hermitian conjugate (dagger) for unitary transformation
    Lx = orbs.conj().T @ Lx @ orbs
    Ly = orbs.conj().T @ Ly @ orbs
    Lz = orbs.conj().T @ Lz @ orbs
    # L.S = (1/2) * [[Lz, Lx-iLy], [Lx+iLy, -Lz]]
    return 0.5*np.block([[Lz, Lx-1j*Ly], [Lx+1j*Ly, -Lz]])


class TestConstructSOCterm:
    """Test suite for constructSOCterm using verified reference implementation."""

    def test_hermiticity(self):
        """SOC Hamiltonian must be Hermitian."""
        lambdas = [0.1, 0.2, 0.3]  # arbitrary test values
        Hsoc = constructSOCterm(lambdas)

        assert np.allclose(Hsoc, Hsoc.conj().T), "SOC Hamiltonian must be Hermitian"

    def test_s_block_zero(self):
        """s orbitals (l=0) have zero spin-orbit coupling."""
        lambdas = [1.0, 1.0, 1.0]  # use unit lambdas
        Hsoc = constructSOCterm(lambdas)

        # s-block is first 2x2 (s_up, s_down)
        s_block = Hsoc[:2, :2]
        assert np.allclose(s_block, 0), "s-orbital L.S must be zero (l=0)"

    def test_p_block_correct(self):
        """p-orbital L.S block matches verified reference."""
        lambdas = [0.0, 1.0, 0.0]  # only p-orbital coupling
        Hsoc = constructSOCterm(lambdas)

        # p-block is 6x6 starting at index 2
        p_block = Hsoc[2:8, 2:8]

        # Reference from genLSMatrix(l=1)
        p_reference = genLSMatrix(1)

        assert np.allclose(p_block, p_reference), \
            f"p-orbital L.S block does not match reference\nGot:\n{p_block}\nExpected:\n{p_reference}"

    def test_d_block_correct(self):
        """d-orbital L.S block matches verified reference."""
        lambdas = [0.0, 0.0, 1.0]  # only d-orbital coupling
        Hsoc = constructSOCterm(lambdas)

        # d-block is 10x10 starting at index 8
        d_block = Hsoc[8:18, 8:18]

        # Reference from genLSMatrix(l=2)
        d_reference = genLSMatrix(2)

        assert np.allclose(d_block, d_reference), \
            f"d-orbital L.S block does not match reference\nGot:\n{d_block}\nExpected:\n{d_reference}"

    def test_eigenvalues_p_orbital(self):
        """p-orbital L.S eigenvalues match j-j coupling theory."""
        lambdas = [0.0, 1.0, 0.0]  # unit lambda_p
        Hsoc = constructSOCterm(lambdas)
        p_block = Hsoc[2:8, 2:8]

        eigs = np.sort(np.linalg.eigvalsh(p_block))

        # For l=1: j=1/2 gives E = -1, j=3/2 gives E = +1/2
        # j=1/2 is 2-fold (2j+1=2), j=3/2 is 4-fold (2j+1=4)
        expected = np.array([-1.0, -1.0, 0.5, 0.5, 0.5, 0.5])

        assert np.allclose(eigs, expected, atol=1e-10), \
            f"p-orbital eigenvalues incorrect\nGot: {eigs}\nExpected: {expected}"

    def test_eigenvalues_d_orbital(self):
        """d-orbital L.S eigenvalues match j-j coupling theory."""
        lambdas = [0.0, 0.0, 1.0]  # unit lambda_d
        Hsoc = constructSOCterm(lambdas)
        d_block = Hsoc[8:18, 8:18]

        eigs = np.sort(np.linalg.eigvalsh(d_block))

        # For l=2: j=3/2 gives E = -3/2, j=5/2 gives E = +1
        # j=3/2 is 4-fold (2j+1=4), j=5/2 is 6-fold (2j+1=6)
        expected = np.array([-1.5, -1.5, -1.5, -1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        assert np.allclose(eigs, expected, atol=1e-10), \
            f"d-orbital eigenvalues incorrect\nGot: {eigs}\nExpected: {expected}"

    def test_lambda_scaling(self):
        """SOC matrix scales linearly with lambda parameters."""
        lambdas1 = [0.1, 0.2, 0.3]
        lambdas2 = [0.2, 0.4, 0.6]  # 2x lambdas1

        Hsoc1 = constructSOCterm(lambdas1)
        Hsoc2 = constructSOCterm(lambdas2)

        assert np.allclose(Hsoc2, 2.0 * Hsoc1), \
            "SOC matrix must scale linearly with lambda"

    def test_off_diagonal_blocks_zero(self):
        """s, p, d blocks should not couple to each other."""
        lambdas = [1.0, 1.0, 1.0]
        Hsoc = constructSOCterm(lambdas)

        # s-p coupling
        assert np.allclose(Hsoc[:2, 2:8], 0), "s-p coupling must be zero"
        assert np.allclose(Hsoc[2:8, :2], 0), "p-s coupling must be zero"

        # s-d coupling
        assert np.allclose(Hsoc[:2, 8:18], 0), "s-d coupling must be zero"
        assert np.allclose(Hsoc[8:18, :2], 0), "d-s coupling must be zero"

        # p-d coupling
        assert np.allclose(Hsoc[2:8, 8:18], 0), "p-d coupling must be zero"
        assert np.allclose(Hsoc[8:18, 2:8], 0), "d-p coupling must be zero"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
