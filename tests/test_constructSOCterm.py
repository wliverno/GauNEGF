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
            V[i, l+mi] = ((-1.0)**mi)/np.sqrt(2)
            V[i, l-mi] = 1.0/np.sqrt(2)
        else:
            V[i, l+mi] = -((-1.0)**mi)*1j/np.sqrt(2)
            V[i, l-mi] = 1j/np.sqrt(2)
    return V


def genLSMatrix(l: int):
    """Generate L.S matrix for angular momentum l, in INTERLEAVED spin basis.

    Returns a (2*(2l+1)) x (2*(2l+1)) matrix in the basis
    (orb_0_up, orb_0_dn, orb_1_up, orb_1_dn, ..., orb_(2l)_up, orb_(2l)_dn),
    matching the kron(H0, eye(2)) convention used by surfGBethe.

    Previous version of this reference had two bugs (now fixed):
    (1) used V.conj().T @ L @ V which is the wrong transformation direction
        (gave a matrix that was NOT L in the real orbital basis -- in particular
        the diagonal was non-zero, violating <real_i|Lz|real_i> = 0)
    (2) used np.block([[Lz, L-], [L+, -Lz]]) which is a SPIN-MAJOR layout
        (first all-orbs-up, then all-orbs-dn), incompatible with the
        interleaved kron(H0, eye(2)) it was being added to in production.
    """
    Lx, Ly, Lz = LOps(l)
    V = genOrbList(l)
    # V[i, j] = <Y_{m_j} | real_i>, so |real_i> = sum_j V[i,j] |Y_j>.
    # Therefore A_real[i,k] = sum_{m,n} V*[i,m] A[m,n] V[k,n]
    #                       = (V.conj() @ A @ V.T)[i,k]
    Lx = V.conj() @ Lx @ V.T
    Ly = V.conj() @ Ly @ V.T
    Lz = V.conj() @ Lz @ V.T
    sigx = np.array([[0, 1], [1, 0]], dtype=complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigz = np.array([[1, 0], [0, -1]], dtype=complex)
    return 0.5 * (np.kron(Lx, sigx) + np.kron(Ly, sigy) + np.kron(Lz, sigz))


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

    def test_diagonal_zero_in_real_basis(self):
        """L.S has zero diagonal in any real orbital basis.

        L.S = LxSx + LySy + LzSz. In real orbital basis, all three of Lx, Ly, Lz
        have zero diagonal (real orbs are linear combinations of +/-m_l spherical
        harmonics, so <real_i|Lz|real_i> = (m + (-m))/2 = 0, similarly for Lx, Ly).
        Therefore <real_i, s|L.S|real_i, s> = 0 for any s. This test catches a
        wrong-direction basis transform (V^H L V vs V L V^H) which produces a
        matrix that is NOT L in the real basis and has nonzero diagonals.
        """
        lambdas = [0.0, 1.0, 1.0]
        Hsoc = constructSOCterm(lambdas)
        diag = np.diag(Hsoc)
        assert np.allclose(diag, 0, atol=1e-10), \
            f"L.S diagonal in real-orbital basis must be zero, got nonzero entries:\n{diag}"

    def test_interleaved_intra_orbital_spin_flip_zero(self):
        """H[2i, 2i+1] = 0 for all orbitals i in interleaved basis.

        Within a single real orbital, <orb_up | L.S | orb_dn> = 0.5*<orb|Lx|orb>
        + 0.5j*<orb|Ly|orb> = 0 since Lx and Ly have zero diagonal in real basis.
        If Hsoc were in spin-major layout (first all-orbs-up, then all-orbs-dn),
        H[2i, 2i+1] would correspond to crossing orbital i with orbital i+1 in
        the up sector, which is generally NOT zero -- so this test catches a
        spin-major-vs-interleaved mismatch.
        """
        lambdas = [1.0, 1.0, 1.0]
        Hsoc = constructSOCterm(lambdas)
        for i in range(9):
            assert np.isclose(Hsoc[2*i, 2*i + 1], 0, atol=1e-10), \
                f"Intra-orbital spin-flip H[{2*i},{2*i+1}] = {Hsoc[2*i, 2*i+1]} (should be 0)"

    def test_interleaved_consistent_with_kron_h0(self):
        """kron(H0_9x9, eye(2)) + Hsoc has the same diagonal as kron(H0_9x9, eye(2)).

        Because L.S has zero diagonal in real-orbital basis (test above) and
        interleaved kron preserves orbital diagonal in 2x2 blocks, adding the
        two matrices must not modify the diagonal. A spin-major Hsoc would
        contaminate diagonal entries of orbital pairs that are nonzero in L.S.
        """
        lambdas = [0.0, 0.05, 0.023]  # realistic Au-like
        Hsoc = constructSOCterm(lambdas)
        # Fake H0: distinct diagonal entries for s, p, d to make any contamination visible
        H0 = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        H_total = np.kron(H0, np.eye(2)) + Hsoc
        expected_diag = np.kron(np.diag(H0), np.array([1.0, 1.0]))
        actual_diag = np.diag(H_total).real
        assert np.allclose(actual_diag, expected_diag, atol=1e-10), \
            f"kron(H0,I2)+Hsoc diagonal does not match kron(diag(H0), [1,1])\n" \
            f"Got: {actual_diag}\nExpected: {expected_diag}"

    def test_d_block_no_spin_block_segregation(self):
        """The d-block should NOT show 5x5 spin-block segregation.

        In the buggy spin-major layout, ||H[8:13, 8:13]||_F was zero (spin-up
        block of Lz only) and ||H[13:18, 13:18]||_F was large. In the correct
        interleaved layout, both halves should have comparable non-zero norms
        because spin-up and spin-dn d-orbital pairs are interleaved.
        """
        lambdas = [0.0, 0.0, 1.0]
        Hsoc = constructSOCterm(lambdas)
        # In interleaved layout: rows 8,10,12,14,16 are d_up; rows 9,11,13,15,17 are d_dn.
        # H[8:13, 8:13] is d_up--d_up + d_dn--d_dn mixed in 5 alternating rows,
        # and should have non-trivial norm comparable to the full d-block.
        n_block1 = np.linalg.norm(Hsoc[8:13, 8:13])
        n_block2 = np.linalg.norm(Hsoc[13:18, 13:18])
        n_full = np.linalg.norm(Hsoc[8:18, 8:18])
        assert n_block1 > 0.01 * n_full, \
            f"H[8:13, 8:13] norm {n_block1} is too small vs full {n_full} -- " \
            f"suggests spin-major layout (would give ~0 for spin-up-only block)"
        # And the spin-flip cross-block should also be non-trivial
        n_cross = np.linalg.norm(Hsoc[8:13, 13:18])
        assert n_cross > 0.01 * n_full, \
            f"Cross-block norm {n_cross} unexpectedly small"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
