"""
Test the public I/O methods of the NEGF class in gauNEGF.scf.

Tests cover the following public I/O methods:
  - setFock(F_) / getHOMOLUMO() / setDen(P_)
  - setContacts(lContact, rContact) / setSigma(sig, sig2) / getSigma()
  - setVoltage(qV, fermi)
  - updatePulay(nPulay)
  - saveMAT(matfile)
  - SCF(conv, damping, maxcycles, checkpoint, pulay)
  - writeChk() [TDD: expected RED initially]

The test fixture creates an ethane molecule (2 C + 6 H, Z_total=18)
in restricted (B3LYP/LANL2DZ) configuration and runs a minimal SCF loop.
For writeChk tests (13-16), the methods are expected to fail initially
during the TDD red phase.
"""

# Standard library
import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Scientific computing
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

# Testing
import pytest
from scipy import io

# Gaussian interface and NEGF
from gauopen import QCBinAr as qcb
from gauNEGF.scf import NEGF
from gauNEGF.matTools import getDen, getFock


# ===========================================================================
# SESSION-SCOPED FIXTURE: ethane_negf
# ===========================================================================

@pytest.fixture(scope="session")
def ethane_negf(tmp_path_factory):
    """
    Create a temporary directory, copy ethane.gjf, instantiate NEGF, and yield.

    Performs chdir() into scratch directory for NEGF file I/O.
    On teardown, restores the original working directory.
    """
    # Create scratch directory
    scratch_dir = tmp_path_factory.mktemp("scratch_scf_io")

    # Copy ethane.gjf into scratch
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    ethane_src = os.path.join(repo_root, 'examples', 'ethane.gjf')
    ethane_dst = os.path.join(scratch_dir, 'ethane.gjf')
    shutil.copy(ethane_src, ethane_dst)

    # Save original cwd and chdir to scratch
    original_cwd = os.getcwd()
    os.chdir(scratch_dir)

    try:
        # Instantiate NEGF
        negf = NEGF(fn="ethane", basis="lanl2dz", func="b3lyp", spin="r")
        yield negf
    finally:
        # Restore original cwd
        os.chdir(original_cwd)


# ===========================================================================
# FUNCTION-SCOPED AUTOUSE FIXTURE: _ensure_wired
# ===========================================================================

@pytest.fixture(autouse=True)
def _ensure_wired(ethane_negf):
    """
    Re-apply setContacts, setSigma, setVoltage at the start of each test.

    This ensures consistent wiring state across all tests despite mutations
    from previous tests. The fixture is idempotent and uses a session-scoped
    NEGF object.
    """
    ethane_negf.setContacts(lContact=[1], rContact=[2])
    ethane_negf.setSigma(sig=-0.1j)
    ethane_negf.setVoltage(0.0)


# ===========================================================================
# TEST: test_init_state
# ===========================================================================

def test_init_state(ethane_negf):
    """
    Verify initialization: shapes, nelec, locs, Hermiticity of S.
    """
    negf = ethane_negf

    # Check shapes
    assert negf.F.shape == negf.P.shape, f"F shape {negf.F.shape} != P shape {negf.P.shape}"
    assert negf.F.shape == negf.S.shape, f"F shape {negf.F.shape} != S shape {negf.S.shape}"
    assert negf.F.shape[0] == negf.F.shape[1], "Fock matrix not square"

    # Check nelec (ethane: 2*C (Z=6) + 6*H (Z=1) = 18 electrons)
    np.testing.assert_allclose(negf.nelec, 18.0, atol=0.5, err_msg="nelec not close to 18")

    # Check locs matches nsto
    assert len(negf.locs) == negf.nsto, f"len(locs)={len(negf.locs)} != nsto={negf.nsto}"

    # Check S is Hermitian
    np.testing.assert_allclose(negf.S, negf.S.conj().T, atol=1e-10,
                               err_msg="Overlap matrix S not Hermitian")


# ===========================================================================
# TEST: test_setFock_roundtrip
# ===========================================================================

def test_setFock_roundtrip(ethane_negf):
    """
    Save F in Hartree, convert to eV, setFock, verify round-trip.
    """
    negf = ethane_negf

    # Save original Fock in Hartree
    F_orig = negf.F.copy()

    # Convert to eV and call setFock
    negf.setFock(F_orig * 27.211386)

    # Round-trip should recover original Hartree Fock
    np.testing.assert_allclose(negf.F, F_orig, atol=1e-10,
                               err_msg="setFock round-trip failed")


# ===========================================================================
# TEST: test_setDen_roundtrip
# ===========================================================================

def test_setDen_roundtrip(ethane_negf):
    """
    Save P and nelec, call setDen(P), verify nelec and getDen round-trip.
    """
    negf = ethane_negf

    # Save original density and electron count
    P_orig = negf.P.copy()
    nelec_orig = negf.nelec

    # Call setDen
    negf.setDen(P_orig)

    # Check nelec preserved
    np.testing.assert_allclose(negf.nelec, nelec_orig, atol=1e-6,
                               err_msg="setDen changed nelec")

    # Check getDen round-trip
    P_retrieved = getDen(negf.bar, negf.spin)
    np.testing.assert_allclose(P_retrieved, P_orig, atol=1e-10,
                               err_msg="getDen round-trip failed")


# ===========================================================================
# TEST: test_getHOMOLUMO
# ===========================================================================

def test_getHOMOLUMO(ethane_negf):
    """
    Verify getHOMOLUMO returns [HOMO, LUMO] with HOMO < LUMO, both finite.
    """
    negf = ethane_negf

    homo_lumo = negf.getHOMOLUMO()

    # Should be array of length 2
    assert len(homo_lumo) == 2, f"Expected length 2, got {len(homo_lumo)}"

    # HOMO < LUMO
    assert homo_lumo[0] < homo_lumo[1], f"HOMO {homo_lumo[0]} >= LUMO {homo_lumo[1]}"

    # Both finite
    assert np.isfinite(homo_lumo[0]), "HOMO is not finite"
    assert np.isfinite(homo_lumo[1]), "LUMO is not finite"


# ===========================================================================
# TEST: test_setContacts
# ===========================================================================

def test_setContacts(ethane_negf):
    """
    Verify setContacts(lContact=[1], rContact=[2]) returns orbital indices
    and sets nelecContacts correctly.
    """
    negf = ethane_negf

    # Call setContacts
    lInd, rInd = negf.setContacts(lContact=[1], rContact=[2])

    # Both should be ndarrays of positive length
    assert isinstance(lInd, np.ndarray), "lInd not ndarray"
    assert isinstance(rInd, np.ndarray), "rInd not ndarray"
    assert len(lInd) > 0, "lInd is empty"
    assert len(rInd) > 0, "rInd is empty"

    # lInd and rInd should be disjoint (no shared orbitals)
    shared = len(set(lInd) & set(rInd))
    assert shared == 0, f"lInd and rInd share {shared} orbital(s)"

    # nelecContacts should be 12 (Z_C * 2 for two carbons)
    assert negf.nelecContacts == 12, f"Expected nelecContacts=12, got {negf.nelecContacts}"

    # All lInd entries should point to orbitals on atom 1
    for i in lInd:
        assert abs(negf.locs[i]) == 1, f"lInd[{i}] points to atom {abs(negf.locs[i])}, not 1"


# ===========================================================================
# TEST: test_setSigma_scalar
# ===========================================================================

def test_setSigma_scalar(ethane_negf):
    """
    Verify setSigma with scalar works and creates Hermitian Gam1.
    """
    negf = ethane_negf

    # Already wired by _ensure_wired fixture
    # Verify sigma matrices are set and have correct shape
    assert negf.sigma1.shape == negf.F.shape, f"sigma1 shape {negf.sigma1.shape} != F shape {negf.F.shape}"
    assert negf.sigma2.shape == negf.F.shape, f"sigma2 shape {negf.sigma2.shape} != F shape {negf.F.shape}"

    # Gam1 should be Hermitian
    np.testing.assert_allclose(negf.Gam1, negf.Gam1.conj().T, atol=1e-10,
                               err_msg="Gam1 not Hermitian")

    # Eigenvalues of Gam1 should be non-negative (since Gam = sigma - sigma.conj().T in 1j)
    eigvals = np.linalg.eigvalsh(negf.Gam1)
    assert np.all(eigvals >= -1e-8), f"Gam1 has negative eigenvalues: {eigvals[eigvals < 0]}"


# ===========================================================================
# TEST: test_setSigma_dim_mismatch
# ===========================================================================

def test_setSigma_dim_mismatch(ethane_negf):
    """
    Verify setSigma raises Exception for dimension mismatch.
    """
    negf = ethane_negf

    # setSigma with wrong-length vector should raise
    with pytest.raises(Exception):
        negf.setSigma(lContact=[1], rContact=[2], sig=np.array([1.0, 2.0, 3.0]))


# ===========================================================================
# TEST: test_setVoltage
# ===========================================================================

def test_setVoltage(ethane_negf):
    """
    Verify setVoltage sets mu1, mu2, qV, and electric field.
    """
    negf = ethane_negf

    # Call setVoltage with qV = 0.5
    negf.setVoltage(qV=0.5)

    # Check mu1 - mu2 == qV
    assert negf.mu1 - negf.mu2 == pytest.approx(0.5), \
        f"mu1 - mu2 = {negf.mu1 - negf.mu2}, expected 0.5"

    # Check qV is stored
    assert negf.qV == 0.5, f"qV = {negf.qV}, expected 0.5"

    # Check that electric field is stored (calling scalar with one arg should return value)
    try:
        efield_x = negf.bar.scalar("X-EFIELD")
        assert efield_x is not None, "X-EFIELD not set"
    except Exception as e:
        pytest.fail(f"Failed to retrieve X-EFIELD: {e}")


# ===========================================================================
# TEST: test_getSigma
# ===========================================================================

def test_getSigma(ethane_negf):
    """
    Verify getSigma returns (sigma1, sigma2) that match stored values.
    """
    negf = ethane_negf

    # Already wired by fixture
    s1, s2 = negf.getSigma()

    # Should equal stored sigma matrices
    np.testing.assert_array_equal(s1, negf.sigma1, err_msg="getSigma s1 != sigma1")
    np.testing.assert_array_equal(s2, negf.sigma2, err_msg="getSigma s2 != sigma2")


# ===========================================================================
# TEST: test_updatePulay
# ===========================================================================

def test_updatePulay(ethane_negf):
    """
    Verify updatePulay resizes Pulay buffers and validates input.
    """
    negf = ethane_negf

    # Record initial nPulay
    initial_nPulay = len(negf.pList)

    # Call updatePulay(3)
    negf.updatePulay(3)

    # Check shapes
    assert negf.pList.shape == (3, negf.nsto, negf.nsto), \
        f"pList shape {negf.pList.shape} != (3, {negf.nsto}, {negf.nsto})"
    assert negf.DPList.shape == (3, negf.nsto, negf.nsto), \
        f"DPList shape {negf.DPList.shape} != (3, {negf.nsto}, {negf.nsto})"
    assert negf.pMat.shape == (4, 4), f"pMat shape {negf.pMat.shape} != (4, 4)"
    assert negf.pB.shape == (4,), f"pB shape {negf.pB.shape} != (4,)"

    # Check Pulay matrix boundary conditions
    assert negf.pMat[-1, -1] == 0, f"pMat[-1, -1] = {negf.pMat[-1, -1]}, expected 0"
    assert negf.pB[-1] == -1, f"pB[-1] = {negf.pB[-1]}, expected -1"

    # Check pList[0] seeded with current P
    np.testing.assert_allclose(negf.pList[0], negf.P, atol=1e-10,
                               err_msg="pList[0] not seeded with current P")

    # Call updatePulay(8)
    negf.updatePulay(8)
    assert negf.pList.shape == (8, negf.nsto, negf.nsto), \
        f"pList shape {negf.pList.shape} != (8, {negf.nsto}, {negf.nsto})"
    assert negf.pMat.shape == (9, 9), f"pMat shape {negf.pMat.shape} != (9, 9)"

    # Verify raises ValueError for nPulay < 1
    with pytest.raises(ValueError):
        negf.updatePulay(0)

    # Restore original size for subsequent tests
    negf.updatePulay(initial_nPulay)


# ===========================================================================
# TEST: test_saveMAT
# ===========================================================================

def test_saveMAT(ethane_negf, tmp_path):
    """
    Verify saveMAT writes .mat file with all required keys.
    """
    negf = ethane_negf

    # Save to temporary file
    matfile = str(tmp_path / "out.mat")
    negf.saveMAT(matfile)

    # Load and verify keys
    data = io.loadmat(matfile)
    required_keys = ['F', 'sig1', 'sig2', 'S', 'fermi', 'qV', 'spin', 'den', 'conv']
    for key in required_keys:
        assert key in data, f"Key '{key}' not found in saved .mat file"

    # Verify shapes match
    assert data['F'].shape == negf.F.shape, f"F shape mismatch"
    assert data['S'].shape == negf.S.shape, f"S shape mismatch"
    assert data['den'].shape == negf.P.shape, f"den shape mismatch"
    assert data['sig1'].shape == negf.sigma1.shape, f"sig1 shape mismatch"
    assert data['sig2'].shape == negf.sigma2.shape, f"sig2 shape mismatch"


# ===========================================================================
# TEST: test_short_SCF_and_updatePulay_inloop
# ===========================================================================

def test_short_SCF_and_updatePulay_inloop(ethane_negf):
    """
    Run a short SCF loop with huge convergence threshold (exits on maxcycles).
    Then updatePulay and run again. Both should complete without error.
    """
    negf = ethane_negf

    # Ensure wiring is present
    assert hasattr(negf, 'mu1') and hasattr(negf, 'mu2'), "Voltage not set"
    assert hasattr(negf, 'lInd') and hasattr(negf, 'rInd'), "Contacts not set"

    # Run SCF with conv=-1 so the convergence branch never fires; the loop
    # exits via maxcycles. Note: SCF's maxcycles check is `Niter >= maxcycles`
    # AFTER the append for that iteration, so maxcycles=3 produces 4 energies
    # (Niter = 0,1,2,3). We test the I/O shape, not the exact count.
    maxc = 3
    count1, PP1, TotalE1 = negf.SCF(conv=-1, damping=0.02, maxcycles=maxc,
                                     checkpoint=True, pulay=True)

    # Check return values
    assert 1 <= len(TotalE1) <= maxc + 1, \
        f"Expected 1..{maxc+1} energy values, got {len(TotalE1)}"
    assert len(count1) <= len(TotalE1), "count longer than TotalE"
    assert len(PP1) <= len(TotalE1), "PP longer than TotalE"
    assert len(count1) >= 1, "count is empty"

    # Check checkpoint file exists
    checkpoint_file = "ethane_P.mat"
    final_file = "ethane_Final.mat"
    assert os.path.exists(checkpoint_file) or os.path.exists(final_file), \
        f"Checkpoint file not found: {checkpoint_file} or {final_file}"

    # Now updatePulay and run again
    negf.updatePulay(2)
    count2, PP2, TotalE2 = negf.SCF(conv=-1, damping=0.02, maxcycles=2,
                                     checkpoint=True, pulay=True)

    # Should complete without error; same off-by-one applies to maxcycles=2.
    assert 1 <= len(TotalE2) <= 3, \
        f"Expected 1..3 energy values in second SCF, got {len(TotalE2)}"


# ===========================================================================
# TDD RED TESTS: writeChk (expected to fail initially)
# ===========================================================================

def test_writeChk_creates_file(ethane_negf):
    """
    Verify writeChk() creates a checkpoint file with non-zero size.
    """
    negf = ethane_negf

    # Remove any pre-existing chk in the test cwd to ensure we verify
    # this writeChk call actually produced it.
    if os.path.exists("ethane.chk"):
        os.unlink("ethane.chk")

    negf.writeChk()

    assert os.path.exists("ethane.chk"), "ethane.chk not created"
    assert os.path.getsize("ethane.chk") > 0, "ethane.chk is empty"


def test_writeChk_roundtrip(ethane_negf):
    """
    Verify writeChk() output can be re-loaded by qcb.BinAr and the
    overlap matrix survives the round-trip.
    """
    negf = ethane_negf

    negf.writeChk()

    # Load from written file (qcb.BinAr handles .chk via formchk internally)
    bar2 = qcb.BinAr(debug=False, lenint=8, inputfile="ethane.chk")

    overlap_orig = np.array(negf.bar.matlist["OVERLAP"].expand())
    overlap_read = np.array(bar2.matlist["OVERLAP"].expand())

    np.testing.assert_allclose(overlap_orig, overlap_read, atol=1e-8,
                               err_msg="Overlap matrix mismatch after writeChk roundtrip")


def test_writeChk_no_fchk_artifact(ethane_negf):
    """
    The current writeChk implementation routes BAF -> .chk via gauopen's
    direct unfchk path, so no .fchk intermediate is created. Verify
    nothing of that sort is left behind.
    """
    negf = ethane_negf

    # Clean slate
    if os.path.exists("ethane.fchk"):
        os.unlink("ethane.fchk")

    negf.writeChk()

    assert not os.path.exists("ethane.fchk"), \
        "ethane.fchk should not be produced by writeChk"


def test_writeChk_idempotent_overwrite(ethane_negf):
    """
    Calling writeChk twice in a row should leave a valid .chk in place
    (overwrite, not append-corrupt).
    """
    negf = ethane_negf

    negf.writeChk()
    size1 = os.path.getsize("ethane.chk")
    negf.writeChk()
    size2 = os.path.getsize("ethane.chk")

    assert size2 > 0, "ethane.chk empty after second writeChk"
    # Sizes should be the same or very close: same content written twice.
    # Allow a small slack for any temp metadata embedded by Gaussian.
    assert abs(size2 - size1) < max(1024, size1 // 100), \
        f"writeChk sizes differ unexpectedly: {size1} vs {size2}"
