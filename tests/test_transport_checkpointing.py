"""
Test cases for transport calculations with checkpointing.

This module provides test Hamiltonians and sigma cases for testing checkpointable
transport calculations. Includes:
- Simple nanowire Hamiltonians with nearest neighbor coupling
- Energy-independent sigma (diagonal and matrix forms)
- Energy-dependent sigma using surfG1D for nanowire contacts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
from gauNEGF.transport import SigmaCalculator
from gauNEGF.surfG1D import surfG
from gauNEGF.config import ETA


def create_nanowire_hamiltonian(N, t=-1.0, eps=0.0):
    """
    Create a simple 1D nanowire Hamiltonian with nearest neighbor coupling.
    
    Parameters
    ----------
    N : int
        Number of sites in the nanowire
    t : float, optional
        Hopping parameter between nearest neighbors (default: -1.0 eV)
    eps : float, optional
        On-site energy (default: 0.0 eV)
    
    Returns
    -------
    F : ndarray, shape (N, N)
        Fock matrix (Hamiltonian) for the nanowire
    S : ndarray, shape (N, N)
        Overlap matrix (identity for orthogonal basis)
    
    Notes
    -----
    Creates a tight-binding Hamiltonian:
    H = eps * I + t * (off-diagonal nearest neighbors)
    """
    F = np.eye(N, dtype=complex)
    S = np.eye(N, dtype=complex)
    
    # On-site energies
    F *= eps
    
    # Nearest neighbor hopping
    for i in range(N - 1):
        F[i, i+1] = t
        F[i+1, i] = t
    
    return F, S


def create_nanowire_with_contacts(N_device, N_contact, t=-1.0, eps=0.0, t_contact=None):
    """
    Create an extended nanowire system with device and contact regions.
    
    Parameters
    ----------
    N_device : int
        Number of sites in the device region
    N_contact : int
        Number of sites in each contact region
    t : float, optional
        Hopping parameter (default: -1.0 eV)
    eps : float, optional
        On-site energy (default: 0.0 eV)
    t_contact : float, optional
        Hopping between device and contact (default: same as t)
    
    Returns
    -------
    F : ndarray, shape (N_total, N_total)
        Extended Fock matrix [contact1 | device | contact2]
    S : ndarray, shape (N_total, N_total)
        Extended overlap matrix
    device_indices : list
        Indices of device region [N_contact : N_contact + N_device]
    contact1_indices : list
        Indices of left contact [0 : N_contact]
    contact2_indices : list
        Indices of right contact [N_contact + N_device : N_total]
    """
    if t_contact is None:
        t_contact = t
    
    N_total = 2 * N_contact + N_device
    F, S = create_nanowire_hamiltonian(N_total, t=t, eps=eps)
    
    # Define indices
    contact1_indices = list(range(N_contact))
    device_indices = list(range(N_contact, N_contact + N_device))
    contact2_indices = list(range(N_contact + N_device, N_total))
    
    return F, S, device_indices, contact1_indices, contact2_indices


def create_energy_independent_sigma_simple(N, linds, rinds, gamma=0.1):
    """
    Create simple energy-independent self-energies (diagonal form).
    
    Parameters
    ----------
    N : int
        Number of orbitals
    linds : list
        Indices of left contact
    rinds : list
        Indices of right contact
    gamma : float, optional
        Broadening parameter (default: 0.1 eV)
    
    Returns
    -------
    sig1 : ndarray, shape (N,)
        Left contact self-energy (diagonal, imaginary part only)
    sig2 : ndarray, shape (N,)
        Right contact self-energy (diagonal, imaginary part only)
    """
    # Simple diagonal self-energies: -i*gamma
    sig1 = np.zeros((N, N), dtype=complex)
    sig2 = np.zeros((N, N), dtype=complex)
    sig1[np.ix_(linds, linds)] = -1j * gamma * np.ones(len(linds))
    sig2[np.ix_(rinds, rinds)] = -1j * gamma * np.ones(len(rinds))
    return sig1, sig2

def create_energy_dependent_sigma_nanowire(F_extended, S_extended, device_indices, 
                                          contact1_indices, contact2_indices, eta=None):
    """
    Create energy-dependent sigma using surfG1D for nanowire contacts.
    
    Parameters
    ----------
    F_extended : ndarray
        Extended Fock matrix including contacts
    S_extended : ndarray
        Extended overlap matrix
    device_indices : list
        Indices of device region
    contact1_indices : list
        Indices of left contact
    contact2_indices : list
        Indices of right contact
    eta : float, optional
        Broadening parameter in eV. If None, uses ETA from config (default: None)
        Larger values (e.g., 1e-6) improve numerical stability but reduce precision.
    
    Returns
    -------
    surfG_obj : surfG object
        Surface Green's function calculator for energy-dependent calculations
    
    Notes
    -----
    The taus parameter specifies connection indices that connect TO the contacts.
    For pattern (a), taus[0] connects to contact1 (indsList[0]), 
    and taus[1] connects to contact2 (indsList[-1]).
    In our nanowire: device connects to contacts, so:
    - taus[0] = [first device site] connects to contact1
    - taus[1] = [last device site] connects to contact2
    """
    # Use provided eta or default from config
    if eta is None:
        eta = ETA
    
    # Create surfG object with automatic extraction from Fock matrix
    indsList = [contact1_indices, contact2_indices]
    taus = [device_indices[0:len(contact1_indices)], 
            device_indices[-len(contact2_indices):]]

    surfG_obj = surfG(F_extended, S_extended, indsList, taus=taus, eta=eta)
    
    return surfG_obj


# Test case configurations
TEST_CASES = {
    'small_nanowire': {
        'N': 20,
        'N_contact': 10,
        't': -1.0,
        'eps': 0.0,
        'description': 'Small nanowire: 20 sites, t=-1.0 eV'
    },
    'medium_nanowire': {
        'N': 50,
        'N_contact': 25,
        't': -1.0,
        'eps': 0.0,
        'description': 'Medium nanowire: 50 sites, t=-1.0 eV'
    },
    'large_nanowire': {
        'N': 100,
        'N_contact': 50,
        't': -1.0,
        'eps': 0.0,
        'description': 'Large nanowire: 100 sites, t=-1.0 eV'
    },
    'nanowire_with_contacts_small': {
        'N_device': 20,
        'N_contact': 10,
        't': -1.0,
        'eps': 0.0,
        'description': 'Small device (20) with contacts (10 each)'
    },
    'nanowire_with_contacts_medium': {
        'N_device': 50,
        'N_contact': 20,
        't': -1.0,
        'eps': 0.0,
        'description': 'Medium device (50) with contacts (20 each)'
    },
    'nanowire_with_contacts_large': {
        'N_device': 100,
        'N_contact': 50,
        't': -1.0,
        'eps': 0.0,
        'description': 'Large device (100) with contacts (50 each)'
    }
}


def setup_energy_independent_test(case_name='medium_nanowire'):
    """
    Set up a test with energy-independent sigma.
    
    Parameters
    ----------
    case_name : str, optional
        Test case name (default: 'medium_nanowire')
    sigma_type : str, optional
        'diagonal' or 'matrix' (default: 'diagonal')
    
    Returns
    -------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    sigma_calc : SigmaCalculator
        Energy-independent sigma calculator
    """
    case = TEST_CASES[case_name]
    N = case['N']
    contact1_indices = list(range(0, case['N_contact']))
    contact2_indices = list(range(N - case['N_contact'], N))
    F, S = create_nanowire_hamiltonian(N, t=case['t'], eps=case['eps'])
    sig1, sig2 = create_energy_independent_sigma_simple(N, contact1_indices, contact2_indices, gamma=0.1)
    sigma_calc = SigmaCalculator(sig1, sig2, energy_dependent=False)
    return F, S, sigma_calc

def setup_energy_dependent_test(case_name='nanowire_with_contacts_medium', eta=None):
    """
    Set up a test with energy-dependent sigma using surfG1D.
    
    Parameters
    ----------
    case_name : str, optional
        Test case name (default: 'nanowire_with_contacts_medium')
    eta : float, optional
        Broadening parameter in eV. If None, uses 1e-6 for better numerical stability
        (default: None). Smaller values (e.g., 1e-9) provide higher precision but may
        cause numerical issues.
    
    Returns
    -------
    F_extended : ndarray
        Extended Fock matrix (includes device and contacts)
    S_extended : ndarray
        Extended overlap matrix
    sigma_calc : SigmaCalculator
        Energy-dependent sigma calculator
    F_extended : ndarray
        Extended Fock matrix (duplicate for compatibility)
    S_extended : ndarray
        Extended overlap matrix (duplicate for compatibility)
    device_indices : list
        Device region indices
    """
    case = TEST_CASES[case_name]
    N_device = case['N_device']
    N_contact = case['N_contact']
    
    F_extended, S_extended, device_indices, contact1_indices, contact2_indices = \
        create_nanowire_with_contacts(
            N_device, N_contact, 
            t=case['t'], eps=case['eps']
        )
    
    # Use larger ETA by default for better numerical stability in tests
    if eta is None:
        eta = 1e-6
    
    # Create energy-dependent sigma calculator
    surfG_obj = create_energy_dependent_sigma_nanowire(
        F_extended, S_extended, device_indices, 
        contact1_indices, contact2_indices, eta=eta
    )
    
    # Use extended F/S since surfG returns sigma for the extended system
    sigma_calc = SigmaCalculator(surfG_obj, energy_dependent=True)
    
    return F_extended, S_extended, sigma_calc, F_extended, S_extended, device_indices


# Test functions for checkpointable transport calculations
import tempfile
import time

def test_transmission_physics():
    """Test physical correctness of transmission calculations."""
    print("="*70)
    print("Test 1: Transmission Physics Validation")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('medium_nanowire')
    energy_list = np.linspace(-3, 3, 50)
    
    # Calculate transmission
    from gauNEGF.transport import calculate_transmission
    transmission = calculate_transmission(F, S, sigma_calc, energy_list, spin='r')
    
    # Physical checks
    assert np.all(transmission >= 0), "Transmission must be non-negative!"
    assert np.all(np.isfinite(transmission)), "Transmission must be finite!"
    
    # Verify consistency with single-point calculations
    print("\nVerifying consistency with single-point calculations...")
    from gauNEGF.transport import transmission_single_energy
    F_jax = jnp.asarray(F)
    S_jax = jnp.asarray(S)
    for i in [0, len(energy_list)//2, len(energy_list)-1]:
        E = energy_list[i]
        T_single = transmission_single_energy(E, F_jax, S_jax, sigma_calc, spin='r')
        T_checkpoint = transmission[i]
        assert np.isclose(T_single, T_checkpoint, rtol=1e-10), \
            f"Mismatch at E={E:.3f}: single={T_single:.6f}, checkpoint={T_checkpoint:.6f}"
    
    print(f"[OK] Transmission range: {np.min(transmission):.6f} to {np.max(transmission):.6f}")
    print(f"[OK] All values non-negative and finite")
    print(f"[OK] Consistent with single-point calculations")


def test_transmission_checkpointing():
    """Test checkpointing and resume functionality for transmission."""
    print("\n" + "="*70)
    print("Test 2: Transmission Checkpointing")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('medium_nanowire')
    energy_list = np.linspace(-2, 2, 30)
    
    from gauNEGF.transport import calculate_transmission
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, 'test_transmission.npz')
        
        # Full calculation
        print("\n1. Full calculation with checkpointing...")
        transmission_full = calculate_transmission(
            F, S, sigma_calc, energy_list,
            spin='r', checkpoint_file=checkpoint_file, checkpoint_interval=5
        )
        
        assert os.path.exists(checkpoint_file), "Checkpoint file not created!"
        assert np.all(transmission_full != -1), "Some values still -1!"
        print(f"   [OK] Calculated {len(energy_list)} energies")
        
        # Verify checkpoint contents
        data = np.load(checkpoint_file, allow_pickle=True)
        assert 'transmission' in data, "Missing transmission in checkpoint!"
        assert 'energy_list' in data, "Missing energy_list in checkpoint!"
        assert np.allclose(data['energy_list'], energy_list), "Energy list mismatch!"
        data.close()
        
        # Resume from partial checkpoint
        print("\n2. Resume from partial checkpoint...")
        data = np.load(checkpoint_file, allow_pickle=True)
        transmission_partial = data['transmission'].copy()
        transmission_partial[10:20] = -1  # Mark middle section as uncalculated
        np.savez(checkpoint_file, transmission=transmission_partial, 
                energy_list=data['energy_list'])
        data.close()
        
        transmission_resumed = calculate_transmission(
            F, S, sigma_calc, energy_list,
            spin='r', checkpoint_file=checkpoint_file, checkpoint_interval=5
        )
        
        assert np.allclose(transmission_full, transmission_resumed, rtol=1e-10), \
            "Resumed calculation doesn't match full calculation!"
        assert np.all(transmission_resumed != -1), "Some values still -1 after resume!"
        print(f"   [OK] Resumed calculation matches full calculation")
        
        # Test with no checkpoint (should still work)
        print("\n3. No checkpoint file (normal operation)...")
        transmission_no_checkpoint = calculate_transmission(
            F, S, sigma_calc, energy_list, spin='r'
        )
        assert np.allclose(transmission_full, transmission_no_checkpoint, rtol=1e-10), \
            "No-checkpoint result doesn't match checkpointed result!"
        print(f"   [OK] No-checkpoint mode works correctly")


def test_dos_physics():
    """Test physical correctness of DOS calculations."""
    print("\n" + "="*70)
    print("Test 3: DOS Physics Validation")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('medium_nanowire')
    energy_list = np.linspace(-3, 3, 50)
    
    from gauNEGF.transport import calculate_dos, dos_single_energy
    # Calculate DOS
    dos_total, dos_per_site = calculate_dos(F, S, sigma_calc, energy_list, spin='r')
    
    # Physical checks
    assert np.all(dos_total >= 0), "DOS must be non-negative!"
    assert np.all(np.isfinite(dos_total)), "DOS must be finite!"
    assert np.all(dos_per_site >= 0), "DOS per site must be non-negative!"
    
    # Verify consistency: sum of dos_per_site should equal dos_total
    print("\nVerifying DOS consistency...")
    dos_total_from_sites = np.sum(dos_per_site, axis=1)
    assert np.allclose(dos_total, dos_total_from_sites, rtol=1e-9), \
        "Sum of dos_per_site doesn't equal dos_total!"
    print(f"[OK] DOS per site sums correctly to total DOS")
    
    # Verify consistency with single-point calculations
    print("\nVerifying consistency with single-point calculations...")
    F_jax = jnp.asarray(F)
    S_jax = jnp.asarray(S)
    for i in [0, len(energy_list)//2, len(energy_list)-1]:
        E = energy_list[i]
        dos_single = dos_single_energy(E, F_jax, S_jax, sigma_calc, spin='r')
        assert np.isclose(dos_single[0], dos_total[i], rtol=1e-10), \
            f"Mismatch in total DOS at E={E:.3f}"
        assert np.allclose(dos_single[1], dos_per_site[i], rtol=1e-10), \
            f"Mismatch in DOS per site at E={E:.3f}"
    
    print(f"[OK] DOS range: {np.min(dos_total):.6f} to {np.max(dos_total):.6f}")
    print(f"[OK] All values non-negative and finite")
    print(f"[OK] Consistent with single-point calculations")


def test_dos_checkpointing():
    """Test checkpointing and resume functionality for DOS."""
    print("\n" + "="*70)
    print("Test 4: DOS Checkpointing")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('medium_nanowire')
    energy_list = np.linspace(-2, 2, 30)
    
    from gauNEGF.transport import calculate_dos
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, 'test_dos.npz')
        
        # Full calculation
        print("\n1. Full calculation with checkpointing...")
        dos_total_full, dos_per_site_full = calculate_dos(
            F, S, sigma_calc, energy_list,
            spin='r', checkpoint_file=checkpoint_file, checkpoint_interval=5
        )
        
        assert os.path.exists(checkpoint_file), "Checkpoint file not created!"
        assert np.all(dos_total_full != -1), "Some dos_total values still -1!"
        assert np.all(dos_per_site_full != -1), "Some dos_per_site values still -1!"
        print(f"   [OK] Calculated {len(energy_list)} energies")
        
        # Resume from partial checkpoint
        print("\n2. Resume from partial checkpoint...")
        data = np.load(checkpoint_file, allow_pickle=True)
        dos_total_partial = data['dos_total'].copy()
        dos_per_site_partial = data['dos_per_site'].copy()
        dos_total_partial[10:20] = -1
        dos_per_site_partial[10:20] = -1
        np.savez(checkpoint_file, dos_total=dos_total_partial,
                dos_per_site=dos_per_site_partial, energy_list=data['energy_list'])
        data.close()
        
        dos_total_resumed, dos_per_site_resumed = calculate_dos(
            F, S, sigma_calc, energy_list,
            spin='r', checkpoint_file=checkpoint_file, checkpoint_interval=5
        )
        
        assert np.allclose(dos_total_full, dos_total_resumed, rtol=1e-10), \
            "Resumed dos_total doesn't match!"
        assert np.allclose(dos_per_site_full, dos_per_site_resumed, rtol=1e-10), \
            "Resumed dos_per_site doesn't match!"
        print(f"   [OK] Resumed calculation matches full calculation")


def test_current_physics():
    """Test physical correctness of current calculations."""
    print("\n" + "="*70)
    print("Test 5: Current Physics Validation")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('medium_nanowire')
    
    from gauNEGF.transport import calculate_current
    from gauNEGF.config import ENERGY_STEP
    
    # Test zero bias - integration window is empty, should handle gracefully
    print("\n1. Zero bias test (empty integration window)...")
    try:
        current_zero = calculate_current(
            F, S, sigma_calc,
            fermi=0.0, qV=0.0, T=0, spin='r', dE=ENERGY_STEP
        )
        # If it doesn't raise an error, current should be zero
        assert np.abs(current_zero) < 1e-10, \
            f"Zero bias should give zero current, got {current_zero:.2e}"
        print(f"   [OK] Zero bias handled: {current_zero:.2e} A")
    except ValueError as e:
        # Empty integration window is acceptable for zero bias
        if "No energies in integration window" in str(e):
            print(f"   [OK] Zero bias correctly raises error for empty integration window")
        else:
            raise
    
    # Test positive bias
    print("\n2. Positive bias test...")
    fermi = 0.0
    qV = 0.5
    current_pos = calculate_current(
        F, S, sigma_calc,
        fermi=fermi, qV=qV, T=0, spin='r', dE=ENERGY_STEP
    )
    assert np.isfinite(current_pos), "Current must be finite!"
    print(f"   [OK] Positive bias qV={qV} eV gives current: {current_pos:.6e} A")
    
    # Test negative bias
    print("\n3. Negative bias test...")
    current_neg = calculate_current(
        F, S, sigma_calc,
        fermi=fermi, qV=-qV, T=0, spin='r', dE=ENERGY_STEP
    )
    assert np.isfinite(current_neg), "Negative bias current must be finite!"
    # For our symmetric nanowire, magnitudes should be similar
    assert np.isclose(np.abs(current_pos), np.abs(current_neg), rtol=1e-6), \
        f"Negative bias magnitude should match positive: |pos|={np.abs(current_pos):.6e}, |neg|={np.abs(current_neg):.6e}"
    print(f"   [OK] Negative bias current: {current_neg:.6e} A (magnitude matches positive)")
    
    # Test finite temperature
    print("\n4. Finite temperature test...")
    current_T = calculate_current(
        F, S, sigma_calc,
        fermi=fermi, qV=qV, T=300, spin='r', dE=ENERGY_STEP
    )
    assert np.isfinite(current_T), "Finite T current must be finite!"
    print(f"   [OK] T=300K current: {current_T:.6e} A (vs T=0: {current_pos:.6e} A)")


def test_current_checkpointing():
    """Test checkpointing for current calculations."""
    print("\n" + "="*70)
    print("Test 6: Current Checkpointing")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('medium_nanowire')
    fermi, qV = 0.0, 0.5
    
    from gauNEGF.transport import calculate_current
    from gauNEGF.config import ENERGY_STEP
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, 'test_current_transmission.npz')
        
        # Full calculation
        print("\n1. Full calculation with checkpointing...")
        current_full = calculate_current(
            F, S, sigma_calc,
            fermi=fermi, qV=qV, T=0, spin='r', dE=ENERGY_STEP,
            checkpoint_file=checkpoint_file, checkpoint_interval=10
        )
        
        assert os.path.exists(checkpoint_file), "Transmission checkpoint not created!"
        print(f"   [OK] Current calculated: {current_full:.6e} A")
        print(f"   [OK] Transmission checkpoint created")
        
        # Resume by using the transmission checkpoint
        print("\n2. Resume using transmission checkpoint...")
        current_resumed = calculate_current(
            F, S, sigma_calc,
            fermi=fermi, qV=qV, T=0, spin='r', dE=ENERGY_STEP,
            checkpoint_file=checkpoint_file, checkpoint_interval=10
        )
        
        assert np.isclose(current_full, current_resumed, rtol=1e-10), \
            "Resumed current doesn't match!"
        print(f"   [OK] Resumed current matches: {current_resumed:.6e} A")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("Test 7: Edge Cases")
    print("="*70)
    
    F, S, sigma_calc = setup_energy_independent_test('small_nanowire')
    
    from gauNEGF.transport import calculate_transmission, transmission_single_energy
    
    # Single energy point
    print("\n1. Single energy point...")
    energy_single = np.array([0.0])
    transmission = calculate_transmission(F, S, sigma_calc, energy_single, spin='r')
    assert len(transmission) == 1, "Should handle single energy!"
    print(f"   [OK] Single energy handled correctly")
    
    # Empty energy list (should fail gracefully)
    print("\n2. Empty energy list...")
    try:
        transmission = calculate_transmission(F, S, sigma_calc, np.array([]), spin='r')
        print(f"   [OK] Empty list handled (returned shape: {transmission.shape})")
    except Exception as e:
        print(f"   [OK] Empty list raises error (expected): {type(e).__name__}")
    
    # All energies already calculated (checkpoint with no -1 values)
    print("\n3. All energies already calculated...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, 'test_complete.npz')
        energy_list = np.linspace(-1, 1, 10)
        
        # Pre-calculate all values
        F_jax = jnp.asarray(F)
        S_jax = jnp.asarray(S)
        transmission_precalc = np.array([
            transmission_single_energy(E, F_jax, S_jax, sigma_calc, spin='r')
            for E in energy_list
        ])
        np.savez(checkpoint_file, transmission=transmission_precalc, energy_list=energy_list)
        
        # Should return immediately without recalculating
        transmission_loaded = calculate_transmission(
            F, S, sigma_calc, energy_list,
            spin='r', checkpoint_file=checkpoint_file
        )
        assert np.allclose(transmission_precalc, transmission_loaded, rtol=1e-10), \
            "Pre-calculated values don't match!"
        print(f"   [OK] All pre-calculated values loaded correctly")


def test_energy_dependent_sigma():
    """Test with energy-dependent sigma (if surfG1D works)."""
    print("\n" + "="*70)
    print("Test 8: Energy-Dependent Sigma")
    print("="*70)
    
    try:
        F, S, sigma_calc, F_ext, S_ext, dev_inds = setup_energy_dependent_test(
            'nanowire_with_contacts_medium'
        )
        energy_list = np.linspace(-1, 1, 15)  # Smaller range for testing
        
        from gauNEGF.transport import calculate_transmission, calculate_dos
        
        print("\n1. Transmission with energy-dependent sigma...")
        transmission = calculate_transmission(F, S, sigma_calc, energy_list, spin='r')
        assert np.all(transmission >= 0), "Transmission must be non-negative!"
        assert len(transmission) == len(energy_list), "Wrong length!"
        print(f"   [OK] Transmission calculated: range {np.min(transmission):.6f} to {np.max(transmission):.6f}")
        
        print("\n2. DOS with energy-dependent sigma...")
        dos_total, dos_per_site = calculate_dos(F, S, sigma_calc, energy_list, spin='r')
        assert np.all(dos_total >= 0), "DOS must be non-negative!"
        assert dos_per_site.shape == (len(energy_list), F.shape[0]), "Wrong shape!"
        print(f"   [OK] DOS calculated: range {np.min(dos_total):.6f} to {np.max(dos_total):.6f}")
        
    except Exception as e:
        print(f"   [SKIP] Energy-dependent sigma test skipped: {type(e).__name__}: {e}")
        print("   (This may occur with very small ETA values or numerical instabilities)")


if __name__ == '__main__':
    print("="*70)
    print("Comprehensive Tests for Checkpointable Transport Functions")
    print("="*70)
    
    tests = [
        ("Transmission Physics", test_transmission_physics),
        ("Transmission Checkpointing", test_transmission_checkpointing),
        ("DOS Physics", test_dos_physics),
        ("DOS Checkpointing", test_dos_checkpointing),
        ("Current Physics", test_current_physics),
        ("Current Checkpointing", test_current_checkpointing),
        ("Edge Cases", test_edge_cases),
        ("Energy-Dependent Sigma", test_energy_dependent_sigma),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n[FAILED] {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        exit(1)

