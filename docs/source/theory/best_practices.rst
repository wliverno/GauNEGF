Best Practices for Production Calculations
===================================

This section provides guidelines for setting up and running production-quality NEGF-DFT calculations, focusing on accuracy, efficiency, and troubleshooting.

System Preparation
----------------

Geometry Optimization
~~~~~~~~~~~~~~~~~~

1. **Initial Structure**

   * Use standard bond lengths and angles
   * Consider symmetry for efficient calculations
   * Optimize geometry without contacts first

2. **Contact Placement**

   * Choose chemically reasonable contact sites
   * Maintain consistent contact-molecule distances
   * Consider multiple contact configurations

3. **Basis Set Selection**

   * Start with a compact basis like LANL2DZ for testing
   * Use larger basis sets for production
   * Check basis set superposition error

DFT Setup
--------

Functional Selection
~~~~~~~~~~~~~~~~~
Honestly, there are hundreds of DFT functionals out there. Find one that works for your system and doesn't make the reviewers too mad. People seem to like pure functionals for metals and hybrid functionals for organics, if you want both you are SOL.

Contact Models
~~~~~~~~~~~~

1. **Energy-Independent**

   .. code-block:: python
       negf = scf.NEGF('mol')
       # Start with simple diagonal self-energies
       negf.setSigma([1], [2], -0.05j)

2. **Bethe Lattice**

   .. code-block:: python
   
       # Use realistic metallic contacts with extended system
       negf = scfE.NEGFE('molContacts')
       # Assuming triangular contacts on 1,2,3,4 and 5,6,7,8
       inds = setContactBethe([[1,2,3],[6,7,8]], latFile='Au2', eta=1e-5, T=300)

3. **1D Chain**

   .. code-block:: python
   
       # For molecular wire systems
       negf = scfE.NEGFE('molContacts')
       # Assuming repeating infinite chain extending atoms [1,2] and [3,4]
       inds = setContact1D([[2],[3]], [[1],[4]], eta=1e-5, T=300)

Convergence Strategies
-------------------

SCF Convergence
~~~~~~~~~~~~~

1. **Mixing Parameters**

   .. code-block:: python
   
       # Start with conservative mixing
       negf.SCF(damping=0.005, maxcycles=200)
       
       # Increase if convergence is slow
       negf.runSCF(mix=0.02, maxcycles=100)
       
       # Values over 0.05 will be unstable!

2. **Pulay Mixing**

   .. code-block:: python
   
       # Pulay mixing as implemented works well
       # Increase nPulay if you have cyclical convergence values
       negf = NEGF(fn='system', nPulay=9)
       # Note: increasing nPulay will have a similar effect as
       # lowering mixing in the SCF cycle

3. **Change Initial Wavefunction Guess**

   .. code-block:: python
        
       negf.setDen(density_guess)
       # Typical values for convergence, don't read checkpoint
       negf.SCF(conv=1e-4, damping=0.02, maxcycles=300, checkpoint=False)

Integration Parameters
~~~~~~~~~~~~~~~~~~

Note that integration is only used by the NEGFE() class:

1. **Automatic Integration Limits***

   .. code-block:: python
   
       # tol = DOS cutoff value for Emin and
       # limit to MaxDP for density generation
       negf.integralCheck(tol=1e-4)

2. **Manually Set Integration Limits***

   .. code-block:: python
   
       # Set grid size and Emin
       negf.setIntegrationLimits(
            N1=100, #Integration from Emin to mu
            N2=50, #Integration from Eminf to Emin
            Emin=-500
       )


3. **Add Temperature**

   .. code-block:: python
   
       # Include finite temperature (300 Kelvin) 
       # even for energy-independent contacts
       negf.setSigma([1], [2], sig=-0.05j, T=300)

Troubleshooting Guide
------------------

Common Issues
~~~~~~~~~~~

1. **SCF Convergence**

   * Decrease mixing parameter
   * Check Fermi level shifts over SCF cycles
   * Check for unreasonable geometries

2. **Transport Results**

   * Verify energy range applied
   * Start with simple energy independent contact
   * Compare with similar systems

Validation Checks
~~~~~~~~~~~~~~

1. **Zero Bias**

   * Compare with literature conductance values
   * Check Transmission profile between HOMO-LUMO gap
   * Verify DOS features (molecular orbitals vs contact effects)

2. **Finite Bias**

   * Check current symmetry with positive and negative bias
   * Check current hysteresis with increasing convergence
   * Monitor charge conservation

3. **Spin Systems**

   * Check charge and multiplicity
   * Check spin contamination
   * For non-collinear cases check system spin direction
