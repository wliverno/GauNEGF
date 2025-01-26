Self-Consistent NEGF-DFT
=====================

This section details the self-consistent procedure combining NEGF and DFT calculations, including density matrix construction, convergence strategies, and practical considerations.

Theory Overview
-------------

Basic Procedure
~~~~~~~~~~~~~
The self-consistent NEGF-DFT cycle:

1. **Initial Guess**

   * Start with DFT density
   * Set up contact parameters
   * Define integration grid

2. **NEGF Step**

   * Calculate Green's functions
   * Construct density matrix
   * Update chemical potentials

3. **DFT Step**

   * Generate Fock matrix
   * Update electronic structure
   * Check convergence

4. **Iterate** until convergence

Implementation Classes
-------------------

The gauNEGF package provides two main classes for NEGF-DFT calculations:

1. **NEGF Class** (``scf.py``)

   * Energy-independent self-energies
   * Constant broadening
   * Simple contact models
   * Faster calculations
   * Suitable for quick tests and initial setup

2. **NEGFE Class** (``scfE.py``)

   * Energy-dependent self-energies
   * Temperature effects
   * Advanced contact models (Bethe lattice, 1D chain)
   * More accurate results
   * Required for realistic transport calculations

Mathematical Details
-----------------

Density Matrix
~~~~~~~~~~~~
The non-equilibrium density matrix has two components:

.. math::

   P = P_{eq} + P_{neq}

where:

.. math::

   P_{eq} = -\frac{1}{\pi} \Im \int_{-\infty}^{E_F-V/2} G^r(E) dE

   P_{neq} = -\frac{1}{2\pi} \int_{E_F-V/2}^{E_F+V/2} G^r(E)\Gamma(E)G^a(E) dE

For the energy-independent case (NEGF), Γ(E) is constant. For the energy-dependent case (NEGFE), both G(E) and Γ(E) vary with energy.

Implementation
------------

Integration Methods
~~~~~~~~~~~~~~~~

Energy-Independent Case (NEGF):

.. code-block:: python

    from scf import NEGF
    
    # Initialize with constant self-energies
    negf = NEGF('molecule', basis='lanl2dz')
    negf.setSigma([1], [6])  # Simple constant self-energy
    

Energy-Dependent Case (NEGFE):

.. code-block:: python

    from scfE import NEGFE
    
    # Initialize with energy-dependent self-energies
    negf = NEGFE('molecule', basis='lanl2dz')
    negf.setContactBethe([1,2,3], [4,5,6], latFile='Au', T=300)  # Bethe lattice with temperature
    
    # Set integration parameters
    negf.setIntegralLimits(
        N1=100,     # Complex contour points
        N2=50,      # Real axis points
        Emin=-50,   # Lower bound
        T=300       # Temperature in K
    )

Convergence Acceleration
~~~~~~~~~~~~~~~~~~~~~
Density mixing strategies (applicable to both NEGF and NEGFE):

.. code-block:: python

    # Simple mixing
    negf.SCF(damping=0.02, pulay=False)
    
    # Pulay mixing
    negf.SCF(damping=0.02, pulay=True, nPulay=4)

Fermi Energy Search
~~~~~~~~~~~~~~~~
Methods for finding the Fermi energy (NEGFE only):

.. code-block:: python

    # Constant self-energy approximation
    negf.setVoltage(qV, fermiMethod='predict')
    
    # Secant method (recommended for NEGFE)
    negf.setVoltage(qV, fermiMethod='secant')
    
    # Muller method (alternative for NEGFE)
    negf.setVoltage(qV, fermiMethod='muller')

Practical Considerations
---------------------

Choosing Between NEGF and NEGFE
~~~~~~~~~~~~~~~~~~~~~~~~~~
Guidelines for selecting the appropriate class:

1. **Use NEGF when:**

   * Quick initial tests are needed
   * System is well-described by constant self-energies
   * Temperature effects are negligible
   * Performance is critical

2. **Use NEGFE when:**

   * Accurate transport properties are needed
   * Temperature effects are important
   * Realistic contact models are required
   * Energy-dependent effects are significant

Convergence Issues
~~~~~~~~~~~~~~~
Common problems and solutions:

1. **Charge Oscillations**

   * Reduce mixing parameter
   * Increase Pulay vectors
   * Check contact parameters

2. **Energy Conservation**

   * Verify integration limits
   * Check grid density
   * Monitor total energy

3. **Numerical Stability**

   * Use appropriate broadening
   * Monitor matrix condition
   * Check basis set effects

Example Workflows
--------------

Basic NEGF Calculation
~~~~~~~~~~~~~~~~~~
Quick test with energy-independent self-energies:

.. code-block:: python

    from scf import NEGF
    
    # Initialize system
    negf = NEGF('molecule', basis='lanl2dz')
    negf.setContactBethe([1,2,3], [4,5,6])
    
    # Run SCF
    negf.SCF(conv=1e-4, damping=0.05)

Production NEGFE Calculation
~~~~~~~~~~~~~~~~~~~~~~~~
Accurate calculation with temperature effects:

.. code-block:: python

    from scfE import NEGFE
    
    # Initialize system
    negf = NEGFE('molecule', basis='lanl2dz')
    negf.setContactBethe([1,2,3], [4,5,6], latFile='Au', T=300)
    
    # Set voltage and run SCF
    negf.setVoltage(0.1, fermiMethod='secant')
    negf.SCF(conv=1e-6, damping=0.02)

Next Steps
---------
Continue to :doc:`transport` for details on calculating transport properties. 