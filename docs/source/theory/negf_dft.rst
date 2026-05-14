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

   * Energy-independent self-energies [Damle2002]_
   * Constant broadening
   * Simple contact models
   * Faster calculations
   * Suitable for quick tests and initial setup

2. **NEGFE Class** (``scfE.py``)

   * Energy-dependent self-energies
   * Temperature effects
   * Advanced contact models (Bethe lattice [Jacob2011]_, 1D chain)
   * Not approximate
   * Longer calculations (10-100x compute)

Mathematical Details
-----------------

Density Matrix
~~~~~~~~~~~~
The non-equilibrium density matrix has two components:

.. math::

   P = P_{eq} + P_{neq}

which are given by the definite integrals (assuming T=0):

.. math::

   P_{eq} = -\frac{1}{\pi} \Im \int_{-\infty}^{E_F} G^r(E) dE

   P_{neq} = -\frac{1}{2\pi} \int_{E_F}^{E_F+V/2} G^r(E)\Gamma(E)G^a(E) dE
           + -\frac{1}{2\pi} \int_{E_F}^{E_F-V/2} G^r(E)\Gamma(E)G^a(E) dE

For the energy-independent case (NEGF), Γ(E) is constant. For the energy-dependent case (NEGFE), both G(E) and Γ(E) vary with energy.

Implementation
------------

Integration Methods
~~~~~~~~~~~~~~~~

Energy-Independent Case (NEGF):

.. code-block:: python

    from gauNEGF.scf import NEGF
    
    # Initialize with constant self-energies
    negf = NEGF('molecule', basis='lanl2dz')
    negf.setSigma(lContact=[1], rContact=[6], sig=-0.1j)  # Simple constant self-energy
    

Energy-Dependent Case (NEGFE):

.. code-block:: python

    from gauNEGF.scfE import NEGFE
    
    # Initialize with energy-dependent self-energies
    negf = NEGFE('molecule', basis='lanl2dz')
    negf.setContactBethe([1,2,3], [4,5,6], latFile='Au', T=300)  # Bethe lattice with temperature
    

Convergence Acceleration
~~~~~~~~~~~~~~~~~~~~~
Density mixing strategies (applicable to both NEGF and NEGFE):

The Pulay mixing method [Pulay1980]_ is a powerful convergence acceleration technique that uses information from previous iterations to predict the optimal density matrix. This method is particularly effective for systems with challenging convergence behavior.

.. code-block:: python

    # Simple mixing
    negf.SCF(damping=0.02, pulay=False)
    
    # Pulay mixing (DIIS)
    negf.SCF(damping=0.02, pulay=True, nPulay=4)  # Use 4 previous iterations

Fermi Energy Search
~~~~~~~~~~~~~~~~~~~
The Fermi energy is found by a root-finding search on the contact electron
count (NEGFE only). The search algorithm is selected via the ``fermiMethod``
keyword on :meth:`gauNEGF.scfE.NEGFE.setVoltage`:

.. code-block:: python

    # Default search (Muller's method, falls back to bisection)
    negf.setVoltage(qV)

    # Explicit method selection
    negf.setVoltage(qV, fermiMethod='poly')    # 3rd-order polynomial fit
    negf.setVoltage(qV, fermiMethod='secant')  # for strongly-coupled contacts
    negf.setVoltage(qV, fermiMethod='bisect')  # always-safe fallback

For 1D auto-extract contacts, do **not** pass an explicit Fermi energy to
``setVoltage``; let the search run. See :doc:`/guides/config_tuning` for the
full method comparison and selection guidance.

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

2. **Slow convergence**

   * Add broadening (eta) to surfG
   * Change fermi solver
   * Reduce system/basis size

Example Workflows
--------------

Basic NEGF Calculation
~~~~~~~~~~~~~~~~~~~~~~
Quick test with energy-independent self-energies:

.. code-block:: python

    from gauNEGF.scf import NEGF

    # Initialize system
    negf = NEGF('molContact', basis='lanl2dz')
    negf.setSigma(lContact=[1], rContact=[2], sig=-0.05j)
    negf.setVoltage(0.0)

    # Run SCF
    negf.SCF(conv=1e-4, damping=0.02)

Production NEGFE Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accurate calculation with temperature effects:

.. code-block:: python

    from gauNEGF.scfE import NEGFE

    # Initialize system
    negf = NEGFE('molecule', basis='lanl2dz')
    negf.setContactBethe([[1,2,3], [4,5,6]], latFile='Au', T=300)

    # Use defaults for fermiMethod -- see /guides/config_tuning for tuning
    negf.setVoltage(0.0)
    negf.SCF(conv=1e-4, damping=0.02)

See :doc:`/guides/contacts_bethe` and :doc:`/guides/config_tuning` for
production workflows and Fermi-search tuning.

Next Steps
---------
Continue to :doc:`transport` for details on calculating transport properties.

.. [Damle2002] Damle, P., Ghosh, A. W., & Datta, S. (2002). First-principles analysis of molecular conduction using quantum chemistry software. Chemical Physics, 281(2-3), 171-187. DOI: 10.1016/S0301-0104(02)00496-2
.. [Pulay1980] Pulay, P. (1980). Convergence acceleration of iterative sequences. The case of SCF iteration. Chemical Physics Letters, 73(2), 393-398. DOI: 10.1016/0009-2614(80)80396-4
.. [Jacob2011] Jacob, D., & Palacios, J. J. (2011). Critical comparison of electrode models in density functional theory based quantum transport calculations. The Journal of Chemical Physics, 134(4), 044118. DOI: 10.1063/1.3526044 
