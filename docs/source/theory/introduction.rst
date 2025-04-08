Introduction to NEGF-DFT
====================

This section provides an overview of the Non-Equilibrium Green's Function (NEGF) method combined with Density Functional Theory (DFT) for quantum transport calculations.

Historical Context
----------------
The gauNEGF package builds upon the foundational work of ANT.Gaussian [Palacios2002]_, which pioneered the implementation of NEGF-DFT calculations using Gaussian basis sets. This approach has proven particularly effective for molecular electronics and nanoscale transport calculations.

Core Functionality
----------------

The gauNEGF package provides:

* Energy-independent and energy-dependent NEGF calculations [Damle2002]_
* Multiple contact models (Bethe lattice [Jacob2011]_, 1D chain)
* Transmission and current calculations
* Spin-dependent transport [Zollner2020]_
* Temperature and voltage effects

Mathematical Framework
-------------------

Key Equations
-----------
The central quantities in NEGF-DFT are:

1. **Green's Functions**

   .. math::
      :label: eq_retarded_gf

      G^r(E) = [(E+i\eta)S - F - \Sigma(E)]^{-1}

   where:
   
   * :math:`G^r(E)` is the retarded Green's function
   * :math:`E` is the energy
   * :math:`\eta` is a small broadening parameter
   * :math:`S` is the overlap matrix
   * :math:`F` is the Fock matrix
   * :math:`\Sigma(E)` is the self-energy

   The lesser Green's function is given by:

   .. math::
      :label: eq_lesser_gf

      G^\lt(E) = G^r(E) [\Gamma_L f(E-\mu_L) + \Gamma_R f(E-\mu_R)] G^{r\dagger}(E)

   where:

   * :math:`G^\lt(E)` is the lesser Green's function
   * :math:`\Gamma_{L,R}` are the broadening matrices for left and right contacts
   * :math:`f(E)` is the Fermi-Dirac distribution
   * :math:`\mu_{L,R}` are the chemical potentials of the contacts

2. **Self-Energies**

   .. math::

      \Sigma(E) = \sum_i \tau_i g_{s,i}(E) \tau_i^\dagger

   where:
   
   * :math:`\tau_i` is the coupling matrix to contact i
   * :math:`g_{s,i}(E)` is the surface Green's function of contact i
   * The sum runs over all contacts

3. **Density Matrix**

   .. math::

      \rho = -\frac{1}{2\pi} \int_{-\infty}^{\infty} G^\lt(E) dE

   This gives the electron density used in the self-consistent cycle.

Contact Models and Testing
----------------------

The package includes several contact models:

1. **Bethe Lattice** (`surfGBethe.py`)

   * Ideal for metallic contacts
   * Energy-dependent self-energy
   * Realistic density of states

2. **1D Chain** (`surfG1D.py`)

   * Perfect for molecular wires
   * Periodic boundary conditions
   * Band structure effects

3. **Constant Self-Energy** (`surfGTester.py`)

   * Performance benchmarking
   * Adding Temperature dependence
   * Future development: Energy-dependent decoherence

Utility Functions (`matTools.py`)
--------------

Helper functions for common tasks:

* Matrix operations and transformations
* Integration routines
* Density and transmission calculations
* File I/O and checkpointing

Next Steps
--------
Continue to :doc:`negf_dft` for details on the self-consistent procedure.

References
----------

.. [Palacios2002] Palacios, J. J., Pérez-Jiménez, A. J., Louis, E., SanFabián, E., & Vergés, J. A. (2002). First-principles approach to electrical transport in atomic-scale nanostructures. *Physical Review B*, 66(3), 035322. https://doi.org/10.1103/PhysRevB.66.035322

.. [Damle2002] Damle, P., Ghosh, A. W., & Datta, S. (2002). First-principles analysis of molecular conduction using quantum chemistry software. *Chemical Physics*, 281(2-3), 171-187. https://doi.org/10.1016/S0301-0104(02)00496-2

.. [Jacob2011] Jacob, D., & Palacios, J. J. (2011). Critical comparison of electrode models in density functional theory based quantum transport calculations. *The Journal of Chemical Physics*, 134(4), 044118. https://doi.org/10.1063/1.3526044

.. [Zollner2020] Zöllner, M. S., Varela, S., Medina, E., Mujica, V., & Herrmann, C. (2020). Insight into the Origin of Chiral-Induced Spin Selectivity from a Symmetry Analysis of Electronic Transmission. *Journal of Chemical Theory and Computation*, 16(5), 2914-2929. https://doi.org/10.1021/acs.jctc.9b01078 
