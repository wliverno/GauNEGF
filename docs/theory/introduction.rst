Introduction to NEGF-DFT
=====================

This section provides an overview of the Non-Equilibrium Green's Function (NEGF) method combined with Density Functional Theory (DFT) for quantum transport calculations.

Core Functionality
---------------

The gauNEGF package provides:

* Energy-independent and energy-dependent NEGF calculations
* Multiple contact models (Bethe lattice, 1D chain)
* Transmission and current calculations
* Spin-dependent transport
* Temperature and voltage effects

Mathematical Framework
------------------

Key Equations
~~~~~~~~~~
The central quantities in NEGF-DFT are:

1. **Green's Functions**

   .. math::

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