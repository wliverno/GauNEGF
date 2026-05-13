============================
Choosing a Contact Type
============================

Overview
--------

gauNEGF offers four contact type options for modeling electron-electrode coupling:

- **Energy-independent (constant) self-energy:** Static self-energy matrices set via :meth:`gauNEGF.scf.NEGF.setSigma` or :meth:`gauNEGF.scfE.NEGFE.setSigma`. Simplest approach; energy-independent broadening matrices.
  
- **1D chain contacts:** Energy-dependent frequency-dependent self-energy for semi-infinite linear chain electrodes via :meth:`gauNEGF.scfE.NEGFE.setContact1D`. Uses :class:`gauNEGF.surfG1D.surfG` internally.
  
- **Bethe lattice contacts:** Energy-dependent self-energy for metallic FCC [111] surfaces with Slater-Koster parameters via :meth:`gauNEGF.scfE.NEGFE.setContactBethe`. Uses :class:`gauNEGF.surfGBethe.surfGB` internally.
  
- **3D atomic contacts:** Direct explicit 3D bulk k-point contact construction for advanced users via :class:`gauNEGF.surfG3D.surfGAt3D`. Provides full control; manual construction required.

Choosing Your Contact Type
---------------------------

Use the following decision table to select the contact setup method for your simulation:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Scenario
     - Contact type
     - gauNEGF entry point
   * - Quick test, benchmarking, or constant self-energy
     - Energy-independent
     - :meth:`gauNEGF.scf.NEGF.setSigma` or :meth:`gauNEGF.scfE.NEGFE.setSigma`
   * - Periodic 1D electrode (CNT, nanowire, linear chain)
     - 1D chain
     - :meth:`gauNEGF.scfE.NEGFE.setContact1D`
   * - Metallic contact with Slater-Koster parameters (Au, FCC)
     - Bethe lattice
     - :meth:`gauNEGF.scfE.NEGFE.setContactBethe`
   * - Explicit 3D bulk k-point contact (advanced)
     - 3D atomic
     - :class:`gauNEGF.surfG3D.surfGAt3D` (direct construction)

Fermi Energy and 1D Auto-Extract Contacts
-------------------------------------------

.. warning::

   When you call :meth:`gauNEGF.scfE.NEGFE.setContact1D` with only atom indices and no explicit tau/alpha/beta matrices (the "auto-extract" pattern), gauNEGF extracts coupling matrices from your DFT Fock block. In this case, you **MUST** specify a fermiMethod in the :meth:`gauNEGF.scfE.NEGFE.setVoltage` call. Without a Fermi search method, the electron count in the contact region is not physically meaningful.
   
   For Bethe lattice contacts set via :meth:`gauNEGF.scfE.NEGFE.setContactBethe`, the Bethe lattice provides its own Fermi reference. fermiMethod is recommended but not strictly required.

Correct usage:

.. code-block:: python

   # 1D auto-extract: MUST specify fermiMethod
   negf.setVoltage(0.0, fermiMethod='poly')  # 'poly' is a good default

   # Bethe lattice: fermiMethod optional (Bethe provides Fermi reference)
   negf.setContactBethe([[1,2,3],[7,8,9]], 'Au')
   negf.setVoltage(0.0)  # OK -- Bethe provides Fermi energy

For detailed information on available Fermi search methods, see :doc:`config_tuning`.

Where to Go Next
----------------

- **1D chain contacts:** :doc:`contacts_1d`
- **Bethe lattice contacts:** :doc:`contacts_bethe`
- **Configuration and tuning:** :doc:`config_tuning`
- **IV curve sweep and other recipes:** :doc:`workflow_recipes`
