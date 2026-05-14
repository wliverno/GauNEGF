========================================
1D Chain Contacts in gauNEGF
========================================

This guide walks through how to set up an NEGF transport calculation using
energy-dependent 1D chain contacts via the ``NEGFE`` class and ``surfG`` surface
Green's function. This method is ideal for periodic or quasi-periodic structures
where you can define semi-infinite leads as repeating 1D chains.

.. warning::

   **Contact-atom basis must be minimal.** Atoms participating in the 1D
   contact slice must use a minimal basis (e.g. STO-3G for light atoms,
   LANL2DZ for heavy atoms). Polarized or diffuse functions on contact atoms
   produce a poorly-conditioned overlap matrix between adjacent unit cells,
   which breaks the surface Green's function iteration. Interior (device)
   atoms can use any basis. See :doc:`/theory/best_practices` for the
   conditioning background.

--------
Overview
--------

The workflow has three stages:

1. Run a large DFT cluster calculation in Gaussian to get bulk-like matrices
2. Extract the Fock, overlap, and coupling matrices for the device and contacts
3. Set up the ``NEGFE`` object and run the self-consistent field (SCF) loop

--------
Prepare Gaussian Input Files
--------

You need two Gaussian input files (``.gjf``):

a) Large cluster (for matrix extraction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cluster with many unit cells (e.g., 11) so that the interior cells are
bulk-like. This is used *only* to extract initial Fock/overlap matrices and
coupling parameters. Hydrogen caps at the edges are fine -- they will be
discarded during extraction.

Example: ``CNT33_11_minbasis.gjf`` -- an 11-cell (3,3) armchair CNT with STO-3G.

b) Device cluster (for SCF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A smaller cluster matching the device region you will simulate. This file
defines the geometry that Gaussian uses during the SCF loop. It should:

* Contain exactly the atoms in your device (contact cells + interior cells)
* Use the same basis set as the large cluster
* Have NO hydrogen caps (the semi-infinite leads replace them)
* Have coordinates taken from the interior of the large cluster

Example: ``CNTPeriodic5.gjf`` -- 5 cells extracted from the interior of the
11-cell system, STO-3G, no caps.

------------------------------------
Understand the Unit Cell Structure
------------------------------------

For a periodic 1D system, identify:

* **Atoms per unit cell** (``CPerLayer``): e.g., 12 for a (3,3) armchair CNT
* **Orbitals per atom** (``Corbs``): e.g., 5 for carbon with STO-3G (1s, 2s, 2px, 2py, 2pz)
* **Orbitals per unit cell** (``orbsPerCell = Corbs * CPerLayer``): e.g., 60

The device is partitioned as:

::

    |  Left   |         Device          |  Right  |
    | Contact |  Coupling ... Coupling  | Contact |
    | 1 cell  |  (N-2) cells            | 1 cell  |

The contact cells define the repeating unit of the semi-infinite leads.
Buffer cells screen the contact self-energies from the active region.

-----------------------------------------
Extract Matrices from the Large Cluster
-----------------------------------------

.. code-block:: python

    import numpy as np
    from scipy.linalg import fractional_matrix_power
    from gauopen import QCBinAr as qcb

    har_to_eV = 27.211386
    Corbs = 5
    CPerLayer = 12
    orbsPerCell = Corbs * CPerLayer  # 60

    # Run Gaussian on the large cluster
    bar = qcb.BinAr(debug=False, lenint=8, inputfile="CNT33_11_minbasis.gjf")
    bar.update(model='b3lyp', basis='STO-3G', toutput='out.log',
               chkname="CNT33_11_minbasis.chk", dofock=True)

    S_full = np.array(bar.matlist['OVERLAP'].expand())
    P_full = np.array(bar.matlist['ALPHA SCF DENSITY MATRIX'].expand())
    F_full = np.array(bar.matlist['ALPHA FOCK MATRIX'].expand()) * har_to_eV

a) Extract the device block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the most central cells. For a 10-cell system extracting 5 cells:

.. code-block:: python

    nCells = 5
    startCell = 3  # cells 3,4,5,6,7

    ind1 = startCell * orbsPerCell            # 180
    ind2 = (startCell + nCells) * orbsPerCell  # 480

    F = F_full[ind1:ind2, ind1:ind2]
    S = S_full[ind1:ind2, ind1:ind2]

b) Count electrons
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PS_full = P_full @ S_full
    ne = np.trace(PS_full[ind1:ind2, ind1:ind2]).real

This gives the Mulliken electron count in the device block from the
full DFT calculation. Divide by ``nCells`` to get the electron count per
unit cell.

c) Extract coupling matrices from the deep interior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the two most central cells of the large cluster for the most bulk-like
coupling. These define the semi-infinite 1D chain:

.. code-block:: python

    # Interior cells 4 and 5 (deep inside the 10-cell cluster)
    ic1 = np.arange(4 * orbsPerCell, 5 * orbsPerCell)
    ic2 = np.arange(5 * orbsPerCell, 6 * orbsPerCell)

    tau_F = F_full[np.ix_(ic1, ic2)]     # inter-cell coupling (Fock)
    tau_S = S_full[np.ix_(ic1, ic2)]     # inter-cell coupling (overlap)
    alpha_F = F_full[np.ix_(ic2, ic2)]   # on-site energy (Fock)
    alpha_S = S_full[np.ix_(ic2, ic2)]   # on-site overlap

**Key definitions:**

.. list-table::
   :widths: 20 40 20
   :header-rows: 1

   * - Matrix
     - Meaning
     - Size
   * - ``alpha``
     - On-site Fock for one bulk unit cell
     - orbsPerCell x orbsPerCell
   * - ``alpha_S``
     - On-site overlap for one bulk unit cell
     - orbsPerCell x orbsPerCell
   * - ``tau``
     - Coupling Fock from cell n to cell n+1
     - orbsPerCell x orbsPerCell
   * - ``tau_S``
     - Coupling overlap from cell n to cell n+1
     - orbsPerCell x orbsPerCell
   * - ``beta``
     - Hopping between bulk cells (= tau for periodic)
     - orbsPerCell x orbsPerCell

For a periodic system, ``beta = tau`` because every inter-cell coupling is
identical.

-----------
Set Up NEGFE
-----------

.. code-block:: python

    from gauNEGF.scfE import NEGFE

    # NEGFE inherits NEGF.__init__, so standard NEGF arguments apply
    negf = NEGFE(fn='CNTPeriodic5', func='b3lyp', basis='STO-3G', fullSCF=False)
    negf.setFock(F)

* ``fn``: stem of the device ``.gjf`` file (no extension)
* ``fullSCF=False``: use Harris guess for initial DFT (faster)
* ``setFock(F)``: override the initial Fock with the one extracted from the
  large cluster

-------------------------------------
Three Usage Patterns for setContact1D
-------------------------------------

The ``setContact1D`` method supports three different calling patterns, depending
on how much information you have about the bulk electronic structure. Choose the
pattern that matches your workflow.

Pattern A: Auto-Extract from Device Fock (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this pattern when you have a device cluster extracted from a larger
DFT calculation. ``setContact1D`` reads the contact-cell and inter-cell
coupling matrices directly from the device Fock and overlap, and the contact
electron count from the device density matrix. No explicit matrices required.

.. code-block:: python

    # adapted from examples/SiNEGF.py (multi-atom contacts)
    negf = NEGFE(fn='SiNanowire', func='b3lyp', basis='STO-3G')
    negf.setFock(F)

    inds = negf.setContact1D(
        [[1, 2, 3], [4, 5, 6]],      # contactList: atoms per contact
        symmetrize_contacts=True,    # True if both contacts same material
        eta=1e-3                     # broadening (eV)
    )

    negf.setVoltage(0.0)
    negf.SCF(1e-2, 0.02, 100)

**Key arguments:**

* **contactList**: 1-indexed atom numbers within the device ``.gjf`` file.
  Left contact is first unit cell; right contact is last unit cell.
* **symmetrize_contacts**: Set ``True`` when both contacts are the same
  material (e.g., periodic nanotube, nanowire). This averages the on-site
  Fock blocks during SCF to prevent artificial symmetry breaking from
  opposite directional coupling signs in sigma_L vs sigma_R.
* **eta**: Broadening (eV). Larger values (1e-3) speed up surface Green's
  function convergence but smear sharp features. See
  :doc:`config_tuning` for ETA tuning.

For the full signature including ``tauList``, ``stauList``, ``alphas``,
``aOverlaps``, ``betas``, ``bOverlaps``, and ``neList`` (used when overriding
auto-extraction), see :meth:`gauNEGF.scfE.NEGFE.setContact1D` or Pattern B
below.

Pattern B: Custom Coupling with Auto Onsite (tauList only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this pattern when you have only the inter-cell coupling matrices and want
the on-site block computed automatically from the device Fock matrix. This is
useful when the device onsite approximation is adequate.

.. code-block:: python

    # Synthesized example: custom tau, auto alpha/beta
    negf = NEGFE(fn='CNT_device', func='b3lyp', basis='STO-3G')
    negf.setFock(F)  # F is the device Fock (rows/cols for all device atoms)

    # Only provide coupling; on-site blocks are extracted from F
    inds = negf.setContact1D(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],  # contact atom indices
        tauList=[custom_tau_left, custom_tau_right],   # explicit inter-cell coupling
        eta=1e-3
    )

    negf.setVoltage(0.0, fermiMethod='poly')
    negf.SCF(1e-3, 0.02, 1000)

.. note::

   This pattern is useful when your bulk structure matches the device
   geometry well enough that extracting alpha and beta from the device Fock
   is acceptable. If this assumption is invalid (e.g., strongly distorted
   contacts), pass the explicit matrices listed in
   :meth:`gauNEGF.scfE.NEGFE.setContact1D`.

Pattern C: Auto-Extraction from DFT (contactList + neList only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this pattern when you do NOT have pre-extracted bulk matrices and want
gauNEGF to extract contact parameters directly from Gaussian. The coupling
matrices are auto-computed from the DFT geometry stored in the device ``.gjf``.

.. code-block:: python

    # adapted from CNanowire.py (minimal specification)
    negf = NEGFE(fn='C2', func='b3lyp', basis='lanl2dz', fullSCF=False)

    # Only atoms and neList; matrices auto-extracted from Gaussian
    inds = negf.setContact1D(
        [[1], [2]],                      # contactList: atoms per contact
        neList=[ne_per_unit_cell, ne_per_unit_cell],
        eta=1e-4
    )

    # Let Fermi search run -- do NOT pass an explicit fermi energy here
    negf.setVoltage(0.0, fermiMethod='poly')
    negf.SCF(1e-2, 0.02, 100)
    negf.SCF(1e-4, 0.02, 1000, pulay=False)

.. warning::

   With Pattern C (no explicit tau/alpha/beta), **do not pass an explicit
   Fermi energy** as the second positional argument to ``setVoltage`` -- the
   electron count becomes unphysical. Let Fermi search run via the default
   ``fermiMethod='muller'`` or pick another method explicitly:

   .. code-block:: python

      negf.setVoltage(0.0)                       # uses default 'muller'
      # or
      negf.setVoltage(0.0, fermiMethod='poly')   # explicit method

   See :doc:`config_tuning` for the method comparison and selection guidance.

-----------
Run the SCF
-----------

.. code-block:: python

    # Set voltage and initial Fermi level from contact calculation
    negf.setVoltage(0.0, negf.g.fermiList[0])

    # Run SCF: tolerance, damping, max iterations
    negf.SCF(1e-3, 0.02, 1000)

    # Save results
    negf.saveMAT('output.mat')

**SCF parameters**

* **tolerance** (1e-3): convergence criterion on the density matrix change
* **damping** (0.02): linear mixing parameter -- smaller is more stable but
  slower. Range: 0.01 to 0.1.
* **max iterations** (1000): safety cap

-------------------------------------------
Post-SCF: Transmission and Current
-------------------------------------------

.. code-block:: python

    from gauNEGF.transport import cohTransE
    from scipy import io

    Elist = np.linspace(-5, 5, 500)
    T = cohTransE(Elist + negf.fermi, negf.F * har_to_eV, negf.S, negf.g)
    io.savemat('transmission.mat', {'Elist': Elist, 'fermi': negf.fermi, 'T': T})

For IV curve sweeps or multi-temperature workflows, see :doc:`workflow_recipes`.

----------------
Common Pitfalls
----------------

1. **Too few cells**: Using only 3 cells (1+1+1) gives no buffer between
   contacts; the self-energies overlap on the single device cell. Add
   interior cells until the on-site Fock for the central cell looks
   bulk-like.

2. **Edge-contaminated coupling**: Extracting tau/alpha from the edge of the
   large cluster gives non-bulk-like values. Always extract from the deep
   interior.

3. **Mismatched geometries**: The device ``.gjf`` must have coordinates taken
   from the same large cluster used for matrix extraction. Do not mix
   independently generated geometries.

4. **Wrong basis in device .gjf**: The NEGFE constructor overrides the basis
   via the ``basis`` argument, but using a checkpoint from a different basis
   can cause issues. Delete stale ``.chk`` files when changing basis sets.

5. **Forgetting .conj().T for right contact**: In a periodic system, the
   right contact coupling is the Hermitian conjugate of the left. Passing
   the same matrix for both creates an asymmetric system.

6. **H caps in device .gjf**: The device file should have bare dangling bonds
   at the edges. The semi-infinite leads replace the caps. Including H atoms
   adds spurious states.

7. **Passing an explicit Fermi energy with Pattern C**: With auto-extraction
   (Pattern C), passing the second positional argument to ``setVoltage``
   disables the Fermi search and gives an unphysical contact electron count.
   Either omit it (default ``fermiMethod='muller'`` runs) or pass only the
   ``fermiMethod=`` keyword.

8. **Forgetting symmetrize_contacts=True**: For parity-symmetric systems
   (both contacts same material), set ``symmetrize_contacts=True`` in
   ``setContact1D``. This prevents numerical artifacts from breaking
   symmetry during SCF iteration.
