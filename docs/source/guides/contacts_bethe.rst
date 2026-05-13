=============================
Bethe Lattice Contacts
=============================

Overview: What is a Bethe Lattice Contact?
--------------------------------------------

A Bethe lattice contact is an infinite, periodic approximation to a bulk metallic
electrode. Rather than explicitly model atomic layers, the Bethe lattice uses
**Slater-Koster hopping parameters** to approximate the electronic structure of
a 3D bulk material. The approach is particularly well-suited for metallic
contacts like Au, where accurate band structure and density-of-states near the
Fermi level are essential for transport calculations.

When to Use Bethe Lattice Contacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bethe lattice contacts are ideal when you need:

* Metallic electrodes (Au, Fe, Pt, etc.) with well-characterized Slater-Koster parameters
* Accurate bulk density-of-states without modeling finite atomic layers
* Self-consistent SCF + transport calculations with minimized contact artifacts
* Optional spin-orbit coupling for systems with strong relativistic effects

If you are setting up non-metallic contacts or 1D chain electrodes, see
:doc:`contact_choice` for alternative approaches.

Reference: :class:`gauNEGF.surfGBethe.surfGB` (NEGFE integration) and
:class:`gauNEGF.surfGBethe.surfGBAt` (standalone Bethe lattice object).


The setContactBethe Call
------------------------

Syntax
~~~~~~

.. code-block:: python

   negf.setContactBethe(contactList, latFile='Au', eta=ETA, T=TEMPERATURE)

Parameters
~~~~~~~~~~

**contactList** (nested list)
  Specifies which atoms belong to each contact. Format: ``[[left_atoms], [right_atoms]]``.
  Example: ``[[1,2,3],[7,8,9]]`` creates a contact where atoms 1--3 couple to
  the left electrode and atoms 7--9 couple to the right electrode.
  Each inner list becomes a separate contact.

**latFile** (string, default='Au')
  Selects the Slater-Koster parameter file:

  * ``'Au'`` -- Gold FCC [111] surface without spin-orbit coupling
  * ``'AuSOC'`` -- Gold FCC [111] surface with spin-orbit coupling enabled

  Wrong choice leads to incorrect orbital couplings and transport results.

**eta** (float, default 1e-5 eV)
  Broadening parameter for the surface Green's function. Controls the
  numerical stability of the contour integration. Use 1e-6 eV for high
  precision or when testing convergence; 1e-5 eV is standard for production.

**T** (float, default 0 K)
  Temperature in Kelvin. Set to 300 for room-temperature transport;
  0 K gives sharp Fermi-Dirac distribution.

Important Notes on setVoltage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike 1D semi-infinite contacts, Bethe lattice contacts provide their own
Fermi reference energy via ``calcFermi``. You can call ``setVoltage`` without
specifying ``fermiMethod``:

.. code-block:: python

   negf.setVoltage(0.0)  # First pass: initial guess

However, using ``fermiMethod='poly'`` is still **strongly recommended** for
speed and robustness:

.. code-block:: python

   negf.setVoltage(0.0, fermiMethod='poly')  # Faster Fermi search


Basic Bethe Contact Workflow (spin='g', no SOC)
------------------------------------------------

This is the simplest path for Bethe lattice contacts: generalized spin ('g')
without spin-orbit coupling.

Minimal Example
~~~~~~~~~~~~~~~

.. code-block:: python

   # adapted from HemeStudies/AuFe.py
   from gauNEGF.scfE import NEGFE

   fn = 'AuTipFe'
   negf = NEGFE(fn=fn, func='b3lyp', basis='chkbasis', spin='g',
                route='integral=grid=superfine')
   negf.setContactBethe([[1,2,3],[7,8,9]], 'Au')
   
   # First pass: initial guess
   negf.setVoltage(0.0)
   
   # Second pass: refine with computed Fermi energy
   negf.setVoltage(0.0, negf.fermi)
   
   # SCF convergence
   negf.SCF(1e-3, 0.1, 1000, pulay=False)
   
   # Save density for later warm-start
   negf.saveMAT('AuTipFe_ESCF_Bethe_contFermi.mat')

Key Points
~~~~~~~~~~

1. **Two setVoltage calls**: First without Fermi (initial guess), second with
   ``negf.fermi`` (refine). This two-pass approach improves convergence.

2. **pulay=False**: For the first SCF pass with Bethe contacts, linear mixing
   (``pulay=False``) is more stable than Pulay mixing. After warm-starting
   from a saved density, ``pulay=True`` is safe.

3. **dampings**: Use damping ~ 0.1 for initial convergence. Reduce to 0.02
   for final refinement if needed.


SOC Workflow (spin='g', latFile='AuSOC')
-----------------------------------------

Adding spin-orbit coupling (SOC) changes both the Hamiltonian and the orbital
dimensions. Use this path when magnetism or spin-dependent transport is
important.

Changes from Non-SOC Bethe
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **spin must be 'g'** -- Generalized spinor basis (spin='g') is required
   for SOC calculations. This doubles the orbital dimension (N -> 2N).

2. **latFile='AuSOC'** -- Reads spin-orbit coupling constants (soc_p, soc_d)
   from the parameter file.

3. **Memory and compute**: Expect 2x memory and compute time compared to
   non-SOC ('Au') contacts.

Minimal Example
~~~~~~~~~~~~~~~

.. code-block:: python

   # adapted from AuStudies/AuBetheFerrocene.py
   from gauNEGF.scfE import NEGFE
   import numpy as np

   fn = 'AuBetheFerrocene'
   negf = NEGFE(fn=fn, func='b3lyp', basis='chkbasis', spin='g',
                fullSCF=False, route='integral=grid=superfine')
   
   # Set Bethe contacts with SOC
   negf.setContactBethe([[1,2,3,4,5,6],[36,37,38,39,40,41]], 'AuSOC')
   
   # Fermi energy: use fast polynomial fit method
   negf.setVoltage(0.0, fermiMethod='poly')
   
   # SCF with reduced damping for stability
   negf.SCF(1e-3, 0.1, 500, pulay=False)

Why spin='g' is Required
~~~~~~~~~~~~~~~~~~~~~~~~

SOC couples spin and orbital angular momentum, which is not representable in
restricted (spin='r') or unrestricted (spin='u') basis. The generalized spinor
basis ('g') is the only formalism that captures this coupling. Attempting SOC
with spin='r' or spin='u' will produce incorrect results.


Warm-Start from Prior Run
--------------------------

For long SCF calculations (especially with SOC), save the density matrix and
Fermi energy after an initial convergence, then reload for subsequent bias
voltages or refinement passes.

Save After Initial Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initial 200 SCF cycles with coarse damping
   negf.SCF(1e-3, 0.1, 200)
   
   # Save density and Fermi for warm-start
   negf.saveMAT('AuBetheFerrocene_ESCF_BetheSOC_contFermi.mat')

Warm-Start for Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # adapted from AuStudies/AuBetheFerrocene.py
   from scipy import io

   fn = 'AuBetheFerrocene'
   extra = 'ESCF_BetheSOC_contFermi'
   
   negf = NEGFE(fn=fn, func='b3lyp', basis='chkbasis', spin='g',
                fullSCF=False, route='integral=grid=superfine')
   negf.setContactBethe([[1,2,3,4,5,6],[36,37,38,39,40,41]], 'AuSOC')
   
   # Load prior density and Fermi
   A = io.loadmat(f"{fn}_{extra}.mat")
   negf.setDen(A['den'])
   negf.setVoltage(0.0, A['fermi'][0][0])
   
   # Refine with tighter convergence and smaller damping
   negf.SCF(1e-3, 0.02, 200, checkpoint=False)
   
   # Save refined result
   negf.saveMAT(f"{fn}_{extra}_refined.mat")

Why Warm-Start?
~~~~~~~~~~~~~~~

* **Speed**: Avoids re-converging from scratch; typically 3-5x faster.
* **Stability**: Starting from a near-converged density reduces oscillations.
* **IV curves**: For bias-dependent transport, warm-start from V=0 state
  then increment voltage.

For the full IV curve sweep workflow following this setup, see
:doc:`workflow_recipes`.


Standalone Contact Validation (Advanced)
-----------------------------------------

Before running a full SCF calculation, test your Bethe lattice parameters
independently. This allows you to validate the electronic structure, DOS,
and Fermi energy in isolation.

Minimal Test
~~~~~~~~~~~~

.. code-block:: python

   # adapted from tests/test_bethe_cross_term_fermi.py
   import numpy as np
   from gauNEGF.surfGBethe import surfGBAt

   # Assume H0, Slist, Vlist are pre-built (from Slater-Koster parameters)
   # Number of electrons per contact
   ne = 11  # Au has 11 valence electrons per atom
   
   # Create standalone Bethe lattice object
   eta = 1e-6
   gBAt = surfGBAt(H0, Slist, Vlist, eta=eta, T=0)
   
   # Compute Fermi energy for half-filled Bethe (spin-up)
   fermi = gBAt.calcFermi(ne / 2)
   print(f"Fermi energy: {fermi:.4f} eV")
   
   # Compute density-of-states near Fermi
   dos = gBAt.DOS(fermi)
   print(f"DOS(Fermi): {dos:.4f} states/eV")
   
   # Expected: Au bulk Fermi ~ 2.84 eV
   assert abs(fermi - 2.84) < 0.05, "Fermi energy benchmark failed"

When to Use Standalone Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Before first NEGFE run**: Validate contact parameters are sensible.
* **When changing latFile**: Ensure parameter switch (e.g., 'Au' -> 'AuSOC')
  produces expected changes in Fermi energy and DOS.
* **Debugging convergence issues**: If SCF fails, test contacts in isolation
  to rule out contact definition errors.

Reference: :class:`gauNEGF.surfGBethe.surfGBAt` (standalone constructor).


Common Pitfalls
---------------

Pitfall 1: Wrong latFile
~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Using latFile='Au' when SOC is needed (or vice versa) produces
incorrect orbital couplings and Fermi energy.

**Fix**: Always confirm latFile matches your molecular system:

* 'Au' for non-SOC Au contacts
* 'AuSOC' for Au contacts with spin-orbit coupling

Pitfall 2: Incorrect Spin Basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Attempting SOC (latFile='AuSOC') with spin='r' or spin='u'
silently produces wrong results.

**Fix**: Require spin='g' for any SOC calculation:

.. code-block:: python

   # WRONG
   negf = NEGFE(fn=fn, spin='r')
   negf.setContactBethe(contactList, 'AuSOC')  # Incompatible!
   
   # CORRECT
   negf = NEGFE(fn=fn, spin='g')
   negf.setContactBethe(contactList, 'AuSOC')

Pitfall 3: Skipping Two setVoltage Calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Calling setVoltage once without Fermi, then directly to SCF,
misses refinement with computed Fermi energy.

**Fix**: Always use two-pass approach:

.. code-block:: python

   negf.setVoltage(0.0)                 # Pass 1: initial guess
   negf.setVoltage(0.0, negf.fermi)     # Pass 2: refine with Fermi
   negf.SCF(...)

Pitfall 4: Wrong Pulay Setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Using pulay=True on first SCF pass with fresh Bethe contacts
can cause oscillations or divergence.

**Fix**: Use pulay=False for first pass, pulay=True for warm-start refinement:

.. code-block:: python

   # Fresh calculation: linear mixing
   negf.SCF(1e-3, 0.1, 500, pulay=False)
   
   # After warm-start: Pulay is safe
   negf.setDen(A['den'])
   negf.setVoltage(0.0, A['fermi'][0][0])
   negf.SCF(1e-3, 0.02, 200, pulay=True)

Pitfall 5: Memory Explosion with SOC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Doubling orbital dimension (N -> 2N) with SOC can exceed memory
on large systems.

**Fix**: Pre-test on a smaller subset of atoms, or reduce ``eta`` conservatively:

.. code-block:: python

   # Test on smaller system first
   negf = NEGFE(fn='AuSmall_test.chk', spin='g')
   negf.setContactBethe([[1,2],[5,6]], 'AuSOC')
   negf.SCF(1e-3, 0.1, 50, pulay=False)
   print(f"Memory OK. F shape: {negf.F.shape}")
