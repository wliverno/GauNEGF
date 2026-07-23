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

.. warning::

   **Contact-atom basis must be minimal.** Atoms participating in the Bethe
   contact slice must use a minimal basis (typically LANL2DZ for the heavy
   metal). Polarized or diffuse functions on contact atoms produce a
   poorly-conditioned overlap matrix that breaks the Bethe lattice surface
   Green's function construction. Interior (device) atoms can use any basis.
   See :doc:`/theory/best_practices` for the conditioning background.


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
  * ``'Au2'`` -- Orthogonalized Au FCC [111] lattice (all overlap
    Slater-Koster keys are zero). Trades some physical accuracy for
    faster, more stable SCF convergence at extended contacts (many-atom
    leads); see the crossover guidance in the skill for when to prefer
    this over ``'Au'``.
  * ``'AuSOC'`` -- Gold FCC [111] surface with spin-orbit coupling enabled

  Wrong choice leads to incorrect orbital couplings and transport results.

**eta** (float, default 1e-5 eV)
  Broadening parameter for the surface Green's function. Controls the
  numerical stability of the contour integration. 1e-5 eV is the
  recommended value for production Bethe contacts. Going to 1e-6 eV can
  sharpen spectral resolution near band edges but may cause numerical
  issues if the Green's function is poorly conditioned -- prefer 1e-5 eV
  unless you have a specific reason to go lower, and check convergence
  carefully if you do.

**T** (float, default 0 K)
  Temperature in Kelvin. Set to 300 for room-temperature transport;
  0 K gives sharp Fermi-Dirac distribution.

Important Notes on setVoltage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike 1D auto-extract contacts, Bethe lattice contacts provide their own
Fermi reference energy via ``calcFermi``. You can pass an explicit Fermi
energy (e.g. ``setVoltage(0.0, 0.0)``) as a shortcut, or omit it to let the
default ``fermiMethod='muller'`` run. See :doc:`config_tuning` for the full
method comparison.


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
   
   # First pass: constraint device charge
   negf.setVoltage(0.0)
   negf.SCF(1e-2, 0.02, 100)
   
   # Second pass: remove device charge constraint
   negf.setVoltage(0.0, negf.fermi)
   negf.SCF(1e-3, 0.02, 100)
   negf.SCF(1e-3, 0.05, 1000, pulay=False)
   
   # Save density
   negf.saveMAT('AuTipFe_ESCF_Bethe_contFermi.mat')

Key Points
~~~~~~~~~~

1. **Two setVoltage/SCF calls**: First without Fermi to ensure contacts set, second with
   ``negf.fermi`` (refine). This two-pass approach improves convergence.

2. **pulay=False**: For the first SCF pass, Pulay mixing helps to quickly get to convergence
   criteria. In open shell systems, the Pulay mixing can get stuck in a loop. Turning it off
   during second run can help with convergence if the initial run doesn't converge 


SOC Workflow (spin='g', latFile='AuSOC')
-----------------------------------------

Adding spin-orbit coupling (SOC) changes both the Hamiltonian and the orbital
dimensions. Use this path when magnetism or spin-dependent transport is
important.

Changes from Non-SOC Bethe
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **spin must be 'g'** -- Generalized spinor basis (spin='g') is required
   for SOC calculations. This doubles the orbital dimension (N -> 2N). 
   SOC couples spin and orbital angular momentum, which is not representable in
   restricted (spin='r') or unrestricted (spin='u') basis. The generalized spinor
   basis ('g') is the only formalism that captures this coupling. Attempting SOC
   with spin='r' or spin='u' will produce incorrect results.

2. **latFile='AuSOC'** -- Reads spin-orbit coupling constants (soc_p, soc_d)
   from the parameter file.


Warm-Start from Prior Run
--------------------------

For long SCF calculations (especially with SOC), save the density matrix and
Fermi energy after an initial convergence, then reload for subsequent bias
voltages or refinement passes.

Save After Initial Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initial 200 SCF cycles with loose convergence
   negf.SCF(1e-2, 0.01, 200)
   
   # Save density and Fermi for warm-start
   negf.saveMAT('NEGFSystem_loose.mat')

Warm-Start for Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy import io
   
   A = io.loadmat('NEGFSystem_loose.mat')
   negf.setDen(A['den'])
   negf.setVoltage(0.0, A['fermi'][0][0])
   # NOTE: a bare setVoltage(0.0) here would NOT re-enable the Fermi
   # search -- once fermi is pinned, self.fermi is no longer None, so
   # updFermi stays False no matter what you pass next. To re-enable the
   # search, construct a fresh NEGFE object, or set
   # negf.updFermi = True manually before the next SCF() call.

   # Refine with tighter convergence and smaller damping
   # checkpoint=False to ensure density not overwritten
   # by checkpoint file ({fn}_P.mat)
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

