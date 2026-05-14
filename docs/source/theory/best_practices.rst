Best Practices for Production Calculations
==========================================

This page collects practical advice that is not covered by the task-oriented
guides. For step-by-step setup of contacts, SCF tuning, and standard workflows,
see :doc:`/guides/index`.

.. seealso::

   :doc:`/guides/contact_choice` -- choosing between energy-independent self-energies, 1D chain, and Bethe lattice contacts.

   :doc:`/guides/config_tuning` -- SCF damping, Pulay mixing, ETA, fermiMethod, and integration tolerances.

   :doc:`/guides/workflow_recipes` -- warm-starts, IV sweeps, and checkpointing.


System Preparation
------------------

Contact Placement
~~~~~~~~~~~~~~~~~

* Choose chemically reasonable contact sites (e.g. thiol on Au, terminating
  Si atom for a nanowire).
* Maintain consistent contact-molecule distances when comparing geometries.
* Consider multiple contact configurations if conductance is sensitive to
  binding site.

Basis Set Selection
~~~~~~~~~~~~~~~~~~~

* For light-element systems, a compact all-electron basis such as STO-3G or
  6-31G is fine for testing and quick convergence checks.
* **Boundary atoms must use a minimal basis** - This applies to 1D or 3D 
  contacts, due to poor conditioning of the overlap matrix (to be fixed
  in future versions with regularization methods)
* For heavy elements (Au, Pt, transition metals), use an ECP basis such as
  LANL2DZ to keep the orbital count tractable.
* Move to a polarized basis (6-31G(d,p), def2-SVP) for production transport.


DFT Setup
---------

Functional Selection
~~~~~~~~~~~~~~~~~~~~

From a pure physics perspective, using a functional with exchange will never
be accurate: the charges from any atom outside of the device boundary cannot
possibly be included in the two-electron integrals. However, in practice, 
hybrid functionals like B3LYP have produced the best results for isolated
molecular systems. Therefore, benchmarking your results against multiple
functionals is always recommended: find one that works relatively well
for your system and doesn't make the reviewers too mad.


Integration Parameters
----------------------

Integration is used only by the energy-dependent class
(:class:`gauNEGF.scfE.NEGFE`). The default adaptive integrator is correct for
most cases; see :doc:`/guides/config_tuning` for the ``ADAPTIVE_INTEGRATION_TOL``
constraint.

Manually Setting the Integration Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For cases where the adaptive integrator is too slow or unstable, you can fix
the grid directly:

.. code-block:: python

    negf.setIntegralLimits(
         N1=100,     # integration points from Emin to mu
         N2=50,      # integration points from Eminf to Emin
         Emin=-500   # lower bound (eV)
    )

This bypasses adaptive refinement entirely; use only after you have a working
adaptive run for comparison.

Adding Temperature
~~~~~~~~~~~~~~~~~~

Temperature affects the contact Fermi-Dirac distribution. It is set per-contact
on the energy-dependent class:

.. code-block:: python

    from gauNEGF.scfE import NEGFE

    negf = NEGFE('molecule', basis='lanl2dz')
    negf.setSigma(lContact=[1], rContact=[2], sig=-0.05j, T=300)

See :doc:`/guides/config_tuning` for the global ``TEMPERATURE`` config constant.


Validation Checks
-----------------

Zero Bias
~~~~~~~~~

* Compare with literature conductance values for similar systems.
* Check the transmission profile inside the HOMO-LUMO gap (should be small).
* Verify DOS features: molecular orbitals should be sharp, contact-induced
  broadening should be smooth.

Finite Bias
~~~~~~~~~~~

* Check current symmetry under positive and negative bias.
* Check for hysteresis by sweeping voltage forward and backward.
* Monitor charge conservation across the device region.

Spin Systems
~~~~~~~~~~~~

* Verify charge and multiplicity match the intended state.
* Check for spin contamination (``<S^2>``) in unrestricted runs.
* For non-collinear (``spin='g'``) cases, inspect the system spin direction
  after SCF convergence.
