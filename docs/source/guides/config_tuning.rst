============================
Configuration and Tuning
============================

Overview
--------

All gauNEGF default parameters live in :mod:`gauNEGF.config`. You can override them locally
by passing them as arguments to contact setup functions (``eta=``, ``T=``) or to SCF (``tol=``, ``damp=``).
This guide covers the parameters most likely to need tuning when optimizing convergence for your
system.

Default Configuration
---------------------

The table below lists the configuration constants that control convergence behavior.
You need not change most of them; the defaults are conservative and appropriate for
most DFT + NEGF calculations.

.. list-table:: Tunable Configuration Parameters
   :widths: 25 20 15 40
   :header-rows: 1

   * - Parameter
     - Default
     - Type
     - Purpose
   * - ``SCF_DAMPING``
     - 0.02
     - float
     - Controls density matrix update aggressiveness in SCF
   * - ``SCF_CONVERGENCE_TOL``
     - 1e-3
     - float
     - SCF convergence threshold (eV or a.u.)
   * - ``SCF_MAX_CYCLES``
     - 100
     - int
     - Maximum number of SCF iterations before stopping
   * - ``ETA``
     - 1e-5
     - float (eV)
     - Imaginary broadening for Green's function
   * - ``TEMPERATURE``
     - 0.0
     - float (K)
     - Temperature for Fermi-Dirac distribution in contacts
   * - ``FERMI_CALCULATION_TOL``
     - 1e-3
     - float
     - Tolerance for Fermi energy search
   * - ``FERMI_SEARCH_CYCLES``
     - 10
     - int
     - Maximum search cycles before returning Fermi energy
   * - ``SURFACE_GREEN_CONVERGENCE``
     - 1e-5
     - float
     - Green's function convergence tolerance for iterative inversion
   * - ``ADAPTIVE_INTEGRATION_TOL``
     - 1e-4
     - float
     - Tolerance for adaptive energy integration
   * - ``PULAY_MIXING_SIZE``
     - 4
     - int
     - Number of previous densities in Pulay mixing history


Fermi Search Method (fermiMethod)
----------------------------------

The ``fermiMethod`` parameter controls the *root-finding algorithm* used to determine
the Fermi energy during contact setup. This is passed to :meth:`setVoltage` to select
how gauNEGF solves for the electron count.

.. warning::

   For 1D contacts set up with atom indices only (no explicit tau/alpha/beta matrices),
   you **MUST** specify ``fermiMethod`` in ``setVoltage``. Without it, the electron count
   is non-physical. Example: ``negf.setVoltage(0.0, fermiMethod='poly')``
   
   For Bethe lattice contacts, ``fermiMethod`` is recommended but not strictly required.

.. list-table:: Fermi Search Algorithms
   :widths: 15 15 15 55
   :header-rows: 1

   * - Method
     - Stability
     - Speed
     - When to use
   * - ``'bisect'``
     - Highest
     - Slow
     - Always safe; default fallback when others fail. Guaranteed convergence but requires many iterations.
   * - ``'muller'``
     - High
     - Medium
     - Good general-purpose default. Quadratically convergent, low failure rate.
   * - ``'poly'``
     - Medium
     - Medium
     - Recommended in production. Balances stability and speed with polynomial fitting.
   * - ``'secant'``
     - Low
     - Fast
     - Use only on well-conditioned problems with a good initial guess. High failure rate on challenging systems.
   * - ``'predict'``
     - Lowest
     - Very fast
     - Advanced use; unstable for most systems. Use only when you have empirical evidence it works for your geometry.

See :doc:`contact_choice` for guidance on which contact type is appropriate, and :doc:`contacts_1d`
for detailed setup instructions.


SCF_DAMPING
-----------

The ``SCF_DAMPING`` parameter controls how aggressively the density matrix is updated
each SCF iteration. It acts as a mixing coefficient: new density = (1 - damping) * old + damping * computed.

**Default:** 0.02

**Tuning guidelines:**

- **0.02 (default):** Good starting point for most systems. Provides stable convergence without excessive oscillation.

- **0.01 (tighter damping):** Use when SCF oscillates or overshoots. Slower but more stable; particularly useful for charged systems or systems with small band gaps.

- **0.1 (looser damping):** Use when SCF is very slow to converge. Looser damping finds the convergence basin faster but may miss fine structure. After convergence, rerun with tighter damping (0.01-0.02) for final accuracy.

**Production strategy:**

For difficult systems, use a two-phase approach: run SCF with loose damping first to find
the convergence basin, then switch to tight damping for final refinement.

.. code-block:: python

   # Phase 1: Loose damping to find basin (coarse search)
   negf.SCF(conv=1e-3, damping=0.1, maxcycles=200)
   
   # Phase 2: Tight damping to converge (fine refinement)
   negf.SCF(conv=1e-3, damping=0.01, maxcycles=1000)


ETA (Broadening Parameter)
---------------------------

The ``ETA`` parameter controls the imaginary part of the energy (eta: E -> E + i*eta)
added to the Green's function. This broadening stabilizes the inversion and smooths
spectral features.

**Default:** 1e-5 eV

**Typical overrides:**

- **1e-4 eV:** Standard DFT workflows with 1D contacts. Slight broadening improves numerical stability for wide energy ranges (e.g., computing transmission over >10 eV).

- **1e-5 eV (default):** Recommended for Bethe lattice 3D contacts and most production transport calculations. Provides sharp spectral resolution while maintaining good conditioning.

- **1e-6 eV:** Very precise calculations when you have converged basis sets and tight SCF thresholds. Use with caution; may cause numerical issues if the Green's function is poorly conditioned.

**Trade-off:** Higher ETA speeds convergence and broadens spectral features; lower ETA gives sharper spectra but requires better conditioning. For production transport calculations, use the default (1e-5) unless specifically tuning spectral resolution.

**Pass locally:**

.. code-block:: python

   # Set ETA for 1D contact
   negf.setContact1D([[1,2,3],[4,5,6]], eta=1e-4)
   
   # Set ETA for Bethe contact
   negf.setContactBethe([[1,2,3],[4,5,6]], eta=1e-5)


TEMPERATURE
-----------

The ``TEMPERATURE`` parameter controls thermal broadening in the contacts. It does *not*
affect the device Hamiltonian, only the Fermi-Dirac distribution in the contact self-energy.

**Default:** 0.0 K (zero temperature, sharp Fermi cutoff)

**Common values:**

- **0.0 K (default):** Zero-temperature transport. Sharp electron/hole cutoff at the Fermi energy. Appropriate for single-point transmission calculations and most DFT+NEGF workflows.

- **300 K:** Room temperature. Adds thermal broadening (kT ~ 0.026 eV); electrons and holes leak slightly across the Fermi energy. Use for finite-temperature transport or to model thermal effects.

**Pass locally:**

.. code-block:: python

   # Set temperature for 1D contact
   negf.setContact1D([[1,2,3],[4,5,6]], T=300)
   
   # Set temperature for Bethe contact
   negf.setContactBethe([[1,2,3],[4,5,6]], T=300)


Fermi Energy Search Tolerances
-------------------------------

Two tolerances control Fermi energy accuracy:

**FERMI_CALCULATION_TOL (default 1e-3):**
  Relative tolerance for the electron count during Fermi search. Convergence is
  achieved when |N_computed - N_target| < FERMI_CALCULATION_TOL. Loosen to 1e-2
  if Fermi search oscillates; tighten to 1e-4 for strict charge neutrality.

**FERMI_SEARCH_CYCLES (default 10):**
  Maximum number of root-finding iterations before returning. If the solver does not
  converge within this limit, gauNEGF returns the best estimate and logs a warning.
  Increase to 20-30 for difficult systems; decrease to 5 if Fermi search is a bottleneck.


Green's Function Convergence
-----------------------------

**SURFACE_GREEN_CONVERGENCE (default 1e-5):**
  Convergence threshold for the iterative inversion of the surface Green's function
  (used in :meth:`surfG.g` and related methods). Controls accuracy of contact self-energies.

  - **1e-5 (default):** Balanced choice for most calculations.
  - **1e-4:** Faster but less accurate; acceptable for exploratory studies.
  - **1e-6:** Higher accuracy; use if you observe non-physical transmission features.


Integration Tolerances
----------------------

**ADAPTIVE_INTEGRATION_TOL (default 1e-4):**
  Tolerance for adaptive energy integration (e.g., in density and current calculations).
  Controls automatic refinement of the energy grid. Loosen to 1e-3 for speed; tighten
  to 1e-5 for high-resolution spectroscopy.


Integration Methods: Complex Contour vs Real-Axis
--------------------------------------------------

For hands-on comparison of gauNEGF integration approaches, see
``examples/IntegralDemo.ipynb`` -- a step-by-step notebook designed to run on a
compute node and illustrate the trade-offs between complex contour integration and
real-axis energy integration.


Workflow: Tuning Convergence
-----------------------------

**Step 1: Start with defaults.**
  Most systems converge with factory settings. Run SCF and check the convergence history.

**Step 2: If SCF oscillates or diverges:**
  - Tighten SCF_DAMPING to 0.01
  - Enable Pulay mixing: ``negf.SCF(conv=1e-3, damping=0.02, pulay=True)``
  - Check that contacts are physically reasonable (see :doc:`contact_choice`)

**Step 3: If SCF is very slow:**
  - Use two-phase approach: loose damping (0.1) then tight (0.01)
  - Increase SCF_MAX_CYCLES to 200-500
  - Reduce SCF_CONVERGENCE_TOL to 1e-2 for initial runs

**Step 4: If Fermi search fails or gives non-physical electrons:**
  - Explicitly set fermiMethod in setVoltage: ``negf.setVoltage(0.0, fermiMethod='muller')``
  - If muller fails, fall back to 'bisect'
  - Check FERMI_CALCULATION_TOL and FERMI_SEARCH_CYCLES

**Step 5: If transmission or DOS has spurious features:**
  - Reduce ETA from 1e-5 to 1e-6 (sharper spectra)
  - Or increase SURFACE_GREEN_CONVERGENCE to 1e-6 (tighter contact self-energy)
  - Tighten ADAPTIVE_INTEGRATION_TOL to 1e-5


See Also
--------

- :doc:`contact_choice` -- Choosing between 1D and Bethe lattice contacts
- :doc:`contacts_1d` -- 1D contact setup and atom indexing
- :doc:`workflow_recipes` -- Production workflows and best practices
- :mod:`gauNEGF.config` -- Source of all default constants
