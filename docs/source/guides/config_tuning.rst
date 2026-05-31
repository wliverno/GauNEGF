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
the Fermi energy during contact setup. This is passed to
:meth:`gauNEGF.scfE.NEGFE.setVoltage` to select how gauNEGF solves for the
electron count.

.. warning::

   For 1D contacts set up with atom indices only (no explicit tau/alpha/beta matrices), **DO NOT**  specify a fermi energy in ``setVoltage`` or the electron count will become unphysical! Instead use defaults or set ``fermiMethod`` . Example: ``negf.setVoltage(0.0)``
   

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
   * - ``'muller'`` (default)
     - High
     - Medium
     - Good general-purpose default. Overshoot unlikely, defaults to bisection upon failure.
   * - ``'poly'``
     - Medium
     - Medium-Fast
     - Recommended in production. Balances stability and speed with 3rd order polynomial fitting.
   * - ``'secant'``
     - Low
     - Fast
     - Best for highly coupled contacts, unstable for weak contacts. Defaults to bisection upon failure
   * - ``'predict'``
     - Lowest
     - Very fast
     - Stable only for weakly coupled contacts, uses previous guess upon failure (check warnings)

See :doc:`contact_choice` for guidance on which contact type is appropriate, and :doc:`contacts_1d`
for detailed setup instructions.


SCF_DAMPING
-----------

The ``SCF_DAMPING`` parameter controls how aggressively the density matrix is updated
each SCF iteration. It acts as a mixing coefficient: new density = (1 - damping) * old + damping * computed.

**Default:** 0.02

**Tuning guidelines:**

- **Recommended Range: 0.002 to 0.1**, Anything outside of that range highly unstable

- **High Mixing (0.1) for saddle points** Use if iterations seem "stuck", try with or without pulay interpolation.

- **Low Mixing (0.002) for unstable systems** Use when SCF oscillates or overshoots, increase nPulay if unstable

**Production strategy:**

For difficult systems, use a two-phase approach: run SCF with low mixing first
to find the convergence basin, then switch to higher mixing to escape any
saddle points.

.. code-block:: python

   # Phase 1: Can be unstable until contacts mixed into system
   negf.updatePulay(10)
   negf.SCF(conv=1e-3, damping=0.002, maxcycles=100)
   
   # Phase 2: Ensure not in a saddle point
   negf.updatePulay(4)
   negf.SCF(conv=1e-3, damping=0.1, maxcycles=100)


ETA (Broadening Parameter)
---------------------------

The ``ETA`` parameter controls the imaginary part of the energy (eta: E -> E + i*eta)
added to the Green's function. This broadening stabilizes the inversion and smooths
spectral features.

**Default:** 1e-5 eV

**Typical overrides:**

- **1e-3 eV:** Higher eta values can improve stability of 1D contacts especially at band edges

- **1e-5 eV (default):** Recommended for most production transport calculations. Provides sharp spectral resolution while maintaining good conditioning.

- **1e-6 eV:** Use for precise calculations to capture band edges exactly, may cause numerical issues if the Green's function is poorly conditioned.

**Trade-off:** Higher ETA speeds convergence and broadens spectral features; lower ETA gives sharper spectra but requires better conditioning. For production transport calculations, use the default (1e-5) unless specifically tuning spectral resolution.

**Pass locally:**

.. code-block:: python

   # Set ETA for 1D contact
   negf.setContact1D([[1,2,3],[4,5,6]], eta=1e-3)
   
   # Set ETA for Bethe contact
   negf.setContactBethe([[1,2,3],[4,5,6]], eta=1e-5)


TEMPERATURE
-----------

The ``TEMPERATURE`` parameter controls thermal broadening in the contacts. It does not *directly*
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



Surface Green's Function Convergence
------------------------------------

**SURFACE_GREEN_CONVERGENCE (default 1e-5):**
  Convergence threshold for the iterative inversion that builds the contact
  surface Green's function. The default is the lowest safe value: going
  **below 1e-5** can produce numerical instability in the contour integration
  and is not recommended. Going **above 1e-5** (e.g. 1e-4) speeds up contact
  setup at the cost of slightly less accurate self-energies, which is useful
  for exploratory runs or initial convergence sweeps.


Integration Tolerances
----------------------

**ADAPTIVE_INTEGRATION_TOL (default 1e-4):**
  Tolerance for adaptive energy integration in density matrix calculations. Set as a threshold
  for the maximum error (element-wise) in the density matrix.
  **NOTE**: should be *below* SCF convergence criteria


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
  - Tighten mixing to lower value: ``negf.SCF(conv=1e-3, damping=0.005)``
  - Adjust Pulay mixing value if unstable: ``negf.updatePulay(9)``
  - Check that contacts are physically reasonable (see :doc:`contact_choice`)

**Step 3: If SCF is very slow:**
  - Use two-phase approach: higher mixing (0.1) then lower (0.01)
  - Increase ETA in initial runs: ``negf.setContact1D([[1,2],[5,6]], eta=1e-3)``
  - Use closed shell spin for initial runs, then add spin degrees of freedom later

**Step 4: If Fermi search fails or gives non-physical electrons:**
  - Turn on fermi search in initial runs - fermi energy setpoint may be near orbital energies
  - Explicitly set fermiMethod in setVoltage: ``negf.setVoltage(0.0, fermiMethod='muller')``
  - Add a finite temperature to smooth out integration: ``negf.setContact1D([[1,2],[5,6]], T=300)``
  - Otherwise revert back to bisection (``fermiMethod='bisect'``) if all else fails


See Also
--------

- :doc:`contact_choice` -- Choosing between 1D and Bethe lattice contacts
- :doc:`contacts_1d` -- 1D contact setup and atom indexing
- :doc:`workflow_recipes` -- Production workflows and best practices
- :mod:`gauNEGF.config` -- Source of all default constants
