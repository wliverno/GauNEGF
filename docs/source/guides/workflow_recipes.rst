============================
Guide 5: Workflow Recipes
============================

Once you have a converged SCF calculation (see :doc:`contacts_1d` or
:doc:`contacts_bethe`), these recipes show how to compute IV curves, study
temperature effects, and handle long energy scans efficiently.

Overview
--------

This guide covers four advanced workflows for transport calculations:

* **IV Curve Sweep** - Loop over bias voltages to measure current-voltage characteristics
* **Multi-Temperature Sweep** - Compare transport at different temperatures (e.g., 0K vs 300K)
* **Transport Checkpointing** - Save intermediate results during long energy scans and resume gracefully
* **Warm-Start SCF** - Hot-start a new SCF from a prior converged calculation

Each recipe includes working code adapted from the gauNEGF example suite.

Recipe 1: IV Curve Sweep
~~~~~~~~~~~~~~~~~~~~~~~~

Measuring current as a function of applied bias voltage (IV curves) is one of the most
common transport calculations. This recipe shows how to loop over a voltage list,
perform SCF at each voltage, and save results to MAT files for easy resume.

The key insight: if you interrupt a run, you can resume from saved checkpoint files
without recalculating previous voltages.

Code Example
^^^^^^^^^^^^

.. code-block:: python

    # adapted from AuStudies/AuBetheFerrocene.py
    from gauNEGF.scfE import NEGFE
    from gauNEGF.transport import calculate_current, SigmaCalculator
    from scipy import io
    import os
    import numpy as np

    fn = 'AuBetheFerrocene'
    extra = 'ESCF_BetheSOC_contFermi'
    spin = 'g'
    Vhi = 0.5
    dV = 0.1
    conv = 1e-3

    # Initialize NEGF object with Bethe contact
    negf = NEGFE(fn=fn, func='b3lyp', basis='chkbasis', spin=spin,
                 fullSCF=False, route='integral=grid=superfine')
    negf.setContactBethe([[1, 2, 3, 4, 5, 6],
                          [36, 37, 38, 39, 40, 41]], 'AuSOC')

    # Build voltage list: up, then down, then back to zero
    Vlist = np.concatenate((
        np.arange(dV, Vhi, dV),
        np.arange(Vhi, -1*Vhi, -1*dV),
        np.arange(-1*Vhi, 0, dV)
    ))

    # Create output directory and log file
    os.makedirs(f"{fn}_{extra}", exist_ok=True)
    f = open(f'{fn}_{extra}/IV_{Vhi}_{dV}.log', 'w')
    f.write("V (volts),I (Amps),Iuu,Iud,Idu,Idd\n")
    f.write("0.0,0.0,0.0,0.0,0.0,0.0\n")
    f.flush()

    # Loop over voltage points
    IList = np.zeros_like(Vlist)
    IsList = np.zeros((len(Vlist), 4))

    for i, V in enumerate(Vlist):
        matName = f'{fn}_{extra}/num{i}_{V:.2f}V.mat'
        # Resume from checkpoint if it exists
        if os.path.exists(matName):
            A = io.loadmat(matName)
            negf.setDen(A['den'])
            negf.setVoltage(V, A['fermi'][0][0])
        else:
            negf.setVoltage(V, negf.fermi)

        # SCF at this voltage
        negf.SCF(conv, 0.01, 1000, checkpoint=False)
        # Save density and Fock for resume
        negf.saveMAT(matName)

        # Compute current (returns total I and spin-resolved Is)
        har_to_eV = 27.211386
        I, Is = calculate_current(
            negf.F * har_to_eV, negf.S,
            SigmaCalculator(negf.g),
            negf.fermi, V, spin=spin
        )
        print(f"V = {V:.2f} V, I = {I:.3E} A")
        f.write(f"{V:.2f},{I:.3E},{Is[0]:.3E},{Is[1]:.3E},{Is[2]:.3E},{Is[3]:.3E}\n")
        f.flush()

        IList[i] = I
        IsList[i, :] = Is

    # Save results to MAT file
    io.savemat(f'{fn}_{extra}/IV_{Vhi}_{dV}.mat',
               {'Vlist': Vlist, 'IList': IList, 'IsList': IsList})
    f.close()

Key Points
^^^^^^^^^^

* **Voltage list construction**: Use ``np.arange`` to build the up, down, and back portions.
  This explores hysteresis if present.

* **Resume logic**: Before SCF, check if the MAT file exists using ``os.path.exists()``.
  If so, load the prior density and Fermi level to hot-start.

* **Current returns**: ``calculate_current()`` returns a tuple ``(I, Is)``:

  - ``I`` is the total current (scalar).
  - ``Is`` is spin-resolved current with shape (4,) for spin='g': ``[Iuu, Iud, Idu, Idd]``.
    The indices represent the four possible spin configurations in non-collinear spin.

* **Logging**: Write results to a plain-text log file for real-time monitoring.

Cross-references:

* :func:`gauNEGF.transport.calculate_current` - The main transport function.
* :class:`gauNEGF.transport.SigmaCalculator` - Wrapper for contact self-energy.
* :meth:`gauNEGF.scfE.NEGFE.setContactBethe` - Set Bethe contacts.

Recipe 2: Multi-Temperature Sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temperature affects electron transport through the Fermi-Dirac distribution in the contacts.
This recipe shows how to compute transmission (or current) at two different temperatures and compare.

Critical insight: You must run a full SCF at each temperature, not just change a parameter.
The contact self-energy is temperature-dependent, so the density matrix must re-equilibrate.

Code Example
^^^^^^^^^^^^

.. code-block:: python

    # adapted from NEGFTests/CNanowire.py
    from gauNEGF.scfE import NEGFE
    from gauNEGF.transport import cohTransE
    from scipy import io
    import numpy as np

    fn = 'C2'
    har_to_eV = 27.211386
    Elist = np.linspace(-5, 5, 1000)

    # First temperature: 0K (default)
    print("SCF at T=0K...")
    negf = NEGFE(fn=fn, func='b3lyp', basis='lanl2dz', fullSCF=False)
    negf.setContact1D([[1], [2]], eta=1e-4)  # Default T=0
    negf.setVoltage(0.0, fermiMethod='predict')
    negf.SCF(1e-2, 0.02, 100)
    negf.SCF(1e-4, 0.02, 1000, pulay=False)
    negf.saveMAT('CNanowire_ESCF.mat')

    T_0K = cohTransE(Elist + negf.fermi, negf.F * har_to_eV, negf.S, negf.g)
    io.savemat('CNanowire_TESCF_0K.mat',
               {'Elist': Elist, 'fermi': negf.fermi, 'T': T_0K})

    # Second temperature: 300K
    print("SCF at T=300K...")
    negf = NEGFE(fn=fn, func='b3lyp', basis='lanl2dz', fullSCF=False)
    negf.setContact1D([[1], [2]], eta=1e-4, T=300)  # Set T=300K
    negf.setVoltage(0.0, fermiMethod='predict')
    negf.SCF(1e-2, 0.02, 100)
    negf.SCF(1e-4, 0.02, 1000, pulay=False)
    negf.saveMAT('CNanowire_ESCF_300K.mat')

    T_300K = cohTransE(Elist + negf.fermi, negf.F * har_to_eV, negf.S, negf.g)
    io.savemat('CNanowire_TESCF_300K.mat',
               {'Elist': Elist, 'fermi': negf.fermi, 'T': T_300K})

Key Points
^^^^^^^^^^

* **Two independent SCF runs**: Each temperature gets its own NEGFE object and full SCF.
  You cannot reuse one SCF across multiple temperatures.

* **Temperature in setContact1D()**: Pass ``T=<value>`` (in Kelvin) to
  :meth:`gauNEGF.scfE.NEGFE.setContact1D`. The default is 0K (T=0).
  This sets the Fermi-Dirac distribution in the contact self-energy.

* **Fermi level**: Each temperature has its own Fermi level (stored in ``negf.fermi``).
  This is computed during SCF and reflects the electron population at that temperature.

* **Transmission comparison**: Plot ``T_0K`` vs ``T_300K`` to see how thermal broadening
  affects transport. Thermal effects typically broaden the transmission near the Fermi level.

Cross-references:

* :meth:`gauNEGF.scfE.NEGFE.setContact1D` - Set 1D contacts with temperature.
* :func:`gauNEGF.transport.cohTransE` - Compute transmission with energy-dependent sigma.

Recipe 3: Transport Checkpointing (Long Energy Scans)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long transmission or DOS scans (e.g., 5000+ energy points) can take hours or days.
This recipe shows how to use checkpointing to save intermediate results and resume
after an interruption without losing progress.

Code Example
^^^^^^^^^^^^

.. code-block:: python

    # adapted from tests/test_transport_checkpointing.py
    from gauNEGF.transport import (
        SigmaCalculator, calculate_transmission, calculate_dos
    )
    import numpy as np
    import tempfile
    import os

    # Assume you have F, S (Fock and overlap matrices)
    # and sigma_calc (SigmaCalculator object)

    energy_list = np.linspace(-5, 5, 5000)  # Long scan
    checkpoint_file = 'transmission_checkpoint.npz'
    checkpoint_interval = 100  # Save every 100 energies

    # First call: compute transmission with checkpointing
    print("Computing transmission with checkpointing...")
    transmission = calculate_transmission(
        F, S, sigma_calc, energy_list,
        spin='r',
        checkpoint_file=checkpoint_file,
        checkpoint_interval=checkpoint_interval
    )

    # If interrupted, just call again with the same arguments.
    # The function will:
    # 1. Load the checkpoint file
    # 2. Identify which energies are already computed (-1 indicates not done)
    # 3. Resume from the next missing energy
    # 4. Continue saving at the same interval

    print(f"Transmission computed: {len(transmission)} energies")

    # For DOS with checkpointing:
    dos_total, dos_per_site = calculate_dos(
        F, S, sigma_calc, energy_list,
        spin='r',
        checkpoint_file='dos_checkpoint.npz',
        checkpoint_interval=100
    )

Key Points
^^^^^^^^^^

* **checkpoint_file**: Path to an HDF5-like (.npz) file. The file is created on first call
  and reused on resume.

* **checkpoint_interval**: How many energy points to compute before saving.
  Smaller values (e.g., 10) save frequently but add I/O overhead.
  Larger values (e.g., 500) are faster but lose more progress if interrupted.
  A value of 50-100 is typical.

* **Resume behavior**: If you kill the script and restart with identical arguments,
  ``calculate_transmission()`` and ``calculate_dos()`` will:

  1. Load the saved checkpoint.
  2. Detect uncalculated energies (marked -1).
  3. Compute only those missing energies.
  4. Return the complete array.

* **No manual checkpoint management**: You do not need to manage file I/O yourself.
  The function handles loading, checking for completeness, and saving automatically.

Cross-references:

* :func:`gauNEGF.transport.calculate_transmission` - Transmission with checkpointing.
* :func:`gauNEGF.transport.calculate_dos` - DOS with checkpointing.

Recipe 4: Warm-Start SCF from Prior Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running similar geometries or exploring slight parameter variations,
starting SCF from a prior converged density matrix (warm-start) can halve
convergence time compared to starting from scratch.

Code Example
^^^^^^^^^^^^

.. code-block:: python

    # adapted from AuStudies/AuBetheFerrocene.py
    from gauNEGF.scfE import NEGFE
    from scipy import io

    # Load a prior converged result
    A = io.loadmat('AuBetheFerrocene_ESCF_BetheSOC_contFermi.mat')
    prior_density = A['den']
    prior_fermi = A['fermi'][0][0]

    # Create a new NEGF object (same or similar geometry)
    negf = NEGFE(fn='AuBetheFerrocene', func='b3lyp', basis='chkbasis',
                 spin='g', fullSCF=False)
    negf.setContactBethe([[1, 2, 3, 4, 5, 6],
                          [36, 37, 38, 39, 40, 41]], 'AuSOC')

    # Hot-start with prior density and Fermi
    negf.setDen(prior_density)  # Load density matrix
    negf.setVoltage(0.0, prior_fermi)  # Set initial Fermi level

    # SCF from warm-start (converges faster)
    negf.SCF(1e-3, 0.02, 200, checkpoint=False)

Key Points
^^^^^^^^^^

* **setDen()**: Load a prior density matrix using :meth:`gauNEGF.scf.NEGF.setDen`.
  This initializes the density in the SCF loop.

* **setVoltage() second argument**: The second positional argument to
  :meth:`gauNEGF.scf.NEGF.setVoltage` is the Fermi level (or chemical potential).
  Pass a scalar float (not a tuple).

* **MAT file keys**: Standard keys in a gauNEGF MAT file are:

  - ``'den'`` - Density matrix (ndarray)
  - ``'fermi'`` - Fermi level as a 1x1 array (access with ``[0][0]``)
  - ``'F'`` - Fock matrix
  - ``'S'`` - Overlap matrix

* **Speed improvement**: Warm-start typically converges in 10-30% of the iterations
  needed for a cold start, especially if the geometry is similar.

Cross-references:

* :meth:`gauNEGF.scf.NEGF.setDen` - Load density matrix.
* :meth:`gauNEGF.scf.NEGF.setVoltage` - Set applied voltage and Fermi level.
* :meth:`gauNEGF.scf.NEGF.saveMAT` - Save density/Fock to MAT for later resume.

Integration Methods Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recipes above use the real-axis integration method for transport (via
``calculate_current()`` and ``cohTransE()``). For a comparison with complex
contour integration and when to use each, see ``examples/IntegralDemo.ipynb`` --
run it in Jupyter on a compute node.

The demo shows:

* Accuracy vs. speed trade-offs
* How to set energy windows via ``setIntegralLimits()``
* Convergence criteria and numerical precision

Summary
-------

These four recipes cover the most common advanced workflows in gauNEGF:

1. **IV sweeps** - Fast current-voltage measurements with resume capability.
2. **Temperature studies** - Independent SCF runs at different T to study thermal effects.
3. **Checkpointing** - Long scans with automatic save/resume for robustness.
4. **Warm-start** - Fast re-convergence when starting from a prior result.

Combine these recipes as needed. For example:

* IV sweep at T=300K using checkpointing for transmission calculations.
* Multi-temperature comparison of warm-started SCF runs.
* Checkpoint very long DOS scans using energy-dependent sigma from contacts.
