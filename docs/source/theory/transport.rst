Transport Calculations
======================

This section covers the calculation and analysis of quantum transport properties, including transmission functions, current-voltage characteristics, and spin-dependent transport.

Transmission Function
---------------------

Theory
~~~~~~
The transmission function :math:`T(E)` gives the probability of electron transmission at energy E:

.. math::

   T(E) = \mathrm{Tr}[\Gamma_1(E) G^r(E) \Gamma_2(E) G^a(E)]

where :math:`\Gamma_{1,2}` are the contact broadening matrices.

Implementation
~~~~~~~~~~~~~~
Basic transmission calculation:

.. code-block:: python

    from gauNEGF.transport import calculate_transmission, SigmaCalculator
    import numpy as np
    
    # Energy grid
    E = np.linspace(-5, 5, 1000)
    
    # Calculate transmission (constant self-energies from an NEGF object)
    sig1, sig2 = negf.getSigma()
    F_eV = negf.F*27.211386 #Convert from hartrees to eV
    T = calculate_transmission(F_eV, negf.S, SigmaCalculator(sig1, sig2), E)

Current Calculations
--------------------

Landauer Formula
~~~~~~~~~~~~~~~~
The current is calculated using the Landauer formula:

.. math::

   I = \frac{2e}{h} \int_{-\infty}^{\infty} T(E)[f_1(E) - f_2(E)]dE

where :math:`f_{1,2}(E)` are the Fermi functions for the contacts.

Implementation
~~~~~~~~~~~~~~
Current calculation at finite bias:

.. code-block:: python

    from gauNEGF.transport import calculate_current, SigmaCalculator
    
    # Calculate current with energy-independent sigma
    F_eV = negf.F*27.211386 #Convert from hartrees to eV
    I = calculate_current(
        F_eV, negf.S, 
        SigmaCalculator(sig1, sig2),
        fermi=negf.fermi,
        qV=0.1
    )
    
    # Calculate current with energy-dependent sigma
    I = calculate_current(
        F_eV, negf.S,
        SigmaCalculator(negf.g, energy_dependent=True),
        fermi=negf.fermi,
        qV=0.1
    )

IV Characteristics
~~~~~~~~~~~~~~~~~~
Generate current-voltage curves:

.. code-block:: python

    # Equilibrium Fermi energy from a converged zero-bias run, pinned under bias
    fermi0 = negf.fermi

    # Voltage range
    V = np.arange(0, 0.5, 0.1)
    
    # Calculate IV curve
    I = []
    for v in V:
        negf.setVoltage(v, fermi0)
        negf.SCF()
        F_eV = negf.F*27.211386 #Convert from hartrees to eV
        I.append(calculate_current(
            F_eV, negf.S, 
            SigmaCalculator(sig1, sig2),
            fermi=negf.fermi,
            qV=v
        ))

Spin-Dependent Transport
------------------------

Theory
~~~~~~
Spin-dependent transport calculations account for the spin-selective transmission of electrons through molecular systems. This is particularly important for chiral molecules exhibiting the chiral-induced spin selectivity (CISS) effect [Zollner2020]_.

The spin-orbit coupling effects are included using an on-site approximation [Fernandez2006]_, which provides an efficient method for calculating relativistic effects in localized basis sets.

For spin-resolved calculations (``spin='u'``, ``'ro'`` or ``'g'``) the total
transmission is the exact trace over the full spin space,

.. math::

   T_{\text{total}}(E) = \mathrm{Tr}[\Gamma_1(E) G^r(E) \Gamma_2(E) G^a(E)],

and the four spin channels are reported alongside it:

.. math::

   T_{s s'}(E) = \mathrm{Tr}[\Gamma_{1,ss}(E) G^r_{ss'}(E) \Gamma_{2,s's'}(E) G^a_{s's}(E)],
   \qquad s, s' \in \{\uparrow, \downarrow\}

where :math:`\Gamma_{i,ss}` are the spin-diagonal blocks of the broadening
matrices. For spin-scalar leads (broadening block-diagonal in spin) the
total equals
:math:`T_{\uparrow\uparrow} + T_{\uparrow\downarrow} + T_{\downarrow\uparrow} + T_{\downarrow\downarrow}`.
For spin-mixing leads (e.g. the ``AuSOC`` Bethe lattice) the broadening
matrices have off-diagonal spin blocks, and the total differs from the sum
of the four channels; the difference is ``T - Tspin.sum(axis=1)``.
``calculate_current`` integrates the exact total in the same way.

Implementation
~~~~~~~~~~~~~~
Spin-resolved transmission:

.. code-block:: python

    from gauNEGF.transport import calculate_transmission, SigmaCalculator
    
    # Calculate spin-resolved transmission
    Elist = np.linspace(-5, 5, 1000)
    sig1, sig2 = negf.getSigma()
    T, Tspin = calculate_transmission(
        negf.F*27.211386, negf.S, 
        SigmaCalculator(sig1, sig2),
        Elist + negf.fermi,
        spin='u'  # 'u' for unrestricted
    )
    
    # Access components
    T_up_up = Tspin[:, 0]
    T_up_down = Tspin[:, 1]
    T_down_up = Tspin[:, 2]
    T_down_down = Tspin[:, 3]

Analysis Tools
--------------

Density of States
~~~~~~~~~~~~~~~~~
Calculate and analyze DOS:

.. code-block:: python

    from gauNEGF.transport import calculate_dos, SigmaCalculator
    
    # Calculate DOS
    Elist = np.linspace(-5, 5, 1000)
    dos, dos_list = calculate_dos(
        negf.F*27.211386, negf.S, 
        SigmaCalculator(sig1, sig2),
        Elist + negf.fermi
    )

Transmission Analysis
~~~~~~~~~~~~~~~~~~~~~
Analyze transmission features:

.. code-block:: python

    # Plot transmission vs energy
    plt.semilogy(Elist, T)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    
    # Find transmission peaks
    peaks = np.where(T > 0.5)[0]
    for p in peaks:
        plt.axvline(Elist[p], color='r', ls='--')

Checkpointing
~~~~~~~~~~~~~
Long-running calculations can be checkpointed to allow resuming after interruption:

.. code-block:: python

    from gauNEGF.transport import calculate_transmission, calculate_dos, calculate_current
    
    # Calculate transmission with checkpointing
    T = calculate_transmission(
        F, S, sigma_calculator, energy_list,
        checkpoint_file='transmission_checkpoint.npz',
        checkpoint_interval=50  # Save every 50 energies
    )
    
    # Resume from checkpoint (if interrupted, just run again with same parameters)
    T = calculate_transmission(
        F, S, sigma_calculator, energy_list,
        checkpoint_file='transmission_checkpoint.npz',
        checkpoint_interval=50
    )
    
    # DOS checkpointing works the same way
    dos_total, dos_per_site = calculate_dos(
        F, S, sigma_calculator, energy_list,
        checkpoint_file='dos_checkpoint.npz',
        checkpoint_interval=50
    )
    
    # Current calculations use transmission checkpointing internally.
    # Each (fermi, qV, T, dE) gives its own energy grid, so use one
    # checkpoint file per bias point: a file whose stored energy_list does
    # not match is ignored (with a warning) and the scan starts fresh.
    I = calculate_current(
        F, S, sigma_calculator,
        fermi=negf.fermi, qV=0.5,
        checkpoint_file='current_transmission_0.5V.npz',  # Stores transmission data
        checkpoint_interval=50
    )

Example Analysis
----------------

Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~
Example of a comprehensive transport analysis:

.. code-block:: python

    # Initialize system
    negf = NEGFE('molecule', basis='lanl2dz')
    negf.setContactBethe([[1,2,3], [6,7,8]], 'Au')
    har_to_eV = 27.211386

    # Run NEGF-DFT to get equilibrium density and Fermi energy
    negf.setVoltage(0.0)
    negf.SCF(1e-3, 0.02, 200)
    fermi0 = negf.fermi

    # Calculate transmission
    Elist = np.linspace(-5, 5, 1000)
    T = calculate_transmission(negf.F*har_to_eV, negf.S, SigmaCalculator(negf.g), Elist + negf.fermi)

    # Calculate DOS
    dos, _ = calculate_dos(negf.F*har_to_eV, negf.S, SigmaCalculator(negf.g), Elist + negf.fermi)

    # Generate IV curve
    V = np.linspace(0, 2, 21)
    I = []
    for v in V:
        negf.setVoltage(v, fermi0)  # pin the equilibrium Fermi energy under bias
        negf.SCF()
        I.append(calculate_current(
            negf.F*har_to_eV, negf.S,
            SigmaCalculator(negf.g),
            fermi=negf.fermi,
            qV=v
        ))
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Transmission
    ax1.semilogy(Elist, T)
    ax1.set_xlabel(r'$E - E_F$ (eV)')
    ax1.set_ylabel('Transmission')
    
    # DOS
    ax2.plot(Elist, dos)
    ax2.set_xlabel(r'$E - E_F$ (eV)')
    ax2.set_ylabel('DOS')
    
    # IV curve
    ax3.plot(V, I)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Current (A)')
    
    plt.tight_layout()
    plt.show()

Next Steps
----------
Review :doc:`best_practices` for tips on production calculations, and see
:doc:`/guides/workflow_recipes` for IV sweeps and checkpointing patterns.



.. [Fernandez2006] Fernandez-Seivane, L., Oliveira, M. A., Sanvito, S., & Ferrer, J. (2006). On-site approximation for spin-orbit coupling in linear combination of atomic orbitals density functional methods. Journal of Physics: Condensed Matter, 18(34), 7999-8013. DOI: 10.1088/0953-8984/18/34/012 
