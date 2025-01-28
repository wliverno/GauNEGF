Transport Calculations
====================

This section covers the calculation and analysis of quantum transport properties, including transmission functions, current-voltage characteristics, and spin-dependent transport.

Transmission Function
------------------

Theory
~~~~~
The transmission function :math:`T(E)` gives the probability of electron transmission at energy E:

.. math::

   T(E) = \mathrm{Tr}[\Gamma_1(E) G^r(E) \Gamma_2(E) G^a(E)]

where :math:`\Gamma_{1,2}` are the contact broadening matrices.

Implementation
~~~~~~~~~~~~
Basic transmission calculation:

.. code-block:: python

    from transport import cohTrans
    import numpy as np
    
    # Energy grid
    E = np.linspace(-5, 5, 1000)
    
    # Calculate transmission
    T = cohTrans(E, negf.F, negf.S, sig1, sig2)

Current Calculations
-----------------

Landauer Formula
~~~~~~~~~~~~~
The current is calculated using the Landauer formula:

.. math::

   I = \frac{2e}{h} \int_{-\infty}^{\infty} T(E)[f_1(E) - f_2(E)]dE

where :math:`f_{1,2}(E)` are the Fermi functions for the contacts.

Implementation
~~~~~~~~~~~~
Current calculation at finite bias:

.. code-block:: python

    from transport import quickCurrent
    
    # Calculate current
    I = quickCurrent(
        negf.F, negf.S,
        sig1, sig2,
        fermi=negf.fermi,
        qV=0.1
    )

IV Characteristics
~~~~~~~~~~~~~~~
Generate current-voltage curves:

.. code-block:: python

    # Voltage range
    V = np.arange(0, 0.5, 0.1)
    
    # Calculate IV curve
    I = []
    for v in V:
        negf.setVoltage(v)
        negf.SCF()
        I.append(quickCurrent(
            negf.F, negf.S,
            sig1, sig2,
            fermi=negf.fermi,
            qV=v
        ))

Spin-Dependent Transport
---------------------

Theory
~~~~~
Spin-dependent transport calculations account for the spin-selective transmission of electrons through molecular systems. This is particularly important for chiral molecules exhibiting the chiral-induced spin selectivity (CISS) effect [Zoellner2020]_.

The spin-orbit coupling effects are included using an on-site approximation [Fernandez2006]_, which provides an efficient method for calculating relativistic effects in localized basis sets.

For spin-dependent transport, we consider four transmission channels:

.. math::

   T_{\text{total}} = T_{\uparrow\uparrow} + T_{\uparrow\downarrow} + 
                      T_{\downarrow\uparrow} + T_{\downarrow\downarrow}

The spin-dependent transmission function for each channel can be calculated as:

.. math::

   T_{\sigma}(E) = \mathrm{Tr}[\Gamma_{1,\sigma}(E) G^r_{\sigma}(E) \Gamma_{2,\sigma}(E) G^a_{\sigma}(E)]

where σ denotes the spin channel.

Implementation
~~~~~~~~~~~~
Spin-resolved transmission:

.. code-block:: python

    from transport import cohTransSpin
    
    # Calculate spin-resolved transmission
    T, Tspin = cohTransSpin(
        E, negf.F, negf.S,
        sig1, sig2,
        spin='u'  # 'u' for unrestricted
    )
    
    # Access components
    T_up_up = Tspin[:, 0]
    T_up_down = Tspin[:, 1]
    T_down_up = Tspin[:, 2]
    T_down_down = Tspin[:, 3]

Analysis Tools
------------

Density of States
~~~~~~~~~~~~~
Calculate and analyze DOS:

.. code-block:: python

    from transport import DOS
    
    # Calculate DOS
    dos, dos_list = DOS(
        E, negf.F, negf.S,
        sig1, sig2
    )

Transmission Analysis
~~~~~~~~~~~~~~~~~
Analyze transmission features:

.. code-block:: python

    # Plot transmission vs energy
    plt.semilogy(E, T)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    
    # Find transmission peaks
    peaks = np.where(T > 0.5)[0]
    for p in peaks:
        plt.axvline(E[p], color='r', ls='--')

Example Analysis
-------------

Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~
Example of a comprehensive transport analysis:

.. code-block:: python

    # Initialize system
    negf = NEGF('molecule', basis='lanl2dz')
    negf.setContactBethe([1,2,3], [6,7,8], 'Au')

    # Run NEGF-DFT to get quilibrium density
    negf.setVoltage(0.0)
    negf.SCF(1e-3, 0.02, 200)
    
    # Calculate transmission
    E = np.linspace(-5, 5, 1000)
    T = cohTransE(E+negf.fermi, negf.F, negf.S, negf.g)
    
    # Calculate DOS
    dos, _ = DOSE(E+negf.fermi, negf.F, negf.S, negf.g)
    
    # Generate IV curve
    V = np.linspace(0, 2, 21)
    I = []
    for v in V:
        negf.setVoltage(v)
        negf.SCF()
        I.append(quickCurrent(
            negf.F, negf.S,
            sig1, sig2,
            fermi=negf.fermi,
            qV=v
        ))
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Transmission
    ax1.semilogy(E, T)
    ax1.set_xlabel(r'$E - E_F$ (eV)')
    ax1.set_ylabel('Transmission')
    
    # DOS
    ax2.plot(E, dos)
    ax2.set_xlabel(r'$E - E_F$ (eV)')
    ax2.set_ylabel('DOS')
    
    # IV curve
    ax3.plot(V, I)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Current (A)')
    
    plt.tight_layout()
    plt.show()

Next Steps
--------
Review :doc:`best_practices` for tips on production calculations. 

.. [Zoellner2020] Zöllner, M. S., Varela, S., Medina, E., Mujica, V., & Herrmann, C. (2020). Insight into the Origin of Chiral-Induced Spin Selectivity from a Symmetry Analysis of Electronic Transmission. Journal of Chemical Theory and Computation, 16(5), 2914-2929. DOI: 10.1021/acs.jctc.9b01078 

.. [Fernandez2006] Fernández-Seivane, L., Oliveira, M. A., Sanvito, S., & Ferrer, J. (2006). On-site approximation for spin–orbit coupling in linear combination of atomic orbitals density functional methods. Journal of Physics: Condensed Matter, 18(34), 7999-8013. DOI: 10.1088/0953-8984/18/34/012 