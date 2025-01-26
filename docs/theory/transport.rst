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

Spin Transport
------------

Theory
~~~~~
For spin-dependent transport, we consider four transmission channels:

.. math::

   T_{\text{total}} = T_{\uparrow\uparrow} + T_{\uparrow\downarrow} + 
                      T_{\downarrow\uparrow} + T_{\downarrow\downarrow}

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