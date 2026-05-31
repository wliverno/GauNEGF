Advanced Examples
================

This section provides advanced examples demonstrating specialized features of gauNEGF.

Spin Transport
-----------

Example of spin-dependent transport:

.. code-block:: python

    from gauNEGF.scf import NEGF
    from gauNEGF.transport import calculate_transmission, SigmaCalculator
    
    # Initialize with unrestricted calculation
    negf = NEGF(
        fn='molecule',
        spin='u'  # Unrestricted calculation
    )
    
    # Set up spin-dependent contacts
    sig_up = [-0.1j]
    sig_down = [-0.05j]
    # Define contact indices explicitly (NEGF.setSigma does not return them)
    inds = [[1], [2]]
    sig1 = np.diag(sig_up*len(inds[0]) + sig_down*len(inds[0]))
    sig2 = np.diag(sig_down*len(inds[1]) + sig_up*len(inds[1]))
    negf.setSigma(lContact=inds[0], rContact=inds[1], sig=sig1, sig2=sig2)
    
    # Run at equilibrium
    negf.setVoltage(0.0)
    negf.SCF(1e-3, 0.01)
    
    # Calculate spin-resolved transmission
    sigma1, sigma2 = negf.getSigma()
    Elist = np.linspace(-5, 5, 1000)
    T, Tspin = calculate_transmission(
        negf.F, negf.S, SigmaCalculator(sigma1, sigma2), 
        Elist + negf.fermi, spin='u'
    )

Temperature Effects
---------------

Finite temperature can be set globally in the ``config.py`` file:

.. code-block:: python

    TEMPERATURE = 300

Or locally on an NEGFE object by setting the temperature argument
(NEGFE overrides setSigma/setContactBethe/setContact1D to accept the
``T`` keyword; the base NEGF class does not):

.. code-block:: python

    from gauNEGF.scfE import NEGFE

    negf = NEGFE('molecule', basis='lanl2dz')

    # Set basic temperature-dependent contact
    negf.setSigma(
        lContact=[1], rContact=[2],
        sig=-0.05j,
        T=300     # Temperature in Kelvin
    )

    # Set up temperature-dependent Bethe Lattice contacts
    negf.setContactBethe(
        contactList=[[1,2,3], [4,5,6]],
        latFile='Au',
        T=300  # Temperature in Kelvin
    )
   
Energy-Dependent Contacts
---------------------

Using realistic contact models (NEGFE provides setContactBethe and
setContact1D; both are NEGFE-only methods):

.. code-block:: python

    from gauNEGF.scfE import NEGFE

    negf = NEGFE('molContacts', basis='lanl2dz')

    # Bethe lattice contacts at atoms {1,2,3} and {6,7,8}
    negf.setContactBethe(
        contactList=[[1,2,3], [6,7,8]],
        latFile='Au', # Slater Koster parameters define in Au.bethe
        eta=1e-6      # Broadening term (eV)
    )

    # 1D chain contacts auto-extracted from the device Fock block.
    # See /guides/contacts_1d for the explicit-matrix and tau-list patterns.
    negf.setContact1D(
        contactList=[[1], [6]],
        symmetrize_contacts=True,
        eta=1e-3
    )

Custom Analysis
------------

Advanced analysis tools:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gauNEGF.transport import calculate_dos, calculate_transmission, SigmaCalculator
    
    #... run energy dependent NEGFE() calculation

    # Calculate DOS and transmission
    E = np.linspace(-5, 5, 1000)
    dos, dos_list = calculate_dos(negf.F, negf.S, SigmaCalculator(g), E)
    T = calculate_transmission(negf.F, negf.S, SigmaCalculator(negf.g), E)
    
    # Plot correlation
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(E, dos)
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS')
    
    plt.subplot(122)
    plt.semilogy(E, T)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    
    plt.tight_layout()
    plt.show()

See Also
--------

* :doc:`/guides/contacts_bethe` -- Bethe lattice contact setup, including SOC workflows
* :doc:`/guides/contacts_1d` -- full 1D chain contact setup patterns
* :doc:`/guides/workflow_recipes` -- IV sweeps, checkpointing, multi-temperature workflows

