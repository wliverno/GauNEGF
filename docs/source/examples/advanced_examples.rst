Advanced Examples
=================

This section provides advanced examples demonstrating specialized features of gauNEGF.

Spin Transport
--------------

Example of spin-dependent transport:

.. code-block:: python

    import numpy as np
    from gauNEGF.scf import NEGF
    from gauNEGF.transport import calculate_transmission, SigmaCalculator

    har_to_eV = 27.211386

    # Initialize with unrestricted calculation
    negf = NEGF(
        fn='molecule',
        spin='u'  # Unrestricted calculation
    )

    # Orbital indices of the contact atoms; for spin='u' the alpha orbitals
    # come first, then the beta orbitals
    lInd, rInd = negf.setContacts([1], [2])
    nL, nR = len(lInd)//2, len(rInd)//2

    # Spin-dependent contacts: one diagonal matrix per contact
    sig1 = np.diag([-0.1j]*nL + [-0.05j]*nL)    # left: up -0.1j, down -0.05j
    sig2 = np.diag([-0.05j]*nR + [-0.1j]*nR)    # right: up -0.05j, down -0.1j
    negf.setSigma(lContact=[1], rContact=[2], sig=sig1, sig2=sig2)

    # Run at equilibrium
    negf.setVoltage(0.0)
    negf.SCF(1e-3, 0.01)

    # Calculate spin-resolved transmission (F is stored in hartree)
    sigma1, sigma2 = negf.getSigma()
    Elist = np.linspace(-5, 5, 1000)
    T, Tspin = calculate_transmission(
        negf.F*har_to_eV, negf.S, SigmaCalculator(sigma1, sigma2),
        Elist + negf.fermi, spin='u'
    )

Temperature Effects
-------------------

Finite temperature can be set globally through :mod:`gauNEGF.config`,
before the other gauNEGF modules are imported (they bind the default at
import time):

.. code-block:: python

    from gauNEGF import config
    config.TEMPERATURE = 300

    from gauNEGF.scfE import NEGFE  # import after changing the default

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
-------------------------

Using realistic contact models (NEGFE provides setContactBethe and
setContact1D; both are NEGFE-only methods):

.. code-block:: python

    from gauNEGF.scfE import NEGFE

    negf = NEGFE('molContacts', basis='lanl2dz')

    # Either: Bethe lattice contacts at atoms {1,2,3} and {6,7,8}
    negf.setContactBethe(
        contactList=[[1,2,3], [6,7,8]],
        latFile='Au', # Slater-Koster parameters defined in Au.bethe
        eta=1e-6      # Broadening term (eV)
    )

    # Or: 1D chain contacts auto-extracted from the device Fock block
    # (a second contact call replaces the first).
    # See /guides/contacts_1d for the explicit-matrix and tau-list patterns.
    negf.setContact1D(
        contactList=[[1], [6]],
        symmetrize_contacts=True,
        eta=1e-3
    )

Custom Analysis
---------------

Advanced analysis tools:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gauNEGF.transport import calculate_dos, calculate_transmission, SigmaCalculator
    
    #... run energy dependent NEGFE() calculation

    # Calculate DOS and transmission (F is stored in hartree; pass eV)
    har_to_eV = 27.211386
    F_eV = negf.F*har_to_eV
    E = np.linspace(-5, 5, 1000) + negf.fermi
    dos, dos_list = calculate_dos(F_eV, negf.S, SigmaCalculator(negf.g), E)
    T = calculate_transmission(F_eV, negf.S, SigmaCalculator(negf.g), E)
    
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

