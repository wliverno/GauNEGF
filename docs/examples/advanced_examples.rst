Advanced Examples
================

This section provides advanced examples demonstrating specialized features of gauNEGF.

Spin Transport
-----------

Example of spin-dependent transport:

.. code-block:: python

    from gauNEGF.scf import NEGF
    from gauNEGF.transport import cohTransSpin
    
    # Initialize with unrestricted calculation
    negf = NEGF(
        fn='molecule',
        spin='u'  # Unrestricted calculation
    )
    
    # Set up spin-dependent contacts
    sig_up = [-0.1j]
    sig_down = [-0.05j]
    inds = negf.setSigma([1], [2])
    sig1 = np.diag(sig_up*len(inds[0]) + sig_down*len(inds[0]))
    sig2 = np.diag(sig_down*len(inds[1]) + sig_up*len(inds[1]))
    negf.setSigma([1], [2], sig1, sig2)
    
    # Run at equilibrium
    negf.setVoltage(0.0)
    negf.SCF(1e-3, 0.01)
    
    # Calculate spin-resolved transmission
    sigma1, sigma1 = negf.getSigma()
    T, Tspin = cohTransSpin(
        E, negf.F, negf.S,
        sigma1, sigma2,
        spin='u'
    )

Temperature Effects
---------------

Including finite temperature:

.. code-block:: python
 
    # Set basic temperature-dependent contact
    negf.setSigma(
        [1], [2],  
        -0.05j, 
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

Using realistic contact models:

.. code-block:: python

    # Bethe lattice contacts at atoms {1,2,3} and {6,7,8}
    negf.setContactBethe(
        contactList=[[1,2,3], [6,7,8]],
        latFile='Au', # Slater coster parameters define in Au.bethe
        eta=1e-6      # Broadening term (eV)
    )
    
    # 1D chain contacts attached to atoms 1 and 6
    negf.setContact1D(
        contactList= [[1],[6]],
        tauList = [[2], [5]],   # hopping calculated from 1 to 2 and 6 to 5
        neList = [4,  4],       # 4 electrons per cell
        eta = 1e-6              # Broadening term (eV)
    )

Parallel Processing
---------------

If you know what you are doing and want to parallelize *each* integration point then you can manually set this in the densityComplex function:

.. code-block:: python

    from gauNEGF.density import densityComplex
    
    # Parallel density calculation
    P = densityComplex(
        F, S, g,
        Emin=-50,
        mu=0,
        N=100,
        parallel=True,
        numWorkers=4
    )

Note this is usually slower than the default numpy parallelization. *Only proceed if you know what you are doing*

Custom Analysis
------------

Advanced analysis tools:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gauNEGF.transport import DOS, cohTransE
    
    #... run energy dependent NEGFE() calculation

    # Calculate DOS and transmission
    E = np.linspace(-5, 5, 1000)
    dos, dos_list = DOSE(E, negf.F, negf.S, g)
    T = cohTransE(E, negf.F, negf.S, negf.g)
    
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

Next Steps
--------
1. Develop custom contact models
2. Implement new analysis tools
3. Optimize performance
4. Add error handling 
