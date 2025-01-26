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
    
    # Calculate spin-resolved transmission
    T, Tspin = cohTransSpin(
        E, negf.F, negf.S,
        sig1, sig2,
        spin='u'
    )

Temperature Effects
---------------

Including finite temperature:

.. code-block:: python

    # Set up temperature-dependent contacts
    negf.setContactBethe(
        contactList=[1, 6],
        latFile='Au',
        T=300  # Temperature in Kelvin
    )
    
    # Configure integration
    negf.setIntegralLimits(
        N1=100,    # Complex contour points
        N2=50,     # Real axis points
        T=300     # Temperature
    )

Energy-Dependent Contacts
---------------------

Using realistic contact models:

.. code-block:: python

    # Bethe lattice contacts
    negf.setContactBethe(
        contactList=[1, 6],
        latFile='Au',
        eta=1e-6
    )
    
    # 1D chain contacts
    negf.setContact1D(
        contactList=[1, 6],
        taus=[2,5]
    )

Parallel Processing
---------------

If you know what you are doing and want to parallelize *each* integration point then you can manually set this in the densityComplex function:

.. code-block:: python

    from density import densityComplex
    
    # Parallel density calculation
    P = densityComplex(
        F, S, g,
        Emin=-50,
        mu=0,
        N=100,
        parallel=True,
        numWorkers=4
    )

Note this is usually slower than the default numpy parallelization.


Custom Analysis
------------

Advanced analysis tools:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from transport import DOS, cohTransE
    
    # Calculate DOS and transmission
    E = np.linspace(-5, 5, 1000)
    dos, dos_list = DOS(E, negf.F, negf.S, sig1, sig2)
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

Performance Tips
-------------

1. **Memory Management**
   - Use sparse matrices
   - Clean up temporary files
   - Monitor memory usage

2. **Convergence**
   - Start with small systems
   - Validate each step
   - Use appropriate tolerances

3. **Parallelization**
   - Choose optimal workers
   - Balance load distribution
   - Monitor scaling

Next Steps
--------
1. Develop custom contact models
2. Implement new analysis tools
3. Optimize performance
4. Add error handling 