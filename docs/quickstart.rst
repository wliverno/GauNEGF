Quickstart Guide
==============

This guide walks through a basic NEGF-DFT calculation using an ethane molecule as an example.

Setup
----
First, create a Gaussian input file `ethane.gjf`:

.. code-block:: text

    %chk=ethane.chk
    # b3lyp/6-31g(d,p)
    
    Ethane molecule for NEGF-DFT
    
    0 1
    C    0.000000    0.000000    0.762897
    C    0.000000    0.000000   -0.762897
    H    0.000000    1.018967    1.157832
    H    0.882443   -0.509483    1.157832
    H   -0.882443   -0.509483    1.157832
    H    0.000000   -1.018967   -1.157832
    H   -0.882443    0.509483   -1.157832
    H    0.882443    0.509483   -1.157832

Basic Calculation
--------------
Run a basic NEGF-DFT calculation:

.. code-block:: python

    from gauNEGF.scf import NEGF
    
    # Initialize calculator
    negf = NEGF(
        fn='ethane',          # Input file name
        func='b3lyp',         # DFT functional
        basis='6-31g(d,p)',   # Basis set
        spin='r'              # Restricted calculation
    )
    
    # Attach contacts to carbon atoms
    negf.setSigma([1], [2], -0.05j)
    
    # Run SCF calculation
    negf.SCF(1e-3, 0.02, 100)

Transmission Function
------------------
Calculate and plot transmission:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gauNEGF.transport import cohTrans
    
    # Energy grid
    Elist = np.linspace(-5, 5, 1000)
    
    # Calculate transmission
    T = cohTrans(Elist, negf.F, negf.S, -0.05j, -0.05j)
    
    # Plot
    plt.figure()
    plt.plot(Elist, T)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    plt.title('Ethane Transmission Function')
    plt.show()

Current Calculation
----------------
Calculate current at different voltages:

.. code-block:: python

    from gauNEGF.transport import quickCurrent
    
    # Voltage range
    V = np.arange(-0.5, 0.5, 0.1)
    
    # Calculate current
    I = []
    for v in V:
        negf.setVoltage(v)
        negf.SCF(1e-3, 0.02, 100)
        I.append(quickCurrent(negf.F, negf.S, -0.05j, -0.05j, qV=v))
    
    # Plot IV curve
    plt.figure()
    plt.plot(V, I)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('Ethane IV Characteristic')
    plt.show()

Next Steps
---------
1. Try different contact parameters or energy-dependent contacts
2. Check for current hysteresis by using a circular voltage sweep
3. Explore spin-dependent transport using open shell systems
4. Add solver parameters such as solvation models (e.g. `scrf=solvent=water`)

For more detailed examples, see the :doc:`examples/index` section.
For theoretical background, see the :doc:`theory/index` section. 
