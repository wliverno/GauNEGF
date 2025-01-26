Ethane Molecule Tutorial
===================

This tutorial demonstrates how to perform a basic NEGF-DFT calculation using an ethane molecule as an example.

System Setup
----------

First, create the Gaussian input file ``ethane.gjf``:

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

Create a Python script ``ethane.py``:

.. code-block:: python

    from scf import NEGF
    import numpy as np
    import matplotlib.pyplot as plt
    from transport import cohTrans, quickCurrent
    
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
    negf.SCF(mix=0.02)
    
    # Calculate transmission
    E = np.linspace(-5, 5, 1000)
    T = cohTrans(E, negf.F, negf.S, -0.05j, -0.05j)
    
    # Plot transmission
    plt.figure()
    plt.semilogy(E, T)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    plt.title('Ethane Transmission Function')
    plt.show()

IV Characteristics
---------------

Add voltage calculations to your script:

.. code-block:: python

    # Voltage range
    V = np.linspace(0, 2, 21)
    
    # Calculate IV curve
    I = []
    for v in V:
        negf.setVoltage(v)
        negf.SCF()
        I.append(quickCurrent(
            negf.F, negf.S,
            -0.05j, -0.05j,
            fermi=negf.fermi,
            qV=v
        ))
    
    # Plot IV curve
    plt.figure()
    plt.plot(V, I)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('Ethane IV Characteristic')
    plt.show()

Analysis
-------

The calculation demonstrates:

1. **Transmission Function**
   - HOMO-LUMO gap
   - Resonant peaks
   - Off-resonant tunneling

2. **IV Characteristics**
   - Linear regime
   - Current saturation
   - Voltage dependence

3. **Contact Effects**
   - Level broadening
   - Contact coupling
   - Voltage drop

Next Steps
--------
Try modifying the example:

1. Change contact parameters
2. Use different basis sets
3. Try different functionals
4. Add temperature effects 