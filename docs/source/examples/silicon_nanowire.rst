Silicon Nanowire Tutorial
====================

This tutorial demonstrates transport calculations through a silicon nanowire using two different approaches: with and without self-consistent field calculations.

Part 1: Transport Without SCF
-------------------------

This approach uses a long chain (12 Si atoms) with input file given by 

.. code-block:: text

    %chk=SiNanowire12.chk
    #p b3lyp/lanl2dz

    Title Card Required

    0 1
    Si                0.00000000    0.00000000    0.00000000
    Si                3.00000000    0.00000000    0.00000000
    Si                6.00000000    0.00000000    0.00000000
    Si                9.00000000    0.00000000    0.00000000
    Si               12.00000000    0.00000000    0.00000000
    Si               15.00000000    0.00000000    0.00000000
    Si               18.00000000    0.00000000    0.00000000
    Si               21.00000000    0.00000000    0.00000000
    Si               24.00000000    0.00000000    0.00000000
    Si               27.00000000    0.00000000    0.00000000
    Si               30.00000000    0.00000000    0.00000000
    Si               33.00000000    0.00000000    0.00000000

This is used to approximate an infinite chain by cutting out the middle 2 Si atoms:

.. code-block:: python

    from gauNEGF.surfG1D import surfG
    from gauNEGF.density import *
    from gauNEGF.transport import *
    from gauNEGF.utils import fractional_matrix_power

    har_to_eV = 27.211386

    # Run DFT calculation using SiNanowire12.gjf input file:
    bar = qcb.BinAr(debug=False,lenint=8,inputfile="SiNanowire12.gjf")
    bar.update(model='b3lyp', basis='lanl2dz', toutput='out.log',dofock="scf")

    # Collect matrices from Gaussian, generate orthogonal H matrix
    S = np.array(bar.matlist['OVERLAP'].expand())
    P = np.array(bar.matlist['ALPHA SCF DENSITY MATRIX'].expand())
    F = np.array(bar.matlist['ALPHA FOCK MATRIX'].expand())*har_to_eV
    X = fractional_matrix_power(S, -0.5)
    H = np.real(X@F@X)

    # Cut out middle 2 Si atoms to use for generation of infinite chain
    contactInds = np.arange(0, 8)
    onsiteInds = np.arange(8, 16)
    PS = P@S
    ne = np.trace(PS[40:56, 40:56]).real
    F = F[40:56, 40:56]
    S = S[40:56, 40:56]
    H = H[40:56, 40:56]

    # Transport calculations for non-orthogonal case
    print('Coherent transport for non-orth case')
    g = surfG(F, S, [contactInds, onsiteInds], eta=1e-4) #Added broadening to speed up convergence
    fermi = getFermiContact(g, ne)
    Elist = np.linspace(-5, 5, 1000)
    T = calculate_transmission(F, S, SigmaCalculator(g), Elist+fermi)

    # Transport calculations for non-orthogonal case
    print('Coherent transport for orth case')
    g = surfG(H, np.eye(len(H)), [contactInds, onsiteInds])
    fermi = getFermiContact(g, ne)
    Elist = np.linspace(-5, 5, 1000)
    Torth = calculate_transmission(H, np.eye(len(H)), SigmaCalculator(g), Elist+fermi)

    io.savemat('SiNanowire_TnoSCF.mat', {'Elist':Elist, 'fermi':fermi, 'T':T, 'Torth':Torth})


Part 2: Transport With SCF
----------------------

This approach uses self-consistent field calculations with different temperature settings using the input file given by

.. code-block:: text

    %chk=Si2.chk
    #p b3lyp/lanl2dz 

    Title Card Required

    0 1
    Si             3.00000000    0.00000000    0.00000000
    Si             0.00000000    0.00000000    0.00000000

Using the ``negf.setContact1D`` method, we can model the nanowire with self-consistent field calculations by folding the nanowire back on itself.

.. code-block:: python

    from gauNEGF.scfE import NEGFE
    from gauNEGF.transport import *
    from gauNEGF.utils import fractional_matrix_power

    har_to_eV = 27.211386

    negf = NEGFE(fn='Si2', func='b3lyp', basis='lanl2dz')
    inds = negf.setContact1D([[1],[2]], eta=1e-4) #Again, some broadening to speed up convergence
    negf.setVoltage(0)
    # This type of contact is unstable, setting a low damping value
    negf.SCF(1e-2, 0.005, 200)
    negf.saveMAT('SiNanowire_ESCF.mat')

    Torth = calculate_transmission(negf.F*har_to_eV, negf.S, SigmaCalculator(negf.g), Elist+negf.fermi)
    io.savemat('SiNanowire_TESCF.mat', {'Elist':Elist, 'fermi':negf.fermi, 'T':T})

    inds = negf.setContact1D([[1],[2]], T=300, eta=1e-4)
    negf.SCF(1e-3, 0.002, 200)
    negf.saveMAT('SiNanowire_ESCF_300K.mat')

    Torth = calculate_transmission(negf.F*har_to_eV, negf.S, SigmaCalculator(negf.g), Elist+negf.fermi)
    io.savemat('SiNanowire_TESCF_300K.mat', {'Elist':Elist, 'fermi':negf.fermi, 'T':T})


Key Points
--------

1. **Part 1: No SCF**
   - Uses 12 Si atoms to approximate infinite chain
   - Calculates both orthogonal and non-orthogonal cases
   - Uses broadening (eta=1e-4) for convergence

2. **Part 2: With SCF**
   - Uses NEGFE for self-consistent calculations
   - Implements 1D chain contacts
   - Includes both zero and finite temperature (300K)
   - Uses low damping values due to contact instability 
