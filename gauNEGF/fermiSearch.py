"""
Fermi energy search using density of states (DOS) information.

[DEPRECATED] This module has been replaced by the constant sigma approximation
method, which provides better stability and performance. It is kept for
reference and historical purposes.

This module implements a Fermi energy search algorithm based on DOS information
using Taylor series expansion and finite difference methods. The algorithm:
1. Computes DOS and its derivatives using finite difference
2. Constructs a Taylor series approximation
3. Solves for the root to find the Fermi energy shift

The implementation supports:
- Arbitrary-order finite difference calculations
- Automatic step size adjustment
- Debug output for troubleshooting
"""

import numpy as np
from scipy import linalg as LA
from math import factorial

class DOSFermiSearch:
    """
    Fermi energy search using DOS information.

    [DEPRECATED] This class has been replaced by simpler and more stable methods.
    It uses a Taylor series expansion of the DOS to predict Fermi energy shifts
    that will achieve a target electron count.

    Parameters
    ----------
    initialEf : float
        Initial guess for the Fermi energy in eV
    nTarget : float
        Target number of electrons
    deltaE : float, optional
        Step size for finite difference calculations in eV (default: 0.01)
    numPoints : int, optional
        Number of points to use in finite difference method (default: 5)
    debug : bool, optional
        Enable debug output (default: False)

    Notes
    -----
    The algorithm uses finite difference to compute DOS derivatives,
    then constructs and solves a Taylor series equation to predict
    the required Fermi energy shift.
    """
    def __init__(self, initialEf, nTarget, deltaE=0.01, numPoints=5, debug=False):
        """
        Initialize the Fermi energy search object.

        Parameters
        ----------
        initialEf : float
            Initial guess for the Fermi energy in eV
        nTarget : float
            Target number of electrons
        deltaE : float, optional
            Step size for finite difference calculations in eV (default: 0.01)
        numPoints : int, optional
            Number of points to use in finite difference method (default: 5)
        debug : bool, optional
            Enable debug output (default: False)
        """
        self.Ef = initialEf
        self.nTarget = nTarget
        self.deltaE = deltaE
        self.numPoints = numPoints
        self.deltaEf = initialEf
        self.debug = debug

    def getAccuracy(self):
        """
        Get the absolute change in Fermi energy from the last step.

        Returns
        -------
        float
            Absolute change in Fermi energy, or infinity if not set
        """
        return abs(self.deltaEf) if self.deltaEf is not None else float('inf')

    def matrixFiniteDifference(self, dosFunc, E, h, numPoints):
        """
        Compute finite difference approximation of DOS and its derivatives.

        Uses a Vandermonde matrix approach to compute derivatives up to
        order numPoints-1 around energy E.

        Parameters
        ----------
        dosFunc : callable
            Function to compute DOS, takes energy as argument
        E : float
            Energy point around which to compute derivatives in eV
        h : float
            Step size for finite difference in eV
        numPoints : int
            Number of points to use in finite difference

        Returns
        -------
        ndarray
            Array of derivatives [DOS, dDOS/dE, d²DOS/dE², ...]
        """
        points = np.linspace(-h, h, numPoints)
        A = np.zeros((numPoints, numPoints))
        for i in range(numPoints):
            for j in range(numPoints):
                A[i, j] = points[i]**j / factorial(j)
        b = np.array([dosFunc(E + p) for p in points])
        derivatives = np.linalg.solve(A, b)
        return derivatives

    def step(self, dosFunc, nCurr, stepLim=10):
        """
        Perform one step of the Fermi energy search.

        Uses the DOS and its derivatives to predict the Fermi energy
        shift needed to reach the target electron count.

        Parameters
        ----------
        dosFunc : callable
            Function to compute DOS, takes energy as argument
        nCurr : float
            Current number of electrons
        stepLim : float, optional
            Maximum allowed step size in eV (default: 10)

        Returns
        -------
        float
            New Fermi energy estimate in eV

        Notes
        -----
        The method:
        1. Computes DOS derivatives using finite difference
        2. Constructs a Taylor series equation
        3. Solves for the root to find deltaEf
        4. Applies step size limits and sign corrections
        """
        delta_N = self.nTarget - nCurr
        if self.debug:
            print(self.nTarget, " <-- ", nCurr)

        # Get DOS and its derivatives
        h = min(self.deltaE, np.abs(self.deltaEf/10))
        dos_derivatives = self.matrixFiniteDifference(dosFunc, self.Ef, h, self.numPoints)
        if self.debug:
            print('DOS Derivatives list: ', dos_derivatives)

        # Solve the Taylor series equation
        coeffs = [0] * (self.numPoints+1)
        coeffs[0] = -delta_N
        for n in range(0, self.numPoints):
            coeffs[n+1] = dos_derivatives[n] / factorial(n+1)

        # Find the root of the polynomial
        roots = np.roots(coeffs[::-1])
        if self.debug:
            print('Final solver roots:', roots)
        
         # Check if there are any real roots
        real_roots = roots[np.abs(roots.imag)<1e-9].real
        root = 0 
        if len(real_roots) > 0:
            # If there are real roots, return the minimum absolute value
            root = real_roots[np.argmin(np.abs(real_roots))]
        else:
            # If there are no real roots, print a warning and return the smallest real part
            if self.debug:
                print("Warning: No real roots found, using Newton's method...")
            root = delta_N/dos_derivatives[0]
        
        # Relaxation 
        #root *= 0.5 

        if np.abs(root)>stepLim:
            print(f'Warning: deltaEf cutoff reached! Incrementing by {stepLim} eV')
            if self.deltaEf == -np.sign(root)*stepLim:
                self.deltaEf = np.sign(root)*stepLim*0.5
            else:
                self.deltaEf = np.sign(root)*stepLim
        else:
            self.deltaEf = root
        if np.sign(delta_N.real) != np.sign(self.deltaEf.real):
            print('Warning: deltaEf sign error corrected')
            self.deltaEf *= -1
        new_Ef = self.Ef + self.deltaEf
        self.Ef = new_Ef
        return new_Ef

