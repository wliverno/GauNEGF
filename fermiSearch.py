import numpy as np
from scipy import linalg as LA
from math import factorial

class DOSFermiSearch:
    """
    A class for performing Fermi energy search using DOS (Density of States) information.
    It uses a Taylor series expansion and finite difference methods to estimate the Fermi energy.
    """

    def __init__(self, initial_Ef, N_target, delta_E=0.01, numpoints=2, debug=False):
        """
        Initialize the Fermi energy search object.

        :param initial_Ef: Initial guess for the Fermi energy
        :param N_target: Target number of electrons
        :param delta_E: Step size for finite difference calculations
        :param numpoints: Number of points to use in finite difference method
        :param debug: Flag to enable debug output
        """
        self.Ef = initial_Ef
        self.N_target = N_target
        self.delta_E = delta_E
        self.numpoints = numpoints
        self.delta_Ef = None
        self.debug = debug

    def get_accuracy(self):
        """
        Return the absolute change in Fermi energy from the last step, or infinity if not set.

        :return: Absolute change in Fermi energy or infinity
        """
        return abs(self.delta_Ef) if self.delta_Ef is not None else float('inf')

    def matrix_finite_difference(self, dos_func, E, h, numpoints):
        """
        Compute finite difference approximation of DOS and its derivatives.

        :param dos_func: Function to compute DOS
        :param E: Energy point around which to compute derivatives
        :param h: Step size for finite difference
        :param numpoints: Number of points to use in finite difference
        :return: Array of derivatives
        """
        points = np.linspace(-h, h, numpoints)
        A = np.zeros((numpoints, numpoints))
        for i in range(numpoints):
            for j in range(numpoints):
                A[i, j] = points[i]**j / factorial(j)
        b = np.array([dos_func(E + p) for p in points])
        derivatives = np.linalg.solve(A, b)
        return derivatives

    def step(self, dos_func, N_curr, stepLim=10):
        """
        Perform one step of the Fermi energy search.

        :param dos_func: Function to compute DOS
        :param N_curr: Current number of electrons
        :return: New Fermi energy estimate
        """

        delta_N = self.N_target - N_curr
        if self.debug:
            print(self.N_target, " <-- ", N_curr)

        # Get DOS and its derivatives
        dos_derivatives = self.matrix_finite_difference(dos_func, self.Ef, self.delta_E, self.numpoints)
        if self.debug:
            print('DOS Derivatives list: ', dos_derivatives)

        # Solve the Taylor series equation
        coeffs = [0] * (self.numpoints+1)
        coeffs[0] = -delta_N
        for n in range(0, self.numpoints):
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
                print("Warning: No real roots found, using real part of roots")
            root = roots.real[np.argmin(roots.real)]
        
        # Relaxation 
        #root *= 0.5 

        if np.abs(root)>stepLim:
            print(f'WARNING: delta_Ef cutoff reached! Incrementing by {stepLim} eV')
            if self.delta_Ef == -np.sign(root)*stepLim:
                self.delta_Ef = np.sign(root)*stepLim*0.5
            else:
                self.delta_Ef = np.sign(root)*stepLim
        else:
            self.delta_Ef = root
        new_Ef = self.Ef + self.delta_Ef
        self.Ef = new_Ef
        return new_Ef
