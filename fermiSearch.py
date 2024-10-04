import numpy as np
from scipy import linalg as LA
from math import factorial

class DOSFermiSearch:
    """
    A class for performing Fermi energy search using DOS (Density of States) information.
    It uses a Taylor series expansion and finite difference methods to estimate the Fermi energy.
    """

    def __init__(self, initial_Ef, N_target, delta_E=0.01, numpoints=3, debug=False):
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
        left_points = (numpoints - 1) // 2
        right_points = numpoints - 1 - left_points
        points = np.concatenate([
            np.arange(-left_points, 0) * h,
            [0],
            np.arange(1, right_points + 1) * h
        ])
        A = np.zeros((numpoints, numpoints))
        for i in range(numpoints):
            for j in range(numpoints):
                A[i, j] = points[i]**j / factorial(j)
        b = np.array([dos_func(E + p) for p in points])
        derivatives = np.linalg.solve(A, b)
        return derivatives

    def step(self, dos_func, N_curr, stepLim=1e2):
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
        coeffs = [0] * self.numpoints
        coeffs[0] = -delta_N
        for n in range(1, self.numpoints):
            coeffs[n] = dos_derivatives[n-1] / factorial(n)

        # Find the root of the polynomial
        roots = np.roots(coeffs[::-1])
        if self.debug:
            print('Final solver roots:', roots)

        # Select the root with the smallest absolute value
        root = min(roots, key=abs).real
        if np.abs(root)>stepLim:
            print('WARNING: delta_Ef too big! Fermi energy not updated')
            new_Ef = self.Ef
        else:
            self.delta_Ef = root
            new_Ef = self.Ef + self.delta_Ef
            self.Ef = new_Ef
        return new_Ef
