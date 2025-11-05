import numpy as np

def getANTPoints(N):
    """
    Generate integration points and weights matching ANT.Gaussian implementation.

    Follows the IntCompPlane subroutine in device.F90 from ANT.Gaussian package.
    Uses a modified Gauss-Chebyshev quadrature scheme optimized for transport
    calculations.

    Parameters
    ----------
    N : int
        Number of integration points

    Returns
    -------
    tuple
        (points, weights) - Arrays of integration points and weights
    """
    k = np.arange(1,N+1,2)
    theta = k*np.pi/(2*N)
    xs = np.sin(theta)
    xcc = np.cos(theta)

    # Transform points using ANT-like formula
    x = 1.0 + 0.21220659078919378103 * xs * xcc * (3 + 2*xs*xs) - k/(N)
    x = np.concatenate((x,-1*x))
    
    # Generate weights similarly to ANT
    w = xs**4 * 16.0/(3*(N))
    w = np.concatenate((w, w))

    return x, w

if __name__ == "__main__":
    func = lambda x: np.exp(-x**2)

    prev_x = None
    prev_sumW = None
    val = 0.0
    N = 2

    for level in range(1, 8):
        x, w = getANTPoints(N)
        direct = float(np.dot(w, func(x)))
        print(f"N={N}, direct={direct:.12f}, sumW={np.sum(w):.12f}")

        if prev_x is None:
            # first level: no reuse
            val = direct
        else:
            # mark old nodes robustly by value
            old_mask = np.isin(np.round(x, 14), np.round(prev_x, 14))
            # sanity: all previous nodes should be found
            assert int(old_mask.sum()) == prev_x.size, "Old nodes mismatch"

            # exact transfer factor (should be ~1/3)
            ratio = float(np.sum(w[old_mask]) / prev_sumW)
            print(f"level {level}: N={N}, nested-weight ratio ~ {ratio:.12f}")

            # scale previous integral + add only new-node contributions
            new_mask = ~old_mask
            val = val * ratio + float(np.dot(w[new_mask], func(x[new_mask])))

        # update state for next level
        prev_x = x
        prev_sumW = float(np.sum(w))
        N *= 3

    print(f"final value {val:.12f}")