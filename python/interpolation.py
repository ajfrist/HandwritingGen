import numpy as np
from scipy.interpolate import CubicSpline

def newton_interpolation(x, y):
    """
    Generate a function that interpolates points using Newton's divided differences.
    Returns a callable function that can evaluate the polynomial at any point.
    """
    n = len(x)
    # Calculate divided differences table
    coef = np.zeros(n)
    coef[0] = y[0]
    
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            y[i] = (y[i] - y[i-1])/(x[i] - x[i-j])
        coef[j] = y[j]
    
    def polynomial(t):
        """Evaluate the Newton polynomial at point t"""
        p = coef[n-1]
        for k in range(n-2, -1, -1):
            p = p * (t - x[k]) + coef[k]
        return p
    
    return polynomial

def cubic_spline_interpolation(x, y):
    """
    Generate a function that interpolates points using cubic splines.
    Returns a callable function that can evaluate the spline at any point.
    """
    cs = CubicSpline(x, y, bc_type='natural')
    return cs
