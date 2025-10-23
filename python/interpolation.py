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

def build_interpolant(t, v):
    if t.size == 0:
        return None
    if t.size == 1:
        const = float(v[0])
        return lambda tt: np.full_like(np.asarray(tt, dtype=float), const, dtype=float)
    # ensure t is strictly increasing for spline; if not, sort and unique
    sort_idx = np.argsort(t)
    t_s = t[sort_idx]
    v_s = v[sort_idx]
    # remove duplicate timestamps by keeping the last value
    unique_t, unique_idx = np.unique(t_s, return_index=True)
    if unique_t.size != t_s.size:
        # take first occurrence for each unique time
        t_s = t_s[unique_idx]
        v_s = v_s[unique_idx]
    try:
        return cubic_spline_interpolation(t_s, v_s)
    except Exception:
        # fallback to numpy interp (clamped extrapolation)
        print('fall back to numpy interp')
        return lambda tt: np.interp(np.asarray(tt, dtype=float), t_s, v_s, left=v_s[0], right=v_s[-1])

