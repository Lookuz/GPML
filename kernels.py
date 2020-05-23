import numpy as np
import math
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist

'''
    Class that implements kernel functions to be used as covariance function k(x, x')
    Parameters:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        l: Length scale parameter
        d: Signal variance

    Returns:
        Covariance matrix (m x n).
'''

# Radial Basis Function(RBF), or squared exponential/Gaussian kernel
def rbf(X1, X2, d=1.0, l=1.0):
    # (X1 - X2)^2 => X1^2 + X2^2 - 2 * X1X2
    X1_norm = np.sum(X1**2, axis=1)
    X2_norm = np.sum(X2**2, axis=1)
    squared_norm = X1_norm.reshape(-1, 1) + X2_norm - 2 * np.dot(X1, X2.T)

    return d**2 * np.exp(-(0.5/(l**2)) * squared_norm)

# Polynomial kernel given by (bias + <X1, X2>)
def poly(X1, X2, bias=1., order=2.):
    X1X2 = X1.dot(X2.T)
    result = X1X2 + bias
    result = result ** order

    return result

# Matern Kernel 
# Special cases for nv = 0.5, 1.5, 2.5
# For other values that not as above, the general Matern kernel formula will be used
# which can be very expensive
def matern(X1, X2, nv=1.5, l=1.0):
    X1 = X1 / l
    X2 = X2 / l

    abs_dist = cdist(X1, X2, metric='euclidean')

    if nv == 0.5:
        K = np.exp(-abs_dist)
    elif nv == 1.5:
        r = math.sqrt(3.0) * abs_dist
        K = (1.0 + r) * np.exp(-r)
    elif nv == 2.5:
        r = math.sqrt(5) * abs_dist
        K = (1.0 + r + (r**2 / 3.0)) * np.exp(-r)
    else:
        a = (2.0**(1.0 - nv)) / gamma(nv)
        R =  math.sqrt(2.0 * nv) * abs_dist
        K = a * (R**nv) * kv(nv, R)

    return K