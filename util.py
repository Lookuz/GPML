import numpy as np
from scipy.special import ndtr
from scipy.linalg import lapack

_lim_val = np.finfo(np.float64).max
_lim_val_square = np.sqrt(_lim_val)

'''
    Implements the Gaussian or squared exponential kernel
    Used as covariance function k(x, x')
    Parameters:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        l: Length scale parameter
        d: Signal variance

    Returns:
        Covariance matrix (m x n).
    '''
def rbf_kernel(X1, X2, d=1.0, l=1.0):
    # (X1 - X2)^2 => X1^2 + X2^2 - 2 * X1X2
    X1_norm = np.sum(X1**2, axis=1)
    X2_norm = np.sum(X2**2, axis=1)
    squared_norm = X1_norm.reshape(-1, 1) + X2_norm - 2 * np.dot(X1, X2.T)

    return d**2 * np.exp(-(0.5/(l**2)) * squared_norm)

def sigmoid(f):
    return 1  / (1 + np.exp(-f))

def gaussian_cdf(f):
    return ndtr(f)

def gaussian_pdf(f):
    f = np.clip(f, -1e150, 1e150)
    return np.exp(-np.square(f)/2)/np.sqrt(2 * np.pi)

# Faa Di Bruno's methods for chain rule
# Taken from the implementation used by GPy
def chain_2(d2f_dg2, dg_dx, df_dg, d2g_dx2):
    if np.all(dg_dx==1.) and np.all(d2g_dx2 == 0):
        return d2f_dg2
         
    dg_dx_2 = np.clip(dg_dx, -np.inf, _lim_val_square)**2
    return d2f_dg2*(dg_dx_2) + df_dg*d2g_dx2

def dtrtrs(A, B, lower=1, trans=0, unitdiag=0):
    """
    Wrapper for lapack dtrtrs function
    DTRTRS solves a triangular system of the form
        A * X = B  or  A**T * X = B,
    where A is a triangular matrix of order N, and B is an N-by-NRHS
    matrix.  A check is made to verify that A is nonsingular.
    :param A: Matrix A(triangular)
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns: Solution to A * X = B or A**T * X = B
    """
    A = np.asfortranarray(A)
    #Note: B does not seem to need to be F ordered!
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)