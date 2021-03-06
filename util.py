import numpy as np
from scipy.special import ndtr
from scipy.linalg import lapack

_lim_val = np.finfo(np.float64).max
_lim_val_square = np.sqrt(_lim_val)

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