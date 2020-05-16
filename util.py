import numpy as np

'''
    Implements the Gaussian or squared exponential kernel
    Used as covariance function k(x, x')
    Parameters:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
def rbf_kernel(X1, X2, d=1.0, l=1.0):
    # (X1 - X2)^2 => X1^2 + X2^2 - 2 * X1X2
    X1_norm = np.sum(X1**2, axis=1)
    X2_norm = np.sum(X2**2, axis=1)
    squared_norm = X1_norm.reshape(-1, 1) + X2_norm - 2 * np.dot(X1, X2.T)

    return d * np.exp(-(0.5/(l**2)) * squared_norm)