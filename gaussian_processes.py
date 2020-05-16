import numpy as np
from numpy.linalg import inv
from util import rbf_kernel

kernels = {
    'rbf' : rbf_kernel
}

'''
    Computes the posterior distribution of the GP on the new inputs X_test
    conditioned on training dataset (X, y)
    and derives the prediction mean and it's covariance

    Args:
        X: Training inputs (m x d).
        Y: Training target output (m x 1).
        l: Length scale parameter.
        d: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
'''
def gp_regression(X, y, X_new, l=1.0, d=1.0, sigma_y=1e-8, kernel='rbf'):
    k = kernels[kernel]

    # Mean prediction vector
    K_y = k(X, X, d, l) + sigma_y**2 * np.eye(len(X))
    K_ynew = k(X, X_new, d, l)
    K_y_inv = inv(K_y)
    mu_new = (K_ynew.T).dot(K_y_inv).dot(y)

    # Covariance matrix
    K_newnew = k(X_new, X_new, d, l)
    cov_new = K_newnew - (K_ynew.T).dot(K_y_inv).dot(K_ynew)

    return mu_new, cov_new
