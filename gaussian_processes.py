import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky, cho_solve
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

'''
    Faster version of gp_regression. Uses cholesky decomposition instead of direction inversion
    Requires that the inverting matrix is positive definite
'''
def gp_regression_fast(X, y, X_new, l=1.0, d=1.0, sigma_y=1e-8, kernel='rbf'):
    k = kernels[kernel]

    # Mean prediction vector
    K = k(X, X, d, l) + sigma_y**2 * np.eye(len(X))
    # L = cholesky(K, lower=True) # Lower triangular
    # alpha = cho_solve((L, True), y)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    k_star = k(X, X_new, d, l)
    mu_new = k_star.T.dot(alpha)

    # Covariance matrix
    # v = cho_solve((L, True), k_star)
    v = np.linalg.solve(L, k_star)
    k_star2 = k(X_new, X_new, d, l)
    # cov_new = k_star2 - k_star.T.dot(v)
    cov_new = k_star2 - v.T.dot(v)

    # Log-likelihood
    n = len(X)
    log_sum = np.sum(np.log(np.diagonal(L)))
    log_likelihood = -0.5 * y.T.dot(alpha) - log_sum - 0.5*(n * np.log(2 * np.pi))

    return mu_new, cov_new, log_likelihood

