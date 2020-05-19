import numpy as np
import likelihood_classes
from util import dtrtrs

"""
    Laplace Approximation inference for determining the classification probabilities on new inputs
    Given the mode and variance of the posterior distribution on latent functions f on observations

    Parameters:
        X: Observed input points
        X_new: New unobserved points to be evaluated at
        y: Labels of the observed inputs X
        likelihood: Likelihood class of labels y given f p(y|f). For choice of likelihood/probability 
            distribution functions, look at sigmoid.py
        kernel: Kernel function to be used
"""
def laplace_approximation_binary(X_new, X, y, likelihood, kernel, tol=1e-4, max_iter=30, l=1.0, d=1.0):
    # Covariance matrix of observations K
    K = kernel(X, X, d, l)

    # Mode finding to get mean vector of posterior q(f|X, y)
    f_hat, log_ll = laplace_mode_finding(K, y, likelihood, tol=tol, max_iter=max_iter)

    W = np.diagflat(-(likelihood.d2_log_likelihood(f_hat, y)))
    W_12 = np.sqrt(W)
    B = np.eye(len(K)) + W_12.dot(K).dot(W_12)
    L = np.linalg.cholesky(B)

    # Mean vector
    k_star = kernel(X, X_new, d, l)
    d_log_likelihood = likelihood.d_log_likelihood(f_hat, y)
    f_bar = k_star.T.dot(d_log_likelihood)

    # Covariance matrix
    v = np.linalg.solve(L, W_12.dot(k_star))
    k_star2 = kernel(X_new, X_new)
    cov = k_star2 - v.T.dot(v)

    # Additional parameters for computing predictions 
    LiW_12, _ = dtrtrs(L, np.diagflat(W_12), lower=1, trans=0)
    woodbury_inv = LiW_12.T.dot(LiW_12)

    return f_bar, cov

"""
    Mode finding algorithm for binary Laplace GPC.
    Utilizes Newton's method for numerically finding the maximum likelihood values of f
    Parameters:
        K: Covariance matrix of f on X
        y: Labels of observations X
        likelihood: Likelihood class of labels y given f p(y|f). For choice of likelihood/probability 
            distribution functions, look at sigmoid.py
"""
def laplace_mode_finding(K, y, likelihood, tol=1e-4, max_iter=30):
    
    # Initialization
    f = np.zeros_like(y)
    f_old = f
    old_obj = - np.inf
    iteration = 0

    while iteration < max_iter:
        W = np.diagflat(-likelihood.d2_log_likelihood(f, y)) # Make W diagonal matrix
        W_12 = np.sqrt(W)
        B = np.eye(len(K)) + W_12.dot(K).dot(W_12)
        L = np.linalg.cholesky(B)
        
        b = W.dot(f) + likelihood.d_log_likelihood(f, y)
        
        a = b - W_12.dot(np.linalg.solve(L.T, np.linalg.solve(L, W_12.dot(K).dot(b))))

        f = K.dot(a)

        new_obj = laplace_mode_objective(a, f, y, likelihood, tol=tol)
        
        if new_obj < old_obj:
            raise ValueError("Optimization failed. Decrease in objective value from {0} to {1}".format(old_obj, new_obj))
        else:
            if new_obj - old_obj < tol:
                break

        iteration += 1
        # Update previous iteration values
        f_old = f
        old_obj = new_obj
        
    new_obj -= np.sum(np.log(np.diagonal(L)))
    return f, new_obj


# Models the convergence objective of the laplace mode finding algorithm
# Maximize: -0.5 * aT*f + log(p(y|f))
def laplace_mode_objective(a, f, y, likelihood, tol=1e-4):
    obj = -0.5 * np.sum(a.T.dot(f)) + np.sum(likelihood.log_likelihood(f, y))

    return obj