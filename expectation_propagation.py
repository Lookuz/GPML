import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import cholesky, cho_solve
from kernels import rbf, poly, matern, rational_quadratic, exp_sin_sq, nn
from util import gaussian_cdf, gaussian_pdf

kernels = {
    'rbf' : rbf,
    'poly' : poly,
    'matern' : matern,
    'rational_quadratic' : rational_quadratic,
    'exp_sin_sq' : exp_sin_sq,
    'nn' : nn
}

"""
    Function that computes the natural site parameters, nu and tau
    as well as approximate log marginal likelihood as in algorithm 3.5 of GPML
    Parameters:
        K: Covariance matrix of observations X
        y: Target output labels of observations X
"""
def expectation_propagation(X, y, tol=1e-4, max_iter=50, kernel='rbf'):
    stop = False
    k = kernels[kernel]
    K = k(X, X)

    # Approximate likelihood site parameters
    v_t = np.zeros_like(X)
    tau_t = np.zeros_like(X)
    v_prev = np.copy(v_t)
    tau_prev = np.copy(tau_t)

    # Posterior parameters
    # sigma = np.copy(K)
    sigma = K + 1e-6*np.eye(len(K))
    mu = np.zeros_like(X)

    # Cavity parameters
    v = np.zeros_like(X)
    tau = np.zeros_like(X)

    # Marginal moments
    mu_hat = np.zeros_like(X)
    sigma_hat = np.zeros_like(X)

    n = X.shape[0]
    num_iter = 0
    while (not stop) and num_iter < max_iter:
        for i in range(n):
            # Compute v and tau
            tau[i] = 1./sigma[i][i] - tau_t[i]
            v[i] = mu[i]/sigma[i][i] - v_t[i]

            # Compute marginal moments
            y_i = 1. if y[i] == 1 else -1.
            z_i = (y_i * v[i])/np.sqrt(tau[i]**2 + tau[i])

            normal_factor = gaussian_pdf(z_i) / gaussian_cdf(z_i) # N(z_i)/phi(z_i) in 3.58

            mu_hat[i] = v[i]/tau[i] + (y_i*normal_factor)/np.sqrt(tau[i]**2 + tau[i])
            sigma_hat[i] = 1./tau[i] + (normal_factor/(tau[i]**2 + tau[i]))*(z_i + normal_factor)

            # Update approximate likelihood site parameters
            del_tau = 1./sigma_hat[i] - tau[i] - tau_t[i]
            # del_tau = 1./sigma_hat[i] - tau[i]
            # prev_tau_t = tau_t[i]
            tau_t[i] += del_tau

            if tau_t[i] < np.finfo(float).eps:
                del_tau += np.finfo(float).eps - tau_t[i]
                tau_t[i] = np.finfo(float).eps
                # del_tau = tau_t[i] - prev_tau_t

            v_t[i] = mu_hat[i]/sigma_hat[i] - v[i]

            # Update posterior parameters
            s_i = sigma[:, i]
            c_i = del_tau/(1 + del_tau*sigma[i,i])
            sigma -= c_i*np.dot(s_i[:, None], s_i[None, :])
            mu = sigma.dot(v_t)

        # Recompute approximate posterior
        S_12 = np.diag(np.sqrt(tau_t).flatten())
        L = np.linalg.cholesky(np.eye(len(K)) + S_12.dot(K).dot(S_12))
        V = np.linalg.solve(L.T, S_12.dot(K))
        sigma = K - V.T.dot(V)
        mu = sigma.dot(v_t)

        # Check for convergence
        tau_diff = np.mean(np.square(tau_t - tau_prev))
        v_diff = np.mean(np.square(v_t - v_prev))
        if tau_diff.all() < tol and v_diff.all() < tol:
            stop =  True
            
        num_iter += 1

    return v_t, tau_t

"""
    Prediction of new input points using Expectation Propagation as in 
    algorithm 3.6 of GPML textbook
    NOTE: Work in progress, currently still buggy :(
"""
def expectation_propagation_predict(X_new, X, y, tol=1e-4, max_iter=50, kernel='rbf'):
    v_t, tau_t = expectation_propagation(X, y, tol=tol, max_iter=max_iter, kernel=kernel)

    k = kernels[kernel]
    K = k(X, X)
    S_12 = np.diag(np.sqrt(tau_t).flatten())
    L = np.linalg.cholesky(np.eye(len(K)) + S_12.dot(K).dot(S_12))
    z = S_12.dot(np.linalg.solve(L.T, np.linalg.solve(L, S_12.dot(K).dot(v_t))))\
    
    # Predictive distribution
    k_star = k(X, X_new)
    f_bar = k_star.T.dot(v_t - z)
    v = np.linalg.solve(L, S_12.dot(k_star))
    var = k(X_new, X_new) - v.T.dot(v)

    pi_bar = gaussian_cdf(f_bar / np.sqrt(1 + np.diag(var))[:, None])

    return pi_bar, (f_bar, var)
    