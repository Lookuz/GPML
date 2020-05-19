import numpy as np

def laplace_approximation_binary():
    pass


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
        print("Iteration {0}: Marginal Likelihood = {1}".format(iteration, new_obj))
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