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
def laplace_mode_finding(K, f, y, likelihood, tol=1e-4, max_iter=30):
    
    # Initialization
    f = np.zeros_like(y)
    iteration = 0

    while iteration < max_iter:
        # W matrix
        W = - likelihood.d2_log_likelihood(f, y)
        b = W * f + likelihood.d_log_likelihood(f, y)

        