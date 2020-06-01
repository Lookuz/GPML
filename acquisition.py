import numpy as np
from gaussian_processes import gp_regression_fast
from util import gaussian_cdf, gaussian_pdf
from scipy.optimize import minimize

"""
    Expected Improvement acquistion function defined as:
    EI(X_new) = E[max(f(X_new) - f_max, 0)]
    which computes the expected increase in objective function
    from the best observation f_bar from sampled points so far.
    
    Parameters:
        X_new: new input points to compute the EI at
        X: Currently observed inputs
        y: Target labels of currently observed inputs
        xi: Noise label; Higher values reduce the actual improvement values
        relative to the variance, favoring regions with higher uncertainty

    Returns:
        Expected improvement values of new inputs X
"""
def expected_improvement(X_new, X, y, xi=0.01):
    mu_new, cov_new, _ = gp_regression_fast(X, y, X_new)
    f_max = np.max(y) # Take max of current observations so far

    sigma_new = np.diag(cov_new)

    # Expected improve over GP prior on functions
    with np.errstate(divide='warn'):
        improvement = mu_new - f_max - xi
        Z = improvement/sigma_new
        expected_imp = improvement*gaussian_cdf(Z) + sigma_new*gaussian_pdf(Z)
        expected_imp[sigma_new == 0.0] = 0.0
        
    return expected_imp

"""
    Obtains the next input location to sample at that best improves the current objective function value
    Gain in the objective function value is determined by the acquisition function
    Local objective function minimisation is used alongside random restarts from points within the bounds provided
    
    Parameters:
        acquisition: Acquisition function to be used
        bounds: Range of values to be sampled from duiring random restarts
        X: Locations of already observed inputs
        y: Target labels of already observed inputs
        xi: Noise label; Influences exploration(See Expected Improvement)
        n_restarts: Number of restarts to perform during local minimisation
"""
def maximise_acquisition(acquistion, bounds, X, y, xi=0.01, n_restarts=25):
    input_dim = X.shape[1]
    min_improvement = 1.
    x_best = None

    # Maximising acquisition equivalent to minimising the negative of the acquisition
    def objective(X_new):
        return -acquistion(X_new.reshape(-1, input_dim), X, y, xi=xi)

    for x in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, input_dim)):
        res = minimize(objective, x0=x, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_improvement:
            min_improvement = res.fun[0]
            x_best = res.x

    return x_best.reshape(1, input_dim)