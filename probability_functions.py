import numpy as np
import scipy
from util import gaussian_cdf, gaussian_pdf, chain_2

class Bernoulli():
    # Response function for squashing function values
    # Also known as an inverse link function
    # Implements the inverse link of a probit function
    def response_function(self, f):
        return gaussian_cdf(f)

    # First derivative of the response function
    def response_derivative(self, f):
        return gaussian_pdf(f)
    
    def response_second_derivative(self, f):
        return -(f * gaussian_pdf(f))

    #############################
    #   Likelihood Functions    #
    #############################
    # p(y | sig(f))
    def likelihood(self, f, y):
        inverse_link_f = self.response_function(f)
        
        # Obtain probabilities based on the y-values
        # Invert 1 - p to get p(y = 0|f)
        return np.where(y==1, inverse_link_f, 1. - inverse_link_f)

    # log(p(y | sig(f)))
    def log_likelihood(self, f, y):
        p = self.likelihood(f, y)
        p = np.clip(p, 1e-9, np.inf)
        
        return np.log(p)

    # d(log(p(y | sig(f)))) / d(sig(f))
    def d_log_likelihood_link(self, f, y):
        inverse_link_f = self.response_function(f)
        inverse_link_f = np.clip(inverse_link_f, 1e-9, 1 - 1e-9)
        
        return 1. / np.where(y==1, inverse_link_f, -(1 - inverse_link_f))

    # Second derivative of log likelihood w.r.t to link function
    # d2(log(p(y | sig(f)))) / d(sig(f))2
    def d2_log_likelihood_link(self, f, y):
        inverse_link_f = self.response_function(f)
        inverse_link_f = np.where(y==1, inverse_link_f, 1. - inverse_link_f)

        return inverse_link_f

    # Seconmd derivative on the log likelihood of y given the values of f/sig(f)

    def d2_log_likelihood(self, f, y):
        d2_log_likelihood_link_ = self.d2_log_likelihood_link(f, y)
        d_response = self.response_derivative(f)
        d_log_likelihood_link_ = self.d_log_likelihood_link(f, y)
        d2_response = self.response_second_derivative(f)

        d2_log_likelihood_ = chain_2(d2_log_likelihood_link_, d_response, d_log_likelihood_link_, d2_response)
        return d2_log_likelihood_
