from graphlearning.active_learning import acquisition_function
import numpy as np

class dirichlet_var(acquisition_function):
    '''
    Dirichlet Learning Variance

    Note: u now is actually the matrix A in Dirichlet Learning. Need to ensure
    '''
    def compute(self, u, candidate_ind):
        a0 = u.sum(axis=1)
        a = (u * u).sum(axis=1)
        return ((1. - a/(a0**2.))/(1. + a0))[candidate_ind]

class dirichlet_varprop(acquisition_function):
    '''
    Dirichlet Learning Variance, with percentile sampling, not max.
    
    CURRENTLY ONLY IMPLEMENTED FOR SEQUENTIAL, NOT BATCH
    '''
    def __init__(self, seed=42):
        self.K = 10
        self.log_Eps_tilde = np.log(1e150)  # log of square root of roughly the max precision of python float
        self.rand_state = np.random.RandomState(seed)

    def set_K(self, K):
        print(f"Setting K = {K} for betavarprop")
        self.K = K
        
    def compute(self, u, candidate_ind):
        a0 = u.sum(axis=1)
        a = (u * u).sum(axis=1)
        vals = ((1. - a/(a0**2.))/(1. + a0))[candidate_ind]
        
        # scaling for p(x) \propto e^{x/T}, where T is scales as the values change. Ensures no numerical overflow occurs
        M = vals.max()
        T0 = M - np.percentile(vals, 100*(1. - 1./self.K))
        eps = M / (self.log_Eps_tilde - np.log(vals.size))
        T = max(eps, min(1.0,T0))
        p = np.exp(vals/T)
        
        # return values so that this k_choice will be the maximizer
        k_choice = self.rand_state.choice(np.arange(candidate_ind.size), p=p/p.sum())
        acq_vals = np.zeros_like(candidate_ind)
        acq_vals[k_choice] = 1.
        
        return acq_vals

class random(acquisition_function):
    '''
    Random choices
    '''
    def __init__(self, seed=42):
        self.rand_state = np.random.RandomState(seed)

    def compute(self, u, candidate_ind):
        return self.rand_state.rand(candidate_ind.size)