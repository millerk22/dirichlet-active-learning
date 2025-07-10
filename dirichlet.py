import numpy as np 
import graphlearning as gl
import scipy.sparse as sparse


class dirichlet_learning(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, tau=0.0, epsK=None, seed=42):
        """Dirichlet Learning with Epsilon prior
        ===================

        Semi-supervised learning
        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        K : int, default=10
            Number of "known" clusters in the dataset. Parameter for choosing epsilon prior size
        """
        super().__init__(W, class_priors)
        self.tau = tau
        self.train_ind = np.array([])
        self.rand_state = np.random.RandomState(seed)
        
        
        # If have passed K value at this initialization, then can set epsilon prior accordingly
        if epsK is not None:
            # epsilon calculation
            MULT = 5
            num_eps_props = epsK * MULT
            rand_inds = self.rand_state.choice(self.graph.num_nodes, num_eps_props, replace=False)
            props = self.poisson_prop(rand_inds)
            props_to_inds = np.max(props, axis=1)
            epsilons = np.array([np.percentile(props[:,i], 100.*(epsK-1.)/epsK) for i in range(props.shape[1])])
            self.eps = np.max(epsilons)
        else:
            # prior is Beta(1,1,...,1)
            self.eps = 1.0
        
        # Setup accuracy filename
        fname = '_dir' 
        if self.tau > 0:
            fname += '_' + str(int(self.tau * 1e4)).zfill(4)
        self.name = f'Dirichlet Learning, tau = {self.tau}'

        self.accuracy_filename = fname
    
    def set_eps(self, K=10, verbose=False):
        if K is None:
            self.eps = 1.0
        else:
            # epsilon calculation -- Appendix E.1 of paper for explanation
            MULT = 5
            num_eps_props = K * MULT
            rand_inds = self.rand_state.choice(self.graph.num_nodes, num_eps_props, replace=False)
            props = self.poisson_prop(rand_inds)
            props_to_inds = np.max(props, axis=1)
            epsilons = np.array([np.percentile(props[:,i], 100.*(K-1.)/K) for i in range(props.shape[1])])
            self.eps = np.max(epsilons)
            
        if verbose:
            print(f"\tSetting eps = {self.eps} for Dirichlet Learning, tau = {self.tau}")
        return 
        

    def _fit(self, train_ind, train_labels, all_labels=None):
        # Not currently designed for repeated indices in train_ind
        if train_ind.size >= self.train_ind.size:
            mask = ~np.isin(train_ind, self.train_ind)
            prop_ind = train_ind[np.where(mask)[0]]
            prop_labels = train_labels[np.where(mask)[0]]
        else: # if give fewer training labels than before, we assume that this is a "new" instantiation
            prop_ind, prop_labels = train_ind, train_labels
            mask = np.ones(3, dtype=bool)
        self.train_ind = train_ind
        n, nc = self.graph.num_nodes, np.unique(train_labels).size
        
        P = self.poisson_prop(prop_ind)
        
        P /= P[prop_ind,np.arange(prop_ind.size)][np.newaxis,:] # scale by the value at the point sources

        if mask.all(): # prop_ind == train_ind, so all inds are "new"
            self.A = self.eps*np.ones((n, nc))  # Dir(1,1,1,...,1) prior on each node

        # Add propagations according to class for the propagation inds (prop_inds)
        for c in np.unique(prop_labels):
            self.A[:, c] += np.sum(P[:,np.where(prop_labels == c)[0]], axis=1) # sum propagations together according to class
        

        u = self.A / (self.A.sum(axis=1)[:,np.newaxis]) # mean estimator
        return u
    
    def poisson_prop(self, inds):
        # Poisson propagation
        n, num_prop = self.graph.num_nodes, inds.size
        F = np.zeros((n, num_prop))
        F[inds,:] = np.eye(num_prop)
        F -= np.mean(F, axis=0)

        L = self.graph.laplacian()
        if self.tau  > 0.0:
            L += self.tau*sparse.eye(L.shape[0])

        prop = gl.utils.conjgrad(L, F, tol=1e-9)
        prop -= np.min(prop, axis=0)
        return prop
