import graphlearning as gl
import os
import numpy as np
import scipy.sparse as sparse
from copy import deepcopy
import acquisitions
from dirichlet import dirichlet_learning



def get_models(G, model_names):
    MODELS = {'poisson': gl.ssl.poisson(G),  # poisson learning
              'laplace': gl.ssl.laplace(G), # laplace learning
              'rwll1000': gl.ssl.laplace(G, tau=0.1, reweighting='poisson'),  # poisson-reweighted laplace learning
              'dirichlet1000' : dirichlet_learning(G, tau=0.1),
              'dirichlet0100': dirichlet_learning(G, tau=0.01),
              'dirichlet0010': dirichlet_learning(G, tau=0.001),
              'dirichlet0001': dirichlet_learning(G, tau=0.0001),
              }

    return [deepcopy(MODELS[name]) for name in model_names]


def load_graph(dataset, metric, numeigs=200, data_dir="data", returnX=False, returnK=False):
    X, clusters = gl.datasets.load(dataset.split("-")[0], metric=metric)
    if dataset.split("-")[-1] == 'evenodd':
        labels = clusters % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = clusters % modnum
    else:
        labels = clusters

    if metric == "hsi":
        trainset = np.where(labels != 0)[0]
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset}")
    knn = 20
    if dataset in ['box', 'blobs']:
        print(f"knn = 100, {dataset}")
        knn = 100
    
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["mnist", "fashionmnist", "cifar", "emnist", "mnistsmall", "fashionmnistsmall", "salinassub", "paviasub", "mnistimb", "fashionmnistimb", "emnistvcd"]:
        normalization = "normalized"
    if labels.size < 100000:
        method = "exact"
    
    print(f"Eigendata calculation will be {method}")

    try:
        G = gl.graph.load(graph_filename)
        found = True
    except:
        if metric == "hsi":
            sim_name ="angular" # LAND does 100 in HSI
        else:
            sim_name = "euclidean"
        knn_ind, knn_dist = gl.weightmatrix.knnsearch(X, knn, similarity=sim_name, metric=metric, dataset=dataset.split("-")[0])
        W = gl.weightmatrix.knn(X, knn, knn_data=(knn_ind, knn_dist), metric=metric)
        G = gl.graph(W)
        found = False

    if numeigs is not None:
        eigdata = G.eigendata[normalization]['eigenvalues']
        if eigdata is not None:
            prev_numeigs = eigdata.size
            if prev_numeigs >= numeigs:
                print("Retrieving Eigendata...")
            else:
                print(f"Requested {numeigs} eigenvalues, but have only {prev_numeigs} stored. Recomputing Eigendata...")
        else:
            print(f"No Eigendata found, so computing {numeigs} eigenvectors...")

        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method)


    G.save(graph_filename)
    
    if returnX:
        return G, labels, trainset, normalization, X
    
    if returnK:
        return G, labels, trainset, normalization, np.unique(clusters).size
    
    return G, labels, trainset, normalization


def get_eig_data(G, normalization, numeigs):
    # determine if need to recompute eigenvalues/vectors
    recompute = True
    if G.eigendata[normalization]['eigenvalues'] is not None:
        if G.eigendata[normalization]['eigenvalues'].size < numeigs:
            recompute = True
    else:
        recompute = True
        
    if not recompute:
        # Current gl.active_learning is implemented only to allow for exact eigendata compute for "normalized"
        print(f"Using previously stored {normalization} eigendata with {numeigs} evals")
        evals = G.eigendata[normalization]['eigenvalues'][:numeigs]
        evecs = G.eigendata[normalization]['eigenvectors'][:,:numeigs] 
        
    else:
        print("Warning: Computing eigendata with gl.active_learning defaults...")
        evals, evecs = G.eigen_decomp(normalization, k=numeigs)

    return evals, evecs


def get_unc_acq_func(af_name):
    unc_method = None
    if af_name == "random":
        acq_func = acquisitions.random
    elif af_name in ["uncnorm"]:
        acq_func = gl.active_learning.unc_sampling
        unc_method = "unc_2norm"
    elif af_name == "unc":
        acq_func = gl.active_learning.unc_sampling
        unc_method = "smallest_margin"
    else:
        raise NotImplementedError(f"Acquisition function = {af_name} not implemented...")

    return acq_func, unc_method
    
def get_active_learner(acq_func_name, model, labeled_ind, labeled_ind_labels, normalization, args, numeigs=100):
    """
        Based on the acquisition function name, determine if need to compute the covariance matrix for instantiating 
        active_learner object
    """
    if len(acq_func_name.split("-")) > 1:
        numeigs = int(acq_func_name.split("-")[-1])
        
    af_name = acq_func_name.split("-")[0]

    if af_name == "dirichletvar":
        AL = gl.active_learning.active_learner(model, acquisitions.dirichlet_var, labeled_ind.copy(), labeled_ind_labels.copy())
    elif af_name == "dirichletvarprop":
        AL = gl.active_learning.active_learner(model, acquisitions.dirichlet_varprop, labeled_ind.copy(), labeled_ind_labels.copy())
    elif af_name in ["mc", "mcvopt", "vopt", "sopt"]:
        print(f"gamma = {args.gamma}")
        evals, V = get_eig_data(model.graph, normalization, numeigs)
        if acq_func_name.split("-")[0][-1] == "1":
            evals = evals[1:] 
            V = V[:,1:]
        C = np.linalg.inv(np.diag(evals + 1e-11))

        if af_name == "mc":
            acq_func = gl.active_learning.model_change
        elif af_name == "mcvopt":
            acq_func = gl.active_learning.model_change_var_opt
        elif af_name in "vopt":
            acq_func = gl.active_learning.var_opt
        elif af_name == "sopt":
            acq_func = gl.active_learning.sigma_opt

        AL = gl.active_learning.active_learner(model, acq_func, labeled_ind.copy(), labeled_ind_labels.copy(), C=C.copy(), V=V.copy(), gamma2=args.gamma**2.)
        
    elif af_name in ["voptfull", "soptfull"]:
        # add a small diagonal term to make C invertible
        C = sparse.linalg.inv(sparse.csc_matrix(model.graph.laplacian(normalization=normalization) + 
                                                                       0.001*sparse.eye(model.graph.num_nodes))).toarray() 
        if af_name == "voptfull":
            acq_func = gl.active_learning.var_opt
        else:
            acq_func = gl.active_learning.sigma_opt
            
        AL = gl.active_learning.active_learner(model, acq_func, labeled_ind.copy(), labeled_ind_labels.copy(), C=C.copy(), gamma2=args.gamma**2.)
    
    else:
        acq_func, unc_method = get_unc_acq_func(af_name)
        AL = gl.active_learning.active_learner(model, acq_func, labeled_ind.copy(), labeled_ind_labels.copy())
    
    return AL





def get_graph_and_models(acq_funcs_names, model_names, args):
    # Determine if we need to calculate more eigenvectors/values for mc, vopt, mcvopt acquisitions
    maxnumeigs = 0
    for acq_func_name in acq_funcs_names:
        if len(acq_func_name.split("-")) == 1:
            continue
        d = acq_func_name.split("-")[-1]
        if len(d) > 0:
            if maxnumeigs < int(d):
                maxnumeigs = int(d)
    if maxnumeigs == 0:
        maxnumeigs = None

    # Load in the graph and labels
    print("Loading in Graph...")
    G, labels, trainset, normalization, K = load_graph(args.dataset, args.metric, maxnumeigs, returnK=True)
    
    models = get_models(G, model_names)
    
    return models, labels, trainset, normalization,  K
