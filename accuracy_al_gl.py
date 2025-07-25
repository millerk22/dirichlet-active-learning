import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
import yaml
from copy import deepcopy
from glob import glob
from scipy.special import softmax
from functools import reduce
from utils import *

from joblib import Parallel, delayed



if __name__ == "__main__":
    parser = ArgumentParser(description="Compute Accuracies in Parallel of Active Learning Tests for Graph Learning")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=5)
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--knn", type=int, default=0)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    G, labels, trainset, normalization = load_graph(args.dataset, args.metric, numeigs=None) # don't compute any eigenvalues
    model_names = [name for name in config["acc_models"] if name[:3] != "gcn"]
    models = get_models(G, model_names)
    models_dict = {name:model for name, model in zip(model_names, models)}
    results_directories = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}/"))
    acqs_models = config["acqs_models"]
    
    


    for out_num, RESULTS_DIR in enumerate(results_directories):
        choices_fnames = glob(os.path.join(RESULTS_DIR, "choices_*.npy"))
        choices_fnames = [fname for fname in choices_fnames if " ".join(fname.split("_")[-2:]).split(".")[0] in acqs_models ]
        labeled_ind = np.load(os.path.join(RESULTS_DIR, "init_labeled.npy")) # initially labeled points that are common to all acq_func:gbssl modelname pairs
        
        for num, acc_model_name in enumerate(models_dict.keys()):
            acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)

            def compute_accuracies(choices_fname):
                # get acquisition function - gbssl modelname that made this sequence of choices
                acq_func_name, modelname = choices_fname.split("_")[-2:]
                modelname = modelname.split(".")[0]

                # load in the indices of the choices
                choices = np.load(choices_fname)

                # define the filepath of where the results of evaluating this acq_func:modelname combo had in acc_model_name
                acc_fname = os.path.join(acc_dir, f"acc_{acq_func_name}_{modelname}.npy")
                if os.path.exists(acc_fname):
                    print(f"Already computed accuracies in {acc_model_name} for {acc_fname}")
                    return
                else:
                    print(f"Computing accuracies in {acc_model_name} for {acq_func_name} in {modelname}")

                # get copy of model on this cpu
                model = deepcopy(models_dict[acc_model_name])
                
                # Compute accuracies at each sequential subset of choices
                acc = np.array([])
                for j in tqdm(range(labeled_ind.size,choices.size+1), desc=f"Computing Acc of {acq_func_name}-{modelname}"):
                    train_ind = choices[:j]
                    u = model.fit(train_ind, labels[train_ind])
                    acc = np.append(acc, gl.ssl.ssl_accuracy(model.predict(), labels, train_ind))

                # save accuracy results to corresponding filename
                np.save(acc_fname, acc)
                return

            print(f"-------- Computing Accuracies in {acc_model_name}, {num+1}/{len(models_dict)} in {RESULTS_DIR} ({out_num+1}/{len(results_directories)}) -------")

            Parallel(n_jobs=args.numcores)(delayed(compute_accuracies)(choices_fname) for choices_fname \
                    in choices_fnames)
            print()

        # Consolidate results
        print(f"Consolidating accuracy results of run in: {os.path.join(RESULTS_DIR)}...")
        for acc_model_name in models_dict.keys():
            acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
            accs_fnames = glob(os.path.join(acc_dir, "acc_*.npy"))
            columns = {}
            max_length = 0
            for fname in accs_fnames:
                acc = np.load(fname)
                acq_func_name, modelname = fname.split("_")[-2:]
                modelname = modelname.split(".")[0]
                columns[acq_func_name + " : " + modelname] = acc
                if acc.size > max_length:
                    max_length = acc.size
            
            for k, col in columns.items():
                if col.size < max_length:
                    print(f"found col = {k} of too short lenghth, padding with nans")
                    columns[k] = np.concatenate((col, np.full(max_length - col.size, fill_value=np.nan)))
            
            acc_df = pd.DataFrame(columns)
            acc_df.to_csv(os.path.join(acc_dir, "accs.csv"), index=None)

        print("-"*40)
        print("-"*40)
    print()
