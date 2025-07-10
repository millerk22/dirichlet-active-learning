# Dirichlet Active Learning 

This repository contains the code to reproduce the numerical experiments from our paper, Dirichlet Active Learning (see [arXiv link](https://arxiv.org/abs/2311.05501)) by Kevin Miller and Ryan Murray. 

## Requirements

See the ``requirements.txt`` file for a list of Python packages (and versions) of some needed packages to run this repository.

## Running Experiments

We have provided the ``run.sh`` bash script that calls the main Python files to run active learning tests in parallel and evaluate the accuracy performances. The main files are as follows:
* ``config.yaml``: config file the specifies the acquisition functions, random seeds, and graph-based SSL models in which to evaluate the accuracy of each trial.
* ``test_al_gl.py``: main test driver coordinating the tests  specified in the ``config.yaml`` file. Example usage: 

    ``python test_al_gl.py --config config.yaml --dataset mnist --metric raw --resultsdir results``
* ``accuracy_al_gl.py``: once the active learning tests have been run via ``test_al_gl.py``, this script evaluates all the sequences of labeled nodes in the specified graph-based SSL classifiers. For example, an acquisition function might use the classifier outputs of Laplace Learning (Zhu, Gharahmani, Lafferty 2003), but in order to standardize the comparison, we evaluate the accuracy in our Dirichlet Learning classifier. 
* ``compile_summary.py``: this simply reads all of the results in the corresponding experiment's results directory and compiles them into csv file for later plotting and assessment. 

Overall, as you can see in the ``run.sh`` file, the usage block for a dataset's tests look something like:
```
# run the different acquisition functions on "paviasub" dataset
python test_al_gl.py --dataset paviasub --metric hsi --config config.yaml --resultsdir results 
# compute accuracies for previously run acquisition functions
python accuracy_al_gl.py --dataset paviasub --metric hsi --config config.yaml --resultsdir results
# compile results for further analysis/plotting
python compile_summary.py --dataset paviasub --resultsdir results
```

## Plotting

See the ``Plotting-Dirichlet-Learning_Paper.ipynb`` Jupyter notebook (after running ``compile_summary.py``) for reproducing the plots and figures from our numerical experiments of our paper. 

