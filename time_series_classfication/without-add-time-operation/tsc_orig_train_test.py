# University of East Anglia (UEA) time series datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
from time import sleep

from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from WSigKernel import wsigkernel
from WSigKernel import paths_transform

_datasets = [
    'ArticularyWordRecognition',
    'BasicMotions',
    'Cricket',
    'ERing',
    'Libras',
    'NATOPS',
    'RacketSports',
    'FingerMovements',
    'Heartbeat',
    'SelfRegulationSCP1',
    'UWaveGestureLibrary'
]

print(_datasets)

# _kernels = [
#     'original_signature',
#     'factorial_signature',
#     'beta_signature'
# ]

_kernels = [
    'original_signature'
]

# ==================================================================================================
# Training phase
# ==================================================================================================

# store best models in training phase
try:
    with open('trained_models_orig.pkl', 'rb') as file:
        trained_models = pickle.load(file)
except:
    trained_models = {}

# define grid-search hyperparameters for SVC (common to all kernels)
svc_parameters = {'C': np.logspace(0, 4, 5), 'gamma': list(np.logspace(-4, 4, 9)) + ['auto']}

# start grid-search
datasets = tqdm(_datasets, position=0, leave=True)
for name in datasets:

    # record best scores in training phase
    best_scores_train = {k: 0. for k in _kernels}

    transforms = tqdm([(False, False)], position=1, leave=False)

    # grid-search for path-transforms (add-time, lead-lag)
    for (at, ll) in transforms:
        transforms.set_description(f"add-time: {at}, lead-lag: {ll}")

        # load train data
        x_train, y_train, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
        x_train /= x_train.max()

        # encode outputs as labels
        y_train = LabelEncoder().fit_transform(y_train)

        # path-transform
        # x_train = paths_transform.transform(x_train, at=at, ll=ll, scale=.1)

        # subsample every time steps if certain length is exceeded
        subsample = max(int(np.floor(x_train.shape[1] / 149)), 1)
        x_train = x_train[:, ::subsample, :]
        datasets.set_description(f"dataset: {name} --- shape: {x_train.shape}")

        # ==================================================================================
        # Signature PDE kernel
        # ==================================================================================
        # move to cuda (if available and memory doesn't exceed a certain threshold)
        if x_train.shape[0] <= 150 and x_train.shape[1] <= 150 and x_train.shape[2] <= 8:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32
        else:  # otherwise do computations in cython
            device = 'cpu'
            dtype = torch.float64

        # numpy -> torch
        x_train = torch.tensor(x_train, dtype=dtype, device=device).numpy()
        print(x_train.shape)

        # compute Gram matrix on train data
        G_train = wsigkernel.sig_kernel_matrix(x_train, x_train, n=0, solver=0, sym=True, rbf=False)

        # SVC sklearn estimator
        svc = SVC(kernel='precomputed', decision_function_shape='ovo')
        svc_model = GridSearchCV(estimator=svc, param_grid=svc_parameters, cv=3, n_jobs=-1)
        svc_model.fit(G_train, y_train)

        # empty memory
        del G_train
        torch.cuda.empty_cache()

        # store results
        ker = _kernels[0]
        if svc_model.best_score_ > best_scores_train[ker]:
            best_scores_train[ker] = svc_model.best_score_
            trained_models[(name, ker)] = (subsample, at, ll, svc_model)

        sleep(0.5)

    # save trained models
    with open('trained_models_orig.pkl', 'wb') as file:
        pickle.dump(trained_models, file)

# ==================================================================================================
# Testing phase
# ==================================================================================================

# load trained models
try:
    with open('trained_models_orig.pkl', 'rb') as file:
        trained_models = pickle.load(file)
except:
    print('Models need to be trained first')

# load final results from last run
try:
    with open('final_results_orig.pkl', 'rb') as file:
        final_results = pickle.load(file)
except:
    final_results = {}

for name in _datasets:
    for ker in _kernels:

        # load test data
        x_train, y_train, x_test, y_test = UCR_UEA_datasets(use_cache=True).load_dataset(name)
        x_train /= x_train.max()
        x_test /= x_test.max()
        print(x_train.shape)

        # encode outputs as labels
        y_test = LabelEncoder().fit_transform(y_test)

        # extract information from training phase
        subsample, at, ll, estimator = trained_models[(name, ker)]

        # path-transform and subsampling
        x_train = x_train[:, ::subsample, :]
        x_test = x_test[:, ::subsample, :]

        # move to cuda (if available and memory doesn't exceed a certain threshold)
        if x_test.shape[0] <= 150 and x_test.shape[1] <= 150 and x_test.shape[2] <= 10:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32
        else:  # otherwise do computations in cython
            device = 'cpu'
            dtype = torch.float64

        # numpy -> torch
        x_train = torch.tensor(x_train, dtype=dtype, device=device).numpy()
        x_test = torch.tensor(x_test, dtype=dtype, device=device).numpy()

        # compute Gram matrix on test data
        G_test = wsigkernel.sig_kernel_matrix(x_test, x_train, n=0, solver=0, sym=False, rbf=False)

        # record scores
        train_score = estimator.best_score_
        test_score = estimator.score(G_test, y_test)
        final_results[(name, ker)] = {f'training accuracy: {train_score}',
                                      f'testing accuracy: {test_score}'}

        # empty memory
        del G_test
        torch.cuda.empty_cache()

        sleep(0.5)

        print(name, ker, final_results[name, ker])

        # save results
        with open('final_results_orig.pkl', 'wb') as file:
            pickle.dump(final_results, file)

    print('\n')


with open('final_results_orig.pkl', 'rb') as file:
    final = pickle.load(file)
print(final)






