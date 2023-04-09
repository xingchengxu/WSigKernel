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

_kernels = [
    'original_signature'
]

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

    # ==================================================================================================
    # Training phase
    # ==================================================================================================

    # lead-lag only if number of channels is <= 5
    x_train, _, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    print(x_train.shape)

    if x_train.shape[1] <= 200 and x_train.shape[2] <= 8:
        transforms = tqdm([(True, True), (False, True), (True, False), (False, False)], position=1, leave=False)
    else:  # do not try lead-lag as dimension is already high
        transforms = tqdm([(True, False), (False, False)], position=1, leave=False)

    # grid-search for path-transforms (add-time, lead-lag)
    # at=add-time, ll=lead-lag
    for (at, ll) in transforms:
        transforms.set_description(f"add-time: {at}, lead-lag: {ll}")

        # load train data
        x_train, y_train, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
        x_train /= x_train.max()

        # encode outputs as labels
        y_train = LabelEncoder().fit_transform(y_train)

        # path-transform
        x_train = paths_transform.transform(x_train, at=at, ll=ll, scale=.1)

        # subsample every time steps if certain length is exceeded
        subsample = max(int(np.floor(x_train.shape[1] / 149)), 1)
        x_train = x_train[:, ::subsample, :]
        datasets.set_description(f"dataset: {name} --- shape: {x_train.shape}")

        # ==================================================================================
        # Full Signature Kernel
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
        svc_model = GridSearchCV(estimator=svc, param_grid=svc_parameters, cv=5, n_jobs=-1)
        svc_model.fit(G_train, y_train)

        # empty memory
        del G_train
        torch.cuda.empty_cache()

        ker = _kernels[0]
        trained_models[(name, ker, at, ll)] = (subsample, at, ll, svc_model)

        # save trained models
        with open('trained_models_orig.pkl', 'wb') as file:
            pickle.dump(trained_models, file)

        # ==================================================================================================
        # Testing phase
        # ==================================================================================================

        # load final results from last run
        try:
            with open('final_results_orig.pkl', 'rb') as file:
                final_results = pickle.load(file)
        except:
            final_results = {}

        x_train, y_train, x_test, y_test = UCR_UEA_datasets(use_cache=True).load_dataset(name)
        x_train /= x_train.max()
        x_test /= x_test.max()
        print(x_train.shape)

        # encode outputs as labels
        y_test = LabelEncoder().fit_transform(y_test)

        # extract information from training phase
        # at=add-time, ll=lead-lag
        subsample, at, ll, estimator = trained_models[(name, ker, at, ll)]

        # path-transform and subsampling
        x_train = paths_transform.transform(x_train, at=at, ll=ll, scale=.1)[:, ::subsample, :]
        x_test = paths_transform.transform(x_test, at=at, ll=ll, scale=.1)[:, ::subsample, :]

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
        final_results[(name, ker, at, ll)] = {f'training accuracy: {train_score}',
                                      f'testing accuracy: {test_score}'}

        # empty memory
        del G_test
        torch.cuda.empty_cache()

        sleep(0.5)

        print(name, ker, at, ll, final_results[name, ker, at, ll])

        print('\n')

        # save results
        with open('final_results_orig.pkl', 'wb') as file:
            pickle.dump(final_results, file)

with open('final_results_orig.pkl', 'rb') as file:
    final = pickle.load(file)
print(final)






