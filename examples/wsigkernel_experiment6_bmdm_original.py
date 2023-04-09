import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from WSigKernel import optmeasure as om

# ========================n paths with m pieces in d dimension: Brownian motion===================== #

# the number of weights and cubature paths
n = 10
# number of piecewise linear pieces
m = 10
# the dimension of the underlying Brownian motion
d = 2

data = []
a = np.linspace(0, 5, 50)
for epsilon in a[0:36]:
    for p in range(100):
        print([n, m, d, epsilon, p])
        # d-dimensional Brownian motion
        T = 1
        h = T / (m - 1)
        X = np.sqrt(h) * np.random.randn(n, m, d)
        for i in range(X.shape[0]):
            X[i, :, :] = np.cumsum(X[i, :, :], axis=0)

        # epsilon*sqrt((t-U)+)
        t = np.linspace(0, T, m)

        for i in range(X.shape[0]):
            U = np.random.rand(1)
            Z = np.where(t - U > 0, t - U, 0)
            Y = np.sqrt(Z)
            for j in range(X.shape[2]):
                X[i, :, j] = X[i, :, j] + epsilon * Y

        mu, K, h = om.optimal_measure_original(X, solver=1, N_contour=32, contour='cotangent')
        delta = np.ones(n)/n

        try:
            corr1 = om.correlation_wiener_measure_original(mu, K, h, d, N_contour=32, contour='cotangent')
            corr2 = om.correlation_wiener_measure_original(delta, K, h, d, N_contour=32, contour='cotangent')
            corr_ratio = corr1/corr2

            loss1 = om.error_optmeas_original(mu, K, h, d, N_contour=32, contour='cotangent')
            loss2 = om.error_optmeas_original(delta, K, h, d, N_contour=32, contour='cotangent')
            loss_ratio = loss1/loss2

            kl_div = om.kl_divergence(mu, delta)
            js_div = om.js_divergence(mu, delta)

            data_res = np.array([n, m, d, epsilon, corr1, corr2, corr_ratio, loss1, loss2, loss_ratio, kl_div, js_div])
            data = np.concatenate((data, data_res), axis=0)
        except:
            continue


data = data.reshape((-1, 12))
data = pd.DataFrame(data, columns=['n', 'm', 'd', 'epsilon', 'corr1', 'corr2', 'corr_ratio', 'loss1', 'loss2', 'loss_ratio', 'kl_div', 'js_div'])

data.to_excel('signature_kernel_result_t11.xlsx', index=False)





