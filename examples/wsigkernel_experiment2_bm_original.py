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
# for n in [5, 10, 15, 20, 25]:
for d in [2, 3, 4, 5, 6]:
    for p in range(200):
        print(d, p)
        # d-dimensional Brownian motion
        T = 1
        h = T / (m - 1)
        X = np.sqrt(h) * np.random.randn(n, m, d)
        # X = np.random.randn(n, m, d)
        for i in range(X.shape[0]):
            X[i, :, :] = np.cumsum(X[i, :, :], axis=0)

        mu, K, h = om.optimal_measure_original(X, solver=1, N_contour=32, contour='cotangent')
        delta = np.ones(n)/n

        corr1 = om.correlation_wiener_measure_original(mu, K, h, d, N_contour=32, contour='cotangent')
        corr2 = om.correlation_wiener_measure_original(delta, K, h, d, N_contour=32, contour='cotangent')
        corr_ratio = corr1/corr2

        loss1 = om.error_optmeas_original(mu, K, h, d, N_contour=32, contour='cotangent')
        loss2 = om.error_optmeas_original(delta, K, h, d, N_contour=32, contour='cotangent')
        loss_ratio = loss1/loss2

        kl_div = om.kl_divergence(mu, delta)
        js_div = om.js_divergence(mu, delta)

        data_res = np.array([n, m, d, corr1, corr2, corr_ratio, loss1, loss2, loss_ratio, kl_div, js_div])
        data = np.concatenate((data, data_res), axis=0)
        print([n, m, d, p])

data = data.reshape((-1, 11))
data = pd.DataFrame(data, columns=['n', 'm', 'd', 'corr1', 'corr2', 'corr_ratio', 'loss1', 'loss2', 'loss_ratio', 'kl_div', 'js_div'])

data.to_excel('signature_kernel_result_t1.xlsx', index=False)





