import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from WSigKernel import optmeasure as om

# ========================n paths with m pieces in d dimension: Brownian motion+epsilon*sine ================== #

# the number of weights and cubature paths
n = 10
# number of piecewise linear pieces
m = 10
# the dimension of the underlying Brownian motion
d = 2

# d-dimensional Brownian motion
T = 1
h = T/(m-1)
X = np.sqrt(h)*np.random.randn(n, m, d)
for i in range(X.shape[0]):
    X[i, :, :] = np.cumsum(X[i, :, :], axis=0)
#     plt.plot(X[i, :, 0], X[i, :, 1])
# plt.show()

# sine wave
t = np.linspace(0, T, m)
freq = 2   # sine function has some error
Y = np.sin(2*np.pi*freq*t-np.random.rand(1))
plt.plot(Y)
plt.show()

epsilon = 0.1
for i in range(X.shape[0]):
    for j in range(X.shape[2]):
        Y = np.sin(2 * np.pi * freq * t - np.random.rand(1))
        X[i, :, j] = X[i, :, j]+epsilon*Y
    plt.plot(X[i, :, 0], X[i, :, 1])
plt.show()

mu, K, h = om.optimal_measure(X, solver=1, weight=1)

print('optimal discrete measure ', mu)

mu_lab = np.round(mu, 2)
for i in range(X.shape[0]):
    plt.plot(X[i, :, 0], X[i, :, 1], label=mu_lab[i])
plt.legend()
plt.show()

corr1 = om.correlation_wiener_measure(mu, K, h, d)
print('optimal correlation ', corr1)

delta = np.ones(n)/n
corr2 = om.correlation_wiener_measure(delta, K, h, d)
print('empirical correlation ', corr2)
print('correlation ratio ', corr1/corr2)

error1 = om.error_optmeas(mu, K, h, d)
print('optimal error ', error1)
error2 = om.error_optmeas(delta, K, h, d)
print('empirical error ', error2)
print('error ratio ', error1/error2)

kl_div = om.kl_divergence(mu, delta)
print('KL divergence ', kl_div)

js_div = om.js_divergence(mu, delta)
print('JS divergence ', js_div)





