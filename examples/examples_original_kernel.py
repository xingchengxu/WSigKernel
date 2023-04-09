import numpy as np
import matplotlib.pyplot as plt


# ========================n paths with m pieces in d dimension: Brownian motion===================== #

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
    plt.plot(X[i, :, 0], X[i, :, 1])
# plt.show()
plt.close()


from WSigKernel.hyperdevelop_explicit_original import sigkernel_with_bm_original

h_normal = sigkernel_with_bm_original(X, N=32, contour='parabola')
print(h_normal)

h_normal = sigkernel_with_bm_original(X, N=32, contour='hyperbola')
print(h_normal)

h_normal = sigkernel_with_bm_original(X, N=32, contour='cotangent')
print(h_normal)


# norm of BM under original inner product
from WSigKernel.hyperdevelop_explicit_original import norm_of_bm_original

d = 2
norm_bm = norm_of_bm_original(d, N=32, contour='parabola')
print(norm_bm)
norm_bm = norm_of_bm_original(d, N=32, contour='hyperbola')
print(norm_bm)
norm_bm = norm_of_bm_original(d, N=32, contour='cotangent')
print(norm_bm)


from WSigKernel import optmeasure as om
mu, K, h = om.optimal_measure_original(X, solver=1, N_contour=32, contour='cotangent')

print('optimal discrete measure ', mu)

mu_lab = np.round(mu, 2)
for i in range(X.shape[0]):
    plt.plot(X[i, :, 0], X[i, :, 1], label=mu_lab[i])
plt.legend()
# plt.show()
plt.close()

corr1 = om.correlation_wiener_measure_original(mu, K, h, d, N_contour=32, contour='cotangent')
print('optimal correlation ', corr1)

delta = np.ones(n)/n
corr2 = om.correlation_wiener_measure_original(delta, K, h, d, N_contour=32, contour='cotangent')
print('empirical correlation ', corr2)
print('correlation ratio ', corr1/corr2)

error1 = om.error_optmeas_original(mu, K, h, d, N_contour=32, contour='cotangent')
print('optimal error ', error1)
error2 = om.error_optmeas_original(delta, K, h, d, N_contour=32, contour='cotangent')
print('empirical error ', error2)
print('error ratio ', error1/error2)


