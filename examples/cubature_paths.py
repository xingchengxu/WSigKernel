import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from WSigKernel import optmeasure as om


# cubature on Wiener space, the cubature paths with degree 5 and dimension d=2
a = np.array([0, 1, 2, 3, 4])
theta1 = np.array([0, 0.0480779, 0.0893357, 1.1770950, -0.3145086])
theta1 = a*theta1
w1 = np.cumsum(theta1, axis=0)

theta2 = np.array([0, 1.0123685, -1.6196048, 1.7021040, -0.0948677])
theta2 = a*theta2
w2 = np.cumsum(theta2, axis=0)

w1w2 = np.concatenate((w1, w2), axis=0).reshape((2, 5)).T
w2w1 = np.concatenate((w2, w1), axis=0).reshape((2, 5)).T

coff1 = np.array([[0, 0],
                 [1, np.sqrt(3)], [-1, np.sqrt(3)], [1, -1*np.sqrt(3)], [-1, -1*np.sqrt(3)],
                 [1, np.sqrt(3)], [-1, np.sqrt(3)], [1, -1*np.sqrt(3)], [-1, -1*np.sqrt(3)],
                 [2, 0], [-2, 0], [0, 2], [0, -2]]
                )

coff = np.zeros((coff1.shape[0], 2, 2))
for i in range(coff1.shape[0]):
    coff[i, :, :] = np.diag(coff1[i, :])

# the cubature paths
n, m, d = 13, 5, 2
X = np.zeros((n, m, d))
for i in [0, 1, 2, 3, 4, 9, 10]:
    X[i, :, :] = np.matmul(w1w2, coff[i, :, :])
for i in [5, 6, 7, 8, 11, 12]:
    X[i, :, :] = np.matmul(w2w1, coff[i, :, :])
# print(X)

# cabuture weights
mu_cub = np.concatenate((np.array([1/2]), 1/24*np.array(np.ones(12))))
# print(mu_cub)

delta = np.ones(n)/n

mu, K, h = om.optimal_measure(X, n=0, solver=0, weight=1)

print('cabuture discrete measure ', mu_cub)
print('optimal discrete measure ', mu)
print('empirical discrete measure ', delta)

mu_lab = np.round(mu, decimals=5)
for i in range(X.shape[0]):
    plt.plot(X[i, :, 0], X[i, :, 1], label=str(mu_lab[i]))
plt.legend()
# plt.show()
plt.close()

meas = np.concatenate((mu, mu_cub, delta)).reshape(3, -1)
meas = pd.DataFrame(meas, index=['Optimal', 'Cubature', 'Empirical'])
print(meas)
# meas.to_excel('measure.xlsx')


corr0 = om.correlation_wiener_measure(mu_cub, K, h, d)
print('cubature correlation ', corr0)

corr1 = om.correlation_wiener_measure(mu, K, h, d)
print('optimal correlation ', corr1)

corr2 = om.correlation_wiener_measure(delta, K, h, d)
print('empirical correlation ', corr2)
print('correlation ratio (optimal/cubature) ', corr1/corr0)
print('correlation ratio (optimal/empirical) ', corr1/corr2)

error0 = om.error_optmeas(mu_cub, K, h, d)
print('cubature error ', error0)
error1 = om.error_optmeas(mu, K, h, d)
print('optimal error ', error1)
error2 = om.error_optmeas(delta, K, h, d)
print('empirical error ', error2)
print('error ratio (optimal/cubature) ', error1/error0)
print('error ratio (optimal/empirical) ', error1/error2)

kl_div = om.kl_divergence(mu, mu_cub)
print('KL divergence (optimal, cubature) ', kl_div)
kl_div = om.kl_divergence(mu, delta)
print('KL divergence (optimal, empirical) ', kl_div)

js_div = om.js_divergence(mu, mu_cub)
print('JS divergence (optimal, cubature) ', js_div)
js_div = om.js_divergence(mu, delta)
print('JS divergence (optimal, empirical) ', js_div)


# ================== original kernel ============================ #

from WSigKernel import optmeasure as om
mu, K, h = om.optimal_measure_original(X, solver=0, N_contour=32, contour='cotangent')

print('cabuture discrete measure ', mu_cub)
print('optimal discrete measure ', mu)
print('empirical discrete measure ', delta)

mu_lab = np.round(mu, decimals=5)
for i in range(X.shape[0]):
    plt.plot(X[i, :, 0], X[i, :, 1], label=str(mu_lab[i]))
# plt.legend()
# plt.show()
plt.close()

meas = np.concatenate((mu, mu_cub, delta)).reshape(3, -1)
meas = pd.DataFrame(meas, index=['Optimal', 'Cubature', 'Empirical'])
print(meas)
# meas.to_excel('measure.xlsx')


corr0 = om.correlation_wiener_measure_original(mu_cub, K, h, d, N_contour=32, contour='cotangent')
print('cubature correlation ', corr0)

corr1 = om.correlation_wiener_measure_original(mu, K, h, d, N_contour=32, contour='cotangent')
print('optimal correlation ', corr1)

corr2 = om.correlation_wiener_measure_original(delta, K, h, d, N_contour=32, contour='cotangent')
print('empirical correlation ', corr2)
print('correlation ratio (optimal/cubature) ', corr1/corr0)
print('correlation ratio (optimal/empirical) ', corr1/corr2)

error0 = om.error_optmeas_original(mu_cub, K, h, d, N_contour=32, contour='cotangent')
print('cubature error ', error0)
error1 = om.error_optmeas_original(mu, K, h, d, N_contour=32, contour='cotangent')
print('optimal error ', error1)
error2 = om.error_optmeas_original(delta, K, h, d, N_contour=32, contour='cotangent')
print('empirical error ', error2)
print('error ratio (optimal/cubature) ', error1/error0)
print('error ratio (optimal/empirical) ', error1/error2)


# ================== beta kernel ============================ #
# m = 0.001
# m = 17.5
m = 10
from WSigKernel import optmeasure as om
mu, K, h = om.optimal_measure_beta(X, solver=0, N_contour=32, contour='cotangent', theta=1., m=m, N_xp=201)

print('cabuture discrete measure ', mu_cub)
print('optimal discrete measure ', mu)
print('empirical discrete measure ', delta)

mu_lab = np.round(mu, decimals=5)
for i in range(X.shape[0]):
    plt.plot(X[i, :, 0], X[i, :, 1], label=str(mu_lab[i]))
# plt.legend()
# plt.show()
plt.close()

meas = np.concatenate((mu, mu_cub, delta)).reshape(3, -1)
meas = pd.DataFrame(meas, index=['Optimal', 'Cubature', 'Empirical'])
print(meas)
# meas.to_excel('measure.xlsx')


corr0 = om.correlation_wiener_measure_beta(mu_cub, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
print('cubature correlation ', corr0)

corr1 = om.correlation_wiener_measure_beta(mu, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
print('optimal correlation ', corr1)

corr2 = om.correlation_wiener_measure_beta(delta, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
print('empirical correlation ', corr2)
print('correlation ratio (optimal/cubature) ', corr1/corr0)
print('correlation ratio (optimal/empirical) ', corr1/corr2)

error0 = om.error_optmeas_beta(mu_cub, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
print('cubature error ', error0)
error1 = om.error_optmeas_beta(mu, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
print('optimal error ', error1)
error2 = om.error_optmeas_beta(delta, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
print('empirical error ', error2)
print('error ratio (optimal/cubature) ', error1/error0)
print('error ratio (optimal/empirical) ', error1/error2)


# # ================== beta kernel (times Gamma(m+1) version)============================ #
# # m = 0.001
# # m = 17.5
# m = 20
# from WSigKernel import optmeasure as om
# mu, K, h = om.optimal_measure_beta_gmver(X, solver=0, N_contour=32, contour='cotangent', theta=1., m=m, N_xp=201)
#
# print('cabuture discrete measure ', mu_cub)
# print('optimal discrete measure ', mu)
# print('empirical discrete measure ', delta)
#
# mu_lab = np.round(mu, decimals=5)
# for i in range(X.shape[0]):
#     plt.plot(X[i, :, 0], X[i, :, 1], label=str(mu_lab[i]))
# # plt.legend()
# # plt.show()
# plt.close()
#
# meas = np.concatenate((mu, mu_cub, delta)).reshape(3, -1)
# meas = pd.DataFrame(meas, index=['Optimal', 'Cubature', 'Empirical'])
# print(meas)
# # meas.to_excel('measure.xlsx')
#
#
# corr0 = om.correlation_wiener_measure_beta_gmver(mu_cub, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
# print('cubature correlation ', corr0)
#
# corr1 = om.correlation_wiener_measure_beta_gmver(mu, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
# print('optimal correlation ', corr1)
#
# corr2 = om.correlation_wiener_measure_beta_gmver(delta, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
# print('empirical correlation ', corr2)
# print('correlation ratio (optimal/cubature) ', corr1/corr0)
# print('correlation ratio (optimal/empirical) ', corr1/corr2)
#
# error0 = om.error_optmeas_beta_gmver(mu_cub, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
# print('cubature error ', error0)
# error1 = om.error_optmeas_beta_gmver(mu, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
# print('optimal error ', error1)
# error2 = om.error_optmeas_beta_gmver(delta, K, h, d, N_contour=32, contour='cotangent', theta_w=1., m=m)
# print('empirical error ', error2)
# print('error ratio (optimal/cubature) ', error1/error0)
# print('error ratio (optimal/empirical) ', error1/error2)



