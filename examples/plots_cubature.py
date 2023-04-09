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


# from WSigKernel.wsigkernel import sig_kernel_matrix_beta
# K = sig_kernel_matrix_beta(X, n=0, solver=0, theta=1, m=1, N_xp=201)
# print(K)


# from WSigKernel.hyperdevelop_explicit_beta import sigkernel_with_bm_beta
# h = sigkernel_with_bm_beta(X, N=32, contour='cotangent', theta_w=1., m=1.)
# print(h)

# from WSigKernel.hyperdevelop_explicit_beta import norm_of_bm_beta
# norm_bm = norm_of_bm_beta(d, N=32, contour='cotangent', theta_w=1., m=1.)
# print(norm_bm)

import math
print(math.gamma(20))

# ================================ number of sample paths n varying================================

import pandas as pd
file_path = '.\\results\\signature_kernel_result.xlsx'
data = pd.read_excel(file_path, sheet_name='cubature_beta', engine='openpyxl')

print(data)

# plt.plot(data['m'], data['corr_optimal'], label='Optimal measure')
# plt.plot(data['m'], data['corr_cubature'], label='Cubature measure')
# plt.plot(data['m'], data['corr_empirical'], label='Empirical measure')
# plt.legend()
# plt.show()
# plt.close()
#
# plt.plot(data['m'], data['loss_optimal'], label='Optimal measure')
# plt.plot(data['m'], data['loss_cubature'], label='Cubature measure')
# plt.plot(data['m'], data['loss_empirical'], label='Empirical measure')
# plt.legend()
# plt.show()
# plt.close()
#
# # plt.plot(data['m'], data['loss_cubature']-data['loss_optimal'])
# plt.plot(data['m'], data['loss_optimal']/data['loss_cubature'])
# plt.show()
# plt.close()

# ============================================== #
import math
data['g'] = data['m'].apply(lambda x: np.sqrt(math.gamma(x+1)))

plt.subplot(1, 2, 1)
plt.plot(data['m'], data['loss_optimal']*data['g'], label='Optimal measure')
plt.plot(data['m'], data['loss_cubature']*data['g'], label='Cubature measure')
plt.plot(data['m'], data['loss_empirical']*data['g'], label='Empirical measure')
plt.legend()
plt.xlabel('the parameter '+r"$m$")
plt.ylabel('the MMD distance of measures')
# plt.show()
# plt.close()

plt.subplot(1, 2, 2)
# plt.plot(data['m'], (data['loss_cubature']-data['loss_optimal'])*data['g'])
plt.plot(data['m'], data['loss_optimal']/data['loss_cubature'])
plt.xlabel('the parameter '+r"$m$")
plt.ylabel('the ratio of MMD distances (optimal/cubature)')
plt.show()
plt.close()

data['loss_optimal_g'] = data['loss_optimal']*data['g']
print(data['loss_optimal_g'].tolist())

