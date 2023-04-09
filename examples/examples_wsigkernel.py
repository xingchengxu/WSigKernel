import numpy as np
import matplotlib.pyplot as plt
from WSigKernel.wsigkernel import sig_kernel_matrix
from WSigKernel.wsigkernel import sig_kernel_matrix_wt

samples = 10
length_x = 100
length_y = 150
dim = 2

X = np.random.randn(samples, length_x, dim).cumsum(axis=1)
Y = np.random.randn(samples, length_y, dim).cumsum(axis=1)

X /= X.max()
Y /= Y.max()

K = sig_kernel_matrix(X, X, solver=1)
print(K)
print(K.shape)

K = sig_kernel_matrix(2*X, X, solver=1)
print(K)
print(K.shape)

K = sig_kernel_matrix(X, Y, sym=False)
print(K)
print(K.shape)


K = sig_kernel_matrix_wt(X, X, solver=1, weight=0)
print(K)
print(K.shape)




