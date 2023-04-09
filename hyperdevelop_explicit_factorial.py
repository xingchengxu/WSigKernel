import tensorflow as tf
import numpy as np


# =================== calculate the Cartan and hyperbolic development ======================== #


def hyperbolic_development_direction(V):
    """
    :param V: directions of n paths with m pieces in d dimension, V.shape=(n,d,m) in ndarray format
    :return: the last coordinate of the hyperbolic development of paths
    """

    d = V.shape[1]
    m = V.shape[2]
    V = tf.Variable(V, dtype=tf.float64, name="V")
    a = np.identity(d+1)
    I = tf.constant(a, name="I")
    er = tf.zeros([d], tf.float64, name="er")
    e1 = tf.ones([1], tf.float64, name="e1")
    ed = tf.concat([er, e1], 0)
    on = tf.constant(0, dtype=tf.float64)

    A = tf.zeros([d, d], dtype=tf.float64, name="A")
    C = tf.Variable(a, dtype=tf.float64, name="C")
    E = tf.unstack(V, axis=0, name="E")
    F = tf.ones([0], tf.float64, name="F")

    for Y in E:
        E1 = tf.unstack(Y, axis=1, name="E1")
        for X in E1:

            M = tf.unstack(A, axis=0)
            M.append(X)
            D = tf.stack(M, axis=0)
            L = tf.unstack(D, axis=1)
            Xb = tf.unstack(X, axis=0)
            Xb.append(on)
            Z = tf.stack(Xb, axis=0)
            L.append(Z)
            J = tf.stack(L, axis=1)

            norm = tf.norm(X)
            # C is the Cartan development: matrix Gamma
            if norm == 0:
                C = I
            else:
                C = tf.matmul(I+(1/norm)*tf.sinh(norm/m)*J + (1/norm**2)*(tf.cosh(norm/m)-1)*tf.matmul(J, J), C)
            # print(C)

        # minkowski_metric = tf.reduce_sum(tf.square(C[:-1, :]), axis=0) - tf.square(C[-1, :])
        # print('Minkowski metric ', minkowski_metric)

        G = tf.linalg.matvec(C, ed)
        # H is the last coordinate of the hyperbolic development
        H = tf.tensordot(G, ed, [[0], [0]])
        Q = tf.unstack(F, axis=0)
        Q.append(H)
        F = tf.stack(Q, axis=0)
        C = I

    return F.numpy()


def hyperbolic_development_coordinate(X):
    """
    :param X: coordinate of n paths with m pieces in d dimension, X.shape=(n,m,d) in ndarray format
    :return: the last coordinate of the hyperbolic development of paths
    """

    VV = X[:, 1:, :] - X[:, :-1, :]

    b = np.zeros((VV.shape[0], VV.shape[2], VV.shape[1]))
    for i in range(VV.shape[0]):
        b[i, :, :] = VV[i, :, :].T

    # direction tensor for piecewise linear paths
    V = tf.Variable(b, dtype=tf.float64, name="V")

    hd_kernel = hyperbolic_development_direction(V)

    return hd_kernel


def sigkernel_with_bm(X, weight=1):
    """
    :param X: coordinate of n paths with m pieces in d dimension, X.shape=(n,m,d) in ndarray format
    :param weight: weight=1 means (k/2)! weight, weight=2 means 2^(k/2)(k/2)! weight
    :return: the last coordinate of the hyperbolic development of paths
    """
    if weight == 1:
        lbd = np.sqrt(1 / 2)
        hd_kernel = hyperbolic_development_coordinate(lbd * X)
        return hd_kernel

    elif weight == 2:
        hd_kernel = hyperbolic_development_coordinate(X)
        return hd_kernel

    else:
        return print('Weight should be in {1,2}.')


# =================== coordinates of paths to directions of paths: axis paths ======================== #

# # direction v_i, i=1,...,m
# # direction tensor for piecewise linear paths
#
# # Test examples: axis paths
# V = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
# # V = np.array([[[1, 0, -1, 0], [0, 1, 0, -1]], [[0, 1, 0, -1], [1, 0, -1, 0]]])
# # V = np.array([[[1, 0, -1, 0, 1, 0], [0, 1, 0, -1, 0, 1]], [[0, 1, 0, -1, 0, 1], [1, 0, -1, 0, 1, 0]]])
# print(V)
#
# hd_kernel = hyperbolic_development_direction(V)
#
# print(hd_kernel)

# ========================n paths with m pieces in d dimension: Brownian motion===================== #
# #
# # the number of weights and cubature paths
# n = 10
# # the dimension of the underlying Brownian motion
# d = 2
# # number of piecewise linear pieces
# m = 5
#
# import matplotlib.pyplot as plt
# # d-dimensional Brownian motion
# X = np.random.randn(n, m, d)
# for i in range(X.shape[0]):
#     X[i, :, :] = np.cumsum(X[i, :, :], axis=0)
#     plt.plot(X[i, :, 0])
#     plt.plot(X[i, :, 1])
# plt.show()
#
# hd_kernel = hyperbolic_development_coordinate(X)
# print(hd_kernel)
#
# lbd = np.sqrt(1/2)
# hd_kernel = hyperbolic_development_coordinate(lbd*X)
# print(hd_kernel)
#
# kernel_bm = sigkernel_with_bm(X, weight=1)
# print(kernel_bm)

##############################################################################

