import numpy as np


def optimal_measure(X, n=0, solver=0, weight=1):
    """
    Compute the optimal discrete measure on a finite collection of paths
    :param X: (n,m,d), X[n_sample,n_length,n_dimension]
    :param solver: PDE solver {0,1}
    :param weight: weight for signature kernel {1,2}
    :return: optimal discrete measure: mu, weighted signature kernel matrix: K,
             weighted signature kernel with BM: vector h
    """
    from WSigKernel.wsigkernel import sig_kernel_matrix_wt
    from WSigKernel.hyperdevelop_explicit_factorial import sigkernel_with_bm
    from cvxopt import solvers, matrix

    K = sig_kernel_matrix_wt(X, X, n=n, solver=solver, weight=weight)
    h = sigkernel_with_bm(X, weight=weight)

    n = X.shape[0]
    P = matrix(K)
    q = matrix(-1*h)
    G = matrix(-1*np.identity(n))
    hh = matrix(np.zeros(n))
    A = matrix(np.ones(n), (1, n))
    b = matrix([1.0])

    result = solvers.qp(P, q, G, hh, A, b)
    mu = np.array([one for one in result['x']])

    return mu, K, h


def correlation_wiener_measure(mu, K, h, dim_BM):
    """
    Correlation between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :return: correlation
    """

    corr = np.dot(h, mu)/(np.sqrt(np.dot(np.matmul(mu, K), mu))*np.exp(dim_BM/8))
    return corr


def error_optmeas(mu, K, h, dim_BM):
    """
    error between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :return: error
    """

    error = np.sqrt(np.dot(np.matmul(mu, K), mu)-2*np.dot(h, mu)+np.exp(dim_BM/4))
    return error


def optimal_measure_original(X, n=0, solver=0, N_contour=32, contour='cotangent'):
    """
    Compute the optimal discrete measure on a finite collection of paths
    :param X: (n,m,d), X[n_sample,n_length,n_dimension]
    :param solver: PDE solver {0,1}
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: optimal discrete measure: mu, signature kernel matrix: K,
             signature kernel with BM: vector h
    """
    from WSigKernel.wsigkernel import sig_kernel_matrix_wt
    from WSigKernel.hyperdevelop_explicit_original import sigkernel_with_bm_original
    from cvxopt import solvers, matrix

    K = sig_kernel_matrix_wt(X, X, n=n, solver=solver, weight=0)
    h = sigkernel_with_bm_original(X, N=N_contour, contour=contour)

    n = X.shape[0]
    P = matrix(K)
    q = matrix(-1*h)
    G = matrix(-1*np.identity(n))
    hh = matrix(np.zeros(n))
    A = matrix(np.ones(n), (1, n))
    b = matrix([1.0])

    try:
        result = solvers.qp(P, q, G, hh, A, b)
        mu = np.array([one for one in result['x']])
    except:
        mu = None

    return mu, K, h


def correlation_wiener_measure_original(mu, K, h, dim_BM, N_contour=32, contour='cotangent'):
    """
    Correlation between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: correlation
    """
    from WSigKernel.hyperdevelop_explicit_original import norm_of_bm_original

    norm_bm = norm_of_bm_original(dim_BM, N=N_contour, contour=contour)
    corr = np.dot(h, mu)/(np.sqrt(np.dot(np.matmul(mu, K), mu))*norm_bm)
    return corr


def error_optmeas_original(mu, K, h, dim_BM, N_contour=32, contour='cotangent'):
    """
    error between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: error
    """
    from WSigKernel.hyperdevelop_explicit_original import norm_of_bm_original

    norm_bm = norm_of_bm_original(dim_BM, N=N_contour, contour=contour)
    error = np.sqrt(np.dot(np.matmul(mu, K), mu)-2*np.dot(h, mu)+norm_bm**2)
    return error


def optimal_measure_beta(X, n=0, solver=0, N_contour=32, contour='cotangent', theta=1., m=1., N_xp=201):
    """
    Compute the optimal discrete measure on a finite collection of paths
    :param X: (n,m,d), X[n_sample,n_length,n_dimension]
    :param solver: PDE solver {0,1}
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: optimal discrete measure: mu, signature kernel matrix: K,
             signature kernel with BM: vector h
    """
    from WSigKernel.wsigkernel import sig_kernel_matrix_beta
    from WSigKernel.hyperdevelop_explicit_beta import sigkernel_with_bm_beta
    from cvxopt import solvers, matrix

    K = sig_kernel_matrix_beta(X, n=n, solver=solver, theta=theta, m=m, N_xp=N_xp)
    h = sigkernel_with_bm_beta(X, N=N_contour, contour=contour, theta_w=theta, m=m)

    n = X.shape[0]
    P = matrix(K)
    q = matrix(-1*h)
    G = matrix(-1*np.identity(n))
    hh = matrix(np.zeros(n))
    A = matrix(np.ones(n), (1, n))
    b = matrix([1.0])

    result = solvers.qp(P, q, G, hh, A, b)
    mu = np.array([one for one in result['x']])

    return mu, K, h


def correlation_wiener_measure_beta(mu, K, h, dim_BM, N_contour=32, contour='cotangent', theta_w=1., m=1.):
    """
    Correlation between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: correlation
    """
    from WSigKernel.hyperdevelop_explicit_beta import norm_of_bm_beta

    norm_bm = norm_of_bm_beta(dim_BM, N=N_contour, contour=contour, theta_w=theta_w, m=m)
    corr = np.dot(h, mu)/(np.sqrt(np.dot(np.matmul(mu, K), mu))*norm_bm)
    return corr


def error_optmeas_beta(mu, K, h, dim_BM, N_contour=32, contour='cotangent', theta_w=1., m=1.):
    """
    error between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: error
    """
    from WSigKernel.hyperdevelop_explicit_beta import norm_of_bm_beta

    norm_bm = norm_of_bm_beta(dim_BM, N=N_contour, contour=contour, theta_w=theta_w, m=m)
    error = np.sqrt(np.dot(np.matmul(mu, K), mu)-2*np.dot(h, mu)+norm_bm**2)
    return error


def optimal_measure_beta_gmver(X, n=0, solver=0, N_contour=32, contour='cotangent', theta=1., m=1., N_xp=201):
    """
    Compute the optimal discrete measure on a finite collection of paths
    :param X: (n,m,d), X[n_sample,n_length,n_dimension]
    :param solver: PDE solver {0,1}
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: optimal discrete measure: mu, signature kernel matrix: K,
             signature kernel with BM: vector h
    """
    from WSigKernel.wsigkernel import sig_kernel_matrix_beta
    from WSigKernel.hyperdevelop_explicit_beta import sigkernel_with_bm_beta
    from cvxopt import solvers, matrix
    from math import gamma

    K = sig_kernel_matrix_beta(X, n=n, solver=solver, theta=theta, m=m, N_xp=N_xp)*gamma(m+1)
    h = sigkernel_with_bm_beta(X, N=N_contour, contour=contour, theta_w=theta, m=m)*gamma(m+1)

    n = X.shape[0]
    P = matrix(K)
    q = matrix(-1*h)
    G = matrix(-1*np.identity(n))
    hh = matrix(np.zeros(n))
    A = matrix(np.ones(n), (1, n))
    b = matrix([1.0])

    result = solvers.qp(P, q, G, hh, A, b)
    mu = np.array([one for one in result['x']])

    return mu, K, h


def correlation_wiener_measure_beta_gmver(mu, K, h, dim_BM, N_contour=32, contour='cotangent', theta_w=1., m=1.):
    """
    Correlation between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: correlation
    """
    from WSigKernel.hyperdevelop_explicit_beta import norm_of_bm_beta
    from math import gamma

    norm_bm = norm_of_bm_beta(dim_BM, N=N_contour, contour=contour, theta_w=theta_w, m=m)*np.sqrt(gamma(m+1))
    corr = np.dot(h, mu)/(np.sqrt(np.dot(np.matmul(mu, K), mu))*norm_bm)
    return corr


def error_optmeas_beta_gmver(mu, K, h, dim_BM, N_contour=32, contour='cotangent', theta_w=1., m=1.):
    """
    error between discrete measure mu on a finite collection of paths and Wiener measure.
    :param mu: discrete measure
    :param K: signature kernel matrix of the finite collection of paths
    :param h: signature kernel vector of the finite collection of paths with Brownian motion
    :param dim_BM: dimension of underlying Brownian motion
    :param N_contour: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: error
    """
    from WSigKernel.hyperdevelop_explicit_beta import norm_of_bm_beta
    from math import gamma

    norm_bm = norm_of_bm_beta(dim_BM, N=N_contour, contour=contour, theta_w=theta_w, m=m)*np.sqrt(gamma(m+1))
    error = np.sqrt(np.dot(np.matmul(mu, K), mu)-2*np.dot(h, mu)+norm_bm**2)
    return error


def kl_divergence(mu, delta):
    return np.dot(mu, np.log(mu/delta))


def js_divergence(mu, delta):
    return 0.5*kl_divergence(mu, 0.5*(mu+delta))+0.5*kl_divergence(delta, 0.5*(mu+delta))


# # ========================n paths with m pieces in d dimension: Brownian motion===================== #
#
# # the number of weights and cubature paths
# n = 10
# # number of piecewise linear pieces
# m = 10
# # the dimension of the underlying Brownian motion
# d = 2
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
#
# mu, K, h = optimal_measure(X, solver=1, weight=1)
#
# print('optimal discrete measure ', mu)
#
# corr1 = correlation_wiener_measure(mu, K, h, d)
# print('correlation ', corr1)
#
# delta = np.ones(n)/n
# corr2 = correlation_wiener_measure(delta, K, h, d)
# print('correlation ', corr2)
# print('correlation ratio ', corr1/corr2)
#
# error1 = error_optmeas(mu, K, h, d)
# print('error ', error1)
# error2 = error_optmeas(delta, K, h, d)
# print('error ', error2)
# print('error ratio ', error1/error2)
#
# kl_div = kl_divergence(mu, delta)
# print('KL divergence ', kl_div)
#
# js_div = js_divergence(mu, delta)
# print('JS divergence ', js_div)

#########################################################################################



