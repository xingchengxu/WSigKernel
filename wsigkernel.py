import numpy as np


def sig_kernel_matrix(x, y, n=0, solver=0, sym=True, full=False, rbf=False, sigma=1.):
    """
    :param x: x[n_sample,n_length,n_dimension]=X(n,m,d)
    :param y: y[n_sample,n_length,n_dimension]
    :param n: dyadic refinements 2^n pieces for each observation interval
    :param solver: PDE solver scheme
    :param sym: the kernel matrix is symmetric or not
    :param full: return the full kernel K(s,t) or just the last entry
    :param rbf: radial basis function
    :param sigma: parameter
    :return: signature kernel matrix for sample paths in x and y
    """

    A = x.shape[0]
    B = y.shape[0]
    M = x.shape[1]
    N = y.shape[1]
    D = x.shape[2]

    factor = 2 ** (2 * n)
    MM = (2 ** n) * (M - 1)
    NN = (2 ** n) * (N - 1)
    K = np.zeros((A, B, MM + 1, NN + 1), dtype=np.float64)

    if sym:
        for l in range(A):
            for m in range(l, A):

                for i in range(MM + 1):
                    K[l, m, i, 0] = 1.
                    K[m, l, i, 0] = 1.

                for j in range(NN + 1):
                    K[l, m, 0, j] = 1.
                    K[m, l, 0, j] = 1.

                for i in range(MM):
                    for j in range(NN):

                        ii = int(i / (2 ** n))
                        jj = int(j / (2 ** n))

                        increment = 0.
                        d1 = 0.
                        d2 = 0.
                        d3 = 0.
                        d4 = 0.
                        for k in range(D):

                            if rbf:
                                d1 = d1 + (x[l, ii + 1, k] - y[m, jj + 1, k]) ** 2
                                d2 = d2 + (x[l, ii + 1, k] - y[m, jj, k]) ** 2
                                d3 = d3 + (x[l, ii, k] - y[m, jj + 1, k]) ** 2
                                d4 = d4 + (x[l, ii, k] - y[m, jj, k]) ** 2
                            else:
                                increment = increment + (x[l, ii + 1, k] - x[l, ii, k]) * (y[m, jj + 1, k] - y[m, jj, k])

                        if rbf:
                            increment = (np.exp(-d1 / sigma) - np.exp(-d2 / sigma) - np.exp(-d3 / sigma) + np.exp(-d4 / sigma)) / factor
                        else:
                            increment = increment / factor

                        if solver == 0:
                            K[l, m, i + 1, j + 1] = K[l, m, i + 1, j] + K[l, m, i, j + 1] + K[l, m, i, j] * (increment - 1.)
                        elif solver == 1:
                            K[l, m, i + 1, j + 1] = (K[l, m, i + 1, j] + K[l, m, i, j + 1]) * (
                                        1. + 0.5 * increment + (1. / 12) * increment ** 2) - K[l, m, i, j] * (
                                                                1. - (1. / 12) * increment ** 2)
                        else:
                            K[l, m, i + 1, j + 1] = K[l, m, i + 1, j] + K[l, m, i, j + 1] - K[l, m, i, j] + (
                                        np.exp(0.5 * increment) - 1.) * (K[l, m, i + 1, j] + K[l, m, i, j + 1])

                        K[m, l, j + 1, i + 1] = K[l, m, i + 1, j + 1]
    else:
        for l in range(A):
            for m in range(B):

                for i in range(MM + 1):
                    K[l, m, i, 0] = 1.

                for j in range(NN + 1):
                    K[l, m, 0, j] = 1.

                for i in range(MM):
                    for j in range(NN):

                        ii = int(i / (2 ** n))
                        jj = int(j / (2 ** n))

                        increment = 0.
                        d1 = 0.
                        d2 = 0.
                        d3 = 0.
                        d4 = 0.
                        for k in range(D):

                            if rbf:
                                d1 = d1 + (x[l, ii + 1, k] - y[m, jj + 1, k]) ** 2
                                d2 = d2 + (x[l, ii + 1, k] - y[m, jj, k]) ** 2
                                d3 = d3 + (x[l, ii, k] - y[m, jj + 1, k]) ** 2
                                d4 = d4 + (x[l, ii, k] - y[m, jj, k]) ** 2
                            else:
                                increment = increment + (x[l, ii + 1, k] - x[l, ii, k]) * (
                                            y[m, jj + 1, k] - y[m, jj, k])

                        if rbf:
                            increment = (np.exp(-d1 / sigma) - np.exp(-d2 / sigma) - np.exp(-d3 / sigma) + np.exp(
                                -d4 / sigma)) / factor
                        else:
                            increment = increment / factor

                        if solver == 0:
                            K[l, m, i + 1, j + 1] = K[l, m, i + 1, j] + K[l, m, i, j + 1] + K[l, m, i, j] * (
                                        increment - 1.)
                        elif solver == 1:
                            K[l, m, i + 1, j + 1] = (K[l, m, i + 1, j] + K[l, m, i, j + 1]) * (
                                        1. + 0.5 * increment + (1. / 12) * increment ** 2) - K[l, m, i, j] * (
                                                                1. - (1. / 12) * increment ** 2)
                        else:
                            K[l, m, i + 1, j + 1] = K[l, m, i + 1, j] + K[l, m, i, j + 1] - K[l, m, i, j] + (
                                        np.exp(0.5 * increment) - 1.) * (K[l, m, i + 1, j] + K[l, m, i, j + 1])

    if full:
        return np.array(K)
    else:
        return np.array(K[:, :, MM, NN])


def sig_kernel_matrix_wt(x, y, n=0, solver=0, sym=True, rbf=False, sigma=1., weight=0):
    """
    :param x: x[n_sample,n_length,n_dimension]
    :param y: y[n_sample,n_length,n_dimension]
    :param n: dyadic refinements 2^n pieces for each observation interval
    :param solver: PDE solver scheme
    :param sym: the kernel matrix is symmetric or not
    :param full: return the full kernel K(s,t) or just the last entry
    :param rbf: radial basis function
    :param sigma: parameter
    :param weight: version of weight: 0=no weight=the usual kernel, 1=(k/2)! weight, 2=2^(k/2)(k/2)! weight
    :return: (weighted) signature kernel matrix for sample paths in x and y
    """

    A = x.shape[0]
    B = y.shape[0]

    K = np.zeros((A, B), dtype=np.float64)

    w_quad = [0.379531, 0.213681, 0.559586, 0.958717, 0.116908, 0.102936, 0.646825, 0.283191,
              0.836265, 0.159774, 0.187013, 0.124394, 0.420847, 0.605185, 0.264341, 0.152459]
    a_quad = [0.477580, 0.157564, 0.323656, 0.539147, 0.797005, 0.109096, 0.141598, 0.176844,
              0.214615, 0.254837, 0.297590, 0.343148, 0.392069, 0.445412, 0.505367, 0.577848]

    if weight == 0:
        return sig_kernel_matrix(x, y, n=n, solver=solver, sym=sym, full=False, rbf=rbf, sigma=sigma)

    elif weight == 1:
        for i in range(len(a_quad)):
            K_xy = sig_kernel_matrix(a_quad[i]*x, y, n=n, solver=solver, sym=sym, full=False, rbf=rbf, sigma=sigma)
            K += w_quad[i]*K_xy
        return K*2

    elif weight == 2:
        for i in range(len(a_quad)):
            K_xy = sig_kernel_matrix(np.sqrt(2)*a_quad[i]*x, y, n=n, solver=solver, sym=sym, full=False, rbf=rbf, sigma=sigma)
            K += w_quad[i]*K_xy
        return K*2

    else:
        return print('Weight is only accepted in {0,1,2}.')


def sig_kernel_matrix_beta(X, n=0, solver=0, theta=1., m=1., N_xp=201, sym=True):
    """
    kernel under the beta weight
    :param X: (n,m,d) shape
    :param n:
    :param solver:
    :param theta:
    :param m: m>0
    :param N_xp: the number of x taken from [0,1]
    :return: kernel matrix
    """

    # from WSigKernel.sigkernel import sig_kernel_matrix
    import math

    h = 1 / (N_xp - 1)
    xp = np.linspace(h, 1 - h, N_xp - 2)

    A = X.shape[0]
    K = np.zeros((A, A), dtype=np.float64)
    for i in range(len(xp)):
        K_xp = sig_kernel_matrix(theta * xp[i] * X, X, n=n, solver=solver, sym=sym, full=False, rbf=False, sigma=1.)
        K += K_xp * (1 - xp[i]) ** (m - 1)

    return K / math.gamma(m)


def sig_kernel_matrix_beta_unsym(x, y, n=0, solver=0, theta=1., m=1., N_xp=201, sym=True):
    """
    kernel under the beta weight
    :param x: (n,m,d) shape
    :param n:
    :param solver:
    :param theta:
    :param m: m>0
    :param N_xp: the number of x taken from [0,1]
    :return: kernel matrix
    """

    import math

    h = 1 / (N_xp - 1)
    xp = np.linspace(h, 1 - h, N_xp - 2)

    A = x.shape[0]
    B = y.shape[0]

    K = np.zeros((A, B), dtype=np.float64)
    for i in range(len(xp)):
        K_xp = sig_kernel_matrix(theta * xp[i] * x, y, n=n, solver=solver, sym=sym, full=False, rbf=False, sigma=1.)
        K += K_xp * (1 - xp[i]) ** (m - 1)

    return K / math.gamma(m)

##############################################################################



