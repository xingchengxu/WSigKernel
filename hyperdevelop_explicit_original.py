import tensorflow as tf
import numpy as np


# =================== calculate the Cartan and hyperbolic development ======================== #


def hyperbolic_development_direction(zeta, V):
    """
    :param V: directions of n paths with m pieces in d dimension, V.shape=(n,d,m) in ndarray format
    :return: the last coordinate of the hyperbolic development of paths
    """

    d = V.shape[1]
    m = V.shape[2]
    V = tf.Variable(V, dtype=tf.float64, name="V")
    a = np.identity(d+1)
    # I = tf.constant(a, name="I")
    I = tf.complex(a, np.zeros((d+1, d+1)))
    er = tf.zeros([d], tf.float64, name="er")
    e1 = tf.ones([1], tf.float64, name="e1")
    ed = tf.concat([er, e1], 0)
    ed = tf.complex(ed, np.zeros_like(ed))
    on = tf.constant(0, dtype=tf.float64)

    A = tf.zeros([d, d], dtype=tf.float64, name="A")
    C = tf.complex(a, np.zeros((d+1, d+1)))
    E = tf.unstack(V, axis=0, name="E")
    F = tf.ones([0], tf.complex128, name="F")

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
            J = tf.complex(J, np.zeros_like(J))

            norm = tf.norm(X)
            # C is the Cartan development: matrix Gamma
            if norm == 0:
                C = I
            else:
                # zz = np.zeros((1,), dtype=float)
                # ss = tf.complex(1 / (abs(zeta)*norm) * np.sinh(abs(zeta)*norm / m), zz) * zeta*J
                # cc = tf.complex((1 / ((abs(zeta)*norm) ** 2)) * (tf.cosh(abs(zeta)*norm / m)-1), zz)*zeta**2*tf.matmul(J, J)

                imag0 = np.zeros(1)
                imag0 = imag0[0]
                ss = tf.complex(1 / norm, imag0) * tf.sinh(zeta * tf.complex(norm / m, imag0)) * J
                # print(ss)
                cc = tf.complex(1 / norm ** 2, imag0) * (tf.cosh(zeta * tf.complex(norm / m, imag0))-1) * tf.matmul(J, J)
                # print(cc)
                C = tf.matmul(I + ss + cc, C)
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


def hyperbolic_development_coordinate(zeta, X):
    """
    :param X: coordinate of n paths with m pieces in d dimension, X.shape=(n,m,d) in ndarray format
    :return: the last coordinate of the hyperbolic development of paths
    """

    VV = X[:, 1:, :] - X[:, :-1, :]

    b = np.zeros((VV.shape[0], VV.shape[2], VV.shape[1]))
    # b = np.zeros((VV.shape[0], VV.shape[2], VV.shape[1]), dtype=complex)
    for i in range(VV.shape[0]):
        b[i, :, :] = VV[i, :, :].T

    # direction tensor for piecewise linear paths
    V = tf.Variable(b, dtype=tf.float64, name="V")

    hd_kernel = hyperbolic_development_direction(zeta, V)

    return hd_kernel


def sigkernel_with_bm(zeta, X, weight=1):
    """
    :param X: coordinate of n paths with m pieces in d dimension, X.shape=(n,m,d) in ndarray format
    :param weight: weight=1 means (k/2)! weight, weight=2 means 2^(k/2)(k/2)! weight
    :return: the last coordinate of the hyperbolic development of paths
    """
    if weight == 1:
        lbd = np.sqrt(1 / 2)
        hd_kernel = hyperbolic_development_coordinate(zeta, lbd * X)
        return hd_kernel

    elif weight == 2:
        hd_kernel = hyperbolic_development_coordinate(zeta, X)
        return hd_kernel

    else:
        return print('Weight should be in {1,2}.')


def fun_z(zz_array, X):
    """
    compute the f(z) in the complex integral
    :param zz_array: shape (N,) the values of zeta
    :param X: paths (n,m,d)
    :return: f(z) in shape (N,n)
    """

    length = zz_array.shape[0]
    fz = np.zeros((length, X.shape[0]), dtype=complex)
    for i in range(length):
        zz = zz_array[i]
        zeta = np.sqrt(1 / zz)
        fz[i, :] = sigkernel_with_bm(zeta, X, weight=1) / zz

    return fz


def sigkernel_with_bm_original_old(X, N=32, contour='cotangent'):
    """
    compute the signature kernel with BM under original definition
    :param X: (n,m,d) paths
    :param N: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent'}
    :return: the kernel with BM (original)
    """

    theta = np.linspace(-np.pi, np.pi, N)

    if contour == 'parabola':
        z_theta = N*(0.1309-0.1194*theta**2+0.2500j*theta)
        w_theta = N*(-0.1194*2*theta+0.2500j)

    elif contour == 'hyperbola':
        z_theta = 2.246*N*(1-np.sin(1.1721-0.3443j*theta))
        w_theta = 2.246*N*0.3443j*np.cos(1.1721-0.3443j*theta)

    else:
        z_theta = N*(0.5017*theta*(1/np.tan(0.6407*theta))-0.6122+0.2645j*theta)
        w_theta = N*(0.5017*(1/np.tan(0.6407*theta))-0.5017*theta/np.sin(0.6407*theta)**2*0.6407+0.2645j)

    c_theta = 1j*np.exp(z_theta)*w_theta/N

    fz = fun_z(z_theta, X)

    h_normal = -1*np.matmul(c_theta, fz)
    h_normal = h_normal.real

    return h_normal


def sigkernel_with_bm_original(X, N=32, contour='cotangent'):
    """
    compute the signature kernel with BM under original definition
    :param X: (n,m,d) paths
    :param N: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent', 'cycle'}
    :return: the kernel with BM (original)
    """

    theta = np.linspace(-np.pi, np.pi, N)

    if contour == 'parabola':
        z_theta = N*(0.1309-0.1194*theta**2+0.2500j*theta)
        w_theta = N*(-0.1194*2*theta+0.2500j)

    elif contour == 'hyperbola':
        z_theta = 2.246*N*(1-np.sin(1.1721-0.3443j*theta))
        w_theta = 2.246*N*0.3443j*np.cos(1.1721-0.3443j*theta)

    elif contour == 'cotangent':
        z_theta = N*(0.5017*theta*(1/np.tan(0.6407*theta))-0.6122+0.2645j*theta)
        w_theta = N*(0.5017*(1/np.tan(0.6407*theta))-0.5017*theta/np.sin(0.6407*theta)**2*0.6407+0.2645j)
    else:
        z_theta = np.exp(1j*theta)

    if contour in ['parabola', 'hyperbola', 'cotangent']:
        c_theta = 1j*np.exp(z_theta)*w_theta/N

        fz = fun_z(z_theta, X)

        h_normal = -1*np.matmul(c_theta, fz)
        h_normal = h_normal.real
    else:
        c_theta = np.exp(z_theta)*z_theta/N
        fz = fun_z(z_theta, X)
        h_normal = np.matmul(c_theta, fz)
        h_normal = h_normal.real

    return h_normal


def norm_of_bm_original(d, N=32, contour='cotangent'):
    """
    compute the norm of BM under original inner product in [0,1]
    :param d: dimension of BM
    :param N: level to compute the complex integral
    :param contour: {'parabola', 'hyperbola', 'cotangent', 'cycle'}
    :return: the norm of BM (original)
    """

    theta = np.linspace(-np.pi, np.pi, N)

    if contour == 'parabola':
        z_theta = N*(0.1309-0.1194*theta**2+0.2500j*theta)
        w_theta = N*(-0.1194*2*theta+0.2500j)

    elif contour == 'hyperbola':
        z_theta = 2.246*N*(1-np.sin(1.1721-0.3443j*theta))
        w_theta = 2.246*N*0.3443j*np.cos(1.1721-0.3443j*theta)

    elif contour == 'cotangent':
        z_theta = N*(0.5017*theta*(1/np.tan(0.6407*theta))-0.6122+0.2645j*theta)
        w_theta = N*(0.5017*(1/np.tan(0.6407*theta))-0.5017*theta/np.sin(0.6407*theta)**2*0.6407+0.2645j)
    else:
        z_theta = np.exp(1j*theta)

    if contour in ['parabola', 'hyperbola', 'cotangent']:
        c_theta = 1j*np.exp(z_theta)*w_theta/N

        fz = np.exp(d/(4*z_theta))/z_theta

        h_normal = -1*np.dot(c_theta, fz)
        # print(h_normal)
        h_normal = h_normal.real
        h_normal = np.sqrt(h_normal)
    else:
        c_theta = np.exp(z_theta) * z_theta / N
        fz = np.exp(d/(4*z_theta))/z_theta
        h_normal = np.dot(c_theta, fz)
        h_normal = h_normal.real
        h_normal = np.sqrt(h_normal)

    return h_normal




