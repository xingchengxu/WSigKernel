import numpy as np


def fun_z_beta(zz_array, X, xp, theta=1., m=1.):
    """
    compute the f(z) in the complex integral
    :param zz_array: shape (N,) the values of zeta
    :param X: paths (n,m,d)
    :return: f(z) in shape (N,n)
    """

    from WSigKernel.hyperdevelop_explicit_original import sigkernel_with_bm
    length = zz_array.shape[0]
    fz = np.zeros((length, X.shape[0]), dtype=complex)
    for i in range(length):
        zz = zz_array[i]
        zeta = np.sqrt(2)*xp*theta/zz
        fz[i, :] = sigkernel_with_bm(zeta, X, weight=1) / zz**(m+1)

    return fz


def sigkernel_with_bm_beta1(X, N=32, contour='cotangent', xp=1., theta_w=1., m=1.):
    """
    compute the signature kernel with BM under beta weight
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

    fz = fun_z_beta(z_theta, X, xp, theta=theta_w, m=m)

    h_normal = -1*np.matmul(c_theta, fz)
    h_normal = h_normal.real

    return h_normal


def sigkernel_with_bm_beta(X, N=32, contour='cotangent', theta_w=1., m=1.):
    """
    """

    w_quad = [7.64043E-6, 0.00134365, 0.0338744, 0.240139, 0.610863, 0.610863, 0.240139, 0.0338744, 0.00134365, 7.64043E-6]
    a_quad = [-3.43616, -2.53273, -1.75668, -1.03661, -0.342901, 0.342901, 1.03661, 1.75668, 2.53273, 3.43616]

    h = np.zeros((X.shape[0], ), dtype=np.float64)
    for i in range(len(a_quad)):
        fx = sigkernel_with_bm_beta1(X, N=N, contour=contour, xp=np.sqrt(2)*a_quad[i], theta_w=theta_w, m=m)
        fx *= np.exp(-a_quad[i]**2)
        h += w_quad[i]*fx

    return h/np.sqrt(np.pi)


# def norm_of_bm_beta(d, N=32, contour='cotangent', theta_w=1., m=1.):
#     """
#     compute the norm of BM under beta inner product in [0,1]
#     :param d: dimension of BM
#     :param N: level to compute the complex integral
#     :param contour: {'parabola', 'hyperbola', 'cotangent'}
#     :return: the norm of BM (original)
#     """
#
#     theta = np.linspace(-np.pi, np.pi, N)
#
#     if contour == 'parabola':
#         z_theta = N*(0.1309-0.1194*theta**2+0.2500j*theta)
#         w_theta = N*(-0.1194*2*theta+0.2500j)
#
#     elif contour == 'hyperbola':
#         z_theta = 2.246*N*(1-np.sin(1.1721-0.3443j*theta))
#         w_theta = 2.246*N*0.3443j*np.cos(1.1721-0.3443j*theta)
#
#     else:
#         z_theta = N*(0.5017*theta*(1/np.tan(0.6407*theta))-0.6122+0.2645j*theta)
#         w_theta = N*(0.5017*(1/np.tan(0.6407*theta))-0.5017*theta/np.sin(0.6407*theta)**2*0.6407+0.2645j)
#
#     c_theta = 1j*np.exp(z_theta)*w_theta/N
#
#     fz = 1/np.sqrt(1-theta_w**2*d/z_theta**2)/z_theta**(m+1)
#
#     h_normal = -1*np.dot(c_theta, fz)
#     # print(h_normal)
#     h_normal = h_normal.real
#     h_normal = np.sqrt(abs(h_normal))
#
#     return h_normal

def norm_of_bm_beta1(d, N=32, contour='cotangent', xp=1., theta_w=1., m=1.):
    """
    compute the signature kernel with BM under beta weight
    :param d:
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

    fz = np.exp((xp*theta_w)**2*d/(2*z_theta**2))/z_theta**(m+1)

    h_normal = -1*np.matmul(c_theta, fz)
    h_normal = h_normal.real

    return h_normal


def norm_of_bm_beta(d, N=32, contour='cotangent', theta_w=1., m=1.):
    """
    """

    w_quad = [7.64043E-6, 0.00134365, 0.0338744, 0.240139, 0.610863, 0.610863, 0.240139, 0.0338744, 0.00134365, 7.64043E-6]
    a_quad = [-3.43616, -2.53273, -1.75668, -1.03661, -0.342901, 0.342901, 1.03661, 1.75668, 2.53273, 3.43616]

    h = np.zeros(1, dtype=np.float64)
    for i in range(len(a_quad)):
        fx = norm_of_bm_beta1(d, N=N, contour=contour, xp=np.sqrt(2)*a_quad[i], theta_w=theta_w, m=m)
        fx *= np.exp(-a_quad[i]**2)
        h += w_quad[i]*fx

    h = h/np.sqrt(np.pi)
    print(h)

    return np.sqrt(abs(h[0]))


