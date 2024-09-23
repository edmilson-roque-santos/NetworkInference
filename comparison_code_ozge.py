"""
Created on Tue Sep 17 15:51:26 2024

@author: Ozge Canli Usta
"""

import numpy as np
from scipy import spatial as ss
from scipy import special

def cse_new(data, conditional_data, data_delayed, neigh):

    """

    :param data: first observation vector
    :param conditional_data: causation entropy conditional observation vector
    :param data_delayed: time-delayed second observation vector and its given as data_delayed
    :param neigh: number of neighbors
    :return: causation entropy from first observation to second on condition given conditional data

    """
    n, p = data.shape


    if np.size(conditional_data) == 0:
        # print('delayed mutual information')
        data_delayed = np.hstack((data_delayed, data))
        return ksg(data_delayed, neigh, borders=False)

    else:
        # print('causation entropy')

        data_all = np.hstack((data, data_delayed, conditional_data))

        tree = ss.cKDTree(data_all)
        dist, ind = tree.query(data_all, neigh + 1, p=float('inf'))

        tree_xy = ss.cKDTree(conditional_data)

        data_xpx = np.hstack((data_delayed, conditional_data))
        tree_xpx = ss.cKDTree(data_xpx)

        data_yx = np.hstack((conditional_data, data))
        tree_yx = ss.cKDTree(data_yx)

        Neigh_sum = 0

        for i in range(n):

            ee = dist[i, neigh]

            nx = tree_xy.query_ball_point(conditional_data[i, :], ee - 1e-15, p=float('inf'))
            nxpx = tree_xpx.query_ball_point(data_xpx[i, :], ee - 1e-15, p=float('inf'))
            nyx = tree_yx.query_ball_point(data_yx[i, :], ee - 1e-15, p=float('inf'))

            if (len(nx) == 0):
                print('xx')

            else:

                Neigh_sum += (special.digamma(len(nx)) - special.digamma(len(nxpx)) - special.digamma(len((nyx)))) / n

        return special.digamma(neigh) + Neigh_sum


def ksg(data, neig, center=True, borders=True):
    """
    MI KSG estimators in 2 dim I^(2) (X,Y)
    Args:
        data:
        neig: number of neighbors
        center: including center point or not
        borders: including border point or not

    Returns:

    """
    x = data[:, [0]]
    y = data[:, [1]]
    tree = ss.cKDTree(data)  # 2dim-tree
    tree_x = ss.cKDTree(x)  # 1dim-tree
    tree_y = ss.cKDTree(y)  # 1dim-tree
    n, p = data.shape  # number of points, p is the dim of point
    dist_2d, ind_2d = tree.query(data, neig + 1, p=float('inf'))
    Neigh_sum = 0.
    for i in range(n):
        e_x, e_y = np.max(np.fabs(np.tile(data[i, :], (neig + 1, 1)) - data[ind_2d[i, :]]), 0)

        if borders:

            nx = tree_x.query_ball_point([data[i, 0]], e_x, p=float('inf'))
            ny = tree_y.query_ball_point([data[i, 1]], e_y, p=float('inf'))
        else:
            nx = tree_x.query_ball_point([data[i, 0]], e_x - 1e-15, p=float('inf'))
            ny = tree_y.query_ball_point([data[i, 1]], e_y - 1e-15, p=float('inf'))

        if center:
            Neigh_sum += (special.digamma(len(nx)) + special.digamma(len(ny))) / n  # including center point
        else:
            Neigh_sum += special.digamma(len(nx) - 1) + special.digamma(len(ny) - 1) / n  # not including center point

    return special.digamma(neig) - (1 / neig) - Neigh_sum + special.digamma(n)


def coupled_gauss_map(A, noise, init_vec, N):
    """
    $x(t+1) = A x(t) + noise $
    :param A: linear coupling matrix
    :param noise: mean zero, covariance 1 independent gaussian noise
    :param init_vec: initial values of discrete-time map
    :param N: number of length of trajectory
    :return: trajectory of the system


    """
    n, p = np.shape([init_vec])
    empty_array = np.zeros([N, p])
    for i in range(N):
        empty_array[i, :] = init_vec
        x1 = np.dot(A, init_vec) + noise[i, :]
        init_vec = x1

    return empty_array

def mi_cov(data):
    # n, p = data.shape
    x = data[:, [0]]
    y = data[:, [1]]

    cov_x = np.cov(x.T)
    cov_y = np.cov(y.T)

    det_cov_xy = np.linalg.det(np.cov(data.T))

    if np.size(cov_x) ==1:
        c_x = cov_x

    else:
       c_x = np.linalg.det(cov_x)

    if np.size(cov_y) == 1:
        c_y = cov_y

    else:
        c_y = np.linalg.det(cov_y)

    return 0.5 * np.log((c_x * c_y) / det_cov_xy)


def cse_cov(data, conditional_data, data_delayed):
    """
      :param data: first observation vector
      :param conditional_data: causation entropy conditional observation vector
      :param data_delayed: time-delayed second observation vector and its given as data_delayed

      :return: causation entropy from first observation to second on condition given conditional data for gauss r.v.

      """
    # n, p = data.shape

    if np.size(conditional_data) == 0:
        data_all = np.hstack((data_delayed, data))

        return mi_cov(data_all)

    else:
        data_all = np.hstack((data_delayed, data, conditional_data))
        det_cov_data_all = np.linalg.det(np.cov(data_all.T))
        # --------------------------------------
        cov_z = np.cov(conditional_data.T)

        if np.size(cov_z) == 1:
            c_z = cov_z

        else:
            c_z = np.linalg.det(cov_z)

        # --------------------
        datayn_z = np.hstack((data_delayed, conditional_data))
        det_cov_ynz = np.linalg.det(np.cov(datayn_z.T))

        # -----------------------
        datax_z = np.hstack((data, conditional_data))
        det_cov_x_z = np.linalg.det(np.cov(datax_z.T))

        return 0.5 * np.log((det_cov_ynz * det_cov_x_z) / (det_cov_data_all * c_z))


# alternative form for comparison with np.corrcoef


def cse_corrcoef(data, conditional_data, data_delayed):
    if np.size(conditional_data) == 0:
        data_all = np.hstack((data_delayed, data))
        x = data_all[:, [0]]
        y = data_all[:, [1]]

        cov_x = np.corrcoef(x.T)
        cov_y = np.corrcoef(y.T)

        det_cov_xy = np.linalg.det(np.corrcoef(data_all.T))

        if np.size(cov_x) == 1:
            c_x = 1

        else:
            c_x = np.linalg.det(cov_x)

        if np.size(cov_y) == 1:
            c_y = 1

        else:
            c_y = np.linalg.det(cov_y)

        return 0.5 * np.log((c_x * c_y) / det_cov_xy)


    else:

        Cz = 1

        xz = np.hstack([data, conditional_data])

        Cxz = np.linalg.det(np.corrcoef(xz.T))

        yz = np.hstack([data_delayed, conditional_data])

        Cyz = np.linalg.det(np.corrcoef(yz.T))

        xyz = np.hstack([data, data_delayed, conditional_data])

        Cxyz = np.linalg.det(np.corrcoef(xyz.T))

        return 0.5 * np.log((Cxz * Cyz) / (Cz * Cxyz))


# main part starts....
# as an alternative calculation

N = 3

A = np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0]])

mean = np.zeros(N)
cov = np.eye(N)
T = 2000 # time steps

np.random.seed(1)
noise = np.random.multivariate_normal(mean, cov, T)
init_vec = np.random.randn(N)

data = coupled_gauss_map(A, noise[0:T, :], init_vec, T)

# only-one time delay

x = data[:-1, [0]]
y = data[:-1, [1]]
z = data[:-1, [2]]

x_delayed = data[1:, [0]]
y_delayed = data[1:, [1]]
z_delayed = data[1:, [2]]


neigh = 10

'''
print('*********')
print('cse_xyz', cse_cov(x, z, y_delayed))
print('cse_xyz_coeff', cse_corrcoef(x, z, y_delayed))
# print('cse_xyz_knn', cse_new(x, z, y_delayed, neigh))

# -------------------
print('*********')
print('cse_zyy', cse_cov(z, y, y_delayed))
print('cse_zyy_coeff', cse_corrcoef(z, y, y_delayed))
# print('cse_zyy_knn', cse_new(z, y, y_delayed, neigh))
##  print('te_zy_knn', te_new(y_delayed, y, z, neigh))
# ************
print('*********')
print('cse_yxx', cse_cov(y, x, x_delayed))
print('cse_yxx_coeff', cse_corrcoef(y, x, x_delayed))
# print('cse_yxx_knn', cse_new(y, x, x_delayed, neigh))

# --------------------
print('*********')
print('cse_xzy', cse_cov(x, y, z_delayed))
print('cse_xzy_coeff', cse_corrcoef(x, y, z_delayed))
# print('cse_xzy_kNN', cse_new(x, y, z_delayed, neigh))
# ***********
print('*********')
print('cse_zyx', cse_cov(z, x, y_delayed))
print('cse_zyx_coeff', cse_corrcoef(z, x, y_delayed))
# print('cse_zyx_kNN', cse_new(z, x, y_delayed, neigh))

# ***********
print('*********')
print('cse_yxz', cse_cov(y, z, x_delayed))
print('cse_yxz_coeff', cse_corrcoef(y, z, x_delayed))
print('cse_yxz_kNN', cse_new(y, z, x_delayed, neigh))
# ***********
print('*********')
print('cse_yzx', cse_cov(y, x, z_delayed))
print('cse_yzx_coeff', cse_corrcoef(y, x, z_delayed))
# print('cse_yzx_kNN', cse_new(y, x, z_delayed, neigh))
# -------------


# ***********
print('*********')
print('cse_zxy', cse_cov(z, y, x_delayed))
print('cse_zxy_coeff', cse_corrcoef(z, y, x_delayed))
# print('cse_zxy_kNN', cse_new(z, y, x_delayed, neigh))
# -------------
'''

print('*********')
print('cse_xy', cse_cov(y_delayed, [], x))
print('cse_xy_coeff', cse_corrcoef(x, [], y_delayed))
print('cse_xy_knn', cse_new(y_delayed, [], x, neigh))
'''
print('*********')
print('cse_yx', cse_cov(x_delayed, [], y))
print('cse_yx_coeff', cse_corrcoef(x_delayed, [], y))
# print('cse_yx_knn', cse_new(x_delayed, [], y, neigh))

print('*********')
print('cse_xz', cse_cov(z_delayed, [], x))
print('cse_xz_coeff', cse_corrcoef(z_delayed, [], x))
# print('cse_xz_kNN', cse_new(z_delayed, [], x, neigh))
print('*********')
print('cse_zx', cse_cov(x_delayed, [], z))
print('cse_zx_coeff', cse_corrcoef(x_delayed, [], z))
# print('cse_zx_kNN', cse_new(x_delayed, [], z, neigh))
print('*********')
print('cse_yz', cse_cov(z_delayed, [], y))
print('cse_yz_coeff', cse_corrcoef(z_delayed, [], y))
# print('cse_yz_kNN', cse_new(z_delayed, [], y, neigh))
print('*********')
print('cse_zy', cse_cov(y_delayed, [], z))
print('cse_zy_coeff', cse_corrcoef(y_delayed, [], z))
# print('cse_zy_kNN', cse_new(y_delayed, [], z, neigh))


'''


