"""
Time delay determination via mutual information.

@author: Ozge Canli Usta
"""


from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial as ss
from scipy import special
from mpl_toolkits.mplot3d  import Axes3D
import matplotlib as mpl

def lorenz(x, t, sigma=10.0, b=8.0/3.0, r=28.0):
    """
    Lorenz system
    Args:
        x: initial values
        t: time
        sigma: inner parameters of Lorenz
        b: inner parameters of Lorenz
        r: inner parameters of Lorenz

    Returns:
        xdot: right part of differential equation
    """
    xdot = np.array([sigma * (x[1] - x[0]), -x[0] * x[2] + r * x[0] - x[1], x[0] * x[1] - b * x[2]])
    return xdot



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


# ********************
# initialize parameters
T = 50
Td_array = np.arange(1, 100, 2)
Ts = 0.01
transient_index = 1000
t = np.arange(0, T, Ts)
x0 = np.random.rand(3)
res = odeint(lorenz, x0, t)
data = res[transient_index:, [0]]

k = 10 # number of neighbors
I_array = np.zeros([len(Td_array), 1])
for counter, Td in enumerate(Td_array):
    first_signal = data[:-Td].reshape(-1, 1)
    second_signal = data[Td:].reshape(-1, 1)
    s = np.hstack((first_signal, second_signal))

    I = ksg(s, k, borders=False)
    I_array[counter, :] = I


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(res[transient_index:, 0], res[transient_index:, 1], res[transient_index:, 2], label='Lorenz system')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')
plt.grid()

fig = plt.figure()
plt.stem(Td_array, I_array)
plt.title('MI with knn method')
plt.xlabel('time lag')
plt.ylabel('Delayed MI')
plt.grid()

Td = 15
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(res[transient_index:-2*Td, 0], res[transient_index+Td:-Td, 0], 
        res[transient_index+2*Td:, 0], label='Lorenz system')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')
plt.grid()


plt.show()