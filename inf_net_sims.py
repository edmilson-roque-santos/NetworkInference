"""
Computing the AFNs from the NetSim data set.
This is a preliminary test to see the performance in identifying the underlying
network structure.

Created on Wed Dec 11 10:24:10 2024

@author: Edmilson Roque dos Santos
"""

import itertools
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, stats
import scipy.special
from scipy import spatial as ss
import scipy.io

from NetworkInference import NetworkInference
#=============================================================================#
#=============================================================================#
# Set plotting parameters
params_plot = {'axes.labelsize': 16,
              'axes.titlesize': 18,
              'axes.linewidth': 1.0,
              'axes.xmargin':0.0, 
              'axes.ymargin': 0.1,
              'legend.fontsize': 10,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.figsize': (7, 3),
              'figure.titlesize': 18,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm',
              'lines.linewidth': 1.2
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)
#=============================================================================#
#=============================================================================#
#============================##============================##============================#
#Auto correlation function for the input time series     
def x_corr(sign1, sign2, normalize = False):
    
    n = np.max([sign1.shape[0], sign2.shape[0]])
    
    if normalize:
        s1 = (sign1 - sign1.mean())/(sign1.std()*n)
        s2 = (sign2 - sign2.mean())/sign2.std()
    else:    
        s1 = (sign1)/(np.sqrt(n))
        s2 = (sign2)/np.sqrt(n)
        
    lags = signal.correlation_lags(s1.size, s2.size, mode="full")
    cx = signal.correlate(s1, s2, mode='full')
   
    return lags, cx    
    
def plot_corr(X_t, index, normalize, id_xlim = 500):
    
    fig, ax = plt.subplots(2, 1, sharex = True, dpi = 300, figsize = (5, 5))
    
    for id_ in index:
        lags, cx = x_corr(X_t[:, id_], X_t[:, id_], normalize)
        lgn = int(lags.shape[0]/2)
        ax[id_].plot(lags[lgn:lgn+id_xlim], cx[lgn:lgn+id_xlim])
       
        ax[id_].set_ylabel(r"Auto correlation")
        
    ax[id_].set_xlabel(r"$\tau$")
    
#============================##============================##============================#
#Mutual information for the input time series     
    
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
            Neigh_sum += (scipy.special.digamma(len(nx)) + scipy.special.digamma(len(ny))) / n  # including center point
        else:
            Neigh_sum += scipy.special.digamma(len(nx) - 1) + scipy.special.digamma(len(ny) - 1) / n  # not including center point

    return scipy.special.digamma(neig) - (1 / neig) - Neigh_sum + scipy.special.digamma(n)

def mutual_info(X_t, index, id_xlim = 250):
    k = 10 # number of neighbors
    
    Td_array = np.arange(1, id_xlim, 2)
    I_array = np.zeros([len(Td_array), len(index)])
    
    for id_ in index:
        data = X_t[:, id_]
        
        for counter, Td in enumerate(Td_array):
            first_signal = data[:-Td].reshape(-1, 1)
            second_signal = data[Td:].reshape(-1, 1)
            s = np.hstack((first_signal, second_signal))
        
            I = ksg(s, k, borders=False)
            I_array[counter, id_] = I    
    
    return I_array, Td_array
    
def plot_MI(X_t, index, id_xlim = 250):
        
    I_array, Td_array = mutual_info(X_t, index, id_xlim)
    
    fig, ax = plt.subplots(len(index), 1, sharex = True, dpi = 300, figsize = (5, 5))
    
    for id_ in index:
        ax[id_].plot(Td_array, I_array[:, id_])
       
        ax[id_].set_ylabel(r"MI")
        
    ax[id_].set_xlabel(r"$\tau$")  
    

#=============================================================================#
#Import input data from NetSim
folder = 'data/net_sim7/'
A = scipy.io.loadmat(folder+'adj_sim7.mat')['A'].T
X_time_series = scipy.io.loadmat(folder+'ts_sim7.mat')['ts']
Ntimepoints = 5000
Nsubject = 1
data = X_time_series[Nsubject*Ntimepoints:(Nsubject + 1)*Ntimepoints, :].T
T, N = data.shape
#============================##============================##============================#
plot_netsim_data = False
if plot_netsim_data:
    
    fig = plt.figure(figsize=(5,2), dpi = 300)
    plt.plot(data[0, :])
    plt.plot(data[1, :])
    plt.plot(data[2, :])
    plt.plot(data[3, :])
    
    #Plot return map:
    plot_return_map = False
    if plot_return_map:
        fig = plt.figure(figsize=(5,2), dpi = 300)
        plt.plot(data[0, :-1], data[0, 1:])
        plt.plot(data[1, :-1], data[1, 1:])
    
    plot_corr(data.T, index=[0, 1], normalize=True)    

plot_signals = True
if plot_signals:
    fig, ax = plt.subplots(3, 1, figsize=(6, 5), dpi = 300)
    
    for index in [0, 1, 2, 3, 4]:
        ax[0].plot(data[index, :],
                   markersize=5)

        lower_bound, upper_bound = data[index, :].min(), data[index, :].max()
        interval = np.arange(lower_bound, upper_bound, 0.001)
        kernel = stats.gaussian_kde(data[index, :])
        ax[1].plot(interval, 
                   kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                   label="{}".format(index))
    
    ax[1].hlines(0.0, lower_bound, upper_bound, color = 'k', linestyle = '--')
    n_bins = 100
    
    ax[2].hist(data[0, :], n_bins)
    

plot_mutual_info = False
if plot_mutual_info:
    plot_MI(X_time_series, index = [0, 1, 2, 3, 4], id_xlim = 30)

estimate_net = False

if estimate_net:
    #Calculate the tau from first minimum of mutual information.
    #From visual inspection, we can test initially
    Tau = 1
    #============================##============================##============================#
    #=============================================================================#
    
    #SET NETWORK STRUCTURE FOR DATA GENERATION
    NI = NetworkInference()
    NI.set_NetworkAdjacency(A)
    NI.XY = X_time_series
    #SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
    #This sets the number of time steps to simulate. Current default is 100. 
    NI.set_T(T)
    NI.set_sampling_rate(1)
    
    #Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
    #or we can retrieve the data if we wish.
    
    #Now lets estimate the network structure. 
    #SET ALL NECESSARY PARAMETERS
    #Set the inference method
    method = 'Gaussian'
    NI.set_InferenceMethod_oCSE(method)
    
    NI.set_Tau(Tau)
    NI.set_KNN_K(10)
    #Set the number of shuffles (see the Sun, Taylor, Bollt paper for more details)
    NI.set_Num_Shuffles_oCSE(100)
    
    #Set the alpha value (in the Sun, Taylor, Bollt paper alpha = 1- theta that they used). 
    #Essentially this is like a p-value and it is your level of confidence in an edge. 
    #The lower alpha the more confident you are in the edges it finds.
    NI.set_Forward_oCSE_alpha(0.001)
    #There is a forward and a backward stage to the oCSE algorithm...
    NI.set_Backward_oCSE_alpha(0.001)
    
    #Now actually estimate the network using the Gaussian oCSE method. This may take some time, but it will print
    #progress in terms of which node number it is working on (starting from node 0) to estimate incoming edges.
    #note that the run time is mainly dependent on the number of EDGES not the number of nodes...
    B = NI.Estimate_Network()
