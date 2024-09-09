"""
Cluster synchronization for Gaussian processes on network

Created on Tue Sep  3 16:32:28 2024

@author: Edmilson Roque dos Santos
"""

import itertools
import networkx as nx
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import scipy.special

from NetworkInference import NetworkInference
import networkx as nx
from matplotlib import pyplot as plt

colors = ['darkgrey', 'orange', 'darkviolet', 'darkslategrey', 'silver']

# Set plotting parameters
params_plot = {'axes.labelsize': 15,
              'axes.titlesize': 15,
              'axes.linewidth': 1.0,
              'axes.xmargin':0, 
              'axes.ymargin': 0,
              'legend.fontsize': 18,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.figsize': (8, 6),
              'figure.titlesize': 18,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm',
              'axes.linewidth': 1.0
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)


def stochastic_Gaussian_cluster(seed, 
                                Epsilon, 
                                n, 
                                cluster_list, 
                                A,
                                Rho,
                                T):
    np.random.seed(seed)
    #R = 2*(np.random.rand(n, n)-0.5)
    A = np.array(A)# * R
    
    spec_radius = np.max(np.abs(np.linalg.eigvals(A)))
    
    if spec_radius > 0:
        A = A/np.max(np.abs(np.linalg.eigvals(A)))
        
    A = A*Rho

    XY = np.zeros((T, n))
    
    for id_cluster, cluster in enumerate(cluster_list):
        XY[0, cluster] = Epsilon*np.random.randn(1)
    
    for i in range(1, T):
        
        Xi = A @ XY[i - 1, :]

        for cluster in cluster_list:
            Xi[cluster] = Xi[cluster] + Epsilon*np.random.randn(1)
            
        XY[i, :] = Xi
        
    return XY 

def cluster_state_GN(cluster_list, T = 250, Rho = 0.95):
    
    A = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0], 
                  [1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0]])
    
    seed = 1
    Epsilon = 1
    n = A.shape[0]
        
    #SET NETWORK STRUCTURE FOR DATA GENERATION
    NI = NetworkInference()
    NI.set_NetworkAdjacency(A)
    
    #SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
    #This sets the number of time steps to simulate. Current default is 100. 
    NI.set_T(T)
    #This is 0< rho < 1, which is closely related to the signal to noise ratio. The close rho is to 1 the higher the signal to noise...
    #for the actual details on the parameter rho please visit the paper: "Causal Network Inference by Optimal Causation Entropy"
    NI.set_Rho(Rho)
    #Epsilon in the below expression is the variance of the variables in the stochastic process
    #This will generate synthetic data which is stored inside
    #NI.Gen_Stochastic_Gaussian(Epsilon=1)
    
    Data = stochastic_Gaussian_cluster(seed, Epsilon, n, cluster_list, A, Rho, T)
    #Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
    #or we can retrieve the data if we wish.
    
    #Retrieve the data
    #Data = NI.return_XY()
    #plt.plot(Data)
    NI.XY = Data
    
    #Now lets estimate the network structure. 
    #SET ALL NECESSARY PARAMETERS
    #Set the inference method to Gaussian
    NI.set_InferenceMethod_oCSE('Gaussian')
    
    #Set the number of shuffles (see the Sun, Taylor, Bollt paper for more details)
    NI.set_Num_Shuffles_oCSE(1000)
    
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
    
    #Now lets see how we did. We will calculate the true positive and false positive rates.
    TPR,FPR = NI.Compute_TPR_FPR()
    print("This is the TPR and FPR: ",TPR,FPR)
    
    #Finally if we wish we can save the state of all of the parameters we just used, that way we could 
    #come back later and use the same data and number of shuffles and alpha and so on... State will be 
    #saved in the local directory. To load a saved state use .load_state(). You may have to set a date
    #in load_state() and a saved number though as the default is to load the previously saved state from today. 
    #NI.save_state()
    
    return A, B

def quotient_GN(T = 250, Rho = 0.95):
    
    Q = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0], 
                  [1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0]])
    
        
    E = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1]])
    
    A = LA.pinv(E) @ Q @ E
    A[np.absolute(A) > 1] = 1
    
    seed = 1
    Epsilon = 1
    n = A.shape[0]
    
    #SET NETWORK STRUCTURE FOR DATA GENERATION
    NI = NetworkInference()
    NI.set_NetworkAdjacency(A)
    
    #SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
    #This sets the number of time steps to simulate. Current default is 100. 
    NI.set_T(T)
    #This is 0< rho < 1, which is closely related to the signal to noise ratio. The close rho is to 1 the higher the signal to noise...
    #for the actual details on the parameter rho please visit the paper: "Causal Network Inference by Optimal Causation Entropy"
    NI.set_Rho(Rho)
    #Epsilon in the below expression is the variance of the variables in the stochastic process
    #This will generate synthetic data which is stored inside
    NI.Gen_Stochastic_Gaussian(Epsilon=1)
    
    #Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
    #or we can retrieve the data if we wish.
    
    #Retrieve the data
    Data = NI.return_XY()
    #plt.plot(Data)
        
    #Now lets estimate the network structure. 
    #SET ALL NECESSARY PARAMETERS
    #Set the inference method to Gaussian
    NI.set_InferenceMethod_oCSE('Gaussian')
    
    #Set the number of shuffles (see the Sun, Taylor, Bollt paper for more details)
    NI.set_Num_Shuffles_oCSE(1000)
    
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
    
    #Now lets see how we did. We will calculate the true positive and false positive rates.
    TPR,FPR = NI.Compute_TPR_FPR()
    print("This is the TPR and FPR: ",TPR,FPR)
    
    #Finally if we wish we can save the state of all of the parameters we just used, that way we could 
    #come back later and use the same data and number of shuffles and alpha and so on... State will be 
    #saved in the local directory. To load a saved state use .load_state(). You may have to set a date
    #in load_state() and a saved number though as the default is to load the previously saved state from today. 
    #NI.save_state()
    
    return A, B
 
def links_types(G, G_true):
    '''
    Compare a graph G and edges set.

    Parameters
    ----------
    G : graph
        A networkx graph.
    edges_G_true : set
        Edges set from graph that is considered true.

    Returns
    -------
    links : dict
        A dictionary keyed by type of links in the comparison, false positive,
        false negatives, or intersection.

    '''
    
    edges_G_true = G_true.edges()
    edges_G_true = set(edges_G_true)
    
    links = dict()    
    
    links['set_edges_G_estimated'] = set(G.edges())
    links['intersection'] = list(links['set_edges_G_estimated'] & edges_G_true)
    
    links['false_positives'] = list(links['set_edges_G_estimated'] - edges_G_true)
        
    links['false_negatives'] = list(edges_G_true - links['set_edges_G_estimated'])
    
    return links    

def plot_comparison(ax, A, A_est, nodecolors):    

    G_true = nx.from_numpy_array(A, create_using = nx.DiGraph())
    G = nx.from_numpy_array(A_est, create_using = nx.DiGraph())
    
    pos = nx.spring_layout(G)
    
    links_CGN = links_types(G, G_true)
    intersection = links_CGN['intersection']
    false_positives = links_CGN['false_positives']
    false_negatives = links_CGN['false_negatives']
    
    nx.draw_networkx_nodes(G, pos = pos, ax = ax,
                           node_color = nodecolors,
                           node_size = 150,
                           alpha = 1.0)
        
    nx.draw_networkx_edges(G, pos = pos, edgelist = intersection,
                           ax = ax,
                           edge_color = colors[4], alpha = 1.0)
  
    nx.draw_networkx_edges(G, pos = pos, edgelist = false_positives, 
                           edge_color = colors[1],
                           ax = ax,
                           alpha = 1.0,
                           connectionstyle='arc3,rad=0.2')
    
    nx.draw_networkx_edges(G,pos = pos, edgelist = false_negatives, 
                           edge_color = colors[2], 
                           ax = ax,
                           width = 1.0,
                           connectionstyle='arc3,rad=0.2')
    
    ax.margins(0.10)
    ax.axis("off")
    
def fig_comparison(T = 250, Rho = 0.95, filename = None):
    
    fig, ax1 = plt.subplots(1, 2, figsize=(6, 4), dpi = 300)
    
    ax = ax1[0]
    ax.set_title(r'a) Gaussian process on cluster')
    
    #Gaussian network on Cluster state
    cluster_list = [np.array([0]), np.array([1]), np.array([2]), np.array([3, 4])]
    
    nodecolors = ['tab:orange', 'tab:green', 'tab:cyan', 'tab:purple', 'tab:purple']
    
    A, A_est = cluster_state_GN(cluster_list, T = T, Rho = Rho)
    plot_comparison(ax, A, A_est, nodecolors)
    
    ax = ax1[1]
    ax.set_title(r'b) Quotient dynamics')
    nodecolors = ['tab:orange', 'tab:green', 'tab:cyan', 'tab:purple']
    
    A, A_est = quotient_GN(T = T, Rho = Rho)
    plot_comparison(ax, A, A_est, nodecolors)
    
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
    

