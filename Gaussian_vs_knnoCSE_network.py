"""
Optimal Causation Entropy comparison between Gaussian formula and KNN method

Created on Tue May 14 09:59:57 2024

@author: Edmilson Roque dos Santos
"""

import itertools
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.special

from NetworkInference import NetworkInference

def permut(N, k):
    Range = range(0, N)
    comb = list(itertools.permutations(Range,k))
    return np.array(comb)

def true_CE_path(A, epsilon = 1):
    
    N = A.shape[0]
    Epsilon = np.ones(N)*epsilon**2
    
    pairs = permut(N, 2)
    
    num_pairs = pairs.shape[0]

    CE_vector = np.zeros((num_pairs, 3))
    
    for counter, pair in enumerate(pairs):
        CE_vector[counter, 0] = pair[0] 
        CE_vector[counter, 1] = pair[1]
        CE_vector[counter, 2] = 0.5*A[pair[0] , pair[1]]*np.log(1 + Epsilon[0:pair[1]].sum()/Epsilon[pair[0]])
        
    return CE_vector


def personal_test():
    #Generate network
    
    #GENERATE NETWORK FOR SYNTHETIC DATA
    N = 3
    G = nx.path_graph(N, create_using = nx.DiGraph())
    A = nx.adjacency_matrix(G)
    A = A.todense()
    
    #SET NETWORK STRUCTURE FOR DATA GENERATION
    NI = NetworkInference()
    NI.set_NetworkAdjacency(A)
    
    #SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
    #This sets the number of time steps to simulate. Current default is 100. 
    NI.set_T(50)
    #This is 0< rho < 1, which is closely related to the signal to noise ratio. The close rho is to 1 the higher the signal to noise...
    #for the actual details on the parameter rho please visit the paper: "Causal Network Inference by Optimal Causation Entropy"
    NI.set_Rho(0.99)
    #Epsilon in the below expression is the variance of the variables in the stochastic process
    #This will generate synthetic data which is stored inside
    espilon = 1
    NI.Gen_Stochastic_Gaussian(Epsilon=espilon)
    
    #Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
    #or we can retrieve the data if we wish.
    
    #Retrieve the data
    Data = NI.return_XY()
    plt.plot(Data)
    
    #Calculate the Causation Entropy for the directed path
    CE_true = true_CE_path(A, epsilon = espilon)
    
    pairs = permut(N, 2)
    num_pairs = pairs.shape[0]
    NI.Tau = 2
    CE_dict = dict()
    for k in range(1, 11):
        NI.set_KNN_K(k)
        CE_est = np.zeros((num_pairs, 3))
        
        time_eval = np.arange(0, Data.shape[0] - NI.Tau, NI.Tau, dtype = int)
        
        for counter, pair in enumerate(pairs):
            CE_est[counter, 0] = pair[0] 
            CE_est[counter, 1] = pair[1]
            NI.Y = Data[time_eval+1, [pair[0]]] 
            NI.X = Data[time_eval, [pair[0]]]
            
            NI.Z = Data[time_eval, [pair[1]]] 
            
            YcatZ = np.concatenate((NI.Y,NI.Z),axis=1)
            MIYZ = NI.MutualInfo_KNN(NI.Y,NI.Z)
            MIYZX = NI.MutualInfo_KNN(YcatZ,NI.X)
           
            CE_est[counter, 2] = MIYZ - MIYZX
        
        CE_dict[k] = CE_est

def net_inference(T, Rho, Tau, SR, K, seed, method = 'Gaussian'):
    #GENERATE NETWORK FOR SYNTHETIC DATA
    N = 3
    G = nx.path_graph(N, create_using = nx.DiGraph())
    A = nx.adjacency_matrix(G)
    A = A.todense()
    
    #SET NETWORK STRUCTURE FOR DATA GENERATION
    NI = NetworkInference()
    NI.set_NetworkAdjacency(A)
    
    #SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
    #This sets the number of time steps to simulate. Current default is 100. 
    NI.set_T(T)
    NI.set_sampling_rate(SR)
    #This is 0< rho < 1, which is closely related to the signal to noise ratio. The close rho is to 1 the higher the signal to noise...
    #for the actual details on the parameter rho please visit the paper: "Causal Network Inference by Optimal Causation Entropy"
    NI.set_Rho(Rho)
    #Epsilon in the below expression is the variance of the variables in the stochastic process
    #This will generate synthetic data which is stored inside
    espilon = 1
    NI.Gen_Stochastic_Gaussian(Epsilon=espilon)
    
    #Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
    #or we can retrieve the data if we wish.
    
    #Now lets estimate the network structure. 
    #SET ALL NECESSARY PARAMETERS
    #Set the inference method
    NI.set_InferenceMethod_oCSE(method)
    
    NI.set_Tau(Tau)
    NI.set_KNN_K(K)
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

    return B, NI

def lgth_data():
    
    Rho = 0.3
    Tau = 1
    K = 1
    number_seeds = 10
    
    T_vector = np.linspace(100, 1000, 5, dtype = int)
    
    entropies = dict()

    for method in ['Gaussian', 'KNN']:
        entropies[method] = dict()
        for T in T_vector:
            entropies[method][T] = dict()
            for seed in range(1, number_seeds + 1):
                B, NI = net_inference(T, Rho, Tau, K, seed, method)
                entropies[method][T][seed] = NI.Ents_dict        
        
    return entropies

Rho = 0.3
Tau = 1
sampling_rate = 100
K = 10
number_seeds = 1

T_vector = np.linspace(100, 1000, 2, dtype = int)

graphs = dict()
entropies = dict()

for method in ['Gaussian', 'KNN']:
    entropies[method] = dict()
    graphs[method] = dict()
    for T in T_vector:
        entropies[method][T] = dict()
        graphs[method][T] = dict()
        for seed in range(1, number_seeds + 1):
            B, NI = net_inference(T, Rho, Tau, sampling_rate, K, seed, method)
            entropies[method][T][seed] = NI.Ents_dict
            graphs[method][T][seed] = B
