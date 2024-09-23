"""
Comparison between Gaussian formula and KNN method for estimating 
the mutual information.

Created on Mon Sep 16 13:47:11 2024

@author: Edmilson Roque dos Santos
"""

import os
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.special

import h5dict

import utils as uts
from NetworkInference import NetworkInference


def mutual_info_node(A, T, Rho, Tau, SR, K, seed, id_node = 0, method = 'Gaussian'):
    #GENERATE NETWORK FOR SYNTHETIC DATA
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
    NI.Gen_Stochastic_Gaussian(Epsilon=espilon, seed=seed, spec_rad = True)
    
    #Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
    #or we can retrieve the data if we wish.
    
    #Now lets estimate the network structure. 
    #SET ALL NECESSARY PARAMETERS
    #Set the inference method
    NI.set_InferenceMethod_oCSE(method)
    
    NI.set_Tau(Tau)
    NI.set_KNN_K(K)
    
    XY = NI.XY.copy()
    XY_1 = NI.sampling_ts(XY, 0)
    XY_2 = NI.sampling_ts(XY, NI.Tau)
   
    mutual_infos = np.zeros(NI.n)
    
    NI.Y = XY_2[:, [id_node]].copy()
    NI.X = XY_1.copy() 
           
    for i in range(NI.n):
        X = NI.X[:, [i]].copy()
        
        mutual_infos[i] = NI.Compute_CMI(X)
            
    return mutual_infos


Rho = 0.99
Tau = 1
sampling_rate = 10
K = 10
number_seeds = 10
id_node = 1

Ts = [50, 600, 15]
T_vector = np.linspace(Ts[0], Ts[1], Ts[2], dtype = int)


#Number of nodes
N = 3
#Experiment name for saving the results in appropriate filename
exp_name = 'cg'
network_name = 'cycle_graph'
G = nx.read_edgelist("network_structure/{}.txt".format(network_name),
                    nodetype = int, create_using = nx.DiGraph())
A = nx.adjacency_matrix(G)
A = A.todense().T

filename = "minfo_exp_{}_rho_{}_tau_{}_sr_{}_K_{}, ns_{}_node_{}_Ts_{}_{}_{}".format(exp_name,
                                                                                     Rho, Tau,
                                                                                     sampling_rate,
                                                                                     K,
                                                                                     number_seeds, id_node,
                                                                                     Ts[0], Ts[1], Ts[2]) 

out_results_direc = 'results/'

if os.path.isfile(out_results_direc+filename+".hdf5"):
    out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
    mutual_infos = out_results_hdf5.to_dict()  
    out_results_hdf5.close()      
    
else:
    out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')
    
    out_results_hdf5['T_vector'] = T_vector
    out_results_hdf5['sampling_rate'] = sampling_rate
    
    for method in ['Gaussian', 'KNN', 'KNN cdtree']:
        out_results_hdf5[method] = dict()
                
        for T in T_vector:
            out_results_hdf5[method][T] = dict()
            
            for seed in range(1, number_seeds + 1):
                mis = mutual_info_node(A, T, Rho, Tau, sampling_rate, K, seed, id_node, method)
                out_results_hdf5[method][T][seed] = mis

    mutual_infos = out_results_hdf5.to_dict()        
    out_results_hdf5.close()

CE_vector = uts.true_CE_cycle(id_node, A, epsilon = Rho)

uts.plot_error_bar(mutual_infos, CE_vector)
uts.plot_shaded_area(mutual_infos, CE_vector)








