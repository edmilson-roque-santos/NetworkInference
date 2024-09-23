"""
Methods to plot and computed true mutual information expressions.

Created on Mon Sep 23 14:27:07 2024

@author: Edmilson Roque dos Santos
"""

import os
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.special

# Set plotting parameters
params_plot = {'axes.labelsize': 14,
              'axes.titlesize': 14,
              'axes.linewidth': 1.0,
              'axes.xmargin':0.1, 
              'axes.ymargin': 0.1,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'figure.figsize': (7, 3),
              'figure.titlesize': 15,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm',
              'lines.linewidth': 0.8
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)


#=============================================================================#
#Generate network structure
#=============================================================================#

def cycle_graph(N, filename = None):
    G = nx.cycle_graph(N, create_using=nx.DiGraph())
    
    if filename != None:
        nx.write_edgelist(G, "Network_structure/"+filename+".txt", data=False)
    
    return G 

#=============================================================================#
#Analytical expressions for mutual information
#=============================================================================#
def true_CE_path(id_node, A, epsilon = 1):
    
    N = A.shape[0]
    Epsilon = np.ones(N)*epsilon**2
    
    CE_vector = np.zeros(N)
    id_vec = np.array([0])
    for i in range(N):
        CE_vector[i] = 0.5*A[id_node, i]*np.log(1 + Epsilon[id_vec].sum()/Epsilon[id_node])
        id_vec = np.append(id_vec, [i + 1])
    return CE_vector

def true_CE_cycle(id_node, A, epsilon):
    
    N = A.shape[0]
    CE_vector = np.zeros(N)
    id_vec = np.array([0])
    for i in range(N):
        CE_vector[i] = 0.5*A[id_node, i]*np.log(1/(1 - epsilon**2))
        id_vec = np.append(id_vec, [i + 1])
    return CE_vector

#=============================================================================#
#Comparison method to compute the absolute value of the difference
#=============================================================================#
def comparison(mutual_infos, CE_vector):
    
    T_vector = mutual_infos['T_vector']
    methods = ['Gaussian', 'KNN', 'KNN cdtree']
    ave_comp_vec = np.zeros((len(methods), T_vector.shape[0], CE_vector.shape[0]))
    std_comp_vec = np.zeros((len(methods), T_vector.shape[0], CE_vector.shape[0]))

    for id_method, method in enumerate(methods):        
        for id_T, T in enumerate(mutual_infos[method].keys()):
            seeds = mutual_infos[method][T].keys()
            mi_seed = np.zeros((len(seeds), CE_vector.shape[0]))
            for id_seed, seed in enumerate(seeds):
                mi_seed[id_seed, :] = mutual_infos[method][T][seed]
            
            diff = np.absolute(mi_seed - CE_vector)
            
            ave_comp_vec[id_method, id_T, :] = diff.mean(axis=0)
            std_comp_vec[id_method, id_T, :] = diff.std(axis=0)
            
    return ave_comp_vec, std_comp_vec   

def plot_shaded_area(mutual_infos, CE_vector):

    T_vector = mutual_infos['T_vector']
    sampling_rate = mutual_infos['sampling_rate']
    
    methods = list(mutual_infos.keys())
    ave_comp_vec, std_comp_vec = comparison(mutual_infos, CE_vector)

    nrows = ave_comp_vec.shape[0]
    
    fig, ax = plt.subplots(nrows, 1, sharex=True, dpi = 300,
                           figsize = (5, 6))
    
    for id_row in range(nrows):
        for j_node in range(CE_vector.shape[0]):
            
            ax[id_row].plot(T_vector, ave_comp_vec[id_row, :, j_node], 
                            '-o', 
                            label=r'node {}'.format(j_node))
            ax[id_row].fill_between(T_vector, 
                            ave_comp_vec[id_row, :, j_node]-std_comp_vec[id_row, :, j_node], 
                            ave_comp_vec[id_row, :, j_node]+std_comp_vec[id_row, :, j_node],
                            alpha=0.2)
    
        
        ax[id_row].set_ylabel(r'$\hat{I} - I$')
        title = methods[id_row].replace("_", " ")
        
        ax[id_row].set_title(r'{}'.format(title))
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3)
    ax[2].set_xlabel(r'$T$')
    fig.suptitle('Sampling rate {}'.format(sampling_rate))
    
def plot_error_bar(mutual_infos, CE_vector):
    
    T_vector = mutual_infos['T_vector']
    sampling_rate = mutual_infos['sampling_rate']
    
    methods = list(mutual_infos.keys())
    ave_comp_vec, std_comp_vec = comparison(mutual_infos, CE_vector)

    nrows = ave_comp_vec.shape[0]
    
    fig, ax = plt.subplots(nrows, 1, sharex=True, dpi = 300,
                           figsize = (5, 6))
    
    for id_row in range(nrows):
        for j_node in range(CE_vector.shape[0]):
            
            ax[id_row].errorbar(T_vector, ave_comp_vec[id_row, :, j_node], 
                                std_comp_vec[id_row, :, j_node],
                                fmt = 'o',
                                linewidth = 2,
                                capsize = 6,
                                label=r'node {}'.format(j_node))
    
        ax[id_row].set_ylabel(r'$|\hat{I} - I|$')
        title = methods[id_row].replace("_", " ")
        
        ax[id_row].set_title(r'{}'.format(title))
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3)
    ax[2].set_xlabel(r'$T$')
    fig.suptitle('Sampling rate {}'.format(sampling_rate))

