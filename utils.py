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
from scipy import signal, stats

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
        for id_T, T in enumerate(T_vector):
            seeds = mutual_infos[method][T].keys()
            mi_seed = np.zeros((len(seeds), CE_vector.shape[0]))
            for id_seed, seed in enumerate(seeds):
                mi_seed[id_seed, :] = mutual_infos[method][T][seed]
            
            diff = np.absolute(mi_seed - CE_vector)
            
            ave_comp_vec[id_method, id_T, :] = diff.mean(axis=0)
            std_comp_vec[id_method, id_T, :] = diff.std(axis=0)
            
    return ave_comp_vec, std_comp_vec   

def statistics(mutual_infos, CE_vector):
    
    T_vector = mutual_infos['T_vector']
    methods = ['Gaussian', 'KNN', 'KNN cdtree']
    ave_comp_vec = np.zeros((len(methods), T_vector.shape[0], CE_vector.shape[0]))
    std_comp_vec = np.zeros((len(methods), T_vector.shape[0], CE_vector.shape[0]))

    for id_method, method in enumerate(methods):        
        for id_T, T in enumerate(T_vector):
            seeds = mutual_infos[method][T].keys()
            mi_seed = np.zeros((len(seeds), CE_vector.shape[0]))
            for id_seed, seed in enumerate(seeds):
                mi_seed[id_seed, :] = mutual_infos[method][T][seed]
            
            diff = mi_seed
            
            ave_comp_vec[id_method, id_T, :] = diff.mean(axis=0)
            std_comp_vec[id_method, id_T, :] = diff.std(axis=0)
            
    return ave_comp_vec, std_comp_vec   


def plot_shaded_area(mutual_infos, CE_vector):

    T_vector = mutual_infos['T_vector']
    sampling_rate = mutual_infos['sampling_rate']
    
    methods = list(mutual_infos.keys())
    ave_comp_vec, std_comp_vec = statistics(mutual_infos, CE_vector)

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
    
            ax[id_row].hlines(CE_vector[j_node], T_vector[0], T_vector[-1], 
                              color = 'tab:red',
                              linestyle = 'dashed')
        
        ax[id_row].set_ylabel(r'$\hat{I}$')
        title = methods[id_row].replace("_", " ")
        
        ax[id_row].set_title(r'{}'.format(title))
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.1),
          ncol=1)
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
    
    ax[id_row].set_xscale('log')
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3)
    ax[2].set_xlabel(r'$T$')
    fig.suptitle('Sampling rate {}'.format(sampling_rate))


from scipy.stats.kde import gaussian_kde

def ridgeline(ax, data,
              y_axis,
              node, 
              overlap=0, 
              fill=True, 
              alpha = 1.0,
              labels=None, 
              n_points=250,
              limits = [-1, 1]):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(limits[0]),
                     np.max(limits[1]), n_points)
    curves = []
    ys = []
    plot_legend = True
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y =  y_axis[i]*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            if plot_legend:
                ax.fill_between(xx, np.ones(n_points)*y, 
                                curve+y, zorder=len(data)-i+1, 
                                color=fill,
                                alpha = alpha,
                                label = r'node {}'.format(node))
                
                plot_legend = False
            
            if not plot_legend:
                ax.fill_between(xx, np.ones(n_points)*y, 
                                curve+y, zorder=len(data)-i+1, 
                                color=fill,
                                alpha = alpha)

        
        ax.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
    if labels:
        lab = [r"${}$".format(y_axis[::-1][i]) for i in range(y_axis.shape[0])]
        ax.set_yticks(ys, lab)
    
    
    
def plot_ridgeline_method(ax, mutual_infos, nodelist, 
                          method = 'KNN cdtree', 
                          overlap = 0.95,
                          limits = [-0.2, 0.2],
                          seed = 1,
                          add_info = True):
    
    T_vector = mutual_infos['T_vector']
       
    #methods = ['Gaussian', 'KNN', 'KNN cdtree']
    
    alpha = [1.0, 0.6]
    colors = ['tab:purple', 'tab:green']
    
    #fig, ax = plt.subplots(1, 1, sharex=True, dpi = 300,
    #                       figsize = (5, 6))
    
    for j_node in nodelist:
            
        data = [mutual_infos[method][t][seed][j_node, :] for t in T_vector[::-1]]
        ridgeline(ax, data, T_vector, node = j_node,
                  overlap=overlap, 
                  fill=colors[j_node],
                  alpha = alpha[j_node],
                  labels = True,
                  limits = limits)
    
    title = method.replace("_", " ")
    ax.set_title(r'{}'.format(title))
    ax.set_xlabel(r'$\hat{I}$')
    
    
    if add_info:
        ax.legend(loc=0)
        ax.set_ylabel(r'$T$')
        
 
def plot_fig_rls(mutual_infos, nodelist, filename):
    
    fig, ax = plt.subplots(1, 3, dpi = 300,
                           figsize = (10, 6))
    
    plot_ridgeline_method(ax[0], mutual_infos, nodelist, 
                              method = 'Gaussian', 
                              overlap = 0.01,
                              limits = [-0.02, 0.02],
                              seed = 1)
    
    plot_ridgeline_method(ax[1], mutual_infos, nodelist, 
                              method = 'KNN', 
                              overlap = 0.95,
                              limits = [-0.2, 0.2],
                              seed = 1,
                              add_info = False)
    
    plot_ridgeline_method(ax[2], mutual_infos, nodelist, 
                              method = 'KNN cdtree', 
                              overlap = 0.95,
                              limits = [-0.2, 0.2],
                              seed = 1,
                              add_info = False)
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
     
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        

def plot_GP(data):
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), dpi = 300)
    
    for index in range(data.shape[0]):
        ax[0].plot(data[index, :],
                   markersize=5)
        ax[0].set_xlabel(r'time')
        ax[0].set_ylabel(r'$X_t$')
        
        lower_bound, upper_bound = data[index, :].min(), data[index, :].max()
        interval = np.arange(lower_bound, upper_bound, 0.001)
        kernel = stats.gaussian_kde(data[index, :])
        ax[1].plot(interval, 
                   kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                   label="{}".format(index))
        
        ax[1].set_xlabel(r'$X$')
        ax[1].set_ylabel(r'Histogram')
                
    ax[1].hlines(0.0, lower_bound, upper_bound, color = 'k', linestyle = '--')    
    plt.tight_layout()
    plt.show()