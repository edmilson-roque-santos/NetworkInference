"""
Network inference - Gaussian process on network from Jeremie Fish

Created on Wed Sep  4 07:04:53 2024

@author: Edmilson Roque dos Santos
"""


from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from NetworkInference import NetworkInference

#GENERATE NETWORK FOR SYNTHETIC DATA
'''
n = 20
p = 0.1
G = nx.erdos_renyi_graph(n,p)
A = nx.adjacency_matrix(G)
'''
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0], 
              [1, 1, 0, 1, 1],
              [0, 0, 1, 0, 1],
              [0, 0, 1, 1, 0]])
#A.todense()

#SET NETWORK STRUCTURE FOR DATA GENERATION
NI = NetworkInference()
NI.set_NetworkAdjacency(A)

#SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
#This sets the number of time steps to simulate. Current default is 100. 
NI.set_T(250)
#This is 0< rho < 1, which is closely related to the signal to noise ratio. The close rho is to 1 the higher the signal to noise...
#for the actual details on the parameter rho please visit the paper: "Causal Network Inference by Optimal Causation Entropy"
NI.set_Rho(0.95)
#Epsilon in the below expression is the variance of the variables in the stochastic process
#This will generate synthetic data which is stored inside
NI.Gen_Stochastic_Gaussian(Epsilon=1)

#Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
#or we can retrieve the data if we wish.

#Retrieve the data
Data = NI.return_XY()
plt.plot(Data)

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