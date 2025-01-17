"""
NetSim python version.

Created on Tue Jan 14 14:56:29 2025

@author: Edmilson Roque dos Santos
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import scipy.io
from scipy.integrate import solve_ivp

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

# Parameters of the input Poisson process driving dynamics
    
N = 2   #Number of nodes in the network
up_duration = 2.5  # mean duration for "up" state (seconds)
down_duration = 10  # mean duration for "down" state (seconds)
noise_std = 1 / 20  # standard deviation for noise
state_heights = {'up': 1, 'down': 0}  # Heights for each state
total_time = 150  # simulation time in seconds
dt = 0.001  # time step for simulation
random_seed = 1
rng = default_rng(random_seed)

# Time array
time_points = np.arange(0, total_time, dt)
external_input = np.zeros((time_points.shape[0], N))


for id_node in range(N):
    # Poisson process for state transitions
    current_state = 'down'
    state_switch_time = 0

    for i, t in enumerate(time_points):
        if t >= state_switch_time:
            if current_state == 'down':
                state_switch_time += rng.exponential(up_duration)
                current_state = 'up'
            else:
                state_switch_time += rng.exponential(down_duration)
                current_state = 'down'
            
        # Generate external input with added noise
        height = state_heights[current_state]
        noise = rng.normal(0, noise_std)
        external_input[i, id_node] = height + noise

#=============================================================================#
plot_input_Poisson = False
if plot_input_Poisson:
    plt.figure(figsize=(6, 3), dpi = 300)
    plt.plot(time_points, external_input, label='External Input with Noise')
    plt.title("External Input Signal with Poisson State Switching and Noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Input Value")
    plt.show()


def linear_DMC(t, z, A, sigma, C, u):
    
    return sigma*(A @ z) + C @ u(t)


#Import input data from NetSim
#folder = 'data/net_sim7/'
#A = scipy.io.loadmat(folder+'adj_sim7.mat')['A'].T
A = np.array([[-1, 0], [0.4, -1]])
C = np.identity(N)
ic = np.zeros(N)
initial_condition = np.array(ic)
u = lambda t: external_input[int(t/dt), :]
sigma = 20
neural_time = 0.99*total_time
sampling_dt = 0.005
t_eval = np.arange(0.0, neural_time, sampling_dt)

sol = solve_ivp(linear_DMC, [0, neural_time], initial_condition,
                method='RK45',
                args=(A, sigma, C, u), t_eval = t_eval, 
                first_step=0.001, 
                max_step = 0.001,
                rtol = 1e-4,
                atol = 1e-5)

X_t = sol.y.T
    
    
plot_neural_timeseries = False
if plot_neural_timeseries:
    plt.figure(figsize=(6, 3), dpi = 300)
    plt.plot(t_eval, X_t, label='Neural Time series')
    plt.title("Neural Time series")
    plt.xlabel(r"Time (s)")
    plt.ylabel(r"$z(t)$")
    plt.show()
    
    
#=============================================================================#

    
    
    
    
    
    
    
    
    
    
