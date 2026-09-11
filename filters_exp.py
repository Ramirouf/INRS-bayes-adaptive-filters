import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from numba_progress import ProgressBar
from collections import namedtuple

from numpy.typing import NDArray

std_env_parameters = namedtuple('std_env_parameters', ['ho', 
                                                       'AR',
                                                       'var_v',
                                                       'var_x'])
laplace_env_parameters = namedtuple('laplace_env_parameters', ['ho', 
                                                               'AR',
                                                               'scale_v',
                                                               'var_x'])

from filters import (
    # parameter dtypes
    NLMS_params, sKF_params, sKF_L_params, skf_int_params, skf_L_int_params,
    # signal / environment helpers
    autocorr_matrix_calc, AR_settling_time, 
    filter, shift,
    # algorithms
    NLMS_algorithm, sKF_algorithm, sKF_L_algorithm, sKF_L_exact_algorithm,
    sKF_integral_algorithm, sKF_L_integral_algorithm,
    # monte carlo driver
    MC_Simulations_Modular_Variance,
)

@njit(cache=True)
def _generate_normal_input_signal(N:int, AR: NDArray[np.float64], warm_up: bool = True):
    # Determine the input signal x through a AR process
    settling_time = AR_settling_time(AR, error = 0.001)*warm_up
    x = np.random.randn(N + settling_time)
    
    # Generate the correlated signal
    AR = AR/AR[0]
    aux_Rxx = autocorr_matrix_calc(AR, 1, M = len(AR) - 1)
    b = np.sqrt(1/aux_Rxx[0,0])
    x = filter(AR, np.array([b]), x)[settling_time:]
    
    return x

@njit(cache=True)
def std_behavior(N: int, params: std_env_parameters, warm_up: bool = True):
    # AR process order and filter length
    L = len(params.ho)

    # Determine the noise signal
    v = np.sqrt(params.var_v)*np.random.randn(N)

    # Determine the input signal x through a AR process
    x = _generate_normal_input_signal(N + L, params.AR, warm_up = warm_up)

    # Determine the desired signal
    d = v + np.convolve(params.ho, x, mode = 'full')[L:N+L]
    x = x[L:N+L]

    return {'x': x, 'v': v, 'd': d, 'ho_hist': params.ho}

@njit(cache=True)
def laplace_noise_behavior(N: int, params: laplace_env_parameters, warm_up: bool = True):
#(N, ho, var_x, scale_v, AR, settling_time = 0):
    # AR process order and filter length
    L = len(params.ho)

    # Determine the noise signal
    v = np.random.laplace(loc=0.0, scale=params.scale_v, size=(N,))
    
    # Determine the input signal x through a AR process
    x = _generate_normal_input_signal(N + L, params.AR, warm_up = warm_up)

    # Determine the desired signal
    d = v + np.convolve(params.ho, x, mode = 'full')[L:N+L]
    x = x[L: N+L ]

    return {'x': x, 'v': v, 'd': d, 'ho_hist': params.ho}

def MC_Simulations(N, 
                   NR,
                   environment_parameters,
                   environment,
                   Algorithms,
                   Parameters,
                   h0,
                   PBar = None):
    L = len(h0)
    N_Algorithms = len(Algorithms)
    measure_init = lambda taps, N_iter: {'h': np.zeros((N_iter, taps)),
                                         'J': np.zeros(N_iter),
                                         'Jex': np.zeros(N_iter),
                                         'var': np.zeros((N_iter, taps))}
  
    measures = {Parameters[k].label: measure_init(L, N) for k in range(N_Algorithms)}
  
    for k in range(NR):
        signals = environment(N, environment_parameters)
        x = signals['x']
        d = signals['d']
  
        for c in range(N_Algorithms):
            label = Parameters[c].label
            algorithm_signals = Algorithms[c](N, x, d, h0, Parameters[c])
            measures[label]['h'] += algorithm_signals['h']
            measures[label]['J'] += algorithm_signals['e']**2
            measures[label]['Jex'] += (algorithm_signals['e']- signals['v'])**2
            if 'v' in algorithm_signals:
                measures[label]['var'] += algorithm_signals['v']
            else:
                measures[label]['var'] += np.zeros((N, L))
  
        if not PBar is None:
            PBar.update(1)
        else:
            print(f'Realization {k} out of {NR}')
    
    for k in range(N_Algorithms):
        label = Parameters[k].label
        measures[label]['h'] /= NR
        measures[label]['J'] /= NR
        measures[label]['Jex'] /= NR
        measures[label]['var'] /= NR
  
    return measures