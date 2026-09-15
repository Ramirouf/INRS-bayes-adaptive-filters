import os
import dask
import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from numba import njit, literal_unroll
from numba_progress import ProgressBar
from dask.diagnostics import ProgressBar as dask_PB
from collections import namedtuple
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

from numpy.typing import NDArray

# Desativa especificamente o warning de funções como objetos de primeira classe
warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

from filters import (
    # parameter dtypes
    std_env_parameters, laplace_env_parameters,
    GMVC_parameters, NLMS_parameters,
    # signal / environment helpers
    autocorr_matrix_calc, autocorr_matrix_estimate, AR_settling_time, 
    std_gaussian_behavior, laplace_noise_behavior,
    # algorithms
    NLMS_algorithm, GMVC_algorithm, sKF_algorithm, sKF_L_algorithm, sKF_L_exact_algorithm,
    sKF_integral_algorithm, sKF_L_integral_algorithm,
    # monte carlo driver
    MC_Simulations, _compute_MSD
)

def single_run(N, 
               environment_parameters,
               environment, Algorithms,
               Parameters,
               h0):
    ho, signals = environment(N, environment_parameters)
    x = signals['x']
    d = signals['d']
    L = len(h0)

    N_Algorithms = len(Algorithms)
    measures = {Parameters[k].label: measure_init(L, N) for k in range(N_Algorithms)}
    for c in range(N_Algorithms):
        label = Parameters[c].label
        algorithm_signals = Algorithms[c](N, x, d, h0, Parameters[c])

        measures[label]['h'] += algorithm_signals.h
        measures[label]['J'] += algorithm_signals.e**2
        measures[label]['Jex'] += (algorithm_signals.e - signals['v'])**2
        measures[label]['MSD'] += _compute_MSD(algorithm_signals.h, ho)
        measures[label]['var'] += algorithm_signals.v
    return measures

measure_init = lambda taps, N_iter: {'h': np.zeros((N_iter, taps)),
                                     'J': np.zeros(N_iter),
                                     'Jex': np.zeros(N_iter),
                                     'MSD': np.zeros(N_iter),
                                     'var': np.zeros((N_iter, taps))}

def sum_measures(measures_a, measures_b):
    summed_measures = {}
    for label in measures_a:
        summed_measures[label] = {}
        for key in measures_a[label]:
            summed_measures[label][key] = measures_a[label][key] + measures_b[label][key]
    return summed_measures

delayed_iteration = dask.delayed(single_run)
delayed_chunked_iterations = dask.delayed(MC_Simulations)
delayed_sum = dask.delayed(sum_measures)

def create_tasks_tree(tasks):
    while len(tasks) > 1:
        next_stage = []
        for i in range(0, len(tasks), 2):
            if i + 1 < len(tasks):
                combined = delayed_sum(tasks[i], tasks[i+1])
                next_stage.append(combined)
            else:
                next_stage.append(tasks[i])
        tasks = next_stage
    return tasks[0]

def dask_MC_Simulations(N, 
                        NR,
                        environment_parameters,
                        environment,
                        Algorithms,
                        Parameters,
                        h0,
                        num_workers = 1,
                        num_chunks = None):
    L = len(h0)
    N_Algorithms = len(Algorithms)
    if num_chunks is None:
        num_chunks = num_workers

    rest_of_realizations = NR % num_chunks
    tasks = [delayed_chunked_iterations(
        N, NR//num_chunks, environment_parameters, environment, Algorithms, Parameters, h0, "", external_avg = True
    ) for _ in range(num_chunks)]
    
    if rest_of_realizations > 0:
        tasks.append(delayed_chunked_iterations(
            N, rest_of_realizations, environment_parameters, environment, Algorithms, Parameters, h0, "", external_avg = True
        ))
    
    tasks_tree = create_tasks_tree(tasks)
    tasks_tree.visualize()

    with dask_PB():
        measures = dask.compute(tasks_tree, num_workers=num_workers)[0]

    for k in range(N_Algorithms):
        label = Parameters[k].label
        measures[label]['h'] /= NR
        measures[label]['J'] /= NR
        measures[label]['Jex'] /= NR
        measures[label]['var'] /= NR
        measures[label]['MSD'] /= NR

    return measures