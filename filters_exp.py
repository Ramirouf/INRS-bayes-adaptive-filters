import numpy as np
import dask
import copy
import filters
from matplotlib import pyplot as plt
from numba import njit
from numba_progress import ProgressBar
from collections import namedtuple
from dask.diagnostics import ProgressBar as dask_PB

from numpy.typing import NDArray

def measure_init(taps, N_iter, system_metrics: dict, error_metrics: dict):
    measures = {'h': np.zeros((N_iter, taps)),
                'var': np.zeros((N_iter, taps))}
    for key in system_metrics:
        measures[key] = np.zeros(N_iter)
    for key in error_metrics:
        measures[key] = np.zeros(N_iter)
    return measures

def _average_measures(measures, Parameters, system_metrics, error_metrics, NR):
    avg_measures = copy.deepcopy(measures)
    for par in Parameters:
        label = par.label
        avg_measures[label]['h'] /= NR
        avg_measures[label]['var'] /= NR
        for key in system_metrics:
            avg_measures[label][key] /= NR
        for key in error_metrics:
            avg_measures[label][key] /= NR
    return avg_measures

def MC_Simulations(N, 
                   NR,
                   environment_parameters,
                   environment,
                   algorithms,
                   algorithms_parameters,
                   system_metrics,
                   error_metrics,
                   h0,
                   PBar = None,
                   average = True,
                   seed = None):
    L = len(h0)
    np.random.seed(seed)
    N_Algorithms = len(algorithms)
  
    measures = {algorithms_parameters[k].label: measure_init(L, N, system_metrics, error_metrics) for k in range(N_Algorithms)}
  
    for k in range(NR):
        ho, signals = environment(N, environment_parameters)
        x = signals['x']
        d = signals['d']
  
        for c in range(N_Algorithms):
            label = algorithms_parameters[c].label
            algorithm_signals = algorithms[c](N, x, d, h0, algorithms_parameters[c])
            
            measures[label]['h'] += algorithm_signals.h
            measures[label]['var'] += algorithm_signals.v
            for key in system_metrics:
                measures[label][key] += system_metrics[key](algorithm_signals.h, ho)
            for key in error_metrics:
                measures[label][key] += error_metrics[key](algorithm_signals.e, signals['v'])

        filters._PBbar_update(PBar, k, NR)
    
    if average:
        return _average_measures(measures, algorithms_parameters, system_metrics, error_metrics, NR)
    
    return measures

delayed_chunked_iterations = dask.delayed(MC_Simulations)

def _create_task_list(NR, num_chunks, seed_sequence, general_parameters):
    tasks = [
        delayed_chunked_iterations(
            NR = NR//num_chunks, seed = seed_sequence[k].generate_state(1), **general_parameters
        ) for k in range(num_chunks)
    ]
    
    rest_of_realizations = NR % num_chunks
    if rest_of_realizations > 0:
        tasks.append(
            filters.delayed_chunked_iterations(
                    NR = rest_of_realizations, seed = seed_sequence[-1].generate_state(1), **general_parameters
                )
            )
    
    return tasks

def dask_MC_Simulations(N, 
                        NR,
                        environment_parameters,
                        environment,
                        algorithms,
                        algorithms_parameters,
                        system_metrics,
                        error_metrics,
                        h0,
                        num_workers = 1,
                        num_chunks = None,
                        seed = None,
                        scheduler = "threads"):
    seed_sequence = filters._get_seed_sequence(seed, num_chunks)
    num_chunks = filters._check_num_chunks(num_chunks, num_workers)
    filters._check_scheduler_value(scheduler)
    filters._check_numba_algorithms(algorithms, scheduler)

    general_parameters = {
        "N": N,
        "environment_parameters": environment_parameters,
        "environment": environment,
        "algorithms": algorithms,
        "algorithms_parameters": algorithms_parameters,
        "system_metrics": system_metrics,
        "error_metrics": error_metrics,
        "h0": h0,
        "average": False
    }
    tasks = _create_task_list(NR, num_chunks, seed_sequence, general_parameters)
    tasks_tree = filters.create_tasks_tree(tasks)

    with dask_PB():
        extra = {"chunksize": 1} if scheduler == "processes" else {}
        measures = dask.compute(tasks_tree, num_workers=num_workers, scheduler=scheduler, **extra)[0]

    avg_measures = _average_measures(measures, algorithms_parameters, system_metrics, error_metrics, NR)

    return avg_measures, tasks_tree