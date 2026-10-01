"""
Monte Carlo simulations for adaptive filters using Dask for parallel execution.
"""

import numba
import numpy as np
import dask
import copy
import warnings
from numba_progress import ProgressBar
from dask.diagnostics import ProgressBar as dask_PB

from numpy.typing import NDArray

def _PBbar_update(PBar: ProgressBar | str | None, k: int, NR: int) -> None:
    """
    Update the progress bar for Monte Carlo simulations.

    Parameters:
    PBar (ProgressBar | str | None): Progress bar object or "verbose" for console output.
    k (int): Current realization index.
    NR (int): Total number of realizations.
    """
    if PBar is None:
        return
    if PBar == "verbose":
        print(f'Realization {k} out of {NR}')
        return
    if hasattr(PBar, 'update'):
        PBar.update(1)
        return
    warnings.warn(f'Unrecognized PBar type: {type(PBar)}')

def _measure_init(taps: int, N_iter: int, desired_metrics: dict[str, callable]) -> dict[str, NDArray[np.float64]]:
    """
    Initialize the measures dictionary for storing simulation results.

    Parameters:
    taps (int): Number of filter taps.
    N_iter (int): Number of iterations per simulation.
    desired_metrics (dict[str, callable]): Dictionary of metric functions to evaluate the algorithms.

    Returns:
    dict[str, NDArray[np.float64]]: Dictionary containing initialized measures.
    """
    measures = {'h': np.zeros((N_iter, taps)),
                'var': np.zeros((N_iter, taps))}
    for key in desired_metrics:
        measures[key] = np.zeros(N_iter)
    return measures

def _average_measures(
    measures: dict[str, dict[str, NDArray[np.float64]]], 
    Parameters: tuple[object], 
    desired_metrics: dict[str, callable], 
    NR: int
    ) -> dict[str, dict[str, NDArray[np.float64]]]:
    """
    Average the measures over multiple realizations.

    Parameters
    ----------
    measures : dict[str, dict[str, NDArray[np.float64]]]
        Dictionary containing the measures for each algorithm.
    Parameters : tuple[object]
        Tuple of algorithm parameter objects.
    desired_metrics : dict[str, callable]
        Dictionary of metric functions to evaluate the algorithms.
    NR : int
        Number of realizations.

    Returns
    -------
    dict[str, dict[str, NDArray[np.float64]]]
        Dictionary containing the averaged measures for each algorithm.
    """
    avg_measures = copy.deepcopy(measures)
    for par in Parameters:
        label = par.label
        avg_measures[label]['h'] /= NR
        avg_measures[label]['var'] /= NR
        for key in desired_metrics:
            avg_measures[label][key] /= NR
    return avg_measures

def MC_Simulations(
    N: int, 
    NR: int,
    environment_parameters: object,
    environment: callable,
    algorithms: tuple[callable],
    algorithms_parameters: tuple[object],
    desired_metrics: dict[str, callable],
    h0: NDArray[np.float64],
    PBar: ProgressBar = None,
    average: bool = True,
    seed: int = None) -> dict[str, dict[str, NDArray[np.float64]]]:
    """
    Monte Carlo Simulations for adaptive filters.

    Parameters
    ----------
    N : int
        Number of iterations per simulation.
    NR : int
        Number of realizations.
    environment_parameters : object
        Parameters for the environment.
    environment : callable
        Function that generates the environment signals.
    algorithms : tuple[callable]
        Tuple of algorithm functions to be tested.
    algorithms_parameters : tuple[object]
        Tuple of parameters for each algorithm.
    desired_metrics : dict[str, callable]
        Dictionary of metric functions to evaluate the algorithms.
    h0 : NDArray[np.float64]
        Initial filter coefficients.
    PBar : ProgressBar, optional
        Progress bar object for tracking simulation progress.
    average : bool, optional
        Whether to average the measures over realizations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, NDArray[np.float64]]]
        Dictionary containing the averaged measures for each algorithm over the Monte Carlo realizations.
    """
    L = len(h0)
    np.random.seed(seed)
    N_Algorithms = len(algorithms)
    
    measures = {algorithms_parameters[k].label: _measure_init(L, N, desired_metrics=desired_metrics) for k in range(N_Algorithms)}
    
    for k in range(NR):
        ho, signals = environment(N, environment_parameters)
        x = signals['x']
        d = signals['d']

        for c in range(N_Algorithms):
            label = algorithms_parameters[c].label
            algorithm_signals = algorithms[c](N, x, d, h0, algorithms_parameters[c])
            
            measures[label]['h'] += algorithm_signals.h
            measures[label]['var'] += algorithm_signals.v
            for key in desired_metrics:
                measures[label][key] += desired_metrics[key](algorithm_signals, signals['v'], ho)

        _PBbar_update(PBar, k, NR)
    
    if average:
        return _average_measures(measures, algorithms_parameters, desired_metrics, NR)
    
    return measures

def _create_task_list(
    NR: int, 
    num_chunks: int,
    seed_sequence: list[np.random.SeedSequence],
    general_parameters: dict[str, object]
    ) -> list[object]:
    """
    Create a list of Dask tasks for Monte Carlo simulations.

    Parameters
    ----------
    NR : int
        Total number of realizations.
    num_chunks : int
        Number of chunks to divide the simulations into.
    seed_sequence : list[np.random.SeedSequence]
        List of seed sequences for reproducibility.
    general_parameters : dict[str, object]
        General parameters to pass to the simulation function.

    Returns
    -------
    list[object]
        List of Dask delayed tasks.
    """
    tasks = [
        _delayed_chunked_iterations(
            NR = NR//num_chunks, seed = seed_sequence[k].generate_state(1), **general_parameters
        ) for k in range(num_chunks)
    ]
    
    rest_of_realizations = NR % num_chunks
    if rest_of_realizations > 0:
        tasks.append(
            _delayed_chunked_iterations(
                    NR = rest_of_realizations, seed = seed_sequence[-1].generate_state(1), **general_parameters
                )
            )
    
    return tasks

def _sum_measures(
    measures_a: dict[str, dict[str, NDArray[np.float64]]],
    measures_b: dict[str, dict[str, NDArray[np.float64]]]
    ) -> dict[str, dict[str, NDArray[np.float64]]]:
    """
    Sum two sets of measures from Monte Carlo simulations.

    Parameters
    ----------
    measures_a : dict[str, dict[str, NDArray[np.float64]]]
        First set of measures.
    measures_b : dict[str, dict[str, NDArray[np.float64]]]
        Second set of measures.

    Returns
    -------
    dict[str, dict[str, NDArray[np.float64]]]
        Summed measures.
    """
    summed_measures = {}
    for label in measures_a:
        summed_measures[label] = {}
        for key in measures_a[label]:
            summed_measures[label][key] = measures_a[label][key] + measures_b[label][key]
    return summed_measures

_delayed_chunked_iterations = dask.delayed(MC_Simulations)
_delayed_sum = dask.delayed(_sum_measures)

def _create_tasks_tree(tasks: list[object]) -> object:
    """
    Create a tree of Dask tasks to sum measures.

    Parameters
    ----------
    tasks : list[object]
        List of Dask delayed tasks representing individual simulation chunks.

    Returns
    -------
    object
        Dask delayed task representing the combined result of all tasks.
    """
    while len(tasks) > 1:
        next_stage = []
        for i in range(0, len(tasks), 2):
            if i + 1 < len(tasks):
                combined = _delayed_sum(tasks[i], tasks[i+1])
                next_stage.append(combined)
            else:
                next_stage.append(tasks[i])
        tasks = next_stage
    return tasks[0]

def _check_scheduler_value(scheduler: str) -> None:
    """
    Check if the provided scheduler value is valid.

    Parameters
    ----------
    scheduler : str
        Scheduler type, either 'threads' or 'processes'.

    Raises
    ------
    ValueError
        If the scheduler is not 'threads' or 'processes'.
    """
    if scheduler not in ["threads", "processes"]:
        raise ValueError("Scheduler must be either 'threads' or 'processes'")
      
def _check_numba_algorithms(Algorithms: list[callable], scheduler: str) -> None:
    """
    Check if the provided algorithms are compatible with the specified scheduler.

    Parameters
    ----------
    Algorithms : list[callable]
        List of algorithms to be checked.
    scheduler : str
        Scheduler type, either 'threads' or 'processes'.

    Raises
    ------
    Warning
        If an algorithm is not numba-jitted and the scheduler is 'threads'.
    """
    if scheduler == "threads":
        for alg in Algorithms:
            if not isinstance(alg, numba.core.dispatcher.Dispatcher):
                import warnings
                warnings.warn(f"Algorithm {alg.__name__} is not numba-jitted. Using 'threads' scheduler may not be efficient.")
              
def _get_seed_sequence(seed: None | int, num_chunks: int) -> list[np.random.SeedSequence]:
    """
    Generate a list of SeedSequence objects for parallel random number generation.

    Parameters
    ----------
    seed : None | int
        Base random seed. If None, a new seed is generated.
    num_chunks : int
        Number of SeedSequence objects to generate.

    Returns
    -------
    list[np.random.SeedSequence]
        List of SeedSequence objects for each simulation chunk.
        
    Raises
    ------
    ValueError
        If seed is not None or a non-negative integer.
    """
    if seed is None:
        seed = np.random.SeedSequence().entropy
        return np.random.SeedSequence(seed).spawn(num_chunks)
    if isinstance(seed, int) and seed >= 0:
        return np.random.SeedSequence(seed).spawn(num_chunks)
    raise ValueError("Seed must be None or a non-negative integer")

def _check_num_chunks(num_chunks: int | None, num_workers: int) -> int:
    """
    Check and adjust the number of chunks for parallel execution.

    Parameters
    ----------
    num_chunks : int | None
        Number of chunks to divide the simulations into.
    num_workers : int
        Number of parallel workers.

    Returns
    -------
    int
        Adjusted number of chunks.

    Raises
    ------
    ValueError
        If num_chunks is not a positive integer.
    """
    if num_chunks <= 0 or not isinstance(num_chunks, int):
        raise ValueError("num_chunks must be a positive integer")
    if num_chunks < num_workers:
        warnings.warn("num_chunks is less than num_workers. Setting num_chunks to num_workers.")
        return num_workers
    if num_chunks is None:
        return num_workers
    return num_chunks

def dask_MC_Simulations(
    N: int,
    NR: int,
    environment_parameters: dict[str, object],
    environment: callable,
    algorithms: list[callable],
    algorithms_parameters: list[object],
    desired_metrics: dict[str, callable],
    h0: np.ndarray,
    num_workers: int = 1,
    num_chunks: int = None,
    seed: int = None,
    scheduler: str = "threads"
    ) -> tuple[dict[str, dict[str, NDArray[np.float64]]], object]:
    """
    Perform Monte Carlo simulations using Dask for parallel execution.

    Parameters
    ----------
    N : int
        Number of iterations per simulation.
    NR : int
        Total number of realizations.
    environment_parameters : dict[str, object]
        Parameters for the environment.
    environment : callable
        Function representing the environment.
    algorithms : list[callable]
        List of adaptive filtering algorithms to be tested.
    algorithms_parameters : list[object]
        List of parameters for each algorithm.
    desired_metrics : dict[str, callable]
        Dictionary of desired metrics to evaluate the algorithms.
    h0 : np.ndarray
        Initial filter coefficients.
    num_workers : int, optional
        Number of Dask workers (default is 1).
    num_chunks : int, optional
        Number of chunks to divide the simulations into (default is None).
    seed : int, optional
        Random seed for reproducibility (default is None).
    scheduler : str, optional
        Dask scheduler to use ("threads" or "processes", default is "threads").

    Returns
    -------
    tuple[dict[str, dict[str, NDArray[np.float64]]], object]
        A tuple containing the averaged measures and the Dask task tree.
    """
    seed_sequence = _get_seed_sequence(seed, num_chunks)
    num_chunks = _check_num_chunks(num_chunks, num_workers)
    _check_scheduler_value(scheduler)
    _check_numba_algorithms(algorithms, scheduler)

    general_parameters = {
        "N": N,
        "environment_parameters": environment_parameters,
        "environment": environment,
        "algorithms": algorithms,
        "algorithms_parameters": algorithms_parameters,
        "desired_metrics": desired_metrics,
        "h0": h0,
        "average": False
    }
    tasks = _create_task_list(NR, num_chunks, seed_sequence, general_parameters)
    tasks_tree = _create_tasks_tree(tasks)

    with dask_PB():
        extra = {"chunksize": 1} if scheduler == "processes" else {}
        measures = dask.compute(tasks_tree, num_workers=num_workers, scheduler=scheduler, **extra)[0]

    avg_measures = _average_measures(measures, algorithms_parameters, desired_metrics, NR)

    return avg_measures, tasks_tree