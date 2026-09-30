"""
Metrics module for evaluating adaptive filters.

All metrics are defined as functions that take the algorithm output, noise signal, and true system output as inputs and return a numerical evaluation.
Some metrics may include additional optional arguments for specific evaluation criteria using lambda functions.
"""

import numpy as np
import filters

from numba import njit

from numpy.typing import NDArray

@njit(cache=True, nogil=True)
def squared_error(
    algorithm_output: filters.filter_output, 
    noise_signal: NDArray[np.float64], 
    true_system: NDArray[np.float64]
    ):
    """
    Compute the squared error between the algorithm output and the true system output.

    Parameters:
    algorithm_output (array-like): Output of the adaptive filter algorithm.
    noise_signal (array-like): Noise signal added to the system output.
    true_system (array-like): True system without noise.

    Returns:
    float: Mean squared error.
    """
    
    squared_error = (algorithm_output.e)**2
    
    return squared_error

@njit(cache=True, nogil=True)
def power_error(
    algorithm_output: filters.filter_output, 
    noise_signal: NDArray[np.float64], 
    true_system: NDArray[np.float64],
    p: np.float64 = 2):
    """
    Compute the power error between the algorithm output and the true system output.

    Parameters:
    algorithm_output (array-like): Output of the adaptive filter algorithm.
    noise_signal (array-like): Noise signal added to the system output.
    true_system (array-like): True system without noise.
    p (float, optional): Power to raise the error to. Default is 2.

    Returns:
    float: Mean power error.
    """
    
    power_error = algorithm_output.e**p
    
    return power_error
    

@njit(cache=True, nogil=True)
def excess_squared_error(
    algorithm_output: filters.filter_output, 
    noise_signal: NDArray[np.float64], 
    true_system: NDArray[np.float64]
    ):
    """
    Compute the excess squared error between the algorithm output and the true system output.

    Parameters:
    algorithm_output (array-like): Output of the adaptive filter algorithm.
    noise_signal (array-like): Noise signal added to the system output.
    true_system (array-like): True system without noise.

    Returns:
    float: Mean excess squared error.
    """
    
    excess_squared_error = (algorithm_output.e - noise_signal)**2
    
    return excess_squared_error

@njit(cache=True, nogil=True)
def excess_power_error(
    algorithm_output: filters.filter_output, 
    noise_signal: NDArray[np.float64], 
    true_system: NDArray[np.float64],
    p: np.float64 = 2):
    """
    Compute the excess power error between the algorithm output and the true system output.

    Parameters:
    algorithm_output (array-like): Output of the adaptive filter algorithm.
    noise_signal (array-like): Noise signal added to the system output.
    true_system (array-like): True system without noise.
    p (float, optional): Power to raise the error to. Default is 2.

    Returns:
    float: Mean excess power error.
    """
    
    excess_power_error = (algorithm_output.e - noise_signal)**p
    
    return excess_power_error

@njit(cache=True, nogil=True)
def squared_deviation(
    algorithm_output: filters.filter_output, 
    noise_signal: NDArray[np.float64], 
    true_system: NDArray[np.float64],
    ):
    """
    Compute the squared deviation between the algorithm system estimative and the true system plant.

    Parameters:
    algorithm_output (array-like): Output of the adaptive filter algorithm.
    noise_signal (array-like): Noise signal added to the system output.
    true_system (array-like): True system without noise.

    Returns:
    float: Mean squared deviation.
    """
    
    if true_system.ndim == 1:
        normalization_factor = np.dot(true_system, true_system)
    else:
        normalization_factor = np.array([np.dot(true_system[k,:], true_system[k,:]) for k in range(true_system.shape[0])])
    h_error = algorithm_output.h - true_system
        
    squared_deviation = np.zeros(h_error.shape[0])
    for k in range(h_error.shape[0]):
        squared_deviation[k] = np.linalg.norm(h_error[k,:])**2
        if true_system.ndim == 1:
            squared_deviation[k] /= normalization_factor
        else:
            squared_deviation[k] /= normalization_factor[k]
    
    return squared_deviation

@njit(cache=True, nogil=True)
def power_deviation(
    algorithm_output: filters.filter_output, 
    noise_signal: NDArray[np.float64], 
    true_system: NDArray[np.float64],
    p: np.float64 = 2):
    """
    Compute the power deviation between the algorithm system estimative and the true system plant.

    Parameters:
    algorithm_output (array-like): Output of the adaptive filter algorithm.
    noise_signal (array-like): Noise signal added to the system output.
    true_system (array-like): True system without noise.
    p (float, optional): Power to raise the system deviation to. Default is 2.

    Returns:
    float: Mean power deviation.
    """
    
    if true_system.ndim == 1:
        normalization_factor = np.sum(np.abs(true_system)**p)
    else:
        normalization_factor = np.array([np.sum(np.abs(true_system[k,:])**p) for k in range(true_system.shape[0])])
    h_error = algorithm_output.h - true_system
        
    power_deviation = np.zeros(h_error.shape[0])
    for k in range(h_error.shape[0]):
        power_deviation[k] = np.sum(np.abs(h_error[k,:])**p)
        if true_system.ndim == 1:
            power_deviation[k] /= normalization_factor
        else:
            power_deviation[k] /= normalization_factor[k]
    
    return power_deviation