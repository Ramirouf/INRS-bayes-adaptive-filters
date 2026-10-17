# Visualization tools for plotting metrics with different scales (dB, linear, variance)

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from numpy.typing import NDArray
from collections.abc import Callable

@njit(cache=True, nogil=True)
def dB_scale(array: NDArray) -> NDArray:
    return 10 * np.log10(array)

@njit(cache=True, nogil=True)
def linear_scale(array: NDArray) -> NDArray:
    return array

@njit(cache=True, nogil=True)
def var_scale(array: NDArray) -> NDArray:
    return 10 * np.log10(array[:, 0])

def plot_metrics(
    metrics_dict: dict[str, dict[str, NDArray]],
    desired_metrics: list[str],
    metrics_scales: dict[str, Callable[[NDArray], NDArray]] = {},
    subplot_shape: tuple[int, int] = (-1, 1),
    axis_size: tuple[int, int] = (5, 4),
    title: str = "",
    ) -> plt.figure:
    '''
    Plot multiple metrics for different algorithms with customizable scales.

    Parameters
    ----------
    metrics_dict : dict[str, dict[str, NDArray]]
        Dictionary containing metrics for each algorithm.
    desired_metrics : list[str]
        List of metrics to plot.
    metrics_scales : dict[str, Callable[[NDArray], NDArray]], optional
        Dictionary mapping metric names to scale functions (default is linear scale).
    subplot_shape : tuple[int, int], optional
        Shape of the subplot grid (default is (-1, 1), which automatically determines rows).
    axis_size : tuple[int, int], optional
        Size of each subplot axis (default is (5, 4)).
    title : str, optional
        Title for the entire figure (default is an empty string).

    Returns
    -------
    plt.figure
        The created matplotlib figure object.
    '''
    
    # Get the number of rows and columns for the subplots
    n_rows, n_cols = subplot_shape
    if n_rows == -1:
        n_rows = int(np.ceil(len(desired_metrics) / n_cols))

    # Create the figure object and subplots axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(axis_size[0]*n_cols, axis_size[1]*n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    # Plot each desired metric for all algorithms
    for i, metric in enumerate(desired_metrics):
        scale_func = metrics_scales.get(metric, linear_scale)
        ax = axes[i]
        for alg, alg_metrics in metrics_dict.items():
            if metric in alg_metrics:
                ax.plot(scale_func(alg_metrics[metric]), label=alg)
            else:
                raise ValueError(f"Metric '{metric}' not found for algorithm '{alg}'")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        ax.legend()

    # Adjust layout and set the title if provided
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    return fig