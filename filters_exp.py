import numpy as np
import dask
import copy

from collections.abc import Callable
import filters
from matplotlib import pyplot as plt
from numba import njit
from numba_progress import ProgressBar
from collections import namedtuple
from dask.diagnostics import ProgressBar as dask_PB

from numpy.typing import NDArray

@njit(cache=True, nogil=True)
def generalized_gaussian(shape: tuple[int], beta:np.float64, scale:np.float64=1.0):
  """Sample an array from a zero-mean generalized Gaussian distribution.

  pdf(x) = beta / (2*scale*Gamma(1/beta)) * exp(-(|x|/scale)**beta)

  Samples are drawn via |X| = scale * Gamma(1/beta, 1)**(1/beta) with a
  random sign, which recovers a Gaussian for beta=2 and a Laplace for beta=1.

  Parameters
  ----------
  shape : tuple of int
    Shape of the output array.
  beta : float
    Shape parameter of the distribution.
  scale : float
    Scale parameter of the distribution (alpha).

  Returns
  -------
  NDArray
    Array of samples with the requested shape.
  """
  out = np.empty(shape, dtype=np.float64)
  flat = out.ravel()
  for i in range(flat.size):
    g = np.random.gamma(1.0 / beta, 1.0)
    mag = scale * g ** (1.0 / beta)
    flat[i] = mag if np.random.random() < 0.5 else -mag
  return out

''' 
#Current code for base to plot metrics

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(10, 12))
plt.subplot(3,1,1)
k=0
for params in Alg_Parameters:
    label = params.label
    plt.plot(10*np.log10(tvs_laplacian_MC_measures[label]['EMSE']), label=label, color=colors[k])
    k += 1
plt.ylabel("EMSE (dB)")
plt.xlabel("Iterations")
plt.title("Laplacian Noise - Time Varying System")
plt.legend()

plt.subplot(3,1,2)
k=0
for params in Alg_Parameters:
    label = params.label
    plt.semilogy(tvs_laplacian_MC_measures[label]['var'], label=label, color=colors[k])
    k += 1
plt.ylabel("v_tilde")
plt.xlabel("Iterations")
plt.title("Laplacian Noise - Time Varying System")

plt.subplot(3,1,3)
k=0
for params in Alg_Parameters:
    label = params.label
    plt.plot(10*np.log10(tvs_laplacian_MC_measures[label]['MSD']), label=label, color=colors[k])
    k += 1
plt.ylabel("MSD (dB)")
plt.xlabel("Iterations")
plt.title("Laplacian Noise - Time Varying System")
plt.legend()

plt.tight_layout()
plt.show()
'''

@njit(cache=True, nogil=True)
def dB_scale(array: NDArray) -> NDArray:
    return 10 * np.log10(array)

@njit(cache=True, nogil=True)
def linear_scale(array: NDArray) -> NDArray:
    return array
  
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
