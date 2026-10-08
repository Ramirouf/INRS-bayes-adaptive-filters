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


def plot_metrics(
    metrics_dict: dict[str, dict[str, NDArray]],
    desired_metrics: list[str],
    subplot_shape: tuple[int, int] = (-1, 1),
    title: str = ""
    ) -> plt.figure:
    n_rows, n_cols = subplot_shape
    if n_rows == -1:
        n_rows = int(np.ceil(len(desired_metrics) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, metric in enumerate(desired_metrics):
        ax = axes[i]
        for alg, alg_metrics in metrics_dict.items():
            if metric in alg_metrics:
                ax.plot(alg_metrics[metric], label=alg)
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        ax.legend()

    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    return fig
