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

@njit(cache=True, nogil=True)
def impulsive_gaussian(shape: tuple[int], p:np.float64, var_1:np.float64=1.0, var_2:np.float64=1.0):
  """Sample an array from an impulsive Gaussian distribution.

  The distribution is a mixture of two Gaussians:
    with probability p: N(0, var_1)
    with probability 1-p: N(0, var_2)

  Parameters
  ----------
  shape : tuple of int
    Shape of the output array.
  p : float
    Probability of sampling from the first Gaussian.
  var_1 : float
    Variance of the first Gaussian.
  var_2 : float
    Variance of the second Gaussian.

  Returns
  -------
  NDArray
    Array of samples with the requested shape.
  """
  out = np.empty(shape, dtype=np.float64)
  flat = out.ravel()
  for i in range(flat.size):
    if np.random.random() < p:
      flat[i] = np.random.normal(0, np.sqrt(var_1))
    else:
      flat[i] = np.random.normal(0, np.sqrt(var_2))
  return out