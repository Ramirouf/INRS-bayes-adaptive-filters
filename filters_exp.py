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
