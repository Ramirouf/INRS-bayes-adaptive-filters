import numpy as np
import dask
import copy
import itertools

from collections.abc import Callable
import filters
import monte_carlo as mc
from matplotlib import pyplot as plt
from numba import njit
from numba_progress import ProgressBar
from collections import namedtuple
from dask.diagnostics import ProgressBar as dask_PB
from typing import Any
from pathlib import Path
import os
import yaml
import metrics
import gc
import time

from numpy.typing import NDArray
from filters import generalized_gaussian, impulsive_gaussian

# Convergence metrics

def db_area_of_convergence(db_metric: np.typing.NDArray[np.float64]) -> np.float64:
  '''
  Compute the area of convergence of the given dB metric.

  Parameters
  ----------
  db_metric : NDArray[np.float64]
    Array of metric values in dB.

  Returns
  -------
  np.float64
    Area of convergence.
  '''
  
  floor = np.min(db_metric)
  
  area_of_convergence = np.sum(db_metric - floor)
  return area_of_convergence

def db_accumulated_target_error(
    db_metric: np.typing.NDArray[np.float64],
    target: float | np.floating,
    tolerance: float | np.floating = 0.2,
    **kwargs: Any,
) -> float:
  """Compute the accumulated target error of the given dB metric.

  Parameters
  ----------
  db_metric : NDArray[np.float64]
    Array of metric values in dB.
  target : float
    Target value in dB (e.g. minimum value of the slow filter).
  tolerance : float, optional
    Tolerance around the target in dB (default: 0.2 dB).
  **kwargs : Any
    Additional keyword arguments (ignored, for interface compatibility).

  Returns
  -------
  float
    Accumulated target error.
  """
  error = np.maximum(np.abs(db_metric - target) - tolerance, 0.0)
  return float(np.sum(error))

def time_to_convergence(db_metric: np.typing.NDArray[np.float64],
                        cutoff: np.float64 = -3.0,
                        dt: np.float64 = 1.0) -> int:
  ''' 
  Compute the time to convergence of the given dB metric.

  Parameters
  ----------
  db_metric : NDArray[np.float64]
    Array of metric values in dB.
  floor_estimate_samples : int
    Number of samples from the end of the array to estimate the floor.
  cutoff : float
    Threshold below the floor to consider as convergence.
  dt : float
    Time step between samples.

  Returns
  -------
  int
    Time to convergence in units of dt.
  '''
  target = np.min(db_metric)
  previous = db_metric[0]
  for idx, measure in enumerate(db_metric):
    if measure < (target - cutoff) and previous > (target - cutoff):
      return idx*dt
    previous = measure

  raise Exception("Couldn't find a convergence time/iteration")

def mean_absolute_steady_state_derivative(db_metric: np.typing.NDArray[np.float64],
                                          floor_estimate_samples: int = 100) -> np.float64:
  '''
  Compute the mean absolute derivative of the steady-state portion of the metric.

  Parameters
  ----------
  db_metric : NDArray[np.float64]
    Array of metric values.
  floor_estimate_samples : int
    Number of samples from the end of the array to consider as steady-state.

  Returns
  -------
  np.float64
    Mean absolute derivative of the steady-state portion.
  '''
  steady_state = db_metric[-floor_estimate_samples:]
  derivative = np.diff(steady_state)
  return np.mean(np.abs(derivative))


def find_minimum_lowest_slope(
    db_metric: np.typing.NDArray[np.float64],
    n_slope_samples: int = 1000,
) -> tuple[float, int, dict[str, Any]]:
  """Find the minimum/floor of a dB metric array by locating the window with lowest mean absolute slope.

  Parameters
  ----------
  db_metric : NDArray[np.float64]
    Array of metric values in dB.
  n_slope_samples : int, optional
    Window size in samples (default: 1000).

  Returns
  -------
  target_val : float
    Mean metric value in the window with the lowest point-to-point mean absolute derivative.
  best_window_start : int
    Start index of the window with the lowest slope.
  diagnostics : dict[str, Any]
    Diagnostic information including 'window_slopes', 'best_window_center', and 'n_slope_samples'.
  """
  if len(db_metric) < n_slope_samples:
    raise ValueError(
        f"Input array length ({len(db_metric)}) must be >= n_slope_samples ({n_slope_samples})."
    )

  abs_diff = np.abs(np.diff(db_metric))
  kernel = np.ones(n_slope_samples - 1, dtype=np.float64) / float(n_slope_samples - 1)
  window_slopes = np.convolve(abs_diff, kernel, mode='valid')

  best_idx = int(np.nanargmin(window_slopes))
  target_val = float(np.mean(db_metric[best_idx : best_idx + n_slope_samples]))

  diagnostics = {
      'window_slopes': window_slopes,
      'best_window_start': best_idx,
      'best_window_end': best_idx + n_slope_samples,
      'best_window_center': best_idx + n_slope_samples // 2,
      'min_slope': float(window_slopes[best_idx]),
      'n_slope_samples': n_slope_samples,
  }
  return target_val, best_idx, diagnostics

# Perform a grid search on algorithms for a given target MSD, optimizing for convergence speed
def default_db_scale(array: NDArray[np.float64]) -> NDArray[np.float64]:
  """Convert an array of metric values to dB scale, avoiding log of zero."""
  return 10.0 * np.log10(np.maximum(array, 1e-30))


def _format_param_val(val: Any) -> str:
  """Format a parameter value concisely for labels and plot titles."""
  if isinstance(val, (int, np.integer)):
    return str(val)
  if isinstance(val, (float, np.floating)):
    return f"{val:.2e}" if (abs(val) < 1e-2 or abs(val) >= 1e3) else f"{val:.4g}"
  return str(val)


def _build_algorithm_parameters(
    algorithm: Callable,
    param_dict: dict[str, Any],
    algorithm_parameter_class: type | None = None,
) -> object:
  """Construct an algorithm parameter namedtuple instance for any algorithm.

  Parameters
  ----------
  algorithm : Callable
    The adaptive filter algorithm function.
  param_dict : dict
    Merged dictionary of parameters including 'label'.
  algorithm_parameter_class : type or None
    Explicit parameter class. If None, resolves from filters module or creates dynamically.
  """
  if algorithm_parameter_class is not None:
    target_cls = algorithm_parameter_class
  else:
    alg_name = getattr(algorithm, '__name__', str(algorithm))
    candidate_name = f"{alg_name}_parameters"
    if hasattr(filters, candidate_name):
      target_cls = getattr(filters, candidate_name)
    elif hasattr(algorithm, 'parameters_class'):
      target_cls = algorithm.parameters_class
    else:
      target_cls = None

  if target_cls is not None and hasattr(target_cls, '_fields'):
    kwargs = {f: param_dict[f] for f in target_cls._fields if f in param_dict}
    if 'label' in target_cls._fields and 'label' not in kwargs:
      kwargs['label'] = param_dict.get('label', 'algorithm')
    return target_cls(**kwargs)
  else:
    alg_name = getattr(algorithm, '__name__', 'algorithm')
    fields = list(param_dict.keys())
    DynamicParam = namedtuple(f"{alg_name}_parameters", fields)
    return DynamicParam(**param_dict)


def grid_search(
    N: int,
    NR: int,
    h0: NDArray[np.float64],
    environment: Callable,
    environment_parameters: object,
    algorithm: Callable,
    algorithm_convergence_parameters: object,
    algorithm_performance_parameters: object = None,
    performance_metric: Callable = None,
    convergence_metric: Callable = db_area_of_convergence,
    convergence_metric_parameters: dict[str, Any] | None = None,
    performance_metric_scale: Callable[[NDArray], NDArray] | None = None,
    algorithm_parameter_class: type | None = None,
    floor_estimate_samples: int = 100,
    num_workers: int = 1,
    num_chunks: int | None = None,
    seed: int | None = None,
    minimize_convergence_metric: bool = True,
    scheduler: str = "threads",
    baseline_algorithm: Callable | None = None,
    baseline_parameter_class: type | None = None,
    noise_param_name: str = 'var_eta',
    noise_param_val: float | None = None,
    algortihm_performance_parameters: object = None,
) -> dict[str, Any]:
  """Perform a generic grid search on adaptive filter hyperparameters.

  Separates hyperparameters into fixed steady-state performance parameters and
  a grid of convergence hyperparameters. Evaluates each candidate configuration
  across Monte Carlo realizations using Dask.

  Parameters
  ----------
  N : int
    Number of iterations per simulation.
  NR : int
    Number of realizations.
  h0 : NDArray[np.float64]
    Initial filter taps.
  environment : Callable
    Function that generates the environment signals.
  environment_parameters : object
    Parameters for the environment.
  algorithm : Callable
    Adaptive filter algorithm function.
  algorithm_convergence_parameters : object
    Namedtuple (or dict) whose fields are parameter names and values are 1D NDArrays.
  algorithm_performance_parameters : object, optional
    Namedtuple (or dict) containing fixed performance hyperparameters.
  performance_metric : Callable
    Metric function to evaluate algorithm output (e.g. metrics.squared_deviation).
  convergence_metric : Callable, optional
    Scalar metric summarizing convergence speed/quality. Defaults to db_area_of_convergence.
  convergence_metric_parameters : dict, optional
    Additional keyword arguments passed to convergence_metric.
  performance_metric_scale : Callable, optional
    Function to scale the performance metric before convergence evaluation (default: dB scale).
  algorithm_parameter_class : type, optional
    Specific parameter class/namedtuple. If None, auto-detected or dynamically created.
  floor_estimate_samples : int, optional
    Number of final samples used to estimate the steady-state performance floor.
  num_workers : int, optional
    Number of parallel Dask workers.
  num_chunks : int, optional
    Number of chunks to divide the realizations into.
  seed : int, optional
    Random seed for reproducibility.
  minimize_convergence_metric : bool, optional
    Whether lower convergence metric indicates better performance (default: True).
  scheduler : str, optional
    Dask scheduler ('threads' or 'processes').
  algortihm_performance_parameters : object, optional
    Alias for algorithm_performance_parameters.

  Returns
  -------
  dict[str, Any]
    Dictionary containing:
      - 'best_parameters': dict of optimal convergence parameter values.
      - 'best_metric_value': optimal convergence metric score.
      - 'best_steady_state': measured steady-state value of the optimal configuration.
      - 'optimal_index': multi-dimensional index of the optimal configuration.
      - 'convergence_grid': NDArray of convergence scores across the parameter grid.
      - 'steady_state_grid': NDArray of steady-state values across the parameter grid.
      - 'grid_axes': dict mapping parameter names to 1D arrays of tested values.
      - 'measures': raw averaged Monte Carlo simulation measures.
      - 'best_label': label of the best configuration.
      - 'labels_grid': NDArray of labels for all combinations.
  """
  if performance_metric is None:
    raise ValueError("performance_metric must be provided.")

  if algorithm_performance_parameters is None and algortihm_performance_parameters is not None:
    algorithm_performance_parameters = algortihm_performance_parameters

  if convergence_metric_parameters is None:
    convergence_metric_parameters = {}

  if performance_metric_scale is None:
    performance_metric_scale = default_db_scale

  # Extract convergence parameters dict
  if hasattr(algorithm_convergence_parameters, '_asdict'):
    conv_dict = {k: np.asarray(v) for k, v in algorithm_convergence_parameters._asdict().items()}
  elif isinstance(algorithm_convergence_parameters, dict):
    conv_dict = {k: np.asarray(v) for k, v in algorithm_convergence_parameters.items()}
  else:
    raise TypeError("algorithm_convergence_parameters must be a namedtuple or dict.")

  # Extract fixed performance parameters dict
  if algorithm_performance_parameters is not None:
    if hasattr(algorithm_performance_parameters, '_asdict'):
      perf_dict = dict(algorithm_performance_parameters._asdict())
    elif isinstance(algorithm_performance_parameters, dict):
      perf_dict = dict(algorithm_performance_parameters)
    else:
      perf_dict = {k: getattr(algorithm_performance_parameters, k) for k in dir(algorithm_performance_parameters) if not k.startswith('_')}
  else:
    perf_dict = {}

  alg_name = getattr(algorithm, '__name__', 'algorithm')
  param_names = list(conv_dict.keys())
  param_arrays = [conv_dict[k] for k in param_names]
  grid_shape = tuple(len(arr) for arr in param_arrays)

  # Generate all combinations
  all_combos = list(itertools.product(*param_arrays))
  param_objects = []
  labels_grid = np.empty(grid_shape, dtype=object)

  for combo_idx, combo in enumerate(all_combos):
    multi_idx = np.unravel_index(combo_idx, grid_shape)
    curr_conv = dict(zip(param_names, combo))
    merged = {**perf_dict, **curr_conv}

    # Generate unique label
    label_parts = [f"{k}_{_format_param_val(v)}" for k, v in curr_conv.items()]
    label = f"{alg_name}_{'_'.join(label_parts)}"
    merged['label'] = label
    labels_grid[multi_idx] = label

    param_obj = _build_algorithm_parameters(
        algorithm=algorithm,
        param_dict=merged,
        algorithm_parameter_class=algorithm_parameter_class,
    )
    param_objects.append(param_obj)

  # Run Monte Carlo simulations (including slow baseline filter if provided)
  desired_metrics = {'performance': performance_metric}
  algorithms_list = [algorithm] * len(param_objects)
  slow_label = None
  slow_curve = None
  slow_min = None

  if baseline_algorithm is not None and baseline_parameter_class is not None:
    eps_slow = float(perf_dict.get('epsilon_2', 5e-9))
    noise_val = noise_param_val if noise_param_val is not None else float(perf_dict.get(noise_param_name, 1e-3))
    v_tilde_0 = float(perf_dict.get('v_tilde_0', 1.0))
    slow_label = f"{alg_name}_slow_baseline"
    slow_kwargs = {
        'label': slow_label,
        'epsilon': eps_slow,
        noise_param_name: noise_val,
        'v_tilde_0': v_tilde_0,
    }
    slow_param_obj = baseline_parameter_class(**slow_kwargs)
    param_objects.append(slow_param_obj)
    algorithms_list.append(baseline_algorithm)

  res, _ = mc.dask_MC_Simulations(
      N=N,
      NR=NR,
      environment_parameters=environment_parameters,
      environment=environment,
      algorithms=algorithms_list,
      algorithms_parameters=param_objects,
      desired_metrics=desired_metrics,
      h0=h0,
      num_workers=num_workers,
      num_chunks=num_chunks,
      seed=seed,
      scheduler=scheduler,
  )

  slow_curve = None
  slow_min = None
  slow_min_idx = None
  slow_diag = None

  if convergence_metric_parameters is None:
    convergence_metric_parameters = {}
  else:
    convergence_metric_parameters = dict(convergence_metric_parameters)

  n_slope_samples = int(convergence_metric_parameters.get('n_slope_samples', 1000))

  if slow_label is not None and slow_label in res:
    slow_perf = res[slow_label]['performance']
    slow_curve = performance_metric_scale(slow_perf)
    slow_min, slow_min_idx, slow_diag = find_minimum_lowest_slope(
        slow_curve, n_slope_samples=n_slope_samples
    )

  is_target_error = (
      convergence_metric == db_accumulated_target_error
      or getattr(convergence_metric, '__name__', '') == 'db_accumulated_target_error'
  )
  if is_target_error:
    if 'target' not in convergence_metric_parameters or convergence_metric_parameters['target'] is None:
      if slow_min is not None:
        convergence_metric_parameters['target'] = slow_min
      else:
        raise ValueError(
            "Convergence metric 'db_accumulated_target_error' requires 'target' parameter, "
            "or a baseline_algorithm/baseline_parameter_class to compute the slow filter minimum."
        )
    if 'tolerance' not in convergence_metric_parameters or convergence_metric_parameters['tolerance'] is None:
      convergence_metric_parameters['tolerance'] = 0.2
    if 'n_slope_samples' not in convergence_metric_parameters:
      convergence_metric_parameters['n_slope_samples'] = n_slope_samples

  convergence_grid = np.full(grid_shape, np.nan, dtype=np.float64)
  steady_state_grid = np.full(grid_shape, np.nan, dtype=np.float64)
  steady_state_derivative_grid = np.full(grid_shape, np.nan, dtype=np.float64)

  for combo_idx in range(len(all_combos)):
    multi_idx = np.unravel_index(combo_idx, grid_shape)
    label = labels_grid[multi_idx]
    perf_curve = res[label]['performance']
    scaled_curve = performance_metric_scale(perf_curve)

    # Steady-state floor estimation
    try:
      ss_val = float(np.mean(scaled_curve[-floor_estimate_samples:]))
    except Exception:
      ss_val = np.nan
    steady_state_grid[multi_idx] = ss_val

    # Steady-state derivative calculation
    try:
      ss_deriv = float(mean_absolute_steady_state_derivative(scaled_curve, floor_estimate_samples=floor_estimate_samples))
    except Exception:
      ss_deriv = np.nan
    steady_state_derivative_grid[multi_idx] = ss_deriv

    # Convergence metric evaluation with divergence/NaN penalty
    try:
      if np.isnan(scaled_curve).any() or np.isinf(scaled_curve).any():
        c_val = np.nan
      else:
        c_val = float(convergence_metric(scaled_curve, **convergence_metric_parameters))
    except Exception:
      c_val = np.nan
    convergence_grid[multi_idx] = c_val

  # Find best parameters
  valid_mask = ~np.isnan(convergence_grid) & ~np.isinf(convergence_grid)
  if not np.any(valid_mask):
    opt_multi_idx = tuple(0 for _ in grid_shape)
  else:
    if minimize_convergence_metric:
      penalized = np.where(valid_mask, convergence_grid, np.inf)
      best_flat = np.argmin(penalized)
    else:
      penalized = np.where(valid_mask, convergence_grid, -np.inf)
      best_flat = np.argmax(penalized)
    opt_multi_idx = np.unravel_index(best_flat, grid_shape)

  best_parameters = {param_names[d]: param_arrays[d][opt_multi_idx[d]] for d in range(len(param_names))}
  best_metric_value = float(convergence_grid[opt_multi_idx])
  best_steady_state = float(steady_state_grid[opt_multi_idx])
  best_label = labels_grid[opt_multi_idx]

  return {
      'best_parameters': best_parameters,
      'best_metric_value': best_metric_value,
      'best_steady_state': best_steady_state,
      'optimal_index': opt_multi_idx,
      'convergence_grid': convergence_grid,
      'steady_state_grid': steady_state_grid,
      'steady_state_derivative_grid': steady_state_derivative_grid,
      'grid_axes': conv_dict,
      'measures': res,
      'best_label': best_label,
      'labels_grid': labels_grid,
      'slow_baseline_curve': slow_curve,
      'slow_baseline_min': slow_min,
      'slow_baseline_min_idx': slow_min_idx,
      'slow_baseline_diagnostics': slow_diag,
      'convergence_metric_name': getattr(convergence_metric, '__name__', str(convergence_metric)),
      'convergence_metric_target': convergence_metric_parameters.get('target'),
      'convergence_metric_tolerance': convergence_metric_parameters.get('tolerance'),
      'n_slope_samples': n_slope_samples,
  }


def plot_grid_search_histograms(
    grid_results: dict[str, Any],
    bins: int = 15,
    figsize: tuple[float, float] = (12, 4.5),
) -> plt.Figure:
  """Plot histograms of the convergence metric and steady-state performance metric.

  Parameters
  ----------
  grid_results : dict
    Output dictionary returned by grid_search.
  bins : int, optional
    Number of histogram bins (default is 15).
  figsize : tuple, optional
    Figure dimensions (width, height).

  Returns
  -------
  plt.Figure
    The created matplotlib figure.
  """
  conv_data = grid_results['convergence_grid'].ravel()
  ss_data = grid_results['steady_state_grid'].ravel()

  valid_conv = conv_data[np.isfinite(conv_data)]
  valid_ss = ss_data[np.isfinite(ss_data)]

  fig, axes = plt.subplots(1, 2, figsize=figsize)

  # 1. Convergence metric histogram
  ax0 = axes[0]
  ax0.hist(valid_conv, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
  best_conv = grid_results['best_metric_value']
  ax0.axvline(best_conv, color='crimson', linestyle='--', linewidth=2, label=f'Optimal ({best_conv:.2f})')
  ax0.axvline(np.mean(valid_conv), color='orange', linestyle=':', linewidth=1.5, label=f'Mean ({np.mean(valid_conv):.2f})')
  ax0.set_xlabel('Convergence Metric (dB \u00d7 samples)', fontweight='bold')
  ax0.set_ylabel('Frequency (Count)', fontweight='bold')
  ax0.set_title('Convergence Metric Histogram', fontsize=11)
  ax0.grid(True, alpha=0.3)
  ax0.legend(fontsize=9)

  # 2. Steady-state performance metric histogram
  ax1 = axes[1]
  ax1.hist(valid_ss, bins=bins, color='seagreen', edgecolor='black', alpha=0.7)
  best_ss = grid_results['best_steady_state']
  mean_ss = np.mean(valid_ss)
  std_ss = np.std(valid_ss)
  ax1.axvline(best_ss, color='crimson', linestyle='--', linewidth=2, label=f'Optimal ({best_ss:.2f} dB)')
  ax1.axvline(mean_ss, color='orange', linestyle=':', linewidth=1.5, label=f'Mean ({mean_ss:.2f} dB \u00b1 {std_ss:.2f})')
  ax1.set_xlabel('Steady-State Metric [dB]', fontweight='bold')
  ax1.set_ylabel('Frequency (Count)', fontweight='bold')
  ax1.set_title('Steady-State Performance Metric Histogram', fontsize=11)
  ax1.grid(True, alpha=0.3)
  ax1.legend(fontsize=9)

  fig.suptitle('Hyperparameter Grid Search Metric Distributions', fontsize=12, y=1.02)
  plt.tight_layout()
  return fig


def plot_grid_search_pairwise_heatmaps(
    grid_results: dict[str, Any],
    metric_type: str = 'convergence',
    cmap: str = 'viridis',
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
  """Plot all pairwise heatmaps for convergence hyperparameters.

  Each 2D heatmap plots a pair of hyperparameters, while fixing all other
  hyperparameters to their global optimal values. Highlights the optimal
  parameter combination on each subplot.

  Parameters
  ----------
  grid_results : dict
    Output dictionary returned by grid_search.
  metric_type : str, optional
    'convergence' (default) or 'steady_state'.
  cmap : str, optional
    Matplotlib colormap name.
  figsize : tuple, optional
    Figure dimensions (width, height).

  Returns
  -------
  plt.Figure
    The created matplotlib figure.
  """
  grid_axes = grid_results['grid_axes']
  param_names = list(grid_axes.keys())
  K = len(param_names)
  opt_idx = grid_results['optimal_index']
  if metric_type == 'convergence':
    grid_data = grid_results['convergence_grid']
    best_metric_val = grid_results['best_metric_value']
    metric_label = "Convergence (dB Area)"
  elif metric_type == 'steady_state_derivative':
    grid_data = grid_results.get('steady_state_derivative_grid', grid_results['steady_state_grid'])
    best_metric_val = float(grid_data[opt_idx])
    metric_label = "Steady-State Derivative (dB/sample)"
  else:
    grid_data = grid_results['steady_state_grid']
    best_metric_val = grid_results['best_steady_state']
    metric_label = "Steady-State (dB)"

  if K == 1:
    fig, ax = plt.subplots(figsize=figsize or (6, 4))
    x_vals = grid_axes[param_names[0]]
    ax.plot(x_vals, grid_data, marker='o', label='Metric')
    ax.plot(x_vals[opt_idx[0]], best_metric_val, marker='*', markersize=14, color='red', label='Optimal')
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(metric_label)
    ax.set_title(f"{param_names[0]} Optimization\nOptimal: {param_names[0]}={_format_param_val(x_vals[opt_idx[0]])} | Best: {best_metric_val:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

  if K == 2:
    pairs = [(0, 1)]
  else:
    pairs = list(itertools.combinations(range(K), 2))

  num_pairs = len(pairs)
  ncols = min(3, num_pairs)
  nrows = int(np.ceil(num_pairs / ncols))
  if figsize is None:
    figsize = (6.0 * ncols, 5.0 * nrows)

  fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

  for p_idx, (i, j) in enumerate(pairs):
    row, col = p_idx // ncols, p_idx % ncols
    ax = axes[row, col]

    # Build slicing tuple fixing other dimensions to optimal values
    slicing = [opt_idx[d] for d in range(K)]
    slicing[i] = slice(None)
    slicing[j] = slice(None)
    slice_2d = grid_data[tuple(slicing)]
    if i > j:
      slice_2d = slice_2d.T

    y_vals = grid_axes[param_names[i]]
    x_vals = grid_axes[param_names[j]]

    im = ax.imshow(slice_2d, aspect='auto', origin='lower', cmap=cmap)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([_format_param_val(v) for v in x_vals], rotation=30)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([_format_param_val(v) for v in y_vals])
    ax.set_xlabel(param_names[j], fontweight='bold')
    ax.set_ylabel(param_names[i], fontweight='bold')

    # Mark optimal coordinate
    ax.plot(opt_idx[j], opt_idx[i], marker='*', markersize=16, color='red',
            markeredgecolor='white', markeredgewidth=1.5, label='Optimal')

    other_dims = [d for d in range(K) if d not in (i, j)]
    other_str = ", ".join([f"{param_names[d]}*={_format_param_val(grid_axes[param_names[d]][opt_idx[d]])}" for d in other_dims])
    opt_pair_str = f"{param_names[i]}*={_format_param_val(y_vals[opt_idx[i]])}, {param_names[j]}*={_format_param_val(x_vals[opt_idx[j]])}"
    sub_title = f"{param_names[i]} vs {param_names[j]}"
    if other_str:
      sub_title += f"\nFixed @ opt: {other_str}"
    sub_title += f"\nOptimal: {opt_pair_str} | Best: {best_metric_val:.2f}"
    ax.set_title(sub_title, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label, fontsize=8)

  # Turn off unused subplots
  for p_idx in range(num_pairs, nrows * ncols):
    axes[p_idx // ncols, p_idx % ncols].axis('off')

  fig.suptitle(f"Pairwise Hyperparameter Slices at Optimal Settings ({metric_label})", fontsize=13, y=1.02)
  plt.tight_layout()
  return fig


def interactive_grid_search_heatmap(
    grid_results: dict[str, Any],
    default_metric: str = 'convergence',
    cmap: str = 'viridis',
):
  """Create an interactive heatmap with dropdowns for axes and sliders for remaining hyperparameters.

  Parameters
  ----------
  grid_results : dict
    Output dictionary returned by grid_search.
  default_metric : str, optional
    Default metric to display ('convergence' or 'steady_state').
  cmap : str, optional
    Colormap name.

  Returns
  -------
  ipywidgets.Widget or None
    Interactive widget container.
  """
  try:
    import ipywidgets as widgets
    from IPython.display import display
  except ImportError:
    print("ipywidgets is required for interactive heatmap. Please install ipywidgets.")
    return None

  grid_axes = grid_results['grid_axes']
  param_names = list(grid_axes.keys())
  K = len(param_names)
  opt_idx = grid_results['optimal_index']
  best_params = grid_results['best_parameters']
  best_val = grid_results['best_metric_value']

  if K < 2:
    print("Interactive heatmap requires at least 2 hyperparameters in grid_axes.")
    return plot_grid_search_pairwise_heatmaps(grid_results, metric_type=default_metric, cmap=cmap)

  # Dropdowns for axes and metric
  metric_options = [('Convergence Metric', 'convergence'), ('Steady-State Metric', 'steady_state')]
  if 'steady_state_derivative_grid' in grid_results:
    metric_options.append(('Steady-State Derivative', 'steady_state_derivative'))

  metric_dropdown = widgets.Dropdown(
      options=metric_options,
      value=default_metric,
      description='Metric:',
      layout=widgets.Layout(width='260px')
  )
  x_dropdown = widgets.Dropdown(
      options=param_names,
      value=param_names[0],
      description='X-Axis:',
      layout=widgets.Layout(width='260px')
  )
  y_dropdown = widgets.Dropdown(
      options=param_names,
      value=param_names[1],
      description='Y-Axis:',
      layout=widgets.Layout(width='260px')
  )

  # Create a selection slider for each parameter initialized to its optimal index
  param_sliders = {}
  for d, p_name in enumerate(param_names):
    options = [(_format_param_val(v), idx) for idx, v in enumerate(grid_axes[p_name])]
    slider = widgets.SelectionSlider(
        options=options,
        value=opt_idx[d],
        description=f"{p_name}:",
        layout=widgets.Layout(width='320px')
    )
    param_sliders[p_name] = slider

  out = widgets.Output()

  def update_plot(*args):
    x_name = x_dropdown.value
    y_name = y_dropdown.value
    m_type = metric_dropdown.value

    # Update slider enabled state
    for p_name, slider in param_sliders.items():
      slider.disabled = (p_name in (x_name, y_name))

    with out:
      out.clear_output(wait=True)
      if x_name == y_name:
        print("Please choose two different parameters for X-Axis and Y-Axis.")
        return

      i = param_names.index(y_name)
      j = param_names.index(x_name)
      if m_type == 'convergence':
        grid_data = grid_results['convergence_grid']
        metric_str = "Convergence Area"
      elif m_type == 'steady_state_derivative':
        grid_data = grid_results.get('steady_state_derivative_grid', grid_results['steady_state_grid'])
        metric_str = "SS Derivative"
      else:
        grid_data = grid_results['steady_state_grid']
        metric_str = "Steady-State MSD"

      slicing = [param_sliders[p].value for p in param_names]
      slicing[i] = slice(None)
      slicing[j] = slice(None)
      slice_2d = grid_data[tuple(slicing)]
      if i > j:
        slice_2d = slice_2d.T

      y_vals = grid_axes[y_name]
      x_vals = grid_axes[x_name]

      fig, ax = plt.subplots(figsize=(7, 5.5))
      im = ax.imshow(slice_2d, aspect='auto', origin='lower', cmap=cmap)

      ax.set_xticks(range(len(x_vals)))
      ax.set_xticklabels([_format_param_val(v) for v in x_vals], rotation=30)
      ax.set_yticks(range(len(y_vals)))
      ax.set_yticklabels([_format_param_val(v) for v in y_vals])
      ax.set_xlabel(x_name, fontweight='bold', fontsize=11)
      ax.set_ylabel(y_name, fontweight='bold', fontsize=11)

      # Mark optimal point projection
      ax.plot(opt_idx[j], opt_idx[i], marker='*', markersize=18, color='red',
              markeredgecolor='white', markeredgewidth=1.5, label='Global Optimum')

      # Check if sliders are currently at optimal values
      other_params = [p for p in param_names if p not in (x_name, y_name)]
      at_opt = all(param_sliders[p].value == opt_idx[param_names.index(p)] for p in other_params)

      fixed_info = ", ".join([f"{p}={_format_param_val(grid_axes[p][param_sliders[p].value])}" for p in other_params])
      opt_info = ", ".join([f"{k}*={_format_param_val(v)}" for k, v in best_params.items()])
      opt_status = "[Slice @ Global Optimum]" if at_opt else "[Off-optimum Slice]"

      ax.set_title(
          f"{y_name} vs {x_name} {opt_status}\n"
          f"Global Optimum: ({opt_info}) | Best {metric_str}: {best_val:.2f}\n"
          f"Fixed parameters: {fixed_info if fixed_info else 'None'}",
          fontsize=10
      )
      cbar = fig.colorbar(im, ax=ax)
      cbar.set_label(f"{m_type.capitalize()} metric", fontsize=10)
      ax.legend(loc='upper right')
      plt.tight_layout()
      plt.show()

  x_dropdown.observe(update_plot, names='value')
  y_dropdown.observe(update_plot, names='value')
  metric_dropdown.observe(update_plot, names='value')
  for slider in param_sliders.values():
    slider.observe(update_plot, names='value')

  sliders_box = widgets.VBox(list(param_sliders.values()))
  controls = widgets.VBox([
      widgets.HBox([x_dropdown, y_dropdown, metric_dropdown]),
      widgets.Label("Sliders for fixed dimensions:"),
      sliders_box
  ])
  ui = widgets.VBox([controls, out])

  # Initial render
  update_plot()
  return ui


def _validate_numeric(val: Any, param_name: str) -> float | int:
  """Validate that a parameter is strictly numeric and not a string expression."""
  if isinstance(val, str):
    raise TypeError(
        f"Parameter '{param_name}' must be numeric, got string '{val}'. "
        "String expressions are not allowed; please provide numeric values directly."
    )
  if not isinstance(val, (int, float, np.integer, np.floating)):
    raise TypeError(
        f"Parameter '{param_name}' must be numeric, got {type(val).__name__}: {val}"
    )
  return val


def _validate_numeric_array(val: Any, param_name: str) -> NDArray[np.float64]:
  """Validate that a parameter is a list/sequence of numbers and convert to a float64 NDArray."""
  if isinstance(val, str):
    raise TypeError(
        f"Parameter '{param_name}' must be a sequence of numbers, got string '{val}'. "
        "String expressions are not allowed; please provide numeric values directly."
    )
  if not isinstance(val, (list, tuple, np.ndarray)):
    raise TypeError(
        f"Parameter '{param_name}' must be a list or array of numbers, got {type(val).__name__}."
    )
  for item in val:
    if isinstance(item, str):
      raise TypeError(
          f"Elements of parameter '{param_name}' must be numeric, got string '{item}'. "
          "String expressions are not allowed; please provide numeric values directly."
      )
    if not isinstance(item, (int, float, np.integer, np.floating)):
      raise TypeError(
          f"Elements of parameter '{param_name}' must be numeric, got {type(item).__name__}: {item}"
      )
  return np.asarray(val, dtype=np.float64)


def load_experiment_configs(folder_path: str | Path) -> dict[str, Any]:
  """Load and validate standardized environment and algorithm YAML configuration files.

  Expects 'environment_config.yml' and 'algorithm_config.yml' inside folder_path.
  Raises FileNotFoundError if either file is missing.
  Raises TypeError if any numeric parameter is provided as a string expression.

  Parameters
  ----------
  folder_path : str or Path
    Path to the experiment folder containing the configuration files.

  Returns
  -------
  dict[str, Any]
    Dictionary containing parsed and validated parameters ready for grid_search.
  """
  folder = Path(folder_path)
  env_file = folder / "environment_config.yml"
  alg_file = folder / "algorithm_config.yml"

  if not env_file.is_file():
    raise FileNotFoundError(
        f"Environment configuration file not found: {env_file.resolve()}"
    )
  if not alg_file.is_file():
    raise FileNotFoundError(
        f"Algorithm configuration file not found: {alg_file.resolve()}"
    )

  with open(env_file, 'r') as f:
    env_cfg = yaml.safe_load(f) or {}
  with open(alg_file, 'r') as f:
    alg_cfg = yaml.safe_load(f) or {}

  # Validate simulation / Monte Carlo parameters
  N = int(_validate_numeric(env_cfg.get('N', 50000), 'N'))
  NR = int(_validate_numeric(env_cfg.get('NR', 256), 'NR'))
  num_workers = int(_validate_numeric(env_cfg.get('num_workers', 4), 'num_workers'))
  num_chunks = int(_validate_numeric(env_cfg.get('num_chunks', 8), 'num_chunks'))
  seed = int(_validate_numeric(env_cfg.get('seed', 42), 'seed'))

  # Plant & system setup
  L = int(_validate_numeric(env_cfg['L'], 'L'))
  plant_type = env_cfg.get('plant_type', 'normalized_sinc')
  if plant_type == 'normalized_sinc':
    ho = np.sinc(np.linspace(0, 1, L))
    ho = ho / np.linalg.norm(ho)
  elif plant_type == 'custom':
    ho = _validate_numeric_array(env_cfg['ho'], 'ho')
  else:
    raise ValueError(f"Unknown plant_type: {plant_type}")

  h0 = np.zeros(L, dtype=np.float64)
  AR = _validate_numeric_array(env_cfg['AR'], 'AR')
  var_x = float(_validate_numeric(env_cfg['var_x'], 'var_x'))

  # Environment behavior and parameter namedtuple
  behavior_name = env_cfg['behavior']
  if not hasattr(filters, behavior_name):
    raise AttributeError(f"Environment behavior '{behavior_name}' not found in filters module.")
  environment = getattr(filters, behavior_name)

  env_param_cls_name = env_cfg['parameters_class']
  if not hasattr(filters, env_param_cls_name):
    raise AttributeError(f"Environment parameters class '{env_param_cls_name}' not found in filters module.")
  env_param_cls = getattr(filters, env_param_cls_name)

  # Build environment parameters namedtuple
  env_kwargs = {}
  for field in env_param_cls._fields:
    if field == 'ho':
      env_kwargs['ho'] = ho
    elif field == 'AR':
      env_kwargs['AR'] = AR
    elif field == 'var_x':
      env_kwargs['var_x'] = var_x
    elif field in env_cfg:
      env_kwargs[field] = float(_validate_numeric(env_cfg[field], field))
    else:
      raise ValueError(f"Missing required field '{field}' in {env_file}")
  env_parameters = env_param_cls(**env_kwargs)

  # Algorithm specification
  alg_name = alg_cfg['algorithm']
  if not hasattr(filters, alg_name):
    raise AttributeError(f"Algorithm '{alg_name}' not found in filters module.")
  algorithm = getattr(filters, alg_name)

  alg_param_cls_name = alg_cfg.get('algorithm_parameter_class')
  algorithm_parameter_class = (
      getattr(filters, alg_param_cls_name)
      if alg_param_cls_name and hasattr(filters, alg_param_cls_name)
      else None
  )

  # Performance parameters (fixed)
  perf_raw = alg_cfg.get('performance_parameters', {})
  if not isinstance(perf_raw, dict):
    raise TypeError("performance_parameters in algorithm_config.yml must be a mapping (dict).")
  algorithm_performance_parameters = {}
  for k, v in perf_raw.items():
    algorithm_performance_parameters[k] = float(_validate_numeric(v, k))

  # Convergence parameters grid
  conv_raw = alg_cfg.get('convergence_parameters', {})
  if not isinstance(conv_raw, dict):
    raise TypeError("convergence_parameters in algorithm_config.yml must be a mapping (dict).")
  algorithm_convergence_parameters = {}
  for k, v in conv_raw.items():
    algorithm_convergence_parameters[k] = _validate_numeric_array(v, k)

  # Baseline algorithm & parameters for comparative learning curves
  base_alg_name = alg_cfg.get('baseline_algorithm')
  baseline_algorithm = (
      getattr(filters, base_alg_name)
      if base_alg_name and hasattr(filters, base_alg_name)
      else None
  )
  base_cls_name = alg_cfg.get('baseline_parameter_class')
  baseline_parameter_class = (
      getattr(filters, base_cls_name)
      if base_cls_name and hasattr(filters, base_cls_name)
      else None
  )
  noise_param_name = alg_cfg.get('noise_param_name', 'var_eta')

  # Optimization metrics
  perf_metric_name = alg_cfg.get('performance_metric', 'squared_deviation')
  if hasattr(metrics, perf_metric_name):
    performance_metric = getattr(metrics, perf_metric_name)
  else:
    raise AttributeError(f"Performance metric '{perf_metric_name}' not found in metrics module.")

  conv_metric_name = alg_cfg.get('convergence_metric', 'db_accumulated_target_error')
  if conv_metric_name == 'db_accumulated_target_error':
    convergence_metric = db_accumulated_target_error
  elif conv_metric_name == 'db_area_of_convergence':
    convergence_metric = db_area_of_convergence
  elif conv_metric_name == 'time_to_convergence':
    convergence_metric = time_to_convergence
  elif hasattr(metrics, conv_metric_name):
    convergence_metric = getattr(metrics, conv_metric_name)
  else:
    raise AttributeError(f"Convergence metric '{conv_metric_name}' not found.")

  conv_params_raw = alg_cfg.get('convergence_metric_parameters', {})
  if not isinstance(conv_params_raw, dict):
    raise TypeError("convergence_metric_parameters in algorithm_config.yml must be a mapping (dict).")
  convergence_metric_parameters = {}
  for k, v in conv_params_raw.items():
    if v is not None:
      if k == 'n_slope_samples':
        convergence_metric_parameters[k] = int(_validate_numeric(v, k))
      else:
        convergence_metric_parameters[k] = float(_validate_numeric(v, k))
  if conv_metric_name == 'db_accumulated_target_error':
    if 'tolerance' not in convergence_metric_parameters:
      convergence_metric_parameters['tolerance'] = 0.2
    if 'n_slope_samples' not in convergence_metric_parameters:
      convergence_metric_parameters['n_slope_samples'] = 1000

  floor_estimate_samples = int(
      _validate_numeric(alg_cfg.get('floor_estimate_samples', 100), 'floor_estimate_samples')
  )
  minimize_convergence_metric = bool(
      alg_cfg.get('minimize_convergence_metric', True)
  )

  return {
      'folder': folder,
      'N': N,
      'NR': NR,
      'num_workers': num_workers,
      'num_chunks': num_chunks,
      'seed': seed,
      'L': L,
      'ho': ho,
      'h0': h0,
      'environment': environment,
      'environment_parameters': env_parameters,
      'algorithm': algorithm,
      'algorithm_parameter_class': algorithm_parameter_class,
      'algorithm_performance_parameters': algorithm_performance_parameters,
      'algorithm_convergence_parameters': algorithm_convergence_parameters,
      'baseline_algorithm': baseline_algorithm,
      'baseline_parameter_class': baseline_parameter_class,
      'noise_param_name': noise_param_name,
      'performance_metric': performance_metric,
      'convergence_metric': convergence_metric,
      'convergence_metric_parameters': convergence_metric_parameters,
      'floor_estimate_samples': floor_estimate_samples,
      'minimize_convergence_metric': minimize_convergence_metric,
  }


def plot_grid_search_learning_curves(
    grid_results: dict[str, Any],
    environment: Callable,
    environment_parameters: object,
    h0: NDArray[np.float64],
    baseline_algorithm: Callable | None = None,
    baseline_parameter_class: type | None = None,
    noise_param_name: str = 'var_eta',
    noise_param_val: float | None = None,
    fast_epsilon_key: str = 'epsilon_1',
    slow_epsilon_key: str = 'epsilon_2',
    v_tilde_0: float = 1.0,
    N: int = 50000,
    NR: int = 256,
    num_workers: int = 4,
    num_chunks: int = 8,
    seed: int = 42,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
  """Plot comparative MSD learning curves for optimal, median, and worst grid configurations vs constituent baselines.

  Parameters
  ----------
  grid_results : dict
    Output dictionary returned by grid_search.
  environment : Callable
    Environment generator behavior function.
  environment_parameters : object
    Environment parameters namedtuple.
  h0 : NDArray[np.float64]
    Initial filter taps.
  baseline_algorithm : Callable, optional
    Constituent baseline algorithm function (e.g. sKF_algorithm).
  baseline_parameter_class : type, optional
    Parameter namedtuple for the baseline algorithm (e.g. sKF_parameters).
  noise_param_name : str, optional
    Name of the noise variance/scale parameter in baseline ('var_eta' or 'b_eta').
  noise_param_val : float, optional
    Value of the observation noise parameter. If None, extracts from grid_results.
  fast_epsilon_key : str, optional
    Key name for the fast filter process noise variance.
  slow_epsilon_key : str, optional
    Key name for the slow filter process noise variance.
  v_tilde_0 : float, optional
    Initial variance estimate.
  N : int, optional
    Number of iterations.
  NR : int, optional
    Number of Monte Carlo realizations.
  num_workers : int, optional
    Number of Dask workers.
  num_chunks : int, optional
    Number of chunks.
  seed : int, optional
    Random seed.
  figsize : tuple, optional
    Figure dimensions.

  Returns
  -------
  plt.Figure
    The created matplotlib figure.
  """
  c_grid = grid_results['convergence_grid']
  opt_idx = grid_results['optimal_index']
  worst_flat = np.nanargmax(c_grid)
  worst_idx = np.unravel_index(worst_flat, c_grid.shape)

  valid_scores = c_grid[~np.isnan(c_grid)]
  median_val = np.median(valid_scores)
  median_flat = np.argmin(np.abs(c_grid - median_val))
  median_idx = np.unravel_index(median_flat, c_grid.shape)

  best_label = grid_results['labels_grid'][opt_idx]
  worst_label = grid_results['labels_grid'][worst_idx]
  median_label = grid_results['labels_grid'][median_idx]

  msd_best = default_db_scale(grid_results['measures'][best_label]['performance'])
  msd_worst = default_db_scale(grid_results['measures'][worst_label]['performance'])
  msd_median = default_db_scale(grid_results['measures'][median_label]['performance'])

  has_baselines = baseline_algorithm is not None and baseline_parameter_class is not None
  msd_fast = None
  msd_slow = None
  eps_fast = grid_results['best_parameters'].get(fast_epsilon_key, 1e-5)
  perf_params = grid_results.get('algorithm_performance_parameters', {})
  eps_slow = perf_params.get(slow_epsilon_key, 5e-9)

  if has_baselines:
    if noise_param_val is None:
      noise_param_val = perf_params.get(noise_param_name, 1e-3)

    fast_kwargs = {
        'label': 'Fast Baseline',
        'epsilon': eps_fast,
        noise_param_name: noise_param_val,
        'v_tilde_0': v_tilde_0,
    }
    fast_param = baseline_parameter_class(**fast_kwargs)

    # Reuse slow baseline curve if already computed during grid_search
    if grid_results.get('slow_baseline_curve') is not None:
      msd_slow = grid_results['slow_baseline_curve']
      baseline_measures, _ = mc.dask_MC_Simulations(
          N=N,
          NR=NR,
          environment_parameters=environment_parameters,
          environment=environment,
          algorithms=[baseline_algorithm],
          algorithms_parameters=[fast_param],
          desired_metrics={'MSD': metrics.squared_deviation},
          h0=h0,
          num_workers=num_workers,
          num_chunks=num_chunks,
          seed=seed,
      )
      msd_fast = default_db_scale(baseline_measures['Fast Baseline']['MSD'])
    else:
      slow_kwargs = {
          'label': 'Slow Baseline',
          'epsilon': eps_slow,
          noise_param_name: noise_param_val,
          'v_tilde_0': v_tilde_0,
      }
      slow_param = baseline_parameter_class(**slow_kwargs)
      baseline_measures, _ = mc.dask_MC_Simulations(
          N=N,
          NR=NR,
          environment_parameters=environment_parameters,
          environment=environment,
          algorithms=[baseline_algorithm, baseline_algorithm],
          algorithms_parameters=[fast_param, slow_param],
          desired_metrics={'MSD': metrics.squared_deviation},
          h0=h0,
          num_workers=num_workers,
          num_chunks=num_chunks,
          seed=seed,
      )
      msd_fast = default_db_scale(baseline_measures['Fast Baseline']['MSD'])
      msd_slow = default_db_scale(baseline_measures['Slow Baseline']['MSD'])

  conv_metric_name = grid_results.get('convergence_metric_name', '')
  unit_str = 'dB' if 'target_error' in conv_metric_name else 'dB*iter'

  fig, ax = plt.subplots(figsize=figsize)
  ax.plot(
      msd_best,
      color='crimson',
      lw=2.2,
      label=f"Optimal ({grid_results['best_metric_value']:.1f} {unit_str})",
  )
  ax.plot(
      msd_median,
      color='orange',
      lw=1.5,
      ls='--',
      label=f"Median ({c_grid[median_idx]:.1f} {unit_str})",
  )
  ax.plot(
      msd_worst,
      color='brown',
      lw=1.2,
      ls=':',
      label=f"Worst ({c_grid[worst_idx]:.1f} {unit_str})",
  )

  if msd_fast is not None:
    ax.plot(
        msd_fast,
        color='blue',
        lw=1.2,
        ls='-.',
        alpha=0.7,
        label=f"Fast Baseline ({fast_epsilon_key}={eps_fast:.1e})",
    )
  if msd_slow is not None:
    ax.plot(
        msd_slow,
        color='green',
        lw=1.2,
        ls='-.',
        alpha=0.7,
        label=f"Slow Baseline ({slow_epsilon_key}={eps_slow:.1e})",
    )

  target_val = grid_results.get('convergence_metric_target')
  if target_val is not None:
    tol_val = grid_results.get('convergence_metric_tolerance', 0.2)
    ax.axhline(target_val, color='gray', ls=':', lw=1.2, alpha=0.8, label=f"Target ({target_val:.2f} dB)")
    ax.axhspan(target_val - tol_val, target_val + tol_val, color='gray', alpha=0.15, label=f"Tolerance (±{tol_val:.2f} dB)")

  ax.set_xlabel("Iteration", fontsize=11)
  ax.set_ylabel("Normalized MSD [dB]", fontsize=11)
  ax.set_title("MSD Learning Curves: Optimal vs. Worst, Median & Baselines", fontsize=13)
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=9, loc='upper right')

  min_val = np.nanmin(msd_best)
  if msd_slow is not None:
    min_val = min(min_val, np.nanmin(msd_slow))
  ax.set_ylim([min_val - 3, 5])
  plt.tight_layout()
  return fig


def run_grid_search_experiment(
    folder_path: str | Path,
    save_figures: bool = True,
    save_summary_yaml: bool = True,
) -> dict[str, Any]:
  """Run a complete grid search experiment loaded from standardized YAML configs and save figures as PDFs.

  Parameters
  ----------
  folder_path : str or Path
    Path to the experiment folder containing 'environment_config.yml' and 'algorithm_config.yml'.
  save_figures : bool, optional
    Whether to generate and save PDF figures in '<folder_path>/figures/'.
  save_summary_yaml : bool, optional
    Whether to save 'optimal_parameters.yml' and 'grid_search_summary.yml'.

  Returns
  -------
  dict[str, Any]
    Result dictionary from grid_search with additional metadata.
  """
  cfg = load_experiment_configs(folder_path)

  t_start = time.time()
  noise_p_name = cfg.get('noise_param_name', 'var_eta')
  noise_p_val = cfg['algorithm_performance_parameters'].get(noise_p_name)

  grid_results = grid_search(
      N=cfg['N'],
      NR=cfg['NR'],
      h0=cfg['h0'],
      environment=cfg['environment'],
      environment_parameters=cfg['environment_parameters'],
      algorithm=cfg['algorithm'],
      algorithm_convergence_parameters=cfg['algorithm_convergence_parameters'],
      algorithm_performance_parameters=cfg['algorithm_performance_parameters'],
      baseline_algorithm=cfg.get('baseline_algorithm'),
      baseline_parameter_class=cfg.get('baseline_parameter_class'),
      noise_param_name=noise_p_name,
      noise_param_val=noise_p_val,
      performance_metric=cfg['performance_metric'],
      convergence_metric=cfg['convergence_metric'],
      convergence_metric_parameters=cfg.get('convergence_metric_parameters'),
      floor_estimate_samples=cfg['floor_estimate_samples'],
      num_workers=cfg['num_workers'],
      num_chunks=cfg['num_chunks'],
      seed=cfg['seed'],
      minimize_convergence_metric=cfg['minimize_convergence_metric'],
  )
  t_elapsed = time.time() - t_start

  # Store performance parameters and elapsed time in grid_results
  grid_results['algorithm_performance_parameters'] = cfg['algorithm_performance_parameters']
  grid_results['computation_time_seconds'] = t_elapsed

  folder = Path(folder_path)

  if save_summary_yaml:
    # 1. Save optimal_parameters.yml (performance params first, optimal convergence params last)
    opt_dict = {}
    for k, v in cfg['algorithm_performance_parameters'].items():
      opt_dict[k] = float(v)
    for k, v in grid_results['best_parameters'].items():
      opt_dict[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

    opt_file = folder / "optimal_parameters.yml"
    with open(opt_file, 'w') as f:
      yaml.dump(opt_dict, f, sort_keys=False, default_flow_style=False)

    # 2. Save grid_search_summary.yml (all numeric metrics of the grid search)
    c_grid = grid_results['convergence_grid']
    ss_grid = grid_results['steady_state_grid']
    deriv_grid = grid_results.get('steady_state_derivative_grid', np.array([]))

    valid_conv = c_grid[np.isfinite(c_grid)]
    valid_ss = ss_grid[np.isfinite(ss_grid)]
    valid_deriv = deriv_grid[np.isfinite(deriv_grid)] if deriv_grid.size > 0 else np.array([])
    total_combos = int(np.prod(c_grid.shape))

    summary_dict = {
        'computation_time_seconds': float(round(t_elapsed, 2)),
        'total_combinations': total_combos,
        'convergence_metric_name': grid_results.get('convergence_metric_name'),
        'convergence_metric_target': float(grid_results['convergence_metric_target']) if grid_results.get('convergence_metric_target') is not None else None,
        'convergence_metric_tolerance': float(grid_results['convergence_metric_tolerance']) if grid_results.get('convergence_metric_tolerance') is not None else None,
        'n_slope_samples': int(grid_results['n_slope_samples']) if grid_results.get('n_slope_samples') is not None else 1000,
        'slow_baseline_min_db': float(grid_results['slow_baseline_min']) if grid_results.get('slow_baseline_min') is not None else None,
        'slow_baseline_min_slope': float(grid_results['slow_baseline_diagnostics']['min_slope']) if grid_results.get('slow_baseline_diagnostics') and 'min_slope' in grid_results['slow_baseline_diagnostics'] else None,
        'optimal_convergence_metric': float(grid_results['best_metric_value']),
        'mean_convergence_metric': float(np.mean(valid_conv)) if len(valid_conv) > 0 else None,
        'std_convergence_metric': float(np.std(valid_conv)) if len(valid_conv) > 0 else None,
        'optimal_steady_state_db': float(grid_results['best_steady_state']),
        'mean_steady_state_db': float(np.mean(valid_ss)) if len(valid_ss) > 0 else None,
        'std_steady_state_db': float(np.std(valid_ss)) if len(valid_ss) > 0 else None,
    }
    if len(valid_deriv) > 0:
      summary_dict['max_steady_state_derivative'] = float(np.max(valid_deriv))
      summary_dict['mean_steady_state_derivative'] = float(np.mean(valid_deriv))
      summary_dict['std_steady_state_derivative'] = float(np.std(valid_deriv))

    summary_file = folder / "grid_search_summary.yml"
    with open(summary_file, 'w') as f:
      yaml.dump(summary_dict, f, sort_keys=False, default_flow_style=False)

  if save_figures:
    fig_dir = folder / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metric histograms
    fig_hist = plot_grid_search_histograms(grid_results)
    hist_path = fig_dir / "metric_histograms.pdf"
    fig_hist.savefig(hist_path, format='pdf', bbox_inches='tight')
    plt.close(fig_hist)

    # 2. Pairwise heatmaps (convergence)
    fig_conv = plot_grid_search_pairwise_heatmaps(grid_results, metric_type='convergence')
    conv_path = fig_dir / "pairwise_heatmaps_convergence.pdf"
    fig_conv.savefig(conv_path, format='pdf', bbox_inches='tight')
    plt.close(fig_conv)

    # 3. Pairwise heatmaps (steady-state)
    fig_ss = plot_grid_search_pairwise_heatmaps(grid_results, metric_type='steady_state')
    ss_path = fig_dir / "pairwise_heatmaps_steady_state.pdf"
    fig_ss.savefig(ss_path, format='pdf', bbox_inches='tight')
    plt.close(fig_ss)

    # 4. Learning curves comparison
    perf_p = cfg['algorithm_performance_parameters']
    noise_val = perf_p.get(cfg['noise_param_name'])
    v_0 = perf_p.get('v_tilde_0', 1.0)

    fig_lc = plot_grid_search_learning_curves(
        grid_results=grid_results,
        environment=cfg['environment'],
        environment_parameters=cfg['environment_parameters'],
        h0=cfg['h0'],
        baseline_algorithm=cfg['baseline_algorithm'],
        baseline_parameter_class=cfg['baseline_parameter_class'],
        noise_param_name=cfg['noise_param_name'],
        noise_param_val=noise_val,
        v_tilde_0=v_0,
        N=cfg['N'],
        NR=cfg['NR'],
        num_workers=cfg['num_workers'],
        num_chunks=cfg['num_chunks'],
        seed=cfg['seed'],
    )
    lc_path = fig_dir / "learning_curves_comparison.pdf"
    fig_lc.savefig(lc_path, format='pdf', bbox_inches='tight')
    plt.close(fig_lc)

  # Explicit memory cleanup
  plt.close('all')
  gc.collect()

  return grid_results


# ==============================================================================
# Grid Search Cross-Experiment Summary & Distribution Analysis
# ==============================================================================

def collect_all_optimal_parameters(
    experiment_dirs: list[str | Path],
) -> list[dict[str, Any]]:
  """Collect optimal parameters and summary metrics from a list of experiment directories.

  Parameters
  ----------
  experiment_dirs : list of str or Path
    List of experiment leaf directories.

  Returns
  -------
  list of dict
    A list of dictionaries containing metadata, optimal parameters, and summary metrics.
  """
  records: list[dict[str, Any]] = []

  for d in experiment_dirs:
    p = Path(d)
    opt_file = p / "optimal_parameters.yml"
    summary_file = p / "grid_search_summary.yml"

    if not opt_file.is_file():
      continue

    with open(opt_file, 'r') as f:
      opt_data = yaml.safe_load(f) or {}

    summary_data = {}
    if summary_file.is_file():
      with open(summary_file, 'r') as f:
        summary_data = yaml.safe_load(f) or {}

    # Extract directory hierarchy metadata:
    # .../grid_searches/<algorithm>/<environment>/<target_msd>/<input_characteristic>
    parts = p.parts
    if 'grid_searches' in parts:
      idx = parts.index('grid_searches')
      subparts = parts[idx + 1:]
      alg = subparts[0] if len(subparts) > 0 else 'unknown'
      env = subparts[1] if len(subparts) > 1 else 'unknown'
      target_msd = subparts[2] if len(subparts) > 2 else 'unknown'
      input_char = subparts[3] if len(subparts) > 3 else 'unknown'
    else:
      alg = env = target_msd = input_char = 'unknown'

    rec = {
        'directory': str(p),
        'algorithm': alg,
        'environment': env,
        'target_msd': target_msd,
        'input_characteristic': input_char,
        'mu_a': float(opt_data['mu_a']) if 'mu_a' in opt_data else np.nan,
        'epsilon_1': float(opt_data['epsilon_1']) if 'epsilon_1' in opt_data else np.nan,
        'a_plus': float(opt_data['a_plus']) if 'a_plus' in opt_data else np.nan,
        'l': float(opt_data['l']) if 'l' in opt_data else np.nan,
        'optimal_convergence_metric': float(summary_data.get('optimal_convergence_metric', np.nan)),
        'optimal_steady_state_db': float(summary_data.get('optimal_steady_state_db', np.nan)),
        'computation_time_seconds': float(summary_data.get('computation_time_seconds', np.nan)),
    }
    records.append(rec)

  return records


def plot_optimal_parameters_histograms(
    records: list[dict[str, Any]],
    figsize: tuple[float, float] = (12, 9),
    title: str = "Histograms of Optimal Hyperparameters Across Grid Searches",
) -> plt.Figure:
  """Plot 2x2 categorical histograms of optimal hyperparameters across grid search experiments.

  Parameters
  ----------
  records : list of dict
    List of dictionaries returned by collect_all_optimal_parameters.
  figsize : tuple of float, optional
    Figure dimensions (width, height).
  title : str, optional
    Overall figure super-title.

  Returns
  -------
  plt.Figure
    The created matplotlib figure.
  """
  n_total = len(records)
  if n_total == 0:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, "No experiment records available.", ha='center', va='center')
    return fig

  fig, axes = plt.subplots(2, 2, figsize=figsize)

  configs = [
      {
          'ax': axes[0, 0],
          'key': 'mu_a',
          'name': r'Step Size $\mu_a^*$',
          'candidates': [1.0, 2.5, 5.0, 7.5, 10.0],
          'labels': ['1.0', '2.5', '5.0', '7.5', '10.0'],
          'color': 'steelblue',
      },
      {
          'ax': axes[0, 1],
          'key': 'epsilon_1',
          'name': r'Fast Filter Regularization $\epsilon_1^*$',
          'candidates': [1.0e-06, 1.0e-05, 0.0001, 0.001],
          'labels': [r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'],
          'color': 'seagreen',
      },
      {
          'ax': axes[1, 0],
          'key': 'a_plus',
          'name': r'Asymmetry Parameter $a_+^*$',
          'candidates': [2.0, 4.0, 8.0, 16.0],
          'labels': ['2.0', '4.0', '8.0', '16.0'],
          'color': 'darkorange',
      },
      {
          'ax': axes[1, 1],
          'key': 'l',
          'name': r'Leakage Factor $l^*$',
          'candidates': [0.9375, 0.99375, 0.999375, 0.9999375],
          'labels': ['0.9375\n(1-1/L)', '0.99375\n(1-0.1/L)', '0.999375\n(1-0.01/L)', '0.9999375\n(1-0.001/L)'],
          'color': 'purple',
      },
  ]

  for cfg in configs:
    ax = cfg['ax']
    key = cfg['key']
    cands = cfg['candidates']
    x_indices = np.arange(len(cands))
    vals = [r[key] for r in records if not np.isnan(r[key])]

    counts = [sum(np.isclose(v, c, rtol=1e-5, atol=1e-8) for v in vals) for c in cands]

    bars = ax.bar(x_indices, counts, color=cfg['color'], edgecolor='black', alpha=0.75, width=0.55)

    # Annotate counts and percentages on top of each bar
    for bar, count in zip(bars, counts):
      pct = (count / n_total * 100.0) if n_total > 0 else 0.0
      height = bar.get_height()
      if height > 0:
        ax.annotate(
            f"{count}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(cfg['labels'], fontsize=9)
    ax.set_ylabel("Frequency (Count)", fontweight='bold')
    ax.set_title(cfg['name'], fontsize=11, fontweight='bold')
    ax.set_ylim([0, max(counts) * 1.25 + 1 if counts else 5])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

  fig.suptitle(f"{title} (N = {n_total} Experiments)", fontsize=13, fontweight='bold')
  plt.tight_layout()
  return fig


def plot_optimal_parameters_by_group_histograms(
    records: list[dict[str, Any]],
    group_key: str = 'algorithm',
    figsize: tuple[float, float] = (14, 10),
    title: str | None = None,
) -> plt.Figure:
  """Plot 2x2 grouped histograms of optimal hyperparameters broken down by a category.

  Parameters
  ----------
  records : list of dict
    List of dictionaries returned by collect_all_optimal_parameters.
  group_key : str, optional
    Key in records to group by ('algorithm', 'input_characteristic', 'environment', 'target_msd').
  figsize : tuple of float, optional
    Figure dimensions (width, height).
  title : str, optional
    Figure super-title.

  Returns
  -------
  plt.Figure
    The created matplotlib figure.
  """
  n_total = len(records)
  if n_total == 0:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, "No experiment records available.", ha='center', va='center')
    return fig

  groups = sorted(list(set(r[group_key] for r in records)))
  n_groups = len(groups)
  palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
  group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

  fig, axes = plt.subplots(2, 2, figsize=figsize)

  configs = [
      {
          'ax': axes[0, 0],
          'key': 'mu_a',
          'name': r'Step Size $\mu_a^*$',
          'candidates': [1.0, 2.5, 5.0, 7.5, 10.0],
          'labels': ['1.0', '2.5', '5.0', '7.5', '10.0'],
      },
      {
          'ax': axes[0, 1],
          'key': 'epsilon_1',
          'name': r'Fast Filter Regularization $\epsilon_1^*$',
          'candidates': [1.0e-06, 1.0e-05, 0.0001, 0.001],
          'labels': [r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'],
      },
      {
          'ax': axes[1, 0],
          'key': 'a_plus',
          'name': r'Asymmetry Parameter $a_+^*$',
          'candidates': [2.0, 4.0, 8.0, 16.0],
          'labels': ['2.0', '4.0', '8.0', '16.0'],
      },
      {
          'ax': axes[1, 1],
          'key': 'l',
          'name': r'Leakage Factor $l^*$',
          'candidates': [0.9375, 0.99375, 0.999375, 0.9999375],
          'labels': ['0.9375\n(1-1/L)', '0.99375\n(1-0.1/L)', '0.999375\n(1-0.01/L)', '0.9999375\n(1-0.001/L)'],
      },
  ]

  bar_width = 0.8 / n_groups

  for cfg in configs:
    ax = cfg['ax']
    key = cfg['key']
    cands = cfg['candidates']
    x_indices = np.arange(len(cands))
    max_count = 0

    for g_idx, g in enumerate(groups):
      g_records = [r for r in records if r[group_key] == g]
      g_vals = [r[key] for r in g_records if not np.isnan(r[key])]
      counts = [sum(np.isclose(v, c, rtol=1e-5, atol=1e-8) for v in g_vals) for c in cands]
      max_count = max(max_count, max(counts) if counts else 0)

      offset = (g_idx - (n_groups - 1) / 2) * bar_width
      bars = ax.bar(
          x_indices + offset,
          counts,
          width=bar_width,
          color=group_colors[g],
          edgecolor='black',
          alpha=0.8,
          label=f"{g} (n={len(g_records)})",
      )

      for bar, count in zip(bars, counts):
        if count > 0:
          ax.annotate(
              f"{count}",
              xy=(bar.get_x() + bar.get_width() / 2, count),
              xytext=(0, 2),
              textcoords="offset points",
              ha='center',
              va='bottom',
              fontsize=8,
          )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(cfg['labels'], fontsize=9)
    ax.set_ylabel("Frequency (Count)", fontweight='bold')
    ax.set_title(cfg['name'], fontsize=11, fontweight='bold')
    ax.set_ylim([0, max_count * 1.3 + 1])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='upper right')

  if title is None:
    title = f"Optimal Hyperparameters Grouped by {group_key.replace('_', ' ').title()}"
  fig.suptitle(f"{title} (N = {n_total} Experiments)", fontsize=13, fontweight='bold')
  plt.tight_layout()
  return fig


def plot_optimal_metrics_histograms(
    records: list[dict[str, Any]],
    figsize: tuple[float, float] = (12, 4.5),
) -> plt.Figure:
  """Plot distributions of optimal convergence metric and optimal steady-state error.

  Parameters
  ----------
  records : list of dict
    List of dictionaries returned by collect_all_optimal_parameters.
  figsize : tuple of float, optional
    Figure dimensions (width, height).

  Returns
  -------
  plt.Figure
    The created matplotlib figure.
  """
  n_total = len(records)
  conv_vals = [r['optimal_convergence_metric'] for r in records if not np.isnan(r['optimal_convergence_metric'])]
  ss_vals = [r['optimal_steady_state_db'] for r in records if not np.isnan(r['optimal_steady_state_db'])]

  fig, axes = plt.subplots(1, 2, figsize=figsize)

  # 1. Convergence metric
  ax0 = axes[0]
  if conv_vals:
    ax0.hist(conv_vals, bins=12, color='steelblue', edgecolor='black', alpha=0.75)
    mean_c = np.mean(conv_vals)
    median_c = np.median(conv_vals)
    ax0.axvline(mean_c, color='crimson', ls='--', lw=1.8, label=f"Mean: {mean_c:.1f}")
    ax0.axvline(median_c, color='orange', ls=':', lw=1.8, label=f"Median: {median_c:.1f}")
  ax0.set_xlabel("Optimal Convergence Metric (Accumulated Target Error)", fontweight='bold')
  ax0.set_ylabel("Frequency (Count)", fontweight='bold')
  ax0.set_title("Optimal Convergence Metric Distribution", fontsize=11, fontweight='bold')
  ax0.grid(True, alpha=0.3, linestyle='--')
  ax0.legend(fontsize=9)

  # 2. Steady-state error
  ax1 = axes[1]
  if ss_vals:
    ax1.hist(ss_vals, bins=12, color='seagreen', edgecolor='black', alpha=0.75)
    mean_ss = np.mean(ss_vals)
    median_ss = np.median(ss_vals)
    ax1.axvline(mean_ss, color='crimson', ls='--', lw=1.8, label=f"Mean: {mean_ss:.2f} dB")
    ax1.axvline(median_ss, color='orange', ls=':', lw=1.8, label=f"Median: {median_ss:.2f} dB")
  ax1.set_xlabel("Optimal Steady-State MSD [dB]", fontweight='bold')
  ax1.set_ylabel("Frequency (Count)", fontweight='bold')
  ax1.set_title("Optimal Steady-State MSD Distribution", fontsize=11, fontweight='bold')
  ax1.grid(True, alpha=0.3, linestyle='--')
  ax1.legend(fontsize=9)

  fig.suptitle(f"Optimal Outcome Metrics Distribution (N = {n_total} Experiments)", fontsize=12, fontweight='bold')
  plt.tight_layout()
  return fig


