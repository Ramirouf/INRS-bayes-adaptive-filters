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

  # Run Monte Carlo simulations
  desired_metrics = {'performance': performance_metric}
  algorithms_list = [algorithm] * len(param_objects)

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
