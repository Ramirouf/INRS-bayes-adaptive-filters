#!/usr/bin/env python3
"""CLI runner for performance comparisons across adaptive filtering algorithms.

Compares the performance of:
  - sKF_algorithm
  - sKF_L_algorithm
  - GISV_sKF_algorithm
  - GISV_sKF_L_algorithm
  - cvx_comb_sKF
  - cvx_comb_sKF_L
across standardized noise environments, floor targets, and input characteristics.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

import filters
import filters_exp as exp
import metrics
import monte_carlo as mc


class CustomArgumentParser(argparse.ArgumentParser):
  """Custom argument parser that prints full usage and a clear error message on invalid commands."""

  def error(self, message: str) -> None:
    sys.stderr.write(f"\n{'!' * 70}\n")
    sys.stderr.write(f"ERROR: Invalid command or argument: {message}\n")
    sys.stderr.write(f"{'!' * 70}\n\n")
    self.print_help(sys.stderr)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
  parser = CustomArgumentParser(
      description=(
          "Run performance comparison simulations for adaptive filters:\n"
          "  - sKF_algorithm\n"
          "  - sKF_L_algorithm\n"
          "  - GISV_sKF_algorithm\n"
          "  - GISV_sKF_L_algorithm\n"
          "  - cvx_comb_sKF\n"
          "  - cvx_comb_sKF_L\n\n"
          "Examples:\n"
          "  python run_comparison.py --config experiments/comparisons/gisv_vs_combinations/gaussian/-30dB/white_input\n"
          "  python run_comparison.py --all\n"
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "--config",
      type=str,
      metavar="<leaf_directory>",
      help="Path to a comparison leaf directory containing 'environment_config.yml' and algorithm configuration files.",
  )
  group.add_argument(
      "--all",
      action="store_true",
      help="Scan all leaf subfolders of 'experiments/comparisons/gisv_vs_combinations' and run all comparisons.",
  )
  return parser.parse_args()


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


def _validate_numeric_array(val: Any, param_name: str) -> np.ndarray:
  """Validate that a parameter is a list/sequence of numbers and convert to float64 ndarray."""
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


def extract_experiment_axes(leaf_dir: str | Path) -> tuple[str, str, str]:
  """Extract (environment, target_msd, input_characteristic) from a leaf directory path."""
  leaf = Path(leaf_dir).resolve()
  parts = leaf.parts
  if len(parts) < 3:
    raise ValueError(f"Leaf directory path '{leaf_dir}' is too short to determine experiment axes.")
  env, target, inp = parts[-3], parts[-2], parts[-1]
  valid_envs = {"gaussian", "laplacian", "heavy_tail"}
  valid_targets = {"-30dB", "-40dB", "-50dB"}
  valid_inputs = {"white_input", "correlated_input"}
  if env not in valid_envs or target not in valid_targets or inp not in valid_inputs:
    raise ValueError(
        f"Directory '{leaf_dir}' does not match expected structure: "
        f"<environment>/{valid_envs}/<target_msd>/{valid_targets}/<input_characteristic>/{valid_inputs}. "
        f"Parsed: env='{env}', target='{target}', input='{inp}'."
    )
  return env, target, inp


def resolve_leaf_dir(
    leaf_dir: str | Path,
    comparisons_root: str | Path = "experiments/comparisons/gisv_vs_combinations",
) -> Path:
  """Resolve a user-provided leaf directory to an absolute Path under comparisons_root."""
  path = Path(leaf_dir)
  if path.is_dir():
    return path.resolve()
  comp_root = Path(comparisons_root).resolve()
  candidate = comp_root / leaf_dir
  if candidate.is_dir() or len(Path(leaf_dir).parts) == 3:
    return candidate.resolve()
  try:
    env, target, inp = extract_experiment_axes(path)
    return (comp_root / env / target / inp).resolve()
  except Exception:
    pass
  return path.resolve()


def update_leaf_from_grid_searches(
    leaf_dir: str | Path,
    grid_searches_root: str | Path = "experiments/grid_searches",
) -> Path:
  """Load optimal parameters and appropriate sample size N from grid searches and write comparison configs."""
  leaf_dir = Path(leaf_dir).resolve()
  grid_root = Path(grid_searches_root).resolve()

  env, target, inp = extract_experiment_axes(leaf_dir)

  skf_comb_dir = grid_root / "skf_combination" / env / target / inp
  skf_l_comb_dir = grid_root / "skf_l_combination" / env / target / inp

  skf_comb_opt_file = skf_comb_dir / "optimal_parameters.yml"
  skf_l_comb_opt_file = skf_l_comb_dir / "optimal_parameters.yml"
  skf_env_file = skf_comb_dir / "environment_config.yml"
  skf_l_env_file = skf_l_comb_dir / "environment_config.yml"

  for fpath in (skf_comb_opt_file, skf_l_comb_opt_file, skf_env_file, skf_l_env_file):
    if not fpath.is_file():
      raise FileNotFoundError(f"Required grid search file not found: {fpath}")

  leaf_dir.mkdir(parents=True, exist_ok=True)

  with open(skf_env_file, "r") as f:
    skf_env_cfg = yaml.safe_load(f) or {}
  with open(skf_l_env_file, "r") as f:
    skf_l_env_cfg = yaml.safe_load(f) or {}

  # Load appropriate number of samples N across grid search experiments
  skf_N = int(_validate_numeric(skf_env_cfg.get("N", 20000), "skf_combination:N"))
  skf_l_N = int(_validate_numeric(skf_l_env_cfg.get("N", 20000), "skf_l_combination:N"))
  N = max(skf_N, skf_l_N)

  # Choose environment config base (prefer the one with larger N for worker/chunk tuning)
  base_env_cfg = copy.deepcopy(skf_l_env_cfg if skf_l_N >= skf_N else skf_env_cfg)
  base_env_cfg["N"] = N

  # Memory safety for very large N simulations (e.g. N = 640000)
  if N >= 640000:
    base_env_cfg["num_workers"] = min(int(base_env_cfg.get("num_workers", 2)), 2)
    base_env_cfg["num_chunks"] = min(int(base_env_cfg.get("num_chunks", 2)), 4)

  with open(leaf_dir / "environment_config.yml", "w") as f:
    yaml.safe_dump(base_env_cfg, f, sort_keys=False, default_flow_style=False)

  # Read optimal parameters from skf_combination
  with open(skf_comb_opt_file, "r") as f:
    skf_comb_opt = yaml.safe_load(f) or {}

  # Read optimal parameters from skf_l_combination
  with open(skf_l_comb_opt_file, "r") as f:
    skf_l_comb_opt = yaml.safe_load(f) or {}

  slow_epsilon_skf = float(_validate_numeric(skf_comb_opt["epsilon_2"], "skf_combination:epsilon_2"))
  var_eta = float(_validate_numeric(skf_comb_opt.get("var_eta", 0.001), "var_eta"))
  v_tilde_0 = float(_validate_numeric(skf_comb_opt.get("v_tilde_0", 1.0), "v_tilde_0"))

  slow_epsilon_skf_l = float(_validate_numeric(skf_l_comb_opt["epsilon_2"], "skf_l_combination:epsilon_2"))
  b_eta = float(_validate_numeric(skf_l_comb_opt.get("b_eta", np.sqrt(1e-3 / 2)), "b_eta"))

  cvx_comb_skf_params = {
      "label": "cvx_comb_sKF",
      "epsilon_1": float(_validate_numeric(skf_comb_opt["epsilon_1"], "epsilon_1")),
      "epsilon_2": slow_epsilon_skf,
      "var_eta": var_eta,
      "v_tilde_0": v_tilde_0,
      "mu_a": float(_validate_numeric(skf_comb_opt["mu_a"], "mu_a")),
      "a_plus": float(_validate_numeric(skf_comb_opt["a_plus"], "a_plus")),
      "tau": float(_validate_numeric(skf_comb_opt["tau"], "tau")),
      "l": float(_validate_numeric(skf_comb_opt["l"], "l")),
      "reg_rho": float(_validate_numeric(skf_comb_opt["reg_rho"], "reg_rho")),
      "lambda_0": float(_validate_numeric(skf_comb_opt["lambda_0"], "lambda_0")),
  }

  cvx_comb_skf_l_params = {
      "label": "cvx_comb_sKF_L",
      "epsilon_1": float(_validate_numeric(skf_l_comb_opt["epsilon_1"], "epsilon_1")),
      "epsilon_2": slow_epsilon_skf_l,
      "b_eta": b_eta,
      "v_tilde_0": v_tilde_0,
      "mu_a": float(_validate_numeric(skf_l_comb_opt["mu_a"], "mu_a")),
      "a_plus": float(_validate_numeric(skf_l_comb_opt["a_plus"], "a_plus")),
      "tau": float(_validate_numeric(skf_l_comb_opt["tau"], "tau")),
      "l": float(_validate_numeric(skf_l_comb_opt["l"], "l")),
      "reg_rho": float(_validate_numeric(skf_l_comb_opt["reg_rho"], "reg_rho")),
      "lambda_0": float(_validate_numeric(skf_l_comb_opt["lambda_0"], "lambda_0")),
  }

  algo_configs = {
      "skf_algorithm.yml": {
          "algorithm": "sKF_algorithm",
          "parameter_class": "sKF_parameters",
          "parameters": {
              "label": "sKF",
              "epsilon": slow_epsilon_skf,
              "var_eta": var_eta,
              "v_tilde_0": v_tilde_0,
          },
      },
      "skf_l_algorithm.yml": {
          "algorithm": "sKF_L_algorithm",
          "parameter_class": "sKF_L_parameters",
          "parameters": {
              "label": "sKF-L",
              "epsilon": slow_epsilon_skf_l,
              "b_eta": b_eta,
              "v_tilde_0": v_tilde_0,
          },
      },
      "gisv_skf_algorithm.yml": {
          "algorithm": "GISV_sKF_algorithm",
          "parameter_class": "sKF_parameters",
          "parameters": {
              "label": "GISV-sKF",
              "epsilon": 0.0,
              "var_eta": var_eta,
              "v_tilde_0": v_tilde_0,
          },
      },
      "gisv_skf_l_algorithm.yml": {
          "algorithm": "GISV_sKF_L_algorithm",
          "parameter_class": "GISV_sKF_L_parameters",
          "parameters": {
              "label": "GISV-sKF-L",
              "b_eta": b_eta,
              "v_tilde_0": v_tilde_0,
              "epsilon_min": slow_epsilon_skf_l,
          },
      },
      "cvx_comb_skf.yml": {
          "algorithm": "cvx_comb_sKF",
          "parameter_class": "cvx_comb_sKF_parameters",
          "parameters": cvx_comb_skf_params,
      },
      "cvx_comb_skf_l.yml": {
          "algorithm": "cvx_comb_sKF_L",
          "parameter_class": "cvx_comb_sKF_L_parameters",
          "parameters": cvx_comb_skf_l_params,
      },
  }

  for fname, cfg in algo_configs.items():
    with open(leaf_dir / fname, "w") as f:
      yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

  print(f"Loaded optimal parameters and N = {N} from grid searches for: {env}/{target}/{inp}")
  return leaf_dir


def import_optimal_parameters_to_configs(
    grid_searches_root: str | Path = "experiments/grid_searches",
    comparisons_root: str | Path = "experiments/comparisons/gisv_vs_combinations",
) -> list[Path]:
  """Import optimal parameters from grid searches and generate standardized YAML configs per algorithm.

  For each environment/target_msd/input_characteristic leaf:
    1. Loads N from skf_combination and skf_l_combination and selects max(skf_N, skf_l_N).
    2. Reads optimal_parameters.yml from skf_combination and skf_l_combination.
    3. Generates 6 algorithm YAML files:
       - skf_algorithm.yml (adjusted with slow filter epsilon from skf_combination)
       - skf_l_algorithm.yml (adjusted with slow filter epsilon from skf_l_combination)
       - gisv_skf_algorithm.yml
       - gisv_skf_l_algorithm.yml (epsilon_min set to slow filter epsilon from skf_l_combination)
       - cvx_comb_skf.yml (with optimal grid search parameters from skf_combination)
       - cvx_comb_skf_l.yml (with optimal grid search parameters from skf_l_combination)

  Returns
  -------
  list[Path]
    List of leaf directory paths created/updated.
  """
  grid_root = Path(grid_searches_root).resolve()
  comp_root = Path(comparisons_root).resolve()

  environments = ["gaussian", "laplacian", "heavy_tail"]
  target_msds = ["-30dB", "-40dB", "-50dB"]
  inputs = ["white_input", "correlated_input"]

  created_leaves: list[Path] = []

  for env in environments:
    for target in target_msds:
      for inp in inputs:
        leaf_dir = comp_root / env / target / inp
        try:
          update_leaf_from_grid_searches(leaf_dir, grid_root)
          created_leaves.append(leaf_dir)
        except FileNotFoundError:
          continue

  return created_leaves


def load_leaf_configurations(leaf_dir: str | Path) -> dict[str, Any]:
  """Load and strictly validate environment and all per-algorithm YAML configs for a leaf experiment."""
  leaf = Path(leaf_dir)
  env_file = leaf / "environment_config.yml"
  if not env_file.is_file():
    raise FileNotFoundError(f"Environment configuration file not found in: {leaf.resolve()}")

  with open(env_file, "r") as f:
    env_cfg = yaml.safe_load(f) or {}

  # Validate simulation / Monte Carlo parameters
  N = int(_validate_numeric(env_cfg.get("N", 50000), "N"))
  NR = int(_validate_numeric(env_cfg.get("NR", 256), "NR"))
  num_workers = int(_validate_numeric(env_cfg.get("num_workers", 4), "num_workers"))
  num_chunks = int(_validate_numeric(env_cfg.get("num_chunks", 16), "num_chunks"))
  seed = int(_validate_numeric(env_cfg.get("seed", 42), "seed"))

  # Plant & system setup
  L = int(_validate_numeric(env_cfg["L"], "L"))
  plant_type = env_cfg.get("plant_type", "normalized_sinc")
  if plant_type == "normalized_sinc":
    ho = np.sinc(np.linspace(0, 1, L))
    ho = ho / np.linalg.norm(ho)
  elif plant_type == "custom":
    ho = _validate_numeric_array(env_cfg["ho"], "ho")
  else:
    raise ValueError(f"Unknown plant_type: {plant_type}")

  h0 = np.zeros(L, dtype=np.float64)
  AR = _validate_numeric_array(env_cfg["AR"], "AR")
  var_x = float(_validate_numeric(env_cfg["var_x"], "var_x"))

  # Behavior and environment parameters namedtuple
  behavior_name = env_cfg["behavior"]
  if not hasattr(filters, behavior_name):
    raise AttributeError(f"Environment behavior '{behavior_name}' not found in filters module.")
  environment = getattr(filters, behavior_name)

  env_param_cls_name = env_cfg["parameters_class"]
  if not hasattr(filters, env_param_cls_name):
    raise AttributeError(f"Environment parameters class '{env_param_cls_name}' not found in filters module.")
  env_param_cls = getattr(filters, env_param_cls_name)

  env_kwargs: dict[str, Any] = {}
  for field in env_param_cls._fields:
    if field == "ho":
      env_kwargs["ho"] = ho
    elif field == "AR":
      env_kwargs["AR"] = AR
    elif field == "var_x":
      env_kwargs["var_x"] = var_x
    elif field in env_cfg:
      env_kwargs[field] = float(_validate_numeric(env_cfg[field], field))
    else:
      raise ValueError(f"Missing required field '{field}' in {env_file}")
  env_parameters = env_param_cls(**env_kwargs)

  # Load all algorithm YAML configs in the leaf folder in standard order
  preferred_order = [
      "skf_algorithm.yml",
      "skf_l_algorithm.yml",
      "gisv_skf_algorithm.yml",
      "gisv_skf_l_algorithm.yml",
      "cvx_comb_skf.yml",
      "cvx_comb_skf_l.yml",
  ]
  existing_algo_files = {
      p.name: p
      for p in leaf.glob("*.yml")
      if p.name not in ("environment_config.yml", "metrics_summary.yml")
  }
  algo_files: list[Path] = []
  for name in preferred_order:
    if name in existing_algo_files:
      algo_files.append(existing_algo_files[name])
  # Add any remaining unlisted algorithm files
  for name, path in sorted(existing_algo_files.items()):
    if path not in algo_files:
      algo_files.append(path)
  if not algo_files:
    raise FileNotFoundError(f"No algorithm configuration files found in leaf directory: {leaf.resolve()}")

  algorithms = []
  algorithms_parameters = []
  algorithm_labels = []

  for a_file in algo_files:
    with open(a_file, "r") as f:
      a_cfg = yaml.safe_load(f) or {}

    alg_func_name = a_cfg.get("algorithm")
    if not hasattr(filters, alg_func_name):
      raise AttributeError(f"Algorithm function '{alg_func_name}' not found in filters module (from {a_file.name}).")
    alg_func = getattr(filters, alg_func_name)

    param_cls_name = a_cfg.get("parameter_class")
    if not hasattr(filters, param_cls_name):
      raise AttributeError(f"Parameter class '{param_cls_name}' not found in filters module (from {a_file.name}).")
    param_cls = getattr(filters, param_cls_name)

    raw_params = a_cfg.get("parameters", {})
    validated_params: dict[str, Any] = {}
    for k, v in raw_params.items():
      if k == "label":
        validated_params[k] = str(v)
      else:
        validated_params[k] = float(_validate_numeric(v, f"{a_file.name}:{k}"))

    # Instantiate namedtuple
    param_obj = param_cls(**validated_params)

    algorithms.append(alg_func)
    algorithms_parameters.append(param_obj)
    algorithm_labels.append(param_obj.label)

  return {
      "N": N,
      "NR": NR,
      "num_workers": num_workers,
      "num_chunks": num_chunks,
      "seed": seed,
      "L": L,
      "h0": h0,
      "AR": AR,
      "var_x": var_x,
      "behavior_name": behavior_name,
      "environment": environment,
      "environment_parameters": env_parameters,
      "algorithms": algorithms,
      "algorithms_parameters": algorithms_parameters,
      "algorithm_labels": algorithm_labels,
  }


def run_comparison_leaf(
    leaf_dir: str | Path,
    grid_searches_root: str | Path = "experiments/grid_searches",
) -> dict[str, Any]:
  """Execute the Monte Carlo comparison for a single leaf directory."""
  leaf = Path(leaf_dir).resolve()
  # Load optimal parameters and appropriate number of samples from grid searches
  update_leaf_from_grid_searches(leaf, grid_searches_root)

  print(f"\n{'=' * 70}")
  print(f"Running comparison for: {leaf}")
  print(f"{'=' * 70}")

  t_start = time.time()
  cfg = load_leaf_configurations(leaf)

  desired_metrics = {
      "MSD": metrics.squared_deviation,
      "EMSE": metrics.excess_squared_error,
  }

  print(f"Algorithms to evaluate ({len(cfg['algorithms'])}): {', '.join(cfg['algorithm_labels'])}")
  print(f"Simulations: N = {cfg['N']}, NR = {cfg['NR']}, workers = {cfg['num_workers']}, chunks = {cfg['num_chunks']}")

  measures, _ = mc.dask_MC_Simulations(
      N=cfg["N"],
      NR=cfg["NR"],
      environment_parameters=cfg["environment_parameters"],
      environment=cfg["environment"],
      algorithms=cfg["algorithms"],
      algorithms_parameters=cfg["algorithms_parameters"],
      desired_metrics=desired_metrics,
      h0=cfg["h0"],
      num_workers=cfg["num_workers"],
      num_chunks=cfg["num_chunks"],
      seed=cfg["seed"],
      scheduler="threads",
  )

  t_elapsed = time.time() - t_start
  print(f"Monte Carlo simulations completed in {t_elapsed:.2f} seconds.")

  # Compute accumulated target error and steady-state metrics
  metrics_summary: dict[str, Any] = {
      "leaf_directory": str(leaf),
      "computation_time_seconds": float(t_elapsed),
      "N": cfg["N"],
      "NR": cfg["NR"],
      "algorithms": {},
  }

  floor_estimate_samples = 100

  for label in cfg["algorithm_labels"]:
    msd_raw = np.asarray(measures[label]["MSD"], dtype=np.float64)
    emse_raw = np.asarray(measures[label]["EMSE"], dtype=np.float64)

    # Convert to dB scale
    msd_db = 10.0 * np.log10(np.maximum(msd_raw, 1e-15))
    emse_db = 10.0 * np.log10(np.maximum(emse_raw, 1e-15))

    # Calculate target values using minimum linear regression slope
    msd_target, _, _ = exp.find_minimum_linear_regression_slope(msd_db, n_window_samples=None)
    emse_target, _, _ = exp.find_minimum_linear_regression_slope(emse_db, n_window_samples=None)

    # Accumulated target error
    msd_ate = float(exp.db_accumulated_target_error(msd_db, target=msd_target, tolerance=0.2, normalize=True))
    emse_ate = float(exp.db_accumulated_target_error(emse_db, target=emse_target, tolerance=0.2, normalize=True))

    # Steady-state floor (mean of last floor_estimate_samples)
    msd_ss_db = float(np.mean(msd_db[-floor_estimate_samples:]))
    emse_ss_db = float(np.mean(emse_db[-floor_estimate_samples:]))

    # Steady-state derivative
    msd_ss_deriv = float(
        exp.mean_absolute_steady_state_derivative(msd_db, floor_estimate_samples=floor_estimate_samples)
    )
    emse_ss_deriv = float(
        exp.mean_absolute_steady_state_derivative(emse_db, floor_estimate_samples=floor_estimate_samples)
    )

    metrics_summary["algorithms"][label] = {
        "msd": {
            "accumulated_target_error": msd_ate,
            "target_db": msd_target,
            "steady_state_db": msd_ss_db,
            "steady_state_derivative": msd_ss_deriv,
        },
        "emse": {
            "accumulated_target_error": emse_ate,
            "target_db": emse_target,
            "steady_state_db": emse_ss_db,
            "steady_state_derivative": emse_ss_deriv,
        },
    }

  # Write metrics_summary.yml
  summary_file = leaf / "metrics_summary.yml"
  with open(summary_file, "w") as f:
    yaml.safe_dump(metrics_summary, f, sort_keys=False, default_flow_style=False)
  print(f"Metrics saved to: {summary_file}")

  # Generate and save learning curves
  fig_dir = leaf / "figures"
  fig_dir.mkdir(parents=True, exist_ok=True)

  # Color and style palette for clear distinction
  style_map = {
      "sKF": {"color": "#1f77b4", "linestyle": "--", "label": "sKF"},
      "sKF-L": {"color": "#17becf", "linestyle": "--", "label": "sKF-L"},
      "GISV-sKF": {"color": "#ff7f0e", "linestyle": ":", "label": "GISV-sKF"},
      "GISV-sKF-L": {"color": "#d62728", "linestyle": ":", "label": "GISV-sKF-L"},
      "cvx_comb_sKF": {"color": "#2ca02c", "linestyle": "-", "label": "cvx_comb_sKF"},
      "cvx_comb_sKF_L": {"color": "#9467bd", "linestyle": "-", "label": "cvx_comb_sKF_L"},
  }

  # Determine noise name and eigenvalue spread for plot titles
  behavior_name = cfg.get("behavior_name", "")
  if "general_gaussian" in behavior_name or "heavy_tail" in leaf.parts:
    noise_title = "General Gaussian Noise"
  elif "laplacian" in behavior_name or "laplacian" in leaf.parts:
    noise_title = "Laplacian Noise"
  elif "gaussian" in behavior_name or "gaussian" in leaf.parts:
    noise_title = "Gaussian Noise"
  else:
    noise_title = "Noise"

  Rxx = filters.autocorr_matrix_calc(cfg["AR"], 1.0, M=cfg["L"])
  Rxx = cfg["var_x"] * Rxx / Rxx[0, 0]
  eigenvalues = np.real(np.linalg.eigvals(Rxx))
  rxx_spread = float(np.max(eigenvalues) / np.min(eigenvalues))

  title_msd = f"MSD Learning Curves - {noise_title} - Spread = {rxx_spread:.2f}"
  title_emse = f"EMSE Learning Curves - {noise_title} - Spread = {rxx_spread:.2f}"

  iterations = np.arange(cfg["N"])

  # 1. MSD Figure
  fig_msd, ax_msd = plt.subplots(figsize=(10, 5))
  for label in cfg["algorithm_labels"]:
    msd_raw = np.asarray(measures[label]["MSD"], dtype=np.float64)
    msd_db = 10.0 * np.log10(np.maximum(msd_raw, 1e-15))
    msd_ate = metrics_summary["algorithms"][label]["msd"]["accumulated_target_error"]

    if label == "GISV-sKF":
      plot_label = label
    else:
      plot_label = f"{label} (ATE = {msd_ate:.2e})"

    st = style_map.get(label, {"color": None, "linestyle": "-", "label": label})
    ax_msd.plot(iterations, msd_db, color=st["color"], linestyle=st["linestyle"], label=plot_label, linewidth=1.5)

  ax_msd.set_xlabel("Iteration ($k$)")
  ax_msd.set_ylabel("MSD [dB]")
  ax_msd.set_title(title_msd)
  ax_msd.grid(True, linestyle="--", alpha=0.7)
  ax_msd.legend(loc="upper right", framealpha=0.9)
  fig_msd_path = fig_dir / "learning_curves_msd.pdf"
  fig_msd.savefig(fig_msd_path, format="pdf", bbox_inches="tight")
  plt.close(fig_msd)

  # 2. EMSE Figure
  fig_emse, ax_emse = plt.subplots(figsize=(10, 5))
  for label in cfg["algorithm_labels"]:
    emse_raw = np.asarray(measures[label]["EMSE"], dtype=np.float64)
    emse_db = 10.0 * np.log10(np.maximum(emse_raw, 1e-15))
    emse_ate = metrics_summary["algorithms"][label]["emse"]["accumulated_target_error"]

    if label == "GISV-sKF":
      plot_label = label
    else:
      plot_label = f"{label} (ATE = {emse_ate:.2e})"

    st = style_map.get(label, {"color": None, "linestyle": "-", "label": label})
    ax_emse.plot(iterations, emse_db, color=st["color"], linestyle=st["linestyle"], label=plot_label, linewidth=1.5)

  ax_emse.set_xlabel("Iteration ($k$)")
  ax_emse.set_ylabel("EMSE [dB]")
  ax_emse.set_title(title_emse)
  ax_emse.grid(True, linestyle="--", alpha=0.7)
  ax_emse.legend(loc="upper right", framealpha=0.9)
  fig_emse_path = fig_dir / "learning_curves_emse.pdf"
  fig_emse.savefig(fig_emse_path, format="pdf", bbox_inches="tight")
  plt.close(fig_emse)

  # Explicit memory cleanup
  plt.close("all")
  gc.collect()

  print(f"Figures saved to: {fig_dir}")
  return metrics_summary


def find_comparison_leaves(root: str | Path = "experiments/comparisons/gisv_vs_combinations") -> list[Path]:
  """Scan the comparison tree and find all leaf directories containing environment_config.yml."""
  root_path = Path(root).resolve()
  if not root_path.is_dir():
    return []

  leaves: list[Path] = []
  for current_dir_str, _, files in os.walk(str(root_path)):
    current_dir = Path(current_dir_str)
    if "environment_config.yml" in files:
      leaves.append(current_dir)

  return sorted(leaves)


def main() -> None:
  args = parse_args()

  comparisons_root = Path("experiments/comparisons/gisv_vs_combinations").resolve()
  grid_searches_root = Path("experiments/grid_searches").resolve()

  # Always ensure configs are populated from grid searches if comparisons directory is missing or empty
  if not comparisons_root.is_dir() or not any(comparisons_root.iterdir()):
    print("Initializing comparison configuration files from grid searches...")
    import_optimal_parameters_to_configs(grid_searches_root, comparisons_root)

  if args.config:
    leaf_dir = resolve_leaf_dir(args.config, comparisons_root)
    if not leaf_dir.is_dir():
      # If leaf_dir does not exist but is under comparisons_root, try populating configs
      import_optimal_parameters_to_configs(grid_searches_root, comparisons_root)

    if not leaf_dir.is_dir():
      sys.stderr.write(f"\n{'!' * 70}\n")
      sys.stderr.write(f"ERROR: Specified leaf directory does not exist: {leaf_dir.resolve()}\n")
      sys.stderr.write(f"{'!' * 70}\n\n")
      sys.exit(1)

    run_comparison_leaf(leaf_dir, grid_searches_root)

  elif args.all:
    # Ensure all 18 configs are generated and updated from grid searches
    import_optimal_parameters_to_configs(grid_searches_root, comparisons_root)
    leaves = find_comparison_leaves(comparisons_root)

    if not leaves:
      print("No comparison leaf directories found.")
      sys.exit(1)

    print(f"Found {len(leaves)} comparison leaf experiments.")
    total_start = time.time()

    for idx, leaf in enumerate(leaves, 1):
      print(f"\n>>> Running leaf [{idx}/{len(leaves)}]: {leaf.relative_to(comparisons_root)}")
      run_comparison_leaf(leaf, grid_searches_root)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"All {len(leaves)} comparisons finished in {total_time:.2f} seconds.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
  main()
