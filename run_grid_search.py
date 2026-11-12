#!/usr/bin/env python3
"""CLI runner for hyperparameter grid search on adaptive filters."""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import filters_exp as exp


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
          "Run hyperparameter grid search for adaptive filter combinations.\n\n"
          "Examples:\n"
          "  python run_grid_search.py --algorithm experiments/grid_searches/skf_combination/white_input\n"
          "  python run_grid_search.py --all\n"
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "--algorithm",
      type=str,
      metavar="<experiments/grid_searches/...>",
      help="Path to an experiment directory containing 'environment_config.yml' and 'algorithm_config.yml'.",
  )
  group.add_argument(
      "--all",
      action="store_true",
      help="Scan all leaf subfolders of 'experiments/grid_searches' and run all experiments.",
  )
  return parser.parse_args()


def find_leaf_experiments(root: str | Path) -> list[Path]:
  """Find all leaf experiment folders in the tree under root.

  A directory is a leaf experiment folder if it has no subdirectories,
  or if its only subdirectory is 'figures/'. Intermediate parent directories
  and 'figures/' folders are ignored as non-experiment nodes.
  """
  root = Path(root)
  if not root.is_dir():
    raise FileNotFoundError(f"Root directory does not exist: {root.resolve()}")

  leaf_experiments: list[Path] = []

  for current_dir_str, subdirs, _ in os.walk(str(root)):
    current_dir = Path(current_dir_str)
    # Filter out 'figures' from subdirs so we don't traverse into figures
    if "figures" in subdirs:
      subdirs.remove("figures")

    # If current_dir is root itself, it cannot be a leaf experiment
    if current_dir == root:
      continue

    # If current_dir has no non-figures child directories, it is a leaf experiment
    if len(subdirs) == 0:
      leaf_experiments.append(current_dir)

  leaf_experiments.sort()
  return leaf_experiments


def validate_all_experiments(experiment_dirs: list[Path]) -> None:
  """Validate that all leaf experiment folders contain valid config files before running."""
  if not experiment_dirs:
    raise FileNotFoundError("No leaf experiment directories found.")

  print(f"Validating configuration files for {len(experiment_dirs)} experiment(s)...")
  for exp_dir in experiment_dirs:
    # load_experiment_configs raises FileNotFoundError or TypeError/ValueError if invalid
    exp.load_experiment_configs(exp_dir)
  print("✓ All configuration files successfully validated.\n")


def run_single_experiment(experiment_dir: Path) -> None:
  """Run grid search on a single experiment folder, print combinations, and clean memory."""
  print(f"\n{'=' * 70}")
  print(f"Running grid search for experiment: {experiment_dir.resolve()}")
  print(f"{'=' * 70}")

  # 1. Load config and print grid search dimensions and total combinations
  cfg = exp.load_experiment_configs(experiment_dir)
  conv_grid = cfg['algorithm_convergence_parameters']
  combos = int(np.prod([len(v) for v in conv_grid.values()]))

  print("Grid Search Dimensions:")
  for k, v in conv_grid.items():
    print(f"  {k}: {len(v)} values ({v})")
  print(f"Total grid combinations to evaluate: {combos}\n")

  # 2. Execute grid search experiment
  results = exp.run_grid_search_experiment(
      experiment_dir,
      save_figures=True,
      save_summary_yaml=True,
  )

  t_elapsed = results.get('computation_time_seconds', 0.0)
  print(f"\nExperiment completed in {t_elapsed:.2f} seconds.")
  print("\nOptimal Hyperparameters Found:")
  for k, v in results['best_parameters'].items():
    print(f"  {k}*: {v}")
  print(f"Best Convergence Metric Value: {results['best_metric_value']:.2f}")
  print(f"Steady-State MSD at Optimal: {results['best_steady_state']:.2f} dB")

  figures_dir = experiment_dir / "figures"
  print(f"\nFigures successfully saved to: {figures_dir.resolve()}")
  for fig_file in sorted(figures_dir.glob("*.pdf")):
    print(f"  - {fig_file.name}")

  print("\nOutput YAML files saved to:")
  print(f"  - {(experiment_dir / 'optimal_parameters.yml').resolve()}")
  print(f"  - {(experiment_dir / 'grid_search_summary.yml').resolve()}")

  # 3. Explicit memory cleanup
  del results
  del cfg
  plt.close('all')
  gc.collect()


def main() -> None:
  args = parse_args()

  if args.algorithm:
    exp_path = Path(args.algorithm)
    if not exp_path.is_dir():
      raise FileNotFoundError(f"Specified experiment directory does not exist: {exp_path.resolve()}")
    run_single_experiment(exp_path)

  elif args.all:
    grid_searches_root = Path("experiments") / "grid_searches"
    if not grid_searches_root.is_dir():
      raise FileNotFoundError(f"Grid searches root folder does not exist: {grid_searches_root.resolve()}")

    # Find and validate all leaf experiments in the tree
    leaf_experiments = find_leaf_experiments(grid_searches_root)
    if not leaf_experiments:
      raise FileNotFoundError(f"No leaf experiment directories found in {grid_searches_root.resolve()}")

    print(f"Found {len(leaf_experiments)} leaf experiment directories in {grid_searches_root}:")
    for d in leaf_experiments:
      print(f"  - {d}")
    print()

    # Pre-validate all configuration files; aborts immediately if any file is invalid
    validate_all_experiments(leaf_experiments)

    # Run all experiments
    t_all_start = time.time()
    for exp_dir in leaf_experiments:
      run_single_experiment(exp_dir)
    t_all_elapsed = time.time() - t_all_start

    print(f"\n{'=' * 70}")
    print(f"All {len(leaf_experiments)} experiments completed successfully in {t_all_elapsed:.2f} seconds.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
  main()
