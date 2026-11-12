#!/usr/bin/env python3
"""CLI runner for hyperparameter grid search on adaptive filters."""

import argparse
import sys
import time
from pathlib import Path

import filters_exp as exp


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run hyperparameter grid search for adaptive filter combinations."
  )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "--algorithm",
      type=str,
      help="Path to an experiment directory containing 'environment_config.yml' and 'algorithm_config.yml'.",
  )
  group.add_argument(
      "--all",
      action="store_true",
      help="Run all grid search experiments found inside the 'experiments' folder.",
  )
  return parser.parse_args()


def run_single_experiment(experiment_dir: Path) -> None:
  print(f"\n{'=' * 70}")
  print(f"Running grid search for experiment: {experiment_dir.resolve()}")
  print(f"{'=' * 70}")

  t_start = time.time()
  results = exp.run_grid_search_experiment(experiment_dir, save_figures=True)
  t_elapsed = time.time() - t_start

  print(f"\nExperiment completed in {t_elapsed:.2f} seconds.")
  print("\nOptimal Hyperparameters Found:")
  for k, v in results['best_parameters'].items():
    print(f"  {k}*: {v}")
  print(f"Best Convergence Metric Value: {results['best_metric_value']:.2f}")
  print(f"Steady-State MSD at Optimal: {results['best_steady_state']:.2f} dB")

  figures_dir = experiment_dir / "figures"
  print(f"\nFigures successfully saved to: {figures_dir.resolve()}")
  for fig_file in figures_dir.glob("*.pdf"):
    print(f"  - {fig_file.name}")


def main() -> None:
  args = parse_args()

  if args.algorithm:
    exp_path = Path(args.algorithm)
    if not exp_path.is_dir():
      raise FileNotFoundError(f"Specified experiment directory does not exist: {exp_path.resolve()}")
    run_single_experiment(exp_path)

  elif args.all:
    experiments_root = Path("experiments")
    if not experiments_root.is_dir():
      raise FileNotFoundError(f"Experiments root folder does not exist: {experiments_root.resolve()}")

    # Find all immediate subdirectories in experiments/
    subdirs = sorted([d for d in experiments_root.iterdir() if d.is_dir() and d.name != "figures"])
    if not subdirs:
      raise FileNotFoundError(f"No experiment directories found in {experiments_root.resolve()}")

    print(f"Found {len(subdirs)} experiment directories in {experiments_root}:")
    for d in subdirs:
      print(f"  - {d}")

    for exp_dir in subdirs:
      run_single_experiment(exp_dir)


if __name__ == "__main__":
  main()
