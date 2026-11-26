#!/usr/bin/env python3
"""Run grid search re-evaluations for all experiments whose target diverged from nominal steady state."""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

import filters_exp as exp
from run_grid_search import find_leaf_experiments


def scan_divergent_experiments(
    root: str | Path = "experiments/grid_searches",
    diff_tolerance_db: float = 3.0,
) -> list[dict]:
  """Scan leaf experiments and return those where |target - nominal| > diff_tolerance_db."""
  leaves = find_leaf_experiments(root)
  divergent = []

  for leaf_dir in leaves:
    parts = leaf_dir.parts
    # ('experiments', 'grid_searches', alg, noise, target_msd, input_char)
    alg, noise, target_msd, input_char = parts[2], parts[3], parts[4], parts[5]
    nominal = float(target_msd.replace("dB", ""))

    summary_file = leaf_dir / "grid_search_summary.yml"
    if not summary_file.is_file():
      continue

    with open(summary_file, "r") as f:
      summary = yaml.safe_load(f)

    env_file = leaf_dir / "environment_config.yml"
    with open(env_file, "r") as f:
      env = yaml.safe_load(f)

    target = summary.get("convergence_metric_target")
    diff = abs(target - nominal) if target is not None else 999.0
    N = env.get("N", 50000)
    combos = summary.get("total_combinations", 320)

    if diff > diff_tolerance_db:
      divergent.append({
          "path": str(leaf_dir),
          "alg": alg,
          "noise": noise,
          "target_msd": target_msd,
          "input": input_char,
          "nominal": nominal,
          "target": target,
          "diff": diff,
          "N": N,
          "combos": combos,
      })

  # Sort: shorter N first (white input), then alg, noise, nominal
  divergent.sort(key=lambda x: (x["N"], x["alg"], x["noise"], x["nominal"]))
  return divergent


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Scan and re-run grid searches with target divergence > tolerance (default 3.0 dB)."
  )
  parser.add_argument(
      "--tolerance",
      type=float,
      default=3.0,
      help="Difference tolerance in dB relative to nominal steady state (default: 3.0 dB).",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="List divergent experiments without executing simulations.",
  )
  parser.add_argument(
      "--root",
      type=str,
      default="experiments/grid_searches",
      help="Root directory of grid searches (default: experiments/grid_searches).",
  )
  args = parser.parse_args()

  divergent = scan_divergent_experiments(root=args.root, diff_tolerance_db=args.tolerance)

  print(f"\n{'=' * 75}")
  print(f"DIVERGENT EXPERIMENT SCAN (|target - nominal| > {args.tolerance:.1f} dB)")
  print(f"{'=' * 75}")
  print(f"Found {len(divergent)} divergent experiment(s):\n")

  for idx, item in enumerate(divergent, 1):
    print(
        f"{idx:2d}. [{item['alg'][:10]}|{item['noise'][:7]}|{item['target_msd']:>5}|{item['input'][:5]}] "
        f"target = {item['target']:6.2f} dB, nominal = {item['nominal']:5.1f} dB, "
        f"diff = {item['diff']:5.2f} dB (N = {item['N']}, combos = {item['combos']})\n"
        f"    Path: {item['path']}"
    )

  if args.dry_run:
    print(f"\nDry run complete. No experiments executed.")
    return

  total_start = time.time()
  successes = []
  failures = []

  print(f"\n{'=' * 75}")
  print(f"STARTING BATCH RE-EVALUATION OF {len(divergent)} DIVERGENT EXPERIMENTS")
  print(f"{'=' * 75}\n")

  for idx, item in enumerate(divergent, 1):
    exp_dir = item["path"]
    print(f"\n{'#' * 75}")
    print(f"[{idx}/{len(divergent)}] Running: {exp_dir}")
    print(f"{'#' * 75}\n")

    start_t = time.time()
    try:
      exp.run_grid_search_experiment(exp_dir)
      elapsed = time.time() - start_t
      print(f"\n[OK] Experiment {idx}/{len(divergent)} completed in {elapsed:.1f} seconds: {exp_dir}")
      successes.append((exp_dir, elapsed))
    except Exception as e:
      elapsed = time.time() - start_t
      print(f"\n[ERROR] Experiment {idx}/{len(divergent)} failed after {elapsed:.1f} seconds: {exp_dir}\nError: {e}")
      failures.append((exp_dir, elapsed, str(e)))

    plt.close("all")
    gc.collect()

  total_elapsed = time.time() - total_start
  print(f"\n{'=' * 75}")
  print(f"BATCH EXECUTION COMPLETE")
  print(f"{'=' * 75}")
  print(f"Total time elapsed: {total_elapsed:.1f} seconds ({total_elapsed / 60:.1f} minutes)")
  print(f"Successful: {len(successes)}/{len(divergent)}")
  print(f"Failed:     {len(failures)}/{len(divergent)}")

  if failures:
    print("\nFailed experiments:")
    for f_path, f_time, f_err in failures:
      print(f"  - {f_path} ({f_time:.1f}s): {f_err}")

  # Automatically regenerate cross-experiment summary plots
  summary_script = Path("grid_search_summary.py")
  if not summary_script.is_file():
    summary_script = Path("experiments/grid_search_summary/gs_summary.py")

  if summary_script.is_file():
    print(f"\nRegenerating cross-experiment summary figures via {summary_script.name}...")
    try:
      import subprocess
      subprocess.run([sys.executable, str(summary_script)], check=True)
      print("Summary figures updated successfully.")
    except Exception as err:
      print(f"Warning: Could not run {summary_script.name}: {err}")


if __name__ == "__main__":
  main()
