#!/usr/bin/env python3
"""Cross-experiment summary script for adaptive filter comparisons.

Scans all 'metrics_summary.yml' files under 'experiments/comparisons/gisv_vs_combinations',
discovers all available metrics and algorithms, and generates multi-page PDF summaries
with value-count histogram grids grouped by:
  1. Overall (all simulations)
  2. Target steady-state MSD (-30dB, -40dB, -50dB)
  3. Environment noise (gaussian, laplacian, heavy_tail)
  4. Input correlation (white_input, correlated_input)

In each PDF:
  - Each page corresponds to a distinct metric.
  - Subplot grid: Rows = Algorithms, Columns = Group categories.
  - Each cell displays a value-count histogram of the simulations.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

# Ensure repository root and .venv site-packages are on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

venv_site = REPO_ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if venv_site.is_dir() and str(venv_site) not in sys.path:
  sys.path.append(str(venv_site))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml


# Standard algorithm ordering and colors matching run_comparison.py
PREFERRED_ALGORITHM_ORDER = [
    "sKF",
    "sKF-L",
    "GISV-sKF",
    "GISV-sKF-L",
    "cvx_comb_sKF",
    "cvx_comb_sKF_L",
]

ALGORITHM_COLORS: dict[str, str] = {
    "sKF": "#1f77b4",          # steel blue
    "sKF-L": "#17becf",        # cyan
    "GISV-sKF": "#ff7f0e",     # orange
    "GISV-sKF-L": "#d62728",   # red
    "cvx_comb_sKF": "#2ca02c", # green
    "cvx_comb_sKF_L": "#9467bd",  # purple
}

FALLBACK_PALETTE = [
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79",
    "#637939", "#8c6d31", "#843c39", "#7b4173",
]

# Preferred metric order for multi-page PDF output
PREFERRED_METRIC_ORDER = [
    "msd.accumulated_target_error",
    "msd.steady_state_db",
    "msd.steady_state_derivative",
    "msd.target_db",
    "emse.accumulated_target_error",
    "emse.steady_state_db",
    "emse.steady_state_derivative",
    "emse.target_db",
]

METRIC_LABELS: dict[str, str] = {
    "msd.accumulated_target_error": "MSD Accumulated Target Error (ATE)",
    "msd.steady_state_db": "MSD Steady-State [dB]",
    "msd.steady_state_derivative": "MSD Steady-State Derivative",
    "msd.target_db": "MSD Target [dB]",
    "emse.accumulated_target_error": "EMSE Accumulated Target Error (ATE)",
    "emse.steady_state_db": "EMSE Steady-State [dB]",
    "emse.steady_state_derivative": "EMSE Steady-State Derivative",
    "emse.target_db": "EMSE Target [dB]",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generate multi-page PDF value-count histograms summarizing adaptive filter comparisons.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--comparisons-dir",
      type=str,
      default="experiments/comparisons/gisv_vs_combinations",
      help="Root directory of comparison experiments containing leaf folders with 'metrics_summary.yml'.",
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default="experiments/summaries/comparisons/gisv_vs_combinations",
      help="Directory where output summary multi-page PDFs will be saved.",
  )
  parser.add_argument(
      "--bins",
      type=int,
      default=8,
      help="Number of bins for histogram intervals.",
  )
  parser.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress detailed console progress output.",
  )
  return parser.parse_args()


def flatten_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
  """Recursively flatten nested metric dictionaries into dot-separated keys."""
  items: dict[str, float] = {}
  for key, val in data.items():
    new_key = f"{prefix}.{key}" if prefix else str(key)
    if isinstance(val, dict):
      items.update(flatten_metrics(val, new_key))
    elif isinstance(val, (int, float, np.integer, np.floating)):
      items[new_key] = float(val)
    elif val is None:
      items[new_key] = np.nan
  return items


def extract_experiment_axes(leaf_dir: Path, comparisons_root: Path) -> tuple[str, str, str]:
  """Extract (environment, target_msd, input_characteristic) from directory hierarchy."""
  try:
    rel_parts = leaf_dir.resolve().relative_to(comparisons_root.resolve()).parts
    if len(rel_parts) >= 3:
      return rel_parts[0], rel_parts[1], rel_parts[2]
  except ValueError:
    pass

  parts = leaf_dir.resolve().parts
  if len(parts) >= 3:
    return parts[-3], parts[-2], parts[-1]
  raise ValueError(f"Cannot extract experiment axes from path: {leaf_dir}")


def scan_and_load_metrics(
    comparisons_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
  """Scan comparisons_dir, discover all metrics and algorithms, and load records.

  Returns
  -------
  records : list of dict
    Each record contains experiment metadata and flattened algorithm metrics.
  ordered_algorithms : list of str
    Discovered algorithm names in preferred/deterministic order.
  ordered_metrics : list of str
    Discovered metric names in preferred/deterministic order.
  """
  comp_path = comparisons_dir.resolve()
  if not comp_path.is_dir():
    raise FileNotFoundError(f"Comparisons directory not found: {comp_path}")

  summary_files = sorted(comp_path.rglob("metrics_summary.yml"))
  if not summary_files:
    raise FileNotFoundError(f"No 'metrics_summary.yml' files found under: {comp_path}")

  raw_entries: list[dict[str, Any]] = []
  all_algorithms_set: set[str] = set()
  all_metrics_set: set[str] = set()

  # First pass: load YAMLs and discover all algorithms and metrics
  for file_path in summary_files:
    with open(file_path, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f) or {}

    leaf_dir = file_path.parent
    try:
      env, target, inp = extract_experiment_axes(leaf_dir, comp_path)
    except Exception as e:
      warnings.warn(f"Skipping {file_path}: {e}")
      continue

    alg_data = data.get("algorithms", {})
    if not isinstance(alg_data, dict):
      warnings.warn(f"No valid 'algorithms' dictionary in {file_path}")
      continue

    flattened_alg_metrics: dict[str, dict[str, float]] = {}
    for alg_name, metrics_dict in alg_data.items():
      all_algorithms_set.add(alg_name)
      if isinstance(metrics_dict, dict):
        flat_m = flatten_metrics(metrics_dict)
      else:
        flat_m = {}
      flattened_alg_metrics[alg_name] = flat_m
      all_metrics_set.update(flat_m.keys())

    raw_entries.append({
        "file_path": file_path,
        "leaf_dir": leaf_dir,
        "environment": env,
        "target_msd": target,
        "input_characteristic": inp,
        "algorithms": flattened_alg_metrics,
        "N": data.get("N"),
        "NR": data.get("NR"),
        "computation_time_seconds": data.get("computation_time_seconds"),
    })

  # Determine ordered algorithms
  ordered_algorithms: list[str] = []
  for alg in PREFERRED_ALGORITHM_ORDER:
    if alg in all_algorithms_set:
      ordered_algorithms.append(alg)
  for alg in sorted(all_algorithms_set):
    if alg not in ordered_algorithms:
      ordered_algorithms.append(alg)

  # Determine ordered metrics
  ordered_metrics: list[str] = []
  for m in PREFERRED_METRIC_ORDER:
    if m in all_metrics_set:
      ordered_metrics.append(m)
  for m in sorted(all_metrics_set):
    if m not in ordered_metrics:
      ordered_metrics.append(m)

  # Completeness check & warning reporting
  total_files = len(raw_entries)
  for metric in ordered_metrics:
    missing_occurrences: list[str] = []
    for entry in raw_entries:
      for alg in ordered_algorithms:
        alg_metrics = entry["algorithms"].get(alg, {})
        if metric not in alg_metrics or np.isnan(alg_metrics[metric]):
          missing_occurrences.append(f"{entry['leaf_dir'].name}:{alg}")
          # Ensure NaN is explicitly set
          alg_metrics[metric] = np.nan

    if missing_occurrences:
      warnings.warn(
          f"Metric '{metric}' is missing or NaN in {len(missing_occurrences)} "
          f"out of {total_files * len(ordered_algorithms)} evaluations. Missing values treated as NaN."
      )

  return raw_entries, ordered_algorithms, ordered_metrics


def get_algorithm_color(alg_name: str, fallback_idx: int = 0) -> str:
  """Return dedicated color for an algorithm, or fallback palette entry."""
  if alg_name in ALGORITHM_COLORS:
    return ALGORITHM_COLORS[alg_name]
  return FALLBACK_PALETTE[fallback_idx % len(FALLBACK_PALETTE)]


def compute_shared_bins(
    records: list[dict[str, Any]],
    algorithms: list[str],
    metric: str,
    n_bins: int,
) -> tuple[np.ndarray, bool]:
  """Compute shared histogram bin edges across all algorithms and records for a given metric.

  Returns
  -------
  bins : np.ndarray
    Array of bin edges.
  use_log : bool
    Whether to use a logarithmic scale for this metric (True if all values > 0).
  """
  all_vals: list[float] = []
  for r in records:
    for alg in algorithms:
      val = r["algorithms"].get(alg, {}).get(metric, np.nan)
      if not np.isnan(val):
        all_vals.append(val)

  if not all_vals:
    return np.linspace(0.0, 1.0, n_bins + 1), False

  min_val = float(np.min(all_vals))
  max_val = float(np.max(all_vals))

  # Use log scale only if all values are strictly positive (> 0); keep linear for negative/zero values
  use_log = (min_val > 0)

  if np.isclose(min_val, max_val):
    if use_log:
      min_pad = min_val * 0.9
      max_pad = max_val * 1.1
      return np.geomspace(min_pad, max_pad, n_bins + 1), True
    else:
      padding = max(abs(min_val) * 0.1, 0.1)
      return np.linspace(min_val - padding, max_val + padding, n_bins + 1), False

  if use_log:
    min_pad = min_val * 0.95
    max_pad = max_val * 1.05
    return np.geomspace(min_pad, max_pad, n_bins + 1), True
  else:
    span = max_val - min_val
    pad = span * 0.02
    return np.linspace(min_val - pad, max_val + pad, n_bins + 1), False


def plot_metric_histogram_grid(
    records: list[dict[str, Any]],
    algorithms: list[str],
    metric: str,
    group_title: str,
    categories: list[str],
    category_labels: dict[str, str],
    category_extractor: Any,
    n_bins: int = 8,
) -> plt.Figure:
  """Plot a multi-column grid of value-count histograms for a single metric.

  Rows: Algorithms
  Columns: Group categories
  """
  n_rows = len(algorithms)
  n_cols = len(categories)

  # Dynamic figure dimensions
  col_width = 4.2 if n_cols > 1 else 6.0
  row_height = 2.2
  fig_width = n_cols * col_width + 1.2
  fig_height = n_rows * row_height + 1.5

  fig, axes = plt.subplots(
      n_rows,
      n_cols,
      figsize=(fig_width, fig_height),
      sharex=True,
      squeeze=False,
  )

  # Compute shared bin edges for this metric and determine scale (log vs linear)
  shared_bins, use_log = compute_shared_bins(records, algorithms, metric, n_bins)
  metric_title = METRIC_LABELS.get(metric, metric)

  # Partition records by category
  grouped_records: dict[str, list[dict[str, Any]]] = {c: [] for c in categories}
  for r in records:
    cat = category_extractor(r)
    if cat in grouped_records:
      grouped_records[cat].append(r)

  # Track maximum count per row or overall for uniform y-limits
  col_max_counts: dict[int, int] = {c_idx: 0 for c_idx in range(n_cols)}
  row_max_counts: dict[int, int] = {r_idx: 0 for r_idx in range(n_rows)}

  # First pass: count frequencies to determine clean y-limits
  for r_idx, alg in enumerate(algorithms):
    for c_idx, cat in enumerate(categories):
      cat_recs = grouped_records[cat]
      vals = [
          r["algorithms"].get(alg, {}).get(metric, np.nan)
          for r in cat_recs
      ]
      valid_vals = [v for v in vals if not np.isnan(v)]
      if valid_vals:
        counts, _ = np.histogram(valid_vals, bins=shared_bins)
        max_c = int(np.max(counts)) if len(counts) > 0 else 0
        col_max_counts[c_idx] = max(col_max_counts[c_idx], max_c)
        row_max_counts[r_idx] = max(row_max_counts[r_idx], max_c)

  # Second pass: render histograms
  for r_idx, alg in enumerate(algorithms):
    alg_color = get_algorithm_color(alg, fallback_idx=r_idx)

    for c_idx, cat in enumerate(categories):
      ax = axes[r_idx, c_idx]
      cat_recs = grouped_records[cat]

      vals = [
          r["algorithms"].get(alg, {}).get(metric, np.nan)
          for r in cat_recs
      ]
      valid_vals = [v for v in vals if not np.isnan(v)]

      # Draw value-count histogram
      if valid_vals:
        n, _, patches = ax.hist(
            valid_vals,
            bins=shared_bins,
            color=alg_color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        # Annotate non-zero count on top of bars for readability
        for count, patch in zip(n, patches):
          if count > 0:
            x_left = patch.get_x()
            x_right = x_left + patch.get_width()
            if use_log and x_left > 0 and x_right > 0:
              x_center = np.sqrt(x_left * x_right)
            else:
              x_center = (x_left + x_right) / 2.0
            ax.annotate(
                f"{int(count)}",
                xy=(x_center, count),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
        mean_v = float(np.mean(valid_vals))
        ax.text(
            0.97,
            0.88,
            f"N={len(valid_vals)}\nμ={mean_v:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="#cccccc"),
        )
      else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="#888888",
        )

      # Y-axis styling
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
      max_y = max(col_max_counts[c_idx], row_max_counts[r_idx], 2)
      ax.set_ylim(0, max_y * 1.25 + 0.5)
      ax.grid(True, axis="y", linestyle="--", alpha=0.35)

      # Column title on top row
      if r_idx == 0:
        c_title = category_labels.get(cat, cat)
        n_sims = len(cat_recs)
        ax.set_title(f"{c_title}\n({n_sims} simulations)", fontsize=10, fontweight="bold", pad=8)

      # Algorithm label on leftmost column
      if c_idx == 0:
        ax.set_ylabel(f"{alg}\nCount", fontsize=9, fontweight="bold")

      # X-axis scale (log scale for strictly positive metrics, linear for negative/zero metrics)
      if use_log:
        ax.set_xscale("log")
      else:
        ax.set_xscale("linear")

      # X-axis label on bottom row
      if r_idx == n_rows - 1:
        x_label_suffix = " (Log Scale)" if use_log else ""
        ax.set_xlabel(f"{metric_title}{x_label_suffix}", fontsize=8.5, fontweight="bold")

      # Ensure all subplots display x-axis tick values
      ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=8)
      ax.tick_params(axis="y", labelsize=8)

  scale_info = " [Log Scale]" if use_log else " [Linear Scale]"
  fig.suptitle(
      f"{group_title} — Metric: {metric_title}{scale_info}",
      fontsize=12,
      fontweight="bold",
      y=0.995,
  )
  plt.tight_layout(rect=[0, 0, 1, 0.97])
  return fig


def generate_grouping_pdf(
    records: list[dict[str, Any]],
    algorithms: list[str],
    metrics: list[str],
    output_pdf_path: Path,
    group_title: str,
    categories: list[str],
    category_labels: dict[str, str],
    category_extractor: Any,
    n_bins: int = 8,
    quiet: bool = False,
) -> None:
  """Generate a multi-page PDF for a specific grouping criteria where each page is a metric."""
  output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
  if not quiet:
    print(f"\nGenerating '{output_pdf_path.name}' ({len(metrics)} pages, {len(algorithms)} algorithms, {len(categories)} categories)...")

  with PdfPages(output_pdf_path) as pdf:
    for page_idx, metric in enumerate(metrics, start=1):
      metric_label = METRIC_LABELS.get(metric, metric)
      if not quiet:
        print(f"  Page {page_idx}/{len(metrics)}: {metric_label}")

      fig = plot_metric_histogram_grid(
          records=records,
          algorithms=algorithms,
          metric=metric,
          group_title=group_title,
          categories=categories,
          category_labels=category_labels,
          category_extractor=category_extractor,
          n_bins=n_bins,
      )
      pdf.savefig(fig, bbox_inches="tight")
      plt.close(fig)

  if not quiet:
    print(f"✓ Saved: {output_pdf_path}")


def main() -> int:
  args = parse_args()
  comparisons_dir = Path(args.comparisons_dir).resolve()
  output_dir = Path(args.output_dir).resolve()

  print("=" * 75)
  print("Adaptive Filter Comparisons Summary - Histogram Generator")
  print("=" * 75)
  print(f"Input directory:  {comparisons_dir}")
  print(f"Output directory: {output_dir}")
  print(f"Bins per plot:    {args.bins}")

  # 1. Discover leaf experiments, algorithms, and metrics
  try:
    records, algorithms, metrics = scan_and_load_metrics(comparisons_dir)
  except Exception as e:
    sys.stderr.write(f"\nError scanning comparisons directory: {e}\n")
    return 1

  print(f"\nFound {len(records)} comparison experiments.")
  print(f"Algorithms ({len(algorithms)}): {', '.join(algorithms)}")
  print(f"Metrics available ({len(metrics)}):")
  for m in metrics:
    print(f"  - {m} ({METRIC_LABELS.get(m, m)})")

  output_dir.mkdir(parents=True, exist_ok=True)

  # Grouping configurations
  # 1. Overall
  generate_grouping_pdf(
      records=records,
      algorithms=algorithms,
      metrics=metrics,
      output_pdf_path=output_dir / "overall.pdf",
      group_title="Overall Distribution",
      categories=["All Simulations"],
      category_labels={"All Simulations": "All Simulations"},
      category_extractor=lambda r: "All Simulations",
      n_bins=args.bins,
      quiet=args.quiet,
  )

  # 2. Target steady-state MSD
  target_categories = ["-30dB", "-40dB", "-50dB"]
  # Include any other targets found dynamically
  for r in records:
    if r["target_msd"] not in target_categories:
      target_categories.append(r["target_msd"])

  generate_grouping_pdf(
      records=records,
      algorithms=algorithms,
      metrics=metrics,
      output_pdf_path=output_dir / "target_msd.pdf",
      group_title="Grouped by Target Steady-State MSD",
      categories=target_categories,
      category_labels={t: f"Target MSD: {t}" for t in target_categories},
      category_extractor=lambda r: r["target_msd"],
      n_bins=args.bins,
      quiet=args.quiet,
  )

  # 3. Environment noise
  env_categories = ["gaussian", "laplacian", "heavy_tail"]
  for r in records:
    if r["environment"] not in env_categories:
      env_categories.append(r["environment"])

  env_labels = {
      "gaussian": "Gaussian Noise",
      "laplacian": "Laplacian Noise",
      "heavy_tail": "General Gaussian (Heavy Tail)",
  }
  for env in env_categories:
    if env not in env_labels:
      env_labels[env] = env.replace("_", " ").title()

  generate_grouping_pdf(
      records=records,
      algorithms=algorithms,
      metrics=metrics,
      output_pdf_path=output_dir / "environment_noise.pdf",
      group_title="Grouped by Environment Noise",
      categories=env_categories,
      category_labels=env_labels,
      category_extractor=lambda r: r["environment"],
      n_bins=args.bins,
      quiet=args.quiet,
  )

  # 4. Input signal correlation
  input_categories = ["white_input", "correlated_input"]
  for r in records:
    if r["input_characteristic"] not in input_categories:
      input_categories.append(r["input_characteristic"])

  input_labels = {
      "white_input": "White Input",
      "correlated_input": "Correlated Input",
  }
  for inp in input_categories:
    if inp not in input_labels:
      input_labels[inp] = inp.replace("_", " ").title()

  generate_grouping_pdf(
      records=records,
      algorithms=algorithms,
      metrics=metrics,
      output_pdf_path=output_dir / "input_correlation.pdf",
      group_title="Grouped by Input Signal Correlation",
      categories=input_categories,
      category_labels=input_labels,
      category_extractor=lambda r: r["input_characteristic"],
      n_bins=args.bins,
      quiet=args.quiet,
  )

  print(f"\n{'=' * 75}")
  print(f"✓ All summary PDFs successfully generated in: {output_dir}")
  print(f"{'=' * 75}\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
