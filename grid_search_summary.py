#!/usr/bin/env python3
"""Cross-experiment summary script for adaptive filter hyperparameter grid searches.

Scans all leaf experiment directories in 'experiments/grid_searches', dynamically
reads the convergence parameters from 'experiments/grid_searches/<algorithm>/grid_config.yml',
and generates multi-page PDF summaries with value-count histogram grids grouped by:
  1. Environment Noise (gaussian, heavy_tail, laplacian)
  2. Target Steady-State MSD (-30dB, -40dB, -50dB)
  3. Input Signal Correlation (white_input, correlated_input)
  4. Overall (all simulations, optional)

In each PDF:
  - Each page corresponds to a distinct algorithm.
  - Subplot grid: Rows = Convergence Parameters, Columns = Group categories.
  - Each cell displays a value-count histogram of the optimal hyperparameters.
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

from run_grid_search import find_leaf_experiments

# Preferred order of algorithms
PREFERRED_ALGORITHM_ORDER = [
    "skf_combination",
    "skf_l_combination",
    "cvx_comb_sKF",
    "cvx_comb_sKF_L",
]

# Display titles for known algorithms
ALGORITHM_TITLES: dict[str, str] = {
    "skf_combination": "cvx_comb_sKF",
    "skf_l_combination": "cvx_comb_sKF_L",
    "cvx_comb_sKF": "cvx_comb_sKF",
    "cvx_comb_sKF_L": "cvx_comb_sKF_L",
}

# Clean LaTeX / text labels for convergence parameters
PARAMETER_LABELS: dict[str, str] = {
    "reg_rho": r"$\rho_{\mathrm{reg}}$" + "\n(Regularization)",
    "lambda_0": r"$\lambda_0$" + "\n(Initial Weight)",
    "mu_a": r"$\mu_a$" + "\n(Step Size)",
    "epsilon_1": r"$\epsilon_1$" + "\n(Fast Reg.)",
    "a_plus": r"$a_+$" + "\n(Asymmetry)",
    "l": r"$l$" + "\n(Leakage)",
}

# Distinct palette for parameter rows
PARAM_ROW_COLORS = [
    "#1f77b4",  # steelblue
    "#2ca02c",  # green
    "#ff7f0e",  # darkorange
    "#d62728",  # crimson
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Known categories and display labels
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "environment": {
        "gaussian": "Gaussian Noise",
        "heavy_tail": "Heavy-Tailed Noise",
        "laplacian": "Laplacian Noise",
    },
    "target_msd": {
        "-30dB": "Target MSD: -30 dB",
        "-40dB": "Target MSD: -40 dB",
        "-50dB": "Target MSD: -50 dB",
    },
    "input_characteristic": {
        "white_input": "White Input",
        "correlated_input": "Correlated Input",
    },
    "overall": {
        "all": "All Simulations",
    },
}

PREFERRED_CATEGORY_ORDER: dict[str, list[str]] = {
    "environment": ["gaussian", "heavy_tail", "laplacian"],
    "target_msd": ["-30dB", "-40dB", "-50dB"],
    "input_characteristic": ["white_input", "correlated_input"],
    "overall": ["all"],
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generate multi-page PDF value-count histograms summarizing adaptive filter grid searches.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--grid-searches-dir",
      type=str,
      default="experiments/grid_searches",
      help="Root directory of grid search experiments containing algorithm subdirectories with 'grid_config.yml'.",
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default="experiments/summaries/grid_searches",
      help="Directory where output summary multi-page PDFs will be saved.",
  )
  parser.add_argument(
      "--no-overall",
      dest="generate_overall",
      action="store_false",
      default=True,
      help="Skip generating overall.pdf summarizing all simulations together.",
  )
  parser.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress detailed console progress output.",
  )
  return parser.parse_args()


def format_candidate_label(param: str, cand: Any) -> str:
  """Format candidate parameter value for x-axis tick label."""
  if param == "l":
    known_l = {
        0.9375: "0.9375\n(1-1/L)",
        0.99375: "0.99375\n(1-0.1/L)",
        0.999375: "0.999375\n(1-0.01/L)",
        0.9999375: "0.9999375\n(1-0.001/L)",
    }
    for k, v in known_l.items():
      if np.isclose(float(cand), k, atol=1e-8):
        return v

  try:
    val_f = float(cand)
    if abs(val_f) > 0 and (abs(val_f) <= 0.01 or abs(val_f) >= 10000):
      exp_val = int(round(np.log10(abs(val_f))))
      if np.isclose(val_f, 10 ** exp_val, rtol=1e-5):
        return f"$10^{{{exp_val}}}$"
      mantissa = val_f / (10 ** exp_val)
      return f"{mantissa:.1f}" + r"\times 10^{" + str(exp_val) + "}"
    elif np.isclose(val_f, round(val_f), atol=1e-8):
      return f"{val_f:.1f}"
    else:
      return f"{val_f:g}"
  except (ValueError, TypeError):
    return str(cand)


def load_algorithm_configs(grid_searches_dir: Path) -> dict[str, dict[str, Any]]:
  """Discover all algorithms and load their grid_config.yml configurations."""
  configs: dict[str, dict[str, Any]] = {}
  for item in sorted(grid_searches_dir.iterdir()):
    if item.is_dir():
      cfg_file = item / "grid_config.yml"
      if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as f:
          cfg = yaml.safe_load(f) or {}
        configs[item.name] = cfg
  return configs


def extract_experiment_axes(leaf_dir: Path, grid_searches_root: Path) -> tuple[str, str, str, str]:
  """Extract (algorithm_dir, environment, target_msd, input_characteristic) from directory hierarchy."""
  try:
    rel_parts = leaf_dir.resolve().relative_to(grid_searches_root.resolve()).parts
    if len(rel_parts) >= 4:
      return rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3]
  except ValueError:
    pass

  parts = leaf_dir.resolve().parts
  if len(parts) >= 4:
    return parts[-4], parts[-3], parts[-2], parts[-1]
  raise ValueError(f"Cannot extract experiment axes from path: {leaf_dir}")


def scan_and_load_grid_search_records(
    grid_searches_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
  """Scan grid_searches_dir, discover leaf experiments, and load optimal parameters."""
  gs_path = grid_searches_dir.resolve()
  if not gs_path.is_dir():
    raise FileNotFoundError(f"Grid searches directory not found: {gs_path}")

  alg_configs = load_algorithm_configs(gs_path)
  if not alg_configs:
    raise FileNotFoundError(f"No algorithm folders with 'grid_config.yml' found under: {gs_path}")

  leaves = find_leaf_experiments(gs_path)
  records: list[dict[str, Any]] = []

  for leaf in leaves:
    opt_file = leaf / "optimal_parameters.yml"
    if not opt_file.is_file():
      continue

    with open(opt_file, "r", encoding="utf-8") as f:
      opt_params = yaml.safe_load(f) or {}

    summary_file = leaf / "grid_search_summary.yml"
    summary_data = {}
    if summary_file.is_file():
      with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = yaml.safe_load(f) or {}

    alg_dir, env, target, inp = extract_experiment_axes(leaf, gs_path)
    records.append({
        "leaf_dir": leaf,
        "algorithm_dir": alg_dir,
        "environment": env,
        "target_msd": target,
        "input_characteristic": inp,
        "overall": "all",
        "optimal_parameters": opt_params,
        "summary": summary_data,
    })

  # Order algorithms according to preference
  all_algs = sorted(list(alg_configs.keys()))
  ordered_algorithms: list[str] = []
  for alg in PREFERRED_ALGORITHM_ORDER:
    if alg in all_algs and alg not in ordered_algorithms:
      ordered_algorithms.append(alg)
  for alg in all_algs:
    if alg not in ordered_algorithms:
      ordered_algorithms.append(alg)

  return records, alg_configs, ordered_algorithms


def plot_algorithm_parameter_histograms_grid(
    alg_records: list[dict[str, Any]],
    alg_dir: str,
    alg_config: dict[str, Any],
    group_title: str,
    group_key: str,
    categories: list[str],
    category_labels: dict[str, str],
) -> plt.Figure:
  """Plot a multi-row, multi-column grid of value-count histograms for one algorithm.

  Rows: Convergence parameters listed in grid_config.yml
  Columns: Group categories (e.g., gaussian, heavy_tail, laplacian)
  """
  conv_params: dict[str, list[Any]] = alg_config.get("convergence_parameters", {})
  param_names = list(conv_params.keys())
  n_rows = len(param_names)
  n_cols = len(categories)

  if n_rows == 0:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"No convergence parameters defined for {alg_dir}", ha="center", va="center")
    return fig

  # Dynamically compute figure dimensions: generous space per row and column
  fig_width = max(6.5, 4.4 * n_cols)
  fig_height = max(5.0, 2.25 * n_rows + 1.2)
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

  alg_title = alg_config.get("algorithm", ALGORITHM_TITLES.get(alg_dir, alg_dir))

  # Pre-calculate candidate values and candidate labels for each parameter row
  row_info = []
  for r_idx, param in enumerate(param_names):
    raw_cands = conv_params[param]
    try:
      cands_sorted = sorted(raw_cands, key=float)
    except (ValueError, TypeError):
      cands_sorted = list(raw_cands)
    cand_labels = [format_candidate_label(param, c) for c in cands_sorted]
    param_label = PARAMETER_LABELS.get(param, param.replace("_", " ").title())
    color = PARAM_ROW_COLORS[r_idx % len(PARAM_ROW_COLORS)]
    row_info.append({
        "param": param,
        "cands": cands_sorted,
        "labels": cand_labels,
        "param_label": param_label,
        "color": color,
    })

  for r_idx, r_data in enumerate(row_info):
    param = r_data["param"]
    cands = r_data["cands"]
    x_indices = np.arange(len(cands))
    bar_width = min(0.6, 0.85 / max(len(cands), 1) * 2.0)

    # First pass over columns for this row: determine max count to harmonize y-limits
    col_counts = []
    for c_idx, cat in enumerate(categories):
      cat_recs = [r for r in alg_records if r.get(group_key) == cat]
      vals = [
          r["optimal_parameters"][param]
          for r in cat_recs
          if param in r["optimal_parameters"] and r["optimal_parameters"][param] is not None
      ]
      counts = []
      for c in cands:
        try:
          cnt = sum(np.isclose(float(v), float(c), rtol=1e-5, atol=1e-8) for v in vals)
        except (ValueError, TypeError):
          cnt = sum(str(v) == str(c) for v in vals)
        counts.append(cnt)
      col_counts.append((cat_recs, counts))

    max_row_count = max([max(cnts) if cnts else 0 for _, cnts in col_counts] + [0])
    y_max = max(max_row_count * 1.35 + 0.8, 3.0)

    for c_idx, (cat_recs, counts) in enumerate(col_counts):
      ax = axes[r_idx, c_idx]
      n_cat = len(cat_recs)

      bars = ax.bar(
          x_indices,
          counts,
          width=bar_width,
          color=r_data["color"],
          edgecolor="black",
          alpha=0.75,
      )

      # Annotate bars
      for bar, count in zip(bars, counts):
        if count > 0:
          pct = (count / n_cat * 100.0) if n_cat > 0 else 0.0
          pct_str = f"{pct:.0f}%" if pct.is_integer() else f"{pct:.1f}%"
          ax.annotate(
              f"{count}\n({pct_str})",
              xy=(bar.get_x() + bar.get_width() / 2, count),
              xytext=(0, 2),
              textcoords="offset points",
              ha="center",
              va="bottom",
              fontsize=7.5,
              fontweight="bold",
          )

      ax.set_xticks(x_indices)
      ax.set_xticklabels(r_data["labels"], fontsize=8)
      ax.set_ylim(0, y_max)
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
      ax.grid(True, axis="y", alpha=0.3, linestyle="--")

      # Column title on top row
      if r_idx == 0:
        c_title = category_labels.get(categories[c_idx], categories[c_idx])
        ax.set_title(f"{c_title}\n(N = {n_cat})", fontsize=10.5, fontweight="bold", pad=8)

      # Parameter label on leftmost column
      if c_idx == 0:
        ax.set_ylabel(f"{r_data['param_label']}\nCount", fontsize=8.5, fontweight="bold")

      # Bottom row x-axis label
      if r_idx == n_rows - 1:
        ax.set_xlabel("Candidate Value", fontsize=8.5, fontweight="bold")

      ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=8)
      ax.tick_params(axis="y", labelsize=8)

  fig.suptitle(
      f"{group_title} — Algorithm: {alg_title}",
      fontsize=13,
      fontweight="bold",
      y=0.995,
  )
  plt.tight_layout(rect=[0, 0, 1, 0.97])
  return fig


def generate_grouping_pdf(
    records: list[dict[str, Any]],
    alg_configs: dict[str, dict[str, Any]],
    ordered_algorithms: list[str],
    output_pdf_path: Path,
    group_title: str,
    group_key: str,
    preferred_categories: list[str],
    category_labels: dict[str, str],
    quiet: bool = False,
) -> None:
  """Generate a multi-page PDF for a specific grouping criteria where each page is an algorithm."""
  output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

  # Discover available categories for this group
  all_cats = set(r.get(group_key) for r in records if group_key in r and r.get(group_key) is not None)
  categories = [c for c in preferred_categories if c in all_cats]
  for c in sorted(all_cats):
    if c not in categories:
      categories.append(c)

  if not quiet:
    print(f"\nGenerating '{output_pdf_path.name}' ({len(ordered_algorithms)} algorithms/pages, {len(categories)} categories)...")

  with PdfPages(output_pdf_path) as pdf:
    for page_idx, alg_dir in enumerate(ordered_algorithms, start=1):
      alg_cfg = alg_configs.get(alg_dir, {})
      alg_title = alg_cfg.get("algorithm", ALGORITHM_TITLES.get(alg_dir, alg_dir))
      alg_recs = [r for r in records if r["algorithm_dir"] == alg_dir]

      if not quiet:
        print(f"  Page {page_idx}/{len(ordered_algorithms)}: Algorithm {alg_title} ({len(alg_recs)} experiments)")

      fig = plot_algorithm_parameter_histograms_grid(
          alg_records=alg_recs,
          alg_dir=alg_dir,
          alg_config=alg_cfg,
          group_title=group_title,
          group_key=group_key,
          categories=categories,
          category_labels=category_labels,
      )
      pdf.savefig(fig, bbox_inches="tight")
      plt.close(fig)

  if not quiet:
    print(f"✓ Saved: {output_pdf_path}")


def main() -> int:
  args = parse_args()
  grid_searches_dir = Path(args.grid_searches_dir).resolve()
  output_dir = Path(args.output_dir).resolve()

  print("=" * 75)
  print("Adaptive Filter Grid Searches Summary - Histogram Generator")
  print("=" * 75)
  print(f"Input directory:  {grid_searches_dir}")
  print(f"Output directory: {output_dir}")

  # 1. Discover leaf experiments, algorithms, and configurations
  try:
    records, alg_configs, ordered_algorithms = scan_and_load_grid_search_records(grid_searches_dir)
  except Exception as e:
    sys.stderr.write(f"\nError scanning grid searches directory: {e}\n")
    return 1

  print(f"\nFound {len(records)} completed grid search experiments.")
  print(f"Algorithms ({len(ordered_algorithms)}): {', '.join(ordered_algorithms)}")
  for alg_dir in ordered_algorithms:
    cfg = alg_configs.get(alg_dir, {})
    conv = list(cfg.get("convergence_parameters", {}).keys())
    print(f"  - {alg_dir} ({cfg.get('algorithm', alg_dir)}): {len(conv)} convergence parameters ({', '.join(conv)})")

  output_dir.mkdir(parents=True, exist_ok=True)

  # Grouping configurations:
  # 1. Environment Noise
  # 2. Target MSD
  # 3. Input Correlation
  # 4. Overall (optional)
  groups = [
      {
          "file_name": "environment_noise.pdf",
          "group_title": "Environment Noise",
          "group_key": "environment",
          "preferred_categories": PREFERRED_CATEGORY_ORDER["environment"],
          "category_labels": CATEGORY_LABELS["environment"],
      },
      {
          "file_name": "target_msd.pdf",
          "group_title": "Target Steady-State MSD",
          "group_key": "target_msd",
          "preferred_categories": PREFERRED_CATEGORY_ORDER["target_msd"],
          "category_labels": CATEGORY_LABELS["target_msd"],
      },
      {
          "file_name": "input_correlation.pdf",
          "group_title": "Input Signal Correlation",
          "group_key": "input_characteristic",
          "preferred_categories": PREFERRED_CATEGORY_ORDER["input_characteristic"],
          "category_labels": CATEGORY_LABELS["input_characteristic"],
      },
  ]

  if args.generate_overall:
    groups.append({
        "file_name": "overall.pdf",
        "group_title": "Overall Distribution",
        "group_key": "overall",
        "preferred_categories": PREFERRED_CATEGORY_ORDER["overall"],
        "category_labels": CATEGORY_LABELS["overall"],
    })

  for grp in groups:
    generate_grouping_pdf(
        records=records,
        alg_configs=alg_configs,
        ordered_algorithms=ordered_algorithms,
        output_pdf_path=output_dir / grp["file_name"],
        group_title=grp["group_title"],
        group_key=grp["group_key"],
        preferred_categories=grp["preferred_categories"],
        category_labels=grp["category_labels"],
        quiet=args.quiet,
    )

  print(f"\nAll grid search summary histograms successfully generated in:\n  {output_dir}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
