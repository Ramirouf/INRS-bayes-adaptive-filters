"""Tests for grid_search_summary.py script."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from grid_search_summary import (
    extract_experiment_axes,
    format_candidate_label,
    generate_grouping_pdf,
    load_algorithm_configs,
    plot_algorithm_parameter_histograms_grid,
    scan_and_load_grid_search_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_format_candidate_label():
  assert format_candidate_label("reg_rho", 0.0001) == "$10^{-4}$"
  assert format_candidate_label("reg_rho", 0.001) == "$10^{-3}$"
  assert format_candidate_label("reg_rho", 0.01) == "$10^{-2}$"
  assert format_candidate_label("mu_a", 1.0) == "1.0"
  assert format_candidate_label("mu_a", 2.5) == "2.5"
  assert format_candidate_label("l", 0.9375) == "0.9375\n(1-1/L)"
  assert format_candidate_label("l", 0.9999375) == "0.9999375\n(1-0.001/L)"


def test_extract_experiment_axes():
  gs_root = REPO_ROOT / "experiments" / "grid_searches"
  leaf = gs_root / "skf_combination" / "gaussian" / "-30dB" / "correlated_input"
  alg, env, target, inp = extract_experiment_axes(leaf, gs_root)
  assert alg == "skf_combination"
  assert env == "gaussian"
  assert target == "-30dB"
  assert inp == "correlated_input"


def test_load_algorithm_configs():
  gs_root = REPO_ROOT / "experiments" / "grid_searches"
  configs = load_algorithm_configs(gs_root)
  assert "skf_combination" in configs
  assert "skf_l_combination" in configs
  skf_conv = configs["skf_combination"].get("convergence_parameters", {})
  assert "reg_rho" in skf_conv
  assert "mu_a" in skf_conv
  assert "l" in skf_conv


def test_scan_and_load_grid_search_records_on_repo():
  gs_root = REPO_ROOT / "experiments" / "grid_searches"
  records, alg_configs, ordered_algs = scan_and_load_grid_search_records(gs_root)
  assert len(records) == 36
  assert "skf_combination" in ordered_algs
  assert "skf_l_combination" in ordered_algs
  for r in records:
    assert "optimal_parameters" in r
    assert "reg_rho" in r["optimal_parameters"]


def test_plot_algorithm_parameter_histograms_grid():
  gs_root = REPO_ROOT / "experiments" / "grid_searches"
  records, alg_configs, ordered_algs = scan_and_load_grid_search_records(gs_root)
  alg_recs = [r for r in records if r["algorithm_dir"] == "skf_combination"]
  fig = plot_algorithm_parameter_histograms_grid(
      alg_records=alg_recs,
      alg_dir="skf_combination",
      alg_config=alg_configs["skf_combination"],
      group_title="Environment Noise",
      group_key="environment",
      categories=["gaussian", "heavy_tail", "laplacian"],
      category_labels={
          "gaussian": "Gaussian Noise",
          "heavy_tail": "Heavy-Tailed Noise",
          "laplacian": "Laplacian Noise",
      },
  )
  assert isinstance(fig, plt.Figure)
  plt.close(fig)


def test_generate_grouping_pdf(tmp_path: Path):
  gs_root = REPO_ROOT / "experiments" / "grid_searches"
  records, alg_configs, ordered_algs = scan_and_load_grid_search_records(gs_root)
  pdf_path = tmp_path / "test_env_summary.pdf"
  generate_grouping_pdf(
      records=records,
      alg_configs=alg_configs,
      ordered_algorithms=ordered_algs,
      output_pdf_path=pdf_path,
      group_title="Environment Noise",
      group_key="environment",
      preferred_categories=["gaussian", "heavy_tail", "laplacian"],
      category_labels={
          "gaussian": "Gaussian Noise",
          "heavy_tail": "Heavy-Tailed Noise",
          "laplacian": "Laplacian Noise",
      },
      quiet=True,
  )
  assert pdf_path.is_file()
  assert pdf_path.stat().st_size > 0
