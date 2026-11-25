"""Tests for comparisons_summary.py script."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import yaml

from comparisons_summary import (
    compute_shared_bins,
    extract_experiment_axes,
    flatten_metrics,
    generate_grouping_pdf,
    scan_and_load_metrics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_flatten_metrics():
  nested = {
      "msd": {
          "accumulated_target_error": 0.05,
          "steady_state_db": -30.2,
      },
      "emse": {
          "nested_sub": {
              "val": 1.23,
          },
      },
      "none_metric": None,
  }
  flattened = flatten_metrics(nested)
  assert flattened["msd.accumulated_target_error"] == 0.05
  assert flattened["msd.steady_state_db"] == -30.2
  assert flattened["emse.nested_sub.val"] == 1.23
  assert np.isnan(flattened["none_metric"])


def test_extract_experiment_axes():
  comparisons_root = REPO_ROOT / "experiments" / "comparisons" / "gisv_vs_combinations"
  leaf = comparisons_root / "gaussian" / "-30dB" / "correlated_input"
  env, target, inp = extract_experiment_axes(leaf, comparisons_root)
  assert env == "gaussian"
  assert target == "-30dB"
  assert inp == "correlated_input"


def test_scan_and_load_metrics_on_repo():
  comparisons_dir = REPO_ROOT / "experiments" / "comparisons" / "gisv_vs_combinations"
  records, algorithms, metrics = scan_and_load_metrics(comparisons_dir)

  assert len(records) == 18
  assert len(algorithms) == 6
  assert "sKF" in algorithms
  assert "cvx_comb_sKF" in algorithms
  assert "msd.accumulated_target_error" in metrics
  assert "msd.steady_state_db" in metrics


def test_compute_shared_bins():
  # Positive values -> log scale
  records_pos = [
      {"algorithms": {"sKF": {"msd.val": 1.0}, "sKF-L": {"msd.val": 3.0}}},
      {"algorithms": {"sKF": {"msd.val": 2.0}, "sKF-L": {"msd.val": 5.0}}},
  ]
  bins, use_log = compute_shared_bins(records_pos, ["sKF", "sKF-L"], "msd.val", n_bins=4)
  assert len(bins) == 5
  assert bins[0] <= 1.0
  assert bins[-1] >= 5.0
  assert use_log is True

  # Negative dB values -> linear scale
  records_neg = [
      {"algorithms": {"sKF": {"msd.db": -30.0}, "sKF-L": {"msd.db": -35.0}}},
      {"algorithms": {"sKF": {"msd.db": -40.0}, "sKF-L": {"msd.db": -50.0}}},
  ]
  bins_neg, use_log_neg = compute_shared_bins(records_neg, ["sKF", "sKF-L"], "msd.db", n_bins=4)
  assert len(bins_neg) == 5
  assert use_log_neg is False


def test_missing_metric_warning(tmp_path: Path):
  # Create a dummy structure where one file is missing a metric
  comp_dir = tmp_path / "comparisons"
  leaf1 = comp_dir / "gaussian" / "-30dB" / "white_input"
  leaf2 = comp_dir / "gaussian" / "-30dB" / "correlated_input"
  leaf1.mkdir(parents=True)
  leaf2.mkdir(parents=True)

  data1 = {
      "algorithms": {
          "sKF": {"msd": {"ate": 0.1, "ss": -30.0}},
      }
  }
  data2 = {
      "algorithms": {
          "sKF": {"msd": {"ss": -30.1}},  # missing 'ate'
      }
  }

  with open(leaf1 / "metrics_summary.yml", "w") as f:
    yaml.safe_dump(data1, f)
  with open(leaf2 / "metrics_summary.yml", "w") as f:
    yaml.safe_dump(data2, f)

  with pytest.warns(UserWarning, match="Metric 'msd.ate' is missing"):
    records, algorithms, metrics = scan_and_load_metrics(comp_dir)

  assert "msd.ate" in metrics
  assert "msd.ss" in metrics
  assert len(records) == 2
  rec_leaf2 = next(r for r in records if r["input_characteristic"] == "correlated_input")
  rec_leaf1 = next(r for r in records if r["input_characteristic"] == "white_input")
  assert np.isnan(rec_leaf2["algorithms"]["sKF"]["msd.ate"])
  assert rec_leaf1["algorithms"]["sKF"]["msd.ate"] == 0.1


def test_generate_grouping_pdf(tmp_path: Path):
  records = [
      {
          "target_msd": "-30dB",
          "algorithms": {
              "sKF": {"msd.ate": 0.01},
              "cvx_comb_sKF": {"msd.ate": 0.02},
          },
      },
      {
          "target_msd": "-30dB",
          "algorithms": {
              "sKF": {"msd.ate": 0.015},
              "cvx_comb_sKF": {"msd.ate": 0.025},
          },
      },
  ]
  pdf_path = tmp_path / "test_output.pdf"
  generate_grouping_pdf(
      records=records,
      algorithms=["sKF", "cvx_comb_sKF"],
      metrics=["msd.ate"],
      output_pdf_path=pdf_path,
      group_title="Test Group",
      categories=["-30dB"],
      category_labels={"-30dB": "-30 dB Floor"},
      category_extractor=lambda r: r["target_msd"],
      n_bins=4,
      quiet=True,
  )
  assert pdf_path.is_file()
  assert pdf_path.stat().st_size > 0
