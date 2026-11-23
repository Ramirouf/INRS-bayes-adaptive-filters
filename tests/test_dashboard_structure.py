"""Tests for dashboard directory structure loading and navigation."""

from pathlib import Path
import pytest

from dashboard.structure import (
    DirStructure,
    find_experiment_folders,
    load_dir_structure,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_find_experiment_folders():
    exp_dir = REPO_ROOT / "experiments"
    folders = find_experiment_folders(exp_dir)
    assert len(folders) >= 2

    folder_names = [f["name"] for f in folders]
    assert "grid_searches" in folder_names
    assert "comparisons" in folder_names

    # Check structure flag
    gs = next(f for f in folders if f["name"] == "grid_searches")
    assert gs["has_structure"] is True
    assert gs["has_figures"] is True


def test_load_grid_searches_structure():
    gs_dir = REPO_ROOT / "experiments" / "grid_searches"
    dir_struct = load_dir_structure(gs_dir)

    assert len(dir_struct.levels) == 4
    assert dir_struct.level_names == ["algorithms", "environments", "target_msds", "inputs"]

    # Level 0 options
    opts_l0 = dir_struct.get_options_at_level(0, [])
    assert "skf_combination" in opts_l0
    assert "skf_l_combination" in opts_l0

    # Level 1 options given algorithm
    opts_l1 = dir_struct.get_options_at_level(1, ["skf_combination"])
    assert set(opts_l1) == {"gaussian", "heavy_tail", "laplacian"}

    # Level 2 options given algorithm and environment
    opts_l2 = dir_struct.get_options_at_level(2, ["skf_combination", "gaussian"])
    assert set(opts_l2) == {"-30dB", "-40dB", "-50dB"}

    # Level 3 options given algorithm, environment, and target_msd
    opts_l3 = dir_struct.get_options_at_level(3, ["skf_combination", "gaussian", "-30dB"])
    assert set(opts_l3) == {"correlated_input", "white_input"}

    # Navigation from -30dB to -40dB:
    # Notice input is still valid under -40dB!
    opts_l3_alt = dir_struct.get_options_at_level(3, ["skf_combination", "gaussian", "-40dB"])
    assert "correlated_input" in opts_l3_alt

    # Leaf path resolution and validation
    leaf_path = dir_struct.resolve_leaf_path(["skf_combination", "gaussian", "-30dB", "correlated_input"])
    assert leaf_path.is_dir()

    is_valid, resolved = dir_struct.validate_leaf(["skf_combination", "gaussian", "-30dB", "correlated_input"])
    assert is_valid is True
    assert (resolved / "figures").is_dir()


def test_load_comparisons_structure():
    comp_dir = REPO_ROOT / "experiments" / "comparisons"
    dir_struct = load_dir_structure(comp_dir)

    assert len(dir_struct.levels) == 4
    assert dir_struct.level_names == ["comparisons", "environments", "target_msds", "inputs"]

    opts_l0 = dir_struct.get_options_at_level(0, [])
    assert "gisv_vs_combinations" in opts_l0

    is_valid, _ = dir_struct.validate_leaf(["gisv_vs_combinations", "gaussian", "-30dB", "correlated_input"])
    assert is_valid is True


def test_load_direct_figures_structure():
    summary_dir = REPO_ROOT / "experiments" / "grid_search_summary"
    dir_struct = load_dir_structure(summary_dir)

    assert len(dir_struct.levels) == 0
    is_valid, resolved = dir_struct.validate_leaf([])
    assert is_valid is True
    assert (resolved / "figures").is_dir()

