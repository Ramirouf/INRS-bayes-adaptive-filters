"""Tests for dir_structure.yml generator."""

from pathlib import Path
import tempfile
import yaml
import pytest

from dashboard.generator import (
    find_leaf_directories,
    build_tree_structure,
    scan_directory_tree,
    generate_dir_structure_yml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_find_leaf_directories_grid_searches():
    gs_dir = REPO_ROOT / "experiments" / "grid_searches"
    leaves = find_leaf_directories(gs_dir)
    # 2 algorithms * 3 environments * 3 target_msds * 2 inputs = 36 leaves
    assert len(leaves) == 36

    sample_leaf = leaves[0]
    assert (sample_leaf / "figures").is_dir()


def test_scan_directory_tree_generic_levels():
    gs_dir = REPO_ROOT / "experiments" / "grid_searches"
    result = scan_directory_tree(gs_dir)

    assert "levels" in result
    assert "tree" in result
    assert len(result["levels"]) == 4

    # Without explicit names, levels are named level01..level04
    assert result["levels"][0]["name"] == "level01"
    assert result["levels"][1]["name"] == "level02"


def test_scan_directory_tree_custom_levels():
    gs_dir = REPO_ROOT / "experiments" / "grid_searches"
    custom_names = ["algo", "noise_env", "snr_floor", "input_signal"]
    result = scan_directory_tree(gs_dir, level_names=custom_names)

    assert [lvl["name"] for lvl in result["levels"]] == custom_names


def test_generate_dir_structure_yml_file(tmp_path: Path):
    gs_dir = REPO_ROOT / "experiments" / "grid_searches"
    out_file = tmp_path / "test_dir_structure.yml"

    generate_dir_structure_yml(gs_dir, output_path=out_file)
    assert out_file.is_file()

    with open(out_file, "r") as f:
        data = yaml.safe_load(f)

    assert "levels" in data
    assert "tree" in data
