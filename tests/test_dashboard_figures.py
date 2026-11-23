"""Tests for figure discovery and PDF-to-image rendering."""

from pathlib import Path
from PIL import Image
import pytest

from dashboard.figures import (
    find_figures,
    get_figure_info,
    load_figure_images,
    render_pdf_to_images,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_find_figures():
    leaf_dir = (
        REPO_ROOT
        / "experiments"
        / "grid_searches"
        / "skf_combination"
        / "gaussian"
        / "-30dB"
        / "correlated_input"
    )
    figures = find_figures(leaf_dir)
    assert len(figures) >= 4

    names = [f.name for f in figures]
    assert "learning_curves_comparison.pdf" in names
    assert "metric_histograms.pdf" in names


def test_load_pdf_figure():
    pdf_path = (
        REPO_ROOT
        / "experiments"
        / "grid_searches"
        / "skf_combination"
        / "gaussian"
        / "-30dB"
        / "correlated_input"
        / "figures"
        / "learning_curves_comparison.pdf"
    )
    assert pdf_path.is_file()

    images = load_figure_images(pdf_path, scale=1.0)
    assert len(images) >= 1
    assert isinstance(images[0], Image.Image)
    assert images[0].width > 100
    assert images[0].height > 100

    # Test metadata
    info = get_figure_info(pdf_path)
    assert info["filename"] == "learning_curves_comparison.pdf"
    assert info["stem"] == "learning_curves_comparison"
    assert info["suffix"] == ".pdf"
    assert info["size_kb"] > 0
