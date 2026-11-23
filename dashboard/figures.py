"""Figure discovery, PDF-to-image conversion, and image loading."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from PIL import Image

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

# Figure extensions recognized by the dashboard
SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".svg", ".webp")

# Simple in-memory cache for rendered PDF images: (path_str, page_index, scale) -> Image
_PDF_IMAGE_CACHE: dict[tuple[str, int, float], Image.Image] = {}


def find_figures(folder: str | Path) -> list[Path]:
    """Find all supported figure files in the given folder.

    If the folder has a 'figures' subfolder, searches that. Otherwise searches folder directly.
    """
    target = Path(folder).resolve()
    if (target / "figures").is_dir():
        target = target / "figures"

    if not target.is_dir():
        return []

    figures: list[Path] = []
    for item in sorted(target.iterdir()):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            figures.append(item)

    return figures


def render_pdf_to_images(
    pdf_path: str | Path,
    scale: float = 2.0,
    max_pages: int = 10,
) -> list[Image.Image]:
    """Render pages of a PDF file to PIL Images using pypdfium2.

    Results are cached in memory for fast interactive navigation.
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"PDF file not found: {path_obj}")

    if pdfium is None:
        raise ImportError(
            "pypdfium2 is required for rendering PDF figures. "
            "Please install it using 'pip install pypdfium2'."
        )

    path_str = str(path_obj)
    images: list[Image.Image] = []

    try:
        pdf = pdfium.PdfDocument(path_str)
        n_pages = min(len(pdf), max_pages)

        for page_idx in range(n_pages):
            cache_key = (path_str, page_idx, scale)
            if cache_key in _PDF_IMAGE_CACHE:
                images.append(_PDF_IMAGE_CACHE[cache_key])
                continue

            page = pdf[page_idx]
            # scale=2.0 renders at 144 DPI (72 * 2), providing high resolution
            pil_image = page.render(scale=scale).to_pil()
            _PDF_IMAGE_CACHE[cache_key] = pil_image
            images.append(pil_image)

        pdf.close()
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF {path_obj}: {e}") from e

    return images


def load_figure_images(fig_path: str | Path, scale: float = 2.0) -> list[Image.Image]:
    """Load a figure as a list of PIL Images (1 image per page for PDFs)."""
    path_obj = Path(fig_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Figure file not found: {path_obj}")

    suffix = path_obj.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_to_images(path_obj, scale=scale)

    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        image = Image.open(path_obj)
        # Force load image data
        image.load()
        return [image]

    # For SVGs or other types, fallback to loading if PIL supports it
    try:
        image = Image.open(path_obj)
        image.load()
        return [image]
    except Exception as e:
        raise ValueError(f"Unsupported image format for direct rendering: {suffix} ({e})")


def get_figure_info(fig_path: str | Path) -> dict[str, Any]:
    """Retrieve metadata about a figure file."""
    path_obj = Path(fig_path).resolve()
    if not path_obj.is_file():
        return {}

    stat = path_obj.stat()
    size_kb = stat.st_size / 1024.0

    return {
        "filename": path_obj.name,
        "stem": path_obj.stem,
        "suffix": path_obj.suffix.lower(),
        "size_kb": size_kb,
        "path": str(path_obj),
    }
