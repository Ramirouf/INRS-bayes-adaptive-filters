"""Interactive Streamlit dashboard for exploring experiment figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import yaml

from dashboard.structure import (
    DirStructure,
    find_experiment_folders,
    load_dir_structure,
)
from dashboard.figures import (
    find_figures,
    get_figure_info,
    load_figure_images,
)
from dashboard.generator import generate_dir_structure_yml


def init_session_state() -> None:
    """Initialize session state keys."""
    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = None
    if "level_selections" not in st.session_state:
        st.session_state.level_selections = {}
    if "zoomed_figure" not in st.session_state:
        st.session_state.zoomed_figure = None
    if "gallery_columns" not in st.session_state:
        st.session_state.gallery_columns = 2


def get_experiments_dir() -> Path:
    """Return the experiments root directory."""
    return REPO_ROOT / "experiments"


def render_sidebar(exp_folders: list[dict[str, Any]]) -> tuple[str | None, DirStructure | None]:
    """Render the sidebar with experiment folder selection and level hierarchy widgets."""
    st.sidebar.title("🔬 Experiment Explorer")
    st.sidebar.markdown("---")

    if not exp_folders:
        st.sidebar.warning("No experiment folders found in `experiments/`.")
        return None, None

    exp_names = [item["name"] for item in exp_folders]
    current_exp = st.session_state.selected_experiment
    if current_exp not in exp_names:
        current_exp = exp_names[0]
        st.session_state.selected_experiment = current_exp

    selected_exp_name = st.sidebar.selectbox(
        "Select Experiment Group",
        options=exp_names,
        index=exp_names.index(current_exp),
        help="Select top-level experiment category inside 'experiments/'",
        key="exp_selector",
    )

    if selected_exp_name != st.session_state.selected_experiment:
        st.session_state.selected_experiment = selected_exp_name
        st.session_state.level_selections = {}
        st.session_state.zoomed_figure = None
        st.rerun()

    exp_info = next(item for item in exp_folders if item["name"] == selected_exp_name)
    exp_path: Path = exp_info["path"]

    # Check for dir_structure.yml
    structure_file = exp_path / "dir_structure.yml"
    if not structure_file.is_file():
        st.sidebar.warning(f"No `dir_structure.yml` in `{selected_exp_name}`.")
        if st.sidebar.button("Generate dir_structure.yml"):
            with st.spinner("Generating directory structure..."):
                generate_dir_structure_yml(exp_path)
            st.sidebar.success("Generated dir_structure.yml!")
            st.rerun()
        return selected_exp_name, None

    # Load structure
    try:
        dir_struct = load_dir_structure(exp_path)
    except Exception as e:
        st.sidebar.error(f"Error loading dir_structure.yml: {e}")
        return selected_exp_name, None

    st.sidebar.subheader("Navigation Levels")
    st.sidebar.caption("Change any level directly to jump between experiments:")

    # Build selections level by level
    current_selections: list[str] = []
    level_dict = st.session_state.level_selections.get(selected_exp_name, {})

    for idx, level_spec in enumerate(dir_struct.levels):
        options = dir_struct.get_options_at_level(idx, current_selections)
        if not options:
            st.sidebar.info(f"No options available for {level_spec.title}")
            break

        prev_selected = level_dict.get(level_spec.name)
        if prev_selected in options:
            sel_idx = options.index(prev_selected)
        else:
            sel_idx = 0

        chosen = st.sidebar.selectbox(
            label=f"{level_spec.title}",
            options=options,
            index=sel_idx,
            key=f"level_{selected_exp_name}_{level_spec.name}_{idx}",
        )

        current_selections.append(chosen)
        level_dict[level_spec.name] = chosen

    st.session_state.level_selections[selected_exp_name] = level_dict

    st.sidebar.markdown("---")
    st.sidebar.caption("INRS Bayes Adaptive Filters Dashboard")

    return selected_exp_name, dir_struct


def render_breadcrumb(exp_name: str, selections: list[str], dir_struct: DirStructure) -> None:
    """Render a clean breadcrumb navigation header."""
    parts = [f"**{exp_name}**"]
    for idx, sel in enumerate(selections):
        title = dir_struct.get_level_title(idx)
        parts.append(f"`{title}`: **{sel}**")
    st.markdown(" ➔ ".join(parts))


def render_gallery_view(figures: list[Path], leaf_dir: Path) -> None:
    """Display all figures in a responsive grid layout with zoom-in buttons."""
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(f"Figures Gallery ({len(figures)} figure{'s' if len(figures) != 1 else ''})")
    with header_col2:
        num_cols = st.selectbox(
            "Columns",
            options=[1, 2, 3, 4],
            index=st.session_state.gallery_columns - 1,
            key="col_select",
        )
        st.session_state.gallery_columns = num_cols

    cols = st.columns(num_cols)
    for i, fig_path in enumerate(figures):
        col = cols[i % num_cols]
        with col:
            with st.container(border=True):
                info = get_figure_info(fig_path)
                st.markdown(f"**{info['stem']}** (`{info['suffix']}`)")

                try:
                    images = load_figure_images(fig_path, scale=1.5)
                    if images:
                        st.image(images[0], use_container_width=True)
                        if len(images) > 1:
                            st.caption(f"Page 1 of {len(images)}")
                except Exception as e:
                    st.error(f"Failed to load image: {e}")

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if st.button("🔍 Zoom In", key=f"zoom_btn_{fig_path.name}"):
                        st.session_state.zoomed_figure = fig_path.name
                        st.rerun()
                with btn_col2:
                    with open(fig_path, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download",
                        data=file_bytes,
                        file_name=fig_path.name,
                        mime="application/pdf" if fig_path.suffix == ".pdf" else "image/png",
                        key=f"dl_btn_{fig_path.name}",
                    )


def render_zoom_view(fig_path: Path) -> None:
    """Display a single zoomed figure occupying the full window."""
    info = get_figure_info(fig_path)

    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        if st.button("⬅️ Zoom Out (All Figures)", key="zoom_out_btn", use_container_width=True):
            st.session_state.zoomed_figure = None
            st.rerun()
    with top_col2:
        st.caption(f"Viewing single figure: **{info['filename']}** ({info['size_kb']:.1f} KB)")

    st.markdown("---")
    st.subheader(f"🔍 {info['stem']}")

    try:
        images = load_figure_images(fig_path, scale=2.5)
        for page_idx, img in enumerate(images):
            if len(images) > 1:
                st.markdown(f"**Page {page_idx + 1} of {len(images)}**")
            st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render figure: {e}")

    st.markdown("---")
    btn_col1, btn_col2, _ = st.columns([2, 2, 4])
    with btn_col1:
        if st.button("⬅️ Back to Gallery", key="bottom_back_btn"):
            st.session_state.zoomed_figure = None
            st.rerun()
    with btn_col2:
        with open(fig_path, "rb") as f:
            file_bytes = f.read()
        st.download_button(
            label=f"⬇️ Download {info['filename']}",
            data=file_bytes,
            file_name=fig_path.name,
            mime="application/pdf" if fig_path.suffix == ".pdf" else "image/png",
            key="zoom_dl_btn",
        )


def render_experiment_metadata(leaf_dir: Path) -> None:
    """Render metadata and YAML configurations if present in leaf_dir."""
    yml_files = sorted(p for p in leaf_dir.glob("*.yml") if p.is_file() and p.name != "dir_structure.yml")
    if not yml_files:
        return

    with st.expander("📄 Experiment Configurations & Summaries", expanded=False):
        tabs = st.tabs([p.stem for p in yml_files])
        for tab, yml_path in zip(tabs, yml_files):
            with tab:
                st.caption(f"Path: `{yml_path}`")
                try:
                    with open(yml_path, "r", encoding="utf-8") as f:
                        content = yaml.safe_load(f)
                    st.json(content)
                except Exception:
                    with open(yml_path, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="yaml")


def main() -> None:
    st.set_page_config(
        page_title="INRS Bayes Adaptive Filters - Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    experiments_root = get_experiments_dir()
    if not experiments_root.is_dir():
        st.error(f"Experiments directory not found at: `{experiments_root}`")
        return

    exp_folders = find_experiment_folders(experiments_root)
    selected_exp, dir_struct = render_sidebar(exp_folders)

    if not selected_exp or not dir_struct:
        st.info("Please select an experiment group with a valid `dir_structure.yml` in the sidebar.")
        return

    # Extract chosen selections
    level_dict = st.session_state.level_selections.get(selected_exp, {})
    selections = [level_dict[lvl.name] for lvl in dir_struct.levels if lvl.name in level_dict]

    # Header / Breadcrumb
    render_breadcrumb(selected_exp, selections, dir_struct)
    st.markdown("---")

    # Resolve leaf folder
    leaf_dir = dir_struct.resolve_leaf_path(selections)
    has_figures, _ = dir_struct.validate_leaf(selections)

    if not has_figures:
        st.warning(f"No `figures/` folder found at: `{leaf_dir}`")
        render_experiment_metadata(leaf_dir)
        return

    figures = find_figures(leaf_dir)
    if not figures:
        st.info(f"No supported figures (`.pdf`, `.png`, etc.) found in `{leaf_dir / 'figures'}`.")
        render_experiment_metadata(leaf_dir)
        return

    # Zoomed or Gallery View
    zoomed_name = st.session_state.zoomed_figure
    zoomed_path = (leaf_dir / "figures" / zoomed_name) if zoomed_name else None

    if zoomed_path and zoomed_path.is_file():
        render_zoom_view(zoomed_path)
    else:
        # If previously zoomed figure doesn't exist in current directory, reset to gallery
        st.session_state.zoomed_figure = None
        render_gallery_view(figures, leaf_dir)

    # Metadata / summary configurations footer
    render_experiment_metadata(leaf_dir)


if __name__ == "__main__":
    main()
