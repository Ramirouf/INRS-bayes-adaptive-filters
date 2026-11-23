"""Generator tool to scan experiment directories and build dir_structure.yml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import yaml


def find_leaf_directories(root_dir: str | Path) -> list[Path]:
    """Find all leaf experiment directories under root_dir.

    A directory is considered a leaf if:
      - It has a 'figures' subdirectory (if any 'figures' directory exists under root), OR
      - It has no subdirectories (ignoring hidden dirs) when no 'figures' directory exists.
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    # Check if any 'figures' directories exist under root
    all_figures_dirs = [p for p in root.rglob("figures") if p.is_dir()]
    if all_figures_dirs:
        # Leaves are the parent directories of each 'figures' directory
        leaves = [p.parent for p in all_figures_dirs]
        return sorted(set(leaves))

    leaves: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if path.name.startswith((".", "_")):
            continue

        subdirs = [
            d for d in path.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
        if len(subdirs) == 0:
            leaves.append(path)

    filtered_leaves: list[Path] = []
    for leaf in leaves:
        is_parent_of_another = any(other != leaf and leaf in other.parents for other in leaves)
        if not is_parent_of_another:
            filtered_leaves.append(leaf)

    return sorted(filtered_leaves)


def build_tree_structure(
    root_dir: Path,
    leaf_dirs: list[Path],
) -> tuple[int, dict[str, Any]]:
    """Build a nested dictionary/list tree from leaf directory paths relative to root_dir.

    Returns (max_depth, tree_dict).
    """
    rel_paths = [leaf.relative_to(root_dir).parts for leaf in leaf_dirs]
    if not rel_paths:
        return 0, {}

    max_depth = max(len(p) for p in rel_paths)

    # Recursive insertion into nested dict
    tree: dict[str, Any] = {}
    for parts in rel_paths:
        curr = tree
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if is_last:
                # If last, we can store empty dict or marker
                if part not in curr:
                    curr[part] = None
            else:
                if part not in curr or not isinstance(curr[part], dict):
                    curr[part] = {}
                curr = curr[part]

    # Clean up leaves: if a dict contains only keys with value None, convert to list of keys
    def _clean_leaves(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if all(v is None for v in node.values()):
            return sorted(node.keys())
        return {k: _clean_leaves(v) for k, v in sorted(node.items())}

    cleaned_tree = _clean_leaves(tree)
    return max_depth, cleaned_tree


def scan_directory_tree(
    root_dir: str | Path,
    level_names: list[str] | None = None,
) -> dict[str, Any]:
    """Scan root_dir, find leaf directories, and build the full dir_structure dict."""
    root = Path(root_dir).resolve()
    leaf_dirs = find_leaf_directories(root)

    if not leaf_dirs:
        raise ValueError(f"No leaf experiment directories found in: {root}")

    depth, tree = build_tree_structure(root, leaf_dirs)

    # Build levels specification
    levels: list[dict[str, str]] = []
    for i in range(depth):
        if level_names and i < len(level_names):
            name = level_names[i]
            title = name.replace("_", " ").title()
        else:
            name = f"level{i + 1:02d}"
            title = f"Level {i + 1:02d}"
        levels.append({"name": name, "title": title})

    return {
        "levels": levels,
        "tree": tree,
    }


def generate_dir_structure_yml(
    root_dir: str | Path,
    level_names: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and write dir_structure.yml for the specified root directory."""
    root = Path(root_dir).resolve()
    target_file = Path(output_path).resolve() if output_path else root / "dir_structure.yml"

    data = scan_directory_tree(root, level_names=level_names)

    header = (
        "# Directory Structure Configuration for Experiments Dashboard\n"
        "# Generated automatically. You can customize the level names and titles below.\n\n"
    )

    yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(header + yaml_str)

    return target_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dir_structure.yml for an experiment directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the experiment folder (e.g. experiments/grid_searches)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of level names (e.g. algorithms,environments,target_msds,inputs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom output file path for dir_structure.yml",
    )

    args = parser.parse_args()

    levels_list = [lvl.strip() for lvl in args.levels.split(",")] if args.levels else None

    try:
        out = generate_dir_structure_yml(args.directory, level_names=levels_list, output_path=args.output)
        print(f"✓ Successfully generated: {out}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
