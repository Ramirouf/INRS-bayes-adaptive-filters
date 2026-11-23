"""Directory structure definition and navigation logic for experiment folders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class LevelSpec:
    """Specification for a hierarchy level."""
    name: str
    title: str

    @classmethod
    def from_entry(cls, entry: Any, index: int) -> LevelSpec:
        if isinstance(entry, dict):
            name = str(entry.get("name", f"level{index + 1:02d}"))
            title = str(entry.get("title", name.replace("_", " ").title()))
            return cls(name=name, title=title)
        if isinstance(entry, str):
            return cls(name=entry, title=entry.replace("_", " ").title())
        name = f"level{index + 1:02d}"
        return cls(name=name, title=f"Level {index + 1:02d}")


@dataclass
class DirStructure:
    """Parsed experiment directory structure."""
    experiment_dir: Path
    levels: list[LevelSpec] = field(default_factory=list)
    tree: dict[str, Any] = field(default_factory=dict)

    @property
    def level_names(self) -> list[str]:
        return [lvl.name for lvl in self.levels]

    @property
    def level_titles(self) -> list[str]:
        return [lvl.title for lvl in self.levels]

    def get_level_title(self, level_idx: int) -> str:
        if 0 <= level_idx < len(self.levels):
            return self.levels[level_idx].title
        return f"Level {level_idx + 1:02d}"

    def get_options_at_level(self, level_idx: int, current_selections: list[str]) -> list[str]:
        """Return available options at level_idx given prior selections for levels 0..level_idx-1.

        If the selected path cannot be fully traversed, falls back to the first available
        branch at each intermediate step.
        """
        if level_idx < 0 or level_idx >= len(self.levels):
            return []

        node = self.tree
        for i in range(level_idx):
            desired = current_selections[i] if i < len(current_selections) else None
            if isinstance(node, dict):
                if desired in node:
                    node = node[desired]
                elif node:
                    first_key = next(iter(node.keys()))
                    node = node[first_key]
                else:
                    return []
            elif isinstance(node, list):
                # Node is already a list of leaves, cannot traverse deeper
                return []
            else:
                return []

        if isinstance(node, dict):
            return list(str(k) for k in node.keys())
        if isinstance(node, list):
            return [str(item) for item in node]
        return []

    def get_all_options_for_level(self, level_idx: int) -> list[str]:
        """Return all distinct options appearing anywhere in the tree at the given level index."""
        if level_idx < 0 or level_idx >= len(self.levels):
            return []

        options: set[str] = set()

        def _traverse(current_node: Any, current_depth: int) -> None:
            if current_depth == level_idx:
                if isinstance(current_node, dict):
                    options.update(str(k) for k in current_node.keys())
                elif isinstance(current_node, list):
                    options.update(str(item) for item in current_node)
                return

            if isinstance(current_node, dict):
                for child in current_node.values():
                    _traverse(child, current_depth + 1)

        _traverse(self.tree, 0)
        return sorted(options)

    def resolve_leaf_path(self, selections: list[str]) -> Path:
        """Resolve a full filesystem path for the given sequence of selections."""
        path = self.experiment_dir
        for part in selections:
            path = path / part
        return path

    def validate_leaf(self, selections: list[str]) -> tuple[bool, Path]:
        """Check if the leaf directory exists and contains a 'figures' directory."""
        leaf = self.resolve_leaf_path(selections)
        figures_dir = leaf / "figures"
        return figures_dir.is_dir(), leaf


def load_dir_structure(experiment_dir_or_file: str | Path) -> DirStructure:
    """Load and parse dir_structure.yml from an experiment directory or file path."""
    target = Path(experiment_dir_or_file).resolve()
    if target.is_dir():
        yml_path = target / "dir_structure.yml"
        exp_dir = target
    else:
        yml_path = target
        exp_dir = target.parent

    if not yml_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yml_path}")

    with open(yml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    levels_raw = data.get("levels", [])
    tree = data.get("tree", {})

    # If levels are not explicitly declared, infer depth from tree
    if not levels_raw:
        depth = _infer_tree_depth(tree)
        levels = [LevelSpec(name=f"level{i + 1:02d}", title=f"Level {i + 1:02d}") for i in range(depth)]
    else:
        levels = [LevelSpec.from_entry(entry, idx) for idx, entry in enumerate(levels_raw)]

    return DirStructure(experiment_dir=exp_dir, levels=levels, tree=tree)


def _infer_tree_depth(node: Any) -> int:
    """Infer the maximum depth of a nested dict/list tree."""
    if isinstance(node, dict) and node:
        first_val = next(iter(node.values()))
        return 1 + _infer_tree_depth(first_val)
    if isinstance(node, list) and node:
        return 1
    return 0


def find_experiment_folders(experiments_root: str | Path) -> list[dict[str, Any]]:
    """Scan subdirectories directly under experiments_root.

    Returns a list of dicts with:
      - 'name': folder name
      - 'path': Path object
      - 'has_structure': bool whether dir_structure.yml exists
      - 'has_figures': bool whether any 'figures' folder exists in its subtree
    """
    root = Path(experiments_root).resolve()
    if not root.is_dir():
        return []

    results: list[dict[str, Any]] = []
    # Only direct children of experiments_root
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_", "__")):
            continue
        # Skip utility directories like tests or __pycache__
        if child.name in ("tests", "__pycache__"):
            continue

        has_structure = (child / "dir_structure.yml").is_file()

        # Check if there are any figures folders
        has_figures = any(p.name == "figures" and p.is_dir() for p in child.rglob("figures"))

        results.append({
            "name": child.name,
            "path": child,
            "has_structure": has_structure,
            "has_figures": has_figures,
        })

    return results
