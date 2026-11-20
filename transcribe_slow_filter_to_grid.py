import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
SEARCH_DIR = REPO_ROOT / "experiments" / "slow_filter_adjustment"
DEFAULT_RESULTS_DIR = SEARCH_DIR / "results"
DEFAULT_CONFIG_DIR = SEARCH_DIR / "config_files"
DEFAULT_GRID_DIR = REPO_ROOT / "experiments" / "grid_searches"

GRID_FOLDER_BY_ALGORITHM = {
    "sKF_algorithm": "skf_combination",
    "sKF_L_algorithm": "skf_l_combination",
}
N_GRID_FACTOR = 2
GRID_EPSILON_KEY = "epsilon_2"
ENVIRONMENT_KEYS_NOT_COPIED = ("behavior", "parameters_class")


def load_yaml(path: Path) -> dict:
  with open(path) as f:
    return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
  with open(path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def grid_target_label(target_db: float) -> str:
  return f"{target_db:g}dB"


def update_algorithm_config(path: Path, grid_config: dict, epsilon: float) -> None:
  config = copy.deepcopy(grid_config)
  config["performance_parameters"] = {GRID_EPSILON_KEY: epsilon, **config["performance_parameters"]}
  save_yaml(path, config)


def update_environment_config(path: Path, search_environment: dict, search_input: dict, grid_n: int) -> None:
  config = load_yaml(path)
  config["N"] = grid_n
  for source in (search_environment, search_input):
    for key, value in source.items():
      if key == "N" or key in ENVIRONMENT_KEYS_NOT_COPIED:
        continue
      if key in config:
        config[key] = value
  save_yaml(path, config)




def transcribe(results_dir: Path, config_dir: Path, grid_dir: Path) -> None:
  for algorithm, grid_name in GRID_FOLDER_BY_ALGORITHM.items():
    grid_folder = grid_dir / grid_name
    grid_config = load_yaml(grid_folder / "grid_config.yml")
    for optimal_path in sorted((results_dir / algorithm).rglob("optimal_epsilon.yml")):
      optimal = load_yaml(optimal_path)
      names = (optimal["algorithm"], optimal["environment"], optimal["input"])
      if optimal["status"] != "converged":
        print(f"skipped (status={optimal['status']}): {'/'.join(names)} {optimal['target_db']:g} dB")
        continue

      target_folder = grid_folder / optimal["environment"] / grid_target_label(optimal["target_db"]) / optimal["input"]
      if not target_folder.is_dir():
        raise FileNotFoundError(f"grid folder not found: {target_folder}")

      search_environment = load_yaml(config_dir / "environments" / f"{optimal['environment']}.yml")
      search_input = load_yaml(config_dir / "inputs" / f"{optimal['input']}.yml")

      update_algorithm_config(target_folder / "algorithm_config.yml", grid_config, optimal["optimal_value"])
      update_environment_config(
          target_folder / "environment_config.yml", search_environment, search_input, N_GRID_FACTOR * optimal["N"]
      )
      print(f"updated: {target_folder.relative_to(grid_dir)}")



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Copy slow-filter epsilon and N search results into the grid_searches config files."
  )
  parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
  parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
  parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  transcribe(args.results_dir, args.config_dir, args.grid_dir)


if __name__ == "__main__":
  main()
