import argparse
import math
import sys
from pathlib import Path

import filters_exp as fe

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "slow_filter_adjustment" / "config_files"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "slow_filter_adjustment" / "results"


class CustomArgumentParser(argparse.ArgumentParser):
  """Custom argument parser that prints full usage and a clear error message on invalid commands."""

  def error(self, message: str) -> None:
    sys.stderr.write(f"\n{'!' * 70}\n")
    sys.stderr.write(f"ERROR: Invalid command or argument: {message}\n")
    sys.stderr.write(f"{'!' * 70}\n\n")
    self.print_help(sys.stderr)
    sys.exit(2)


def parse_target(value: str) -> float:
  try:
    target = float(value)
  except ValueError:
    raise argparse.ArgumentTypeError(f"target must be a number in dB, got '{value}'")
  if not math.isfinite(target):
    raise argparse.ArgumentTypeError(f"target must be finite, got '{value}'")
  return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = CustomArgumentParser(
      description=(
          "Find the slow-filter epsilon whose steady-state MSD (dB) matches each target, "
          "using bisection in log scale.\n\n"
          "Every combination of the algorithm, environment and input .yml files found in\n"
          "<config-dir>/algorithms, <config-dir>/environments and <config-dir>/inputs is searched.\n\n"
          "Examples:\n"
          "  python run_slow_filter_adjustment.py -20 -25\n"
          "  python run_slow_filter_adjustment.py -30 --config-dir my_configs --output-dir my_results\n"
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "targets",
      nargs="+",
      type=parse_target,
      metavar="TARGET_DB",
      help="One or more desired steady-state MSD values in dB (finite numbers).",
  )
  parser.add_argument(
      "--config-dir",
      type=Path,
      default=DEFAULT_CONFIG_DIR,
      help=f"Folder with the algorithms/, environments/ and inputs/ .yml files (default: {DEFAULT_CONFIG_DIR}).",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=DEFAULT_OUTPUT_DIR,
      help=f"Folder where results are written (default: {DEFAULT_OUTPUT_DIR}).",
  )
  return parser.parse_args(argv)


def _format_scientific(value: float | None) -> str:
  """Two-decimal scientific notation with a plain exponent, e.g. 1.60e-7."""
  if value is None:
    return "None"
  mantissa, exponent = f"{value:.2e}".split("e")
  return f"{mantissa}e{int(exponent)}"


def _format_fixed(value: float | None) -> str:
  if value is None:
    return "None"
  return f"{value:.2f}"


def output_folder(output_dir: Path, search_config: dict, target_db: float) -> Path:
  names = search_config["names"]
  target_label = f"{target_db:g}dB"
  return output_dir / names["algorithm"] / names["environment"] / target_label / names["input"]


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  search_configs = fe.load_slow_filter_search_configs(args.config_dir)
  total = len(search_configs) * len(args.targets)

  run_index = 0
  for search_config in search_configs:
    for target_db in args.targets:
      run_index += 1
      names = search_config["names"]
      print(
          f"[{run_index}/{total}] {names['algorithm']} | {names['environment']} | "
          f"{names['input']} | target {target_db:g} dB",
          flush=True,
      )
      result = fe.bisection_search_epsilon(target_db, search_config)
      folder = output_folder(args.output_dir, search_config, target_db)
      fe.save_search_results(result, search_config, folder)
      fe.plot_epsilon_convergence(result, folder / "figures" / "convergence.pdf")
      evaluation = result["evaluation"] or {}
      min_slope = evaluation.get("min_slope")
      n_text = f"{result['N']:_}" if result["N"] is not None else "None"
      print(
          f"  N = {n_text} -- epsilon = {_format_scientific(result['epsilon'])} -- "
          f"min_slope = {_format_scientific(min_slope)} dB/iter -- "
          f"convergence_status = {result['status']} -- "
          f"steady_state_db = {_format_fixed(result['steady_state_db'])}",
          flush=True,
      )


if __name__ == "__main__":
  main()
