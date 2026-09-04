"""Regression checks for explicit sampling overrides."""

from pathlib import Path
import sys

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
if str(PROJECT_ROOT / "01_scripts") not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT / "01_scripts"))

from generation.core.sampling import load_sampling_config


def main() -> None:
  config = load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )
  sample = config.sample(
    "instructive",
    "ring",
    np.random.default_rng(7),
    seed=7,
    overrides={"variant": "chain", "count": 6, "chain_pitch": 3.2},
  )
  assert sample.parameters["variant"] == "chain"
  assert sample.parameters["count"] == 6
  assert sample.parameters["chain_pitch"] == 3.2
  assert sample.derived["diameter"] == 6 * 3.2 / np.pi
  assert sample.provenance.overrides == {"variant": "chain", "count": 6, "chain_pitch": 3.2}

  try:
    config.sample(
      "instructive", "ring", np.random.default_rng(7),
      overrides={"variant": "chain", "count": 99},
    )
  except ValueError:
    pass
  else:
    raise AssertionError("Out-of-range categorical override was accepted")

  print("Explicit override checks passed.")


if __name__ == "__main__":
  main()
