"""Print sampling-pipeline snapshots for one parameterized class."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPT_ROOT))

from generation.core.sampling import load_sampling_config


def print_snapshot(title: str, value) -> None:
  print(f"\n=== {title} ===")
  if hasattr(value, "__dict__"):
    value = value.__dict__
  print(json.dumps(value, indent=2, default=str))


def main() -> None:
  ontology_path = PROJECT_ROOT / "06_configs" / "ontology.yaml"
  sampling_path = PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml"
  class_group = "instructive"
  class_name = "ring"
  seed = 1234

  config = load_sampling_config(
    ontology_path=ontology_path,
    sampling_path=sampling_path,
  )
  # print_snapshot("1. loaded config", {
  #   "ontology_path": config.identity.ontology_path,
  #   "ontology_digest": config.identity.ontology_digest,
  #   "sampling_path": config.identity.sampling_path,
  #   "sampling_digest": config.identity.sampling_digest,
  #   "schema_version": config.sampling_config["schema"].get("version"),
  # })

  spec = config.resolve(class_group, class_name)
  # print_snapshot("2. resolved class spec", spec)
  
  rng = np.random.default_rng(seed)
  sample = config.sample(class_group, class_name, rng, seed=seed)
  # print_snapshot("3. sampled parameters", sample.parameters)
  # print_snapshot("4. derived parameters", sample.derived)
  # print_snapshot("5. sampling provenance", sample.provenance)


if __name__ == "__main__":
  main()
