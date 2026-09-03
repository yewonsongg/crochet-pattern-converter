"""Check that every configured symbol class passes through sampling."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_ROOT.parents[3]
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.sampling import load_sampling_config


def main() -> None:
  """Load the pipeline and sample each ontology class once."""

  config = load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )

  class_keys = config.class_keys
  assert len(class_keys) == 16, f"Expected 16 classes, found {len(class_keys)}."

  print(f"Sweeping {len(class_keys)} sampling classes...")
  for index, (class_group, class_name) in enumerate(class_keys):
    seed = 1234 + index
    config.sample(
      class_group,
      class_name,
      np.random.default_rng(seed),
      seed=seed,
    )
    print(f"PASS  {class_group}.{class_name}")

  print("All 16 classes passed through the sampling pipeline.")


if __name__ == "__main__":
  main()
