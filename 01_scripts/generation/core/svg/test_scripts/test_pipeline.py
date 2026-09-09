"""Minimal sampling-to-SVG checks; excludes labeling, rasterization, and I/O."""

from __future__ import annotations

from pathlib import Path
import sys
from xml.etree.ElementTree import fromstring

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GeneratedObject, GenerationConfig
from generation.core.sampling import load_sampling_config
from generation.registry import GENERATOR_REGISTRY


def main() -> None:
  config = load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )
  for index, ((class_group, class_name), generator) in enumerate(GENERATOR_REGISTRY.items()):
    spec = config.resolve(class_group, class_name)
    sample = config.sample(class_group, class_name, np.random.default_rng(1234 + index), seed=1234 + index)
    generated = generator(spec, sample, GenerationConfig())
    assert isinstance(generated, GeneratedObject)
    assert generated.svg
    fromstring(generated.svg)

  spec = config.resolve("primitive", "slst")
  generator = GENERATOR_REGISTRY[("primitive", "slst")]
  for index, overrides in enumerate((
    {"shape": "oval", "aspect_ratio": 1.15},
    {"shape": "circle", "aspect_ratio": 1.04},
  )):
    sample = config.sample("primitive", "slst", np.random.default_rng(2000 + index), seed=2000 + index, overrides=overrides)
    generated = generator(spec, sample, GenerationConfig(rotation_deg=20.0))
    assert generated.metadata["shape"] == overrides["shape"]
    assert generated.metadata["aspect_ratio"] == overrides["aspect_ratio"]
    assert "<ellipse" in generated.svg
    assert "rotate(20.0 50 50)" in generated.svg
    assert generated.obb_pixels is None
    assert generated.yolo_label is None
    fromstring(generated.svg)

  spec = config.resolve("primitive", "sc")
  generator = GENERATOR_REGISTRY[("primitive", "sc")]
  for index, overrides in enumerate((
    {"shape": "symmetric", "asymmetry": 0.0},
    {"shape": "asymmetric", "asymmetry": 0.18},
  )):
    sample = config.sample(
      "primitive", "sc", np.random.default_rng(3000 + index), seed=3000 + index,
      overrides=overrides,
    )
    generated = generator(spec, sample, GenerationConfig())
    assert generated.metadata["shape"] == overrides["shape"]
    assert generated.metadata["asymmetry"] == overrides["asymmetry"]
    assert generated.svg.count("<line") == 2
    assert generated.obb_pixels is None
    assert generated.yolo_label is None
    fromstring(generated.svg)

  spec = config.resolve("primitive", "hdc")
  generator = GENERATOR_REGISTRY[("primitive", "hdc")]
  sample = config.sample(
    "primitive", "hdc", np.random.default_rng(4000), seed=4000,
    overrides={"bar_stem_ratio": 0.5},
  )
  generated = generator(spec, sample, GenerationConfig())
  assert generated.metadata["bar_stem_ratio"] == 0.5
  assert generated.svg.count("<line") == 2
  assert generated.obb_pixels is None
  assert generated.yolo_label is None
  fromstring(generated.svg)

  spec = config.resolve("primitive", "dc")
  generator = GENERATOR_REGISTRY[("primitive", "dc")]
  sample = config.sample(
    "primitive", "dc", np.random.default_rng(5000), seed=5000,
    overrides={"bar_stem_ratio": 0.333, "cross_bar_ratio": 0.36, "cross_bar_y": 0.50, "cross_bar_angle_deg": 15.0},
  )
  generated = generator(spec, sample, GenerationConfig())
  assert generated.metadata["cross_bar_ratio"] == 0.36
  assert generated.metadata["cross_bar_y"] == 0.50
  assert generated.metadata["cross_bar_angle_deg"] == 15.0
  assert generated.svg.count("<line") == 3
  assert generated.obb_pixels is None
  assert generated.yolo_label is None
  fromstring(generated.svg)

  for class_name, expected_count, seed in (("tr", 4, 6000), ("dtr", 5, 7000)):
    spec = config.resolve("primitive", class_name)
    generator = GENERATOR_REGISTRY[("primitive", class_name)]
    sample = config.sample(
      "primitive", class_name, np.random.default_rng(seed), seed=seed,
      overrides={
        "bar_stem_ratio": 0.333,
        "cross_bar_ratio": 0.36,
        "cross_bar_y": 0.50,
        "cross_bar_angle_deg": -10.0,
      },
    )
    generated = generator(spec, sample, GenerationConfig())
    assert generated.svg.count("<line") == expected_count
    assert generated.metadata["cross_bar_count"] == expected_count - 2
    assert generated.obb_pixels is None
    assert generated.yolo_label is None
    fromstring(generated.svg)
  print("Sampling-to-SVG pipeline checks passed.")


if __name__ == "__main__":
  main()
