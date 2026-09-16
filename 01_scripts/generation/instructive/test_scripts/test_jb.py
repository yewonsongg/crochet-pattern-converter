"""Regression checks for filled tapered join-below generation."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
from xml.etree.ElementTree import fromstring

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GenerationConfig
from generation.core.sampling import load_sampling_config
from generation.instructive.jb import generate_jb
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _sample(*, max_width_ratio=0.16, taper_extent=0.72):
  config = _config()
  sample = config.sample(
    "instructive",
    "jb",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "phenotype": "canonical",
      "max_width_ratio": max_width_ratio,
      "taper_extent": taper_extent,
      "stroke_width": 1.8,
    },
  )
  return config, sample


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_filled_tapered_geometry_and_fitting() -> None:
  config, sample = _sample()
  generation = GenerationConfig(
    canvas_width_px=120,
    canvas_height_px=80,
    target_visible_px=50.0,
  )
  generated = generate_jb(
    config.resolve("instructive", "jb"), sample, generation
  )
  metadata = generated.metadata
  root = fromstring(generated.svg)
  groups = [
    element for element in root.iter()
    if element.tag.rsplit("}", 1)[-1] == "g"
  ]
  paths = [element for element in root.iter() if element.tag.endswith("path")]
  assert len(groups) == 1
  assert groups[0].attrib["fill"] == "black"
  assert groups[0].attrib["stroke"] == "none"
  assert len(paths) == 1
  assert paths[0].attrib["d"].count("C") == 4
  assert paths[0].attrib["d"].endswith("Z")

  center = tuple(metadata["center_px"])
  top = tuple(metadata["top_tip_px"])
  bottom = tuple(metadata["bottom_tip_px"])
  left, right = (tuple(point) for point in metadata["center_extrema_px"])
  assert top == (center[0], center[1] - 25.0)
  assert bottom == (center[0], center[1] + 25.0)
  assert left[1] == right[1] == center[1]
  assert math.isclose(center[0] - left[0], right[0] - center[0], abs_tol=1e-10)
  assert math.isclose(metadata["height_px"], 50.0, abs_tol=1e-10)
  assert math.isclose(metadata["max_width_px"], 8.0, abs_tol=1e-10)
  assert metadata["rendered_bounds_px"] == [56.0, 15.0, 64.0, 65.0]
  assert metadata["stroke_width"] == 1.8
  assert metadata["stroke_width_applied"] is False
  assert generated.variant_id is None
  assert generated.sampled_parameters == sample.as_dict()
  assert generated.sampling_provenance is sample.provenance


def test_width_and_taper_semantics() -> None:
  generation = GenerationConfig(target_visible_px=60.0)
  config, narrow_sample = _sample(max_width_ratio=0.11, taper_extent=0.55)
  _, wide_sample = _sample(max_width_ratio=0.22, taper_extent=0.55)
  spec = config.resolve("instructive", "jb")
  narrow = generate_jb(spec, narrow_sample, generation).metadata
  wide = generate_jb(spec, wide_sample, generation).metadata
  assert narrow["height_px"] == wide["height_px"] == 60.0
  assert math.isclose(wide["max_width_px"], 2.0 * narrow["max_width_px"])
  assert narrow["top_tip_px"] == wide["top_tip_px"]
  assert narrow["bottom_tip_px"] == wide["bottom_tip_px"]

  _, full_sample = _sample(max_width_ratio=0.11, taper_extent=0.90)
  full = generate_jb(spec, full_sample, generation).metadata
  assert full["rendered_bounds_px"] == narrow["rendered_bounds_px"]
  assert full["center_extrema_px"] == narrow["center_extrema_px"]
  shallow_control = narrow["path_segments"][0]["control_1_px"]
  full_control = full["path_segments"][0]["control_1_px"]
  assert full_control[0] > shallow_control[0]
  assert full_control[1] == shallow_control[1]


def test_rotation_registry_and_determinism() -> None:
  config, sample = _sample(max_width_ratio=0.18, taper_extent=0.80)
  spec = config.resolve("instructive", "jb")
  assert GENERATOR_REGISTRY[("instructive", "jb")] is generate_jb
  assert ("instructive", "jb") not in COMPOSITE_GENERATORS
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=80,
    target_visible_px=60.0,
  )
  unrotated = generate_jb(spec, sample, generation)
  repeated = generate_jb(spec, sample, generation)
  rotated = generate_jb(spec, sample, replace(generation, rotation_deg=19.0))
  assert repeated.svg == unrotated.svg
  assert repeated.metadata == unrotated.metadata
  assert "rotate(19.0 50 50)" in rotated.svg
  assert rotated.metadata["top_tip_px"] == unrotated.metadata["top_tip_px"]
  assert rotated.metadata["path_segments"] == unrotated.metadata["path_segments"]
  assert rotated.metadata["rendered_bounds_px"] == unrotated.metadata["rendered_bounds_px"]


def test_invalid_inputs() -> None:
  config, sample = _sample()
  spec = config.resolve("instructive", "jb")
  generation = GenerationConfig()
  _expect_error(
    TypeError,
    "requires SampledParameters",
    lambda: generate_jb(spec, object(), generation),
  )
  _expect_error(
    ValueError,
    "instructive.jb class specification",
    lambda: generate_jb(config.resolve("primitive", "ch"), sample, generation),
  )
  _expect_error(
    ValueError,
    "Unsupported jb phenotype",
    lambda: generate_jb(
      spec,
      replace(sample, parameters={**sample.parameters, "phenotype": "other"}),
      generation,
    ),
  )
  old_parameters = dict(sample.parameters)
  old_parameters.pop("max_width_ratio")
  old_parameters.pop("taper_extent")
  old_parameters["aspect_ratio"] = 1.0
  _expect_error(
    ValueError,
    "max_width_ratio must be a finite number",
    lambda: generate_jb(
      spec, replace(sample, parameters=old_parameters), generation
    ),
  )
  for name, value, message in (
    ("max_width_ratio", 1.0, "range (0, 1)"),
    ("taper_extent", 0.0, "range (0, 1]"),
  ):
    _expect_error(
      ValueError,
      message,
      lambda name=name, value=value: generate_jb(
        spec,
        replace(sample, parameters={**sample.parameters, name: value}),
        generation,
      ),
    )
  _expect_error(
    ValueError,
    "target_visible_px must be positive",
    lambda: generate_jb(
      spec, sample, replace(generation, target_visible_px=0.0)
    ),
  )


def main() -> None:
  test_filled_tapered_geometry_and_fitting()
  test_width_and_taper_semantics()
  test_rotation_registry_and_determinism()
  test_invalid_inputs()
  print("Join-below generation checks passed.")


if __name__ == "__main__":
  main()
