"""Regression checks for upper half-chain loop generation."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import fromstring

import numpy as np
from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GenerationConfig
from generation.core.rendering import render_png
from generation.core.sampling import load_sampling_config
from generation.instructive.loop import generate_loop
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _sample(*, aspect_ratio=1.0, curvature=0.5):
  config = _config()
  sample = config.sample(
    "instructive",
    "loop",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "aspect_ratio": aspect_ratio,
      "curvature": curvature,
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


def test_upper_arc_geometry_and_fitting() -> None:
  config, sample = _sample()
  generation = GenerationConfig(
    canvas_width_px=120,
    canvas_height_px=80,
    target_visible_px=50.0,
  )
  generated = generate_loop(
    config.resolve("instructive", "loop"), sample, generation
  )
  metadata = generated.metadata
  root = fromstring(generated.svg)
  paths = [element for element in root.iter() if element.tag.endswith("path")]
  assert len(paths) == 1
  assert paths[0].attrib["d"].count("C") == 2
  assert not [element for element in root.iter() if element.tag.endswith("ellipse")]

  left, right = (tuple(point) for point in metadata["endpoints_px"])
  apex = tuple(metadata["apex_px"])
  assert math.isclose(left[1], right[1], abs_tol=1e-9)
  assert apex[1] < left[1]
  assert math.isclose(apex[0], (left[0] + right[0]) / 2.0, abs_tol=1e-9)
  assert metadata["phenotype"] == "upper_half_chain"
  assert math.isclose(
    metadata["arc_height_px"], metadata["ellipse_height_px"] / 2.0,
    abs_tol=1e-9,
  )

  bounds = metadata["rendered_bounds_px"]
  assert math.isclose(
    max(bounds[2] - bounds[0], bounds[3] - bounds[1]),
    generation.target_visible_px,
    abs_tol=1e-8,
  )
  assert math.isclose((bounds[0] + bounds[2]) / 2.0, 60.0, abs_tol=1e-8)
  assert math.isclose((bounds[1] + bounds[3]) / 2.0, 40.0, abs_tol=1e-8)
  assert generated.variant_id is None
  assert generated.sampled_parameters == sample.as_dict()
  assert generated.sampling_provenance is sample.provenance


def test_aspect_ratio_and_curvature_semantics() -> None:
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  for aspect_ratio in (0.9, 1.0, 1.1):
    config, sample = _sample(aspect_ratio=aspect_ratio)
    generated = generate_loop(
      config.resolve("instructive", "loop"), sample, generation
    )
    metadata = generated.metadata
    assert math.isclose(
      metadata["ellipse_width_px"] / metadata["ellipse_height_px"],
      aspect_ratio,
      abs_tol=1e-9,
    )

  config, shallow_sample = _sample(curvature=0.4)
  _, round_sample = _sample(curvature=0.6)
  spec = config.resolve("instructive", "loop")
  shallow = generate_loop(spec, shallow_sample, generation).metadata
  rounded = generate_loop(spec, round_sample, generation).metadata
  assert shallow["endpoints_px"] == rounded["endpoints_px"]
  assert shallow["apex_px"] == rounded["apex_px"]
  assert shallow["centerline_bounds_px"] == rounded["centerline_bounds_px"]
  assert shallow["control_points_px"] != rounded["control_points_px"]


def test_rotation_rasterization_and_registry() -> None:
  config, sample = _sample(aspect_ratio=1.05, curvature=0.55)
  spec = config.resolve("instructive", "loop")
  assert GENERATOR_REGISTRY[("instructive", "loop")] is generate_loop
  assert ("instructive", "loop") not in COMPOSITE_GENERATORS
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  unrotated = generate_loop(spec, sample, generation)
  rotated = generate_loop(spec, sample, replace(generation, rotation_deg=19.0))
  assert "rotate(19.0 50 50)" in rotated.svg
  assert rotated.metadata["endpoints_px"] == unrotated.metadata["endpoints_px"]
  assert rotated.metadata["apex_px"] == unrotated.metadata["apex_px"]
  assert rotated.metadata["rendered_bounds_px"] == unrotated.metadata["rendered_bounds_px"]

  with TemporaryDirectory() as directory:
    output = Path(directory) / "loop.png"
    render_png(unrotated.svg, output)
    with Image.open(output) as image:
      raster_bounds = image.getbbox()
      assert raster_bounds is not None
      declared = unrotated.metadata["rendered_bounds_px"]
      assert raster_bounds[0] >= math.floor(declared[0]) - 1
      assert raster_bounds[1] >= math.floor(declared[1]) - 1
      assert raster_bounds[2] <= math.ceil(declared[2]) + 1
      assert raster_bounds[3] <= math.ceil(declared[3]) + 1


def test_invalid_inputs() -> None:
  config, sample = _sample()
  spec = config.resolve("instructive", "loop")
  generation = GenerationConfig()
  _expect_error(
    TypeError,
    "requires SampledParameters",
    lambda: generate_loop(spec, object(), generation),
  )
  _expect_error(
    ValueError,
    "instructive.loop class specification",
    lambda: generate_loop(config.resolve("primitive", "ch"), sample, generation),
  )
  _expect_error(
    ValueError,
    "aspect_ratio must be positive",
    lambda: generate_loop(
      spec,
      replace(sample, parameters={**sample.parameters, "aspect_ratio": 0.0}),
      generation,
    ),
  )
  _expect_error(
    ValueError,
    "curvature must be in the range",
    lambda: generate_loop(
      spec,
      replace(sample, parameters={**sample.parameters, "curvature": 1.0}),
      generation,
    ),
  )
  _expect_error(
    ValueError,
    "target_visible_px must exceed",
    lambda: generate_loop(
      spec, sample, replace(generation, target_visible_px=0.1)
    ),
  )


def main() -> None:
  test_upper_arc_geometry_and_fitting()
  test_aspect_ratio_and_curvature_semantics()
  test_rotation_rasterization_and_registry()
  test_invalid_inputs()
  print("Loop generation checks passed.")


if __name__ == "__main__":
  main()
