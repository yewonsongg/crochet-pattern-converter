"""Regression checks for joined-top compound stitch generation."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import Element, fromstring

import numpy as np
from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.compound.fan import symmetric_fan_layout
from generation.compound.together import generate_together
from generation.core.models import CompositeSample, GenerationConfig
from generation.core.rendering import render_png
from generation.core.sampling import load_sampling_config
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _composite(stitch: str, count: int = 3) -> tuple[object, CompositeSample]:
  config = _config()
  parent = config.sample(
    "compound",
    "together",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "stitch": stitch,
      "count": count,
      "spread_angle_deg": 24.0,
      "connector_length": 0.0 if stitch == "sc" else 0.2,
      "stroke_width": 1.8,
    },
  )
  return config.resolve("compound", "together"), config.realize_components(
    "compound", "together", parent, np.random.default_rng(202)
  )


def _lines(svg: str) -> list[Element]:
  root = fromstring(svg)
  return [element for element in root.iter() if element.tag.endswith("line")]


def _line_pixels(element: Element, config: GenerationConfig):
  scale = min(config.canvas_width_px, config.canvas_height_px) / 100.0
  offset_x = (config.canvas_width_px - 100.0 * scale) / 2.0
  offset_y = (config.canvas_height_px - 100.0 * scale) / 2.0

  def point(x_name: str, y_name: str):
    return (
      float(element.attrib[x_name]) * scale + offset_x,
      float(element.attrib[y_name]) * scale + offset_y,
    )

  return point("x1", "y1"), point("x2", "y2")


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_shared_fan_layout() -> None:
  top_joined = symmetric_fan_layout(
    count=4,
    spread_angle_deg=30.0,
    stem_length_px=20.0,
    join_px=(50.0, 20.0),
    joined_at="top",
  )
  assert top_joined.axis_angles_deg == (15.0, 5.0, -5.0, -15.0)
  assert all(placement.top_px == (50.0, 20.0) for placement in top_joined.placements)
  assert all(
    math.isclose(
      math.dist(placement.base_px, placement.top_px), 20.0, abs_tol=1e-12
    )
    for placement in top_joined.placements
  )
  assert [placement.base_px[0] for placement in top_joined.placements] == sorted(
    placement.base_px[0] for placement in top_joined.placements
  )

  base_joined = symmetric_fan_layout(
    count=2,
    spread_angle_deg=20.0,
    stem_length_px=10.0,
    join_px=(12.0, 30.0),
    joined_at="base",
  )
  assert all(placement.base_px == (12.0, 30.0) for placement in base_joined.placements)
  assert base_joined.placements[0].top_px[0] < 12.0
  assert base_joined.placements[1].top_px[0] > 12.0


def test_all_stitches_and_counts() -> None:
  crossbars = {"sc": 1, "hdc": 0, "dc": 1, "tr": 2, "dtr": 3}
  for stitch in crossbars:
    for count in (2, 3, 6):
      spec, composite = _composite(stitch, count)
      snapshot = asdict(composite)
      config = GenerationConfig(
        canvas_width_px=120,
        canvas_height_px=80,
        target_visible_px=50.0,
      )
      generated = generate_together(spec, composite, config)
      repeated = generate_together(spec, composite, config)
      metadata = generated.metadata

      assert asdict(composite) == snapshot
      assert repeated.svg == generated.svg
      assert repeated.metadata == generated.metadata
      assert generated.variant_id is None
      assert generated.sampled_parameters == composite.parent.as_dict()
      assert generated.sampling_provenance is composite.parent.provenance
      assert metadata["axis_angles_deg"][0] == 12.0
      assert metadata["axis_angles_deg"][-1] == -12.0
      assert len(metadata["placements"]) == count
      tops = [tuple(item["top_px"]) for item in metadata["placements"]]
      assert all(top == tops[0] for top in tops)
      assert all(
        math.isclose(
          math.dist(tuple(item["base_px"]), tuple(item["top_px"])),
          metadata["stem_length_px"],
          abs_tol=1e-8,
        )
        for item in metadata["placements"]
      )

      connector_count = 0 if stitch == "sc" else 1
      per_stitch_lines = 2 if stitch == "sc" else 1 + crossbars[stitch]
      assert len(_lines(generated.svg)) == count * per_stitch_lines + connector_count
      assert metadata["component_prototype"]["occurrence_count"] == count
      assert metadata["component_prototype"]["class_name"] == stitch
      assert metadata["component_prototype"]["sampled_parameters"] == (
        composite.prototypes["stitch"].sample.as_dict()
      )
      assert metadata["component_prototype"]["sampling_provenance"]["seed"] == (
        composite.prototypes["stitch"].sample.provenance.seed
      )

      bounds = metadata["rendered_bounds_px"]
      assert math.isclose(
        max(bounds[2] - bounds[0], bounds[3] - bounds[1]),
        config.target_visible_px,
        abs_tol=1e-8,
      )
      assert math.isclose((bounds[0] + bounds[2]) / 2.0, 60.0, abs_tol=1e-8)
      assert math.isclose((bounds[1] + bounds[3]) / 2.0, 40.0, abs_tol=1e-8)

      if stitch == "sc":
        assert metadata["connector_type"] == "none"
        assert metadata["connector_length_px"] == 0.0
      else:
        assert metadata["connector_type"] == "bar"
        assert math.isclose(
          metadata["connector_length_px"],
          0.2 * metadata["stem_length_px"],
          abs_tol=1e-10,
        )
        connector_start, connector_end = _line_pixels(_lines(generated.svg)[-1], config)
        assert math.isclose(connector_start[1], connector_end[1], abs_tol=1e-8)
        assert math.isclose(
          math.dist(connector_start, connector_end),
          metadata["connector_length_px"],
          abs_tol=1e-7,
        )


def test_rotation_rasterization_and_registry() -> None:
  assert GENERATOR_REGISTRY[("compound", "together")] is generate_together
  assert ("compound", "together") in COMPOSITE_GENERATORS
  spec, composite = _composite("dc", 3)
  config = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
    rotation_deg=19.0,
  )
  generated = generate_together(spec, composite, config)
  unrotated = generate_together(
    spec,
    composite,
    replace(config, rotation_deg=0.0),
  )
  assert "rotate(19.0 50 50)" in generated.svg
  assert generated.metadata["visual_rotation_deg"] == 19.0
  assert generated.metadata["placements"] == unrotated.metadata["placements"]
  assert generated.metadata["centerline_bounds_px"] == (
    unrotated.metadata["centerline_bounds_px"]
  )
  assert generated.metadata["rendered_bounds_px"] == (
    unrotated.metadata["rendered_bounds_px"]
  )

  with TemporaryDirectory() as directory:
    output = Path(directory) / "together.png"
    render_png(unrotated.svg, output)
    with Image.open(output) as image:
      raster_bounds = image.getbbox()
      assert raster_bounds is not None
      declared = unrotated.metadata["rendered_bounds_px"]
      assert raster_bounds[0] >= math.floor(declared[0]) - 1
      assert raster_bounds[1] >= math.floor(declared[1]) - 1
      assert raster_bounds[2] <= math.ceil(declared[2]) + 1
      assert raster_bounds[3] <= math.ceil(declared[3]) + 1


def test_crossbars_rotate_with_reused_prototype() -> None:
  spec, composite = _composite("dc", 3)
  config = GenerationConfig(canvas_width_px=120, canvas_height_px=80)
  generated = generate_together(spec, composite, config)
  lines = _lines(generated.svg)
  expected_angle = composite.prototypes["stitch"].sample.parameters[
    "cross_bar_angle_deg"
  ]

  for index, placement in enumerate(generated.metadata["placements"]):
    base = tuple(placement["base_px"])
    top = tuple(placement["top_px"])
    stem_length = math.dist(base, top)
    axis = ((top[0] - base[0]) / stem_length, (top[1] - base[1]) / stem_length)
    perpendicular = (-axis[1], axis[0])
    cross_start, cross_end = _line_pixels(lines[index * 2 + 1], config)
    cross_length = math.dist(cross_start, cross_end)
    direction = (
      (cross_end[0] - cross_start[0]) / cross_length,
      (cross_end[1] - cross_start[1]) / cross_length,
    )
    actual_angle = math.degrees(math.atan2(
      direction[0] * axis[0] + direction[1] * axis[1],
      direction[0] * perpendicular[0] + direction[1] * perpendicular[1],
    ))
    assert math.isclose(actual_angle, expected_angle, abs_tol=1e-6)


def test_invalid_inputs() -> None:
  spec, composite = _composite("dc", 3)
  config = GenerationConfig()
  prototype = composite.prototypes["stitch"]

  _expect_error(
    TypeError,
    "requires a realized CompositeSample",
    lambda: generate_together(spec, composite.parent, config),
  )
  _expect_error(
    ValueError,
    "compound.together class specification",
    lambda: generate_together(_config().resolve("primitive", "dc"), composite, config),
  )
  _expect_error(
    ValueError,
    "exactly one 'stitch'",
    lambda: generate_together(spec, replace(composite, prototypes={}), config),
  )
  _expect_error(
    ValueError,
    "occurrence_count must equal count",
    lambda: generate_together(
      spec,
      replace(composite, prototypes={"stitch": replace(prototype, occurrence_count=2)}),
      config,
    ),
  )
  _expect_error(
    ValueError,
    "sampled primitive stitch class",
    lambda: generate_together(
      spec,
      replace(composite, prototypes={"stitch": replace(prototype, class_name="tr")}),
      config,
    ),
  )
  _expect_error(
    ValueError,
    "inherit the parent stroke_width",
    lambda: generate_together(
      spec,
      replace(
        composite,
        prototypes={
          "stitch": replace(prototype, inherited_parameters={"stroke_width": 2.0})
        },
      ),
      config,
    ),
  )
  invalid_topology = replace(
    composite.parent,
    topology={**composite.parent.topology, "top_relation": "separate"},
  )
  _expect_error(
    ValueError,
    "top_relation must be 'joined'",
    lambda: generate_together(spec, replace(composite, parent=invalid_topology), config),
  )
  invalid_connector = replace(
    composite.parent,
    parameters={**composite.parent.parameters, "connector_type": "none"},
  )
  _expect_error(
    ValueError,
    "require connector_type='bar'",
    lambda: generate_together(spec, replace(composite, parent=invalid_connector), config),
  )
  malformed_child = replace(
    prototype.sample,
    parameters={
      key: value
      for key, value in prototype.sample.parameters.items()
      if key != "bar_stem_ratio"
    },
  )
  _expect_error(
    ValueError,
    "bar_stem_ratio must be a finite number",
    lambda: generate_together(
      spec,
      replace(
        composite,
        prototypes={"stitch": replace(prototype, sample=malformed_child)},
      ),
      config,
    ),
  )
  _expect_error(
    ValueError,
    "must exceed its rendered stroke width",
    lambda: generate_together(
      spec,
      composite,
      GenerationConfig(target_visible_px=0.1),
    ),
  )


def main() -> None:
  test_shared_fan_layout()
  test_all_stitches_and_counts()
  test_rotation_rasterization_and_registry()
  test_crossbars_rotate_with_reused_prototype()
  test_invalid_inputs()
  print("Together compound generation checks passed.")


if __name__ == "__main__":
  main()
