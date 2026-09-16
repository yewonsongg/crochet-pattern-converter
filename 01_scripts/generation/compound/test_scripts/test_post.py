"""Regression checks for front- and back-post generation."""

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

from generation.compound.post import generate_post
from generation.core.models import GenerationConfig
from generation.core.rendering import render_png
from generation.core.sampling import load_sampling_config
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _parent_and_composite(
  stitch: str,
  post_type: str,
  variant: str,
  *,
  hook_curvature: float | None = None,
):
  config = _config()
  curvature = (
    hook_curvature
    if hook_curvature is not None
    else (0.15 if variant == "hook" else 0.25)
  )
  parent = config.sample(
    "compound",
    "post",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "post_type": post_type,
      "variant": variant,
      "stitch": stitch,
      "hook_curvature": curvature,
      "stroke_width": 1.8,
    },
  )
  composite = config.realize_components(
    "compound", "post", parent, np.random.default_rng(202)
  )
  return config, parent, composite


def _elements(svg: str, suffix: str) -> list[Element]:
  root = fromstring(svg)
  return [element for element in root.iter() if element.tag.endswith(suffix)]


def _path_points(metadata) -> list[tuple[float, float]]:
  result = [tuple(metadata["path_start_px"])]
  for segment in metadata["path_segments"]:
    result.extend((
      tuple(segment["control_1_px"]),
      tuple(segment["control_2_px"]),
      tuple(segment["end_px"]),
    ))
  return result


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_all_stitches_post_types_and_variants() -> None:
  line_counts = {"sc": 2, "hdc": 2, "dc": 3, "tr": 4, "dtr": 5}
  generation = GenerationConfig(
    canvas_width_px=120,
    canvas_height_px=80,
    target_visible_px=50.0,
  )
  for stitch, line_count in line_counts.items():
    for post_type in ("front", "back"):
      for variant in ("hook", "j_shape"):
        config, parent, composite = _parent_and_composite(
          stitch, post_type, variant
        )
        snapshot = asdict(composite)
        spec = config.resolve("compound", "post")
        generated = generate_post(spec, composite, generation)
        repeated = generate_post(spec, composite, generation)
        metadata = generated.metadata

        assert asdict(composite) == snapshot
        assert repeated.svg == generated.svg
        assert repeated.metadata == metadata
        assert generated.variant_id == variant
        assert generated.sampled_parameters == parent.as_dict()
        assert generated.sampling_provenance is parent.provenance
        assert len(_elements(generated.svg, "line")) == line_count
        assert len(_elements(generated.svg, "path")) == 1
        drawing_group = next(
          element for element in fromstring(generated.svg)
          if element.tag.endswith("g")
        )
        assert drawing_group[-1].tag.endswith("path")
        assert metadata["attachment_px"] == metadata["stitch_placement"]["base_px"]
        assert metadata["path_start_px"] == metadata["attachment_px"]
        assert metadata["opening_side"] == (
          "left" if post_type == "front" else "right"
        )
        assert metadata["component_prototype"]["occurrence_count"] is None
        assert metadata["component_prototype"]["arrangement"] is None

        bounds = metadata["rendered_bounds_px"]
        assert math.isclose(
          max(bounds[2] - bounds[0], bounds[3] - bounds[1]),
          generation.target_visible_px,
          abs_tol=1e-8,
        )
        assert math.isclose((bounds[0] + bounds[2]) / 2.0, 60.0, abs_tol=1e-8)
        assert math.isclose((bounds[1] + bounds[3]) / 2.0, 40.0, abs_tol=1e-8)
        assert math.isclose(
          metadata["hook_radius_px"] / metadata["stem_length_px"],
          parent.parameters["hook_curvature"],
          abs_tol=1e-10,
        )


def test_front_and_back_are_mirrors() -> None:
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  for variant in ("hook", "j_shape"):
    config, _, front_composite = _parent_and_composite("dc", "front", variant)
    _, _, back_composite = _parent_and_composite("dc", "back", variant)
    spec = config.resolve("compound", "post")
    front = generate_post(spec, front_composite, generation).metadata
    back = generate_post(spec, back_composite, generation).metadata
    front_attachment = tuple(front["attachment_px"])
    back_attachment = tuple(back["attachment_px"])
    assert math.isclose(
      front_attachment[0] + back_attachment[0],
      generation.canvas_width_px,
      abs_tol=1e-9,
    )
    assert math.isclose(front_attachment[1], back_attachment[1], abs_tol=1e-9)
    for front_point, back_point in zip(
      _path_points(front), _path_points(back), strict=True
    ):
      assert math.isclose(front_point[1], back_point[1], abs_tol=1e-9)
      assert math.isclose(
        front_point[0] - front_attachment[0],
        -(back_point[0] - back_attachment[0]),
        abs_tol=1e-9,
      )


def test_hook_and_j_shape_contracts() -> None:
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  for post_type, terminal_sign, opening in (
    ("back", 1.0, [0.0, 90.0]),
    ("front", -1.0, [90.0, 180.0]),
  ):
    config, _, composite = _parent_and_composite("hdc", post_type, "hook")
    hook = generate_post(
      config.resolve("compound", "post"), composite, generation
    ).metadata
    attachment = tuple(hook["attachment_px"])
    radius = hook["hook_radius_px"]
    center = tuple(hook["circle_center_px"])
    end = tuple(hook["path_end_px"])
    assert len(hook["path_segments"]) == 3
    assert hook["attachment_angle_deg"] == 90.0
    assert hook["opening_angle_interval_deg"] == opening
    assert math.isclose(center[0], attachment[0], abs_tol=1e-9)
    assert math.isclose(center[1], attachment[1] + radius, abs_tol=1e-9)
    assert math.isclose(end[0], attachment[0] + terminal_sign * radius, abs_tol=1e-9)
    assert math.isclose(end[1], attachment[1] + radius, abs_tol=1e-9)
    assert math.isclose(hook["hook_width_px"], 2.0 * radius, abs_tol=1e-8)
    assert math.isclose(hook["hook_height_px"], 2.0 * radius, abs_tol=1e-8)

    _, _, composite = _parent_and_composite("hdc", post_type, "j_shape")
    j_shape = generate_post(
      config.resolve("compound", "post"), composite, generation
    ).metadata
    attachment = tuple(j_shape["attachment_px"])
    radius = j_shape["hook_radius_px"]
    first = j_shape["path_segments"][0]
    second = j_shape["path_segments"][1]
    terminal = tuple(j_shape["path_end_px"])
    assert len(j_shape["path_segments"]) == 2
    assert first["control_1_px"][0] == attachment[0]
    incoming = (
      first["end_px"][0] - first["control_2_px"][0],
      first["end_px"][1] - first["control_2_px"][1],
    )
    outgoing = (
      second["control_1_px"][0] - first["end_px"][0],
      second["control_1_px"][1] - first["end_px"][1],
    )
    assert math.isclose(incoming[1], 0.0, abs_tol=1e-9)
    assert math.isclose(outgoing[1], 0.0, abs_tol=1e-9)
    assert terminal_sign * (terminal[0] - attachment[0]) > 0.0
    assert math.isclose(
      abs(terminal[0] - attachment[0]), 1.15 * radius, abs_tol=1e-8
    )


def test_rotation_rasterization_and_registry() -> None:
  config, _, composite = _parent_and_composite("tr", "back", "j_shape")
  spec = config.resolve("compound", "post")
  assert GENERATOR_REGISTRY[("compound", "post")] is generate_post
  assert ("compound", "post") in COMPOSITE_GENERATORS
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  unrotated = generate_post(spec, composite, generation)
  rotated = generate_post(spec, composite, replace(generation, rotation_deg=21.0))
  assert "rotate(21.0 50 50)" in rotated.svg
  assert rotated.metadata["stitch_placement"] == unrotated.metadata["stitch_placement"]
  assert rotated.metadata["path_segments"] == unrotated.metadata["path_segments"]
  assert rotated.metadata["rendered_bounds_px"] == unrotated.metadata["rendered_bounds_px"]

  with TemporaryDirectory() as directory:
    output = Path(directory) / "post.png"
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
  config, parent, composite = _parent_and_composite("dc", "back", "hook")
  spec = config.resolve("compound", "post")
  generation = GenerationConfig()
  stitch = composite.prototypes["stitch"]
  _expect_error(
    TypeError,
    "requires a realized CompositeSample",
    lambda: generate_post(spec, parent, generation),
  )
  _expect_error(
    ValueError,
    "compound.post class specification",
    lambda: generate_post(config.resolve("primitive", "dc"), composite, generation),
  )
  _expect_error(
    ValueError,
    "exactly one 'stitch'",
    lambda: generate_post(spec, replace(composite, prototypes={}), generation),
  )
  _expect_error(
    ValueError,
    "occurrence_count must be unspecified",
    lambda: generate_post(
      spec,
      replace(composite, prototypes={"stitch": replace(stitch, occurrence_count=1)}),
      generation,
    ),
  )
  invalid_curvature = replace(
    composite.parent,
    parameters={**composite.parent.parameters, "hook_curvature": 0.0},
  )
  _expect_error(
    ValueError,
    "hook_curvature must be positive",
    lambda: generate_post(
      spec, replace(composite, parent=invalid_curvature), generation
    ),
  )
  invalid_stroke = replace(
    stitch,
    inherited_parameters={"stroke_width": 2.0},
  )
  _expect_error(
    ValueError,
    "inherit the parent stroke_width",
    lambda: generate_post(
      spec,
      replace(composite, prototypes={"stitch": invalid_stroke}),
      generation,
    ),
  )


def main() -> None:
  test_all_stitches_post_types_and_variants()
  test_front_and_back_are_mirrors()
  test_hook_and_j_shape_contracts()
  test_rotation_rasterization_and_registry()
  test_invalid_inputs()
  print("Post generation checks passed.")


if __name__ == "__main__":
  main()
