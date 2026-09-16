"""Regression checks for symmetric crossed-stitch generation."""

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

from generation.compound.crossed import generate_crossed
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
  chain_presence: bool,
  *,
  realization_seed: int = 202,
):
  config = _config()
  parent = config.sample(
    "compound",
    "crossed",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "chain_presence": chain_presence,
      "stitch": stitch,
      "crossing_angle_deg": 60.0,
      "chain_gap_ratio": 0.55 if chain_presence else 0.0,
      "stroke_width": 1.8,
    },
  )
  composite = config.realize_components(
    "compound",
    "crossed",
    parent,
    np.random.default_rng(realization_seed),
  )
  return config, parent, composite


def _elements(svg: str, suffix: str) -> list[Element]:
  root = fromstring(svg)
  return [element for element in root.iter() if element.tag.endswith(suffix)]


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_activation_and_seed_stability() -> None:
  _, absent_parent, absent = _parent_and_composite("dc", False)
  _, present_parent, present = _parent_and_composite("dc", True)

  assert set(absent_parent.components) == {"stitch"}
  assert set(absent.prototypes) == {"stitch"}
  assert set(present_parent.components) == {"stitch", "chains"}
  assert set(present.prototypes) == {"stitch", "chains"}
  absent_stitch = absent.prototypes["stitch"]
  present_stitch = present.prototypes["stitch"]
  assert absent_stitch.sample.parameters == present_stitch.sample.parameters
  assert absent_stitch.sample.provenance.seed == present_stitch.sample.provenance.seed
  assert present_stitch.occurrence_count == 2
  assert present_stitch.arrangement == "crossed"

  chains = present.prototypes["chains"]
  assert chains.class_name == "ch"
  assert chains.occurrence_count == 1
  assert chains.arrangement == "between_adjacent_tops"
  assert chains.sample.parameters["shape"] == "oval"
  assert chains.replaced_parameters == ("shape",)


def test_all_stitches_and_chain_states() -> None:
  per_stitch_lines = {"hdc": 2, "dc": 3, "tr": 4, "dtr": 5}
  for stitch, line_count in per_stitch_lines.items():
    for chain_presence in (False, True):
      config, parent, composite = _parent_and_composite(stitch, chain_presence)
      snapshot = asdict(composite)
      generation = GenerationConfig(
        canvas_width_px=120,
        canvas_height_px=80,
        target_visible_px=50.0,
      )
      spec = config.resolve("compound", "crossed")
      generated = generate_crossed(spec, composite, generation)
      repeated = generate_crossed(spec, composite, generation)
      metadata = generated.metadata

      assert asdict(composite) == snapshot
      assert repeated.svg == generated.svg
      assert repeated.metadata == metadata
      assert generated.variant_id is None
      assert generated.sampled_parameters == parent.as_dict()
      assert generated.sampling_provenance is parent.provenance
      assert len(_elements(generated.svg, "line")) == 2 * line_count
      assert len(_elements(generated.svg, "ellipse")) == int(chain_presence)
      lines = _elements(generated.svg, "line")
      for top_bar_index in (1, line_count + 1):
        top_bar = lines[top_bar_index]
        assert math.isclose(
          float(top_bar.attrib["y1"]),
          float(top_bar.attrib["y2"]),
          abs_tol=1e-8,
        )
      drawing_group = next(
        element for element in fromstring(generated.svg)
        if element.tag.endswith("g")
      )
      emitted_tags = [element.tag.rsplit("}", 1)[-1] for element in drawing_group]
      if chain_presence:
        assert emitted_tags[-1] == "ellipse"
        assert all(tag == "line" for tag in emitted_tags[:-1])
      else:
        assert all(tag == "line" for tag in emitted_tags)

      assert metadata["axis_angles_deg"] == [30.0, -30.0]
      assert metadata["top_bar_angle_deg"] == 0.0
      assert len(metadata["placements"]) == 2
      left, right = metadata["placements"]
      assert left["base_px"][0] < right["base_px"][0]
      assert left["top_px"][0] > right["top_px"][0]
      crossing = tuple(metadata["crossing_point_px"])
      for placement in (left, right):
        base = tuple(placement["base_px"])
        top = tuple(placement["top_px"])
        midpoint = ((base[0] + top[0]) / 2.0, (base[1] + top[1]) / 2.0)
        assert math.dist(midpoint, crossing) < 1e-8
        assert math.isclose(
          math.dist(base, top), metadata["stem_length_px"], abs_tol=1e-8
        )

      top_chord = math.dist(tuple(left["top_px"]), tuple(right["top_px"]))
      assert math.isclose(
        top_chord, metadata["top_chord_length_px"], abs_tol=1e-8
      )
      bounds = metadata["rendered_bounds_px"]
      assert math.isclose(
        max(bounds[2] - bounds[0], bounds[3] - bounds[1]),
        generation.target_visible_px,
        abs_tol=1e-8,
      )
      assert math.isclose((bounds[0] + bounds[2]) / 2.0, 60.0, abs_tol=1e-8)
      assert math.isclose((bounds[1] + bounds[3]) / 2.0, 40.0, abs_tol=1e-8)

      assert set(metadata["component_prototypes"]) == (
        {"stitch", "chains"} if chain_presence else {"stitch"}
      )
      assert len(metadata["chain_placements"]) == int(chain_presence)
      if chain_presence:
        chain = metadata["chain_placements"][0]
        expected_center = (
          (left["top_px"][0] + right["top_px"][0]) / 2.0,
          (left["top_px"][1] + right["top_px"][1]) / 2.0,
        )
        assert math.dist(tuple(chain["center_px"]), expected_center) < 1e-8
        assert chain["rotation_deg"] == 0.0
        assert math.isclose(
          chain["visible_span_px"], 0.55 * top_chord, abs_tol=1e-8
        )


def test_sampling_distributions_and_policy() -> None:
  config = _config()
  spec = config.resolve("compound", "crossed")
  assert spec.parameters["crossing_angle_deg"].distribution == {
    "type": "truncated_normal",
    "mean": 60.0,
    "std": 6.0,
    "min": 45.0,
    "max": 75.0,
  }
  assert spec.parameters["chain_gap_ratio"].distribution == {
    "type": "conditional",
    "depends_on": ["chain_presence"],
    "cases": {
      True: {
        "type": "truncated_normal",
        "mean": 0.55,
        "std": 0.04,
        "min": 0.45,
        "max": 0.65,
      },
      False: {"type": "fixed", "value": 0.0},
    },
  }
  for stitch in ("hdc", "dc", "tr", "dtr"):
    _, _, composite = _parent_and_composite(stitch, False)
    prototype = composite.prototypes["stitch"]
    expected_parameters = (
      ("bar_stem_ratio",)
      if stitch == "hdc"
      else (
        "bar_stem_ratio",
        "cross_bar_ratio",
        "cross_bar_y",
        "cross_bar_angle_deg",
      )
    )
    assert prototype.replaced_parameters == expected_parameters
    assert prototype.sampling_policy["bar_stem_ratio"]["distribution"] == {
      "type": "truncated_normal",
      "mean": 0.12,
      "std": 0.015,
      "min": 0.09,
      "max": 0.14,
    }
    assert 0.09 <= prototype.sample.parameters["bar_stem_ratio"] <= 0.14


def test_rotation_rasterization_and_registry() -> None:
  config, _, composite = _parent_and_composite("dc", True)
  spec = config.resolve("compound", "crossed")
  assert GENERATOR_REGISTRY[("compound", "crossed")] is generate_crossed
  assert ("compound", "crossed") in COMPOSITE_GENERATORS
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  unrotated = generate_crossed(spec, composite, generation)
  rotated = generate_crossed(
    spec, composite, replace(generation, rotation_deg=17.0)
  )
  assert "rotate(17.0 50 50)" in rotated.svg
  assert rotated.metadata["placements"] == unrotated.metadata["placements"]
  assert rotated.metadata["chain_placements"] == unrotated.metadata["chain_placements"]
  assert rotated.metadata["rendered_bounds_px"] == unrotated.metadata["rendered_bounds_px"]

  with TemporaryDirectory() as directory:
    output = Path(directory) / "crossed.png"
    render_png(unrotated.svg, output)
    with Image.open(output) as image:
      raster_bounds = image.getbbox()
      assert raster_bounds is not None
      declared = unrotated.metadata["rendered_bounds_px"]
      assert raster_bounds[0] >= math.floor(declared[0]) - 1
      assert raster_bounds[1] >= math.floor(declared[1]) - 1
      assert raster_bounds[2] <= math.ceil(declared[2]) + 1
      assert raster_bounds[3] <= math.ceil(declared[3]) + 1


def test_invalid_generation_inputs() -> None:
  config, parent, composite = _parent_and_composite("dc", True)
  spec = config.resolve("compound", "crossed")
  generation = GenerationConfig()
  stitch = composite.prototypes["stitch"]
  chains = composite.prototypes["chains"]

  _expect_error(
    TypeError,
    "requires a realized CompositeSample",
    lambda: generate_crossed(spec, parent, generation),
  )
  _expect_error(
    ValueError,
    "compound.crossed class specification",
    lambda: generate_crossed(config.resolve("primitive", "dc"), composite, generation),
  )
  _expect_error(
    ValueError,
    "component prototype roles",
    lambda: generate_crossed(
      spec, replace(composite, prototypes={"stitch": stitch}), generation
    ),
  )
  _expect_error(
    ValueError,
    "occurrence_count must equal 2",
    lambda: generate_crossed(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "stitch": replace(stitch, occurrence_count=3),
      }),
      generation,
    ),
  )
  _expect_error(
    ValueError,
    "arrangement must be 'crossed'",
    lambda: generate_crossed(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "stitch": replace(stitch, arrangement=None),
      }),
      generation,
    ),
  )
  invalid_topology = replace(
    composite.parent,
    topology={**composite.parent.topology, "base_relation": "joined"},
  )
  _expect_error(
    ValueError,
    "base_relation must be 'separate'",
    lambda: generate_crossed(
      spec, replace(composite, parent=invalid_topology), generation
    ),
  )
  _expect_error(
    ValueError,
    "arrangement must be 'between_adjacent_tops'",
    lambda: generate_crossed(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, arrangement="ring"),
      }),
      generation,
    ),
  )
  invalid_chain = replace(
    chains.sample,
    parameters={**chains.sample.parameters, "shape": "circle"},
  )
  _expect_error(
    ValueError,
    "shape must be 'oval'",
    lambda: generate_crossed(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, sample=invalid_chain),
      }),
      generation,
    ),
  )
  invalid_parent = replace(
    composite.parent,
    parameters={**composite.parent.parameters, "chain_gap_ratio": 0.0},
  )
  _expect_error(
    ValueError,
    "range (0, 1) when chains are present",
    lambda: generate_crossed(
      spec, replace(composite, parent=invalid_parent), generation
    ),
  )


if __name__ == "__main__":
  test_activation_and_seed_stability()
  test_all_stitches_and_chain_states()
  test_sampling_distributions_and_policy()
  test_rotation_rasterization_and_registry()
  test_invalid_generation_inputs()
  print("crossed generator tests passed")
