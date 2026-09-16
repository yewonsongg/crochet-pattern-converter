"""Regression checks for three-chain picot generation."""

from __future__ import annotations

from dataclasses import asdict, replace
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

from generation.compound.ch3picot import generate_ch3picot
from generation.core.models import CompositeSample, GenerationConfig
from generation.core.rendering import render_png
from generation.core.sampling import load_sampling_config
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _sample(junction_gap_ratio=0.10, offset=0.0):
  config = _config()
  parent = config.sample(
    "compound",
    "ch3picot",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "junction_gap_ratio": junction_gap_ratio,
      "closure_offset": offset,
      "stroke_width": 1.8,
    },
  )
  composite = config.realize_components(
    "compound", "ch3picot", parent, np.random.default_rng(202)
  )
  return config, parent, composite


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_geometry_and_reuse() -> None:
  config, parent, composite = _sample()
  spec = config.resolve("compound", "ch3picot")
  snapshot = asdict(composite)
  generation = GenerationConfig(
    canvas_width_px=120,
    canvas_height_px=80,
    target_visible_px=50.0,
  )
  generated = generate_ch3picot(spec, composite, generation)
  repeated = generate_ch3picot(spec, composite, generation)
  metadata = generated.metadata

  assert asdict(composite) == snapshot
  assert repeated.svg == generated.svg
  assert repeated.metadata == metadata
  assert generated.sampled_parameters == parent.as_dict()
  assert generated.sampling_provenance is parent.provenance
  assert generated.variant_id is None
  assert set(metadata["component_prototypes"]) == {"chains", "closure"}
  assert metadata["component_prototypes"]["chains"]["occurrence_count"] == 3
  assert metadata["component_prototypes"]["closure"]["occurrence_count"] == 1

  ellipses = [node for node in fromstring(generated.svg).iter()
              if node.tag.endswith("ellipse")]
  assert len(ellipses) == 4
  assert all(node.attrib["fill"] == "none" for node in ellipses[:3])
  assert ellipses[3].attrib["fill"] == "black"

  chains = metadata["chain_placements"]
  assert [item["position"] for item in chains] == [
    "top", "lower_left", "lower_right",
  ]
  assert [item["rotation_deg"] for item in chains] == [0.0, 90.0, 90.0]
  assert all(
    math.isclose(item["visible_span_px"], metadata["chain_span_px"], abs_tol=1e-10)
    for item in chains
  )

  scale = metadata["construction_scale_px_per_unit"]
  top = tuple(chains[0]["center_px"])
  lower_left = tuple(chains[1]["center_px"])
  lower_right = tuple(chains[2]["center_px"])
  assert math.isclose(lower_left[1], lower_right[1], abs_tol=1e-10)
  assert math.isclose(top[0], (lower_left[0] + lower_right[0]) / 2.0, abs_tol=1e-10)
  assert math.isclose(
    lower_right[0] - lower_left[0], metadata["base_span_px"], abs_tol=1e-10
  )
  assert math.isclose(
    lower_left[1] - top[1], metadata["trace_height_px"], abs_tol=1e-10
  )
  closure = metadata["closure_placement"]
  assert math.dist(
    tuple(closure["center_px"]),
    ((lower_left[0] + lower_right[0]) / 2.0, lower_left[1]),
  ) < 1e-8
  assert math.isclose(closure["visible_span_px"], 0.30 * scale, abs_tol=1e-10)

  bounds = metadata["rendered_bounds_px"]
  assert math.isclose(
    max(bounds[2] - bounds[0], bounds[3] - bounds[1]), 50.0, abs_tol=1e-8
  )
  assert math.isclose((bounds[0] + bounds[2]) / 2.0, 60.0, abs_tol=1e-8)
  assert math.isclose((bounds[1] + bounds[3]) / 2.0, 40.0, abs_tol=1e-8)


def test_layout_distributions_and_closure_bias() -> None:
  config = _config()
  spec = config.resolve("compound", "ch3picot")
  assert spec.parameters["junction_gap_ratio"].distribution == {
    "type": "truncated_normal",
    "mean": 0.10,
    "std": 0.015,
    "min": 0.07,
    "max": 0.13,
  }
  closure_distribution = spec.parameters["closure_offset"].distribution
  assert closure_distribution["type"] == "mixture"
  assert [item["weight"] for item in closure_distribution["components"]] == [
    0.70, 0.30,
  ]

  offsets: list[float] = []
  zero_provenance = None
  for seed in range(500):
    sample = config.sample(
      "compound", "ch3picot", np.random.default_rng(seed), seed=seed
    )
    offset = sample.parameters["closure_offset"]
    offsets.append(offset)
    if offset == 0.0 and zero_provenance is None:
      zero_provenance = sample.provenance
  zero_fraction = sum(value == 0.0 for value in offsets) / len(offsets)
  assert 0.62 <= zero_fraction <= 0.78
  assert all(-0.10 <= value <= 0.10 for value in offsets)
  assert any(value != 0.0 for value in offsets)
  assert zero_provenance is not None
  assert zero_provenance.decisions["closure_offset"]["component"] == 0


def test_offset_frame_rotation_and_rasterization() -> None:
  for offset in (-0.10, 0.0, 0.10):
    config, _, composite = _sample(junction_gap_ratio=0.07, offset=offset)
    spec = config.resolve("compound", "ch3picot")
    base = GenerationConfig(
      canvas_width_px=100,
      canvas_height_px=100,
      target_visible_px=60.0,
    )
    generated = generate_ch3picot(spec, composite, base)
    rotated = generate_ch3picot(spec, composite, replace(base, rotation_deg=23.0))
    assert generated.metadata["chain_placements"] == rotated.metadata["chain_placements"]
    assert generated.metadata["closure_placement"] == rotated.metadata["closure_placement"]
    assert generated.metadata["rendered_bounds_px"] == rotated.metadata["rendered_bounds_px"]

    metadata = generated.metadata
    lower_left = metadata["chain_placements"][1]
    lower_right = metadata["chain_placements"][2]
    midpoint_x = (lower_left["center_px"][0] + lower_right["center_px"][0]) / 2.0
    closure_x = metadata["closure_placement"]["center_px"][0]
    assert math.isclose(
      closure_x - midpoint_x,
      offset * metadata["base_span_px"] / 2.0,
      abs_tol=1e-8,
    )
    left_bounds = lower_left["rendered_bounds_px"]
    right_bounds = lower_right["rendered_bounds_px"]
    closure_bounds = metadata["closure_placement"]["rendered_bounds_px"]
    assert left_bounds[2] < closure_bounds[0]
    assert closure_bounds[2] < right_bounds[0]

    with TemporaryDirectory() as directory:
      output = Path(directory) / "ch3picot.png"
      render_png(generated.svg, output)
      with Image.open(output) as image:
        raster_bounds = image.getbbox()
        assert raster_bounds is not None
        declared = metadata["rendered_bounds_px"]
        assert raster_bounds[0] >= math.floor(declared[0]) - 1
        assert raster_bounds[1] >= math.floor(declared[1]) - 1
        assert raster_bounds[2] <= math.ceil(declared[2]) + 1
        assert raster_bounds[3] <= math.ceil(declared[3]) + 1


def test_endpoint_clearance() -> None:
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  for junction_gap_ratio in (0.07, 0.13):
    for offset in (-0.10, 0.10):
      config, _, composite = _sample(
        junction_gap_ratio=junction_gap_ratio, offset=offset
      )
      chains = composite.prototypes["chains"]
      closure = composite.prototypes["closure"]
      for chain_aspect in (1.30, 1.85):
        for closure_aspect in (0.92, 1.08):
          adjusted = replace(composite, prototypes={
            "chains": replace(chains, sample=replace(
              chains.sample,
              parameters={
                **chains.sample.parameters,
                "aspect_ratio": chain_aspect,
              },
            )),
            "closure": replace(closure, sample=replace(
              closure.sample,
              parameters={
                **closure.sample.parameters,
                "aspect_ratio": closure_aspect,
              },
            )),
          })
          generated = generate_ch3picot(
            config.resolve("compound", "ch3picot"), adjusted, generation
          )
          assert generated.metadata["construction_junction_gap"] >= (
            junction_gap_ratio
          )
          assert min(generated.metadata["construction_closure_gaps"]) >= (
            junction_gap_ratio - 1e-12
          )


def test_validation_and_registry() -> None:
  config, parent, composite = _sample()
  spec = config.resolve("compound", "ch3picot")
  generation = GenerationConfig()
  assert GENERATOR_REGISTRY[("compound", "ch3picot")] is generate_ch3picot
  assert ("compound", "ch3picot") in COMPOSITE_GENERATORS

  _expect_error(
    TypeError,
    "requires a realized CompositeSample",
    lambda: generate_ch3picot(spec, parent, generation),
  )
  _expect_error(
    ValueError,
    "compound.ch3picot class specification",
    lambda: generate_ch3picot(config.resolve("primitive", "ch"), composite, generation),
  )
  _expect_error(
    ValueError,
    "exactly 'chains' and 'closure'",
    lambda: generate_ch3picot(
      spec, replace(composite, prototypes={"chains": composite.prototypes["chains"]}), generation
    ),
  )
  chains = composite.prototypes["chains"]
  _expect_error(
    ValueError,
    "occurrence_count must equal 3",
    lambda: generate_ch3picot(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, occurrence_count=2),
      }),
      generation,
    ),
  )
  invalid_shape = replace(
    chains.sample,
    parameters={**chains.sample.parameters, "shape": "circle"},
  )
  _expect_error(
    ValueError,
    "shape must be 'oval'",
    lambda: generate_ch3picot(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, sample=invalid_shape),
      }),
      generation,
    ),
  )
  obsolete_parent = replace(
    parent,
    parameters={**parent.parameters, "radius": 1.0},
  )
  _expect_error(
    ValueError,
    "obsolete parameters",
    lambda: generate_ch3picot(
      spec, replace(composite, parent=obsolete_parent), generation
    ),
  )
  _expect_error(
    ValueError,
    "must exceed its rendered stroke width",
    lambda: generate_ch3picot(
      spec, composite, GenerationConfig(target_visible_px=0.1)
    ),
  )


def main() -> None:
  test_geometry_and_reuse()
  test_layout_distributions_and_closure_bias()
  test_offset_frame_rotation_and_rasterization()
  test_endpoint_clearance()
  test_validation_and_registry()
  print("Three-chain picot generation checks passed.")


if __name__ == "__main__":
  main()
