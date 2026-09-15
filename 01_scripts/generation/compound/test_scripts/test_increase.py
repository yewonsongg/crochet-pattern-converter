"""Regression checks for joined-base increase generation."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import Element, fromstring

import numpy as np
from PIL import Image
import yaml


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.compound.increase import generate_increase
from generation.core.models import CompositeSample, GenerationConfig
from generation.core.rendering import render_png
from generation.core.sampling import load_sampling_config
from generation.core.sampling.validator import (
  SamplingValidationError,
  validate_sampling_config,
)
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _parent_and_composite(
  stitch: str,
  count: int,
  chain_presence: bool,
  *,
  realization_seed: int = 202,
):
  config = _config()
  parent = config.sample(
    "compound",
    "increase",
    np.random.default_rng(101),
    seed=101,
    overrides={
      "chain_presence": chain_presence,
      "stitch": stitch,
      "count": count,
      "chain_gap_ratio": 0.65 if chain_presence else 0.0,
      "spread_angle_deg": 26.0,
      "stroke_width": 1.8,
    },
  )
  composite = config.realize_components(
    "compound",
    "increase",
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
  config, absent_parent, absent = _parent_and_composite("dc", 4, False)
  _, present_parent, present = _parent_and_composite("dc", 4, True)
  assert set(absent_parent.components) == {"stitch"}
  assert set(absent.prototypes) == {"stitch"}
  assert set(present_parent.components) == {"stitch", "chains"}
  assert set(present.prototypes) == {"stitch", "chains"}
  assert present_parent.components["chains"]["when"] == {
    "parameter": "chain_presence",
    "equals": True,
  }

  absent_stitch = absent.prototypes["stitch"]
  present_stitch = present.prototypes["stitch"]
  assert absent_stitch.sample.parameters == present_stitch.sample.parameters
  assert absent_stitch.sample.provenance.seed == present_stitch.sample.provenance.seed
  chains = present.prototypes["chains"]
  assert chains.class_name == "ch"
  assert chains.occurrence_count == 3
  assert chains.arrangement == "between_adjacent_tops"
  assert chains.sample.parameters["shape"] == "oval"
  assert chains.replaced_parameters == ("shape",)

  base_spec = config.resolve("primitive", "ch")
  config.realize_components(
    "compound", "increase", present_parent, np.random.default_rng(202)
  )
  assert config.resolve("primitive", "ch") == base_spec


def test_all_stitches_counts_and_chain_states() -> None:
  per_stitch_lines = {"hdc": 2, "dc": 3, "tr": 4, "dtr": 5}
  for stitch, line_count in per_stitch_lines.items():
    for count in (2, 3, 6):
      for chain_presence in (False, True):
        config, parent, composite = _parent_and_composite(
          stitch, count, chain_presence
        )
        snapshot = asdict(composite)
        generation = GenerationConfig(
          canvas_width_px=120,
          canvas_height_px=80,
          target_visible_px=50.0,
        )
        generated = generate_increase(
          config.resolve("compound", "increase"), composite, generation
        )
        repeated = generate_increase(
          config.resolve("compound", "increase"), composite, generation
        )
        metadata = generated.metadata

        assert asdict(composite) == snapshot
        assert repeated.svg == generated.svg
        assert repeated.metadata == metadata
        assert generated.variant_id is None
        assert generated.sampled_parameters == parent.as_dict()
        assert generated.sampling_provenance is parent.provenance
        assert len(_elements(generated.svg, "line")) == count * line_count
        assert len(_elements(generated.svg, "ellipse")) == (
          count - 1 if chain_presence else 0
        )
        assert metadata["axis_angles_deg"][0] == -13.0
        assert metadata["axis_angles_deg"][-1] == 13.0
        assert len(metadata["placements"]) == count
        bases = [tuple(item["base_px"]) for item in metadata["placements"]]
        assert all(base == bases[0] for base in bases)
        assert [item["top_px"][0] for item in metadata["placements"]] == sorted(
          item["top_px"][0] for item in metadata["placements"]
        )
        assert all(
          math.isclose(
            math.dist(tuple(item["base_px"]), tuple(item["top_px"])),
            metadata["stem_length_px"],
            abs_tol=1e-8,
          )
          for item in metadata["placements"]
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
        assert metadata["component_prototypes"]["stitch"]["occurrence_count"] == count
        assert len(metadata["chain_placements"]) == (
          count - 1 if chain_presence else 0
        )
        if stitch in {"dc", "tr", "dtr"}:
          angle = composite.prototypes["stitch"].sample.parameters[
            "cross_bar_angle_deg"
          ]
          assert abs(angle) >= 8.0

        if chain_presence:
          chain_prototype = composite.prototypes["chains"]
          aspect_ratio = chain_prototype.sample.parameters["aspect_ratio"]
          for index, chain in enumerate(metadata["chain_placements"]):
            left_top = tuple(metadata["placements"][index]["top_px"])
            right_top = tuple(metadata["placements"][index + 1]["top_px"])
            expected_center = (
              (left_top[0] + right_top[0]) / 2.0,
              (left_top[1] + right_top[1]) / 2.0,
            )
            assert math.dist(tuple(chain["center_px"]), expected_center) < 1e-8
            chord_length = math.dist(left_top, right_top)
            assert math.isclose(
              chain["visible_span_px"], 0.65 * chord_length, abs_tol=1e-8
            )
            assert math.isclose(
              chain["height_px"], chain["width_px"] / aspect_ratio, abs_tol=1e-8
            )
            expected_rotation = math.degrees(math.atan2(
              right_top[1] - left_top[1], right_top[0] - left_top[0]
            ))
            assert math.isclose(chain["rotation_deg"], expected_rotation, abs_tol=1e-8)


def test_distribution_policies() -> None:
  config = _config()
  spread = config.resolve("compound", "increase").parameters[
    "spread_angle_deg"
  ].distribution
  assert spread == {
    "type": "truncated_normal",
    "mean": 24.0,
    "std": 5.0,
    "min": 10.0,
    "max": 40.0,
  }

  expected_bar_policies = {
    "hdc": (0.28, 0.03, 0.22, 0.34),
    "dc": (0.24, 0.03, 0.18, 0.30),
    "tr": (0.20, 0.025, 0.15, 0.25),
    "dtr": (0.18, 0.02, 0.14, 0.22),
  }
  for stitch, (mean, std, minimum, maximum) in expected_bar_policies.items():
    _, _, composite = _parent_and_composite(stitch, 3, False)
    prototype = composite.prototypes["stitch"]
    assert "bar_stem_ratio" in prototype.replaced_parameters
    assert "bar_stem_ratio" in prototype.sampling_policy
    distribution = prototype.sampling_policy["bar_stem_ratio"]["distribution"]
    assert distribution == {
      "type": "truncated_normal",
      "mean": mean,
      "std": std,
      "min": minimum,
      "max": maximum,
    }
    ratio = prototype.sample.parameters["bar_stem_ratio"]
    assert minimum <= ratio <= maximum

  for class_name in ("together", "increase"):
    seen_signs: set[int] = set()
    for seed in range(80):
      parent = config.sample(
        "compound",
        class_name,
        np.random.default_rng(seed),
        seed=seed,
        overrides={
          **({"chain_presence": False, "chain_gap_ratio": 0.0} if class_name == "increase" else {}),
          "stitch": "dc",
          "count": 3,
          "spread_angle_deg": 26.0 if class_name == "increase" else 55.0,
          **({"connector_length": 0.3} if class_name == "together" else {}),
          "stroke_width": 1.8,
        },
      )
      composite = config.realize_components(
        "compound", class_name, parent, np.random.default_rng(seed + 1000)
      )
      angle = composite.prototypes["stitch"].sample.parameters["cross_bar_angle_deg"]
      assert abs(angle) >= 8.0
      seen_signs.add(1 if angle > 0.0 else -1)
    assert seen_signs == {-1, 1}


def test_rotation_rasterization_and_registry() -> None:
  config, _, composite = _parent_and_composite("dc", 3, True)
  spec = config.resolve("compound", "increase")
  assert GENERATOR_REGISTRY[("compound", "increase")] is generate_increase
  assert ("compound", "increase") in COMPOSITE_GENERATORS
  generation = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
    rotation_deg=17.0,
  )
  rotated = generate_increase(spec, composite, generation)
  unrotated = generate_increase(spec, composite, replace(generation, rotation_deg=0.0))
  assert "rotate(17.0 50 50)" in rotated.svg
  assert rotated.metadata["placements"] == unrotated.metadata["placements"]
  assert rotated.metadata["chain_placements"] == unrotated.metadata["chain_placements"]
  assert rotated.metadata["rendered_bounds_px"] == unrotated.metadata["rendered_bounds_px"]

  with TemporaryDirectory() as directory:
    output = Path(directory) / "increase.png"
    render_png(unrotated.svg, output)
    with Image.open(output) as image:
      raster_bounds = image.getbbox()
      assert raster_bounds is not None
      declared = unrotated.metadata["rendered_bounds_px"]
      assert raster_bounds[0] >= math.floor(declared[0]) - 1
      assert raster_bounds[1] >= math.floor(declared[1]) - 1
      assert raster_bounds[2] <= math.ceil(declared[2]) + 1
      assert raster_bounds[3] <= math.ceil(declared[3]) + 1


def test_activation_validation() -> None:
  source_path = PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml"
  with source_path.open("r", encoding="utf-8") as handle:
    valid = yaml.safe_load(handle)

  def invalid(mutator, expected: str) -> None:
    source = copy.deepcopy(valid)
    mutator(source)
    _expect_error(
      SamplingValidationError,
      expected,
      lambda: validate_sampling_config(source),
    )

  chain = lambda source: source["classes"]["compound"]["increase"]["components"]["chains"]
  invalid(lambda source: chain(source).update({"when": {}}), "must be a non-empty mapping")
  invalid(
    lambda source: chain(source).update({
      "when": {"parameter": "chain_presence", "equals": True, "extra": 1}
    }),
    "must contain exactly parameter and equals",
  )
  invalid(
    lambda source: chain(source).update({
      "when": {"parameter": "spread_angle_deg", "equals": 20.0}
    }),
    "direct categorical parent parameter",
  )
  invalid(
    lambda source: chain(source).update({
      "when": {"parameter": "chain_presence", "equals": "yes"}
    }),
    "equals is not reachable",
  )


def test_invalid_generation_inputs() -> None:
  config, parent, composite = _parent_and_composite("dc", 3, True)
  spec = config.resolve("compound", "increase")
  generation = GenerationConfig()
  chains = composite.prototypes["chains"]

  _expect_error(
    TypeError,
    "requires a realized CompositeSample",
    lambda: generate_increase(spec, parent, generation),
  )
  _expect_error(
    ValueError,
    "compound.increase class specification",
    lambda: generate_increase(config.resolve("primitive", "dc"), composite, generation),
  )
  _expect_error(
    ValueError,
    "component prototype roles",
    lambda: generate_increase(
      spec, replace(composite, prototypes={"stitch": composite.prototypes["stitch"]}), generation
    ),
  )
  _expect_error(
    ValueError,
    "occurrence_count must equal chain_count",
    lambda: generate_increase(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, occurrence_count=1),
      }),
      generation,
    ),
  )
  _expect_error(
    ValueError,
    "arrangement must be 'between_adjacent_tops'",
    lambda: generate_increase(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, arrangement="ring"),
      }),
      generation,
    ),
  )
  invalid_chain_sample = replace(
    chains.sample,
    parameters={**chains.sample.parameters, "shape": "circle"},
  )
  _expect_error(
    ValueError,
    "shape must be 'oval'",
    lambda: generate_increase(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, sample=invalid_chain_sample),
      }),
      generation,
    ),
  )
  invalid_chain_aspect = replace(
    chains.sample,
    parameters={**chains.sample.parameters, "aspect_ratio": 0.8},
  )
  _expect_error(
    ValueError,
    "aspect_ratio must exceed 1",
    lambda: generate_increase(
      spec,
      replace(composite, prototypes={
        **composite.prototypes,
        "chains": replace(chains, sample=invalid_chain_aspect),
      }),
      generation,
    ),
  )
  invalid_topology = replace(
    parent,
    topology={**parent.topology, "base_relation": "separate"},
  )
  _expect_error(
    ValueError,
    "base_relation must be 'joined'",
    lambda: generate_increase(
      spec, replace(composite, parent=invalid_topology), generation
    ),
  )
  invalid_ratio = replace(
    parent,
    parameters={**parent.parameters, "chain_gap_ratio": 1.2},
  )
  _expect_error(
    ValueError,
    "chain_gap_ratio must be in the range",
    lambda: generate_increase(spec, replace(composite, parent=invalid_ratio), generation),
  )
  _expect_error(
    ValueError,
    "must exceed its rendered stroke width",
    lambda: generate_increase(
      spec, composite, GenerationConfig(target_visible_px=0.1)
    ),
  )


def main() -> None:
  test_activation_and_seed_stability()
  test_all_stitches_counts_and_chain_states()
  test_distribution_policies()
  test_rotation_rasterization_and_registry()
  test_activation_validation()
  test_invalid_generation_inputs()
  print("Increase compound generation checks passed.")


if __name__ == "__main__":
  main()
