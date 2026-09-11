"""Regression checks for reusable component-prototype sampling."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import yaml


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
if str(PROJECT_ROOT / "01_scripts") not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT / "01_scripts"))

from generation.core.sampling import load_sampling_config
from generation.core.sampling.validator import SamplingValidationError, validate_sampling_config


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _sample(config, group: str, name: str, seed: int, overrides: dict):
  return config.sample(
    group,
    name,
    np.random.default_rng(seed),
    seed=seed,
    overrides=overrides,
  )


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_ring_prototype_reuse_and_determinism() -> None:
  config = _config()
  base_chain_spec = copy.deepcopy(config.resolve("primitive", "ch"))
  parent = _sample(config, "instructive", "ring", 100, {
    "variant": "chain",
    "stroke_width": 1.9,
    "count": 6,
    "chain_pitch": 3.2,
  })
  parent_snapshot = copy.deepcopy(parent)

  first = config.realize_components(
    "instructive", "ring", parent, np.random.default_rng(200)
  )
  second = config.realize_components(
    "instructive", "ring", parent, np.random.default_rng(200)
  )

  assert first.parent is parent
  assert parent == parent_snapshot
  assert asdict(first) == asdict(second)
  assert set(first.prototypes) == {"chain"}

  prototype = first.prototypes["chain"]
  assert prototype.class_group == "primitive"
  assert prototype.class_name == "ch"
  assert prototype.occurrence_count == 6
  assert prototype.arrangement == "ring"
  assert prototype.inherited_parameters == {"stroke_width": 1.9}
  assert prototype.sample.parameters["stroke_width"] == 1.9
  assert prototype.sample.parameters["shape"] == "oval"
  assert "aspect_ratio" in prototype.sample.parameters
  assert prototype.sampling_policy == {
    "shape": {
      "kind": "discrete",
      "distribution": {
        "type": "categorical",
        "probabilities": {"oval": 1.0, "circle": 0.0},
      },
    },
  }
  assert prototype.replaced_parameters == ("shape",)
  assert prototype.sample.provenance is not None
  assert prototype.sample.provenance.seed is not None
  assert prototype.sample.provenance.overrides == {"stroke_width": 1.9}
  assert config.resolve("primitive", "ch") == base_chain_spec


def test_magic_ring_has_no_prototypes() -> None:
  config = _config()
  parent = _sample(config, "instructive", "ring", 300, {
    "variant": "magic",
    "stroke_width": 1.8,
  })
  composite = config.realize_components(
    "instructive", "ring", parent, np.random.default_rng(301)
  )
  assert composite.parent is parent
  assert composite.prototypes == {}


def test_dynamic_component_class_selection() -> None:
  config = _config()
  parent = _sample(config, "compound", "together", 400, {
    "stitch": "dc",
    "count": 3,
    "spread_angle_deg": 24.0,
    "connector_length": 0.2,
    "stroke_width": 1.7,
  })
  composite = config.realize_components(
    "compound", "together", parent, np.random.default_rng(401)
  )
  prototype = composite.prototypes["stitch"]
  assert prototype.class_name == "dc"
  assert prototype.occurrence_count == 3
  assert prototype.sample.parameters["stroke_width"] == 1.7
  assert prototype.sampling_policy == {}
  assert prototype.replaced_parameters == ()

  sc_parent = _sample(config, "compound", "together", 403, {
    "stitch": "sc",
    "count": 2,
    "spread_angle_deg": 20.0,
    "connector_length": 0.0,
    "stroke_width": 1.7,
  })
  sc_composite = config.realize_components(
    "compound", "together", sc_parent, np.random.default_rng(404)
  )
  sc_prototype = sc_composite.prototypes["stitch"]
  assert sc_prototype.sample.parameters["shape"] == "asymmetric"
  assert sc_prototype.replaced_parameters == ("shape",)
  assert set(sc_prototype.sampling_policy) == {"shape"}
  assert "asymmetry" in sc_prototype.sample.parameters
  assert "cross_bar_ratio" in sc_prototype.sample.parameters

  invalid = replace(
    parent,
    parameters={**parent.parameters, "stitch": "ch"},
  )
  _expect_error(
    ValueError,
    "selected class 'ch' is not in allowed_classes",
    lambda: config.realize_components(
      "compound", "together", invalid, np.random.default_rng(402)
    ),
  )

  post = _sample(config, "compound", "post", 410, {
    "post_type": "front",
    "variant": "hook",
    "stitch": "hdc",
    "hook_curvature": 0.15,
    "stroke_width": 1.6,
  })
  post_composite = config.realize_components(
    "compound", "post", post, np.random.default_rng(411)
  )
  assert post_composite.prototypes["stitch"].class_name == "hdc"
  assert post_composite.prototypes["stitch"].occurrence_count is None

  crossed = _sample(config, "compound", "crossed", 420, {
    "chain_presence": False,
    "stitch": "tr",
    "stroke_width": 2.1,
  })
  crossed_composite = config.realize_components(
    "compound", "crossed", crossed, np.random.default_rng(421)
  )
  crossed_stitch = crossed_composite.prototypes["stitch"]
  assert crossed_stitch.class_name == "tr"
  assert crossed_stitch.inherited_parameters == {"stroke_width": 2.1}


def test_multiple_roles_and_schema_aware_stroke_inheritance() -> None:
  config = _config()
  parent = _sample(config, "compound", "ch3picot", 500, {
    "radius": 1.0,
    "opening_angle_deg": 60.0,
    "closure_offset": 0.0,
    "stroke_width": 2.0,
  })
  composite = config.realize_components(
    "compound", "ch3picot", parent, np.random.default_rng(501)
  )

  assert set(composite.prototypes) == {"chains", "closure"}
  chains = composite.prototypes["chains"]
  closure = composite.prototypes["closure"]
  assert chains.class_name == "ch"
  assert chains.occurrence_count == 3
  assert chains.arrangement == "triangular"
  assert chains.sample.provenance is not None
  assert chains.sample.provenance.overrides == {"stroke_width": 2.0}
  assert chains.sample.parameters["shape"] == "oval"
  assert chains.replaced_parameters == ("shape",)
  assert closure.class_name == "slst"
  assert closure.occurrence_count == 1
  assert closure.inherited_parameters == {"stroke_width": 2.0}
  assert closure.sample.provenance is not None
  assert closure.sample.provenance.overrides == {}
  assert "stroke_width" not in closure.sample.parameters
  assert closure.sample.parameters["shape"] == "circle"
  assert closure.replaced_parameters == ("shape",)


def test_nested_components_are_rejected() -> None:
  config = _config()
  parent = _sample(config, "instructive", "ring", 600, {
    "variant": "chain",
    "stroke_width": 1.8,
    "count": 4,
    "chain_pitch": 3.0,
  })
  nested = replace(parent, components={
    "chain": {
      "class": "ch3picot",
      "count": 4,
      "arrangement": "ring",
      "inherit_generator": True,
    },
  })
  _expect_error(
    ValueError,
    "nested component-bearing class compound.ch3picot",
    lambda: config.realize_components(
      "instructive", "ring", nested, np.random.default_rng(601)
    ),
  )


def test_component_declaration_validation() -> None:
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

  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"].update(
      {"inherit_generator": "yes"}
    ),
    "inherit_generator must be a boolean",
  )
  invalid(
    lambda source: source["classes"]["compound"]["together"]["components"]["stitch"]["allowed_classes"].append("missing"),
    "allowed_classes references unknown classes",
  )
  invalid(
    lambda source: source["classes"]["compound"]["ch3picot"]["components"]["chains"].update({"count": 0}),
    "count must be positive",
  )
  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"].update({"arrangement": ""}),
    "arrangement must be a non-empty string",
  )
  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"].update(
      {"sampling": {}}
    ),
    "sampling must be a non-empty mapping",
  )
  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"].update(
      {"sampling": {"by_class": {"ch": {"shape": source["classes"]["primitive"]["ch"]["parameters"]["shape"]}}}}
    ),
    "cannot use by_class for a fixed component",
  )
  invalid(
    lambda source: source["classes"]["compound"]["together"]["components"]["stitch"]["sampling"]["by_class"].update(
      {"ch": {"shape": source["classes"]["primitive"]["ch"]["parameters"]["shape"]}}
    ),
    "contains classes outside allowed_classes",
  )
  invalid(
    lambda source: source["classes"]["compound"]["together"]["components"]["stitch"].update(
      {"sampling": {"shape": copy.deepcopy(source["classes"]["primitive"]["sc"]["parameters"]["shape"])}}
    ),
    "must contain only by_class for a dynamic component",
  )
  invalid(
    lambda source: source["classes"]["compound"]["together"]["components"]["stitch"]["sampling"]["by_class"].update(
      {"sc": {}}
    ),
    "sampling.by_class.sc must be a non-empty mapping",
  )
  invalid(
    lambda source: source["classes"]["compound"]["together"]["components"]["stitch"]["sampling"]["by_class"]["sc"].update(
      {"missing": {"kind": "continuous", "distribution": {"type": "fixed", "value": 1.0}}}
    ),
    "references unknown top-level parameter",
  )
  invalid(
    lambda source: source["classes"]["compound"]["together"]["components"]["stitch"]["sampling"]["by_class"]["sc"]["shape"].update(
      {"kind": "continuous"}
    ),
    "kind must match primitive.sc kind 'discrete'",
  )
  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"]["sampling"].update(
      {"stroke_width": copy.deepcopy(source["classes"]["primitive"]["ch"]["parameters"]["stroke_width"])}
    ),
    "cannot replace inherited parameter 'stroke_width'",
  )
  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"]["sampling"]["shape"]["distribution"]["probabilities"].update(
      {"oval": 0.4, "circle": 0.4}
    ),
    "probabilities must sum to 1",
  )
  invalid(
    lambda source: source["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"]["sampling"]["shape"]["distribution"].update(
      {"probabilities": {"oval": 1.0}}
    ),
    "cases must match reachable values of shape",
  )

  derived = copy.deepcopy(valid)
  derived["classes"]["primitive"]["ch"]["parameters"]["derived_probe"] = {
    "kind": "derived",
    "depends_on": ["aspect_ratio"],
    "formula": "aspect_ratio",
  }
  derived["classes"]["instructive"]["ring"]["variants"]["chain"]["components"]["chain"]["sampling"]["derived_probe"] = {
    "kind": "derived",
    "depends_on": ["aspect_ratio"],
    "formula": "aspect_ratio",
  }
  _expect_error(
    SamplingValidationError,
    "cannot replace derived child parameter 'derived_probe'",
    lambda: validate_sampling_config(derived),
  )


def main() -> None:
  test_ring_prototype_reuse_and_determinism()
  test_magic_ring_has_no_prototypes()
  test_dynamic_component_class_selection()
  test_multiple_roles_and_schema_aware_stroke_inheritance()
  test_nested_components_are_rejected()
  test_component_declaration_validation()
  print("Component-prototype sampling checks passed.")


if __name__ == "__main__":
  main()
