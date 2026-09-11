"""Explicit realization of reusable component prototypes."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..models import ComponentPrototype, CompositeSample, SampledParameters
from .sampler import ClassSampler
from .schema import ClassSpec, ParameterSpec

if TYPE_CHECKING:
  from .config import SamplingConfig


_MAX_CHILD_SEED = int(np.iinfo(np.int64).max)


def realize_component_prototypes(
  *,
  sampling_config: SamplingConfig,
  owner_spec: ClassSpec,
  parent_sample: SampledParameters,
  rng: np.random.Generator,
) -> CompositeSample:
  """Return an explicit parent/prototype tree without mutating the parent.

  Exactly one child is sampled for each active component role. A resolved
  occurrence count is metadata describing later reuse, not a sampling count.
  """

  if not isinstance(rng, np.random.Generator):
    raise TypeError("Component realization rng must be a numpy.random.Generator.")

  _validate_parent_identity(owner_spec, parent_sample)
  parent_values = parent_sample.as_dict()
  inherited = _inherited_parameters(owner_spec, parent_values)
  prototypes: dict[str, ComponentPrototype] = {}

  for role, source in parent_sample.components.items():
    path = f"{owner_spec.class_group}.{owner_spec.class_name}.components.{role}"
    if not isinstance(role, str) or not role:
      raise ValueError(f"{path} has an invalid component role.")
    if not isinstance(source, Mapping):
      raise ValueError(f"{path} must be a mapping.")

    declaration = copy.deepcopy(dict(source))
    child_name = _resolve_child_name(role, declaration, parent_values, path)
    child_group = _resolve_child_group(sampling_config, child_name, path)
    child_spec = sampling_config.resolve(child_group, child_name)
    if _declares_components(child_spec):
      raise ValueError(
        f"{path} resolves to nested component-bearing class "
        f"{child_group}.{child_name}; nested component realization is not supported."
      )
    sampling_policy = _select_sampling_policy(
      declaration=declaration,
      child_name=child_name,
      path=path,
    )
    if sampling_policy and child_group != "primitive":
      raise ValueError(
        f"{path}.sampling targets non-primitive child class {child_group}.{child_name}."
      )
    effective_spec = _apply_sampling_policy(
      child_spec=child_spec,
      sampling_policy=sampling_policy,
      path=path,
    )
    child_seed = int(rng.integers(0, _MAX_CHILD_SEED, dtype=np.int64))
    child_overrides = (
      {"stroke_width": inherited["stroke_width"]}
      if "stroke_width" in inherited and "stroke_width" in child_spec.parameters
      else None
    )
    child_sample = ClassSampler(
      spec=effective_spec,
      config_identity=sampling_config.identity,
    ).sample(
      np.random.default_rng(child_seed),
      seed=child_seed,
      overrides=child_overrides,
      case_id=_child_case_id(parent_sample, role),
    )

    prototypes[role] = ComponentPrototype(
      role=role,
      class_group=child_group,
      class_name=child_name,
      declaration=declaration,
      occurrence_count=_resolve_occurrence_count(declaration, parent_values, path),
      arrangement=_resolve_arrangement(declaration, path),
      inherited_parameters=dict(inherited),
      sample=child_sample,
      sampling_policy=copy.deepcopy(sampling_policy),
      replaced_parameters=tuple(sampling_policy),
    )

  return CompositeSample(parent=parent_sample, prototypes=prototypes)


def _select_sampling_policy(
  *,
  declaration: Mapping[str, Any],
  child_name: str,
  path: str,
) -> dict[str, Any]:
  """Return the parameter policy applicable to one resolved child class."""

  sampling = declaration.get("sampling")
  if sampling is None:
    return {}
  if not isinstance(sampling, Mapping) or not sampling:
    raise ValueError(f"{path}.sampling must be a non-empty mapping.")

  if "allowed_classes" in declaration:
    if set(sampling) != {"by_class"}:
      raise ValueError(f"{path}.sampling must contain only by_class for a dynamic component.")
    by_class = sampling["by_class"]
    if not isinstance(by_class, Mapping) or not by_class:
      raise ValueError(f"{path}.sampling.by_class must be a non-empty mapping.")
    selected = by_class.get(child_name)
    if selected is None:
      return {}
    if not isinstance(selected, Mapping) or not selected:
      raise ValueError(
        f"{path}.sampling.by_class.{child_name} must be a non-empty mapping."
      )
    return copy.deepcopy(dict(selected))

  if "by_class" in sampling:
    raise ValueError(f"{path}.sampling cannot use by_class for a fixed component.")
  return copy.deepcopy(dict(sampling))


def _apply_sampling_policy(
  *,
  child_spec: ClassSpec,
  sampling_policy: Mapping[str, Any],
  path: str,
) -> ClassSpec:
  """Return an uncached child spec with complete parameter replacements applied."""

  if not sampling_policy:
    return child_spec

  parameters = dict(child_spec.parameters)
  for name, raw_parameter in sampling_policy.items():
    parameter_path = f"{path}.sampling.{name}"
    if name == "stroke_width":
      raise ValueError(
        f"{parameter_path} cannot replace inherited parameter 'stroke_width'."
      )
    if name not in parameters:
      raise ValueError(
        f"{parameter_path} references unknown top-level child parameter {name!r}."
      )
    if not isinstance(raw_parameter, Mapping):
      raise ValueError(f"{parameter_path} must be a mapping.")
    base_parameter = parameters[name]
    if base_parameter.kind == "derived":
      raise ValueError(
        f"{parameter_path} cannot replace derived child parameter {name!r}."
      )
    replacement_kind = raw_parameter.get("kind")
    if replacement_kind != base_parameter.kind:
      raise ValueError(
        f"{parameter_path}.kind must match {child_spec.class_group}.{child_spec.class_name} "
        f"kind {base_parameter.kind!r}."
      )
    distribution = raw_parameter.get("distribution")
    if not isinstance(distribution, Mapping):
      raise ValueError(f"{parameter_path}.distribution must be a mapping.")
    parameters[name] = ParameterSpec(
      name=name,
      kind=replacement_kind,
      distribution=copy.deepcopy(dict(distribution)),
    )

  return replace(child_spec, parameters=parameters)


def _declares_components(spec: ClassSpec) -> bool:
  if spec.components:
    return True
  return any(bool(bundle.get("components")) for bundle in spec.variants.values())


def _validate_parent_identity(spec: ClassSpec, sample: SampledParameters) -> None:
  provenance = sample.provenance
  if provenance is None:
    return
  expected = (spec.class_group, spec.class_name)
  actual = (provenance.class_group, provenance.class_name)
  if actual != expected:
    raise ValueError(
      f"Parent sample belongs to {actual[0]}.{actual[1]}, not {expected[0]}.{expected[1]}."
    )


def _inherited_parameters(spec: ClassSpec, values: Mapping[str, Any]) -> dict[str, Any]:
  if "stroke_width" not in values:
    return {}
  value = values["stroke_width"]
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{spec.class_group}.{spec.class_name} stroke_width must be numeric.")
  stroke_width = float(value)
  if not math.isfinite(stroke_width) or stroke_width <= 0.0:
    raise ValueError(f"{spec.class_group}.{spec.class_name} stroke_width must be positive and finite.")
  return {"stroke_width": stroke_width}


def _resolve_child_name(
  role: str,
  declaration: Mapping[str, Any],
  parent_values: Mapping[str, Any],
  path: str,
) -> str:
  fixed = declaration.get("class")
  allowed = declaration.get("allowed_classes")
  if fixed is not None and allowed is not None:
    raise ValueError(f"{path} cannot declare both class and allowed_classes.")
  if fixed is not None:
    if not isinstance(fixed, str) or not fixed:
      raise ValueError(f"{path}.class must be a non-empty string.")
    return fixed
  if not isinstance(allowed, list) or not allowed:
    raise ValueError(f"{path} requires class or a non-empty allowed_classes list.")
  selected = parent_values.get(role)
  if not isinstance(selected, str) or not selected:
    raise ValueError(f"{path} requires parent parameter {role!r} to select its class.")
  if selected not in allowed:
    raise ValueError(f"{path} selected class {selected!r} is not in allowed_classes.")
  return selected


def _resolve_child_group(
  sampling_config: SamplingConfig,
  child_name: str,
  path: str,
) -> str:
  matches = [group for group, name in sampling_config.class_keys if name == child_name]
  if not matches:
    raise ValueError(f"{path} references unknown child class {child_name!r}.")
  if len(matches) != 1:
    raise ValueError(f"{path} child class {child_name!r} is ambiguous across families.")
  return matches[0]


def _resolve_occurrence_count(
  declaration: Mapping[str, Any],
  parent_values: Mapping[str, Any],
  path: str,
) -> int | None:
  value = declaration.get("count", parent_values.get("count"))
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"{path}.count must resolve to a positive integer.")
  return value


def _resolve_arrangement(declaration: Mapping[str, Any], path: str) -> str | None:
  arrangement = declaration.get("arrangement")
  if arrangement is None:
    return None
  if not isinstance(arrangement, str) or not arrangement:
    raise ValueError(f"{path}.arrangement must be a non-empty string.")
  return arrangement


def _child_case_id(parent_sample: SampledParameters, role: str) -> str | None:
  provenance = parent_sample.provenance
  if provenance is None or provenance.case_id is None:
    return None
  return f"{provenance.case_id}.components.{role}.prototype"
