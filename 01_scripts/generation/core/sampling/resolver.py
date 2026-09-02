from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from .distributions import sample_distribution
from .schema import ClassSpec, ComponentSpec, ParameterSpec, TopologySpec


@dataclass(frozen=True)
class SamplingResult:
  parameters: dict[str, Any]
  derived: dict[str, Any]
  topology: dict[str, Any]
  components: dict[str, Any]
  decisions: dict[str, Any]

def resolve_class_spec(raw: Mapping[str, Any], group: str, name: str) -> ClassSpec:
  try:
    source = raw["classes"][group][name]
  except KeyError as exc:
    raise KeyError(f"Unknown class: {group}.{name}") from exc
  params = {k: ParameterSpec(k, v["kind"], v["distribution"]) for k, v in source.get("parameters", {}).items() if "distribution" in v}
  components = {k: ComponentSpec(k, v) for k, v in source.get("components", {}).items()}
  return ClassSpec(group, name, params, source.get("variants", {}), TopologySpec(source.get("topology", {})), components, source)


build_class_spec = resolve_class_spec

def sample_class(spec: ClassSpec, rng: np.random.Generator) -> SamplingResult:
  values, derived, decisions = {}, {}, {}
  active = dict(spec.parameters)
  if "variant" in active:
    variant = sample_distribution(active["variant"].distribution, rng, values, decisions, "variant")
    values["variant"] = variant
    bundle = spec.variants.get(variant, {})
    active.update({k: ParameterSpec(k, v["kind"], v["distribution"]) for k, v in bundle.get("parameters", {}).items()})
  pending = dict(active)
  while pending:
    progressed = False
    context = {**values, **derived}
    for name, parameter in list(pending.items()):
      if all(dep in context for dep in parameter.distribution.get("depends_on", [])):
        result = sample_distribution(parameter.distribution, rng, context, decisions, name)
        (derived if parameter.kind == "derived" else values)[name] = result
        del pending[name]
        progressed = True
        context = {**values, **derived}
    if not progressed:
      raise ValueError(f"Unresolvable sampling dependencies: {list(pending)}")
  topology = dict(spec.topology.raw)
  connector = topology.get("connector", {})
  if connector.get("enabled") and connector.get("distribution"):
    values["connector_type"] = sample_distribution(connector["distribution"], rng, {**values, **derived}, decisions, "topology.connector")
  components = dict(spec.components)
  if values.get("variant") in spec.variants:
    components.update({k: ComponentSpec(k, v) for k, v in spec.variants[values["variant"]].get("components", {}).items()})
  resolved_components = {}
  context = {**values, **derived}
  for name, component in components.items():
    item = dict(component.raw)
    for key, value in item.items():
      if isinstance(value, str) and value in context:
        item[key] = context[value]
    resolved_components[name] = item
  return SamplingResult(values, derived, topology, resolved_components, decisions)
