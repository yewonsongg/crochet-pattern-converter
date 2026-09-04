from typing import Any, Mapping
import math

import numpy as np

from ..models import SamplingResult
from .distributions import sample_distribution
from .schema import ClassSpec, ComponentSpec, ParameterSpec, TopologySpec


def resolve_class_spec(
  ontology_config: Mapping[str, Any],
  sampling_config: Mapping[str, Any], 
  class_group: str, 
  class_name: str, 
) -> ClassSpec:
  """Resolve one class specification from validated configuration mappings.

  Converts the raw sampling declarationf or one class into a :class:`ClassSpec` containing parameter, component, and topology specifications.

  If an ontology configuration is supplied, the matching ontology entry is required and its class ID is added tot he resolved class's raw metadata.

  Args:
    sampling_config: Validated symbol-sampling configuration.
    class_group: Sampling/ontology group containing the class.
    class_name: Name of the class to resolve.
    ontology_config: Optional validated ontology configuration.

  Returns:
    :class:`ClassSpec`: The resolved class specification
  
  Raises:
    KeyError: If the class is absent from the sampling configuration, or if ``ontology_config`` is supplied and does not contain the class.
    TypeError: If validated mappings do not have the expected structure.
  """

  try:
    classes = sampling_config["classes"]
    group = classes[class_group]
    source = group[class_name]
  except KeyError as exc:
    raise KeyError(f"Unknown class: {class_group}.{class_name} .") from exc
  
  ontology_entry: Mapping[str, Any] | None = None

  if ontology_config is not None:
    try:
      ontology_classes = ontology_config["classes"]
    except KeyError as exc:
      raise KeyError("Ontology configuration is missing 'classes'.") from exc

    ontology_entry = next((
      entry 
      for entry in ontology_classes 
      if (
        entry["family"] == class_group 
        and entry["name"] == class_name
      )), None)
    
    if ontology_entry is None:
      raise KeyError(f"Class is absent from ontology: {class_group}.{class_name} .")

  raw_parameters = source.get("parameters", {})
  if not isinstance(raw_parameters, Mapping):
    raise TypeError(f"{class_group}.{class_name}.parameters must be a mapping.")

  parameters: dict[str, ParameterSpec] = {}
  for parameter_name, parameter in raw_parameters.items():
    if not isinstance(parameter, Mapping):
      raise TypeError(f"{class_group}.{class_name}.parameters.{parameter_name} must be a mapping.")

    kind = parameter["kind"]
    if "distribution" in parameter:
      distribution = parameter["distribution"]
    elif kind == "derived":
      distribution = {
        "type": "derived",
        **{
          key: value
          for key, value in parameter.items()
          if key != "kind"
        },
      }
    else:
      continue

    parameters[parameter_name] = ParameterSpec(
      name = parameter_name,
      kind = kind,
      distribution = distribution,
    )

  raw_components = source.get("components", {})
  if not isinstance(raw_components, Mapping):
    raise TypeError(f"{class_group}.{class_name}.components must be a mapping.")

  components = {
    component_name: ComponentSpec(
      component_name,
      component_spec,
    )
    for component_name, component_spec in raw_components.items()
  }

  raw = dict(source)

  if ontology_entry is not None:
    raw["class_id"] = ontology_entry["id"]
  
  return ClassSpec(
    class_group = class_group,
    class_name = class_name,
    class_id = ontology_entry["id"],
    parameters = parameters,
    variants = source.get("variants", {}),
    topology = TopologySpec(
      source.get("topology", {})
    ),
    components = components,
    raw = raw,
  )


build_class_spec = resolve_class_spec

def sample_class(
  spec: ClassSpec, 
  rng: np.random.Generator,
  overrides: Mapping[str, Any] | None = None,
) -> SamplingResult:
  """Sample one concrete instance from a resolved class specification.

  Samples variant selections, direct parameters, derived parameters, conditional distributions, connector types, and component references. 
  Parameter dependencies are resolved incrementally until all pending parameters have been sampled.

  Args:
    spec: Resolved class specification to sample.
    rng: Random-number generator used for stochastic sampling.

  Returns:
    A ``SamplingResult`` containing sampled values, derived values, topology, resolved components, and sampling decisions.

  Raises:
    ValueError: If parameter dependencies cannot be resolved.
    KeyError: If a selected variant or component declaration is malformed.
  """

  values: dict[str, Any] = {}
  derived: dict[str, Any] = {}
  decisions: dict[str, Any] = {}
  overrides = dict(overrides or {})

  allowed_names = set(spec.parameters)
  for variant_spec in spec.variants.values():
    allowed_names.update(variant_spec.get("parameters", {}))
  unknown = set(overrides) - allowed_names
  if unknown:
    raise KeyError(f"Unknown sampling override(s) for {spec.class_group}.{spec.class_name}: {sorted(unknown)!r}.")

  active: dict[str, ParameterSpec] = dict(spec.parameters)

  variant = None
  if "variant" in active:
    variant_parameter = active.pop("variant")

    if "variant" in overrides:
      variant = overrides["variant"]
      _validate_override(variant_parameter.distribution, variant, "variant")
      decisions["variant"] = {"source": "override", "value": variant}
    else:
      variant = sample_distribution(
        spec = variant_parameter.distribution,
        rng = rng,
        values = values,
        decisions = decisions,
        path = "variant",
      )

    if not isinstance(variant, str):
      raise ValueError(f"Sampled variant must be a string; got {variant!r}.")

    values["variant"] = variant

    bundle = spec.variants.get(variant, {})

    if bundle is None:
      raise ValueError(f"Unknown variant {variant!r} for {spec.class_group}.{spec.class_name}.")

    bundle_parameters = bundle.get("parameters", {})

    # print("SELECTED VARIANT:", variant)
    # print("VARIANT BUNDLE:", bundle)
    # print("VARIANT PARAMETERS:", bundle_parameters)

    for name, raw_parameter in bundle_parameters.items():
      if "distribution" in raw_parameter:
        distribution = raw_parameter["distribution"]
      elif raw_parameter.get("kind") == "derived":
        distribution = {
          "type": "derived",
          **{
            key: value
            for key, value in raw_parameter.items()
            if key != "kind"
          },
        }
      else:
        raise ValueError(f"Parameter {name!r} has no distribution.")

      active[name] = ParameterSpec(
        name = name,
        kind = raw_parameter["kind"],
        distribution = distribution,
      )

  # print("ACTIVE PARAMETERS")
  # for name, parameter in active.items():
  #   print(name, parameter.kind, parameter.distribution,)

  pending = dict(active)
  consumed_overrides = {"variant"} if "variant" in overrides else set()

  while pending:
    progressed = False
    context = {**values, **derived}

    # print("PENDING", list(pending))
    # print("CONTEXT", context)

    for name, parameter in list(pending.items()):
      dependencies = parameter.distribution.get("depends_on", [])

      # print("CHECK:", name, "dependencies=", dependencies, "available=", context.keys())

      if not all(
        dependency in context
        for dependency in dependencies
      ):
        continue

      if name in overrides:
        _validate_override(
          _override_distribution(parameter.distribution, context),
          overrides[name],
          name,
        )
        result = overrides[name]
        decisions[name] = {"source": "override", "value": result}
        consumed_overrides.add(name)
      else:
        result = sample_distribution(
          spec = parameter.distribution,
          rng = rng,
          values = context,
          decisions = decisions,
          path = name,
        )

      target = (
        derived 
        if parameter.kind == "derived"
        else 
        values
      )

      target[name] = result
      del pending[name]
      progressed = True

    if not progressed:
      raise ValueError(f"Unresolvable sampling dependencies: {sorted(pending)}")
    
  topology = dict(spec.topology.raw)
  connector = topology.get("connector", {})

  if (
    isinstance(connector, dict)
    and connector.get("enabled")
    and connector.get("distribution")
  ):
    values["connector_type"] = sample_distribution(
      spec = connector["distribution"],
      rng = rng,
      values = {
        **values,
        **derived,
      },
      decisions = decisions,
      path = "topology.connector",
    )

  unused = set(overrides) - consumed_overrides
  if unused:
    raise ValueError(
      f"Sampling override(s) are not active for selected variant: {sorted(unused)!r}."
    )

  components = dict(spec.components)

  if variant in spec.variants:
    variant_components = spec.variants[variant].get("components", {})

    components.update({
      name: ComponentSpec(
        name = name,
        raw = component,
      )
      for name, component in variant_components.items()
    })

  context = {
    **values,
    **derived,
  }

  resolved_components: dict[str, dict[str, Any]] = {}

  for name, component in components.items():
    item = dict(component.raw)

    for key, value in item.items():
      if isinstance(value, str) and value in context:
        item[key] = context[value]

    resolved_components[name] = item

  return SamplingResult(
    parameters = values, 
    derived = derived, 
    topology = topology, 
    components = resolved_components, 
    decisions = decisions,
  )


def _validate_override(distribution: Mapping[str, Any], value: Any, path: str) -> None:
  """Validate a concrete override against a distribution's support."""
  dtype = distribution.get("type")
  if dtype == "derived":
    raise ValueError(f"Cannot override derived parameter {path!r}; it is recomputed.")
  if dtype == "categorical":
    if value not in distribution["probabilities"]:
      raise ValueError(f"Override {path}={value!r} is not one of the declared categorical values.")
    return
  if dtype == "fixed":
    if value != distribution.get("value"):
      raise ValueError(f"Override {path}={value!r} conflicts with fixed value {distribution.get('value')!r}.")
    return
  if dtype == "truncated_normal":
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
      raise ValueError(f"Override {path} must be a finite number.")
    if not float(distribution["min"]) <= float(value) <= float(distribution["max"]):
      raise ValueError(f"Override {path}={value!r} is outside [{distribution['min']}, {distribution['max']}].")
    return
  if dtype == "conditional":
    raise ValueError(f"Override {path!r} requires its dependency's case to be selected first.")
  if dtype == "mixture":
    if not any(_supports_override(component["distribution"], value) for component in distribution["components"]):
      raise ValueError(f"Override {path}={value!r} is outside all mixture component supports.")
    return
  raise ValueError(f"Unsupported override distribution {dtype!r} at {path}.")


def _override_distribution(
  distribution: Mapping[str, Any],
  context: Mapping[str, Any],
) -> Mapping[str, Any]:
  """Return the concrete branch used to validate a conditional override."""
  if distribution.get("type") != "conditional":
    return distribution
  dependency = distribution["depends_on"][0]
  if dependency not in context:
    raise ValueError(f"Cannot validate conditional override before {dependency!r} is resolved.")
  try:
    return distribution["cases"][context[dependency]]
  except KeyError as exc:
    raise ValueError(
      f"No conditional override case for {dependency}={context[dependency]!r}."
    ) from exc


def _supports_override(distribution: Mapping[str, Any], value: Any) -> bool:
  try:
    if distribution.get("type") == "conditional":
      return False
    _validate_override(distribution, value, "mixture")
    return True
  except (TypeError, ValueError, KeyError):
    return False
