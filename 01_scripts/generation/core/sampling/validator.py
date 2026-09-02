import math
from typing import Any


class SamplingValidationError(ValueError):
  pass


SUPPORTED_DISTRIBUTIONS = {"categorical", "truncated_normal", "fixed", "conditional", "mixture", "derived"}


def validate_sampling(parsed_source: Any) -> Any:
  """
  
  """

  if not isinstance(parsed_source, dict):
    raise SamplingValidationError("Sampling config root must be a mapping")
  if not isinstance(parsed_source.get("schema"), dict):
    raise SamplingValidationError("Missing schema mapping")
  
  classes = parsed_source.get("classes")
  if not isinstance(classes, dict):
    raise SamplingValidationError("Missing classes mapping")
  
  known = {name for group in classes.values() if isinstance(group, dict) for name in group}
  for group_name, group in classes.items():
    if not isinstance(group, dict):
      raise SamplingValidationError(f"classes.{group_name} must be a mapping")
    
    for name, spec in group.items():
      _class(spec, f"classes.{group_name}.{name}", known)


def validate_ontology(parsed_source: Any) -> Any:
  print("Hello World")


def validate_ontology_sampling_agreement(ontology: Any, sampling: Any) -> Any:
  print("Hello World")


def _class(spec, path, known):
  """
  
  """

  if not isinstance(spec, dict):
    raise SamplingValidationError(f"{path} must be a mapping")
  
  params = spec.get("parameters", {})
  if not isinstance(params, dict):
    raise SamplingValidationError(f"{path}.parameters must be a mapping")
  
  names = set(params)
  for name, value in params.items():
    _parameter(value, f"{path}.parameters.{name}", params)

  variants = spec.get("variants", {}) or {}
  if not isinstance(variants, dict):
    raise SamplingValidationError(f"{path}.variants must be a mapping")
  
  for variant, bundle in variants.items():
    if not isinstance(bundle, dict):
      raise SamplingValidationError(f"{path}.variants.{variant} must be a mapping")
    
    bundle_params = bundle.get("parameters", {})
    if not isinstance(bundle_params, dict):
      raise SamplingValidationError(f"{path}.variants.{variant}.parameters must be a mapping")
    
    active_params = {**params, **bundle_params}
    for name, value in bundle_params.items():
      _parameter(value, f"{path}.variants.{variant}.parameters.{name}", active_params)

    _components(bundle.get("components", {}), f"{path}.variants.{variant}.components", known)

  variant_parameter = params.get("variant")
  if variant_parameter and variant_parameter.get("distribution", {}).get("type") == "categorical":
    declared = set(variants)
    reachable = set(variant_parameter["distribution"].get("probabilities", {}))
    if declared != reachable:
      raise SamplingValidationError(
        f"{path}.variants must match reachable values of variant: "
        f"expected {sorted(reachable)!r}, got {sorted(declared)!r}"
      )

  _components(spec.get("components", {}), f"{path}.components", known)

  topology = spec.get("topology", {})
  connector = topology.get("connector") if isinstance(topology, dict) else None
  if isinstance(connector, dict) and connector.get("enabled") is False and "distribution" in connector:
    raise SamplingValidationError(f"{path}.topology.connector cannot have distribution when disabled")
  
  if isinstance(connector, dict) and connector.get("enabled") and connector.get("distribution"):
    _distribution(connector["distribution"], f"{path}.topology.connector.distribution", params)


def _parameter(value, path, parameters):
  """
  
  """

  if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
    raise SamplingValidationError(f"{path} requires kind")
  
  distribution = value.get("distribution")
  if not isinstance(distribution, dict):
    raise SamplingValidationError(f"{path}.distribution must be a mapping")
  
  _distribution(distribution, f"{path}.distribution", parameters)


def _distribution(spec, path, parameters):
  """
  
  """

  dtype = spec.get("type")
  if dtype not in SUPPORTED_DISTRIBUTIONS:
    raise SamplingValidationError(f"{path}.type unsupported: {dtype!r}")
  
  if dtype == "categorical":
    probs = spec.get("probabilities")
    if not isinstance(probs, dict) or not probs or any(float(x) < 0 for x in probs.values()) or not math.isclose(sum(map(float, probs.values())), 1, 
    abs_tol=1e-6):
      raise SamplingValidationError(f"{path}.probabilities must be non-negative and sum to 1")
    
  elif dtype == "truncated_normal":
    if any(key not in spec for key in ("mean", "std", "min", "max")):
      raise SamplingValidationError(f"{path} requires mean, std, min, max")
    
    if float(spec["std"]) <= 0 or float(spec["min"]) > float(spec["max"]):
      raise SamplingValidationError(f"{path} has invalid std or bounds")
    
  elif dtype == "fixed" and "value" not in spec:
    raise SamplingValidationError(f"{path}.value required")
  
  elif dtype == "conditional":
    dependency = spec.get("depends_on")
    if dependency not in parameters or not isinstance(spec.get("cases"), dict):
      raise SamplingValidationError(f"{path} requires known depends_on and cases")
    
    dependency_distribution = parameters[dependency].get("distribution", {})
    if dependency_distribution.get("type") == "categorical":
      expected = set(dependency_distribution.get("probabilities", {}))
      actual = set(spec["cases"])
      if expected != actual:
        raise SamplingValidationError(
          f"{path}.cases must match reachable values of {dependency}: "
          f"expected {sorted(expected)!r}, got {sorted(actual)!r}"
        )
    
    for key, case in spec["cases"].items():
      if not isinstance(case, dict):
        raise SamplingValidationError(f"{path}.cases.{key} must be a mapping")
      _distribution(case, f"{path}.cases.{key}", parameters)

  elif dtype == "mixture":
    components = spec.get("components")
    if not isinstance(components, list) or not components:
      raise SamplingValidationError(f"{path}.components must be non-empty")
    
    weights = [float(c.get("weight", -1)) for c in components]
    if any(x < 0 for x in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-6):
      raise SamplingValidationError(f"{path}.weights must be non-negative and sum to 1")
    
    for index, component in enumerate(components):
      nested = component.get("distribution")
      if not isinstance(nested, dict):
        raise SamplingValidationError(f"{path}.components[{index}].distribution required")
      _distribution(nested, f"{path}.components[{index}].distribution", parameters)

  elif dtype == "derived":
    if not spec.get("formula"):
      raise SamplingValidationError(f"{path}.formula required")
    
    for dependency in spec.get("depends_on", []):
      if dependency not in parameters:
        raise SamplingValidationError(f"{path}.depends_on unknown parameter: {dependency!r}")


def _components(components, path, known):
  """
  
  """

  if not components:
    return
  if not isinstance(components, dict):
    raise SamplingValidationError(f"{path} must be a mapping")
  
  for name, component in components.items():
    if not isinstance(component, dict):
      raise SamplingValidationError(f"{path}.{name} must be a mapping")
    
    if component.get("class") and component["class"] not in known:
      raise SamplingValidationError(f"{path}.{name}.class unknown: {component['class']!r}")
