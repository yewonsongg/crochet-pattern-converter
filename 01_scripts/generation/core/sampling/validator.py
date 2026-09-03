import math

from collections.abc import Mapping
from typing import Any, Mapping
from math import isclose, isfinite


class SamplingValidationError(ValueError):
  pass


SUPPORTED_DISTRIBUTIONS = {"categorical", "truncated_normal", "fixed", "conditional", "mixture", "derived"}

SUPPORTED_COORDINATE_CONVENTIONS = {"normalized_100"}

SUPPORTED_ANGLE_UNITS = {"degrees"}


def validate_sampling_config(parsed_source: Any) -> Mapping[str, Any]:
  """Validates the root structure of a symbol-sampling configuration.

  This function validates the presence and basic structure of the schema and class mappings, then delegates each class specification to :func:`_class`.

  It does not validate agreement with ``ontology.yaml``. That belongs in a separate cross-validation step.
  
  Args:
    parsed_source: Parsed YAML document.

  Returns:
    The validated source mapping.

  Raises:
    SamplingValidationError: If the root structure is invalid.
  """

  if not isinstance(parsed_source, Mapping):
    raise SamplingValidationError("Sampling config root must be a mapping.")

  schema = parsed_source.get("schema")
  if not isinstance(schema, Mapping):
    raise SamplingValidationError("Sampling config requires a schema mapping.")
  _schema(
    schema = schema
  )
  
  classes = parsed_source.get("classes")
  if not isinstance(classes, Mapping):
    raise SamplingValidationError("Sampling config classes mapping cannot be empty.")
  
  for group_name, group in classes.items():
    if not isinstance(group_name, str) or not group_name:
      raise SamplingValidationError("Each sampling class group name must be a non-empty string.")

    if not isinstance(group, Mapping):
      raise SamplingValidationError(f"classes.{group_name} must be a mapping.")

    if not group:
      raise SamplingValidationError(f"classes.{group_name} cannot be empty.")
    
    for class_name in group:
      if not isinstance(class_name, str) or not class_name:
        raise SamplingValidationError(f"classes.{group_name} contains an invalid class name: {class_name!r}")

  known = {
    class_name
    for group in classes.values()
    for class_name in group
  }

  for group_name, group in classes.items():
    for class_name, spec in group.items():
      _class(
        spec = spec,
        path = f"classes.{group_name}.{class_name}",
        known= known,
      )

  return parsed_source


def validate_ontology(parsed_source: Any) -> Mapping[str, Any]:
  """Validate and normalize the ontology configuration.

  Validates that the ontology contains a non-empty list of class entries with unique names, unique non-negative integer IDs, and supported class families.

  Digit-only string IDs are normalied to integers in place. For example, ``"07"`` becomes ``7``. Name whitespace normalization is also performed.

  Args:
    parsed_source: Parsed ontology YAML document.
  
  Returns:
    The validated ontology mapping.

  Raises:
    SamplingValidationError: If the ontology root, class entries, names, IDs, or families are invalid.  
  """

  if not isinstance(parsed_source, dict):
    raise SamplingValidationError("ontology must be a mapping.")

  classes = parsed_source.get("classes")
  if not isinstance(classes, list) or not classes:
    raise SamplingValidationError("ontology.classes must be a non-empty list.")

  allowed_families = {"primitive", "compound", "instructive"}
  names: set[str] = set()
  ids: set[int] = set()

  for index, entry in enumerate(classes):
    path = f"ontology.classes[{index}]"
    if not isinstance(entry, dict):
      raise SamplingValidationError(f"{path} must be a mapping.")

    for field_name in ("name", "id", "family"):
      if field_name not in entry:
        raise SamplingValidationError(f"{path}.{field_name} is required.")

    name = entry["name"]
    if not isinstance(name, str) or not name.strip():
      raise SamplingValidationError(f"{path}.name must be a non-empty string.")

    name = name.strip()
    entry["name"] = name
    
    if name in names:
      raise SamplingValidationError(f"{path}.name is duplicated: {name!r}.")
    names.add(name)

    class_id = entry["id"]
    if isinstance(class_id, str) and class_id.isdigit():
      class_id = int(class_id, 10)
      entry["id"] = class_id

    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
      raise SamplingValidationError(f"{path}.id must be a non-negative integer.")
    
    if class_id in ids:
      raise SamplingValidationError(f"{path}.id is duplicated: {class_id}.")
    
    ids.add(class_id)

    family = entry["family"]

    if not isinstance(family, str):
      raise SamplingValidationError(f"{path}.family must be a string.")

    if family not in allowed_families:
      raise SamplingValidationError(f"{path}.family must be one of {sorted(allowed_families)!r}.")

  return parsed_source


def validate_ontology_sampling_agreement(
  ontology: Mapping[str, Any],
  sampling: Mapping[str, Any],
) -> None:
  """Validate agreement between ontology and symbol-sampling configurations.

  Ensures that both configurations contain exactly the same ``(family, class_name)`` pairs and validates component references against the ontology's known class names.

  This function assumes that ``ontology`` and ``sampling`` have already passed their respective individual validators.

  Args:
    ontology: Validated ontology configuration.
    sampling: Validated symbol-sampling configuration.

  Raises:
    SamplingValidationError: If a sampling family is invalid, a sampling family or class mapping is malformed, the two configurations disagree, or a component references an unknown class.  
  """

  allowed_families = {"primitive", "compound", "instructive"}

  ontology_classes: dict[tuple[str, str], Mapping[str, Any]] = {}

  for index, entry in enumerate(ontology["classes"]):
    family = entry["family"]
    name = entry["name"]

    if not isinstance(family, str):
      raise SamplingValidationError(f"agreement.ontology.classes[{index}].family must be a string.")

    if not isinstance(name, str):
      raise SamplingValidationError(f"agreement.ontology.classes[{index}].name must be a string.")

    ontology_classes[(family, name)] = entry

  sampling_root = sampling.get("classes")

  if not isinstance(sampling_root, Mapping):
    raise SamplingValidationError("agreement.sampling.classes must be a mapping.")

  sampling_classes: dict[tuple[str, str], Any] = {}

  for family, classes in sampling_root.items():
    if not isinstance(family, str):
      raise SamplingValidationError(f"agreement.classes family must be a string; got {family!r}.")
    
    if family not in allowed_families:
      raise SamplingValidationError(f"agreement.classes.{family} is not a valid ontology family.")
    
    if not isinstance(classes, Mapping):
      raise SamplingValidationError(f"agreement.classes.{family} must be a mapping.")
    
    for name, spec in classes.items():
      if not isinstance(name, str):
        raise SamplingValidationError(f"agreement.calsses.{family} class name must be a string; got {name!r}.")
      
      key = (family, name)
      if key in sampling_classes:
        raise SamplingValidationError(f"agreement.class_duplicate: {key!r}.")
      
      sampling_classes[key] = spec

  ontology_keys = set(ontology_classes)
  sampling_keys = set(sampling_classes)

  missing = sorted(ontology_keys - sampling_keys)

  if missing:
    raise SamplingValidationError(f"agreement.class_missing: {missing!r}.")

  extra = sorted(set(sampling_classes) - set(ontology_classes))

  if extra:
    raise SamplingValidationError(f"agreement.class_unknown: {extra!r}.")

  known_names = set(
    entry["name"] 
    for entry in ontology["classes"]
  )

  _validate_component_references(
    source = sampling, 
    known_names = known_names
  )


def _validate_component_references(
  source: Mapping[str, Any], 
  known_names: set[str]
) -> None:
  """Validate class references nested in component declarations.

  Recursively traverses mappings and lists beneath ``source["classes"]``. 
  Whenever it encounters a mapping key named ``"class"``, it verifies that the associated value is a known class name.

  Args: 
    source: Validated sampling configuration mapping.
    known_names: Known ontology class names.

  Raises:
    SamplingValidationError: If a ``class`` reference is not a string or does not identify a known class.
  """

  def walk(
    value: Any, 
    path: str
  ) -> None:
    if isinstance(value, Mapping):
      for key, child in value.items():
        child_path = f"{path}.{key}"

        if key == "class":
          if not isinstance(child, str):
            raise SamplingValidationError(f"agreement.{child_path} must be a class-name string; got {child!r}.")

        if child not in known_names:
          raise SamplingValidationError(f"agreement.{child_path} references unknown class: {child!r}.")
        
        walk(child, child_path)

    elif isinstance(value, list):
      for index, child in enumerate(value):
        walk(child, f"{path}[{index}]")

  walk(source.get("classes", {}), "classes")


def _schema(schema): 
  """Validate sampling-configuration schema metadata.

  Checks that the schema contains the required version, coordinate-convention, and angle-unit fields, and rejects unsupported fields or values.

  Args:
    schema: Parsed ``schema`` mapping from the sampling configuration.

  Raises:
    SamplingValidationError: If the schema is not a mapping, is missing required keys, contains unsupported keys, or contains invalid values.
  """

  if not isinstance(schema, Mapping):
    raise SamplingValidationError("Sampling config schema must be a mapping.")

  required_keys = {
    "version",
    "coordinate_convention",
    "angle_units",
  }

  missing_keys = required_keys - schema.keys()
  if missing_keys:
    missing = ", ".join(sorted(repr(key) for key in missing_keys))
    raise SamplingValidationError(f"Sampling config schema is missing required key(s): {missing}.")

  unknown_keys = set(schema) - required_keys
  if unknown_keys:
    unknown = ", ".join(sorted(repr(key) for key in unknown_keys))
    raise SamplingValidationError(f"Sampling config schema contains unsupported key(s): {unknown}.")

  version = schema["version"]
  # bool is technically an int subclass so reject it explicitly
  if isinstance(version, bool) or not isinstance(version, int):
    raise SamplingValidationError(f"schema.version must be a positive integer.")
  if version < 1:
    raise SamplingValidationError(f"schema.version must be a >= 1; got {version}.")

  coordinate_convention = schema["coordinate_convention"]
  if not isinstance(coordinate_convention, str):
    raise SamplingValidationError("schema.coordinate_convention must be a string.")
  if coordinate_convention not in SUPPORTED_COORDINATE_CONVENTIONS:
    raise SamplingValidationError(f"schema.coordinate_convention unsupported: {coordinate_convention!r}.")

  angle_units = schema["angle_units"]
  if not isinstance(angle_units, str):
    raise SamplingValidationError("schema.angle_units must be a string.")
  if angle_units not in SUPPORTED_ANGLE_UNITS:
    raise SamplingValidationError(f"schema.angle_units unsupported: {angle_units!r}.")

  
def _class(
  spec: Any, 
  path: str, 
  known: set[str]
):
  """Validate one class sampling specification.

  Validates class-level parameters, variant bundles, component references, variant reachability, and connector distribution declarations.

  Args:
    spec: Parsed class specification.
    path: Human-readable configuration path used in error messages.
    known: Known ontology class names for component-reference validation.

  Raises:
    SamplingValidationError: If the class specification is malformed or internally inconsistent.
  """

  if not isinstance(spec, Mapping):
    raise SamplingValidationError(f"{path} must be a mapping.")
  
  params = spec.get("parameters", {})
  if not isinstance(params, Mapping):
    raise SamplingValidationError(f"{path}.parameters must be a mapping.")
  
  for name, value in params.items():
    if not isinstance(name, str):
      raise SamplingValidationError(f"{path}.parameters contains a non-string name: {path!r}.")
    
    _parameter(
      value = value, 
      path = f"{path}.parameters.{name}", 
      parameters = params,
    )

  variants = spec.get("variants", {}) or {}
  if not isinstance(variants, Mapping):
    raise SamplingValidationError(f"{path}.variants must be a mapping.")
  
  for variant, bundle in variants.items():
    if not isinstance(variant, str) or not variant:
      raise SamplingValidationError(f"{path}.variants contains an invalid variant name: {variant!r}.")
    
    if not isinstance(bundle, Mapping):
      raise SamplingValidationError(f"{path}.variants.{variant} must be a mapping.")
    
    bundle_params = bundle.get("parameters", {})
    if not isinstance(bundle_params, Mapping):
      raise SamplingValidationError(f"{path}.variants.{variant}.parameters must be a mapping.")
    
    active_params = {**params, **bundle_params}
    for name, value in bundle_params.items():
      if not isinstance(name, str):
        raise SamplingValidationError(f"{path}.variants.{variant}.parameters contains a non-string name: {name!r}.")
      
      _parameter(
        value = value, 
        path = f"{path}.variants.{variant}.parameters.{name}", 
        parameters = active_params,
      )

    _components(
      components = bundle.get("components", {}), 
      path = f"{path}.variants.{variant}.components", 
      known = known,
    )

  variant_parameter = params.get("variant")

  if variants and variant_parameter is not None:
    if not isinstance(variant_parameter, Mapping):
      raise SamplingValidationError(f"{path}.parameters.variant must be a mapping.")

    distribution = variant_parameter.get("distribution")

    if isinstance(distribution, Mapping):
      if distribution.get("type") == "categorical":
        probabilities = distribution.get("probabilities", {})

        if not isinstance(probabilities, Mapping):
          raise SamplingValidationError(f"{path}.parameters.variant.distribution.probabilities must be a mapping.")

        declared = set(variants)
        reachable = set(probabilities)

        if declared != reachable:
          raise SamplingValidationError(f"{path}.variants must match reachable values of variant: expected {sorted(reachable)!r}, got {sorted(declared)!r}.")

  _components(
    components = spec.get("components", {}), 
    path = f"{path}.components", 
    known = known,
  )

  topology = spec.get("topology", {})

  if topology is None:
    topology = {}
  
  if not isinstance(topology, Mapping):
    raise SamplingValidationError(f"{path}.topology must be a mapping.")

  connector = topology.get("connector")
  if connector is None:
    return

  if not isinstance(connector, Mapping):
    raise SamplingValidationError(f"{path}.topology.connector must be a mapping.")

  enabled = connector.get("enabled")
  if enabled is not None and not isinstance(enabled, bool):
    raise SamplingValidationError(f"{path}.topology.connector.enabled must be a boolean.")

  distribution = connector.get("distribution")
  if enabled is False and distribution is not None:
    raise SamplingValidationError(f"{path}.topology.connector cannot have a distribution when disabled.")

  if distribution is not None:
    _distribution(
      spec = distribution,
      path = f"{path}.topology.connector.distribution",
      parameters = params,
    )


def _parameter(
  value: Any, 
  path: str, 
  parameters: Mapping[str, Any],
) -> None:
  """Validate one parameter declaration.

  A parameter must declare a string ``kind`` and either provide an explicit ``distribution`` mapping or use the shorthand derived-parameter form.

  For example:
  
  .. code-block:: yaml
    
    diameter:
      kind: derived
      formula: "count * chain_pitch / pi"
      depends_on: [count, chain_pitch]

  is normalized internally to:

  .. code-block:: yaml

    diameter:
      kind: derived
      distribution:
        type: derived
        formula: "count * chain_pitch / pi"
        depends_on: [count, chain_pitch]

  Args:
    value: Parsed parameter declaration.
    path: Configuration path used in error messages.
    parameters: Parameter names available for dependency validation.

  Raises:
    SamplingValidationError: If the declaration is malformed or its distribution is invalid.  
  """

  if not isinstance(value, Mapping):
    raise SamplingValidationError(f"{path} must be a mapping.")

  kind = value.get("kind")
  if not isinstance(kind, str) or not kind:
    raise SamplingValidationError(f"{path}.kind must be a non-empty string.")
 
  distribution = value.get("distribution")
  if distribution is None and value.get("kind") == "derived":
    distribution = {
      "type": "derived", 
      **{
        key: item 
        for key, item in value.items() 
        if key != "kind"
      },
    }

  if not isinstance(distribution, Mapping):
    raise SamplingValidationError(f"{path}.distribution must be a mapping.")
  
  _distribution(
    spec = distribution, 
    path = f"{path}.distribution", 
    parameters = parameters,
  )


def _distribution(
  spec: Any, 
  path: str, 
  parameters: Mapping[str, Any]
) -> None:
  """Validate one distribution specification.

  Supports categorical, truncated-normal, fixed, conditional, mixture, and derived distributions. Conditional dependencies and derived dependencies must refer to known parameters in ``parameters``.

  Args:
    spec: Parsed distribution specifications.
    path: Configuration path used in error messages.
    parameters: Parameters available for dependency validation.

  Raises:
    SamplingValidationError: If the distribution is malformed, unsupported, numerically invalid, or references unkonwn parameters.
  """

  def _finite_number(
    value: Any,
    path: str,
  ) -> float:
    """Convert a value to a finite float or raise a sampling error."""

    try: 
      number = float(value)
    except (TypeError, ValueError) as exc:
      raise SamplingValidationError(f"{path} must be numeric; got {value!r}.")

    if not isfinite(number):
      raise SamplingValidationError(f"{path} must be finite; got {value!r}.")

    return number

  if not isinstance(spec, Mapping):
    raise SamplingValidationError(f"{path} must be a mapping.")

  dtype = spec.get("type")
  if not isinstance(dtype, str):
    raise SamplingValidationError(f"{path}.type must be a string.")
  if dtype not in SUPPORTED_DISTRIBUTIONS:
    raise SamplingValidationError(f"{path}.type unsupported: {dtype!r}")
  
  if dtype == "categorical":
    probs = spec.get("probabilities")
    if not isinstance(probs, Mapping) or not probs:
      raise SamplingValidationError(f"{path}.probabilities must be a non-empty mapping.")

    values = []
    for key, value in probs.items():
      if isinstance(key, (dict, list, set)):
        raise SamplingValidationError(f"{path}.probabilities contains an invalid key: {key!r}.")

      values.append(_finite_number(
        value = value,
        path = f"{path}.probabilities.{key}",
      ))

    if any(value < 0 for value in values):
      raise SamplingValidationError(f"{path}.probabilities must be non-negative.")

    if not isclose(sum(values), 1.0, abs_tol=1e-6,):
      raise SamplingValidationError(f"{path}.probabilities must sum to 1.")

        
  elif dtype == "truncated_normal":
    required = ("mean", "std", "min", "max")
    missing = [
      key 
      for key in required
      if key not in spec
    ]
    
    if missing:
      raise SamplingValidationError(f"{path} requires {', '.join(missing)}.")

    mean = _finite_number(
      value = spec["mean"],
      path = f"{path}.mean",
    )
    std = _finite_number(
      value = spec["std"],
      path = f"{path}.std",
    )
    minimum = _finite_number(
      value = spec["min"],
      path = f"{path}.min",
    )
    maximum = _finite_number(
      value = spec["max"],
      path = f"{path}.max",
    )

    if std <= 0:
      raise SamplingValidationError(f"{path}.std must be positive.")
    if minimum > maximum:
      raise SamplingValidationError(f"{path}.min must not exceed {path}.max .")
    if not minimum <= mean <= maximum:
      raise SamplingValidationError(f"{path}.mean must lie between min and max.")
 
    
  elif dtype == "fixed": 
    if "value" not in spec:
      raise SamplingValidationError(f"{path}.value required")

  
  elif dtype == "conditional":
    dependency = spec.get("depends_on")
    cases = spec.get("cases")

    if not isinstance(dependency, str):
      raise SamplingValidationError(f"{path}.depends_on must be a parameter name.")
    if dependency not in parameters:
      raise SamplingValidationError(f"{path}.depends_on references unknown parameter: {dependency!r}.")
    if not isinstance(cases, Mapping) or not cases:
      raise SamplingValidationError(f"{path}.cases must be a non-empty mapping.")

    dependency_spec = parameters[dependency]
    if not isinstance(dependency_spec, Mapping):
      raise SamplingValidationError(f"{path} dependency {dependency_spec!r} must be a parameter mapping.")
        
    dependency_distribution = dependency_spec.get("distribution")
    if isinstance(dependency_distribution, Mapping):
      if dependency_distribution.get("type") == "categorical":
        probabilities = (dependency_distribution.get("probabilities"))
        if isinstance(probabilities, Mapping):
          expected = set(probabilities)
          actual = set(cases)

          if expected != actual:
            raise SamplingValidationError(f"{path}.cases must match reachable values of {dependency}: expected {sorted(expected)!r}, got {sorted(actual)!r}.")
    
    for key, case in cases.items():
      if not isinstance(case, Mapping):
        raise SamplingValidationError(f"{path}.cases.{key} must be a mapping.")
      
      _distribution(
        spec = case, 
        path = f"{path}.cases.{key}", 
        parameters = parameters,
      )


  elif dtype == "mixture":
    components = spec.get("components")
    if not isinstance(components, list) or not components:
      raise SamplingValidationError(f"{path}.components must be a non-empty list.")
    
    weights: list[float] = []
    for index, component in enumerate(components):
      component_path = f"{path}.components[{index}]"

      if not isinstance(component, Mapping):
        raise SamplingValidationError(f"{component_path} must be a mapping.")

      if "weight" not in component:
        raise SamplingValidationError(f"{component_path}.weight is required.")

      weights.append(
        _finite_number(
          value = component["weight"],
          path = f"{component_path}.weight",
        ))

      nested = component.get("distribution")
      if not isinstance(nested, Mapping):
        raise SamplingValidationError(f"{component_path}.distribution must be a mapping.")

      _distribution(
        spec = nested,
        path = f"{component_path}.distribution",
        parameters = parameters,
      )

    if any(weight < 0 for weight in weights):
      raise SamplingValidationError(f"{path}.weights must be non-negative.")

    if not isclose(sum(weights), 1.0, abs_tol=1e-6):
      raise SamplingValidationError(f"{path}.weights must sum to 1.")


  elif dtype == "derived":
    formula = spec.get("formula")
    dependencies = spec.get("depends_on", [])

    if not isinstance(formula, str) or not formula.strip():
      raise SamplingValidationError(f"{path}.formula must be a non-empty string.")
    
    if not isinstance(dependencies, list):
      raise SamplingValidationError(f"{path}.depends_on must be a list.")

    for dependency in dependencies:
      if not isinstance(dependency, str):
        raise SamplingValidationError(f"{path}.depends_on entries must be strings.")
      if dependency not in parameters:
        raise SamplingValidationError(f"{path}.depends_on references unknown parameter: {dependency!r}.")


def _components(
  components: Any, 
  path: str, 
  known: set[str],
) -> None:

  """Validate component declarations and referenced class names.

  Each component declaration must be a mapping. If it includes a ``class`` field, that field must contain the name of a known ontology class.

  Args:
    components: Parsed component declarations.
    path: Configuration path used in error messages.
    known: Known ontology class names.

  Raises:
    SamplingValidationError: If component declarations are malformed or a component references an unknown class.
  """

  if components is None:
    return

  if not isinstance(components, Mapping):
    raise SamplingValidationError(f"{path} must be a mapping")
  
  for name, component in components.items():
    if not isinstance(name, str) or not name:
      raise SamplingValidationError(f"{path} contains an invalid component name: {name!r}.")
    
    if not isinstance(component, Mapping):
      raise SamplingValidationError(f"{path}.{name} must be a mapping.")

    if "class" not in component:
      continue

    component_class = component["class"]

    if not isinstance(component_class, str) or not component_class:
      raise SamplingValidationError(f"{path}.{name}.class must be a non-empty class name string.")
    
    if component_class not in known:
      raise SamplingValidationError(f"{path}.{name}.class references unknown class: {component_class!r}.")
