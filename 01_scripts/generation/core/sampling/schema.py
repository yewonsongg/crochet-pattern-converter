from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ParameterSpec:
  """Resolved specification for one clas parameter.
  
  Attributes;
    name: Parameter name.
    kind: Parameter category, such as ``"discrete"``, ``"continuous"``, or ``"derived"``.
    distribution: Normalized distribution declaration used to sample the parameter.
  """

  name: str
  kind: str
  distribution: dict[str, Any]


@dataclass(frozen=True)
class DistributionSpec:
  """Resolved description of one sampling distribution.

  This model is currently available for future use. At present, parameter distributions are stored as raw dictionaries in ``ParameterSpec``.

  Attributes:
    type: Distribution type, such as ``"categorical"``, or ``"truncated-normal"``.
    raw: Complete normalized distribution declaration.
  """

  type: str
  raw: dict[str, Any]


@dataclass(frozen=True)
class ComponentSpec:
  """Specification for one component of a compound class.

  Attributes:
    name: Component name within the enclosing class.
    raw: Component declaration, including its referenced class and any component-specific parameters.
  """

  name: str
  raw: dict[str, Any]


@dataclass(frozen=True)
class TopologySpec:
  """Structural specification for a class or compound.
  
  Attributes:
    raw: Normalized topology declaration, such as relations between component parts or connector configuration.
  """

  raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassSpec:
  """Resolved sampling specification for one ontology class.

  A ``ClassSpec`` binds a class identity to its parameter distributions, optional variants, topology, components, and normalized source metadata.

  Attributes:
    class_group: Ontology/sampling group containing the class.
    class_name: Canonical class name.
    parameters: Resolved parameter specifications keyed by parameter name.
    variants: Variant-specific parameter and component declarations.
    topology: Resolved topology specification.
    components: Resolved component specifications keyed by component name.
    raw: Normalized source fragment for the class.
  """

  class_group: str
  class_name: str
  parameters: dict[str, ParameterSpec] = field(default_factory=dict)
  variants: dict[str, dict[str, Any]] = field(default_factory=dict)
  topology: TopologySpec = field(default_factory=TopologySpec)
  components: dict[str, ComponentSpec] = field(default_factory=dict)
  raw: dict[str, Any] = field(default_factory=dict)
