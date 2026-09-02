from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ParameterSpec:
  name: str
  kind: str
  distribution: dict[str, Any]


@dataclass(frozen=True)
class DistributionSpec:
  type: str
  raw: dict[str, Any]


@dataclass(frozen=True)
class ComponentSpec:
  name: str
  raw: dict[str, Any]


@dataclass(frozen=True)
class TopologySpec:
  raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassSpec:
  class_group: str
  class_name: str
  parameters: dict[str, ParameterSpec] = field(default_factory=dict)
  variants: dict[str, dict[str, Any]] = field(default_factory=dict)
  topology: TopologySpec = field(default_factory=TopologySpec)
  components: dict[str, ComponentSpec] = field(default_factory=dict)
  raw: dict[str, Any] = field(default_factory=dict)
