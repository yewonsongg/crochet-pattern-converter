from __future__ import annotations

import math

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping
from math import isclose, isfinite

import numpy as np


@dataclass(frozen=True)
class ConfigIdentity:
  """Identity and provenance of the configuration sources.

  Attributes:
    ontology_path: Resolved path to the ontology configuration.
    ontology_digest: Content digest of the ontology configuration.
    sampling_path: Resolved path to the symbol-sampling configuration.
    sampling_digest: Content digest of the symbol-sampling configuration.
    schema_version: Schema version used to interpret the configuration.
  """

  ontology_path: str
  ontology_digest: str
  sampling_path: str
  sampling_digest: str
  schema_version: Any


@dataclass(frozen=True)
class SamplingProvenance:
  """Provenance metadata for one sampled class instance.
  
  Records the configuration identity, class identity, random seed, sampling decisions, resulting values, and any explicit overrides used to produce one sampled instance.

  Attributes:
    config_identity: Configuration identity used for sampling.
    class_group: Configuration group containing the sampled class.
    class_name: Sampled class name.
    seed: Optional seed associated with the sample.
    decisions: Discrete sampling decisions.
    parameters: Directly sampled parameter values.
    derived: Values derived from sampled parameters.
    overrides: Parameter values explicitly supplied by the caller instead of being sampled naturally.
    case_id: Optional identifier for the inspection, regression, or coverage case that produced this sample.
  """

  config_identity: ConfigIdentity
  class_group: str
  class_name: str
  seed: int | None = None
  decisions: dict[str, Any] = field(default_factory=dict)
  parameters: dict[str, Any] = field(default_factory=dict)
  derived: dict[str, Any] = field(default_factory=dict)
  overrides: dict[str, Any] = field(default_factory=dict)
  case_id: str | None = None


@dataclass(frozen=True)
class SamplingResult:
  """Intermediate result produced by class-level sampling.
  
  Attributes:
    parameters: Directly sampled parameter values.
    derived: Values calculated from sampled parameters.
    topology: Resolved topology information.
    components: Resolved component declarations.
    decisions: Discrete sampling decisions.
  """

  parameters: dict[str, Any]
  derived: dict[str, Any]
  topology: dict[str, Any]
  components: dict[str, Any]
  decisions: dict[str, Any]


@dataclass(frozen=True)
class SampledParameters:
  """Concrete sampled values for one generated class instance.

  Attributes:
    parameters: Directly sampled values.
    derived: Values computed from the sampled values.
    topology: Resolved topology information.
    components: Resolved component declarations.
    provenance: Optional provenance record for one sample.
  """

  def as_dict(self) -> dict[str, Any]:
    """Return direct and derived values as one mapping.
    
    Derived values override direct values when both mappings contain the same key.
    """

    return {**self.parameters, **self.derived}

  parameters: dict[str, Any]
  derived: dict[str, Any] = field(default_factory=dict)
  topology: dict[str, Any] = field(default_factory=dict)
  components: dict[str, Any] = field(default_factory=dict)
  provenance: SamplingProvenance | None = None


@dataclass(frozen=True)
class GenerationConfig:
  """Rendering configuration for one generated symbol.

  This configuration controls the isolated rendering context used by a class generator. It describes canvas dimensions, target visible-symbol size, visual rotation, and normalized SVG stroke width.

  Attributes:
    canvas_width_px: Output canvas width in pixels.
    canvas_height_px: Output canvas height in pixels.
    target_visible_px: Target visible-symbol size in pixels.
    rotation_deg: Visual rotation applied to the symbol.
    stroke_width_normalized: Stroke width in normalized SVG coordinates.
  """

  canvas_width_px: int = 25
  canvas_height_px: int = 25
  target_visible_px: float = 15.0
  rotation_deg: float = 0.0
  stroke_width_normalized: float = 4.0

  def with_overrides(
    self, 
    overrides: dict[str, Any] | None = None
  ) -> "GenerationConfig":
    """Return a copy with validated rendering overrides applied.
    
    The original configuration is not modified. Only fields declared by ``GenerationConfig`` may be overridden.

    Args:
      overrides: Mapping of field names to replacement values. If ``None`` or empty, an equivalent copy of the current configuration is returned.

    Returns:
      A new validated ``GenerationConfig`` instance.

    Raises: 
      KeyError: If ``overrides`` contains an unknown configuration field.
      ValueError: If the resulting configuration contains invalid dimensions
      TypeError: If ``overrides`` is not a mapping.
    """

    if overrides is not None and not isinstance(overrides, Mapping):
      raise TypeError("Generation override must be a mapping or None.")

    values = {
      field.name: getattr(self, field.name) 
      for field in fields(self)
    }

    for name, value in (overrides or {}).items():
      if name not in values:
        raise KeyError(f"Unknown generation override: {name!r}.")
      values[name] = value

    result = GenerationConfig(**values)

    if (
      isinstance(result.canvas_width_px, bool)
      or not isinstance(result.canvas_width_px, int)
      or result.canvas_width_px <= 0
    ):
      raise ValueError("canvas_width_px must be a positive integer.")

    if (
      isinstance(result.canvas_height_px, bool)
      or not isinstance(result.canvas_width_px, int)
      or result.canvas_height_px <= 0
    ):
      raise ValueError("canvas_height_px must be a positive integer.")

    for name in ("target_visible_px", "rotation_deg", "stroke_width_normalized"):
      value = getattr(result, name)

      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")

      if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")

    if result.target_visible_px <= 0:
      raise ValueError("target_visible_px muts be positive.")

    if result.stroke_width_normalized <= 0:
      raise ValueError("stroke_width_normalized must be positive.")
    
    return result


@dataclass
class GeneratedObject:
  """In-memory representation of one generated symbol or compound.

  A ``GeneratedObject`` contains the generated SVG along with the metadata required for inspection, labeling, serialization, and later scene orchestration. It can be used without writing any files to disk.

  The object may be created before downstream artifacts such as OBB labels or output paths are available, depending on the generation pipeline.

  Attributes:
    class_id: Numeric ontology/class identifer used by the detector.
    class_name: Internal class name, such as ```"ch"``` or ```"together"```.
    variant_id: Optional discrete visual phenotype or variant identifier.
    svg: Serialized SVG representation of the generated object.
    metadata: Generation metadata, including geometry, canvas, and sampled configuration information.
    obb_pixels: Four OBB corner points in pixel coordinates, typically stored as an ``(4, 2)`` NumPy array.
    obb_normalized: Four OBB corner points normalized to the model's expected coordinate convention, typically stored as an ``(4, 2)`` NumPy array.
    yolo_label: Serialized YOLO OBB label line for this object.
    sampled_parameters: Concrete parameter values used to generate the object.
    sampling_provenance: Optional record identifying the configuration, class, decisions, parameters, derived values, and seed used to sample the object.
    svg_path: Optional path where the SVG was written.
    png_path: Optional path where a rasterized PNG was written.
    metadata_path: Optional path where metadata was written.
    label_path: Optional path where the YOLO OBB label was written.
    extras: Additional implementation- or inspection-specific metadata.
  """

  class_id: int
  class_name: str
  variant_id: str | None
  svg: str
  metadata: dict[str, Any]
  obb_pixels: np.ndarray
  obb_normalized: np.ndarray
  yolo_label: str
  sampled_parameters: dict[str, Any] = field(default_factory=dict)
  sampling_provenance: SamplingProvenance | None = None
  svg_path: Path | None = None
  png_path: Path | None = None
  metadata_path: Path | None = None
  label_path: Path | None = None
  extras: dict[str, Any] = field(default_factory=dict)

  @property
  def canvas_size_px(self) -> tuple[int, int]:
    """Return the generated canvas dimensions in pixels.
    
    Returns:
      A ``(width, height)`` tuple in pixels.

    If canvas metadata is absent, ``(0, 0)`` is returned.
    """
    
    canvas = self.metadata.get("canvas", {})
    return canvas.get("width_px", 0), canvas.get("height_px", 0)
