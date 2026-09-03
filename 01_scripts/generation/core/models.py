from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
  
  Attributes:
    config_identity: Configuration identity used for sampling.
    class_group: Configuration group containing the sampled class.
    class_name: Sampled class name.
    seed: Optional seed associated with the sample.
    decisions: Discrete sampling decisions.
    parameters: Directly sampled parameter values.
    derived: Values derived from sampled parameters.
  """

  config_identity: ConfigIdentity
  class_group: str
  class_name: str
  seed: int | None = None
  decisions: dict[str, Any] = field(default_factory=dict)
  parameters: dict[str, Any] = field(default_factory=dict)
  derived: dict[str, Any] = field(default_factory=dict)


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