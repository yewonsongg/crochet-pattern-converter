from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import SamplingConfig, CacheMode
from .provenance import create_config_identity
from .validator import validate_ontology, validate_sampling_config, validate_ontology_sampling_agreement


def _load_yaml(
  path: Path,
  yaml_module: Any
) -> Any:
  """Load one YAML document from disk."""

  with path.open("r", encoding="utf-8") as handle:
    parsed = yaml_module.safe_load(handle)

  if parsed is None: 
    raise ValueError(f"YAML document is empty: {path}")

  return parsed


def load_sampling_config(
  *, 
  ontology_path: str | Path, 
  sampling_path: str | Path, 
  cache_mode: CacheMode = "lazy"
) -> SamplingConfig:
  """Load and validate ontology and symbol-sampling configuration.

  The loader reads both configuration documents, validates their individual structures, validates their cross-file agreement, creates configuration provenance, and constructs a :class:`SamplingConfig`.

  Args:
    ontology_path: Path to the canonical ontology YAML file.
    sampling_path: Path to the symbol-sampling YAML file.
    cache_mode: Whether class samplers are prepared lazily or eagerly.

  Returns:
    :class:`SamplingConfig` - validated and signed

  Raises:
    RuntimeError: If PyYAML dependency is unavailable.
    FileNoteFoundError: If either configuration file does not exist.
    ValueError: If either YAML document is empty.
    SamplingValidationError: If either document or their agreement is invalid.
  """

  try:
    import yaml
  except ImportError as exc:
    raise RuntimeError("Loading sampling YAML requires PyYAML.") from exc

  ontology_source = Path(ontology_path)
  sampling_source = Path(sampling_path)

  # load the YAMLs
  parsed_ontology = _load_yaml(
    path = ontology_source,
    yaml_module = yaml,
  )
  parsed_sampling = _load_yaml(
    path = sampling_source,
    yaml_module = yaml,
  )

  # validation - from `validator.py`
  validated_ontology = validate_ontology(
    parsed_source = parsed_ontology,
  )
  validated_sampling = validate_sampling_config(
    parsed_source = parsed_sampling,
  )

  # check agreement - from `validator.py`
  validate_ontology_sampling_agreement(
    ontology = validated_ontology,
    sampling = validated_sampling,
  )

  # get path, hash, and schema version - from `provenance.py`
  identity = create_config_identity(
    ontology_path = ontology_source,
    sampling_path = sampling_source,
    ontology_config = validated_ontology,
    sampling_config = validated_sampling,
    schema_version = validated_sampling["schema"].get("version")
  )

  # `SamplingConfig` construction handover to `config.py`
  return SamplingConfig(
    ontology_config = validated_ontology, 
    sampling_config = validated_sampling,
    identity = identity,
    cache_mode = cache_mode
  )
