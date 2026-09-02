import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import ConfigIdentity, SamplingProvenance


def config_hash(raw: Any) -> str:
  """Compute stable hash of parsed YAML content."""

  payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_source(path: Path, raw: Any) -> tuple[str, str]:
  """
  Normalize the source path and obtain stable hash.
  
  Returns the identity information (path, hash) needed by `create_config_identity()`.
  """

  return str(path.resolve()), config_hash(raw)


def create_config_identity(
  ontology_path: Path, 
  sampling_path: Path,
  ontology_config: Any,
  sampling_config: Any, 
  schema_version: Any
) -> ConfigIdentity:
  """
  Authors the `ConfigIdentity` record for config provenance from resolved path, hash, and schema version.

  Returns `ConfigIdentity`.
  """

  resolved, digest = config_source(sampling_path, sampling_config)
  return ConfigIdentity(resolved, digest, schema_version)


def create_sampling_provenance(
  config_identity: ConfigIdentity, 
  class_group: str, 
  class_name: str, 
  seed: int | None, 
  decisions: dict[str, Any], 
  parameters: dict[str, Any], 
  derived: dict[str, Any]
) -> SamplingProvenance:
  """
  Authors the `SamplingProvenance` record for sampling provenance.

  Returns `SamplingProvenance`.
  """

  return SamplingProvenance(config_identity, class_group, class_name, seed, decisions, parameters, derived)
