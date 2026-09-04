import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..models import ConfigIdentity, SamplingProvenance


def config_hash(
  source: Any
) -> str:
  """Compute a deterministic SHA-256 digest for parsed configuration data.

  Configuration keys are sorted and JSON separators are minimized so that equivalent mappings produce the same digest regardless of Mappingionary insertion order.

  Args:
    source: Parsed configuration data. It should contain only values that can be represented deterministically by ``json.dumps``.

  Returns:
    A lowercase hexadecimal SHA-256 digest.

  Note:
    The ``default=str`` fallback allows otherwise unsupported values to be serialized, but their string representations must themselves be stable if reproducible hashes are required.
  """

  payload = json.dumps(
    source,
    sort_keys=True,
    separators=(",", ":"),
    default=str,
  )

  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_source(
  path: Path, 
  source: Any
) -> tuple[str, str]:
  """Resolve a configuration path and compute its content digest.

  Args:
    path: Configuration file path.
    source: Parsed configuration content.

  Returns:
    A ``(resolved_path, digest)`` tuple.
  """

  resolved_path = str(path.resolve())
  digest = config_hash(source)

  return resolved_path, digest


def create_config_identity(
  *,
  ontology_path: Path, 
  sampling_path: Path,
  ontology_config: Any,
  sampling_config: Any, 
  schema_version: Any
) -> ConfigIdentity:
  """Create provenance identity for ontology and sampling configurations.

  The resulting identity records the resolved path and content digest of both configuration files, together with the schema version used to interpret them.

  Args:
    ontology_path: Path to the ontology configuration.
    sampling_path: Path to the symbol-sampling configuration.
    ontology_config: Parsed and validated ontology content.
    sampling_config: Parsed and validated sampling content.
    schema_version: Supported schema version for the configuration format.
  
  Returns:
    A configuration identity containing both source paths, both content digests, and the schema version.
  """

  ontology_resolved, ontology_digest = config_source(
    path = ontology_path, 
    source = ontology_config
  )

  sampling_resolved, sampling_digest = config_source(
    path = sampling_path, 
    source = sampling_config
  )

  return ConfigIdentity(
    ontology_path = ontology_resolved, 
    ontology_digest = ontology_digest, 
    sampling_path = sampling_resolved, 
    sampling_digest = sampling_digest, 
    schema_version = schema_version
  )


def create_sampling_provenance(
  *,
  config_identity: ConfigIdentity, 
  class_group: str, 
  class_name: str, 
  seed: int | None, 
  decisions: Mapping[str, Any], 
  parameters: Mapping[str, Any], 
  derived: Mapping[str, Any]
) -> SamplingProvenance:
  """Create provenance for one sampled class instance.

  Args:
    config_identity: Identity of the ontology and sampling configurations used for sampling.
    class_group: Configuration group containing the sampled class.
    class_name: Name of the sampled class.
    seed: Optional seed associated with the sample.
    decisions: Discrete sampling decisions made for the instance.
    parameters: Continuous and discrete sampled parameter values.
    derived: Values computed from the sampled parameters.

  Returns:
    A sampling-provenance record for the generated instance.
  """

  return SamplingProvenance(
    config_identity = config_identity,
    class_group = class_group,
    class_name = class_name,
    seed = seed,
    decisions = dict(decisions),
    parameters = dict(parameters),
    derived = dict(derived),
  )
