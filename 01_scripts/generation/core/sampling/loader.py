from pathlib import Path

from .config import SamplingConfig
from .provenance import create_config_identity
from .validator import validate_sampling_config


def load_sampling_config(path: str | Path) -> SamplingConfig:
  """Point-of-entry: load, validate, identify config, and then pass to `SamplingConfig` constructor in `config.py`.

  Args:
    path: config YAML path.

  Returns:
    :class:`SamplingConfig` with validation-passed resolved config map and provenance-signed config identity.

  Raises:
    RuntimeError: If PyYAML dependency is not found.
  """

  source = Path(path)
  try:
    import yaml
  except ImportError as exc:
    raise RuntimeError("Loading sampling YAML requires PyYAML.") from exc

  # load the YAML
  with source.open("r", encoding="utf-8") as handle:
    raw = yaml.safe_load(handle)

  # validation - from `validator.py`
  validate_sampling_config(raw = raw)

  # get path, hash, and schema version - from `provenance.py`
  identity = create_config_identity(
    path = source, 
    raw = raw, 
    schema_version = raw["schema"].get("version")
  )

  # `SamplingConfig` construction handover to `config.py`
  return SamplingConfig(
    raw = raw, 
    identity = identity
  )