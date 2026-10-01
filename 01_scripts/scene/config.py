"""Load and validate shared scene patterns, sampling profiles, and audit cases."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from generation.core.metrics import SymbolMetrics, load_symbol_metrics
from generation.core.sampling import SamplingConfig

from .models import ClassKey, SceneConfigIdentity


@dataclass(frozen=True)
class SceneConfiguration:
  patterns: Mapping[str, Any]
  profile: Mapping[str, Any]
  rasterization: Mapping[str, Any]
  symbol_metrics: SymbolMetrics
  identity: SceneConfigIdentity


@dataclass(frozen=True)
class AuditCase:
  case_id: str
  profile: str
  seed: int
  pattern: str
  pattern_overrides: Mapping[str, Any]
  forced_role_classes: Mapping[str, tuple[ClassKey, ...]]
  chart_scale: float | None = None


def _load_yaml(path: Path) -> Mapping[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
  if not isinstance(value, Mapping):
    raise ValueError(f"Expected a YAML mapping: {path}")
  return value


def _digest(value: Any) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_class_key(value: str) -> ClassKey:
  if not isinstance(value, str) or value.count(".") != 1:
    raise ValueError(f"Class reference must be 'family.name': {value!r}.")
  family, name = value.split(".", 1)
  if not family or not name:
    raise ValueError(f"Invalid class reference: {value!r}.")
  return family, name


def load_scene_configuration(
  *,
  patterns_path: str | Path,
  profile_path: str | Path,
  metrics_path: str | Path,
  rasterization_path: str | Path,
  sampling_config: SamplingConfig,
) -> SceneConfiguration:
  patterns_source = Path(patterns_path)
  profile_source = Path(profile_path)
  metrics_source = Path(metrics_path)
  rasterization_source = Path(rasterization_path)
  patterns = _load_yaml(patterns_source)
  profile = _load_yaml(profile_source)
  rasterization = _load_yaml(rasterization_source)
  _validate_patterns(patterns, set(sampling_config.class_keys))
  _validate_profile(profile, patterns)
  _validate_rasterization(rasterization)
  metrics = load_symbol_metrics(metrics_source, sampling_config.class_keys)
  identity = SceneConfigIdentity(
    patterns_path=str(patterns_source.resolve()),
    patterns_digest=_digest(patterns),
    profile_path=str(profile_source.resolve()),
    profile_digest=_digest(profile),
    metrics_path=str(metrics_source.resolve()),
    metrics_digest=_digest(_load_yaml(metrics_source)),
    rasterization_path=str(rasterization_source.resolve()),
    rasterization_digest=_digest(rasterization),
  )
  return SceneConfiguration(patterns, profile, rasterization, metrics, identity)


def _validate_rasterization(document: Mapping[str, Any]) -> None:
  expected = {
    "schema_version",
    "scene_supersample_factor",
    "local_label_supersample_factor",
  }
  if set(document) != expected or document.get("schema_version") != 1:
    raise ValueError(
      "Rasterization config requires schema_version: 1 and exactly the two "
      "supersample factors."
    )
  for name in ("scene_supersample_factor", "local_label_supersample_factor"):
    value = document[name]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
      raise ValueError(f"{name} must be an integer from 1 through 4.")


def _validate_patterns(document: Mapping[str, Any], ontology: set[ClassKey]) -> None:
  if document.get("schema_version") != 1:
    raise ValueError("Scene patterns require schema_version: 1.")
  canvas = document.get("canvas")
  if canvas != {"width_px": 1280, "height_px": 1280}:
    raise ValueError("Scene canvas must be exactly 1280x1280.")
  generation = document.get("symbol_generation")
  if not isinstance(generation, Mapping):
    raise ValueError("symbol_generation must be a mapping.")
  for key in ("canvas_width_px", "canvas_height_px", "reference_visible_px", "chart_scale"):
    if key not in generation:
      raise ValueError(f"symbol_generation.{key} is required.")
  pools = document.get("class_pools")
  if not isinstance(pools, Mapping) or not pools:
    raise ValueError("class_pools must be a non-empty mapping.")
  parsed_pools: dict[str, tuple[ClassKey, ...]] = {}
  for pool_name, entries in pools.items():
    if not isinstance(entries, list) or not entries:
      raise ValueError(f"class_pools.{pool_name} must be a non-empty list.")
    parsed = tuple(parse_class_key(entry) for entry in entries)
    unknown = set(parsed) - ontology
    if unknown:
      raise ValueError(f"class_pools.{pool_name} contains unknown classes: {sorted(unknown)}.")
    parsed_pools[str(pool_name)] = parsed
  patterns = document.get("patterns")
  if not isinstance(patterns, Mapping) or set(patterns) != {"grid", "radial"}:
    raise ValueError("v1 scene patterns must contain exactly grid and radial.")
  for pattern_name, pattern in patterns.items():
    if not isinstance(pattern, Mapping):
      raise ValueError(f"patterns.{pattern_name} must be a mapping.")
    roles = pattern.get("roles")
    if not isinstance(roles, Mapping) or not roles:
      raise ValueError(f"patterns.{pattern_name}.roles must be a non-empty mapping.")
    for role_name, role in roles.items():
      pool = role.get("class_pool") if isinstance(role, Mapping) else None
      if pool not in parsed_pools:
        raise ValueError(f"patterns.{pattern_name}.roles.{role_name} has unknown class_pool.")


def _positive_weights(value: Any, path: str) -> Mapping[str, float]:
  if not isinstance(value, Mapping) or not value:
    raise ValueError(f"{path} must be a non-empty mapping.")
  result: dict[str, float] = {}
  for key, weight in value.items():
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
      raise ValueError(f"{path}.{key} must be finite and nonnegative.")
    result[str(key)] = float(weight)
  if sum(result.values()) <= 0:
    raise ValueError(f"{path} must contain a positive total weight.")
  return result


def _validate_profile(profile: Mapping[str, Any], patterns: Mapping[str, Any]) -> None:
  if profile.get("schema_version") != 1 or not isinstance(profile.get("name"), str):
    raise ValueError("Scene profile requires schema_version: 1 and a name.")
  pattern_weights = _positive_weights(profile.get("pattern_weights"), "pattern_weights")
  if set(pattern_weights) != set(patterns["patterns"]):
    raise ValueError("Profile pattern weights must cover every shared pattern.")
  class_weights = profile.get("class_weights")
  if not isinstance(class_weights, Mapping):
    raise ValueError("Profile class_weights must be a mapping.")
  pools = patterns["class_pools"]
  if set(class_weights) != set(pools):
    raise ValueError("Profile class_weights must cover every class pool.")
  for pool_name, pool_entries in pools.items():
    weights = _positive_weights(class_weights[pool_name], f"class_weights.{pool_name}")
    if set(weights) != set(pool_entries):
      raise ValueError(f"class_weights.{pool_name} must cover its shared class pool exactly.")


def load_audit_cases(path: str | Path) -> dict[str, AuditCase]:
  document = _load_yaml(Path(path))
  if document.get("schema_version") != 1 or not isinstance(document.get("cases"), list):
    raise ValueError("Scene audit cases require schema_version: 1 and a cases list.")
  result: dict[str, AuditCase] = {}
  for raw in document["cases"]:
    case_id = raw.get("id") if isinstance(raw, Mapping) else None
    if not isinstance(case_id, str) or not case_id or case_id in result:
      raise ValueError(f"Invalid or duplicate audit case ID: {case_id!r}.")
    forced: dict[str, tuple[ClassKey, ...]] = {}
    for role, values in (raw.get("forced_role_classes") or {}).items():
      if not isinstance(values, list) or not values:
        raise ValueError(f"Audit case {case_id} forced role {role} must be a non-empty list.")
      forced[str(role)] = tuple(parse_class_key(item) for item in values)
    result[case_id] = AuditCase(
      case_id=case_id,
      profile=str(raw.get("profile")),
      seed=int(raw.get("seed")),
      pattern=str(raw.get("pattern")),
      pattern_overrides=dict(raw.get("pattern_overrides") or {}),
      forced_role_classes=forced,
      chart_scale=None if raw.get("chart_scale") is None else float(raw["chart_scale"]),
    )
  return result
