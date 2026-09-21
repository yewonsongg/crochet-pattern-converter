"""Declarative rendering cases and the sampler-to-artifact entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .models import GeneratedObject, GenerationConfig
from .metrics import SymbolMetrics, load_symbol_metrics
from .sampling import SamplingConfig


@dataclass(frozen=True)
class RenderingCase:
  case_id: str
  class_group: str
  class_name: str
  seed: int | None = None
  sampling: dict[str, Any] | None = None
  generation: dict[str, Any] | None = None


def load_rendering_cases(path: str | Path, sampling_config: SamplingConfig) -> dict[str, RenderingCase]:
  """Load and validate named rendering cases against a sampling config."""
  try:
    import yaml
  except ImportError as exc:
    raise RuntimeError("Loading rendering cases requires PyYAML.") from exc

  source = Path(path)
  with source.open("r", encoding="utf-8") as handle:
    document = yaml.safe_load(handle) or {}
  raw_cases = document.get("cases") if isinstance(document, Mapping) else None
  if not isinstance(raw_cases, list):
    raise ValueError("Rendering-case manifest requires a 'cases' list.")

  cases: dict[str, RenderingCase] = {}
  valid_generation = {"canvas_width_px", "canvas_height_px", "target_visible_px", "stroke_width_normalized"}
  for index, raw in enumerate(raw_cases):
    prefix = f"cases[{index}]"
    if not isinstance(raw, Mapping):
      raise ValueError(f"{prefix} must be a mapping.")
    case_id = raw.get("id")
    class_ref = raw.get("class")
    if not isinstance(case_id, str) or not case_id:
      raise ValueError(f"{prefix}.id must be a non-empty string.")
    if case_id in cases:
      raise ValueError(f"Duplicate rendering case ID: {case_id!r}.")
    if not isinstance(class_ref, Mapping) or not isinstance(class_ref.get("family"), str) or not isinstance(class_ref.get("name"), str):
      raise ValueError(f"{prefix}.class requires string family and name fields.")
    group, name = class_ref["family"], class_ref["name"]
    if (group, name) not in sampling_config.class_keys:
      raise ValueError(f"{prefix}.class is unknown: {group}.{name}.")
    sampling = raw.get("sampling", {})
    generation = raw.get("generation", {})
    if not isinstance(sampling, Mapping) or not isinstance(generation, Mapping):
      raise ValueError(f"{prefix}.sampling and {prefix}.generation must be mappings.")
    spec = sampling_config.resolve(group, name)
    valid_sampling = set(spec.parameters)
    for variant in spec.variants.values():
      valid_sampling.update(variant.get("parameters", {}))
    unknown = set(sampling) - valid_sampling
    if unknown:
      raise ValueError(f"{prefix}.sampling contains unknown parameter(s): {sorted(unknown)!r}.")
    derived = {
      key for key, parameter in spec.parameters.items()
      if parameter.kind == "derived"
    }
    for variant in spec.variants.values():
      derived.update(key for key, parameter in variant.get("parameters", {}).items() if parameter.get("kind") == "derived")
    invalid_derived = set(sampling) & derived
    if invalid_derived:
      raise ValueError(f"{prefix}.sampling cannot override derived parameter(s): {sorted(invalid_derived)!r}.")
    invalid_generation = set(generation) - valid_generation
    if invalid_generation:
      raise ValueError(f"{prefix}.generation contains unknown field(s): {sorted(invalid_generation)!r}.")
    seed = raw.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
      raise ValueError(f"{prefix}.seed must be an integer or null.")
    # Reuse the sampler's support/dependency validation so the manifest and
    # direct API cannot acquire different override semantics.
    import numpy as np
    sampling_config.sample(
      group,
      name,
      np.random.default_rng(seed),
      seed=seed,
      overrides=sampling,
      case_id=case_id,
    )
    cases[case_id] = RenderingCase(case_id, group, name, seed, dict(sampling), dict(generation))
  return cases


def generate_rendering_case(
  case: RenderingCase,
  sampling_config: SamplingConfig,
  generator_registry: Mapping[tuple[str, str], Any],
  *,
  symbol_metrics: SymbolMetrics | None = None,
  output_dir: str | Path | None = None,
) -> GeneratedObject:
  """Sample, size, generate, and optionally persist one rendering case."""
  import numpy as np
  from generation.registry import COMPOSITE_GENERATORS

  class_key = (case.class_group, case.class_name)
  rng = np.random.default_rng(case.seed)
  sample = sampling_config.sample(
    case.class_group,
    case.class_name,
    rng,
    seed=case.seed,
    overrides=case.sampling,
    case_id=case.case_id,
  )
  generator = generator_registry.get(class_key)
  if generator is None:
    raise KeyError(f"No generator registered for {case.class_group}.{case.class_name}.")
  if symbol_metrics is None:
    metrics_path = Path(sampling_config.identity.ontology_path).with_name("symbol_metrics.yaml")
    symbol_metrics = load_symbol_metrics(metrics_path, sampling_config.class_keys)
  base_config = GenerationConfig().with_overrides(case.generation)
  config = symbol_metrics.config_for(class_key, base_config)
  generator_input = (
    sampling_config.realize_components(*class_key, sample, rng)
    if class_key in COMPOSITE_GENERATORS else sample
  )
  generated = generator(sampling_config.resolve(*class_key), generator_input, config)
  generated.sampling_provenance = sample.provenance
  generated.metadata["rendering_case"] = {
    "id": case.case_id,
    "seed": case.seed,
    "sampling_overrides": dict(case.sampling or {}),
    "generation_overrides": dict(case.generation or {}),
    "relative_size": symbol_metrics.relative_sizes[class_key],
    "effective_target_visible_px": config.target_visible_px,
  }
  if output_dir is not None:
    from .artifacts import write_generated_artifacts
    write_generated_artifacts(generated, Path(output_dir) / case.case_id)
  return generated
