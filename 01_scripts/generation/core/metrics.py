"""Class size ratios used when configuring upright symbol generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import GenerationConfig


ClassKey = tuple[str, str]


@dataclass(frozen=True)
class SymbolMetrics:
  relative_sizes: Mapping[ClassKey, float]

  def config_for(self, class_key: ClassKey, base: GenerationConfig) -> GenerationConfig:
    """Apply a class ratio to the base visible size before SVG generation."""
    try:
      ratio = self.relative_sizes[class_key]
    except KeyError as exc:
      raise KeyError(f"No relative size configured for {class_key!r}.") from exc
    return base.with_overrides({
      "target_visible_px": base.target_visible_px * ratio,
    })


def load_symbol_metrics(
  path: str | Path,
  class_keys: tuple[ClassKey, ...],
) -> SymbolMetrics:
  """Load positive ratios and require one entry per ontology class."""
  try:
    import yaml
  except ImportError as exc:
    raise RuntimeError("Loading symbol metrics requires PyYAML.") from exc

  with Path(path).open("r", encoding="utf-8") as handle:
    document = yaml.safe_load(handle)
  if not isinstance(document, Mapping) or document.get("schema_version") != 1:
    raise ValueError("Symbol metrics require schema_version: 1.")
  classes = document.get("classes")
  if not isinstance(classes, Mapping):
    raise ValueError("Symbol metrics require a classes mapping.")

  ratios: dict[ClassKey, float] = {}
  for family, members in classes.items():
    if not isinstance(family, str) or not isinstance(members, Mapping):
      raise ValueError("Each symbol-metrics family must map class names to metrics.")
    for name, metrics in members.items():
      if not isinstance(name, str) or not isinstance(metrics, Mapping) or set(metrics) != {"relative_size"}:
        raise ValueError(f"{family}.{name} must contain only relative_size.")
      ratio = metrics["relative_size"]
      if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"{family}.{name}.relative_size must be positive and finite.")
      ratios[(family, name)] = float(ratio)

  expected = set(class_keys)
  missing = expected - ratios.keys()
  extra = ratios.keys() - expected
  if missing or extra:
    raise ValueError(f"Symbol metrics class mismatch: missing={sorted(missing)}, extra={sorted(extra)}.")
  return SymbolMetrics(ratios)
