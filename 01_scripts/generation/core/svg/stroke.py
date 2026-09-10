"""Shared stroke-width resolution for SVG generators."""

from __future__ import annotations

import math
from typing import Any

from ..models import GenerationConfig, SampledParameters


def resolve_stroke_width(
  sample: SampledParameters,
  config: GenerationConfig,
  *,
  class_name: str,
) -> float:
  """Return the sampled stroke width, with a legacy config fallback."""
  value: Any = sample.as_dict().get("stroke_width", config.stroke_width_normalized)
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{class_name} stroke_width must be a finite number.")
  result = float(value)
  if not math.isfinite(result) or result <= 0.0:
    raise ValueError(f"{class_name} stroke_width must be positive and finite.")
  return result
