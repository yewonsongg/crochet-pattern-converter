"""Persistence helpers for generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .models import GeneratedObject
from .rendering import render_png


def write_generated_artifacts(generated: GeneratedObject, stem: Path) -> None:
  """Write SVG, PNG, metadata, and YOLO label for one generated object."""
  stem.parent.mkdir(parents=True, exist_ok=True)
  svg_path = stem.with_suffix(".svg")
  png_path = stem.with_suffix(".png")
  metadata_path = stem.with_suffix(".json")
  label_path = stem.with_suffix(".txt")
  svg_path.write_text(generated.svg, encoding="utf-8")
  render_png(generated.svg, png_path)
  metadata_path.write_text(json.dumps(generated.metadata, indent=2, default=str), encoding="utf-8")
  label_path.write_text(generated.yolo_label + "\n", encoding="utf-8")
  generated.svg_path = svg_path
  generated.png_path = png_path
  generated.metadata_path = metadata_path
  generated.label_path = label_path
