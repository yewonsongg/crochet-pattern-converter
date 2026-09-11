"""Sampling-to-SVG regression checks for instructive rings."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import fromstring

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GenerationConfig
from generation.core.rendering import render_png
from generation.core.sampling import load_sampling_config
from generation.instructive.ring import generate_ring


def _config():
  return load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )


def _generate(overrides: dict, *, seed: int, generation: GenerationConfig):
  config = _config()
  rng = np.random.default_rng(seed)
  parent = config.sample(
    "instructive", "ring", rng, seed=seed, overrides=overrides
  )
  composite = config.realize_components(
    "instructive", "ring", parent, rng
  )
  return generate_ring(config.resolve("instructive", "ring"), composite, generation)


def test_magic_ring() -> None:
  generated = _generate(
    {"variant": "magic", "stroke_width": 1.8},
    seed=100,
    generation=GenerationConfig(
      canvas_width_px=40,
      canvas_height_px=30,
      target_visible_px=20.0,
      rotation_deg=12.0,
    ),
  )
  root = fromstring(generated.svg)
  assert root.tag.endswith("svg")
  assert generated.svg.count("<path") == 1
  assert generated.svg.count("<ellipse") == 0
  path = next(element for element in root.iter() if element.tag.endswith("path"))
  assert path.attrib["d"].count("C") == 38
  assert "Z" not in path.attrib["d"].upper()
  assert path.attrib["stroke-linecap"] == "round"
  assert "rotate(12.0 50 50)" in generated.svg
  assert generated.variant_id == "magic"
  assert generated.metadata["phenotype"] == "canonical_spiral"
  assert generated.metadata["turns"] == 1.9
  assert generated.metadata["winding"] == "clockwise_outward"
  assert np.allclose(generated.metadata["endpoint_angles_deg"], [-72.0, -108.0])
  assert np.isclose(generated.metadata["rendered_width_px"], 20.0)
  assert generated.metadata["rendered_height_px"] <= 20.0
  assert "diameter" not in generated.sampled_parameters
  assert generated.obb_pixels is None
  assert generated.yolo_label is None

  try:
    _generate(
      {"variant": "magic", "stroke_width": 1.8, "diameter": 0.25},
      seed=101,
      generation=GenerationConfig(),
    )
  except ValueError as exc:
    assert "diameter" in str(exc)
  else:
    raise AssertionError("Removed magic diameter override was accepted.")


def test_chain_ring_reuses_one_prototype() -> None:
  generated = _generate(
    {
      "variant": "chain",
      "stroke_width": 2.0,
      "count": 6,
      "chain_pitch": 3.2,
    },
    seed=200,
    generation=GenerationConfig(
      canvas_width_px=30,
      canvas_height_px=30,
      target_visible_px=15.0,
      rotation_deg=-18.0,
    ),
  )
  fromstring(generated.svg)
  assert generated.svg.count("<ellipse") == 6
  assert "rotate(-18.0 50 50)" in generated.svg
  assert generated.variant_id == "chain"
  assert generated.metadata["count"] == 6
  assert generated.metadata["chain_pitch"] == 3.2
  assert np.isclose(generated.metadata["chain_pitch_px"], 4.8)
  assert np.isclose(generated.metadata["construction_scale_px_per_unit"], 1.5)

  prototype = generated.metadata["chain_prototype"]
  assert prototype["class_group"] == "primitive"
  assert prototype["class_name"] == "ch"
  assert prototype["occurrence_count"] == 6
  assert prototype["arrangement"] == "ring"
  assert prototype["inherited_parameters"] == {"stroke_width": 2.0}
  assert prototype["sampling_provenance"]["overrides"] == {"stroke_width": 2.0}

  root = fromstring(generated.svg)
  ellipses = [element for element in root.iter() if element.tag.endswith("ellipse")]
  assert len({element.attrib["rx"] for element in ellipses}) == 1
  assert len({element.attrib["ry"] for element in ellipses}) == 1


def test_ring_rasterization() -> None:
  for index, overrides in enumerate((
    {"variant": "magic", "stroke_width": 1.8},
    {"variant": "chain", "stroke_width": 1.8, "count": 8, "chain_pitch": 3.0},
  )):
    generated = _generate(
      overrides,
      seed=300 + index,
      generation=GenerationConfig(canvas_width_px=64, canvas_height_px=64),
    )
    with TemporaryDirectory() as directory:
      path = Path(directory) / f"ring-{index}.png"
      render_png(generated.svg, path)
      assert path.stat().st_size > 0


def main() -> None:
  test_magic_ring()
  test_chain_ring_reuses_one_prototype()
  test_ring_rasterization()
  print("Ring generation checks passed.")


if __name__ == "__main__":
  main()
