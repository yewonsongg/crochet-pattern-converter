"""Regression checks for reusable rendered-pixel slip-stitch geometry."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from xml.etree.ElementTree import Element


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GenerationConfig
from generation.core.svg import SlipStitchPlacement, append_slip_stitch_geometry


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def main() -> None:
  config = GenerationConfig(canvas_width_px=120, canvas_height_px=80)
  parent = Element("g")
  geometry = append_slip_stitch_geometry(
    parent,
    sampled_values={"shape": "circle", "aspect_ratio": 1.0},
    config=config,
    placement=SlipStitchPlacement((60.0, 40.0), 12.0),
    stroke_width=2.0,
  )
  assert geometry.class_name == "slst"
  assert geometry.width_px == geometry.height_px == 12.0
  assert geometry.centerline_bounds_px == (54.0, 34.0, 66.0, 46.0)
  assert geometry.rendered_bounds_px == (53.2, 33.2, 66.8, 46.8)
  assert parent[0].attrib["fill"] == "black"
  assert math.isclose(float(parent[0].attrib["rx"]) * 1.6, 12.0)

  base = dict(
    parent=Element("g"),
    sampled_values={"shape": "circle", "aspect_ratio": 1.0},
    config=config,
    placement=SlipStitchPlacement((60.0, 40.0), 12.0),
    stroke_width=2.0,
  )
  _expect_error(
    ValueError,
    "Unsupported slip-stitch shape",
    lambda: append_slip_stitch_geometry(
      **{**base, "sampled_values": {"shape": "invalid", "aspect_ratio": 1.0}}
    ),
  )
  _expect_error(
    ValueError,
    "visible_span_px must be positive",
    lambda: append_slip_stitch_geometry(
      **{**base, "placement": SlipStitchPlacement((60.0, 40.0), 0.0)}
    ),
  )
  _expect_error(
    ValueError,
    "center_px[0] must be a finite number",
    lambda: append_slip_stitch_geometry(
      **{**base, "placement": SlipStitchPlacement((math.inf, 40.0), 12.0)}
    ),
  )
  print("Reusable slip-stitch geometry checks passed.")


if __name__ == "__main__":
  main()
