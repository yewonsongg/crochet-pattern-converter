"""Regression checks for reusable rendered-pixel chain geometry."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import Element, tostring

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GenerationConfig
from generation.core.rendering import render_png
from generation.core.svg import ChainPlacement, append_chain_geometry


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_rectangular_chain_geometry() -> None:
  config = GenerationConfig(canvas_width_px=120, canvas_height_px=80)
  parent = Element("g")
  geometry = append_chain_geometry(
    parent,
    sampled_values={"shape": "oval", "aspect_ratio": 1.5},
    config=config,
    placement=ChainPlacement(
      center_px=(60.0, 40.0),
      visible_span_px=30.0,
      rotation_deg=0.0,
    ),
    stroke_width=2.0,
  )
  assert geometry.width_px == 30.0
  assert geometry.height_px == 20.0
  assert geometry.centerline_bounds_px == (45.0, 30.0, 75.0, 50.0)
  assert geometry.rendered_bounds_px == (44.2, 29.2, 75.8, 50.8)
  ellipse = parent[0]
  scale = 0.8
  assert math.isclose(float(ellipse.attrib["rx"]) * 2.0 * scale, 30.0)
  assert math.isclose(float(ellipse.attrib["ry"]) * 2.0 * scale, 20.0)

  rotated = append_chain_geometry(
    Element("g"),
    sampled_values={"shape": "oval", "aspect_ratio": 1.5},
    config=config,
    placement=ChainPlacement((60.0, 40.0), 30.0, 90.0),
    stroke_width=2.0,
  )
  assert math.isclose(rotated.centerline_bounds_px[0], 50.0, abs_tol=1e-10)
  assert math.isclose(rotated.centerline_bounds_px[1], 25.0, abs_tol=1e-10)
  assert math.isclose(rotated.centerline_bounds_px[2], 70.0, abs_tol=1e-10)
  assert math.isclose(rotated.centerline_bounds_px[3], 55.0, abs_tol=1e-10)


def test_chain_rasterization_and_validation() -> None:
  config = GenerationConfig(canvas_width_px=100, canvas_height_px=100)
  svg = Element("svg", {
    "xmlns": "http://www.w3.org/2000/svg",
    "width": "100px",
    "height": "100px",
    "viewBox": "0 0 100 100",
  })
  group = Element("g", {"fill": "none", "stroke": "black", "stroke-width": "2"})
  svg.append(group)
  append_chain_geometry(
    group,
    sampled_values={"shape": "oval", "aspect_ratio": 1.6},
    config=config,
    placement=ChainPlacement((50.0, 50.0), 30.0, 25.0),
    stroke_width=2.0,
  )
  with TemporaryDirectory() as directory:
    output = Path(directory) / "chain.png"
    render_png(tostring(svg, encoding="unicode"), output)
    with Image.open(output) as image:
      assert image.getbbox() is not None

  base = dict(
    parent=Element("g"),
    sampled_values={"shape": "oval", "aspect_ratio": 1.5},
    config=config,
    placement=ChainPlacement((50.0, 50.0), 20.0),
    stroke_width=2.0,
  )
  _expect_error(
    ValueError,
    "Unsupported chain shape",
    lambda: append_chain_geometry(**{**base, "sampled_values": {"shape": "x"}}),
  )
  _expect_error(
    ValueError,
    "visible_span_px must be positive",
    lambda: append_chain_geometry(
      **{**base, "placement": ChainPlacement((50.0, 50.0), 0.0)}
    ),
  )
  _expect_error(
    ValueError,
    "center_px[0] must be a finite number",
    lambda: append_chain_geometry(
      **{**base, "placement": ChainPlacement((math.inf, 50.0), 20.0)}
    ),
  )
  _expect_error(
    ValueError,
    "stroke_width must be positive",
    lambda: append_chain_geometry(**{**base, "stroke_width": 0.0}),
  )


def main() -> None:
  test_rectangular_chain_geometry()
  test_chain_rasterization_and_validation()
  print("Reusable chain-geometry checks passed.")


if __name__ == "__main__":
  main()
