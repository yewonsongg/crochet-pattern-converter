"""Regression checks for reusable anchor-based stitch geometry."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import Element, fromstring

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.models import GenerationConfig
from generation.core.rendering import render_png
from generation.core.svg.bar_stem import build_bar_stem_svg
from generation.core.svg.stitch import (
  STITCH_GEOMETRY_REGISTRY,
  StitchPlacement,
  append_stitch_geometry,
  build_centered_stitch_svg,
)


def _line_pixels(element: Element, config: GenerationConfig):
  scale = min(config.canvas_width_px, config.canvas_height_px) / 100.0
  offset_x = (config.canvas_width_px - 100.0 * scale) / 2.0
  offset_y = (config.canvas_height_px - 100.0 * scale) / 2.0

  def point(x_name: str, y_name: str):
    return (
      float(element.attrib[x_name]) * scale + offset_x,
      float(element.attrib[y_name]) * scale + offset_y,
    )

  return point("x1", "y1"), point("x2", "y2")


def _lines(svg: str) -> list[Element]:
  root = fromstring(svg)
  return [element for element in root.iter() if element.tag.endswith("line")]


def _assert_point(actual, expected) -> None:
  assert math.isclose(actual[0], expected[0], abs_tol=1e-7), (actual, expected)
  assert math.isclose(actual[1], expected[1], abs_tol=1e-7), (actual, expected)


def _expect_error(error_type, message: str, function) -> None:
  try:
    function()
  except error_type as exc:
    assert message in str(exc), str(exc)
  else:
    raise AssertionError(f"Expected {error_type.__name__} containing {message!r}.")


def test_registry_and_centered_geometry() -> None:
  assert set(STITCH_GEOMETRY_REGISTRY) == {"sc", "hdc", "dc", "tr", "dtr"}

  def mutate_registry() -> None:
    STITCH_GEOMETRY_REGISTRY["missing"] = object()  # type: ignore[index]

  _expect_error(
    TypeError,
    "does not support item assignment",
    mutate_registry,
  )

  config = GenerationConfig(
    canvas_width_px=100,
    canvas_height_px=100,
    target_visible_px=60.0,
  )
  sc_svg, sc = build_centered_stitch_svg(
    class_name="sc",
    sampled_values={"shape": "asymmetric", "asymmetry": 0.2, "cross_bar_ratio": 0.5},
    config=config,
    stroke_width=2.0,
  )
  sc_lines = _lines(sc_svg)
  standalone_svgs = [("sc", sc_svg)]
  assert len(sc_lines) == 2
  _assert_point(_line_pixels(sc_lines[0], config)[0], (35.0, 44.0))
  _assert_point(_line_pixels(sc_lines[0], config)[1], (65.0, 44.0))
  _assert_point(_line_pixels(sc_lines[1], config)[0], (50.0, 20.0))
  _assert_point(_line_pixels(sc_lines[1], config)[1], (50.0, 80.0))
  assert sc.cross_bar_positions == (0.6,)

  expected_lines = {"hdc": 2, "dc": 3, "tr": 4, "dtr": 5}
  for class_name, count in expected_lines.items():
    values = {"bar_stem_ratio": 0.5}
    if class_name != "hdc":
      values.update({
        "cross_bar_ratio": 0.3,
        "cross_bar_y": 0.5,
        "cross_bar_angle_deg": 0.0,
      })
    svg, geometry = build_centered_stitch_svg(
      class_name=class_name,
      sampled_values=values,
      config=config,
      stroke_width=2.0,
    )
    assert len(_lines(svg)) == count
    standalone_svgs.append((class_name, svg))
    assert geometry.stem_length_px == 60.0
    assert geometry.top_bar_length_px == 30.0
    assert geometry.cross_bar_length_px == (0.0 if class_name == "hdc" else 18.0)

  with TemporaryDirectory() as directory:
    for class_name, svg in standalone_svgs:
      output = Path(directory) / f"{class_name}.png"
      render_png(svg, output)
      alpha_bounds = Image.open(output).convert("RGBA").getchannel("A").getbbox()
      assert alpha_bounds is not None


def test_rectangular_and_arbitrary_anchor_geometry() -> None:
  config = GenerationConfig(canvas_width_px=200, canvas_height_px=100)
  parent = Element("g")
  placement = StitchPlacement(base_px=(100.0, 80.0), top_px=(130.0, 40.0))
  geometry = append_stitch_geometry(
    parent,
    class_name="dc",
    sampled_values={
      "bar_stem_ratio": 0.4,
      "cross_bar_ratio": 0.3,
      "cross_bar_y": 0.5,
      "cross_bar_angle_deg": 20.0,
    },
    config=config,
    placement=placement,
    stroke_width=2.0,
  )
  assert math.isclose(geometry.stem_length_px, 50.0)
  lines = list(parent)
  stem_start, stem_end = _line_pixels(lines[0], config)
  _assert_point(stem_start, placement.top_px)
  _assert_point(stem_end, placement.base_px)

  top_start, top_end = _line_pixels(lines[1], config)
  _assert_point(top_start, (122.0, 34.0))
  _assert_point(top_end, (138.0, 46.0))
  cross_start, cross_end = _line_pixels(lines[2], config)
  _assert_point(
    ((cross_start[0] + cross_end[0]) / 2.0, (cross_start[1] + cross_end[1]) / 2.0),
    (115.0, 60.0),
  )

  axis = (placement.top_px[0] - placement.base_px[0], placement.top_px[1] - placement.base_px[1])
  cross = (cross_end[0] - cross_start[0], cross_end[1] - cross_start[1])
  axis_unit = (axis[0] / 50.0, axis[1] / 50.0)
  perpendicular = (-axis_unit[1], axis_unit[0])
  cross_unit = (cross[0] / 15.0, cross[1] / 15.0)
  assert math.isclose(
    cross_unit[0] * perpendicular[0] + cross_unit[1] * perpendicular[1],
    math.cos(math.radians(20.0)),
    abs_tol=1e-7,
  )
  assert math.isclose(
    cross_unit[0] * axis_unit[0] + cross_unit[1] * axis_unit[1],
    math.sin(math.radians(20.0)),
    abs_tol=1e-7,
  )
  for line in lines:
    for point in _line_pixels(line, config):
      left, top, right, bottom = geometry.centerline_bounds_px
      assert left - 1e-7 <= point[0] <= right + 1e-7
      assert top - 1e-7 <= point[1] <= bottom + 1e-7

  for anchors in (
    StitchPlacement((20.0, 50.0), (80.0, 50.0)),
    StitchPlacement((80.0, 50.0), (20.0, 50.0)),
    StitchPlacement((20.0, 80.0), (80.0, 20.0)),
  ):
    result = append_stitch_geometry(
      Element("g"),
      class_name="hdc",
      sampled_values={"bar_stem_ratio": 0.5},
      config=config,
      placement=anchors,
      stroke_width=2.0,
    )
    assert math.isclose(
      result.stem_length_px,
      math.dist(anchors.base_px, anchors.top_px),
      abs_tol=1e-9,
    )


def test_top_bar_suppression_and_bounds() -> None:
  config = GenerationConfig(canvas_width_px=100, canvas_height_px=100)
  placement = StitchPlacement((50.0, 80.0), (50.0, 20.0))
  parent = Element("g")
  hdc = append_stitch_geometry(
    parent,
    class_name="hdc",
    sampled_values={"bar_stem_ratio": 0.5},
    config=config,
    placement=placement,
    stroke_width=2.0,
    include_top_bar=False,
  )
  assert len(parent) == 1
  assert hdc.top_bar_length_px == 0.0
  assert hdc.centerline_bounds_px == (50.0, 20.0, 50.0, 80.0)
  assert hdc.rendered_bounds_px == (49.0, 19.0, 51.0, 81.0)

  sc_parent = Element("g")
  sc = append_stitch_geometry(
    sc_parent,
    class_name="sc",
    sampled_values={"shape": "symmetric", "asymmetry": 0.0, "cross_bar_ratio": 0.5},
    config=config,
    placement=placement,
    stroke_width=2.0,
    include_top_bar=False,
  )
  assert len(sc_parent) == 2
  assert sc.top_bar_length_px == 0.0


def test_compatibility_wrapper_and_rasterization() -> None:
  config = GenerationConfig(canvas_width_px=120, canvas_height_px=80, target_visible_px=50.0)
  legacy = build_bar_stem_svg(
    config,
    bar_stem_ratio=0.5,
    cross_bar_ratio=0.3,
    cross_bar_y=0.5,
    cross_bar_angle_deg=10.0,
    cross_bar_count=2,
    stroke_width=2.0,
  )
  assert len(_lines(legacy.svg)) == 4
  assert legacy.stem_length_px == 50.0
  assert legacy.bar_length_px == 25.0
  assert legacy.cross_bar_length_px == 15.0
  with TemporaryDirectory() as directory:
    output = Path(directory) / "bar-stem.png"
    render_png(legacy.svg, output)
    assert output.stat().st_size > 0


def test_invalid_geometry_inputs() -> None:
  config = GenerationConfig()
  placement = StitchPlacement((5.0, 20.0), (5.0, 5.0))
  base = dict(
    parent=Element("g"),
    sampled_values={"bar_stem_ratio": 0.5},
    config=config,
    placement=placement,
    stroke_width=2.0,
  )
  _expect_error(
    KeyError,
    "Unsupported reusable stitch class",
    lambda: append_stitch_geometry(class_name="missing", **base),
  )
  _expect_error(
    ValueError,
    "must be distinct",
    lambda: append_stitch_geometry(
      class_name="hdc", **{**base, "placement": StitchPlacement((1.0, 1.0), (1.0, 1.0))}
    ),
  )
  _expect_error(
    ValueError,
    "base_px[0] must be a finite number",
    lambda: append_stitch_geometry(
      class_name="hdc",
      **{**base, "placement": StitchPlacement((math.inf, 1.0), (1.0, 2.0))},
    ),
  )
  _expect_error(
    ValueError,
    "bar_stem_ratio must be a finite number",
    lambda: append_stitch_geometry(
      class_name="hdc", **{**base, "sampled_values": {}}
    ),
  )
  _expect_error(
    ValueError,
    "stroke_width must be positive",
    lambda: append_stitch_geometry(class_name="hdc", **{**base, "stroke_width": 0.0}),
  )
  _expect_error(
    ValueError,
    "cross_bar_angle_deg must be in the range",
    lambda: append_stitch_geometry(
      class_name="dc",
      **{
        **base,
        "sampled_values": {
          "bar_stem_ratio": 0.5,
          "cross_bar_ratio": 0.3,
          "cross_bar_y": 0.5,
          "cross_bar_angle_deg": 60.0,
        },
      },
    ),
  )


def main() -> None:
  test_registry_and_centered_geometry()
  test_rectangular_and_arbitrary_anchor_geometry()
  test_top_bar_suppression_and_bounds()
  test_compatibility_wrapper_and_rasterization()
  test_invalid_geometry_inputs()
  print("Reusable stitch-geometry checks passed.")


if __name__ == "__main__":
  main()
