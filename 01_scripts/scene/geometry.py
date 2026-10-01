"""Affine placement and polygon diagnostics for scene nodes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import Point


def placement_matrix(
  *,
  local_center: Point,
  scene_center: Point,
  rotation_deg: float,
) -> np.ndarray:
  angle = math.radians(rotation_deg)
  cosine, sine = math.cos(angle), math.sin(angle)
  cx, cy = local_center
  tx = scene_center[0] - cosine * cx + sine * cy
  ty = scene_center[1] - sine * cx - cosine * cy
  return np.array([
    [cosine, -sine, tx],
    [sine, cosine, ty],
    [0.0, 0.0, 1.0],
  ], dtype=np.float64)


def transform_points(points: Iterable[Point], matrix: np.ndarray) -> np.ndarray:
  values = np.asarray(tuple(points), dtype=np.float64)
  homogeneous = np.column_stack((values, np.ones(len(values))))
  return (homogeneous @ matrix.T)[:, :2]


def polygon_area(points: Iterable[Point]) -> float:
  polygon = np.asarray(tuple(points), dtype=np.float64)
  if len(polygon) < 3:
    return 0.0
  x, y = polygon[:, 0], polygon[:, 1]
  return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def clip_polygon_to_canvas(
  points: Iterable[Point],
  width_px: float,
  height_px: float,
) -> tuple[Point, ...]:
  polygon = [tuple(map(float, point)) for point in points]
  boundaries = (
    (lambda p: p[0] >= 0.0, lambda a, b: _vertical_intersection(a, b, 0.0)),
    (lambda p: p[0] <= width_px, lambda a, b: _vertical_intersection(a, b, width_px)),
    (lambda p: p[1] >= 0.0, lambda a, b: _horizontal_intersection(a, b, 0.0)),
    (lambda p: p[1] <= height_px, lambda a, b: _horizontal_intersection(a, b, height_px)),
  )
  for inside, intersection in boundaries:
    polygon = _clip_edge(polygon, inside, intersection)
  return tuple(polygon)


def _clip_edge(polygon, inside, intersection):
  if not polygon:
    return []
  result = []
  previous = polygon[-1]
  previous_inside = inside(previous)
  for current in polygon:
    current_inside = inside(current)
    if current_inside != previous_inside:
      result.append(intersection(previous, current))
    if current_inside:
      result.append(current)
    previous, previous_inside = current, current_inside
  return result


def _vertical_intersection(a: Point, b: Point, x: float) -> Point:
  if b[0] == a[0]:
    return x, a[1]
  t = (x - a[0]) / (b[0] - a[0])
  return x, a[1] + t * (b[1] - a[1])


def _horizontal_intersection(a: Point, b: Point, y: float) -> Point:
  if b[1] == a[1]:
    return a[0], y
  t = (y - a[1]) / (b[1] - a[1])
  return a[0] + t * (b[0] - a[0]), y


def visibility_fraction(points: Iterable[Point], width_px: int, height_px: int) -> float:
  polygon = tuple(points)
  area = polygon_area(polygon)
  if area <= 0:
    return 0.0
  visible = polygon_area(clip_polygon_to_canvas(polygon, width_px, height_px))
  return min(1.0, max(0.0, visible / area))


def projected_signed_gap(first: Iterable[Point], second: Iterable[Point]) -> float:
  a = np.asarray(tuple(first), dtype=np.float64)
  b = np.asarray(tuple(second), dtype=np.float64)
  center_a, center_b = a.mean(axis=0), b.mean(axis=0)
  axis = center_b - center_a
  distance = float(np.linalg.norm(axis))
  if distance == 0:
    return -min(_projection_span(a, np.array([1.0, 0.0])), _projection_span(b, np.array([1.0, 0.0])))
  unit = axis / distance
  radius_a = _projection_span(a - center_a, unit) / 2.0
  radius_b = _projection_span(b - center_b, unit) / 2.0
  return distance - radius_a - radius_b


def _projection_span(points: np.ndarray, axis: np.ndarray) -> float:
  values = points @ axis
  return float(values.max() - values.min())


def convex_intersection(subject: Iterable[Point], clip: Iterable[Point]) -> tuple[Point, ...]:
  output = [tuple(map(float, point)) for point in subject]
  clipper = [tuple(map(float, point)) for point in clip]
  orientation = _signed_area(clipper)
  for index, edge_start in enumerate(clipper):
    edge_end = clipper[(index + 1) % len(clipper)]
    if not output:
      break
    input_points, output = output, []
    previous = input_points[-1]
    for current in input_points:
      current_inside = _inside_edge(current, edge_start, edge_end, orientation)
      previous_inside = _inside_edge(previous, edge_start, edge_end, orientation)
      if current_inside != previous_inside:
        output.append(_line_intersection(previous, current, edge_start, edge_end))
      if current_inside:
        output.append(current)
      previous = current
  return tuple(output)


def _signed_area(points: list[Point]) -> float:
  return sum(
    points[index][0] * points[(index + 1) % len(points)][1]
    - points[(index + 1) % len(points)][0] * points[index][1]
    for index in range(len(points))
  ) / 2.0


def _inside_edge(point: Point, start: Point, end: Point, orientation: float) -> bool:
  cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
  return cross >= -1e-9 if orientation >= 0 else cross <= 1e-9


def _line_intersection(a: Point, b: Point, c: Point, d: Point) -> Point:
  ab = np.asarray(b) - np.asarray(a)
  cd = np.asarray(d) - np.asarray(c)
  denominator = ab[0] * cd[1] - ab[1] * cd[0]
  if abs(denominator) < 1e-12:
    return a
  offset = np.asarray(c) - np.asarray(a)
  t = (offset[0] * cd[1] - offset[1] * cd[0]) / denominator
  point = np.asarray(a) + t * ab
  return float(point[0]), float(point[1])
