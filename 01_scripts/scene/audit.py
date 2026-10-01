"""Pillow visualizations for every in-memory scene stage."""

from __future__ import annotations

from io import BytesIO
from math import ceil

from PIL import Image, ImageDraw

from .models import AssignedGraph, PlacedGraph, RealizedGraph, SlotGraph


def draw_slot_graph(graph: SlotGraph, *, size: int = 720) -> Image.Image:
  positions = _fit_positions(
    {node.slot_id: node.structural_position for node in graph.nodes}, size, 55,
  )
  image = Image.new("RGB", (size, size), "white")
  draw = ImageDraw.Draw(image)
  for edge in graph.edges:
    draw.line((positions[edge.source_id], positions[edge.target_id]), fill="#c8c8c8", width=2)
  role_colors = {"cells": "#4477aa", "center": "#cc6677", "outer": "#228833"}
  for node in graph.nodes:
    x, y = positions[node.slot_id]
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=role_colors.get(node.role, "#666666"))
    draw.text((x + 9, y - 7), node.slot_id, fill="#222222")
  draw.text((12, 10), f"Slot graph: {graph.pattern.pattern_type} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)", fill="black")
  return image


def draw_assigned_graph(graph: AssignedGraph, *, size: int = 720) -> Image.Image:
  positions = _fit_positions(
    {node.slot.slot_id: node.slot.structural_position for node in graph.nodes}, size, 55,
  )
  image = Image.new("RGB", (size, size), "white")
  draw = ImageDraw.Draw(image)
  for edge in graph.slot_graph.edges:
    draw.line((positions[edge.source_id], positions[edge.target_id]), fill="#d5d5d5", width=2)
  for node in graph.nodes:
    x, y = positions[node.slot.slot_id]
    color = _class_color(node.class_key)
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="black")
    draw.text((x + 10, y - 7), f"{node.class_key[1]}", fill="#111111")
  draw.text((12, 10), "Assigned graph: node colors and labels show selected classes", fill="black")
  return image


def draw_realized_contact_sheet(
  graph: RealizedGraph,
  *,
  columns: int = 6,
  tile_size: int = 170,
) -> Image.Image:
  rows = ceil(len(graph.nodes) / columns)
  sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
  for index, node in enumerate(graph.nodes):
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline="#dddddd")
    label = f"{node.assignment.slot.slot_id}: {node.assignment.class_key[1]}"
    draw.text((6, 5), label, fill="black")
    with Image.open(BytesIO(node.png_bytes)) as source:
      symbol = source.convert("RGBA")
    background = Image.new("RGBA", symbol.size, "white")
    background.alpha_composite(symbol)
    available = tile_size - 35
    scale = min(available / symbol.width, available / symbol.height)
    preview_size = (round(symbol.width * scale), round(symbol.height * scale))
    preview = background.convert("RGB").resize(preview_size, Image.Resampling.NEAREST)
    origin = ((tile_size - preview.width) // 2, 25 + (tile_size - 25 - preview.height) // 2)
    tile.paste(preview, origin)
    points = [
      (origin[0] + x * preview.width / symbol.width, origin[1] + y * preview.height / symbol.height)
      for x, y in node.local_obb
    ]
    draw.line(points + [points[0]], fill="red", width=2)
    sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
  return sheet


def draw_placed_graph(graph: PlacedGraph, *, preview_size: int = 900) -> Image.Image:
  canvas_width, canvas_height = graph.realized_graph.assigned_graph.slot_graph.pattern.canvas_size_px
  scale = min(preview_size / canvas_width, preview_size / canvas_height)
  image = Image.new("RGB", (round(canvas_width * scale), round(canvas_height * scale)), "white")
  draw = ImageDraw.Draw(image)
  placements = {item.node.assignment.slot.slot_id: item for item in graph.nodes}
  interaction_by_edge = {item.edge.edge_id: item for item in graph.interactions}
  for edge in graph.realized_graph.assigned_graph.slot_graph.edges:
    first, second = placements[edge.source_id], placements[edge.target_id]
    interaction = interaction_by_edge[edge.edge_id]
    color = "#cc3311" if interaction.intersects else "#bbbbbb"
    draw.line((_scale_point(first.center_px, scale), _scale_point(second.center_px, scale)), fill=color, width=2)
  for placement in graph.nodes:
    points = [_scale_point(point, scale) for point in placement.scene_obb]
    color = "#ee3377" if placement.visibility_fraction < 0.999999 else "#0077bb"
    draw.line(points + [points[0]], fill=color, width=2)
    center = _scale_point(placement.center_px, scale)
    draw.ellipse((center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3), fill="black")
  draw.text((10, 10), "Placed graph: blue OBB; red edge = intersecting; magenta OBB = clipped", fill="black")
  return image


def draw_final_overlay(graph: PlacedGraph, png_bytes: bytes, *, preview_size: int = 1000) -> Image.Image:
  with Image.open(BytesIO(png_bytes)) as source:
    image = source.convert("RGB")
  scale = min(preview_size / image.width, preview_size / image.height)
  if scale != 1.0:
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
  draw = ImageDraw.Draw(image)
  for placement in graph.nodes:
    points = [_scale_point(point, scale) for point in placement.scene_obb]
    draw.line(points + [points[0]], fill="red", width=2)
    draw.text(points[0], placement.node.assignment.class_key[1], fill="#0055cc")
  return image


def interaction_rows(graph: PlacedGraph) -> list[dict[str, object]]:
  return [
    {
      "edge": item.edge.edge_id,
      "relationship": item.edge.relationship,
      "source": item.edge.source_id,
      "target": item.edge.target_id,
      "distance_px": round(item.center_distance_px, 3),
      "signed_gap_px": round(item.signed_gap_px, 3),
      "overlap_area_px2": round(item.overlap_area_px2, 3),
      "intersects": item.intersects,
    }
    for item in graph.interactions
  ]


def node_rows(graph: PlacedGraph) -> list[dict[str, object]]:
  return [
    {
      "slot": item.node.assignment.slot.slot_id,
      "role": item.node.assignment.slot.role,
      "class": ".".join(item.node.assignment.class_key),
      "center_px": tuple(round(value, 3) for value in item.center_px),
      "rotation_deg": round(item.rotation_deg, 3),
      "visibility": round(item.visibility_fraction, 4),
      "symbol_seed": item.node.assignment.symbol_seed,
    }
    for item in graph.nodes
  ]


def _fit_positions(positions, size, margin):
  xs = [point[0] for point in positions.values()]
  ys = [point[1] for point in positions.values()]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span_x, span_y = max(max_x - min_x, 1e-9), max(max_y - min_y, 1e-9)
  scale = min((size - 2 * margin) / span_x, (size - 2 * margin) / span_y)
  center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
  return {
    key: (size / 2.0 + (point[0] - center_x) * scale, size / 2.0 + (point[1] - center_y) * scale)
    for key, point in positions.items()
  }


def _class_color(class_key):
  value = sum((index + 1) * ord(char) for index, char in enumerate(".".join(class_key)))
  return (60 + value % 150, 60 + (value // 7) % 150, 60 + (value // 17) % 150)


def _scale_point(point, scale):
  return point[0] * scale, point[1] * scale
