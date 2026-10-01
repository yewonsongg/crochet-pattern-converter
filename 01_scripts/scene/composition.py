"""Compose generated node SVGs into one vector scene."""

from __future__ import annotations

from copy import deepcopy
from xml.etree.ElementTree import Element, SubElement, fromstring, tostring

from .models import NodePlacement


SVG_NS = "http://www.w3.org/2000/svg"


def compose_scene_svg(
  placements: tuple[NodePlacement, ...],
  width_px: int,
  height_px: int,
) -> str:
  root = Element("svg", {
    "xmlns": SVG_NS,
    "width": f"{width_px}px",
    "height": f"{height_px}px",
    "viewBox": f"0 0 {width_px} {height_px}",
  })
  SubElement(root, "rect", {
    "x": "0", "y": "0", "width": str(width_px), "height": str(height_px),
    "fill": "white",
  })
  for placement in placements:
    generated = placement.node.generated
    local_width, local_height = generated.canvas_size_px
    cx, cy = placement.center_px
    group = SubElement(root, "g", {
      "id": placement.node.assignment.slot.slot_id,
      "transform": f"rotate({placement.rotation_deg:.8f} {cx:.8f} {cy:.8f})",
    })
    source = fromstring(generated.svg)
    nested = SubElement(group, "svg", {
      "x": f"{cx - local_width / 2.0:.8f}",
      "y": f"{cy - local_height / 2.0:.8f}",
      "width": f"{local_width:.8f}",
      "height": f"{local_height:.8f}",
      "viewBox": source.attrib.get("viewBox", f"0 0 {local_width} {local_height}"),
      "overflow": "visible",
    })
    for child in source:
      nested.append(deepcopy(child))
  return tostring(root, encoding="unicode")
