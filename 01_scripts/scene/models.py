"""Immutable records for each auditable scene-construction boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from generation.core.models import GeneratedObject, GenerationConfig


ClassKey = tuple[str, str]
Point = tuple[float, float]


@dataclass(frozen=True)
class SceneConfigIdentity:
  patterns_path: str
  patterns_digest: str
  profile_path: str
  profile_digest: str
  metrics_path: str
  metrics_digest: str
  rasterization_path: str
  rasterization_digest: str


@dataclass(frozen=True)
class PatternSample:
  scene_id: str
  scene_seed: int
  profile_name: str
  pattern_type: str
  parameters: Mapping[str, Any]
  chart_scale: float
  canvas_size_px: tuple[int, int]
  config_identity: SceneConfigIdentity


@dataclass(frozen=True)
class SlotNode:
  slot_id: str
  role: str
  ordinal: int
  structural_position: Point
  context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
  edge_id: str
  source_id: str
  target_id: str
  relationship: str


@dataclass(frozen=True)
class SlotGraph:
  pattern: PatternSample
  nodes: tuple[SlotNode, ...]
  edges: tuple[GraphEdge, ...]


@dataclass(frozen=True)
class AssignedNode:
  slot: SlotNode
  class_key: ClassKey
  eligible_classes: tuple[ClassKey, ...]
  eligible_weights: tuple[float, ...]
  class_seed: int
  symbol_seed: int
  pose_seed: int


@dataclass(frozen=True)
class AssignedGraph:
  slot_graph: SlotGraph
  nodes: tuple[AssignedNode, ...]


@dataclass(frozen=True)
class RealizedNode:
  assignment: AssignedNode
  generated: GeneratedObject
  png_bytes: bytes
  generation_config: GenerationConfig
  local_obb: tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class RealizedGraph:
  assigned_graph: AssignedGraph
  nodes: tuple[RealizedNode, ...]


@dataclass(frozen=True)
class NodePlacement:
  node: RealizedNode
  center_px: Point
  rotation_deg: float
  transform: tuple[tuple[float, float, float], ...]
  scene_obb: tuple[Point, Point, Point, Point]
  visibility_fraction: float


@dataclass(frozen=True)
class EdgeInteraction:
  edge: GraphEdge
  center_distance_px: float
  signed_gap_px: float
  overlap_area_px2: float
  intersects: bool


@dataclass(frozen=True)
class PlacedGraph:
  realized_graph: RealizedGraph
  nodes: tuple[NodePlacement, ...]
  interactions: tuple[EdgeInteraction, ...]
  warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedScene:
  scene_svg: str
  png_bytes: bytes
  yolo_labels: tuple[str, ...]
  labeled_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class SceneTrace:
  pattern_sample: PatternSample
  slot_graph: SlotGraph
  assigned_graph: AssignedGraph
  realized_graph: RealizedGraph
  placed_graph: PlacedGraph
  rendered_scene: RenderedScene

  def placement_by_id(self) -> dict[str, NodePlacement]:
    return {item.node.assignment.slot.slot_id: item for item in self.placed_graph.nodes}
