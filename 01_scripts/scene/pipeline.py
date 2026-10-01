"""End-to-end construction of an in-memory, auditable scene trace."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from generation.core.models import GenerationConfig
from generation.core.sampling import SamplingConfig
from generation.registry import COMPOSITE_GENERATORS, GENERATOR_REGISTRY
from labeling import format_yolo_obb_label, rasterize_and_label_upright_svg
from rasterization import rasterize_svg_supersampled

from .composition import compose_scene_svg
from .config import SceneConfiguration, parse_class_key
from .geometry import (
  convex_intersection,
  placement_matrix,
  polygon_area,
  projected_signed_gap,
  transform_points,
  visibility_fraction,
)
from .models import (
  AssignedGraph,
  AssignedNode,
  EdgeInteraction,
  GraphEdge,
  NodePlacement,
  PatternSample,
  PlacedGraph,
  RealizedGraph,
  RealizedNode,
  RenderedScene,
  SceneTrace,
  SlotGraph,
  SlotNode,
)
from .seeds import derive_seed


def generate_scene_trace(
  *,
  scene_config: SceneConfiguration,
  sampling_config: SamplingConfig,
  scene_seed: int,
  scene_id: str | None = None,
  pattern: str | None = None,
  pattern_overrides: Mapping[str, Any] | None = None,
  forced_role_classes: Mapping[str, tuple[tuple[str, str], ...]] | None = None,
  chart_scale: float | None = None,
  alpha_threshold: int = 0,
) -> SceneTrace:
  """Run every scene stage and retain each immutable boundary in memory."""
  pattern_sample = _sample_pattern(
    scene_config, scene_seed, scene_id, pattern, pattern_overrides, chart_scale,
  )
  slot_graph = _construct_slot_graph(scene_config, pattern_sample)
  assigned_graph = _assign_classes(
    scene_config, slot_graph, forced_role_classes or {},
  )
  realized_graph = _realize_nodes(
    scene_config, sampling_config, assigned_graph, alpha_threshold,
  )
  placed_graph = _place_graph(scene_config, realized_graph)
  rendered_scene = _render_scene(scene_config, placed_graph)
  return SceneTrace(
    pattern_sample, slot_graph, assigned_graph, realized_graph, placed_graph, rendered_scene,
  )


def _sample_pattern(
  config: SceneConfiguration,
  scene_seed: int,
  scene_id: str | None,
  forced_pattern: str | None,
  overrides: Mapping[str, Any] | None,
  forced_chart_scale: float | None,
) -> PatternSample:
  profile = config.profile
  weights = profile["pattern_weights"]
  pattern_rng = np.random.default_rng(derive_seed(scene_seed, "pattern"))
  pattern_type = forced_pattern or _weighted_choice(weights, pattern_rng)
  if pattern_type not in config.patterns["patterns"]:
    raise KeyError(f"Unknown scene pattern: {pattern_type!r}.")
  definitions = config.patterns["patterns"][pattern_type].get("parameters", {})
  overrides = dict(overrides or {})
  unknown = set(overrides) - set(definitions)
  if unknown:
    raise KeyError(f"Unknown {pattern_type} pattern override(s): {sorted(unknown)}.")
  parameters = {
    name: overrides[name] if name in overrides else _sample_value(
      definition, np.random.default_rng(derive_seed(scene_seed, "pattern_parameter", name)),
    )
    for name, definition in definitions.items()
  }
  scale_definition = config.patterns["symbol_generation"]["chart_scale"]
  scale = forced_chart_scale if forced_chart_scale is not None else _sample_value(
    scale_definition, np.random.default_rng(derive_seed(scene_seed, "chart_scale")),
  )
  if not math.isfinite(float(scale)) or float(scale) <= 0:
    raise ValueError("chart_scale must be positive and finite.")
  canvas = config.patterns["canvas"]
  return PatternSample(
    scene_id=scene_id or f"scene_{scene_seed}",
    scene_seed=scene_seed,
    profile_name=profile["name"],
    pattern_type=pattern_type,
    parameters=parameters,
    chart_scale=float(scale),
    canvas_size_px=(canvas["width_px"], canvas["height_px"]),
    config_identity=config.identity,
  )


def _sample_value(definition: Mapping[str, Any], rng: np.random.Generator) -> float | int:
  kind = definition.get("distribution")
  if kind == "uniform":
    return float(rng.uniform(float(definition["min"]), float(definition["max"])))
  if kind == "integer_uniform":
    return int(rng.integers(int(definition["min"]), int(definition["max"]) + 1))
  if kind == "truncated_normal":
    mean, std = float(definition["mean"]), float(definition["std"])
    low, high = float(definition["min"]), float(definition["max"])
    for _ in range(10_000):
      value = float(rng.normal(mean, std))
      if low <= value <= high:
        return value
    return float(np.clip(mean, low, high))
  raise ValueError(f"Unsupported scene distribution: {kind!r}.")


def _weighted_choice(weights: Mapping[str, float], rng: np.random.Generator) -> str:
  names = tuple(weights)
  probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
  probabilities /= probabilities.sum()
  return names[int(rng.choice(len(names), p=probabilities))]


def _construct_slot_graph(config: SceneConfiguration, sample: PatternSample) -> SlotGraph:
  if sample.pattern_type == "grid":
    return _grid_slots(sample)
  if sample.pattern_type == "radial":
    return _radial_slots(sample)
  raise KeyError(sample.pattern_type)


def _grid_slots(sample: PatternSample) -> SlotGraph:
  rows, columns = int(sample.parameters["rows"]), int(sample.parameters["columns"])
  nodes = []
  for row in range(rows):
    for column in range(columns):
      ordinal = row * columns + column
      is_perimeter = row in (0, rows - 1) or column in (0, columns - 1)
      nodes.append(SlotNode(
        slot_id=f"cells_{row:02d}_{column:02d}",
        role="outer_cells" if is_perimeter else "inner_cells",
        ordinal=ordinal,
        structural_position=(column - (columns - 1) / 2.0, row - (rows - 1) / 2.0),
        context={"row": row, "column": column, "rows": rows, "columns": columns},
      ))
  base_edges = []
  for row in range(rows):
    for column in range(columns):
      source = f"cells_{row:02d}_{column:02d}"
      if column + 1 < columns:
        target = f"cells_{row:02d}_{column + 1:02d}"
        base_edges.append(GraphEdge(f"h_{row:02d}_{column:02d}", source, target, "horizontal_neighbor"))
      if row + 1 < rows:
        target = f"cells_{row + 1:02d}_{column:02d}"
        base_edges.append(GraphEdge(f"v_{row:02d}_{column:02d}", source, target, "vertical_neighbor"))
  instruction_count = min(int(sample.parameters["instruction_count"]), len(base_edges))
  rng = np.random.default_rng(derive_seed(sample.scene_seed, "grid", "interstitial_edges"))
  selected_indices = set(
    int(value) for value in rng.choice(len(base_edges), size=instruction_count, replace=False)
  )
  positions = {node.slot_id: node.structural_position for node in nodes}
  edges = []
  instruction_index = 0
  for edge_index, edge in enumerate(base_edges):
    if edge_index not in selected_indices:
      edges.append(edge)
      continue
    source_position = positions[edge.source_id]
    target_position = positions[edge.target_id]
    midpoint = (
      (source_position[0] + target_position[0]) / 2.0,
      (source_position[1] + target_position[1]) / 2.0,
    )
    edge_angle = 0.0 if edge.relationship == "horizontal_neighbor" else 90.0
    slot_id = f"interstitial_{instruction_index:02d}"
    nodes.append(SlotNode(
      slot_id=slot_id,
      role="interstitial",
      ordinal=rows * columns + instruction_index,
      structural_position=midpoint,
      context={
        "between_source": edge.source_id,
        "between_target": edge.target_id,
        "replaced_edge_id": edge.edge_id,
        "replaced_relationship": edge.relationship,
        "edge_angle_deg": edge_angle,
      },
    ))
    edges.extend((
      GraphEdge(f"{edge.edge_id}_before", edge.source_id, slot_id, "interstitial_neighbor"),
      GraphEdge(f"{edge.edge_id}_after", slot_id, edge.target_id, "interstitial_neighbor"),
    ))
    instruction_index += 1
  return SlotGraph(sample, tuple(nodes), tuple(edges))


def _radial_slots(sample: PatternSample) -> SlotGraph:
  ring_count = int(sample.parameters["ring_count"])
  outer_count = int(sample.parameters["outer_count"])
  phase = float(sample.parameters["phase_deg"])
  nodes = [SlotNode("center_00", "center", 0, (0.0, 0.0), {"kind": "center"})]
  rings: list[list[str]] = []
  ordinal = 1
  for ring_index in range(1, ring_count + 1):
    radius_fraction = ring_index / ring_count
    count = max(1, int(math.floor(outer_count * radius_fraction + 0.5)))
    role = "ring_a" if ring_index % 2 == 1 else "ring_b"
    ring_ids = []
    for index in range(count):
      angle = phase + 360.0 * index / count
      radians = math.radians(angle)
      slot_id = f"ring_{ring_index:02d}_{index:03d}"
      ring_ids.append(slot_id)
      nodes.append(SlotNode(
        slot_id=slot_id,
        role=role,
        ordinal=ordinal,
        structural_position=(
          math.sin(radians) * radius_fraction,
          -math.cos(radians) * radius_fraction,
        ),
        context={
          "angle_deg": angle,
          "index": index,
          "count": count,
          "ring_index": ring_index,
          "ring_count": ring_count,
          "radius_fraction": radius_fraction,
        },
      ))
      ordinal += 1
    rings.append(ring_ids)
  cycle_edges = []
  radial_edges = []
  for ring_offset, ring_ids in enumerate(rings):
    ring_index = ring_offset + 1
    for index, source in enumerate(ring_ids):
      target = ring_ids[(index + 1) % len(ring_ids)]
      cycle_edges.append(GraphEdge(
        f"cycle_{ring_index:02d}_{index:03d}", source, target, "cycle_next",
      ))
      if ring_offset == 0:
        parent = "center_00"
        relationship = "center_to_ring"
      else:
        inner_ids = rings[ring_offset - 1]
        parent_index = int(math.floor(index * len(inner_ids) / len(ring_ids) + 0.5))
        parent = inner_ids[parent_index % len(inner_ids)]
        relationship = "adjacent_ring"
      radial_edges.append(GraphEdge(
        f"radial_{ring_index:02d}_{index:03d}", parent, source, relationship,
      ))
  instruction_count = min(int(sample.parameters["instruction_count"]), len(cycle_edges))
  rng = np.random.default_rng(derive_seed(sample.scene_seed, "radial", "interstitial_edges"))
  selected_indices = set(
    int(value) for value in rng.choice(len(cycle_edges), size=instruction_count, replace=False)
  )
  node_by_id = {node.slot_id: node for node in nodes}
  edges = []
  instruction_index = 0
  for edge_index, edge in enumerate(cycle_edges):
    if edge_index not in selected_indices:
      edges.append(edge)
      continue
    source_node = node_by_id[edge.source_id]
    target_node = node_by_id[edge.target_id]
    count = int(source_node.context["count"])
    ring_index = int(source_node.context["ring_index"])
    radius_fraction = float(source_node.context["radius_fraction"])
    angle = float(source_node.context["angle_deg"]) + 180.0 / count
    radians = math.radians(angle)
    midpoint = (
      math.sin(radians) * radius_fraction,
      -math.cos(radians) * radius_fraction,
    )
    delta_x = target_node.structural_position[0] - source_node.structural_position[0]
    delta_y = target_node.structural_position[1] - source_node.structural_position[1]
    edge_angle = math.degrees(math.atan2(delta_y, delta_x))
    slot_id = f"ring_interstitial_{instruction_index:02d}"
    nodes.append(SlotNode(
      slot_id=slot_id,
      role="interstitial",
      ordinal=ordinal + instruction_index,
      structural_position=midpoint,
      context={
        "between_source": edge.source_id,
        "between_target": edge.target_id,
        "replaced_edge_id": edge.edge_id,
        "replaced_relationship": edge.relationship,
        "edge_angle_deg": edge_angle,
        "angle_deg": angle,
        "ring_index": ring_index,
        "ring_count": ring_count,
        "radius_fraction": radius_fraction,
      },
    ))
    edges.extend((
      GraphEdge(f"{edge.edge_id}_before", edge.source_id, slot_id, "interstitial_neighbor"),
      GraphEdge(f"{edge.edge_id}_after", slot_id, edge.target_id, "interstitial_neighbor"),
    ))
    instruction_index += 1
  edges.extend(radial_edges)
  return SlotGraph(sample, tuple(nodes), tuple(edges))


def _assign_classes(
  config: SceneConfiguration,
  graph: SlotGraph,
  forced_role_classes: Mapping[str, tuple[tuple[str, str], ...]],
) -> AssignedGraph:
  pattern_definition = config.patterns["patterns"][graph.pattern.pattern_type]
  pool_definitions = config.patterns["class_pools"]
  profile_weights = config.profile["class_weights"]
  nodes = []
  role_indices: dict[str, int] = {}
  for slot in graph.nodes:
    pool_name = pattern_definition["roles"][slot.role]["class_pool"]
    eligible = tuple(parse_class_key(item) for item in pool_definitions[pool_name])
    weights = tuple(float(profile_weights[pool_name][f"{key[0]}.{key[1]}"]) for key in eligible)
    class_seed = derive_seed(graph.pattern.scene_seed, slot.slot_id, "class")
    forced = forced_role_classes.get(slot.role)
    if forced:
      index = role_indices.get(slot.role, 0)
      class_key = forced[index % len(forced)]
      role_indices[slot.role] = index + 1
      if class_key not in eligible:
        raise ValueError(f"Forced class {class_key!r} is not eligible for role {slot.role!r}.")
    else:
      rng = np.random.default_rng(class_seed)
      probabilities = np.asarray(weights, dtype=float)
      probabilities /= probabilities.sum()
      class_key = eligible[int(rng.choice(len(eligible), p=probabilities))]
    nodes.append(AssignedNode(
      slot=slot,
      class_key=class_key,
      eligible_classes=eligible,
      eligible_weights=weights,
      class_seed=class_seed,
      symbol_seed=derive_seed(graph.pattern.scene_seed, slot.slot_id, "symbol"),
      pose_seed=derive_seed(graph.pattern.scene_seed, slot.slot_id, "pose"),
    ))
  return AssignedGraph(graph, tuple(nodes))


def _realize_nodes(
  config: SceneConfiguration,
  sampling_config: SamplingConfig,
  graph: AssignedGraph,
  alpha_threshold: int,
) -> RealizedGraph:
  settings = config.patterns["symbol_generation"]
  raster_scale = int(config.rasterization["local_label_supersample_factor"])
  base = GenerationConfig(
    canvas_width_px=int(settings["canvas_width_px"]),
    canvas_height_px=int(settings["canvas_height_px"]),
    target_visible_px=float(settings["reference_visible_px"]) * graph.slot_graph.pattern.chart_scale,
  )
  nodes = []
  for assignment in graph.nodes:
    rng = np.random.default_rng(assignment.symbol_seed)
    sample = sampling_config.sample(
      *assignment.class_key, rng, seed=assignment.symbol_seed,
      case_id=graph.slot_graph.pattern.scene_id,
    )
    generator_input = (
      sampling_config.realize_components(*assignment.class_key, sample, rng)
      if assignment.class_key in COMPOSITE_GENERATORS else sample
    )
    generation_config = config.symbol_metrics.config_for(assignment.class_key, base)
    spec = sampling_config.resolve(*assignment.class_key)
    generated = GENERATOR_REGISTRY[assignment.class_key](spec, generator_input, generation_config)
    png_bytes = rasterize_and_label_upright_svg(
      generated,
      supersample_factor=raster_scale,
      alpha_threshold=alpha_threshold,
    )
    local_obb = tuple(tuple(map(float, point)) for point in generated.obb_pixels)
    nodes.append(RealizedNode(assignment, generated, png_bytes, generation_config, local_obb))
  return RealizedGraph(graph, tuple(nodes))


def _place_graph(config: SceneConfiguration, graph: RealizedGraph) -> PlacedGraph:
  sample = graph.assigned_graph.slot_graph.pattern
  width, height = sample.canvas_size_px
  canvas_center = (width / 2.0, height / 2.0)
  pattern_definition = config.patterns["patterns"][sample.pattern_type]
  placements = []
  warnings = []
  for node in graph.nodes:
    slot = node.assignment.slot
    if sample.pattern_type == "grid":
      center = (
        canvas_center[0] + slot.structural_position[0] * float(sample.parameters["pitch_x_px"]),
        canvas_center[1] + slot.structural_position[1] * float(sample.parameters["pitch_y_px"]),
      )
    else:
      radius = float(sample.parameters["radius_px"])
      center = (
        canvas_center[0] + slot.structural_position[0] * radius,
        canvas_center[1] + slot.structural_position[1] * radius,
      )
    orientation = pattern_definition["roles"][slot.role]["orientation"]
    orientation_kind = orientation["kind"]
    base_angle = float(orientation.get("angle_deg", 0.0))
    if orientation_kind == "radial_outward":
      base_angle = float(slot.context["angle_deg"]) + float(orientation.get("offset_deg", 0.0))
    elif orientation_kind == "edge_aligned":
      base_angle = float(slot.context["edge_angle_deg"]) + float(orientation.get("offset_deg", 0.0))
    elif orientation_kind != "fixed":
      raise ValueError(f"Unsupported orientation kind: {orientation_kind!r}.")
    jitter = orientation.get("jitter_deg")
    if jitter:
      base_angle += float(_sample_value(jitter, np.random.default_rng(node.assignment.pose_seed)))
    local_width, local_height = node.generated.canvas_size_px
    matrix = placement_matrix(
      local_center=(local_width / 2.0, local_height / 2.0),
      scene_center=center,
      rotation_deg=base_angle,
    )
    scene_obb_array = transform_points(node.local_obb, matrix)
    scene_obb = tuple(tuple(map(float, point)) for point in scene_obb_array)
    visibility = visibility_fraction(scene_obb, width, height)
    if visibility < 0.999999:
      warnings.append(f"{slot.slot_id} visibility={visibility:.3f}")
    placements.append(NodePlacement(
      node=node,
      center_px=center,
      rotation_deg=base_angle,
      transform=tuple(tuple(map(float, row)) for row in matrix),
      scene_obb=scene_obb,
      visibility_fraction=visibility,
    ))
  by_id = {item.node.assignment.slot.slot_id: item for item in placements}
  interactions = []
  for edge in graph.assigned_graph.slot_graph.edges:
    first, second = by_id[edge.source_id], by_id[edge.target_id]
    center_distance = math.dist(first.center_px, second.center_px)
    overlap_area = polygon_area(convex_intersection(first.scene_obb, second.scene_obb))
    interactions.append(EdgeInteraction(
      edge=edge,
      center_distance_px=center_distance,
      signed_gap_px=projected_signed_gap(first.scene_obb, second.scene_obb),
      overlap_area_px2=overlap_area,
      intersects=overlap_area > 1e-8,
    ))
  return PlacedGraph(graph, tuple(placements), tuple(interactions), tuple(warnings))


def _render_scene(config: SceneConfiguration, graph: PlacedGraph) -> RenderedScene:
  realized_graph = graph.realized_graph
  assigned_graph = realized_graph.assigned_graph
  width, height = assigned_graph.slot_graph.pattern.canvas_size_px
  svg = compose_scene_svg(graph.nodes, width, height)
  png_bytes = rasterize_svg_supersampled(
    svg,
    output_width=width,
    output_height=height,
    supersample_factor=int(config.rasterization["scene_supersample_factor"]),
  )
  labels, node_ids = [], []
  for placement in graph.nodes:
    labels.append(format_yolo_obb_label(
      placement.node.generated.class_id,
      np.asarray(placement.scene_obb, dtype=np.float64),
      width,
      height,
    ))
    node_ids.append(placement.node.assignment.slot.slot_id)
  return RenderedScene(svg, png_bytes, tuple(labels), tuple(node_ids))
