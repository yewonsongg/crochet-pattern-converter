"""Deterministic end-to-end checks for the auditable scene pipeline."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import sys
from xml.etree.ElementTree import fromstring

from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "01_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.sampling import load_sampling_config
from scene import generate_scene_trace, load_audit_cases, load_scene_configuration


CONFIG_ROOT = PROJECT_ROOT / "06_configs"
SCENE_CONFIG_ROOT = CONFIG_ROOT / "scene_sampling"


def _load(profile: str):
  sampling = load_sampling_config(
    ontology_path=CONFIG_ROOT / "ontology.yaml",
    sampling_path=CONFIG_ROOT / "symbol_sampling.yaml",
  )
  scene = load_scene_configuration(
    patterns_path=SCENE_CONFIG_ROOT / "scene_patterns.yaml",
    profile_path=SCENE_CONFIG_ROOT / f"{profile}.yaml",
    metrics_path=CONFIG_ROOT / "symbol_metrics.yaml",
    rasterization_path=CONFIG_ROOT / "rasterization.yaml",
    sampling_config=sampling,
  )
  return sampling, scene


def _generate(case, sampling, scene):
  return generate_scene_trace(
    scene_config=scene,
    sampling_config=sampling,
    scene_seed=case.seed,
    scene_id=case.case_id,
    pattern=case.pattern,
    pattern_overrides=case.pattern_overrides,
    forced_role_classes=case.forced_role_classes,
    chart_scale=case.chart_scale,
  )


def _assert_common_contract(trace) -> None:
  count = len(trace.slot_graph.nodes)
  assert count > 0
  assert len(trace.assigned_graph.nodes) == count
  assert len(trace.realized_graph.nodes) == count
  assert len(trace.placed_graph.nodes) == count
  assert len(trace.rendered_scene.yolo_labels) == count
  assert len(trace.rendered_scene.labeled_node_ids) == count
  assert trace.pattern_sample.canvas_size_px == (1280, 1280)
  assert trace.pattern_sample.config_identity.rasterization_digest
  assert Image.open(BytesIO(trace.rendered_scene.png_bytes)).size == (1280, 1280)
  fromstring(trace.rendered_scene.scene_svg)
  for node, placement, label in zip(
    trace.realized_graph.nodes,
    trace.placed_graph.nodes,
    trace.rendered_scene.yolo_labels,
  ):
    assert len(node.local_obb) == 4
    assert Image.open(BytesIO(node.png_bytes)).size == node.generated.canvas_size_px
    assert len(placement.scene_obb) == 4
    assert 0.0 <= placement.visibility_fraction <= 1.0
    fields = label.split()
    assert len(fields) == 9
    assert int(fields[0]) == node.generated.class_id


def main() -> None:
  cases = load_audit_cases(SCENE_CONFIG_ROOT / "audit_cases.yaml")
  sampling, scene = _load("exposure")

  sparse = _generate(cases["grid_sparse"], sampling, scene)
  _assert_common_contract(sparse)
  assert scene.rasterization["scene_supersample_factor"] == 2
  assert scene.rasterization["local_label_supersample_factor"] == 4
  instruction_count = int(sparse.pattern_sample.parameters["instruction_count"])
  assert len(sparse.slot_graph.nodes) == 16 + instruction_count
  assert len(sparse.slot_graph.edges) == 24 + instruction_count
  assert sum(node.role == "outer_cells" for node in sparse.slot_graph.nodes) == 12
  assert sum(node.role == "inner_cells" for node in sparse.slot_graph.nodes) == 4
  assert sum(node.role == "interstitial" for node in sparse.slot_graph.nodes) == instruction_count
  logical_values = [
    value
    for node in sparse.realized_graph.nodes
    for point in node.local_obb
    for value in point
  ]
  assert all(abs(value * 4 - round(value * 4)) < 1e-9 for value in logical_values)
  assert any(abs(value - round(value)) > 1e-9 for value in logical_values)

  native_render_config = replace(
    scene,
    rasterization={**scene.rasterization, "scene_supersample_factor": 1},
  )
  native_render = _generate(cases["grid_sparse"], sampling, native_render_config)
  assert native_render.rendered_scene.yolo_labels == sparse.rendered_scene.yolo_labels
  assert Image.open(BytesIO(native_render.rendered_scene.png_bytes)).size == (1280, 1280)
  assert sha256(native_render.rendered_scene.png_bytes).digest() != sha256(
    sparse.rendered_scene.png_bytes
  ).digest()

  repeated = _generate(cases["grid_sparse"], sampling, scene)
  assert [node.class_key for node in sparse.assigned_graph.nodes] == [
    node.class_key for node in repeated.assigned_graph.nodes
  ]
  assert [node.generated.sampled_parameters for node in sparse.realized_graph.nodes] == [
    node.generated.sampled_parameters for node in repeated.realized_graph.nodes
  ]
  assert sparse.rendered_scene.yolo_labels == repeated.rendered_scene.yolo_labels
  assert sha256(sparse.rendered_scene.png_bytes).digest() == sha256(
    repeated.rendered_scene.png_bytes
  ).digest()

  all_classes = _generate(cases["grid_all_classes"], sampling, scene)
  _assert_common_contract(all_classes)
  for role, expected in cases["grid_all_classes"].forced_role_classes.items():
    actual = tuple(
      node.class_key for node in all_classes.assigned_graph.nodes if node.slot.role == role
    )
    assert actual == tuple(expected[index % len(expected)] for index in range(len(actual)))

  interstitial = _generate(cases["grid_interstitial"], sampling, scene)
  _assert_common_contract(interstitial)
  inserted = [node for node in interstitial.slot_graph.nodes if node.role == "interstitial"]
  assert len(inserted) == 4
  assert len({node.context["replaced_edge_id"] for node in inserted}) == 4
  positions = {node.slot_id: node.structural_position for node in interstitial.slot_graph.nodes}
  edge_ids = {edge.edge_id for edge in interstitial.slot_graph.edges}
  for index, node in enumerate(inserted):
    source = positions[node.context["between_source"]]
    target = positions[node.context["between_target"]]
    assert node.structural_position == (
      (source[0] + target[0]) / 2.0,
      (source[1] + target[1]) / 2.0,
    )
    assert node.context["replaced_edge_id"] not in edge_ids
    assert node.context["edge_angle_deg"] in (0.0, 90.0)
    expected_classes = cases["grid_interstitial"].forced_role_classes["interstitial"]
    assigned = next(item for item in interstitial.assigned_graph.nodes if item.slot.slot_id == node.slot_id)
    assert assigned.class_key == expected_classes[index % len(expected_classes)]

  radial = _generate(cases["radial_center_ring"], sampling, scene)
  _assert_common_contract(radial)
  assert radial.assigned_graph.nodes[0].class_key == ("instructive", "ring")
  ring_count = int(radial.pattern_sample.parameters["ring_count"])
  outer_count = int(radial.pattern_sample.parameters["outer_count"])
  ring_node_count = sum(
    max(1, math.floor(outer_count * ring_index / ring_count + 0.5))
    for ring_index in range(1, ring_count + 1)
  )
  radial_instruction_count = int(radial.pattern_sample.parameters["instruction_count"])
  assert len(radial.slot_graph.nodes) == ring_node_count + 1 + radial_instruction_count
  assert len(radial.slot_graph.edges) == ring_node_count * 2 + radial_instruction_count
  for node in radial.slot_graph.nodes:
    if node.role not in {"ring_a", "ring_b"}:
      continue
    expected_role = "ring_a" if node.context["ring_index"] % 2 == 1 else "ring_b"
    assert node.role == expected_role
  assert {
    node.role for node in radial.slot_graph.nodes
    if node.context.get("ring_index") == 1 and node.role in {"ring_a", "ring_b"}
  } == {
    "ring_a"
  }

  radial_interstitial = _generate(cases["radial_interstitial"], sampling, scene)
  _assert_common_contract(radial_interstitial)
  inserted = [
    node for node in radial_interstitial.slot_graph.nodes if node.role == "interstitial"
  ]
  assert len(inserted) == 4
  edge_ids = {edge.edge_id for edge in radial_interstitial.slot_graph.edges}
  expected_classes = cases["radial_interstitial"].forced_role_classes["interstitial"]
  for index, node in enumerate(inserted):
    assert node.context["replaced_relationship"] == "cycle_next"
    assert node.context["replaced_edge_id"] not in edge_ids
    assert math.isclose(
      math.hypot(*node.structural_position),
      node.context["radius_fraction"],
      abs_tol=1e-12,
    )
    assigned = next(
      item for item in radial_interstitial.assigned_graph.nodes
      if item.slot.slot_id == node.slot_id
    )
    assert assigned.class_key == expected_classes[index % len(expected_classes)]

  overlap = _generate(cases["overlap_stress"], sampling, scene)
  _assert_common_contract(overlap)
  assert any(interaction.intersects for interaction in overlap.placed_graph.interactions)
  assert any(interaction.overlap_area_px2 > 0 for interaction in overlap.placed_graph.interactions)

  checked = {
    "grid_sparse",
    "grid_all_classes",
    "grid_interstitial",
    "radial_center_ring",
    "radial_interstitial",
    "overlap_stress",
  }
  for case_id, case in cases.items():
    if case_id not in checked:
      _assert_common_contract(_generate(case, sampling, scene))

  # Loading the second profile proves both override files satisfy the shared schema.
  _load("realism")
  print("Auditable scene pipeline checks passed.")


if __name__ == "__main__":
  main()
