"""Deterministic end-to-end checks for the auditable scene pipeline."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
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
  assert Image.open(BytesIO(trace.rendered_scene.png_bytes)).size == (1280, 1280)
  fromstring(trace.rendered_scene.scene_svg)
  for node, placement, label in zip(
    trace.realized_graph.nodes,
    trace.placed_graph.nodes,
    trace.rendered_scene.yolo_labels,
  ):
    assert len(node.local_obb) == 4
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
  assert len(sparse.slot_graph.nodes) == 16
  assert len(sparse.slot_graph.edges) == 24

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
  expected = cases["grid_all_classes"].forced_role_classes["cells"]
  actual = tuple(node.class_key for node in all_classes.assigned_graph.nodes)
  assert actual == tuple(expected[index % len(expected)] for index in range(len(actual)))

  radial = _generate(cases["radial_center_ring"], sampling, scene)
  _assert_common_contract(radial)
  assert radial.assigned_graph.nodes[0].class_key == ("instructive", "ring")
  assert len(radial.slot_graph.nodes) == 13
  assert len(radial.slot_graph.edges) == 24

  overlap = _generate(cases["overlap_stress"], sampling, scene)
  _assert_common_contract(overlap)
  assert any(interaction.intersects for interaction in overlap.placed_graph.interactions)
  assert any(interaction.overlap_area_px2 > 0 for interaction in overlap.placed_graph.interactions)

  checked = {"grid_sparse", "grid_all_classes", "radial_center_ring", "overlap_stress"}
  for case_id, case in cases.items():
    if case_id not in checked:
      _assert_common_contract(_generate(case, sampling, scene))

  # Loading the second profile proves both override files satisfy the shared schema.
  _load("realism")
  print("Auditable scene pipeline checks passed.")


if __name__ == "__main__":
  main()
