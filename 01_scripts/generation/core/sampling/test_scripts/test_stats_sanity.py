"""Sample every configured class and perform basic distribution sanity checks."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_ROOT.parents[3]
PROJECT_ROOT = SCRIPT_ROOT.parents[4]
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from generation.core.sampling import load_sampling_config


SAMPLE_COUNT = 1_000
BASE_SEED = 1234
REPORT_PATH = SCRIPT_ROOT / "stats_sanity_report.json"


def _distribution_expectations(
  spec: Mapping[str, Any],
  path: str,
  expected: dict[str, set[Any]],
  bounds: dict[str, list[tuple[float, float]]],
) -> None:
  """Collect support and bounds from one distribution, recursively."""

  distribution_type = spec["type"]

  if distribution_type == "categorical":
    expected[path].update(
      value for value, probability in spec["probabilities"].items()
      if float(probability) > 0
    )
  elif distribution_type == "truncated_normal":
    bounds[path].append((float(spec["min"]), float(spec["max"])))
  elif distribution_type == "conditional":
    expected[path].update(spec["cases"])
    for case, case_spec in spec["cases"].items():
      _distribution_expectations(
        case_spec,
        f"{path}.cases[{case}]",
        expected,
        bounds,
      )
  elif distribution_type == "mixture":
    expected[path].update(
      index for index, component in enumerate(spec["components"])
      if float(component["weight"]) > 0
    )
    for index, component in enumerate(spec["components"]):
      _distribution_expectations(
        component["distribution"],
        f"{path}.components[{index}]",
        expected,
        bounds,
      )


def _class_expectations(spec) -> tuple[dict[str, set[Any]], dict[str, list[tuple[float, float]]]]:
  expected: dict[str, set[Any]] = defaultdict(set)
  bounds: dict[str, list[tuple[float, float]]] = defaultdict(list)

  for name, parameter in spec.parameters.items():
    _distribution_expectations(parameter.distribution, name, expected, bounds)

  for variant in spec.variants.values():
    for name, parameter in variant.get("parameters", {}).items():
      distribution = parameter.get("distribution")
      if distribution is None and parameter.get("kind") == "derived":
        distribution = {
          "type": "derived",
          **{key: value for key, value in parameter.items() if key != "kind"},
        }
      if distribution is not None:
        _distribution_expectations(distribution, name, expected, bounds)

  connector = spec.topology.raw.get("connector", {})
  if connector.get("enabled") and connector.get("distribution"):
    _distribution_expectations(
      connector["distribution"],
      "topology.connector",
      expected,
      bounds,
    )

  return expected, bounds


def _parameter_distributions(spec) -> dict[str, list[dict[str, Any]]]:
  distributions: dict[str, list[dict[str, Any]]] = defaultdict(list)

  for name, parameter in spec.parameters.items():
    distributions[name].append(dict(parameter.distribution))

  for variant_name, variant in spec.variants.items():
    for name, parameter in variant.get("parameters", {}).items():
      distribution = parameter.get("distribution")
      if distribution is None and parameter.get("kind") == "derived":
        distribution = {
          "type": "derived",
          **{key: value for key, value in parameter.items() if key != "kind"},
        }
      if distribution is not None:
        distributions[name].append({"variant": variant_name, **distribution})

  return distributions


def _actual_statistics(values: list[Any], *, categorical: bool = False) -> dict[str, Any]:
  if not values:
    return {"sample_count": 0, "observed": []}

  observed = sorted(set(values), key=str)
  if not categorical and all(
    isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
    for value in values
  ):
    numeric = [float(value) for value in values]
    return {
      "sample_count": len(values),
      "min": min(numeric),
      "max": max(numeric),
      "mean": float(np.mean(numeric)),
    }

  counts = {str(value): values.count(value) for value in observed}
  return {
    "sample_count": len(values),
    "observed": observed,
    "counts": counts,
    "frequencies": {
      value: count / len(values)
      for value, count in counts.items()
    },
  }


def _conditional_actual_statistics(
  values: dict[str, list[Any]],
  parameter_name: str,
  *,
  categorical: bool,
) -> dict[str, Any]:
  prefix = f"{parameter_name}.cases["
  branches = {
    path[len(prefix):-1]: _actual_statistics(branch_values, categorical=categorical)
    for path, branch_values in values.items()
    if path.startswith(prefix) and branch_values
  }
  return {"by_dependency_value": branches}


def _variant_actual_statistics(
  values: dict[str, list[Any]],
  parameter_name: str,
  distributions: list[dict[str, Any]],
) -> dict[str, Any]:
  """Summarize a parameter separately for each declared variant."""

  by_variant: dict[str, dict[str, Any]] = {}
  for distribution in distributions:
    variant = distribution.get("variant")
    if variant is None:
      continue
    variant_values = values.get(variant, [])
    by_variant[variant] = _actual_statistics(
      variant_values,
      categorical=_contains_distribution_type([distribution], "categorical"),
    )
  return {"by_variant": by_variant}


def _contains_distribution_type(distributions: list[dict[str, Any]], distribution_type: str) -> bool:
  """Return whether a parameter has the requested type anywhere in its declaration."""

  def contains(spec: Mapping[str, Any]) -> bool:
    if spec.get("type") == distribution_type:
      return True
    if spec.get("type") == "conditional":
      return any(contains(case) for case in spec["cases"].values())
    if spec.get("type") == "mixture":
      return any(contains(component["distribution"]) for component in spec["components"])
    return False

  return any(contains(distribution) for distribution in distributions)


def _parameter_actual_statistics(name, distributions, parameter_values, parameter_values_by_variant, decision_values):
  if any("variant" in distribution for distribution in distributions):
    return _variant_actual_statistics(parameter_values_by_variant[name], name, distributions)
  if any(distribution.get("type") == "conditional" for distribution in distributions):
    return _conditional_actual_statistics(
      decision_values, name,
      categorical=_contains_distribution_type(distributions, "categorical"),
    )
  return _actual_statistics(
    parameter_values.get(name, []),
    categorical=_contains_distribution_type(distributions, "categorical"),
  )


def _validate_class(config, class_group: str, class_name: str, rng: np.random.Generator) -> dict[str, Any]:
  spec = config.resolve(class_group, class_name)
  expected, bounds = _class_expectations(spec)
  parameter_distributions = _parameter_distributions(spec)
  observed: dict[str, set[Any]] = defaultdict(set)
  numeric_ranges: dict[str, list[float]] = defaultdict(list)
  parameter_values: dict[str, list[Any]] = defaultdict(list)
  parameter_values_by_variant: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
  decision_values: dict[str, list[Any]] = defaultdict(list)

  for _ in range(SAMPLE_COUNT):
    sample = config.sample(class_group, class_name, rng)
    selected_variant = sample.parameters.get("variant")
    for name, value in sample.as_dict().items():
      parameter_values[name].append(value)
      if selected_variant is not None:
        parameter_values_by_variant[name][selected_variant].append(value)
    for path, decision in sample.provenance.decisions.items():
      if "value" in decision:
        observed[path].add(decision["value"])
        decision_values[path].append(decision["value"])
      if (
        path in bounds
        and decision.get("distribution") == "truncated_normal"
        and "value" in decision
      ):
        value = float(decision["value"])
        assert any(low <= value <= high for low, high in bounds[path]), (
          f"{class_group}.{class_name} {path}={value} outside {bounds[path]}"
        )
        numeric_ranges[path].append(value)
      if path in expected and "case" in decision:
        observed[path].add(decision["case"])
      if path in expected and "component" in decision:
        observed[path].add(decision["component"])

  for path, required in expected.items():
    missing = required - observed[path]
    assert not missing, (
      f"{class_group}.{class_name} {path} missing outcomes: {sorted(missing, key=str)}"
    )

  discrete = ", ".join(
    f"{path}={sorted(values, key=str)}"
    for path, values in sorted(observed.items())
    if path in expected
  )
  numeric = ", ".join(
    f"{path}=[{min(values):.4g}, {max(values):.4g}]"
    for path, values in sorted(numeric_ranges.items())
  )
  details = "; ".join(part for part in (discrete, numeric) if part)
  print(f"PASS  {class_group}.{class_name}: {SAMPLE_COUNT} samples" + (f" ({details})" if details else ""))
  return {
    "class_group": class_group,
    "class_name": class_name,
    "sample_count": SAMPLE_COUNT,
    "parameters": {
        name: {
          "expected_distributions": distributions,
        "actual": _parameter_actual_statistics(
          name, distributions, parameter_values,
          parameter_values_by_variant, decision_values,
        ),
      }
      for name, distributions in sorted(parameter_distributions.items())
    },
    "decision_coverage": {
      path: sorted(values, key=str)
      for path, values in sorted(observed.items())
      if path in expected
    },
  }


def main() -> None:
  config = load_sampling_config(
    ontology_path=PROJECT_ROOT / "06_configs" / "ontology.yaml",
    sampling_path=PROJECT_ROOT / "06_configs" / "symbol_sampling.yaml",
  )
  assert len(config.class_keys) == 16, f"Expected 16 classes, found {len(config.class_keys)}."

  print(f"Sampling {SAMPLE_COUNT} instances per class...")
  rng = np.random.default_rng(BASE_SEED)
  reports = []
  for class_group, class_name in config.class_keys:
    reports.append(_validate_class(config, class_group, class_name, rng))
  REPORT_PATH.write_text(
    json.dumps(
      {
        "sample_count_per_class": SAMPLE_COUNT,
        "base_seed": BASE_SEED,
        "classes": reports,
      },
      indent=2,
      default=str,
    ) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote report to {REPORT_PATH}")
  print("Distribution sanity checks passed for all 16 classes.")


if __name__ == "__main__":
  main()
