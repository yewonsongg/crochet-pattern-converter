import ast
from typing import Any

import numpy as np


def sample_distribution(spec: dict[str, Any], rng: np.random.Generator, values: dict[str, Any], decisions: dict[str, Any], path: str):
  dtype = spec["type"]
  decisions[path] = {"distribution": dtype}
  if dtype == "fixed":
    return spec["value"]
  if dtype == "categorical":
    keys = list(spec["probabilities"])
    value = keys[int(rng.choice(len(keys), p=np.asarray(list(spec["probabilities"].values()), dtype=float)))]
    decisions[path]["value"] = value
    return value
  if dtype == "truncated_normal":
    low, high = float(spec["min"]), float(spec["max"])
    value = float(spec["mean"])
    for _ in range(10000):
      candidate = float(rng.normal(float(spec["mean"]), float(spec["std"])))
      if low <= candidate <= high:
        return candidate
      value = candidate
    return float(np.clip(value, low, high))
  if dtype == "conditional":
    selected = values[spec["depends_on"]]
    decisions[path]["depends_on"] = spec["depends_on"]
    decisions[path]["case"] = selected
    return sample_distribution(spec["cases"][selected], rng, values, decisions, f"{path}.cases[{selected}]")
  if dtype == "mixture":
    index = int(rng.choice(len(spec["components"]), p=np.asarray([x["weight"] for x in spec["components"]], dtype=float)))
    decisions[path]["component"] = index
    return sample_distribution(spec["components"][index]["distribution"], rng, values, decisions, f"{path}.components[{index}]")
  if dtype == "derived":
    return evaluate_formula(spec["formula"], values)
  raise ValueError(f"Unsupported distribution: {dtype}")


def evaluate_formula(formula: str, values: dict[str, Any]):
  tree = ast.parse(formula, mode="eval")
  allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load, ast.FloorDiv)
  if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
    raise ValueError(f"Unsupported formula expression: {formula}")
  return eval(compile(tree, "<sampling formula>", "eval"), {"__builtins__": {}}, {**values, "pi": float(np.pi)})
