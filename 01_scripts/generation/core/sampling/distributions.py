import ast
from typing import Any, Mapping

import numpy as np


_ALLOWED_FORMULA_NODES = (
  ast.Expression,
  ast.BinOp,
  ast.UnaryOp,
  ast.Add,
  ast.Sub,
  ast.Mult,
  ast.Div,
  ast.Pow,
  ast.Mod,
  ast.USub,
  ast.UAdd,
  ast.Constant,
  ast.Name,
  ast.Load,
  ast.FloorDiv,
)

def sample_distribution(
  spec: Mapping[str, Any], 
  rng: np.random.Generator, 
  values: Mapping[str, Any], 
  decisions: dict[str, Any], 
  path: str
) -> Any:
  """Sample one value from a validated distribution specification.

  Records distribution decisions under ``path``. Conditional distributions use values already present in ``values``; derived distributions evaluate their formulas against the same mapping.

  Args:
    spec: Validated distribution specification.
    rng: Random-number generator used for stochastic distributions.
    values: Previously sampled and derived values available as context.
    decisions: Mutable provenance mapping updated with sampling decisions.
    path: Configuration path used as the decision-record key.

  Returns:
    One sampled or derived value.

  Raises:
    KeyError: If a required distribution field, dependency, or conditional case is missing.
    ValueError: If the distribution type is unsupported or a valid sample cannot be produced.
  """

  dtype = spec.get("type")

  if not isinstance(dtype, str):
    raise ValueError(f"{path}.type must be a distribution name.")
  
  decisions[path] = {"distribution": dtype}

  if dtype == "fixed":
    if "value" not in spec:
      raise ValueError(f"{path}.value is required.")

    value = spec["value"]
    decisions[path]["value"] = value
    return value

  
  if dtype == "categorical":
    probabilities = spec["probabilities"]

    if not isinstance(probabilities, Mapping) or not probabilities:
      raise ValueError(f"{path}.probabilities must be a non-empty mapping.")

    keys = list(probabilities)
    weights = np.asarray(
      list(probabilities.values()),
      dtype=float,
    )

    index = int(
      rng.choice(
        len(keys),
        p=weights,
      )
    )

    value = keys[index]
    decisions[path]["value"] = value

    return value


  if dtype == "truncated_normal":
    mean = float(spec["mean"])
    std = float(spec["std"])
    low = float(spec["min"])
    high = float(spec["max"])

    for _ in range(10_000):
      candidate = float(
        rng.normal(mean, std)
      )

      if low <= candidate <= high:
        decisions[path]["value"] = candidate
        return candidate

    fallback = float(np.clip(mean, low, high))

    decisions[path]["value"] = fallback
    decisions[path]["fallback"] = "clipped_mean"

    return fallback

    
  if dtype == "conditional":
    dependency = spec["depends_on"]

    if dependency not in values:
      raise KeyError(f"{path} depends on unavailable value: {dependency!r}.")

    selected = values[dependency]
    cases = spec["cases"]

    if selected not in cases:
      raise KeyError(f"{path}.cases has no case for {dependency}={selected!r}.")

    decisions[path]["depends_on"] = dependency
    decisions[path]["case"] = selected

    case_spec = cases[selected]

    # supports either:
    #   case: {type: truncated_normal, ...}
    # or:
    #   case: {distribution: {type: truncated_normal, ...}}
    if (
      isinstance(case_spec, Mapping)
      and "distribution" in case_spec
      and "type" not in case_spec
    ):
      case_spec = case_spec["distribution"]

    return sample_distribution(
      spec = case_spec,
      rng = rng,
      values = values,
      decisions = decisions,
      path = f"{path}.cases[{selected}]",
    )
  
    
  if dtype == "mixture":
    components = spec["components"]

    weights = np.asarray([
      component["weight"]
      for component in components
    ], dtype=float)

    index = int(
      rng.choice(len(components), p=weights)
    )

    decisions[path]["component"] = index

    return sample_distribution(
      spec = components[index]["distribution"],
      rng = rng,
      values = values,
      decisions = decisions,
      path = f"{path}.components[{index}]",
    )
  
    
  if dtype == "derived":
    formula = spec["formula"]
    result = evaluate_formula(
      formula = formula,
      values = values,
    )

    decisions[path]["formula"] = formula
    decisions[path]["value"] = result

    return result
    
  raise ValueError(f"Unsupported distribution: {dtype!r}.")


def evaluate_formula(
  formula: str, 
  values: Mapping[str, Any]
) -> Any:
  """Safely evaluate a restricted arithmetic sampling formula.

  Supported expressions contain numeric constants, parameter names, the constant ``pi``, arithmetic operators, unary signs, floor division, modulo, and exponentiation. Function calls, attribute access, indexing, collections, comprehensions, and other Python syntax are rejected.

  Args:
    formula: Arithmetic expression to evaluate.
    values: Names and values available to the expression.

  Returns:
    The evaluated formula result.

  Raises:
    ValueError: If ``formula`` is not a non-empty string or contains unsupported syntax.
    NameError: If the formula references an unknown name.
    TypeError: If an expression combines incompatible values.
    ZeroDivisionError: If the formula divides by zero.  
  """

  if not isinstance(formula, str) or not formula.strip():
    raise ValueError("Sampling formula must be a non-empty string.")

  try:
    tree = ast.parse(formula, mode="eval")
  except SyntaxError as exc:
    raise ValueError(f"Invalid sampling formula: {formula!r}.") from exc

  for node in ast.walk(tree):
    if not isinstance(node, _ALLOWED_FORMULA_NODES):
      raise ValueError(f"Unsupported formula expression: {formula!r}.")

    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
      raise ValueError(f"Formula constants must be numeric: {formula!r}.")

  context = {
    **values,
    "pi": float(np.pi),
  }

  try:
    compiled = compile(
      tree,
      "<sampling formula>",
      "eval",
    )

    return eval(
      compiled,
      {"__builtins__": {}},
      context,
    )
  except NameError:
    raise
  except (TypeError, ZeroDivisionError, OverflowError):
    raise
