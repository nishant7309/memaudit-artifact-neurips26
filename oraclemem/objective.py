"""Semantic coverage objective for OracleMem.

The benchmark utility is

  F(X) = sum_r w_r h(sum_{u in X} a_ur),  h(z)=min(1,z).

The helpers below accept the local ``schema.py`` dataclasses, but also work
with plain dictionaries or objects that expose the same field names.  This
keeps the objective usable before a larger package schema is finalized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable

try:
  from .schema import CandidateMemory, Instance
except Exception:  # pragma: no cover - used only if schema.py is absent.
  CandidateMemory = Any  # type: ignore
  Instance = Any  # type: ignore


def _read(obj: Any, name: str, default: Any = None) -> Any:
  if isinstance(obj, Mapping):
    return obj.get(name, default)
  return getattr(obj, name, default)


def _read_first(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
  for name in names:
    value = _read(obj, name, None)
    if value is not None:
      return value
  return default


def h_min_one(z: float) -> float:
  """OracleMem's default saturation function."""
  if z <= 0:
    return 0.0
  return 1.0 if z >= 1.0 else float(z)


def candidate_id(candidate: CandidateMemory) -> str:
  return str(_read_first(candidate, ("candidate_id", "memory_id", "id"), repr(candidate)))


def experience_id(candidate: CandidateMemory) -> str:
  value = _read_first(candidate, ("experience_id", "exp_id", "group_id", "item_id"), None)
  return str(value) if value is not None else candidate_id(candidate)


def representation_type(candidate: CandidateMemory) -> str:
  return str(_read_first(candidate, ("representation", "representation_type", "type", "tier"), ""))


def is_discard_candidate(candidate: CandidateMemory) -> bool:
  return representation_type(candidate).strip().lower() in {"discard", "skip", "none", "empty"}


def candidate_cost(candidate: CandidateMemory) -> int:
  raw = _read_first(candidate, ("cost", "total_cost", "storage_tokens", "tokens", "weight"), 0)
  if isinstance(raw, Mapping):
    raw = _read_first(raw, ("total", "total_tokens", "storage_tokens", "tokens", "weight"), 0)
  cost = int(raw)
  if cost < 0:
    raise ValueError(f"{candidate_id(candidate)} has negative cost {cost}")
  return cost


def candidate_coverage(candidate: CandidateMemory) -> dict[str, float]:
  raw = _read_first(candidate, ("coverage", "covers", "coverage_vector"), {})
  coverage: dict[str, float] = {}
  if isinstance(raw, Mapping):
    items = raw.items()
  else:
    items = []
    for entry in raw:
      if isinstance(entry, Mapping):
        unit = _read_first(entry, ("unit_id", "semantic_unit_id", "unit"), None)
        value = _read_first(entry, ("fidelity", "coverage", "value", "score"), 1.0)
        if unit is not None:
          items.append((unit, value))
      elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
        items.append((entry[0], entry[1]))
  for unit_id, value in items:
    fidelity = float(value)
    if fidelity < 0:
      raise ValueError(f"{candidate_id(candidate)} has negative coverage")
    if fidelity > 0:
      coverage[str(unit_id)] = coverage.get(str(unit_id), 0.0) + fidelity
  return coverage


def unit_weights(instance: Instance) -> dict[str, float]:
  """Weight units by held-out query demand."""
  counts: Counter[str] = Counter()
  for query in instance.queries:
    for unit_id in query.required_unit_ids:
      counts[unit_id] += 1
  return {unit_id: float(count) for unit_id, count in counts.items()}


def selected_cost(candidates: Iterable[CandidateMemory]) -> int:
  return sum(candidate_cost(candidate) for candidate in candidates)


def coverage_utility(
    candidates: Iterable[CandidateMemory],
    weights: dict[str, float],
) -> float:
  """Concave coverage utility with h(z)=min(1,z)."""
  coverage: dict[str, float] = {}
  for candidate in candidates:
    for unit_id, value in candidate_coverage(candidate).items():
      coverage[unit_id] = coverage.get(unit_id, 0.0) + float(value)
  return sum(weights.get(unit_id, 0.0) * h_min_one(value) for unit_id, value in coverage.items())


def marginal_gain(
    selected: Iterable[CandidateMemory],
    candidate: CandidateMemory,
    weights: dict[str, float],
) -> float:
  selected_tuple = tuple(selected)
  return coverage_utility((*selected_tuple, candidate), weights) - coverage_utility(selected_tuple, weights)


def candidate_maps(instance: Instance) -> tuple[dict[str, CandidateMemory], dict[str, list[CandidateMemory]]]:
  by_id = {candidate_id(candidate): candidate for candidate in instance.candidates}
  by_exp: dict[str, list[CandidateMemory]] = {}
  for candidate in instance.candidates:
    if is_discard_candidate(candidate):
      continue
    by_exp.setdefault(experience_id(candidate), []).append(candidate)
  return by_id, by_exp


class SemanticCoverageObjective:
  """Reusable object wrapper around ``coverage_utility``.

  If ``weights`` is omitted and an instance is provided, weights are derived
  from held-out query demand.  If candidates are provided without query
  weights, every observed unit receives weight 1.
  """

  def __init__(
      self,
      candidates: Iterable[CandidateMemory] | None = None,
      weights: dict[str, float] | None = None,
      instance: Instance | None = None,
  ) -> None:
    self.candidates = tuple(candidates or (getattr(instance, "candidates", ()) if instance is not None else ()))
    if weights is not None:
      self.weights = dict(weights)
    elif instance is not None:
      self.weights = unit_weights(instance)
    else:
      inferred: dict[str, float] = {}
      for candidate in self.candidates:
        for unit_id in candidate_coverage(candidate):
          inferred.setdefault(unit_id, 1.0)
      self.weights = inferred

  def value(self, selected: Iterable[CandidateMemory]) -> float:
    return coverage_utility(selected, self.weights)

  def marginal_gain(self, selected: Iterable[CandidateMemory], candidate: CandidateMemory) -> float:
    return marginal_gain(selected, candidate, self.weights)

  def singleton_value(self, candidate: CandidateMemory) -> float:
    return self.marginal_gain((), candidate)
