"""Online and reference memory-writing algorithms for OracleMem."""

from __future__ import annotations

from .objective import candidate_maps, coverage_utility, marginal_gain, selected_cost, unit_weights
from .schema import CandidateMemory, Instance, SolverResult
from .solvers import _result


def greedy_reference(instance: Instance, budget: int) -> SolverResult:
  weights = unit_weights(instance)
  remaining = list(instance.candidates)
  selected: list[CandidateMemory] = []
  used_exp: set[str] = set()
  while True:
    best = None
    best_density = 0.0
    for candidate in remaining:
      if candidate.experience_id in used_exp:
        continue
      if selected_cost(selected) + candidate.cost > budget:
        continue
      gain = marginal_gain(selected, candidate, weights)
      density = gain / max(candidate.cost, 1)
      if density > best_density:
        best_density = density
        best = candidate
    if best is None:
      break
    selected.append(best)
    used_exp.add(best.experience_id)
  return _result("greedy_reference", budget, selected, weights)


def recency_raw(instance: Instance, budget: int) -> SolverResult:
  weights = unit_weights(instance)
  _, by_exp = candidate_maps(instance)
  selected: list[CandidateMemory] = []
  cost = 0
  for experience in reversed(instance.experiences):
    raw = [c for c in by_exp.get(experience.experience_id, []) if c.representation == "raw"]
    if not raw:
      continue
    candidate = raw[0]
    if cost + candidate.cost <= budget:
      selected.append(candidate)
      cost += candidate.cost
  selected.reverse()
  return _result("recency_raw", budget, selected, weights)


def no_tombstone_greedy(instance: Instance, budget: int) -> SolverResult:
  filtered = tuple(
      candidate for candidate in instance.candidates
      if candidate.representation not in {"tombstone", "compound_update"}
  )
  filtered_instance = type(instance)(
      instance_id=instance.instance_id,
      seed=instance.seed,
      units=instance.units,
      experiences=instance.experiences,
      candidates=filtered,
      queries=instance.queries,
      metadata=instance.metadata,
  )
  result = greedy_reference(filtered_instance, budget)
  return SolverResult("no_tombstone_greedy", budget, result.selected_ids, result.utility, result.cost)


def density_only(instance: Instance, budget: int) -> SolverResult:
  weights = unit_weights(instance)
  _, by_exp = candidate_maps(instance)
  selected: list[CandidateMemory] = []
  cost = 0
  for experience in instance.experiences:
    best = None
    best_density = 0.0
    for candidate in by_exp.get(experience.experience_id, []):
      if cost + candidate.cost > budget:
        continue
      gain = marginal_gain(selected, candidate, weights)
      density = gain / max(candidate.cost, 1)
      if density > best_density:
        best = candidate
        best_density = density
    if best is not None:
      selected.append(best)
      cost += best.cost
  return _result("density_only", budget, selected, weights)


def grouped_value_threshold(instance: Instance, budget: int, threshold: float) -> SolverResult:
  weights = unit_weights(instance)
  _, by_exp = candidate_maps(instance)
  selected: list[CandidateMemory] = []
  cost = 0
  for experience in instance.experiences:
    admissible: list[tuple[float, CandidateMemory]] = []
    for candidate in by_exp.get(experience.experience_id, []):
      if cost + candidate.cost > budget:
        continue
      gain = marginal_gain(selected, candidate, weights)
      density = gain / max(candidate.cost, 1)
      if density >= threshold and gain > 0:
        admissible.append((gain, candidate))
    if admissible:
      _, chosen = max(admissible, key=lambda item: (item[0], -item[1].cost))
      selected.append(chosen)
      cost += chosen.cost
  return _result(f"gvt_threshold_{threshold:.4g}", budget, selected, weights)


def grouped_value_threshold_grid(instance: Instance, budget: int) -> SolverResult:
  weights = unit_weights(instance)
  densities = []
  for candidate in instance.candidates:
    gain = coverage_utility([candidate], weights)
    if gain > 0:
      densities.append(gain / max(candidate.cost, 1))
  if not densities:
    return SolverResult("gvt_grid", budget, tuple(), 0.0, 0)
  thresholds = sorted(set([0.0, *densities]))
  best = None
  for threshold in thresholds:
    result = grouped_value_threshold(instance, budget, threshold)
    if best is None or result.utility > best.utility:
      best = result
  assert best is not None
  return SolverResult("gvt_grid", budget, best.selected_ids, best.utility, best.cost)
