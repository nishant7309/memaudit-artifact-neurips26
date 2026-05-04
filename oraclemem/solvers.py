"""Exact and reference solvers for OracleMem memory writing."""

from __future__ import annotations

from dataclasses import replace

from .objective import (
    candidate_cost,
    candidate_coverage,
    candidate_id,
    candidate_maps,
    coverage_utility,
    experience_id,
    marginal_gain,
    selected_cost,
    unit_weights,
)
from .schema import CandidateMemory, Instance, SolverResult


def is_feasible(candidates: list[CandidateMemory], budget: int) -> bool:
  if selected_cost(candidates) > budget:
    return False
  groups = [experience_id(candidate) for candidate in candidates]
  return len(groups) == len(set(groups))


def _result(method: str, budget: int, selected: list[CandidateMemory], weights: dict[str, float]) -> SolverResult:
  return SolverResult(
      method=method,
      budget=budget,
      selected_ids=tuple(candidate_id(candidate) for candidate in selected),
      utility=coverage_utility(selected, weights),
      cost=selected_cost(selected),
  )


def exact_bruteforce(instance: Instance, budget: int, method: str = "exact_opt") -> SolverResult:
  """Exact search by enumerating one representation or discard per experience."""
  weights = unit_weights(instance)
  _, by_exp = candidate_maps(instance)
  exp_ids = [experience.experience_id for experience in instance.experiences]
  best: list[CandidateMemory] = []
  best_utility = -1.0

  def dfs(index: int, selected: list[CandidateMemory], cost: int) -> None:
    nonlocal best, best_utility
    if cost > budget:
      return
    if index == len(exp_ids):
      utility = coverage_utility(selected, weights)
      if utility > best_utility:
        best_utility = utility
        best = list(selected)
      return
    exp_id = exp_ids[index]
    dfs(index + 1, selected, cost)
    for candidate in by_exp.get(exp_id, []):
      candidate_storage = candidate_cost(candidate)
      if cost + candidate_storage <= budget:
        selected.append(candidate)
        dfs(index + 1, selected, cost + candidate_storage)
        selected.pop()

  dfs(0, [], 0)
  return _result(method, budget, best, weights)


def exact_branch_and_bound(instance: Instance, budget: int) -> SolverResult:
  """Exact DFS with a simple admissible remaining-unit upper bound."""
  weights = unit_weights(instance)
  _, by_exp = candidate_maps(instance)
  exp_ids = [experience.experience_id for experience in instance.experiences]
  suffix_units: list[set[str]] = [set() for _ in range(len(exp_ids) + 1)]
  for index in range(len(exp_ids) - 1, -1, -1):
    units = set(suffix_units[index + 1])
    for candidate in by_exp.get(exp_ids[index], []):
      units.update(unit for unit, value in candidate_coverage(candidate).items() if value > 0)
    suffix_units[index] = units

  best: list[CandidateMemory] = []
  best_utility = -1.0

  def dfs(index: int, selected: list[CandidateMemory], cost: int) -> None:
    nonlocal best, best_utility
    if cost > budget:
      return
    current_utility = coverage_utility(selected, weights)
    optimistic = current_utility + sum(weights.get(unit, 0.0) for unit in suffix_units[index])
    if optimistic + 1e-12 < best_utility:
      return
    if index == len(exp_ids):
      if current_utility > best_utility:
        best_utility = current_utility
        best = list(selected)
      return
    exp_id = exp_ids[index]
    choices = sorted(
        by_exp.get(exp_id, []),
        key=lambda candidate: (
            sum(weights.get(unit, 0.0) for unit in candidate_coverage(candidate)) /
            max(candidate_cost(candidate), 1)
        ),
        reverse=True,
    )
    for candidate in choices:
      candidate_storage = candidate_cost(candidate)
      if cost + candidate_storage <= budget:
        selected.append(candidate)
        dfs(index + 1, selected, cost + candidate_storage)
        selected.pop()
    dfs(index + 1, selected, cost)

  dfs(0, [], 0)
  return _result("exact_branch_bound", budget, best, weights)


def attach_ratio(result: SolverResult, optimum: SolverResult, basis: str = "opt_exact") -> SolverResult:
  ratio = result.utility / optimum.utility if optimum.utility > 0 else 1.0
  return replace(result, ratio=ratio, ratio_basis=basis)


def greedy_reference(instance: Instance, budget: int, method: str = "greedy_reference") -> SolverResult:
  """Offline greedy reference by marginal utility density.

  This is a reference denominator/baseline, not an exact optimum. It enforces
  both the storage budget and the one-candidate-per-experience constraint.
  """
  weights = unit_weights(instance)
  remaining = list(instance.candidates)
  selected: list[CandidateMemory] = []
  used_exp: set[str] = set()
  cost = 0
  while True:
    best = None
    best_key = (0.0, 0.0, "")
    for candidate in remaining:
      exp_id = experience_id(candidate)
      candidate_storage = candidate_cost(candidate)
      if exp_id in used_exp or cost + candidate_storage > budget:
        continue
      gain = marginal_gain(selected, candidate, weights)
      if gain <= 0:
        continue
      density = gain / max(candidate_storage, 1)
      key = (density, gain, candidate_id(candidate))
      if key > best_key:
        best_key = key
        best = candidate
    if best is None:
      break
    selected.append(best)
    used_exp.add(experience_id(best))
    cost += candidate_cost(best)
    remaining.remove(best)
  return _result(method, budget, selected, weights)


# Canonical API aliases.
brute_force_exact = exact_bruteforce
branch_and_bound_exact = exact_branch_and_bound
solve_bruteforce_exact = exact_bruteforce
solve_exact_bruteforce = exact_bruteforce
solve_branch_and_bound_exact = exact_branch_and_bound
solve_exact_branch_and_bound = exact_branch_and_bound
solve_greedy_reference = greedy_reference
