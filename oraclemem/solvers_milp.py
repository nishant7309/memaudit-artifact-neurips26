"""Optional MILP backend for exact OracleMem solves.

The default exact-small benchmark uses the pure-stdlib branch-and-bound solver
in :mod:`oraclemem.evaluate`. This module is intentionally optional: it imports
``pulp`` only inside the solver function, so users without a MILP package can
still run all default experiments.
"""

from __future__ import annotations

import time
from typing import Dict

from .evaluate import (
    OracleMemInstance,
    SelectionResult,
    _make_result,
    objective_value,
    ordered_groups,
    selected_candidates,
)


def milp_solve(
    instance: OracleMemInstance,
    budget: int,
    *,
    saturation: str = "cap1",
) -> SelectionResult:
    """Solve the cap-1 OracleMem objective exactly with an optional MILP solver."""

    if saturation != "cap1":
        raise ValueError("MILP backend currently supports only saturation='cap1'")
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "MILP solving requires the optional dependency 'pulp'. "
            "Install it with `pip install pulp` or use `--solver exact_stdlib`."
        ) from exc

    start = time.perf_counter()
    problem = pulp.LpProblem("oraclemem_exact", pulp.LpMaximize)
    x: Dict[str, object] = {
        candidate.candidate_id: pulp.LpVariable(f"x_{candidate.candidate_id}", 0, 1, cat="Binary")
        for candidate in instance.candidates
    }
    y: Dict[str, object] = {
        unit_id: pulp.LpVariable(f"y_{unit_id}", 0, 1, cat="Continuous")
        for unit_id in instance.unit_weights
    }

    problem += pulp.lpSum(instance.unit_weights[unit_id] * y[unit_id] for unit_id in y)
    problem += (
        pulp.lpSum(candidate.cost * x[candidate.candidate_id] for candidate in instance.candidates)
        <= budget
    )
    for group in ordered_groups(instance.candidates):
        problem += pulp.lpSum(x[candidate.candidate_id] for candidate in group) <= 1
    for unit_id in instance.unit_weights:
        problem += y[unit_id] <= pulp.lpSum(
            candidate.coverage.get(unit_id, 0.0) * x[candidate.candidate_id]
            for candidate in instance.candidates
        )

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":  # pragma: no cover - solver/environment dependent
        raise RuntimeError(f"MILP solver did not certify optimality: {pulp.LpStatus[status]}")

    selected_ids = tuple(
        candidate.candidate_id
        for candidate in instance.candidates
        if float(pulp.value(x[candidate.candidate_id]) or 0.0) >= 0.5
    )
    selected = selected_candidates(instance.candidates, selected_ids)
    value = objective_value(selected, instance.unit_weights, saturation=saturation)
    return _make_result(
        instance,
        budget,
        "opt",
        selected_ids,
        selected,
        value,
        optimum_value=value,
        upper_bound=value,
        upper_bound_source="milp_exact",
        reference_value=None,
        runtime_sec=time.perf_counter() - start,
        denominator_label="exact_opt",
    )
