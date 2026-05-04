"""Evaluate human-edited OracleMem natural examples as a finite package.

The JSONL examples in ``llm_memory_validation/human_style_examples`` already
contain candidate memories, costs, evidence units, and coverage edges. This
script converts them into one OracleMem instance and evaluates standard writer
policies against an exact package optimum.

The exact solver here is a dynamic program for this artifact: every example is
one multiple-choice group and evidence-unit ids are namespaced by example, so
candidate singleton values are additive across groups.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oraclemem.evaluate import (
    CandidateMemory,
    DEFAULT_ESTIMATOR_MODEL,
    DEFAULT_ESTIMATOR_PROFILE,
    EstimatedUtilityModel,
    OracleMemInstance,
    SelectionResult,
    TOMBSTONE_TYPES,
    feasibility_report,
    greedy_select,
    objective_value,
    policy_metadata_for_method,
    representation_mix,
    select_method,
    selected_candidates,
    total_cost,
    update_metrics,
    write_benchmark_outputs,
)


DEFAULT_METHODS = (
    "opt",
    "oracle_gvt",
    "estimated_gvt",
    "memgpt_tiered",
    "amem_graph",
    "amac_admission",
    "mem0_extract",
    "density_only",
    "greedy",
    "fact_only",
    "summary_only",
    "recency_raw",
    "no_tombstone_opt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate human-edited OracleMem natural examples."
    )
    parser.add_argument(
        "--examples-jsonl",
        default="llm_memory_validation/human_style_examples/examples_100.jsonl",
        help="Canonical human-style examples JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        default="llm_memory_validation/human_style_examples/eval_package_100",
        help="Output directory for raw_results.jsonl and summaries.",
    )
    parser.add_argument(
        "--budgets",
        default="150,300,600,1000",
        help="Comma or space separated integer storage budgets.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma or space separated methods.",
    )
    return parser.parse_args()


def parse_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in value.replace(",", " ").split() if token)


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def _unit_key(example_id: str, unit_id: str) -> str:
    return f"{example_id}::{unit_id}"


def build_instance(rows: Sequence[Mapping[str, Any]]) -> OracleMemInstance:
    candidates: list[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    current_units: list[str] = []
    invalidation_units: list[str] = []
    stale_units: list[str] = []

    for time_index, row in enumerate(rows):
        example_id = str(row["example_id"])
        required = {
            _unit_key(example_id, str(unit_id))
            for unit_id in row.get("required_unit_ids_for_query", [])
        }
        unit_states = {
            _unit_key(example_id, str(unit["unit_id"])): str(unit.get("state", "current"))
            for unit in row.get("evidence_units", [])
        }
        for unit_id in required:
            unit_weights[unit_id] = 1.0
            state = unit_states.get(unit_id, "")
            if any(marker in state for marker in ("update", "current", "query_required", "correction")):
                current_units.append(unit_id)
            if any(marker in state for marker in ("invalidation", "tombstone", "update", "correction")):
                invalidation_units.append(unit_id)
            if any(marker in state for marker in ("stale", "superseded", "expired")):
                stale_units.append(unit_id)

        for candidate in row.get("candidate_memories", []):
            coverage = {
                _unit_key(example_id, str(unit_id)): float(score)
                for unit_id, score in dict(candidate.get("coverage", {})).items()
                if _unit_key(example_id, str(unit_id)) in required
            }
            candidate_id = f"{example_id}::{candidate['candidate_id']}"
            candidates.append(
                CandidateMemory(
                    candidate_id=candidate_id,
                    experience_id=example_id,
                    representation_type=str(candidate.get("representation_type", "unknown")),
                    serialized=str(candidate.get("text", "")),
                    cost=max(0, int(candidate.get("cost_tokens_estimate", 0))),
                    coverage=coverage,
                    time_index=time_index,
                    generator="human_edited",
                    confidence=1.0,
                )
            )

    return OracleMemInstance(
        instance_id="human_audited_seed_0",
        candidates=candidates,
        unit_weights=unit_weights,
        seed=0,
        current_units=tuple(sorted(set(current_units))),
        invalidation_units=tuple(sorted(set(invalidation_units))),
        stale_units=tuple(sorted(set(stale_units))),
    )


def exact_mckp_dp(
    instance: OracleMemInstance,
    budget: int,
    *,
    disallow_types: Iterable[str] = (),
) -> tuple[str, ...]:
    """Exact multiple-choice DP for disjoint-unit human example groups."""

    disallowed = set(disallow_types)
    groups: dict[str, list[CandidateMemory]] = {}
    for candidate in instance.candidates:
        if candidate.representation_type in disallowed:
            continue
        groups.setdefault(candidate.experience_id, []).append(candidate)

    # budget -> (value, ids, cost)
    states: dict[int, tuple[float, tuple[str, ...], int]] = {0: (0.0, (), 0)}
    for experience_id in sorted(groups):
        next_states = dict(states)
        for used_budget, (value, ids, used_cost) in states.items():
            for candidate in groups[experience_id]:
                new_cost = used_budget + candidate.cost
                if new_cost > budget:
                    continue
                candidate_value = objective_value([candidate], instance.unit_weights)
                new_value = value + candidate_value
                new_ids = ids + (candidate.candidate_id,)
                incumbent = next_states.get(new_cost)
                if incumbent is None or (
                    new_value > incumbent[0] + 1e-12
                    or (abs(new_value - incumbent[0]) <= 1e-12 and new_cost < incumbent[2])
                ):
                    next_states[new_cost] = (new_value, new_ids, new_cost)
        states = next_states

    best = max(states.values(), key=lambda item: (item[0], -item[2], item[1]))
    return best[1]


def make_result(
    instance: OracleMemInstance,
    *,
    budget: int,
    method: str,
    selected_ids: Sequence[str],
    optimum_value: float,
    reference_value: float,
    policy_metadata: Optional[Mapping[str, Any]] = None,
) -> SelectionResult:
    selected = selected_candidates(instance.candidates, selected_ids)
    value = objective_value(selected, instance.unit_weights)
    feasibility = feasibility_report(instance.candidates, selected_ids, budget)
    ratio_to_opt = value / optimum_value if optimum_value > 0 else None
    ratio_to_reference = value / reference_value if reference_value > 0 else None
    return SelectionResult(
        instance_id=instance.instance_id,
        seed=instance.seed,
        distribution="human_audited",
        budget=budget,
        method=method,
        selected_candidate_ids=tuple(selected_ids),
        selected_cost=int(feasibility["selected_cost"]),
        objective_value=value,
        denominator_label="exact_human_audited_package_dp",
        ratio_to_opt=ratio_to_opt,
        ratio_to_upper_bound=ratio_to_opt,
        ratio_to_reference=ratio_to_reference,
        optimum_value=optimum_value,
        upper_bound=optimum_value,
        upper_bound_source="exact_mckp_dp_disjoint_units",
        reference_value=reference_value,
        runtime_sec=0.0,
        budget_feasible=bool(feasibility["budget_feasible"]),
        group_feasible=bool(feasibility["group_feasible"]),
        representation_mix=representation_mix(selected),
        update_metrics=update_metrics(instance, selected),
        retrieval_metrics={},
        policy_metadata=dict(policy_metadata or {}),
    )


def evaluate_human_package(
    instance: OracleMemInstance,
    budgets: Sequence[int],
    methods: Sequence[str],
    *,
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> list[SelectionResult]:
    rows: list[SelectionResult] = []
    for budget in budgets:
        exact_ids = exact_mckp_dp(instance, budget)
        optimum_value = objective_value(
            selected_candidates(instance.candidates, exact_ids), instance.unit_weights
        )
        reference_ids = greedy_select(instance.candidates, budget, instance.unit_weights)
        reference_value = objective_value(
            selected_candidates(instance.candidates, reference_ids), instance.unit_weights
        )
        no_tombstone_ids: Optional[tuple[str, ...]] = None
        if "no_tombstone_opt" in methods:
            no_tombstone_ids = exact_mckp_dp(instance, budget, disallow_types=TOMBSTONE_TYPES)

        for method in methods:
            if method == "opt":
                selected_ids = exact_ids
            elif method == "no_tombstone_opt":
                selected_ids = no_tombstone_ids or ()
            else:
                selected_ids = select_method(
                    method,
                    instance.candidates,
                    budget,
                    instance.unit_weights,
                    exact_ids=exact_ids,
                    estimator_model=estimator_model,
                    estimator_profile=estimator_profile,
                    estimator_state=estimator_state,
                )
            rows.append(
                make_result(
                    instance,
                    budget=budget,
                    method=method,
                    selected_ids=selected_ids,
                    optimum_value=optimum_value,
                    reference_value=reference_value,
                    policy_metadata=policy_metadata_for_method(
                        method,
                        estimator_model=estimator_model,
                        estimator_profile=estimator_profile,
                        estimator_state=estimator_state,
                    ),
                )
            )
    return rows


def write_report(out_dir: Path, examples_path: Path, rows: Sequence[Mapping[str, Any]], results: Sequence[SelectionResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_counts: Dict[str, int] = {}
    for row in rows:
        domain = str(row["domain"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    lines = [
        "# Human-Edited/Audited OracleMem Package Evaluation",
        "",
        f"- Source examples: `{examples_path}`",
        f"- Records: {len(rows)}",
        "- Annotation status: human-edited/audited source examples as provided by the authors; no inter-annotator agreement file is included.",
        "- Denominator: exact dynamic-programming optimum over the finite human-audited package.",
        "- Aggregation: the 100 examples are evaluated as one finite package, so package-level ratios are reported rather than cross-annotator agreement statistics.",
        "",
        "## Domain Counts",
        "",
    ]
    for domain, count in sorted(domain_counts.items()):
        lines.append(f"- `{domain}`: {count}")

    lines.extend(["", "## Package Ratio To OPT", ""])
    by_budget_method: Dict[tuple[int, str], list[float]] = {}
    for result in results:
        by_budget_method.setdefault((result.budget, result.method), []).append(result.ratio_to_opt or 0.0)
    for budget in sorted({result.budget for result in results}):
        lines.append(f"### Budget {budget}")
        for method in sorted({result.method for result in results}):
            values = by_budget_method.get((budget, method), [])
            if values:
                mean = sum(values) / len(values)
                lines.append(f"- `{method}`: {mean:.3f}")
        lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    examples_path = Path(args.examples_jsonl)
    rows = load_examples(examples_path)
    instance = build_instance(rows)
    budgets = tuple(int(token) for token in parse_tokens(args.budgets))
    methods = parse_tokens(args.methods)
    results = evaluate_human_package(instance, budgets, methods)
    paths = write_benchmark_outputs(results, args.out_dir)
    write_report(Path(args.out_dir), examples_path, rows, results)
    print(
        json.dumps(
            {
                "examples": len(rows),
                "candidates": len(instance.candidates),
                "required_units": len(instance.unit_weights),
                "budgets": budgets,
                "methods": methods,
                **paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
