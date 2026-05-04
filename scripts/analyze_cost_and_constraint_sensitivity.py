"""Deterministic cost-model and group-capacity sensitivity checks.

The paper's default storage unit is a normalized word-equivalent cost.  This
script reuses cached MemAudit natural/adjudicated packages and exported-system
coverage rows to answer two review-facing questions without making API calls:

* How much do exact package optima change if one experience can keep two
  representations instead of one?
* Do exported-system diagnostics at B=100 qualitatively change under an
  alternative per-record-overhead + serialized-byte cost rule?
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import (  # noqa: E402
    CandidateMemory,
    OracleMemInstance,
    objective_value,
    ordered_groups,
    solve_exact,
    value_from_totals,
)
from llm_memory_validation.run_mem0_natural_baseline import (  # noqa: E402
    load_package,
    package_instance,
    read_jsonl,
    resolved_queries,
    select_oracle_density_pruned,
    select_recency_pruned,
)
from llm_memory_validation.score_mem0_written_stores import (  # noqa: E402
    attach_salience,
    select_salience_pruned,
)
from llm_memory_validation.gemini_natural_oraclemem import word_count  # noqa: E402
from llm_memory_validation.run_actual_amem_natural_baseline import (  # noqa: E402
    select_native_retrieval_pruned,
)


PACKAGE_DIR = ROOT / "llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package"
OUT_DIR = ROOT / "llm_memory_validation/natural_adjudicated_100_gemini_flash/sensitivity"
BUDGETS = (30, 60, 100)


def mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def read_calls(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def byte_overhead_cost(text: str) -> int:
    """Alternative cost: fixed record overhead plus serialized-byte payload."""

    return max(1, 8 + math.ceil(len(str(text).encode("utf-8")) / 24))


def default_cost(text: str) -> int:
    return max(1, word_count(str(text)))


def clone_candidate(candidate: CandidateMemory, cost_fn: Callable[[str], int]) -> CandidateMemory:
    return CandidateMemory(
        candidate_id=candidate.candidate_id,
        experience_id=candidate.experience_id,
        representation_type=candidate.representation_type,
        serialized=candidate.serialized,
        cost=cost_fn(candidate.serialized),
        coverage=dict(candidate.coverage),
        time_index=candidate.time_index,
        generator=candidate.generator,
        confidence=candidate.confidence,
        estimated_value=candidate.estimated_value,
        estimated_coverage=dict(candidate.estimated_coverage),
        estimator_model=candidate.estimator_model,
    )


def clone_instance(instance: OracleMemInstance, cost_fn: Callable[[str], int]) -> OracleMemInstance:
    return OracleMemInstance(
        instance_id=instance.instance_id,
        candidates=[clone_candidate(candidate, cost_fn) for candidate in instance.candidates],
        unit_weights=dict(instance.unit_weights),
        seed=instance.seed,
        current_units=instance.current_units,
        invalidation_units=instance.invalidation_units,
        stale_units=instance.stale_units,
    )


def add_totals(totals: Mapping[str, float], candidates: Sequence[CandidateMemory]) -> dict[str, float]:
    next_totals = dict(totals)
    for candidate in candidates:
        for unit_id, value in candidate.coverage.items():
            next_totals[unit_id] = next_totals.get(unit_id, 0.0) + value
    return next_totals


def solve_group_cap(instance: OracleMemInstance, budget: int, *, group_cap: int) -> tuple[float, int]:
    """Exact branch-and-bound where each group may choose up to group_cap items."""

    if group_cap == 1:
        result = solve_exact(instance, budget, solver="exact_stdlib")
        return float(result.objective_value), int(result.selected_cost)

    groups = ordered_groups(instance.candidates)
    options: list[list[tuple[CandidateMemory, ...]]] = []
    for group in groups:
        group_options: list[tuple[CandidateMemory, ...]] = [()]
        for size in range(1, min(group_cap, len(group)) + 1):
            for combo in itertools.combinations(group, size):
                if sum(candidate.cost for candidate in combo) <= budget:
                    group_options.append(combo)
        group_options.sort(
            key=lambda combo: (
                -objective_value(list(combo), instance.unit_weights) / max(1, sum(c.cost for c in combo)),
                sum(c.cost for c in combo),
                tuple(c.candidate_id for c in combo),
            )
        )
        options.append(group_options)

    suffix: list[dict[str, float]] = [defaultdict(float) for _ in range(len(groups) + 1)]
    for index in range(len(groups) - 1, -1, -1):
        suffix[index] = defaultdict(float, suffix[index + 1])
        for unit_id in instance.unit_weights:
            best = 0.0
            for combo in options[index]:
                best = max(best, sum(candidate.coverage.get(unit_id, 0.0) for candidate in combo))
            suffix[index][unit_id] += best

    best_value = 0.0
    best_cost = 0

    def optimistic_value(index: int, totals: Mapping[str, float]) -> float:
        optimistic = dict(totals)
        for unit_id, addend in suffix[index].items():
            optimistic[unit_id] = optimistic.get(unit_id, 0.0) + addend
        return value_from_totals(optimistic, instance.unit_weights)

    def recurse(index: int, used_cost: int, totals: Mapping[str, float]) -> None:
        nonlocal best_value, best_cost
        if optimistic_value(index, totals) + 1e-12 < best_value:
            return
        if index == len(options):
            value = value_from_totals(totals, instance.unit_weights)
            if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and used_cost < best_cost):
                best_value = value
                best_cost = used_cost
            return
        for combo in options[index]:
            combo_cost = sum(candidate.cost for candidate in combo)
            if used_cost + combo_cost > budget:
                continue
            recurse(index + 1, used_cost + combo_cost, add_totals(totals, combo))

    recurse(0, 0, {})
    return best_value, best_cost


def candidate_coverage_from_calls(
    calls_path: Path, *, filters: Mapping[str, str] | None = None
) -> dict[str, dict[str, dict[str, float]]]:
    by_instance: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in read_calls(calls_path):
        if filters and any(str(row.get(key, "")) != value for key, value in filters.items()):
            continue
        instance_id = str(row.get("instance_id", ""))
        for edge in row.get("coverage_edges", []) or []:
            memory_id = str(edge.get("memory_id", ""))
            unit_id = str(edge.get("unit_id", ""))
            value = max(0.0, min(1.0, float(edge.get("coverage", edge.get("fidelity", 0.0)) or 0.0)))
            if instance_id and memory_id and unit_id and value > 0:
                by_instance[instance_id][memory_id][unit_id] = max(
                    value, by_instance[instance_id][memory_id].get(unit_id, 0.0)
                )
    return {inst: dict(memories) for inst, memories in by_instance.items()}


def salience_from_calls(calls_path: Path) -> dict[str, dict[str, float]]:
    by_instance: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_calls(calls_path):
        instance_id = str(row.get("instance_id", ""))
        scores = row.get("scores", {}) or {}
        for memory_id, payload in scores.items():
            if instance_id:
                by_instance[instance_id][str(memory_id)] = max(
                    0.0, min(1.0, float((payload or {}).get("salience", 0.0) or 0.0))
                )
    return {inst: dict(scores) for inst, scores in by_instance.items()}


def stores_by_instance(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("instance_id")): row for row in read_jsonl(path) if row.get("instance_id")}


def external_candidates(
    *,
    instance_id: str,
    memories: Sequence[Mapping[str, Any]],
    coverage_by_instance: Mapping[str, Mapping[str, Mapping[str, float]]],
    token: str,
    cost_fn: Callable[[str], int],
    text_key: str = "text",
    salience: Mapping[str, float] | None = None,
) -> list[CandidateMemory]:
    candidates: list[CandidateMemory] = []
    coverage_by_memory = coverage_by_instance.get(instance_id, {})
    for index, memory in enumerate(memories):
        memory_id = str(memory.get("memory_id", f"{token}-{index}"))
        text = str(memory.get(text_key) or memory.get("text") or "")
        candidates.append(
            CandidateMemory(
                candidate_id=f"{instance_id}::{token}::{index:04d}",
                experience_id=f"{instance_id}::{token}::{index:04d}",
                representation_type=token,
                serialized=text,
                cost=cost_fn(text),
                coverage=coverage_by_memory.get(memory_id, {}),
                time_index=int(memory.get("memory_index", index) or index),
                generator=token,
                confidence=float((salience or {}).get(memory_id, 1.0)),
                estimated_value=float((salience or {}).get(memory_id, 0.0)),
                estimator_model="cached_salience" if salience is not None else "",
            )
        )
    return candidates


def union_instance(package: OracleMemInstance, external: Sequence[CandidateMemory], tag: str) -> OracleMemInstance:
    return OracleMemInstance(
        instance_id=f"{package.instance_id}::plus_{tag}",
        candidates=tuple(package.candidates) + tuple(external),
        unit_weights=package.unit_weights,
        current_units=package.current_units,
        invalidation_units=package.invalidation_units,
        stale_units=package.stale_units,
    )


def ratio(value: float, denominator: float) -> float | None:
    return value / denominator if denominator > 0 else None


def selected_ratio(
    selected: Sequence[CandidateMemory],
    package: OracleMemInstance,
    denominator: float,
) -> float | None:
    return ratio(objective_value(selected, package.unit_weights), denominator)


def main() -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_package(PACKAGE_DIR)
    queries = resolved_queries(data, limit=None)

    package_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        k1_values: list[float] = []
        k2_values: list[float] = []
        k1_alt_values: list[float] = []
        for query in queries:
            base = package_instance(data, query)
            base = clone_instance(base, default_cost)
            alt = clone_instance(base, byte_overhead_cost)
            k1, _ = solve_group_cap(base, budget, group_cap=1)
            k2, _ = solve_group_cap(base, budget, group_cap=2)
            k1_alt, _ = solve_group_cap(alt, budget, group_cap=1)
            k1_values.append(k1)
            k2_values.append(k2)
            k1_alt_values.append(k1_alt)
        package_rows.append(
            {
                "budget": budget,
                "n": len(queries),
                "mean_opt_k1_default": mean(k1_values),
                "mean_opt_k2_default": mean(k2_values),
                "mean_k2_over_k1_default": mean(
                    [(b / a if a > 0 else None) for a, b in zip(k1_values, k2_values)]
                ),
                "mean_opt_k1_byte_overhead": mean(k1_alt_values),
                "mean_alt_over_default_k1": mean(
                    [(b / a if a > 0 else None) for a, b in zip(k1_values, k1_alt_values)]
                ),
            }
        )

    mem0_stores = stores_by_instance(ROOT / "llm_memory_validation/mem0_natural200_actual/written_stores.jsonl")
    mem0_cov = candidate_coverage_from_calls(
        ROOT / "llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash/coverage_scoring_calls.jsonl"
    )
    mem0_sal = salience_from_calls(
        ROOT / "llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash/salience_scoring_calls.jsonl"
    )

    letta_dir = ROOT / "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87"
    letta_stores = stores_by_instance(letta_dir / "written_stores.jsonl")
    letta_cov = candidate_coverage_from_calls(letta_dir / "coverage_scoring_calls.jsonl")
    letta_sal = salience_from_calls(letta_dir / "salience_scoring_calls.jsonl")

    amem_dir = ROOT / "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87"
    amem_stores = stores_by_instance(amem_dir / "written_stores.jsonl")
    amem_full_cov = candidate_coverage_from_calls(
        amem_dir / "coverage_scoring_calls.jsonl", filters={"memory_view": "full"}
    )
    amem_metadata_cov = candidate_coverage_from_calls(
        amem_dir / "coverage_scoring_calls.jsonl", filters={"memory_view": "metadata"}
    )

    # Exported-system sensitivity is computed against the package-candidate
    # denominator under each cost rule.  Recomputing every finite union optimum
    # under every alternative cost rule is much slower and does not change the
    # narrow diagnostic needed here: whether budget pruning/ranking is mostly a
    # word-count artifact.  The main paper still uses exact union denominators.
    system_rows: list[dict[str, Any]] = []
    for cost_label, cost_fn in [("word", default_cost), ("byte_overhead", byte_overhead_cost)]:
        values: dict[str, list[float | None]] = defaultdict(list)
        full_min_costs: list[int] = []
        full_min_costs_alt: list[int] = []
        for query in queries:
            instance_id = str(query["query_id"])
            package = clone_instance(package_instance(data, query), cost_fn)

            mem0_store = mem0_stores.get(instance_id, {})
            mem0_memories = mem0_store.get("memories", []) or []
            mem0_candidates = external_candidates(
                instance_id=instance_id,
                memories=mem0_memories,
                coverage_by_instance=mem0_cov,
                token="mem0",
                cost_fn=cost_fn,
                salience=mem0_sal.get(instance_id, {}),
            )
            package_den = float(solve_exact(package, 100, solver="exact_stdlib").objective_value)
            values["mem0_salience"].append(
                selected_ratio(select_salience_pruned(mem0_candidates, 100), package, package_den)
            )
            values["mem0_upper"].append(
                selected_ratio(select_oracle_density_pruned(mem0_candidates, 100, package.unit_weights), package, package_den)
            )

            letta_store = letta_stores.get(instance_id, {})
            letta_memories = letta_store.get("memories", []) or []
            letta_candidates = external_candidates(
                instance_id=instance_id,
                memories=letta_memories,
                coverage_by_instance=letta_cov,
                token="letta",
                cost_fn=cost_fn,
                salience=letta_sal.get(instance_id, {}),
            )
            values["letta_salience"].append(
                selected_ratio(select_salience_pruned(letta_candidates, 100), package, package_den)
            )
            values["letta_upper"].append(
                selected_ratio(select_oracle_density_pruned(letta_candidates, 100, package.unit_weights), package, package_den)
            )

            amem_store = amem_stores.get(instance_id, {})
            amem_memories = amem_store.get("memories", []) or []
            full_memories = [{**memory, "text": str(memory.get("full_text") or memory.get("text") or "")} for memory in amem_memories]
            metadata_memories = [
                {**memory, "text": str(memory.get("metadata_text") or "")}
                for memory in amem_memories
                if str(memory.get("metadata_text") or "").strip()
            ]
            full_candidates = external_candidates(
                instance_id=instance_id,
                memories=full_memories,
                coverage_by_instance=amem_full_cov,
                token="amem_full",
                cost_fn=cost_fn,
            )
            metadata_candidates = external_candidates(
                instance_id=instance_id,
                memories=metadata_memories,
                coverage_by_instance=amem_metadata_cov,
                token="amem_metadata",
                cost_fn=cost_fn,
            )
            if full_candidates:
                full_min_costs.append(min(candidate.cost for candidate in full_candidates))
                full_min_costs_alt.append(min(byte_overhead_cost(candidate.serialized) for candidate in full_candidates))
            native_order = [int(index) for index in amem_store.get("native_order", []) or []]
            values["amem_metadata_recency"].append(
                selected_ratio(select_recency_pruned(metadata_candidates, 100), package, package_den)
            )
            values["amem_full_native"].append(
                selected_ratio(select_native_retrieval_pruned(full_candidates, native_order, 100), package, package_den)
            )

        for method, ratios in sorted(values.items()):
            row = {
                "budget": 100,
                "cost_rule": cost_label,
                "method": method,
                "mean_ratio_to_package_opt": mean(ratios),
                "n": len([value for value in ratios if value is not None]),
            }
            if method == "amem_full_native":
                row["mean_min_full_note_cost_word"] = mean(full_min_costs)
                row["mean_min_full_note_cost_byte_overhead"] = mean(full_min_costs_alt)
            system_rows.append(row)

    summary = {
        "package_dir": str(PACKAGE_DIR.relative_to(ROOT)),
        "n": len(queries),
        "cost_rules": {
            "word": "max(1, whitespace word count)",
            "byte_overhead": "8 + ceil(len(serialized_utf8_bytes) / 24)",
        },
        "package_group_capacity": package_rows,
        "exported_system_b100": system_rows,
        "runtime_sec": time.perf_counter() - start,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# MemAudit Cost and Constraint Sensitivity",
        "",
        "All rows are deterministic recomputations from cached package/export artifacts; no API calls are made.",
        "",
        "## Package group-capacity sensitivity",
        "",
        "| B | OPT k=1 | OPT k=2 | k=2/k=1 | byte-overhead k=1 | byte/default |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in package_rows:
        lines.append(
            "| {budget} | {mean_opt_k1_default:.3f} | {mean_opt_k2_default:.3f} | "
            "{mean_k2_over_k1_default:.3f} | {mean_opt_k1_byte_overhead:.3f} | "
            "{mean_alt_over_default_k1:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Exported-system sensitivity at B=100",
            "",
            "| Cost rule | Method | Mean ratio to package OPT | N |",
            "|---|---|---:|---:|",
        ]
    )
    for row in system_rows:
        lines.append(
            f"| {row['cost_rule']} | {row['method']} | {float(row['mean_ratio_to_package_opt'] or 0.0):.3f} | {row['n']} |"
        )
    amem_full = [row for row in system_rows if row["method"] == "amem_full_native" and row["cost_rule"] == "word"]
    if amem_full:
        row = amem_full[0]
        lines.extend(
            [
                "",
                "A-Mem full native rows are zero at B=100 because the full serialized notes are far above the small budgets.",
                f"Mean minimum full-note cost is {float(row.get('mean_min_full_note_cost_word') or 0.0):.1f} words "
                f"and {float(row.get('mean_min_full_note_cost_byte_overhead') or 0.0):.1f} byte-overhead units.",
            ]
        )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
