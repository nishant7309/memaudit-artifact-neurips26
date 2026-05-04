"""Score existing Mem0-written stores against an OracleMem coverage package.

This script avoids rerunning public Mem0.  It reuses ``written_stores.jsonl``
from a prior Mem0 run, maps those memories to a supplied package's evidence
units, and reports budgeted scores.  It is intended for adjudicated subsets
where the package labels changed but the Mem0-written memories are already
available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import CandidateMemory, OracleMemInstance, objective_value, solve_exact

from llm_memory_validation.gemini_natural_oraclemem import OpenRouterJsonClient, load_env_file, word_count
from llm_memory_validation.run_mem0_natural_baseline import (
    PackageData,
    load_package,
    package_instance,
    read_jsonl,
    score_mem0_coverage,
    select_oracle_density_pruned,
    select_recency_pruned,
    write_json,
    write_jsonl,
)


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def stdev(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return statistics.stdev(clean)


def stores_by_instance(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row.get("instance_id")): row for row in rows if row.get("instance_id")}


def salience_prompt(*, query: Mapping[str, Any], memories: Sequence[Mapping[str, Any]]) -> str:
    memory_rows = [
        {
            "memory_id": str(row.get("memory_id")),
            "text": str(row.get("text", "")),
            "cost_words": word_count(str(row.get("text", ""))),
        }
        for row in memories
    ]
    payload = {
        "query_id": query.get("query_id"),
        "question": query.get("question"),
        "memories": memory_rows,
    }
    return (
        "You are scoring memories for a query-time budget policy.\n"
        "Score each memory for likely usefulness in answering the question, using only the memory text and question.\n"
        "Do not use any gold answer or hidden evidence labels. Scores should be in [0, 1].\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "scores": [{"memory_id": "...", "salience": 0.0, "rationale": "..."}]\n'
        "}\n\n"
        f"PACKAGE:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def score_salience(
    *,
    client: OpenRouterJsonClient,
    query: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not memories:
        return {}
    response = client(salience_prompt(query=query, memories=memories), purpose="mem0_salience_scoring")
    parsed = response.get("parsed", {}) if isinstance(response, Mapping) else {}
    allowed = {str(row.get("memory_id")) for row in memories}
    by_id: dict[str, dict[str, Any]] = {}
    for row in parsed.get("scores", []) or []:
        if not isinstance(row, Mapping):
            continue
        memory_id = str(row.get("memory_id", ""))
        if memory_id not in allowed:
            continue
        by_id[memory_id] = {
            "salience": max(0.0, min(1.0, float(row.get("salience", 0.0) or 0.0))),
            "rationale": str(row.get("rationale", "")),
            "prompt_hash": response.get("prompt_hash"),
            "cache_hit": response.get("cache_hit"),
            "usage": response.get("usage", {}),
        }
    return by_id


def attach_salience(
    candidates: Sequence[CandidateMemory],
    memories: Sequence[Mapping[str, Any]],
    salience_by_memory: Mapping[str, Mapping[str, Any]],
) -> list[CandidateMemory]:
    # score_mem0_coverage names candidate ids by memory order, so use the same order.
    scored: list[CandidateMemory] = []
    for index, candidate in enumerate(candidates):
        memory_id = str(memories[index].get("memory_id")) if index < len(memories) else ""
        salience = float((salience_by_memory.get(memory_id) or {}).get("salience", 0.0) or 0.0)
        scored.append(
            CandidateMemory(
                candidate_id=candidate.candidate_id,
                experience_id=candidate.experience_id,
                representation_type=candidate.representation_type,
                serialized=candidate.serialized,
                cost=candidate.cost,
                coverage=candidate.coverage,
                time_index=candidate.time_index,
                generator=candidate.generator,
                confidence=salience,
                estimated_value=salience,
                estimator_model="gemini_flash_question_salience",
            )
        )
    return scored


def select_salience_pruned(candidates: Sequence[CandidateMemory], budget: int) -> list[CandidateMemory]:
    selected: list[CandidateMemory] = []
    used = 0
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -(float(item.estimated_value or 0.0) / max(1, item.cost)),
            -float(item.estimated_value or 0.0),
            item.cost,
            item.candidate_id,
        ),
    ):
        if float(candidate.estimated_value or 0.0) <= 0:
            continue
        if used + candidate.cost > budget:
            continue
        selected.append(candidate)
        used += candidate.cost
    selected.sort(key=lambda item: item.time_index)
    return selected


def result_row(
    *,
    instance_id: str,
    budget: int,
    method: str,
    selected: Sequence[CandidateMemory],
    package,
    package_denominator: float,
    union_denominator: float,
    runtime_sec: float,
    written_count: int,
    written_cost: int,
) -> dict[str, Any]:
    value = objective_value(selected, package.unit_weights)
    return {
        "instance_id": instance_id,
        "budget": budget,
        "method": method,
        "objective_value": value,
        "package_candidate_exact_opt": package_denominator,
        "package_plus_mem0_exact_opt": union_denominator,
        "ratio_to_package_candidate_opt": value / package_denominator if package_denominator > 0 else None,
        "ratio_to_union_opt": value / union_denominator if union_denominator > 0 else None,
        # Backward-compatible field for older readers. For external Mem0 candidates
        # this is a reference ratio, not an approximation ratio, and can exceed 1.
        "package_exact_opt": package_denominator,
        "package_oracle_ratio": value / package_denominator if package_denominator > 0 else None,
        "selected_cost": sum(candidate.cost for candidate in selected),
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "selected_memory_texts": [candidate.serialized for candidate in selected],
        "written_memory_count": written_count,
        "written_store_cost": written_cost,
        "denominator_label": "package_plus_mem0_exact_opt",
        "runtime_sec": runtime_sec,
    }


def union_instance(package: OracleMemInstance, mem0_candidates: Sequence[CandidateMemory]) -> OracleMemInstance:
    return OracleMemInstance(
        instance_id=f"{package.instance_id}::package_plus_mem0",
        candidates=tuple(package.candidates) + tuple(mem0_candidates),
        unit_weights=package.unit_weights,
        seed=package.seed,
        current_units=package.current_units,
        invalidation_units=package.invalidation_units,
        stale_units=package.stale_units,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--written-stores-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--coverage-model", default="google/gemini-2.5-flash")
    parser.add_argument("--salience-model", default="google/gemini-2.5-flash")
    parser.add_argument("--budgets", default="30,60,100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-salience-pruned", action="store_true")
    parser.add_argument("--include-oracle-pruned-upper", action="store_true")
    parser.add_argument("--request-sleep", type=float, default=0.02)
    args = parser.parse_args()

    env_values = load_env_file(args.api_env)
    for key, value in env_values.items():
        os.environ.setdefault(key, value)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment or api.env")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_package(args.package_dir)
    stores = stores_by_instance(args.written_stores_jsonl)
    budgets = [int(float(item.strip())) for item in args.budgets.split(",") if item.strip()]
    queries = [
        query
        for query in data.queries
        if query.get("required_unit_ids") and str(query.get("query_id")) in stores
    ]
    queries.sort(key=lambda row: str(row.get("query_id", "")))
    if args.limit is not None:
        queries = queries[: args.limit]

    coverage_client = OpenRouterJsonClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.coverage_model,
        cache_path=args.out_dir / "coverage_scoring_cache.json",
        max_tokens=1800,
        request_sleep=args.request_sleep,
    )
    salience_client = OpenRouterJsonClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.salience_model,
        cache_path=args.out_dir / "salience_scoring_cache.json",
        max_tokens=1200,
        request_sleep=args.request_sleep,
    )

    result_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    salience_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for query in queries:
        instance_id = str(query["query_id"])
        started = time.perf_counter()
        store = stores.get(instance_id)
        memories = list((store or {}).get("memories", []) or [])
        if not memories:
            skipped.append({"instance_id": instance_id, "reason": "no_written_memories"})
            continue
        package = package_instance(data, query)
        try:
            mem0_candidates, scoring_record = score_mem0_coverage(
                client=coverage_client,
                data=data,
                query=query,
                memories=memories,
            )
        except Exception as exc:
            skipped.append(
                {
                    "instance_id": instance_id,
                    "reason": "coverage_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        scoring_rows.append(scoring_record)
        salience_candidates = mem0_candidates
        if args.include_salience_pruned:
            try:
                salience_by_memory = score_salience(
                    client=salience_client,
                    query=query,
                    memories=memories,
                )
            except Exception as exc:
                skipped.append(
                    {
                        "instance_id": instance_id,
                        "reason": "salience_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                salience_by_memory = {}
            salience_rows.append(
                {
                    "instance_id": instance_id,
                    "scores": salience_by_memory,
                }
            )
            salience_candidates = attach_salience(mem0_candidates, memories, salience_by_memory)

        for budget in budgets:
            package_exact = solve_exact(package, budget, solver="exact_stdlib")
            union_exact = solve_exact(union_instance(package, mem0_candidates), budget, solver="exact_stdlib")
            package_denominator = package_exact.objective_value
            union_denominator = union_exact.objective_value
            written_cost = sum(candidate.cost for candidate in mem0_candidates)
            result_rows.append(
                result_row(
                    instance_id=instance_id,
                    budget=budget,
                    method="actual_mem0_recency_pruned",
                    selected=select_recency_pruned(mem0_candidates, budget),
                    package=package,
                    package_denominator=package_denominator,
                    union_denominator=union_denominator,
                    runtime_sec=time.perf_counter() - started,
                    written_count=len(mem0_candidates),
                    written_cost=written_cost,
                )
            )
            if args.include_salience_pruned:
                result_rows.append(
                    result_row(
                        instance_id=instance_id,
                        budget=budget,
                        method="actual_mem0_salience_pruned",
                        selected=select_salience_pruned(salience_candidates, budget),
                        package=package,
                        package_denominator=package_denominator,
                        union_denominator=union_denominator,
                        runtime_sec=time.perf_counter() - started,
                        written_count=len(mem0_candidates),
                        written_cost=written_cost,
                    )
                )
            if args.include_oracle_pruned_upper:
                result_rows.append(
                    result_row(
                        instance_id=instance_id,
                        budget=budget,
                        method="actual_mem0_oracle_pruned_upper",
                        selected=select_oracle_density_pruned(mem0_candidates, budget, package.unit_weights),
                        package=package,
                        package_denominator=package_denominator,
                        union_denominator=union_denominator,
                        runtime_sec=time.perf_counter() - started,
                        written_count=len(mem0_candidates),
                        written_cost=written_cost,
                    )
                )

    write_jsonl(args.out_dir / "raw_results.jsonl", result_rows)
    write_jsonl(args.out_dir / "coverage_scoring_calls.jsonl", scoring_rows)
    write_jsonl(args.out_dir / "salience_scoring_calls.jsonl", salience_rows)
    write_jsonl(args.out_dir / "skipped_instances.jsonl", skipped)

    by_method_budget: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        by_method_budget[(str(row["method"]), int(row["budget"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (method, budget), rows in sorted(by_method_budget.items()):
        union_ratios = [row["ratio_to_union_opt"] for row in rows if row.get("ratio_to_union_opt") is not None]
        package_ratios = [
            row["ratio_to_package_candidate_opt"]
            for row in rows
            if row.get("ratio_to_package_candidate_opt") is not None
        ]
        zero_denominator_n = sum(
            1 for row in rows if float(row.get("package_plus_mem0_exact_opt", 0.0) or 0.0) <= 1e-12
        )
        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(rows),
                "ratio_defined_n": len(union_ratios),
                "zero_denominator_n": zero_denominator_n,
                "mean_ratio_to_union_opt": mean(union_ratios),
                "std_ratio_to_union_opt": stdev(union_ratios),
                "mean_ratio_to_package_candidate_opt": mean(package_ratios),
                "std_ratio_to_package_candidate_opt": stdev(package_ratios),
                # Backward-compatible summary fields.
                "mean_package_oracle_ratio": mean(package_ratios),
                "std_package_oracle_ratio": stdev(package_ratios),
                "mean_objective_value": mean([float(row["objective_value"]) for row in rows]),
                "mean_package_candidate_exact_opt": mean(
                    [float(row["package_candidate_exact_opt"]) for row in rows]
                ),
                "mean_package_plus_mem0_exact_opt": mean(
                    [float(row["package_plus_mem0_exact_opt"]) for row in rows]
                ),
                "mean_written_memory_count": mean([float(row["written_memory_count"]) for row in rows]),
                "mean_written_store_cost": mean([float(row["written_store_cost"]) for row in rows]),
            }
        )

    summary = {
        "package_dir": str(args.package_dir),
        "written_stores_jsonl": str(args.written_stores_jsonl),
        "coverage_model": args.coverage_model,
        "salience_model": args.salience_model if args.include_salience_pruned else None,
        "attempted_instances": len(queries),
        "completed_instances": len({row["instance_id"] for row in result_rows}),
        "skipped_instances": len(skipped),
        "budgets": budgets,
        "denominator_label": "package_plus_mem0_exact_opt",
        "summary_rows": summary_rows,
    }
    write_json(args.out_dir / "summary.json", summary)

    lines = [
        "# Mem0 Written Store Rescoring",
        "",
        f"- Package: `{args.package_dir}`",
        f"- Written stores: `{args.written_stores_jsonl}`",
        f"- Coverage judge model: `{args.coverage_model}`",
        f"- Salience model: `{args.salience_model if args.include_salience_pruned else 'not used'}`",
        f"- Attempted instances: {len(queries)}",
        f"- Completed instances: {summary['completed_instances']}",
        f"- Skipped instances: {len(skipped)}",
        "- Primary denominator: exact finite optimum over supplied package candidates plus Mem0-written memories (`package_plus_mem0_exact_opt`).",
        "- Secondary package-candidate ratio is also reported and can exceed 1 for external Mem0 memories.",
        "",
        "| Method | Budget | N | Ratio N | Mean ratio to union OPT | Mean ratio to package-candidate OPT | Mean written memories | Mean store cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {budget} | {n} | {ratio_n} | {union_ratio:.3f} | {package_ratio:.3f} | {count:.2f} | {cost:.1f} |".format(
                method=row["method"],
                budget=row["budget"],
                n=row["n"],
                ratio_n=row["ratio_defined_n"],
                union_ratio=row["mean_ratio_to_union_opt"] if row["mean_ratio_to_union_opt"] is not None else float("nan"),
                package_ratio=(
                    row["mean_ratio_to_package_candidate_opt"]
                    if row["mean_ratio_to_package_candidate_opt"] is not None
                    else float("nan")
                ),
                count=row["mean_written_memory_count"] if row["mean_written_memory_count"] is not None else float("nan"),
                cost=row["mean_written_store_cost"] if row["mean_written_store_cost"] is not None else float("nan"),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "`actual_mem0_salience_pruned` is a query-time Gemini Flash budget heuristic over Mem0-written memories. "
            "It is fairer than pure recency, but it is still not a native Mem0 write-time budget policy. "
            "`actual_mem0_oracle_pruned_upper` uses package coverage labels and is analysis-only. "
            "Ratios to package-candidate OPT are reference ratios, not approximation ratios, because external Mem0 memories are not part of the copied package candidate set.",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
