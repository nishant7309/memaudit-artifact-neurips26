"""Run the actual checked-out A-Mem writer on an OracleMem coverage package.

This is a true-system bridge for the cloned ``external_repos/AgenticMemory``
repository. It feeds package experiences into A-Mem's ``AgenticMemorySystem``,
uses Gemini through OpenRouter for A-Mem metadata/evolution calls, maps the
written A-Mem memories back to OracleMem evidence units with a cached judge, and
reports budgeted scores.

External A-Mem memories are scored against a finite union denominator:
package candidates plus A-Mem-written memories. Package-only ratios are retained
as diagnostics and can exceed or differ from union ratios.

The primary "full" view scores A-Mem's actual stored notes. Because A-Mem stores
large conversation chunks, those notes often exceed the small OracleMem word
budgets. The secondary "metadata" view scores a compact serialization of
A-Mem-generated context/keywords/tags/links; it is a diagnostic for whether
A-Mem's actual metadata contains budget-feasible evidence, not a claim that
A-Mem natively stores only those fields.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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

from llm_memory_validation.gemini_natural_oraclemem import (
    OpenRouterJsonClient,
    load_env_file,
    safe_token,
    word_count,
)
from llm_memory_validation.run_mem0_natural_baseline import (
    PackageData,
    load_package,
    package_instance,
    read_jsonl,
    resolved_queries,
    select_oracle_density_pruned,
    select_recency_pruned,
    write_json,
    write_jsonl,
)
from llm_memory_validation.score_mem0_written_stores import select_salience_pruned, union_instance


DEFAULT_MODEL = "google/gemini-2.5-flash"


class AemOpenRouterLLM:
    """Adapter matching A-Mem's ``get_completion`` interface."""

    def __init__(self, client: OpenRouterJsonClient) -> None:
        self.client = client

    def get_completion(
        self,
        prompt: str,
        response_format: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        _ = response_format, temperature
        response = self.client(prompt, purpose="actual_amem_llm")
        parsed = response.get("parsed", {}) if isinstance(response, Mapping) else {}
        if parsed:
            return json.dumps(parsed, sort_keys=True)
        return str(response.get("raw_content", "{}") if isinstance(response, Mapping) else "{}")


def ensure_amem_importable() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    repo = ROOT / "external_repos" / "AgenticMemory"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def stdev(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return statistics.stdev(clean)


def coverage_prompt(
    *,
    instance_id: str,
    query: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    memories: Sequence[Mapping[str, Any]],
) -> str:
    units = [
        {
            "unit_id": row.get("unit_id"),
            "kind": row.get("kind"),
            "canonical_text": row.get("canonical_text"),
            "unit_weight": row.get("unit_weight"),
            "source_quotes": [
                str(span.get("text", ""))[:500]
                for span in row.get("source_spans", []) or []
                if isinstance(span, Mapping)
            ][:2],
        }
        for row in evidence_rows
    ]
    memory_rows = [
        {"memory_id": str(row.get("memory_id")), "text": str(row.get("text", ""))}
        for row in memories
    ]
    payload = {
        "instance_id": instance_id,
        "question": query.get("question"),
        "required_unit_ids": query.get("required_unit_ids", []),
        "evidence_units": units,
        "amem_memories": memory_rows,
    }
    return (
        "You are auditing A-Mem-written memories for an OracleMem benchmark package.\n"
        "Map each written memory to evidence units only when the memory text entails the unit.\n"
        "Use coverage 1.0 for complete entailment, 0.5 for partial but useful entailment, and omit non-covered pairs.\n"
        "Do not infer missing details from the question or any hidden answer; use only the memory text.\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "coverage_edges": [\n'
        '    {"memory_id": "...", "unit_id": "...", "coverage": 1.0, "rationale": "..."}\n'
        "  ],\n"
        '  "notes": "..."\n'
        "}\n\n"
        f"PACKAGE:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def score_amem_coverage(
    *,
    client: OpenRouterJsonClient,
    data: PackageData,
    query: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
    memory_view: str,
) -> tuple[list[CandidateMemory], dict[str, Any]]:
    instance_id = str(query["query_id"])
    if not memories:
        return [], {"coverage_edges": [], "notes": "No A-Mem memories written.", "cache_hit": None}
    response = client(
        coverage_prompt(
            instance_id=instance_id,
            query=query,
            evidence_rows=data.evidence_by_instance.get(instance_id, []),
            memories=memories,
        ),
        purpose="actual_amem_coverage_scoring",
    )
    parsed = response.get("parsed", {}) if isinstance(response, Mapping) else {}
    allowed_memory_ids = {str(memory["memory_id"]) for memory in memories}
    allowed_unit_ids = {str(row.get("unit_id")) for row in data.evidence_by_instance.get(instance_id, [])}
    coverage_by_memory: dict[str, dict[str, float]] = defaultdict(dict)
    clean_edges: list[dict[str, Any]] = []
    for edge in parsed.get("coverage_edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        memory_id = str(edge.get("memory_id", ""))
        unit_id = str(edge.get("unit_id", ""))
        if memory_id not in allowed_memory_ids or unit_id not in allowed_unit_ids:
            continue
        value = max(0.0, min(1.0, float(edge.get("coverage", edge.get("fidelity", 0.0)) or 0.0)))
        if value <= 0:
            continue
        coverage_by_memory[memory_id][unit_id] = max(value, coverage_by_memory[memory_id].get(unit_id, 0.0))
        clean_edges.append(
            {
                "instance_id": instance_id,
                "memory_id": memory_id,
                "unit_id": unit_id,
                "coverage": value,
                "rationale": str(edge.get("rationale", "")),
            }
        )

    candidates: list[CandidateMemory] = []
    for index, memory in enumerate(memories):
        memory_id = str(memory["memory_id"])
        text = str(memory["text"])
        candidates.append(
            CandidateMemory(
                candidate_id=f"{instance_id}::actual_amem_{safe_token(memory_view)}::{index:04d}",
                experience_id=f"{instance_id}::actual_amem::{index:04d}",
                representation_type=f"actual_amem_{safe_token(memory_view)}",
                serialized=text,
                cost=max(1, word_count(text)),
                coverage=coverage_by_memory.get(memory_id, {}),
                time_index=index,
                generator="actual_amem",
                confidence=float(memory.get("confidence", 1.0) or 1.0),
            )
        )
    return candidates, {
        "instance_id": instance_id,
        "memory_view": memory_view,
        "model": response.get("model") if isinstance(response, Mapping) else None,
        "cache_hit": response.get("cache_hit") if isinstance(response, Mapping) else None,
        "prompt_hash": response.get("prompt_hash") if isinstance(response, Mapping) else None,
        "usage": response.get("usage", {}) if isinstance(response, Mapping) else {},
        "coverage_edges": clean_edges,
        "notes": parsed.get("notes", ""),
    }


def memory_text(note: Any) -> str:
    return "\n".join(
        [
            f"content: {getattr(note, 'content', '')}",
            f"context: {getattr(note, 'context', '')}",
            f"keywords: {', '.join(str(x) for x in getattr(note, 'keywords', []) or [])}",
            f"tags: {', '.join(str(x) for x in getattr(note, 'tags', []) or [])}",
        ]
    ).strip()


def truncate_words(text: str, limit: int) -> str:
    words = str(text).split()
    if len(words) <= limit:
        return str(text)
    return " ".join(words[:limit]) + " ..."


def memory_metadata_text(note: Any) -> str:
    keywords = [str(x) for x in getattr(note, "keywords", []) or []][:12]
    tags = [str(x) for x in getattr(note, "tags", []) or []][:12]
    links = [str(link) for link in getattr(note, "links", []) or []][:8]
    link_text = ", ".join(str(link) for link in links)
    pieces = [
        f"context: {truncate_words(str(getattr(note, 'context', '')), 80)}",
        f"keywords: {', '.join(keywords)}",
        f"tags: {', '.join(tags)}",
    ]
    if link_text:
        pieces.append(f"links: {link_text}")
    return "\n".join(piece for piece in pieces if piece.strip()).strip()


def run_amem_writer(
    *,
    data: PackageData,
    query: Mapping[str, Any],
    llm_client: OpenRouterJsonClient,
    embed_model: str,
    evo_threshold: int,
) -> tuple[list[dict[str, Any]], list[int], str]:
    ensure_amem_importable()
    from memory_layer import AgenticMemorySystem

    system = AgenticMemorySystem(
        model_name=embed_model,
        llm_backend="sglang",
        llm_model="unused",
        evo_threshold=evo_threshold,
    )
    system.llm_controller.llm = AemOpenRouterLLM(llm_client)
    instance_id = str(query["query_id"])
    experiences = sorted(
        data.experiences_by_instance.get(instance_id, []),
        key=lambda row: (int(row.get("time_index", 0) or 0), str(row.get("experience_id", ""))),
    )
    debug = io.StringIO()
    with contextlib.redirect_stdout(debug):
        for row in experiences:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            timestamp = str(row.get("timestamp") or row.get("date") or row.get("experience_id") or "")
            system.add_note(text, time=timestamp)

    memories: list[dict[str, Any]] = []
    for index, (memory_id, note) in enumerate(system.memories.items()):
        memories.append(
            {
                "memory_id": str(memory_id),
                "full_text": memory_text(note),
                "metadata_text": memory_metadata_text(note),
                "text": memory_text(note),
                "content": getattr(note, "content", ""),
                "context": getattr(note, "context", ""),
                "keywords": list(getattr(note, "keywords", []) or []),
                "tags": list(getattr(note, "tags", []) or []),
                "links": list(getattr(note, "links", []) or []),
                "time_index": index,
            }
        )
    query_text = str(query.get("question", ""))
    try:
        native_order = [int(index) for index in system.retriever.search(query_text, k=len(memories))]
    except Exception:
        native_order = list(range(len(memories) - 1, -1, -1))
    return memories, native_order, debug.getvalue()[-20000:]


def select_native_retrieval_pruned(
    candidates: Sequence[CandidateMemory],
    native_order: Sequence[int],
    budget: int,
) -> list[CandidateMemory]:
    selected: list[CandidateMemory] = []
    used = 0
    for index in native_order:
        if index < 0 or index >= len(candidates):
            continue
        candidate = candidates[index]
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
    package: OracleMemInstance,
    package_denominator: float,
    union_denominator: float,
    runtime_sec: float,
    written_count: int,
    written_cost: int,
    memory_view: str,
) -> dict[str, Any]:
    value = objective_value(selected, package.unit_weights)
    return {
        "instance_id": instance_id,
        "budget": budget,
        "method": method,
        "objective_value": value,
        "package_candidate_exact_opt": package_denominator,
        "package_plus_amem_exact_opt": union_denominator,
        "ratio_to_package_candidate_opt": value / package_denominator if package_denominator > 0 else None,
        "ratio_to_union_opt": value / union_denominator if union_denominator > 0 else None,
        "selected_cost": sum(candidate.cost for candidate in selected),
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "selected_memory_texts": [candidate.serialized for candidate in selected],
        "written_memory_count": written_count,
        "written_store_cost": written_cost,
        "memory_view": memory_view,
        "denominator_label": "package_plus_amem_exact_opt",
        "runtime_sec": runtime_sec,
    }


def write_report(out_dir: Path, *, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Actual A-Mem Natural Baseline",
        "",
        f"- Package: `{manifest['package_dir']}`",
        f"- Queries attempted: {manifest['query_count']}",
        f"- A-Mem writer model: `{manifest['amem_model']}`",
        f"- Coverage scorer model: `{manifest['coverage_model']}`",
        "- Denominator: exact finite union OPT over package candidates plus A-Mem-written memories.",
        "- System status: actual checked-out `external_repos/AgenticMemory` writer path, not the local `amem_graph` adapter.",
    ]
    api_usage = manifest.get("api_usage") if isinstance(manifest.get("api_usage"), Mapping) else {}
    if api_usage:
        lines.extend(
            [
                f"- Cached API prompts: {sum(int(row.get('cached_prompts') or 0) for row in api_usage.values() if isinstance(row, Mapping))}",
                f"- API tokens: {int(api_usage.get('total_tokens') or 0)}",
                f"- Estimated OpenRouter cost: ${float(api_usage.get('total_estimated_cost_usd') or 0.0):.3f}",
            ]
        )
    lines.extend(["", "## Mean Ratio To Union OPT", ""])
    budgets = sorted({int(row["budget"]) for row in summary.get("by_method_budget", [])})
    methods = sorted({str(row["method"]) for row in summary.get("by_method_budget", [])})
    lines.append("| Method | " + " | ".join(f"B={budget}" for budget in budgets) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in budgets) + " |")
    by_key = {
        (int(row["budget"]), str(row["method"])): row
        for row in summary.get("by_method_budget", [])
    }
    for method in methods:
        cells = []
        for budget in budgets:
            value = (by_key.get((budget, method)) or {}).get("mean_ratio_to_union_opt")
            cells.append("--" if value is None else f"{float(value):.3f}")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    lines.extend(["", "## Notes", ""])
    lines.append("- `actual_amem_full_*` scores A-Mem's actual full stored notes. These can be much larger than the benchmark budgets.")
    lines.append("- `actual_amem_metadata_*` scores a compact serialization of A-Mem-generated context/keywords/tags/links. This is a diagnostic view, not A-Mem's raw storage policy.")
    lines.append("- `*_native_retrieval_pruned` uses A-Mem's query-time retriever, so it is a retrieval/context diagnostic rather than a pure write-time budget policy.")
    lines.append("- `*_oracle_pruned_upper` is analysis-only and uses hidden coverage to upper-bound the value present in A-Mem's written store.")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(rows: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["budget"]))].append(row)
    summary_rows = []
    for (method, budget), items in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(items),
                "mean_ratio_to_union_opt": mean([row.get("ratio_to_union_opt") for row in items]),
                "std_ratio_to_union_opt": stdev([row.get("ratio_to_union_opt") for row in items]),
                "mean_ratio_to_package_candidate_opt": mean([row.get("ratio_to_package_candidate_opt") for row in items]),
                "mean_objective": mean([row.get("objective_value") for row in items]),
                "mean_selected_cost": mean([row.get("selected_cost") for row in items]),
                "mean_written_memory_count": mean([row.get("written_memory_count") for row in items]),
                "mean_written_store_cost": mean([row.get("written_store_cost") for row in items]),
            }
        )
    return {
        "by_method_budget": summary_rows,
        "result_rows": len(rows),
        "skipped_rows": len(skipped),
        "skipped": list(skipped),
    }


def api_usage_summary(out_dir: Path) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for name in ("amem_llm_cache.json", "coverage_scoring_cache.json"):
        path = out_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        usage[name] = {
            "cached_prompts": len(data),
            "total_tokens": sum(int((row.get("usage") or {}).get("total_tokens") or 0) for row in data.values()),
            "estimated_cost_usd": sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in data.values()),
        }
    cache_rows = [row for row in usage.values() if isinstance(row, Mapping)]
    usage["total_estimated_cost_usd"] = sum(float(row.get("estimated_cost_usd") or 0.0) for row in cache_rows)
    usage["total_tokens"] = sum(int(row.get("total_tokens") or 0) for row in cache_rows)
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package"))
    parser.add_argument("--out-dir", type=Path, default=Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash"))
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--amem-model", default=DEFAULT_MODEL)
    parser.add_argument("--coverage-model", default=DEFAULT_MODEL)
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--budgets", default="30,60,100")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--request-sleep", type=float, default=0.02)
    parser.add_argument("--evo-threshold", type=int, default=100)
    parser.add_argument("--amem-max-tokens", type=int, default=3000)
    parser.add_argument("--coverage-max-tokens", type=int, default=2200)
    args = parser.parse_args()

    env_values = load_env_file(args.api_env)
    for key, value in env_values.items():
        os.environ.setdefault(key, value)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required in api.env or environment")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(float(item.strip())) for item in args.budgets.split(",") if item.strip()]
    data = load_package(args.package_dir)
    queries = resolved_queries(data, args.limit)
    amem_client = OpenRouterJsonClient(
        api_key=api_key,
        model=args.amem_model,
        cache_path=args.out_dir / "amem_llm_cache.json",
        max_tokens=args.amem_max_tokens,
        request_sleep=args.request_sleep,
    )
    coverage_client = OpenRouterJsonClient(
        api_key=api_key,
        model=args.coverage_model,
        cache_path=args.out_dir / "coverage_scoring_cache.json",
        max_tokens=args.coverage_max_tokens,
        request_sleep=args.request_sleep,
    )

    result_rows: list[dict[str, Any]] = []
    written_store_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for query in queries:
        instance_id = str(query["query_id"])
        started = time.perf_counter()
        package = package_instance(data, query)
        if not package.candidates:
            skipped_rows.append({"instance_id": instance_id, "reason": "no_package_candidates"})
            continue
        try:
            memories, native_order, debug_log = run_amem_writer(
                data=data,
                query=query,
                llm_client=amem_client,
                embed_model=args.embed_model,
                evo_threshold=args.evo_threshold,
            )
            full_memories = [
                {**memory, "text": str(memory.get("full_text", memory.get("text", "")))}
                for memory in memories
            ]
            metadata_memories = [
                {**memory, "text": str(memory.get("metadata_text", ""))}
                for memory in memories
                if str(memory.get("metadata_text", "")).strip()
            ]
            full_candidates, full_scoring_record = score_amem_coverage(
                client=coverage_client,
                data=data,
                query=query,
                memories=full_memories,
                memory_view="full",
            )
            metadata_candidates, metadata_scoring_record = score_amem_coverage(
                client=coverage_client,
                data=data,
                query=query,
                memories=metadata_memories,
                memory_view="metadata",
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "instance_id": instance_id,
                    "reason": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        written_store_rows.append(
            {
                "instance_id": instance_id,
                "question": query.get("question"),
                "memories": memories,
                "memory_count": len(memories),
                "native_order": native_order,
            }
        )
        scoring_rows.append(full_scoring_record)
        scoring_rows.append(metadata_scoring_record)
        debug_rows.append({"instance_id": instance_id, "debug_tail": debug_log})

        union = union_instance(package, full_candidates + metadata_candidates)
        for budget in budgets:
            package_exact = solve_exact(package, budget, solver="exact_stdlib")
            union_exact = solve_exact(union, budget, solver="exact_stdlib")
            selectors: dict[str, tuple[list[CandidateMemory], Sequence[CandidateMemory], str]] = {
                "actual_amem_full_recency_pruned": (select_recency_pruned(full_candidates, budget), full_candidates, "full"),
                "actual_amem_full_native_retrieval_pruned": (select_native_retrieval_pruned(full_candidates, native_order, budget), full_candidates, "full"),
                "actual_amem_full_oracle_pruned_upper": (select_oracle_density_pruned(full_candidates, budget, package.unit_weights), full_candidates, "full"),
                "actual_amem_metadata_recency_pruned": (select_recency_pruned(metadata_candidates, budget), metadata_candidates, "metadata"),
                "actual_amem_metadata_native_retrieval_pruned": (select_native_retrieval_pruned(metadata_candidates, native_order, budget), metadata_candidates, "metadata"),
                "actual_amem_metadata_oracle_pruned_upper": (select_oracle_density_pruned(metadata_candidates, budget, package.unit_weights), metadata_candidates, "metadata"),
            }
            for method, (selected, candidate_pool, memory_view) in selectors.items():
                result_rows.append(
                    result_row(
                        instance_id=instance_id,
                        budget=budget,
                        method=method,
                        selected=selected,
                        package=package,
                        package_denominator=package_exact.objective_value,
                        union_denominator=union_exact.objective_value,
                        runtime_sec=time.perf_counter() - started,
                        written_count=len(candidate_pool),
                        written_cost=sum(candidate.cost for candidate in candidate_pool),
                        memory_view=memory_view,
                    )
                )

    write_jsonl(args.out_dir / "raw_results.jsonl", result_rows)
    write_jsonl(args.out_dir / "written_stores.jsonl", written_store_rows)
    write_jsonl(args.out_dir / "coverage_scoring_calls.jsonl", scoring_rows)
    write_jsonl(args.out_dir / "debug_logs.jsonl", debug_rows)
    write_jsonl(args.out_dir / "skipped_instances.jsonl", skipped_rows)
    summary = summarize(result_rows, skipped_rows)
    manifest = {
        "package_dir": str(args.package_dir),
        "out_dir": str(args.out_dir),
        "query_count": len(queries),
        "budgets": budgets,
        "amem_model": args.amem_model,
        "coverage_model": args.coverage_model,
        "embed_model": args.embed_model,
        "limit": args.limit,
        "amem_max_tokens": args.amem_max_tokens,
        "coverage_max_tokens": args.coverage_max_tokens,
        "denominator": "package_plus_amem_exact_opt",
        "actual_system_repo": "external_repos/AgenticMemory",
        "result_rows": len(result_rows),
        "skipped_rows": len(skipped_rows),
    }
    manifest["api_usage"] = api_usage_summary(args.out_dir)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "run_manifest.json", manifest)
    write_report(args.out_dir, summary=summary, manifest=manifest)
    print(json.dumps({"results": len(result_rows), "skipped": len(skipped_rows), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
