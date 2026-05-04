"""Run an actual Mem0 writer on a natural OracleMem coverage package.

This is a benchmark bridge, not a synthetic OracleMem runner.  It feeds the
same package experiences to public Mem0, maps the memories Mem0 writes back to
the package evidence units with a cached OpenRouter judge, and reports the
budgeted value of the resulting store against the package's exact finite OPT.

The denominator is the exact optimum over the package candidate set, not an
optimum over all possible natural-language memories.  Output labels therefore
use ``package_exact_opt`` and ``package_oracle_ratio``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import (
    CandidateMemory,
    OracleMemInstance,
    objective_value,
    solve_exact,
)

from llm_memory_validation.gemini_natural_oraclemem import (
    DEFAULT_MODEL,
    OpenRouterJsonClient,
    load_env_file,
    safe_token,
    word_count,
)


def ensure_mem0_importable() -> None:
    """Prefer an installed Mem0 package, fall back to the checked-out repo."""

    try:
        __import__("mem0")
        return
    except ModuleNotFoundError:
        pass
    local_repo = ROOT / "external_repos" / "mem0"
    if local_repo.exists():
        sys.path.insert(0, str(local_repo))
        # The source checkout expects installed package metadata for mem0ai.
        # For benchmark runs from the git checkout, provide a local version
        # shim without modifying the external repository.
        import importlib.metadata

        original_version = importlib.metadata.version

        def version_with_local_mem0(name: str) -> str:
            if name == "mem0ai":
                return "local-source"
            return original_version(name)

        importlib.metadata.version = version_with_local_mem0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def prefix_of(item_id: str) -> str:
    return str(item_id).split("::", 1)[0]


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def stdev(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return 0.0 if clean else None
    return statistics.stdev(clean)


@dataclass(frozen=True)
class PackageData:
    package_dir: Path
    queries: list[dict[str, Any]]
    experiences_by_instance: Mapping[str, list[dict[str, Any]]]
    evidence_by_instance: Mapping[str, list[dict[str, Any]]]
    candidate_rows_by_instance: Mapping[str, list[dict[str, Any]]]
    coverage_by_candidate: Mapping[str, dict[str, float]]


def load_package(package_dir: Path) -> PackageData:
    queries = read_jsonl(package_dir / "queries.jsonl")
    experiences = read_jsonl(package_dir / "experiences.jsonl")
    evidence_units = read_jsonl(package_dir / "evidence_units.jsonl")
    candidate_rows = read_jsonl(package_dir / "candidate_memories.jsonl")
    coverage_rows = read_jsonl(package_dir / "coverage_matrix.jsonl")

    experiences_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in experiences:
        experiences_by_instance[prefix_of(str(row.get("experience_id", "")))].append(row)

    evidence_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_units:
        evidence_by_instance[prefix_of(str(row.get("unit_id", "")))].append(row)

    candidate_rows_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_rows_by_instance[prefix_of(str(row.get("candidate_id", "")))].append(row)

    coverage_by_candidate: dict[str, dict[str, float]] = defaultdict(dict)
    for row in coverage_rows:
        value = float(row.get("coverage", row.get("fidelity", 0.0)) or 0.0)
        if value <= 0:
            continue
        coverage_by_candidate[str(row["candidate_id"])][str(row["unit_id"])] = value

    return PackageData(
        package_dir=package_dir,
        queries=queries,
        experiences_by_instance=experiences_by_instance,
        evidence_by_instance=evidence_by_instance,
        candidate_rows_by_instance=candidate_rows_by_instance,
        coverage_by_candidate=coverage_by_candidate,
    )


def package_instance(data: PackageData, query: Mapping[str, Any]) -> OracleMemInstance:
    instance_id = str(query["query_id"])
    candidates: list[CandidateMemory] = []
    for row in data.candidate_rows_by_instance.get(instance_id, []):
        candidate_id = str(row["candidate_id"])
        candidates.append(
            CandidateMemory(
                candidate_id=candidate_id,
                experience_id=str(row.get("experience_id") or row.get("candidate_group") or candidate_id),
                representation_type=str(row.get("representation_type", "unknown")),
                serialized=str(row.get("serialized") or row.get("text") or ""),
                cost=max(1, int(row.get("cost", row.get("cost_tokens", 1)) or 1)),
                coverage=data.coverage_by_candidate.get(candidate_id, {}),
                time_index=int(row.get("time_index", 0) or 0),
                generator=str(row.get("generator_id", row.get("generator", "package"))),
                confidence=float(row.get("confidence", 1.0) or 1.0),
            )
        )

    unit_weights = {
        str(row["unit_id"]): float(row.get("unit_weight", 0.0) or 0.0)
        for row in data.evidence_by_instance.get(instance_id, [])
    }
    for unit_id in query.get("required_unit_ids", []) or []:
        unit_weights[str(unit_id)] = max(1.0, float(unit_weights.get(str(unit_id), 0.0)))

    return OracleMemInstance(
        instance_id=instance_id,
        candidates=candidates,
        unit_weights=unit_weights,
        current_units=tuple(unit for unit, weight in unit_weights.items() if weight > 0),
    )


def resolved_queries(data: PackageData, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        query
        for query in data.queries
        if query.get("required_unit_ids")
        and data.candidate_rows_by_instance.get(str(query.get("query_id")))
        and data.evidence_by_instance.get(str(query.get("query_id")))
    ]
    rows.sort(key=lambda row: str(row.get("query_id", "")))
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_mem0_config(out_dir: Path, instance_id: str, model: str) -> dict[str, Any]:
    safe_id = safe_token(instance_id)
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "temperature": 0.0,
                "max_tokens": 700,
                "openrouter_base_url": "https://openrouter.ai/api/v1",
                "site_url": "https://localhost/oraclemem",
                "app_name": "OracleMem Mem0 Natural Baseline",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"oraclemem_mem0_{safe_id[:48]}",
                "path": str(out_dir / "qdrant" / safe_id),
                "embedding_model_dims": 384,
            },
        },
        "history_db_path": str(out_dir / "history" / f"{safe_id}.db"),
        "version": "v1.1",
    }


def ordered_experiences(data: PackageData, instance_id: str) -> list[dict[str, Any]]:
    rows = list(data.experiences_by_instance.get(instance_id, []))
    time_by_experience: dict[str, int] = {}
    for candidate in data.candidate_rows_by_instance.get(instance_id, []):
        exp_id = str(candidate.get("experience_id", ""))
        time_by_experience[exp_id] = min(
            int(candidate.get("time_index", 0) or 0),
            time_by_experience.get(exp_id, int(candidate.get("time_index", 0) or 0)),
        )
    rows.sort(
        key=lambda row: (
            time_by_experience.get(str(row.get("experience_id", "")), 10**9),
            str(row.get("timestamp", "")),
            str(row.get("experience_id", "")),
        )
    )
    return rows


def extract_mem0_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, Mapping):
        raw_results = raw.get("results", raw.get("memories", []))
    else:
        raw_results = raw
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_results or []):
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("memory") or item.get("text") or item.get("content") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "memory_index": index,
                "memory_id": str(item.get("id") or f"mem0_{index}"),
                "text": text,
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", "")),
                "raw": dict(item),
            }
        )
    rows.sort(key=lambda row: (row["created_at"], row["updated_at"], row["memory_index"], row["memory_id"]))
    return rows


def run_mem0_writer(
    *,
    data: PackageData,
    query: Mapping[str, Any],
    out_dir: Path,
    model: str,
    reuse_store: bool,
    max_experience_words: int,
    memory: Any | None = None,
    store_dir: Path | None = None,
) -> dict[str, Any]:
    instance_id = str(query["query_id"])
    safe_id = safe_token(instance_id)
    instance_dir = store_dir or (out_dir / "stores" / safe_id)
    if memory is None:
        ensure_mem0_importable()
        from mem0 import Memory

        if instance_dir.exists() and not reuse_store:
            shutil.rmtree(instance_dir)
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "history").mkdir(parents=True, exist_ok=True)
        (instance_dir / "qdrant").mkdir(parents=True, exist_ok=True)
        config = build_mem0_config(instance_dir, instance_id, model)
        memory = Memory.from_config(config)
    user_id = f"oraclemem::{instance_id}"
    add_rows: list[dict[str, Any]] = []

    if not reuse_store:
        for experience in ordered_experiences(data, instance_id):
            text = str(experience.get("text", "")).strip()
            if not text:
                continue
            if max_experience_words > 0 and word_count(text) > max_experience_words:
                words = text.split()
                text = " ".join(words[:max_experience_words]) + " ..."
            started = time.perf_counter()
            result = memory.add([{"role": "user", "content": text}], user_id=user_id)
            add_rows.append(
                {
                    "instance_id": instance_id,
                    "experience_id": experience.get("experience_id"),
                    "source_kind": experience.get("source_kind"),
                    "text_words": word_count(text),
                    "runtime_sec": time.perf_counter() - started,
                    "result": result,
                }
            )

    all_result = memory.get_all(filters={"user_id": user_id}, top_k=200)
    memories = extract_mem0_results(all_result)
    return {
        "instance_id": instance_id,
        "add_rows": add_rows,
        "all_result": all_result,
        "memories": memories,
        "memory_count": len(memories),
        "store_dir": str(instance_dir),
    }


def coverage_prompt(
    *,
    instance_id: str,
    query: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    memories: Sequence[Mapping[str, Any]],
) -> str:
    units = [
        {
            "unit_id": str(row.get("unit_id")),
            "canonical_text": str(row.get("canonical_text", "")),
            "unit_weight": float(row.get("unit_weight", 0.0) or 0.0),
            "source_quotes": [
                str(span.get("text", ""))
                for span in row.get("source_spans", []) or []
                if isinstance(span, Mapping)
            ][:2],
        }
        for row in evidence_rows
    ]
    memory_rows = [
        {
            "memory_id": str(row.get("memory_id")),
            "text": str(row.get("text", "")),
        }
        for row in memories
    ]
    payload = {
        "instance_id": instance_id,
        "question": query.get("question"),
        "gold_answer": query.get("answer"),
        "required_unit_ids": query.get("required_unit_ids", []),
        "evidence_units": units,
        "mem0_memories": memory_rows,
    }
    return (
        "You are auditing a memory writer for an OracleMem benchmark package.\n"
        "Map Mem0-written memories to evidence units only when the memory text entails the unit.\n"
        "Use coverage 1.0 for complete entailment, 0.5 for partial but useful entailment, and omit non-covered pairs.\n"
        "Do not infer missing details from the question or gold answer; use only the memory text.\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "coverage_edges": [\n'
        '    {"memory_id": "...", "unit_id": "...", "coverage": 1.0, "rationale": "..."}\n'
        "  ],\n"
        '  "notes": "..."\n'
        "}\n\n"
        f"PACKAGE:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def score_mem0_coverage(
    *,
    client: OpenRouterJsonClient,
    data: PackageData,
    query: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
) -> tuple[list[CandidateMemory], dict[str, Any]]:
    instance_id = str(query["query_id"])
    if not memories:
        return [], {"coverage_edges": [], "notes": "No Mem0 memories written.", "cache_hit": None}
    response = client(
        coverage_prompt(
            instance_id=instance_id,
            query=query,
            evidence_rows=data.evidence_by_instance.get(instance_id, []),
            memories=memories,
        ),
        purpose="mem0_coverage_scoring",
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
                candidate_id=f"{instance_id}::mem0::{index:04d}",
                experience_id=f"{instance_id}::mem0::{index:04d}",
                representation_type="mem0_memory",
                serialized=text,
                cost=max(1, word_count(text)),
                coverage=coverage_by_memory.get(memory_id, {}),
                time_index=index,
                generator="actual_mem0",
                confidence=1.0,
            )
        )
    scoring_record = {
        "instance_id": instance_id,
        "model": response.get("model") if isinstance(response, Mapping) else None,
        "cache_hit": response.get("cache_hit") if isinstance(response, Mapping) else None,
        "prompt_hash": response.get("prompt_hash") if isinstance(response, Mapping) else None,
        "usage": response.get("usage", {}) if isinstance(response, Mapping) else {},
        "coverage_edges": clean_edges,
        "notes": parsed.get("notes", ""),
    }
    return candidates, scoring_record


def select_recency_pruned(candidates: Sequence[CandidateMemory], budget: int) -> list[CandidateMemory]:
    selected: list[CandidateMemory] = []
    used = 0
    for candidate in sorted(candidates, key=lambda item: item.time_index, reverse=True):
        if used + candidate.cost > budget:
            continue
        selected.append(candidate)
        used += candidate.cost
    selected.sort(key=lambda item: item.time_index)
    return selected


def select_oracle_density_pruned(
    candidates: Sequence[CandidateMemory],
    budget: int,
    unit_weights: Mapping[str, float],
) -> list[CandidateMemory]:
    selected: list[CandidateMemory] = []
    used = 0
    totals: dict[str, float] = {}
    remaining = list(candidates)
    while remaining:
        best: tuple[float, CandidateMemory] | None = None
        for candidate in remaining:
            if used + candidate.cost > budget:
                continue
            before = objective_value(selected, unit_weights)
            after = objective_value(selected + [candidate], unit_weights)
            density = (after - before) / max(1, candidate.cost)
            if best is None or density > best[0]:
                best = (density, candidate)
        if best is None or best[0] <= 0:
            break
        chosen = best[1]
        selected.append(chosen)
        used += chosen.cost
        for unit_id, value in chosen.coverage.items():
            totals[unit_id] = totals.get(unit_id, 0.0) + value
        remaining.remove(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("llm_memory_validation/mem0_natural_baseline"))
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--coverage-model", default=DEFAULT_MODEL)
    parser.add_argument("--budgets", default="30,60,100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-store", action="store_true")
    parser.add_argument("--max-experience-words", type=int, default=1800)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-oracle-pruned-upper", action="store_true")
    parser.add_argument("--per-instance-store", action="store_true")
    parser.add_argument("--request-sleep", type=float, default=0.02)
    args = parser.parse_args()

    env_values = load_env_file(args.api_env)
    for key, value in env_values.items():
        os.environ.setdefault(key, value)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment or api.env")

    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    budgets = [int(float(item.strip())) for item in args.budgets.split(",") if item.strip()]
    data = load_package(args.package_dir)
    queries = resolved_queries(data, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenRouterJsonClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.coverage_model,
        cache_path=args.out_dir / "coverage_scoring_cache.json",
        max_tokens=1800,
        request_sleep=args.request_sleep,
    )

    shared_memory: Any | None = None
    shared_store_dir: Path | None = None
    if not args.per_instance_store:
        ensure_mem0_importable()
        from mem0 import Memory

        shared_store_dir = args.out_dir / "stores" / "shared"
        if shared_store_dir.exists() and not args.reuse_store:
            shutil.rmtree(shared_store_dir)
        shared_store_dir.mkdir(parents=True, exist_ok=True)
        (shared_store_dir / "history").mkdir(parents=True, exist_ok=True)
        (shared_store_dir / "qdrant").mkdir(parents=True, exist_ok=True)
        shared_memory = Memory.from_config(build_mem0_config(shared_store_dir, "shared", args.model))

    raw_store_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    add_rows_all: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for query in queries:
        instance_id = str(query["query_id"])
        result_marker = args.out_dir / "per_instance" / f"{safe_token(instance_id)}.done.json"
        if args.skip_existing and result_marker.exists():
            continue
        started = time.perf_counter()
        package = package_instance(data, query)
        if not package.candidates:
            skipped_rows.append({"instance_id": instance_id, "reason": "no_package_candidates"})
            continue

        try:
            store = run_mem0_writer(
                data=data,
                query=query,
                out_dir=args.out_dir,
                model=args.model,
                reuse_store=args.reuse_store,
                max_experience_words=args.max_experience_words,
                memory=shared_memory,
                store_dir=shared_store_dir,
            )
            mem0_candidates, scoring_record = score_mem0_coverage(
                client=client,
                data=data,
                query=query,
                memories=store["memories"],
            )
        except Exception as exc:  # keep long runs resumable and auditable
            skipped_rows.append(
                {
                    "instance_id": instance_id,
                    "reason": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        raw_store_rows.append(
            {
                "instance_id": instance_id,
                "question": query.get("question"),
                "answer": query.get("answer"),
                "memories": store["memories"],
                "memory_count": store["memory_count"],
                "store_dir": store["store_dir"],
            }
        )
        add_rows_all.extend(store["add_rows"])
        scoring_rows.append(scoring_record)

        for budget in budgets:
            exact = solve_exact(package, budget, solver="exact_stdlib")
            selected = select_recency_pruned(mem0_candidates, budget)
            value = objective_value(selected, package.unit_weights)
            denominator = exact.objective_value
            result_rows.append(
                {
                    "instance_id": instance_id,
                    "budget": budget,
                    "method": "actual_mem0_recency_pruned",
                    "objective_value": value,
                    "package_exact_opt": denominator,
                    "package_oracle_ratio": value / denominator if denominator > 0 else None,
                    "selected_cost": sum(candidate.cost for candidate in selected),
                    "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
                    "selected_memory_texts": [candidate.serialized for candidate in selected],
                    "written_memory_count": len(mem0_candidates),
                    "written_store_cost": sum(candidate.cost for candidate in mem0_candidates),
                    "denominator_label": "package_exact_opt",
                    "runtime_sec": time.perf_counter() - started,
                }
            )
            if args.include_oracle_pruned_upper:
                oracle_selected = select_oracle_density_pruned(mem0_candidates, budget, package.unit_weights)
                oracle_value = objective_value(oracle_selected, package.unit_weights)
                result_rows.append(
                    {
                        "instance_id": instance_id,
                        "budget": budget,
                        "method": "actual_mem0_oracle_pruned_upper",
                        "objective_value": oracle_value,
                        "package_exact_opt": denominator,
                        "package_oracle_ratio": oracle_value / denominator if denominator > 0 else None,
                        "selected_cost": sum(candidate.cost for candidate in oracle_selected),
                        "selected_candidate_ids": [candidate.candidate_id for candidate in oracle_selected],
                        "selected_memory_texts": [candidate.serialized for candidate in oracle_selected],
                        "written_memory_count": len(mem0_candidates),
                        "written_store_cost": sum(candidate.cost for candidate in mem0_candidates),
                        "denominator_label": "package_exact_opt",
                        "runtime_sec": time.perf_counter() - started,
                    }
                )

        result_marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            result_marker,
            {
                "instance_id": instance_id,
                "memory_count": store["memory_count"],
                "runtime_sec": time.perf_counter() - started,
            },
        )

    write_jsonl(args.out_dir / "written_stores.jsonl", raw_store_rows)
    write_jsonl(args.out_dir / "mem0_add_calls.jsonl", add_rows_all)
    write_jsonl(args.out_dir / "coverage_scoring_calls.jsonl", scoring_rows)
    write_jsonl(args.out_dir / "raw_results.jsonl", result_rows)
    write_jsonl(args.out_dir / "skipped_instances.jsonl", skipped_rows)

    by_method_budget: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        by_method_budget[(str(row["method"]), int(row["budget"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (method, budget), rows in sorted(by_method_budget.items()):
        ratios = [row["package_oracle_ratio"] for row in rows if row.get("package_oracle_ratio") is not None]
        zero_denominator_n = sum(1 for row in rows if float(row.get("package_exact_opt", 0.0) or 0.0) <= 1e-12)
        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(rows),
                "ratio_defined_n": len(ratios),
                "zero_denominator_n": zero_denominator_n,
                "mean_package_oracle_ratio": mean(ratios),
                "std_package_oracle_ratio": stdev(ratios),
                "mean_objective_value": mean([float(row["objective_value"]) for row in rows]),
                "mean_package_exact_opt": mean([float(row["package_exact_opt"]) for row in rows]),
                "mean_written_memory_count": mean([float(row["written_memory_count"]) for row in rows]),
                "mean_written_store_cost": mean([float(row["written_store_cost"]) for row in rows]),
            }
        )

    summary = {
        "package_dir": str(args.package_dir),
        "model": args.model,
        "coverage_model": args.coverage_model,
        "attempted_instances": len(queries),
        "completed_instances": len({row["instance_id"] for row in result_rows}),
        "skipped_instances": len(skipped_rows),
        "budgets": budgets,
        "denominator_label": "package_exact_opt",
        "summary_rows": summary_rows,
    }
    write_json(args.out_dir / "summary.json", summary)

    report_lines = [
        "# Actual Mem0 Natural OracleMem Baseline",
        "",
        f"- Package: `{args.package_dir}`",
        f"- Mem0 LLM model: `{args.model}`",
        f"- Coverage judge model: `{args.coverage_model}`",
        f"- Attempted resolved instances: {len(queries)}",
        f"- Completed instances: {summary['completed_instances']}",
        f"- Skipped instances: {len(skipped_rows)}",
        f"- Denominator: exact finite optimum over package candidates (`package_exact_opt`).",
        "",
        "| Method | Budget | N | Ratio N | Mean package oracle ratio | Std | Mean written memories | Mean store cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| {method} | {budget} | {n} | {ratio_n} | {ratio:.3f} | {std:.3f} | {count:.2f} | {cost:.1f} |".format(
                method=row["method"],
                budget=row["budget"],
                n=row["n"],
                ratio_n=row["ratio_defined_n"],
                ratio=row["mean_package_oracle_ratio"] if row["mean_package_oracle_ratio"] is not None else float("nan"),
                std=row["std_package_oracle_ratio"] if row["std_package_oracle_ratio"] is not None else float("nan"),
                count=row["mean_written_memory_count"] if row["mean_written_memory_count"] is not None else float("nan"),
                cost=row["mean_written_store_cost"] if row["mean_written_store_cost"] is not None else float("nan"),
            )
        )
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
