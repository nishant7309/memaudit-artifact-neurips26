"""Adjudicate a natural OracleMem coverage package with a separate LLM judge.

The Natural-200 package is useful only if its evidence-unit labels and coverage
edges are semantically stable.  This script builds a smaller adjudicated package
from an existing natural package:

* candidate memories are copied from the primary package;
* a separate Gemini Flash adjudicator reviews required evidence units and
  candidate-unit coverage edges;
* only accepted/corrected adjudications are exported into a new coverage
  package;
* exact package-OPT and baseline scores are recomputed on the adjudicated
  package.

This is not human adjudication.  It is an intermediate validity check that is
cheaper than human review and more useful than treating primary annotation as
ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import evaluate_instance, write_benchmark_outputs

from llm_memory_validation.gemini_natural_oraclemem import (
    OpenRouterJsonClient,
    load_env_file,
    stable_hash,
    truncate_words,
    word_count,
)
from llm_memory_validation.run_mem0_natural_baseline import (
    PackageData,
    load_package,
    package_instance,
    prefix_of,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_ADJUDICATOR_MODEL = "google/gemini-2.5-flash"
DEFAULT_METHODS = (
    "opt",
    "oracle_gvt",
    "summary_only",
    "fact_only",
    "mem0_extract",
    "amem_graph",
    "recency_raw",
    "estimated_gvt",
)


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return statistics.fmean(clean)


def read_disagreement_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        label = str(row.get("agreement_label", row.get("status", ""))).lower()
        if "major" in label or "disagreement" in label or "unresolved" in label:
            query_id = row.get("query_id")
            if query_id:
                ids.add(str(query_id))
    return ids


def read_mem0_gap_by_instance(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    by_instance_budget: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in read_jsonl(path):
        ratio = row.get("package_oracle_ratio")
        if ratio is None:
            continue
        key = (str(row.get("instance_id")), int(row.get("budget", 0) or 0))
        by_instance_budget[key][str(row.get("method"))] = float(ratio)
    gaps: dict[str, list[float]] = defaultdict(list)
    for (instance_id, _budget), scores in by_instance_budget.items():
        if "actual_mem0_oracle_pruned_upper" not in scores or "actual_mem0_recency_pruned" not in scores:
            continue
        gaps[instance_id].append(
            max(0.0, scores["actual_mem0_oracle_pruned_upper"] - scores["actual_mem0_recency_pruned"])
        )
    return {instance_id: statistics.fmean(values) for instance_id, values in gaps.items() if values}


def select_queries(
    queries: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    disagreement_ids: set[str],
    mem0_gap_by_instance: Mapping[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic stratified subset for adjudication."""

    rng = random.Random(seed)
    eligible = [dict(row) for row in queries if row.get("required_unit_ids")]
    by_id = {str(row["query_id"]): row for row in eligible}
    selected_ids: list[str] = []

    def add(query_id: str) -> None:
        if query_id in by_id and query_id not in selected_ids and len(selected_ids) < limit:
            selected_ids.append(query_id)

    # First include examples where the previous independent annotation disagreed.
    for query_id in sorted(disagreement_ids):
        add(query_id)

    # Then include examples where Mem0 extraction and budget selection diverged.
    for query_id, _gap in sorted(mem0_gap_by_instance.items(), key=lambda item: (-item[1], item[0])):
        add(query_id)

    # Ensure category diversity.
    categories: dict[str, list[str]] = defaultdict(list)
    for row in eligible:
        categories[str(row.get("category", "unknown"))].append(str(row["query_id"]))
    for ids in categories.values():
        rng.shuffle(ids)
    while len(selected_ids) < min(limit, len(eligible)):
        made_progress = False
        for category in sorted(categories):
            while categories[category]:
                query_id = categories[category].pop()
                if query_id not in selected_ids:
                    add(query_id)
                    made_progress = True
                    break
            if len(selected_ids) >= limit:
                break
        if not made_progress:
            break

    # Fill any remaining slots randomly but deterministically.
    remaining = [str(row["query_id"]) for row in eligible if str(row["query_id"]) not in selected_ids]
    rng.shuffle(remaining)
    for query_id in remaining:
        add(query_id)

    return [dict(by_id[query_id]) for query_id in selected_ids]


def unit_rows_for_query(data: PackageData, query_id: str) -> list[dict[str, Any]]:
    rows = list(data.evidence_by_instance.get(query_id, []))
    rows.sort(key=lambda row: str(row.get("unit_id", "")))
    return rows


def candidate_rows_for_query(data: PackageData, query_id: str) -> list[dict[str, Any]]:
    rows = list(data.candidate_rows_by_instance.get(query_id, []))
    rows.sort(
        key=lambda row: (
            int(row.get("time_index", 0) or 0),
            str(row.get("experience_id", "")),
            int(row.get("cost", row.get("cost_tokens", 0)) or 0),
            str(row.get("candidate_id", "")),
        )
    )
    return rows


def compact_experience_rows(data: PackageData, query_id: str, max_words: int) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(data.experiences_by_instance.get(query_id, []), key=lambda item: str(item.get("experience_id", ""))):
        text = str(row.get("text", ""))
        rows.append(
            {
                "experience_id": row.get("experience_id"),
                "source_kind": row.get("source_kind"),
                "timestamp": row.get("timestamp"),
                "text": truncate_words(text, max_words),
            }
        )
    return rows


def adjudication_prompt(
    *,
    query: Mapping[str, Any],
    evidence_units: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    experiences: Sequence[Mapping[str, Any]],
    max_candidate_words: int,
) -> str:
    units = [
        {
            "unit_id": row.get("unit_id"),
            "kind": row.get("kind"),
            "canonical_text": row.get("canonical_text"),
            "primary_required": str(row.get("unit_id")) in set(query.get("required_unit_ids", []) or []),
            "primary_unit_weight": float(row.get("unit_weight", 0.0) or 0.0),
            "source_quotes": [
                truncate_words(str(span.get("text", "")), 80)
                for span in row.get("source_spans", []) or []
                if isinstance(span, Mapping)
            ][:2],
        }
        for row in evidence_units
    ]
    candidates = [
        {
            "candidate_id": row.get("candidate_id"),
            "experience_id": row.get("experience_id"),
            "representation_type": row.get("representation_type"),
            "generator_id": row.get("generator_id", row.get("generator")),
            "cost": int(row.get("cost", row.get("cost_tokens", 1)) or 1),
            "text": truncate_words(str(row.get("serialized") or row.get("text") or ""), max_candidate_words),
        }
        for row in candidate_rows
    ]
    payload = {
        "query_id": query.get("query_id"),
        "question": query.get("question"),
        "gold_answer": query.get("answer"),
        "category": query.get("category"),
        "primary_required_unit_ids": query.get("required_unit_ids", []),
        "primary_annotation_rationale": query.get("annotation_rationale", ""),
        "support_experiences": experiences,
        "evidence_units": units,
        "candidate_memories": candidates,
    }
    return (
        "You are adjudicating an OracleMem natural-trace coverage package.\n"
        "Your job is to produce conservative benchmark labels. Use the question and gold answer only for adjudication.\n"
        "Do not create new evidence unit ids. Select only from the existing evidence_units.\n"
        "First choose the minimal existing evidence_unit ids needed to answer the question exactly.\n"
        "Then map candidate memories to evidence units only when the candidate text entails the unit.\n"
        "Coverage values: 1.0 for complete entailment, 0.5 for partial but useful entailment. Omit unsupported pairs.\n"
        "If the existing units are insufficient, mark status='rejected'. If the answer is ambiguous, mark status='ambiguous'.\n"
        "If the primary labels are basically correct, mark status='accepted'. If you change required units or coverage, mark status='corrected'.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "accepted|corrected|ambiguous|rejected",\n'
        '  "required_unit_ids": ["..."],\n'
        '  "coverage_edges": [\n'
        '    {"candidate_id": "...", "unit_id": "...", "coverage": 1.0, "rationale": "..."}\n'
        "  ],\n"
        '  "confidence": 0.0,\n'
        '  "rationale": "..."\n'
        "}\n\n"
        f"PACKAGE:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def clean_adjudication(
    *,
    parsed: Mapping[str, Any],
    query: Mapping[str, Any],
    evidence_units: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed_units = {str(row.get("unit_id")) for row in evidence_units}
    allowed_candidates = {str(row.get("candidate_id")) for row in candidate_rows}
    primary_required = set(str(unit_id) for unit_id in query.get("required_unit_ids", []) or [])
    status = str(parsed.get("status", "")).strip().lower()
    if status not in {"accepted", "corrected", "ambiguous", "rejected"}:
        status = "corrected"

    required = []
    for unit_id in parsed.get("required_unit_ids", []) or []:
        unit_id = str(unit_id)
        if unit_id in allowed_units and unit_id not in required:
            required.append(unit_id)
    if status in {"accepted", "corrected"} and not required:
        status = "rejected"

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in parsed.get("coverage_edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        candidate_id = str(edge.get("candidate_id", ""))
        unit_id = str(edge.get("unit_id", ""))
        if candidate_id not in allowed_candidates or unit_id not in allowed_units:
            continue
        coverage = max(0.0, min(1.0, float(edge.get("coverage", edge.get("fidelity", 0.0)) or 0.0)))
        if coverage <= 0:
            continue
        key = (candidate_id, unit_id)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(
            {
                "candidate_id": candidate_id,
                "unit_id": unit_id,
                "coverage": coverage,
                "coverage_label": "full" if coverage >= 0.999 else "partial",
                "rationale": str(edge.get("rationale", "")),
            }
        )

    if status == "accepted" and set(required) != primary_required:
        status = "corrected"
    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    return {
        "query_id": str(query.get("query_id")),
        "status": status,
        "required_unit_ids": required,
        "coverage_edges": edges,
        "confidence": confidence,
        "rationale": str(parsed.get("rationale", "")),
        "primary_required_unit_ids": sorted(primary_required),
        "required_changed": sorted(primary_required) != sorted(required),
    }


def export_adjudicated_package(
    *,
    primary_data: PackageData,
    accepted_queries: Sequence[Mapping[str, Any]],
    adjudications: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    adjudicator_model: str,
    primary_package_dir: Path,
) -> None:
    package_dir = out_dir / "coverage_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    accepted_ids = {str(query["query_id"]) for query in accepted_queries}
    experience_rows = [
        row
        for query_id in accepted_ids
        for row in primary_data.experiences_by_instance.get(query_id, [])
    ]
    candidate_rows = [
        row
        for query_id in accepted_ids
        for row in primary_data.candidate_rows_by_instance.get(query_id, [])
    ]

    evidence_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for query in accepted_queries:
        query_id = str(query["query_id"])
        adjudication = adjudications[query_id]
        required = set(str(unit_id) for unit_id in adjudication.get("required_unit_ids", []) or [])
        for row in primary_data.evidence_by_instance.get(query_id, []):
            updated = dict(row)
            updated["unit_weight"] = 1.0 if str(updated.get("unit_id")) in required else 0.0
            updated["adjudication_status"] = "model_adjudicated"
            updated["annotator_ids"] = list(dict.fromkeys([*(updated.get("annotator_ids", []) or []), adjudicator_model]))
            evidence_rows.append(updated)
        updated_query = dict(query)
        updated_query["primary_required_unit_ids"] = list(query.get("required_unit_ids", []) or [])
        updated_query["required_unit_ids"] = sorted(required)
        updated_query["annotation_rationale"] = str(adjudication.get("rationale", ""))
        updated_query["adjudication_status"] = str(adjudication.get("status"))
        updated_query["adjudicator_model"] = adjudicator_model
        query_rows.append(updated_query)
        for edge in adjudication.get("coverage_edges", []) or []:
            coverage_rows.append(
                {
                    "candidate_id": edge["candidate_id"],
                    "unit_id": edge["unit_id"],
                    "coverage": edge["coverage"],
                    "coverage_label": edge["coverage_label"],
                    "rationale": edge["rationale"],
                    "adjudication_status": "model_adjudicated",
                    "annotator_ids": [adjudicator_model],
                    "experience_id": str(edge["candidate_id"]).rsplit("::", 1)[0],
                    "candidate_group": str(edge["candidate_id"]).rsplit("::", 1)[0],
                }
            )
        decision_rows.append(dict(adjudication))

    write_jsonl(package_dir / "experiences.jsonl", experience_rows)
    write_jsonl(package_dir / "evidence_units.jsonl", evidence_rows)
    write_jsonl(package_dir / "queries.jsonl", query_rows)
    write_jsonl(package_dir / "candidate_memories.jsonl", candidate_rows)
    write_jsonl(package_dir / "coverage_matrix.jsonl", coverage_rows)
    write_jsonl(package_dir / "annotation_decisions.jsonl", decision_rows)

    file_hashes = {}
    for name in (
        "experiences.jsonl",
        "evidence_units.jsonl",
        "queries.jsonl",
        "candidate_memories.jsonl",
        "coverage_matrix.jsonl",
        "annotation_decisions.jsonl",
    ):
        file_hashes[name] = stable_hash((package_dir / name).read_text(encoding="utf-8"))

    manifest = {
        "schema_version": 1,
        "package_kind": "natural_adjudicated_subset",
        "primary_package_dir": str(primary_package_dir),
        "adjudicator_model": adjudicator_model,
        "counts": {
            "instances": len(query_rows),
            "experiences": len(experience_rows),
            "evidence_units": len(evidence_rows),
            "candidate_memories": len(candidate_rows),
            "positive_coverage_rows": len(coverage_rows),
            "queries": len(query_rows),
        },
        "allowed_inputs": [
            "primary package support-slice experiences",
            "primary package evidence units and candidates",
            "question and gold answer for adjudication only",
        ],
        "forbidden_inputs_for_candidate_generation": [
            "adjudicated required_unit_ids",
            "adjudicated coverage edges",
            "solver outputs",
        ],
        "limitations": [
            "LLM adjudicated, not human adjudicated",
            "support-sliced, not full-haystack",
            "exact OPT is finite package OPT over copied primary candidates",
        ],
        "file_hashes": file_hashes,
    }
    write_json(package_dir / "candidate_generation_manifest.json", manifest)
    (package_dir / "README.md").write_text(
        "# OracleMem Natural Adjudicated Coverage Package\n\n"
        "This package is a model-adjudicated subset exported from the primary Natural package. "
        "It is intended as a semantic-stability diagnostic, not as human ground truth.\n",
        encoding="utf-8",
    )


def evaluate_package(
    package_dir: Path,
    budgets: Sequence[int],
    methods: Sequence[str],
    out_dir: Path,
    *,
    estimator_model: str,
) -> dict[str, str]:
    data = load_package(package_dir)
    results = []
    for query in data.queries:
        instance = package_instance(data, query)
        results.extend(
            evaluate_instance(
                instance,
                budgets,
                methods=methods,
                estimator_model=estimator_model,
                estimator_profile="gemini_flash_lite_v1",
            )
        )
    return write_benchmark_outputs(results, out_dir)


def write_report(
    *,
    out_dir: Path,
    selected_queries: Sequence[Mapping[str, Any]],
    accepted_queries: Sequence[Mapping[str, Any]],
    rejected_queries: Sequence[Mapping[str, Any]],
    adjudications: Mapping[str, Mapping[str, Any]],
    benchmark_summary_path: Path | None,
    model: str,
    usage_rows: Sequence[Mapping[str, Any]],
) -> None:
    status_counts: dict[str, int] = defaultdict(int)
    changed = 0
    for adj in adjudications.values():
        status_counts[str(adj.get("status", "unknown"))] += 1
        changed += int(bool(adj.get("required_changed")))
    usage_totals: dict[str, float] = defaultdict(float)
    for row in usage_rows:
        usage = row.get("usage", {}) if isinstance(row, Mapping) else {}
        if not isinstance(usage, Mapping):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            try:
                usage_totals[key] += float(usage.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass

    summary = {
        "model": model,
        "attempted": len(selected_queries),
        "accepted_or_corrected": len(accepted_queries),
        "rejected_or_ambiguous": len(rejected_queries),
        "status_counts": dict(sorted(status_counts.items())),
        "required_changed_n": changed,
        "required_changed_rate": changed / max(1, len(adjudications)),
        "usage": dict(sorted(usage_totals.items())),
        "benchmark_summary_path": str(benchmark_summary_path) if benchmark_summary_path else None,
    }
    write_json(out_dir / "adjudication_summary.json", summary)

    lines = [
        "# Natural Package Adjudication Report",
        "",
        f"- Adjudicator model: `{model}`",
        f"- Attempted examples: {summary['attempted']}",
        f"- Accepted/corrected examples exported: {summary['accepted_or_corrected']}",
        f"- Rejected/ambiguous examples: {summary['rejected_or_ambiguous']}",
        f"- Required-unit changed rate: {summary['required_changed_rate']:.3f}",
        f"- API total tokens: {usage_totals.get('total_tokens', 0.0):.0f}",
        f"- API cost reported by OpenRouter: ${usage_totals.get('cost', 0.0):.4f}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: {count}")
    if benchmark_summary_path and benchmark_summary_path.exists():
        benchmark = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Adjudicated Package Scores",
                "",
                "| Budget | Method | N | Mean ratio to exact package OPT | Bootstrap 95% CI |",
                "|---:|---|---:|---:|---|",
            ]
        )
        for row in benchmark.get("by_budget_method", []):
            lines.append(
                "| {budget} | `{method}` | {n} | {ratio:.3f} | [{lo:.3f}, {hi:.3f}] |".format(
                    budget=row.get("budget"),
                    method=row.get("method"),
                    n=row.get("n"),
                    ratio=row.get("mean_ratio_to_opt", float("nan")),
                    lo=row.get("bootstrap95_ratio_to_opt_low", float("nan")),
                    hi=row.get("bootstrap95_ratio_to_opt_high", float("nan")),
                )
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is model adjudication with Gemini Flash, not human ground truth. It is useful as a stricter semantic-stability diagnostic than the primary single-annotator package, but any main-paper claim should still call it model-adjudicated rather than human-adjudicated.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--model", default=DEFAULT_ADJUDICATOR_MODEL)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--budgets", default="30,60,100")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--secondary-agreement-rows", type=Path, default=None)
    parser.add_argument("--mem0-raw-results", type=Path, default=None)
    parser.add_argument("--max-experience-words", type=int, default=900)
    parser.add_argument("--max-candidate-words", type=int, default=220)
    parser.add_argument("--request-sleep", type=float, default=0.02)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    env_values = load_env_file(args.api_env)
    for key, value in env_values.items():
        os.environ.setdefault(key, value)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment or api.env")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_package(args.primary_package_dir)
    disagreement_ids = read_disagreement_ids(args.secondary_agreement_rows)
    mem0_gap_by_instance = read_mem0_gap_by_instance(args.mem0_raw_results)
    selected_queries = select_queries(
        data.queries,
        limit=args.limit,
        disagreement_ids=disagreement_ids,
        mem0_gap_by_instance=mem0_gap_by_instance,
        seed=args.seed,
    )
    write_jsonl(args.out_dir / "selected_queries.jsonl", selected_queries)

    client = OpenRouterJsonClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.model,
        cache_path=args.out_dir / "openrouter_cache_adjudication.json",
        max_tokens=3500,
        request_sleep=args.request_sleep,
    )

    usage_rows: list[dict[str, Any]] = []
    adjudications: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    for index, query in enumerate(selected_queries, start=1):
        query_id = str(query["query_id"])
        marker = args.out_dir / "per_instance" / f"{query_id}.done.json"
        if args.skip_existing and marker.exists():
            cached = json.loads(marker.read_text(encoding="utf-8"))
            adjudications[query_id] = cached["adjudication"]
            continue
        evidence_units = unit_rows_for_query(data, query_id)
        candidate_rows = candidate_rows_for_query(data, query_id)
        experiences = compact_experience_rows(data, query_id, args.max_experience_words)
        started = time.perf_counter()
        response = client(
            adjudication_prompt(
                query=query,
                evidence_units=evidence_units,
                candidate_rows=candidate_rows,
                experiences=experiences,
                max_candidate_words=args.max_candidate_words,
            ),
            purpose="natural_package_adjudication",
        )
        parsed = response.get("parsed", {}) if isinstance(response, Mapping) else {}
        adjudication = clean_adjudication(
            parsed=parsed,
            query=query,
            evidence_units=evidence_units,
            candidate_rows=candidate_rows,
        )
        adjudication.update(
            {
                "model": args.model,
                "prompt_hash": response.get("prompt_hash"),
                "cache_hit": response.get("cache_hit"),
                "runtime_sec": time.perf_counter() - started,
                "selected_index": index,
            }
        )
        adjudications[query_id] = adjudication
        usage_rows.append(
            {
                "query_id": query_id,
                "prompt_hash": response.get("prompt_hash"),
                "usage": response.get("usage", {}),
                "cache_hit": response.get("cache_hit"),
            }
        )
        raw_rows.append(
            {
                "query_id": query_id,
                "response": response,
                "adjudication": adjudication,
            }
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"query_id": query_id, "adjudication": adjudication})

    write_jsonl(args.out_dir / "adjudication_raw.jsonl", raw_rows)
    write_jsonl(args.out_dir / "api_usage.jsonl", usage_rows)
    write_jsonl(args.out_dir / "adjudication_decisions.jsonl", list(adjudications.values()))

    accepted_queries = [
        query
        for query in selected_queries
        if str(adjudications.get(str(query["query_id"]), {}).get("status")) in {"accepted", "corrected"}
    ]
    rejected_queries = [query for query in selected_queries if query not in accepted_queries]
    export_adjudicated_package(
        primary_data=data,
        accepted_queries=accepted_queries,
        adjudications=adjudications,
        out_dir=args.out_dir,
        adjudicator_model=args.model,
        primary_package_dir=args.primary_package_dir,
    )

    budgets = [int(float(item.strip())) for item in args.budgets.split(",") if item.strip()]
    methods = tuple(args.methods.replace(",", " ").split())
    benchmark_paths: dict[str, str] | None = None
    if accepted_queries:
        benchmark_paths = evaluate_package(
            args.out_dir / "coverage_package",
            budgets,
            methods,
            args.out_dir,
            estimator_model=args.model,
        )

    write_report(
        out_dir=args.out_dir,
        selected_queries=selected_queries,
        accepted_queries=accepted_queries,
        rejected_queries=rejected_queries,
        adjudications=adjudications,
        benchmark_summary_path=Path(benchmark_paths["summary_json"]) if benchmark_paths else None,
        model=args.model,
        usage_rows=usage_rows,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "attempted": len(selected_queries),
                "accepted_or_corrected": len(accepted_queries),
                "rejected_or_ambiguous": len(rejected_queries),
                "model": args.model,
                "benchmark_summary": benchmark_paths["summary_json"] if benchmark_paths else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
