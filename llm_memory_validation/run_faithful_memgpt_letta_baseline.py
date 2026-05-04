"""Run a faithful no-API MemGPT/Letta-style writer on a coverage package.

This runner is the practical fallback for environments where the checked-out
Letta server/API stack is too heavy to execute locally. It does not call an LLM
or run a Letta server. Instead, it simulates the MemGPT/Letta memory architecture
over an exported OracleMem coverage package:

* core memory stores compact, durable facts/preferences/updates;
* archival memory stores longer summaries or structured notes;
* query-time retrieval can use recency or lexical archival search.

The written memories are package-derived candidate texts selected only from
visible metadata. Their audited package coverage is inherited for scoring, so
the resulting store can be scored without new API calls. Primary ratios use the
same finite union denominator convention as the Mem0/A-Mem rescoring scripts:
exact OPT over package candidates plus MemGPT/Letta-written memories.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import CandidateMemory, OracleMemInstance, objective_value, solve_exact

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


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_:-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
    "you",
}

EXCLUDED_TYPES = {"do_not_store"}
CORE_TYPES = {
    "abstain",
    "abstention",
    "atomic_fact",
    "commitment",
    "compound_update",
    "deadline",
    "disambiguation",
    "fact",
    "interval_fact",
    "preference",
    "procedural_preference",
    "scheduled_event",
    "skill",
    "task_state",
    "temporal_fact",
    "temporal_validity",
    "tombstone",
    "tool_result",
    "uncertainty",
}
ARCHIVAL_TYPES = {"compound_evidence", "graph_edge", "raw", "raw_span", "summary"}

TYPE_PRIOR = {
    "compound_update": 1.42,
    "tombstone": 1.36,
    "procedural_preference": 1.30,
    "task_state": 1.26,
    "temporal_validity": 1.24,
    "atomic_fact": 1.22,
    "fact": 1.22,
    "temporal_fact": 1.20,
    "scheduled_event": 1.18,
    "deadline": 1.18,
    "commitment": 1.16,
    "skill": 1.12,
    "tool_result": 1.10,
    "abstain": 1.08,
    "abstention": 1.08,
    "uncertainty": 1.08,
    "disambiguation": 1.08,
    "summary": 1.04,
    "compound_evidence": 1.02,
    "graph_edge": 0.98,
    "raw_span": 0.66,
    "raw": 0.66,
}

GENERATOR_PRIOR = {
    "gemini_memgpt": 1.14,
    "human_edited": 1.10,
    "gemini_validity": 1.08,
    "gemini_mem0": 1.02,
    "gemini_amem": 0.98,
    "longmemeval_raw": 0.78,
}

SALIENT_TERMS = {
    "actually",
    "changed",
    "current",
    "default",
    "deadline",
    "except",
    "invalid",
    "mild",
    "not",
    "now",
    "prefer",
    "preference",
    "remember",
    "scheduled",
    "stop",
    "superseded",
    "update",
    "unless",
}


@dataclass(frozen=True)
class WrittenMemory:
    candidate: CandidateMemory
    source_candidate_id: str
    source_experience_id: str
    tier: str
    write_reason: str
    visible_score: float
    source_representation_type: str
    source_generator: str


def parse_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in value.replace(",", " ").split() if token)


def parse_budgets(value: str) -> list[int]:
    return [int(float(token)) for token in parse_tokens(value)]


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


def tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(str(text).lower()) if token not in STOPWORDS}


def word_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


def source_generator(candidate: CandidateMemory) -> str:
    return str(candidate.generator or "")


def candidate_tier(candidate: CandidateMemory) -> str:
    representation_type = str(candidate.representation_type)
    if representation_type in CORE_TYPES:
        return "core"
    if representation_type in ARCHIVAL_TYPES:
        if representation_type in {"raw", "raw_span"}:
            return "recall"
        return "archival"
    return "archival"


def visible_write_score(candidate: CandidateMemory, universe: Sequence[CandidateMemory]) -> float:
    representation_type = str(candidate.representation_type)
    if representation_type in EXCLUDED_TYPES:
        return -1.0
    text_tokens = tokens(candidate.serialized)
    salient_hits = len(text_tokens & SALIENT_TERMS)
    recency = recency_score(candidate, universe)
    confidence = max(0.0, min(1.25, float(candidate.confidence or 1.0)))
    type_prior = TYPE_PRIOR.get(representation_type, 1.0)
    generator_prior = GENERATOR_PRIOR.get(source_generator(candidate), 1.0)
    compactness = 1.0 / (max(1.0, float(candidate.cost)) ** 0.18)
    salient_bonus = 1.0 + min(0.25, 0.04 * salient_hits)
    return type_prior * generator_prior * confidence * salient_bonus * (0.85 + 0.30 * recency) * compactness


def recency_score(candidate: CandidateMemory, universe: Sequence[CandidateMemory]) -> float:
    if not universe:
        return 0.0
    min_time = min(item.time_index for item in universe)
    max_time = max(item.time_index for item in universe)
    if max_time <= min_time:
        return 1.0
    return (candidate.time_index - min_time) / max(1.0, max_time - min_time)


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / math.sqrt(len(left_tokens) * len(right_tokens))


def group_candidates(candidates: Sequence[CandidateMemory]) -> list[list[CandidateMemory]]:
    groups: dict[str, list[CandidateMemory]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate.experience_id)].append(candidate)
    return [
        sorted(group, key=lambda item: (item.cost, item.candidate_id))
        for _experience_id, group in sorted(
            groups.items(),
            key=lambda item: (
                min(candidate.time_index for candidate in item[1]),
                item[0],
            ),
        )
    ]


def dedupe_key(candidate: CandidateMemory) -> tuple[str, str]:
    compact = " ".join(sorted(tokens(candidate.serialized))[:24])
    return (str(candidate.representation_type), compact)


def add_written_memory(
    written: list[WrittenMemory],
    *,
    source: CandidateMemory,
    instance_id: str,
    memory_index: int,
    tier: str,
    reason: str,
    visible_score: float,
) -> int:
    memory_id = f"{instance_id}::faithful_memgpt_letta::{memory_index:04d}"
    candidate = CandidateMemory(
        candidate_id=memory_id,
        # External written memories are independent memories, matching the
        # Mem0/A-Mem union-denominator convention.
        experience_id=memory_id,
        representation_type=f"faithful_memgpt_letta_{tier}",
        serialized=source.serialized,
        cost=max(1, int(source.cost or word_count(source.serialized) or 1)),
        coverage=dict(source.coverage),
        time_index=source.time_index,
        generator="faithful_memgpt_letta_noapi",
        confidence=visible_score,
        estimated_value=visible_score,
        estimator_model="visible_metadata_memgpt_letta_v1",
    )
    written.append(
        WrittenMemory(
            candidate=candidate,
            source_candidate_id=source.candidate_id,
            source_experience_id=source.experience_id,
            tier=tier,
            write_reason=reason,
            visible_score=visible_score,
            source_representation_type=source.representation_type,
            source_generator=source_generator(source),
        )
    )
    return memory_index + 1


def build_faithful_store(
    package: OracleMemInstance,
    *,
    max_core_per_experience: int,
    max_archival_per_experience: int,
    include_recall_raw: bool,
    max_recall_per_instance: int,
) -> list[WrittenMemory]:
    """Build a package-derived core/archival store without oracle labels."""

    written: list[WrittenMemory] = []
    memory_index = 0
    seen: set[tuple[str, str]] = set()
    raw_candidates: list[tuple[float, CandidateMemory]] = []

    for group in group_candidates(package.candidates):
        ranked = sorted(
            [
                (visible_write_score(candidate, package.candidates), candidate)
                for candidate in group
                if str(candidate.representation_type) not in EXCLUDED_TYPES
            ],
            key=lambda item: (
                item[0],
                GENERATOR_PRIOR.get(source_generator(item[1]), 1.0),
                -item[1].cost,
                item[1].candidate_id,
            ),
            reverse=True,
        )
        core_added = 0
        archival_added = 0
        for score, candidate in ranked:
            if score <= 0:
                continue
            tier = candidate_tier(candidate)
            if tier == "recall":
                raw_candidates.append((score, candidate))
                continue
            if tier == "core":
                if core_added >= max_core_per_experience:
                    continue
                reason = "core_memory_visible_fact_or_update"
            else:
                if archival_added >= max_archival_per_experience:
                    continue
                reason = "archival_memory_visible_summary_or_note"
            key = dedupe_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            memory_index = add_written_memory(
                written,
                source=candidate,
                instance_id=package.instance_id,
                memory_index=memory_index,
                tier=tier,
                reason=reason,
                visible_score=score,
            )
            if tier == "core":
                core_added += 1
            else:
                archival_added += 1

    if include_recall_raw and max_recall_per_instance > 0:
        for score, candidate in sorted(
            raw_candidates,
            key=lambda item: (
                item[1].time_index,
                item[0],
                -item[1].cost,
                item[1].candidate_id,
            ),
            reverse=True,
        )[:max_recall_per_instance]:
            key = dedupe_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            memory_index = add_written_memory(
                written,
                source=candidate,
                instance_id=package.instance_id,
                memory_index=memory_index,
                tier="recall",
                reason="recall_memory_recent_raw_context",
                visible_score=score,
            )

    written.sort(key=lambda item: (item.candidate.time_index, item.candidate.candidate_id))
    return written


def union_instance(package: OracleMemInstance, written: Sequence[CandidateMemory]) -> OracleMemInstance:
    return OracleMemInstance(
        instance_id=f"{package.instance_id}::package_plus_memgpt_letta",
        candidates=tuple(package.candidates) + tuple(written),
        unit_weights=package.unit_weights,
        seed=package.seed,
        current_units=package.current_units,
        invalidation_units=package.invalidation_units,
        stale_units=package.stale_units,
    )


def select_archival_search_pruned(
    candidates: Sequence[CandidateMemory],
    query: Mapping[str, Any],
    budget: int,
) -> list[CandidateMemory]:
    question = str(query.get("question", ""))
    selected: list[CandidateMemory] = []
    used = 0
    if not candidates:
        return selected
    min_time = min(candidate.time_index for candidate in candidates)
    max_time = max(candidate.time_index for candidate in candidates)
    span = max(1, max_time - min_time)

    def score(candidate: CandidateMemory) -> float:
        tier = str(candidate.representation_type).removeprefix("faithful_memgpt_letta_")
        tier_bonus = {"core": 0.18, "archival": 0.08, "recall": 0.02}.get(tier, 0.0)
        recency = (candidate.time_index - min_time) / span
        visible = max(0.0, min(2.0, float(candidate.estimated_value or candidate.confidence or 0.0)))
        return (
            2.0 * lexical_similarity(question, candidate.serialized)
            + 0.26 * visible
            + tier_bonus
            + 0.08 * recency
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            score(item) / (max(1.0, float(item.cost)) ** 0.25),
            score(item),
            -item.cost,
            item.time_index,
            item.candidate_id,
        ),
        reverse=True,
    )
    for candidate in ranked:
        if used + candidate.cost > budget:
            continue
        selected.append(candidate)
        used += candidate.cost
    selected.sort(key=lambda item: (item.time_index, item.candidate_id))
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
    written: Sequence[WrittenMemory],
) -> dict[str, Any]:
    value = objective_value(selected, package.unit_weights)
    tier_mix = Counter(
        str(candidate.representation_type).removeprefix("faithful_memgpt_letta_")
        for candidate in selected
    )
    return {
        "instance_id": instance_id,
        "budget": budget,
        "method": method,
        "objective_value": value,
        "package_candidate_exact_opt": package_denominator,
        "package_plus_memgpt_letta_exact_opt": union_denominator,
        "ratio_to_package_candidate_opt": value / package_denominator if package_denominator > 0 else None,
        "ratio_to_union_opt": value / union_denominator if union_denominator > 0 else None,
        "selected_cost": sum(candidate.cost for candidate in selected),
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "selected_memory_texts": [candidate.serialized for candidate in selected],
        "selected_tier_mix": dict(sorted(tier_mix.items())),
        "written_memory_count": len(written),
        "written_store_cost": sum(item.candidate.cost for item in written),
        "denominator_label": "package_plus_memgpt_letta_exact_opt",
        "runtime_sec": runtime_sec,
    }


def written_store_row(
    *,
    query: Mapping[str, Any],
    written: Sequence[WrittenMemory],
) -> dict[str, Any]:
    return {
        "instance_id": str(query.get("query_id")),
        "question": query.get("question"),
        "answer": query.get("answer"),
        "memories": [
            {
                "memory_id": item.candidate.candidate_id,
                "tier": item.tier,
                "text": item.candidate.serialized,
                "cost": item.candidate.cost,
                "time_index": item.candidate.time_index,
                "source_candidate_id": item.source_candidate_id,
                "source_experience_id": item.source_experience_id,
                "source_representation_type": item.source_representation_type,
                "source_generator": item.source_generator,
                "visible_score": item.visible_score,
                "write_reason": item.write_reason,
            }
            for item in written
        ],
        "memory_count": len(written),
        "store_cost": sum(item.candidate.cost for item in written),
    }


def summarize(rows: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["budget"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (method, budget), items in sorted(grouped.items()):
        union_ratios = [row.get("ratio_to_union_opt") for row in items]
        package_ratios = [row.get("ratio_to_package_candidate_opt") for row in items]
        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(items),
                "ratio_defined_n": len([value for value in union_ratios if value is not None]),
                "zero_denominator_n": sum(
                    1
                    for row in items
                    if float(row.get("package_plus_memgpt_letta_exact_opt", 0.0) or 0.0) <= 1e-12
                ),
                "mean_ratio_to_union_opt": mean(union_ratios),
                "std_ratio_to_union_opt": stdev(union_ratios),
                "mean_ratio_to_package_candidate_opt": mean(package_ratios),
                "std_ratio_to_package_candidate_opt": stdev(package_ratios),
                "mean_objective_value": mean([row.get("objective_value") for row in items]),
                "mean_selected_cost": mean([row.get("selected_cost") for row in items]),
                "mean_written_memory_count": mean([row.get("written_memory_count") for row in items]),
                "mean_written_store_cost": mean([row.get("written_store_cost") for row in items]),
                "mean_package_candidate_exact_opt": mean([row.get("package_candidate_exact_opt") for row in items]),
                "mean_package_plus_memgpt_letta_exact_opt": mean(
                    [row.get("package_plus_memgpt_letta_exact_opt") for row in items]
                ),
            }
        )
    return {
        "by_method_budget": summary_rows,
        "result_rows": len(rows),
        "skipped_rows": len(skipped),
        "skipped": list(skipped),
    }


def actual_letta_probe() -> dict[str, Any]:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, r'external_repos/letta'); "
            "import letta; "
            "print(getattr(letta, '__version__', 'unknown'))"
        ),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {
            "status": "probe_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime_sec": time.perf_counter() - started,
        }
    return {
        "status": "importable" if completed.returncode == 0 else "not_importable",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-2000:],
        "stderr": completed.stderr.strip()[-4000:],
        "runtime_sec": time.perf_counter() - started,
    }


def letta_git_info() -> dict[str, Any]:
    repo = ROOT / "external_repos" / "letta"
    if not repo.exists():
        return {"repo": str(repo), "exists": False}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"repo": str(repo), "exists": True, "error": str(exc)}
    version = None
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                version = line.split("=", 1)[1].strip().strip('"')
                break
    return {
        "repo": "external_repos/letta",
        "exists": True,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "version": version,
        "rev_parse_stderr": commit.stderr.strip() if commit.returncode != 0 else "",
    }


def write_report(
    out_dir: Path,
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    rows = list(summary.get("by_method_budget", []) or [])
    budgets = sorted({int(row["budget"]) for row in rows})
    methods = [
        "faithful_memgpt_letta_archival_search_pruned",
        "faithful_memgpt_letta_recency_pruned",
        "faithful_memgpt_letta_oracle_pruned_upper",
    ]
    by_key = {(str(row["method"]), int(row["budget"])): row for row in rows}

    lines = [
        "# Faithful MemGPT/Letta OracleMem Baseline",
        "",
        f"- Package: `{manifest['package_dir']}`",
        f"- Queries evaluated: {manifest['completed_instances']} / {manifest['query_count']}",
        f"- Budgets: `{','.join(str(budget) for budget in manifest['budgets'])}`",
        "- API calls: 0.",
        "- Denominator: exact finite union OPT over package candidates plus faithful MemGPT/Letta-written memories.",
        "- System status: no-API faithful fallback, not a Letta server/API run.",
        f"- Letta checkout: `{manifest.get('letta_git', {}).get('repo', 'external_repos/letta')}` commit `{manifest.get('letta_git', {}).get('commit')}` version `{manifest.get('letta_git', {}).get('version')}`.",
    ]
    probe = manifest.get("actual_letta_probe")
    if isinstance(probe, Mapping):
        lines.append(f"- Actual Letta import probe: `{probe.get('status')}`.")
    lines.extend(
        [
            "",
            "## Mean Ratio To Union OPT",
            "",
            "| Method | " + " | ".join(f"B={budget}" for budget in budgets) + " |",
            "| --- | " + " | ".join("---:" for _ in budgets) + " |",
        ]
    )
    for method in methods:
        cells = []
        for budget in budgets:
            value = (by_key.get((method, budget)) or {}).get("mean_ratio_to_union_opt")
            cells.append("--" if value is None else f"{float(value):.3f}")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- The writer selects from exported package candidate texts using visible metadata only: representation type, generator label, confidence, cost, recency, and text tokens.",
            "- Coverage labels are inherited only after writing, for no-API scoring; they are not used for admission or retrieval except in the analysis-only oracle upper row.",
            "- `faithful_memgpt_letta_archival_search_pruned` is the main non-oracle query-time policy; it ranks written core/archival memories by lexical query overlap, visible score, tier, and recency.",
            "- `faithful_memgpt_letta_recency_pruned` is a native recency/context diagnostic.",
            "- `faithful_memgpt_letta_oracle_pruned_upper` uses hidden coverage and should be treated as an upper bound on the value present in the written store.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="30,60,100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solver", default="exact_stdlib")
    parser.add_argument("--max-core-per-experience", type=int, default=2)
    parser.add_argument("--max-archival-per-experience", type=int, default=1)
    parser.add_argument("--include-recall-raw", action="store_true")
    parser.add_argument("--max-recall-per-instance", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budgets = parse_budgets(args.budgets)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data: PackageData = load_package(args.package_dir)
    queries = resolved_queries(data, args.limit)
    result_rows: list[dict[str, Any]] = []
    written_store_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for query in queries:
        instance_id = str(query["query_id"])
        started = time.perf_counter()
        package = package_instance(data, query)
        if not package.candidates:
            skipped_rows.append({"instance_id": instance_id, "reason": "no_package_candidates"})
            continue
        written = build_faithful_store(
            package,
            max_core_per_experience=args.max_core_per_experience,
            max_archival_per_experience=args.max_archival_per_experience,
            include_recall_raw=args.include_recall_raw,
            max_recall_per_instance=args.max_recall_per_instance,
        )
        if not written:
            skipped_rows.append({"instance_id": instance_id, "reason": "no_written_memories"})
            continue

        written_candidates = [item.candidate for item in written]
        written_store_rows.append(written_store_row(query=query, written=written))
        union = union_instance(package, written_candidates)

        for budget in budgets:
            package_exact = solve_exact(package, budget, solver=args.solver)
            union_exact = solve_exact(union, budget, solver=args.solver)
            selectors = {
                "faithful_memgpt_letta_archival_search_pruned": select_archival_search_pruned(
                    written_candidates,
                    query,
                    budget,
                ),
                "faithful_memgpt_letta_recency_pruned": select_recency_pruned(written_candidates, budget),
                "faithful_memgpt_letta_oracle_pruned_upper": select_oracle_density_pruned(
                    written_candidates,
                    budget,
                    package.unit_weights,
                ),
            }
            for method, selected in selectors.items():
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
                        written=written,
                    )
                )

    write_jsonl(args.out_dir / "raw_results.jsonl", result_rows)
    write_jsonl(args.out_dir / "written_stores.jsonl", written_store_rows)
    write_jsonl(args.out_dir / "skipped_instances.jsonl", skipped_rows)

    summary = summarize(result_rows, skipped_rows)
    completed_instances = len({row["instance_id"] for row in result_rows})
    manifest = {
        "package_dir": str(args.package_dir),
        "out_dir": str(args.out_dir),
        "query_count": len(queries),
        "completed_instances": completed_instances,
        "skipped_instances": len(skipped_rows),
        "budgets": budgets,
        "limit": args.limit,
        "solver": args.solver,
        "max_core_per_experience": args.max_core_per_experience,
        "max_archival_per_experience": args.max_archival_per_experience,
        "include_recall_raw": args.include_recall_raw,
        "max_recall_per_instance": args.max_recall_per_instance,
        "api_calls": 0,
        "denominator": "package_plus_memgpt_letta_exact_opt",
        "runner": "llm_memory_validation/run_faithful_memgpt_letta_baseline.py",
        "claim_status": "faithful_noapi_package_derived_memgpt_letta_writer",
        "command": " ".join(sys.argv),
        "letta_git": letta_git_info(),
        "actual_letta_probe": actual_letta_probe(),
        "result_rows": len(result_rows),
        "written_store_rows": len(written_store_rows),
        "artifacts": {
            "REPORT.md": str(args.out_dir / "REPORT.md"),
            "summary.json": str(args.out_dir / "summary.json"),
            "raw_results.jsonl": str(args.out_dir / "raw_results.jsonl"),
            "run_manifest.json": str(args.out_dir / "run_manifest.json"),
            "written_stores.jsonl": str(args.out_dir / "written_stores.jsonl"),
        },
    }
    summary = {
        "package_dir": str(args.package_dir),
        "attempted_instances": len(queries),
        "completed_instances": completed_instances,
        "skipped_instances": len(skipped_rows),
        "budgets": budgets,
        "denominator_label": "package_plus_memgpt_letta_exact_opt",
        **summary,
    }
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "run_manifest.json", manifest)
    write_report(args.out_dir, summary=summary, manifest=manifest)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "results": len(result_rows),
                "completed_instances": completed_instances,
                "skipped": len(skipped_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
