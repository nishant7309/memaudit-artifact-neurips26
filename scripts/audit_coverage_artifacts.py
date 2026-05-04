"""Audit OracleMem artifacts for evidence-unit and coverage completeness.

This is a lightweight structural audit. It does not judge annotation quality;
it checks whether artifacts expose the fields needed for a non-synthetic
OracleMem coverage package:

* evidence units;
* query-to-unit requirements;
* candidate memories with costs and groups;
* candidate-by-unit coverage rows.

Existing LongMemEval artifacts are expected to be session-evidence diagnostics,
not full coverage matrices. The report makes that boundary explicit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        "oraclemem_exact_500_results",
        "oraclemem_runs/exact_500/raw_results.jsonl",
        "Synthetic exact result rows; selected ids and metrics, not full matrix.",
    ),
    (
        "oraclemem_stress_exact_500_results",
        "oraclemem_runs/stress_exact_500/raw_results.jsonl",
        "Synthetic stress result rows; selected ids and metrics, not full matrix.",
    ),
    (
        "oraclemem_decomp_det_300_summary",
        "oraclemem_runs/decomp_det_300/summary.json",
        "Synthetic deterministic decomposition summary.",
    ),
    (
        "longmemeval_retrieval_rows",
        "llm_memory_validation/competitor_run_v2/retrieval_rows.json",
        "LongMemEval-S retrieval rows with coarse gold session ids.",
    ),
    (
        "longmemeval_focus_summary",
        "llm_memory_validation/longmemeval_focus_report_core4/summary.json",
        "LongMemEval-S retrieval report summary.",
    ),
    (
        "gpt55_reader_outputs",
        "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/reader_outputs.jsonl",
        "Frozen-context GPT-5.5 reader outputs.",
    ),
    (
        "gpt55_error_audit_rows",
        "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/error_audit_rows.jsonl",
        "Reader error audit rows.",
    ),
    (
        "gpt55_reader_semantic_sample",
        "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/semantic_audit_sample_50.jsonl",
        "Reader semantic audit sample.",
    ),
    (
        "gpt55_scoring_audit_summary",
        "llm_memory_validation/scoring_audit_gpt55/normalized_scoring_v2.json",
        "Normalized scoring audit summary.",
    ),
    (
        "gpt55_scoring_audit_sample",
        "llm_memory_validation/scoring_audit_gpt55/semantic_audit_sample_50.jsonl",
        "Balanced human/judge scoring audit sample.",
    ),
)


UNIT_KEYS = {"unit_id", "evidence_unit_id"}
UNIT_CONTAINER_KEYS = {"units", "evidence_units"}
QUERY_ID_KEYS = {"query_id", "question_id"}
UNIT_REQUIREMENT_KEYS = {"required_unit_ids", "required_units"}
SESSION_REQUIREMENT_KEYS = {"answer_session_ids", "gold_session_ids"}
CANDIDATE_KEYS = {"candidate_id"}
SELECTED_CANDIDATE_KEYS = {"selected_candidate_ids"}
CANDIDATE_DETAIL_KEYS = {
    "candidate_id",
    "experience_id",
    "candidate_group",
    "representation_type",
    "representation",
    "text",
    "serialized",
    "cost",
    "cost_tokens",
}
COVERAGE_KEYS = {"coverage", "coverage_label", "fidelity"}
COVERAGE_SUMMARY_KEYS = {
    "update_metrics",
    "retrieval_metrics",
    "retrieval_summary",
    "gold_recall_in_context",
    "evidence_use",
    "support_in_context",
    "retrieved_at_5",
}
CONTEXT_KEYS = {"context_session_ids", "used_memory_ids", "retrieved_memories"}
PACKAGE_FILE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("manifest", ("candidate_generation_manifest.json", "manifest.json")),
    ("evidence_units", ("evidence_units.jsonl",)),
    ("queries", ("queries.jsonl",)),
    ("experiences", ("experiences.jsonl",)),
    ("candidate_memories", ("candidate_memories.jsonl",)),
    ("coverage_matrix", ("coverage_matrix.jsonl",)),
    ("annotation_decisions", ("annotation_decisions.jsonl",)),
)


@dataclass
class ArtifactAudit:
    name: str
    path: str
    role: str
    exists: bool
    format: str = "missing"
    row_count: int = 0
    sampled_rows: int = 0
    key_counts: dict[str, int] = field(default_factory=dict)
    signals: dict[str, bool] = field(default_factory=dict)
    completeness: dict[str, float] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    package_files: dict[str, bool] = field(default_factory=dict)


def _load_json(path: Path, sample_rows: int) -> tuple[str, int, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    row_count = 0

    if isinstance(data, list):
        row_count = len(data)
        records = [row for row in data[:sample_rows] if isinstance(row, dict)]
        return "json_list", row_count, records

    if isinstance(data, dict):
        list_values = {
            key: value
            for key, value in data.items()
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value[:5])
        }
        if list_values:
            for key, rows in list_values.items():
                row_count += len(rows)
                remaining = max(0, sample_rows - len(records))
                if remaining:
                    for row in rows[:remaining]:
                        item = dict(row)
                        item["_container_key"] = key
                        records.append(item)
            return "json_dict_of_lists", row_count, records

        records = [data]
        return "json_object", 1, records

    return type(data).__name__, 1, []


def _load_jsonl(path: Path, sample_rows: int) -> tuple[str, int, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row_count += 1
            if len(records) >= sample_rows:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at line {line_number}: {exc}") from exc
            if isinstance(row, dict):
                records.append(row)
    return "jsonl", row_count, records


def _load_records(path: Path, sample_rows: int) -> tuple[str, int, list[dict[str, Any]]]:
    if path.is_dir():
        records: list[dict[str, Any]] = []
        row_count = 0
        per_file_sample = max(1, sample_rows // max(1, len(PACKAGE_FILE_ALIASES)))
        for label, child_names in PACKAGE_FILE_ALIASES:
            child = next((path / child_name for child_name in child_names if (path / child_name).exists()), None)
            if child is None:
                continue
            if len(records) >= sample_rows:
                remaining = per_file_sample
            else:
                remaining = max(per_file_sample, sample_rows - len(records))
            child_format, child_count, child_records = _load_records(child, remaining)
            row_count += child_count
            for record in child_records[:remaining]:
                item = dict(record)
                item["_container_key"] = label
                item["_container_file"] = child.name
                item["_container_format"] = child_format
                records.append(item)
        return "coverage_package_dir", row_count, records

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path, sample_rows)
    if suffix == ".json":
        return _load_json(path, sample_rows)
    if suffix == ".md":
        return "markdown", 1, [{"text": path.read_text(encoding="utf-8", errors="replace")}]
    return suffix.lstrip(".") or "unknown", 0, []


def _package_file_presence(path: Path) -> dict[str, bool]:
    if not path.is_dir():
        return {}
    return {
        label: any((path / child_name).exists() for child_name in child_names)
        for label, child_names in PACKAGE_FILE_ALIASES
    }


def _all_keys(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for key in record:
            counts[str(key)] += 1
    return counts


def _has_any_key(records: list[dict[str, Any]], keys: set[str]) -> bool:
    return any(any(key in record for key in keys) for record in records)


def _has_nested_container(records: list[dict[str, Any]], keys: set[str]) -> bool:
    for record in records:
        for key in keys:
            value = record.get(key)
            if isinstance(value, list) and value:
                return True
            if isinstance(value, dict) and value:
                return True
    return False


def _fraction_with(records: list[dict[str, Any]], keys: set[str]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if any(key in record for key in keys)) / len(records)


def _candidate_detail_signal(records: list[dict[str, Any]]) -> bool:
    for record in records:
        if not any(key in record for key in CANDIDATE_KEYS):
            continue
        has_experience = "experience_id" in record
        has_rep = "representation_type" in record or "representation" in record
        has_text = "text" in record or "serialized" in record
        has_cost = "cost" in record or "cost_tokens" in record
        if has_experience and has_rep and has_text and has_cost:
            return True
    return False


def _coverage_matrix_signal(records: list[dict[str, Any]]) -> bool:
    for record in records:
        if "coverage" in record and (
            "candidate_id" in record
            or "experience_id" in record
            or "unit_id" in record
            or "evidence_unit_id" in record
        ):
            return True
        if "candidate_id" in record and ("unit_id" in record or "evidence_unit_id" in record):
            if any(key in record for key in COVERAGE_KEYS):
                return True
    return False


def _invalid_coverage_rows(records: list[dict[str, Any]]) -> int:
    invalid = 0
    for record in records:
        if "coverage" not in record:
            continue
        coverage = record["coverage"]
        values: list[Any] = []
        if isinstance(coverage, dict):
            values.extend(coverage.values())
        elif isinstance(coverage, list):
            for item in coverage:
                if isinstance(item, dict):
                    values.append(item.get("coverage", item.get("fidelity")))
                else:
                    values.append(item)
        else:
            values.append(coverage)
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if numeric < 0.0 or numeric > 1.0:
                invalid += 1
    return invalid


def _status_and_gaps(signals: dict[str, bool], invalid_coverage_rows: int) -> tuple[dict[str, str], list[str]]:
    statuses: dict[str, str] = {}
    gaps: list[str] = []

    if signals["unit_level_evidence"]:
        statuses["evidence_units"] = "unit-level present"
    elif signals["session_level_evidence"]:
        statuses["evidence_units"] = "session-level only"
        gaps.append("No explicit evidence_units/unit_id records; evidence is coarse session id support.")
    else:
        statuses["evidence_units"] = "missing"
        gaps.append("No evidence-unit or session-level gold evidence ids detected.")

    if signals["unit_query_requirements"]:
        statuses["query_requirements"] = "unit requirements present"
    elif signals["session_query_requirements"]:
        statuses["query_requirements"] = "session ids only"
        gaps.append("No required_unit_ids detected for query-level semantic coverage.")
    else:
        statuses["query_requirements"] = "missing"
        gaps.append("No query requirement fields detected.")

    if signals["candidate_details"]:
        statuses["candidate_memories"] = "candidate records present"
    elif signals["selected_candidate_ids"]:
        statuses["candidate_memories"] = "selected ids only"
        gaps.append("Selected candidate ids are present, but candidate texts/costs/coverage are not serialized here.")
    elif signals["retrieved_context"]:
        statuses["candidate_memories"] = "retrieved context only"
        gaps.append("Retrieved/context memories are present, but not finite write-choice candidates.")
    else:
        statuses["candidate_memories"] = "missing"
        gaps.append("No candidate-memory records detected.")

    if signals["coverage_matrix"]:
        statuses["coverage_matrix"] = "candidate-unit coverage present"
    elif signals["coverage_summaries"]:
        statuses["coverage_matrix"] = "summary metrics only"
        gaps.append("Coverage/evidence appears only as aggregate or reader/retrieval metrics.")
    else:
        statuses["coverage_matrix"] = "missing"
        gaps.append("No candidate-by-evidence coverage matrix detected.")

    if invalid_coverage_rows:
        gaps.append(f"Sample contains {invalid_coverage_rows} coverage values outside [0, 1] or nonnumeric.")

    full_ready = (
        signals["unit_level_evidence"]
        and signals["unit_query_requirements"]
        and signals["candidate_details"]
        and signals["coverage_matrix"]
        and invalid_coverage_rows == 0
    )
    synthetic_result_only = signals["exact_ratio"] and signals["selected_candidate_ids"] and not signals["coverage_matrix"]
    session_diagnostic = signals["session_level_evidence"] and not signals["unit_query_requirements"]

    if full_ready:
        statuses["oracle_denominator"] = "machine-checkable coverage package"
    elif synthetic_result_only:
        statuses["oracle_denominator"] = "result rows only; regenerate/serialize hidden synthetic instance for matrix audit"
    elif session_diagnostic:
        statuses["oracle_denominator"] = "not exact OracleMem; session-evidence diagnostic"
    else:
        statuses["oracle_denominator"] = "not coverage-ready"

    return statuses, gaps


def audit_artifact(name: str, path: Path, role: str, sample_rows: int) -> ArtifactAudit:
    audit = ArtifactAudit(name=name, path=str(path), role=role, exists=path.exists())
    if not path.exists():
        audit.errors.append("missing path")
        audit.gaps.append("Artifact path does not exist.")
        return audit

    audit.package_files = _package_file_presence(path)
    try:
        fmt, row_count, records = _load_records(path, sample_rows)
    except Exception as exc:
        audit.errors.append(str(exc))
        audit.gaps.append("Could not parse artifact.")
        return audit

    key_counts = _all_keys(records)
    invalid_coverage_rows = _invalid_coverage_rows(records)
    signals = {
        "unit_level_evidence": _has_any_key(records, UNIT_KEYS)
        or _has_nested_container(records, UNIT_CONTAINER_KEYS),
        "session_level_evidence": _has_any_key(records, SESSION_REQUIREMENT_KEYS),
        "unit_query_requirements": _has_any_key(records, UNIT_REQUIREMENT_KEYS),
        "session_query_requirements": _has_any_key(records, SESSION_REQUIREMENT_KEYS),
        "candidate_details": _candidate_detail_signal(records),
        "selected_candidate_ids": _has_any_key(records, SELECTED_CANDIDATE_KEYS),
        "coverage_matrix": _coverage_matrix_signal(records),
        "coverage_summaries": _has_any_key(records, COVERAGE_SUMMARY_KEYS),
        "retrieved_context": _has_any_key(records, CONTEXT_KEYS),
        "exact_ratio": _has_any_key(records, {"ratio_to_opt", "denominator_label", "optimum_value"}),
    }
    statuses, gaps = _status_and_gaps(signals, invalid_coverage_rows)
    if audit.package_files:
        missing_package_files = [
            label for label, present in audit.package_files.items() if not present
        ]
        if missing_package_files:
            statuses["oracle_denominator"] = "not coverage-ready"
            gaps.append(
                "Coverage package directory is missing required files: "
                + ", ".join(missing_package_files)
                + "."
            )

    audit.format = fmt
    audit.row_count = row_count
    audit.sampled_rows = len(records)
    audit.key_counts = dict(sorted(key_counts.items()))
    audit.signals = signals
    audit.completeness = {
        "sample_with_unit_ids": _fraction_with(records, UNIT_KEYS),
        "sample_with_required_unit_ids": _fraction_with(records, UNIT_REQUIREMENT_KEYS),
        "sample_with_gold_or_answer_session_ids": _fraction_with(records, SESSION_REQUIREMENT_KEYS),
        "sample_with_candidate_ids": _fraction_with(records, CANDIDATE_KEYS),
        "sample_with_selected_candidate_ids": _fraction_with(records, SELECTED_CANDIDATE_KEYS),
        "sample_with_coverage": _fraction_with(records, {"coverage"}),
        "sample_with_context_or_used_ids": _fraction_with(records, CONTEXT_KEYS),
    }
    audit.statuses = statuses
    audit.gaps = gaps
    return audit


def _parse_artifact_arg(value: str) -> tuple[str, str, str]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
        return name.strip(), raw_path.strip(), "User-specified artifact."
    path = Path(value)
    return path.stem, value, "User-specified artifact."


def _markdown_table_row(audit: ArtifactAudit) -> str:
    gap = audit.gaps[0] if audit.gaps else "No structural gap detected."
    return (
        f"| `{audit.name}` | {audit.row_count} | "
        f"{audit.statuses.get('evidence_units', 'missing')} | "
        f"{audit.statuses.get('query_requirements', 'missing')} | "
        f"{audit.statuses.get('candidate_memories', 'missing')} | "
        f"{audit.statuses.get('coverage_matrix', 'missing')} | "
        f"{audit.statuses.get('oracle_denominator', 'not coverage-ready')} | "
        f"{gap} |"
    )


def render_report(audits: list[ArtifactAudit]) -> str:
    ready = [
        audit
        for audit in audits
        if audit.statuses.get("oracle_denominator") == "machine-checkable coverage package"
    ]
    lines = [
        "# Coverage Artifact Audit",
        "",
        "This report checks whether existing artifacts expose the machine-readable fields needed for non-synthetic OracleMem coverage annotation. It is a structural audit only; it does not certify semantic label quality.",
        "",
        "## Verdict",
        "",
    ]
    if ready:
        names = ", ".join(f"`{audit.name}`" for audit in ready)
        lines.append(f"Coverage-ready package candidates detected: {names}.")
    else:
        lines.append(
            "No audited current artifact is a complete non-synthetic coverage package. Existing LongMemEval artifacts remain session-level retrieval/reader/scoring diagnostics; synthetic result rows rely on generator code for hidden coverage and do not serialize a full matrix in the result files."
        )

    lines.extend(
        [
            "",
            "## Artifact Matrix",
            "",
            "| Artifact | Rows | Evidence | Query Requirements | Candidates | Coverage | Denominator Status | First Gap |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for audit in audits:
        lines.append(_markdown_table_row(audit))

    lines.extend(
        [
            "",
            "## Completeness Signals",
            "",
            "| Artifact | Sampled | Unit IDs | Required Units | Gold Sessions | Candidate IDs | Selected IDs | Coverage | Context/Used IDs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for audit in audits:
        c = audit.completeness
        lines.append(
            f"| `{audit.name}` | {audit.sampled_rows} | "
            f"{c.get('sample_with_unit_ids', 0.0):.3f} | "
            f"{c.get('sample_with_required_unit_ids', 0.0):.3f} | "
            f"{c.get('sample_with_gold_or_answer_session_ids', 0.0):.3f} | "
            f"{c.get('sample_with_candidate_ids', 0.0):.3f} | "
            f"{c.get('sample_with_selected_candidate_ids', 0.0):.3f} | "
            f"{c.get('sample_with_coverage', 0.0):.3f} | "
            f"{c.get('sample_with_context_or_used_ids', 0.0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance Gate",
            "",
            "Synthetic exact-small OracleMem instances can be exported for structural inspection with `python run_oraclemem_mvp.py --export-coverage-matrices`; pass an exported package directory to this script with `--artifact name=path/to/coverage_instances/base/seed_0`.",
            "",
            "To upgrade a non-synthetic LongMemEval/LoCoMo-style artifact from diagnostic to exact OracleMem coverage, add the package described in `COVERAGE_VALIDATION_PROTOCOL.md`: `evidence_units.jsonl`, `queries.jsonl` with `required_unit_ids`, `candidate_memories.jsonl`, `coverage_matrix.jsonl`, annotation decisions, and a candidate-generation manifest.",
            "",
            "Hard blockers for non-synthetic coverage are 100% schema completeness for eval queries, evidence units, positive coverage rows, candidate groups/costs, no future-source coverage, no generator leakage, and solver inputs derivable from the artifacts without hidden code defaults.",
            "",
            "| Acceptance item | Required threshold |",
            "| --- | ---: |",
            "| Eval queries with answer/category/session ids/adjudicated required units | 100% |",
            "| Required unit ids resolvable in `evidence_units.jsonl` | 100% |",
            "| Evidence units source-backed and adjudication resolved | 100% |",
            "| Positive coverage rows valid, sourced, rationalized, resolved | 100% |",
            "| Candidate groups, representation types, text, and costs valid | 100% |",
            "| Future-source coverage leakage | 0 rows |",
            "| Forbidden generator inputs leaked | 0 records |",
            "| Binary coverage agreement before adjudication | kappa >= 0.70 |",
            "| None/partial/full coverage agreement before adjudication | weighted kappa >= 0.60 |",
            "| Unit candidate availability at coverage >= 0.75 | >= 0.95 |",
            "| Query feasible support before budget | >= 0.90 |",
            "| Update/current-truth support for validity claims | >= 0.90 |",
            "| Hallucinated coverage after adjudication | 0 rows |",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm_memory_validation/coverage_artifact_audit"),
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Additional artifact as name=path or path. Can be repeated.",
    )
    parser.add_argument("--no-defaults", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=5000)
    args = parser.parse_args()

    specs: list[tuple[str, str, str]] = []
    if not args.no_defaults:
        specs.extend(DEFAULT_ARTIFACTS)
    specs.extend(_parse_artifact_arg(value) for value in args.artifact)
    if not specs:
        raise SystemExit("No artifacts to audit.")

    audits = [
        audit_artifact(name, Path(path), role, sample_rows=args.sample_rows)
        for name, path, role in specs
    ]
    payload = {
        "schema_version": 1,
        "sample_rows": args.sample_rows,
        "coverage_ready_artifacts": [
            audit.name
            for audit in audits
            if audit.statuses.get("oracle_denominator") == "machine-checkable coverage package"
        ],
        "artifacts": [asdict(audit) for audit in audits],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(render_report(audits), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
