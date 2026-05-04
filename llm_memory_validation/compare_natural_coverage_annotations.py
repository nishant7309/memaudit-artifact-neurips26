"""Compare two natural OracleMem coverage packages.

The comparison is intentionally conservative. Unit identifiers can differ
across annotators, so the report compares normalized required-unit text and
candidate-coverage text pairs in addition to exact ids.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(value: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(value).lower()))


def package_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "queries": read_jsonl(path / "queries.jsonl"),
        "evidence_units": read_jsonl(path / "evidence_units.jsonl"),
        "candidate_memories": read_jsonl(path / "candidate_memories.jsonl"),
        "coverage_matrix": read_jsonl(path / "coverage_matrix.jsonl"),
    }


def unit_text_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("unit_id")): norm_text(row.get("canonical_text") or row.get("unit_id"))
        for row in rows
        if row.get("unit_id")
    }


def candidate_text_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("candidate_id")): norm_text(row.get("text") or row.get("serialized") or row.get("candidate_id"))
        for row in rows
        if row.get("candidate_id")
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def required_texts(query: Mapping[str, Any], unit_text: Mapping[str, str]) -> set[str]:
    return {
        unit_text.get(str(unit_id), norm_text(unit_id))
        for unit_id in query.get("required_unit_ids", []) or []
        if unit_text.get(str(unit_id), norm_text(unit_id))
    }


def coverage_text_edges(
    coverage_rows: Iterable[Mapping[str, Any]],
    unit_text: Mapping[str, str],
    candidate_text: Mapping[str, str],
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for row in coverage_rows:
        cov = float(row.get("coverage", 0.0) or 0.0)
        if cov <= 0:
            continue
        ctext = candidate_text.get(str(row.get("candidate_id")), "")
        utext = unit_text.get(str(row.get("unit_id")), "")
        if ctext and utext:
            edges.add((ctext, utext))
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    primary = package_rows(args.primary)
    secondary = package_rows(args.secondary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p_unit_text = unit_text_map(primary["evidence_units"])
    s_unit_text = unit_text_map(secondary["evidence_units"])
    p_candidate_text = candidate_text_map(primary["candidate_memories"])
    s_candidate_text = candidate_text_map(secondary["candidate_memories"])

    p_queries = {str(row.get("query_id")): row for row in primary["queries"] if row.get("query_id")}
    s_queries = {str(row.get("query_id")): row for row in secondary["queries"] if row.get("query_id")}
    common_query_ids = sorted(set(p_queries) & set(s_queries))

    agreement_rows: list[dict[str, Any]] = []
    exact_required_agree = 0
    both_resolved = 0
    primary_resolved = 0
    secondary_resolved = 0
    for query_id in common_query_ids:
        p_required = required_texts(p_queries[query_id], p_unit_text)
        s_required = required_texts(s_queries[query_id], s_unit_text)
        if p_required:
            primary_resolved += 1
        if s_required:
            secondary_resolved += 1
        if p_required and s_required:
            both_resolved += 1
        if p_required == s_required:
            exact_required_agree += 1
        agreement_rows.append(
            {
                "query_id": query_id,
                "primary_required_texts": sorted(p_required),
                "secondary_required_texts": sorted(s_required),
                "required_text_jaccard": jaccard(p_required, s_required),
                "agreement_class": (
                    "AGREE"
                    if p_required == s_required
                    else "UNRESOLVED"
                    if not p_required or not s_required
                    else "MINOR_DISAGREEMENT"
                    if jaccard(p_required, s_required) >= 0.5
                    else "MAJOR_DISAGREEMENT"
                ),
            }
        )

    p_edges = coverage_text_edges(primary["coverage_matrix"], p_unit_text, p_candidate_text)
    s_edges = coverage_text_edges(secondary["coverage_matrix"], s_unit_text, s_candidate_text)
    summary = {
        "schema_version": 1,
        "primary": str(args.primary),
        "secondary": str(args.secondary),
        "common_queries": len(common_query_ids),
        "primary_resolved": primary_resolved,
        "secondary_resolved": secondary_resolved,
        "both_resolved": both_resolved,
        "exact_required_text_agreement_rate": (exact_required_agree / len(common_query_ids)) if common_query_ids else 0.0,
        "mean_required_text_jaccard": (
            sum(float(row["required_text_jaccard"]) for row in agreement_rows) / len(agreement_rows)
            if agreement_rows
            else 0.0
        ),
        "major_disagreement_count": sum(1 for row in agreement_rows if row["agreement_class"] == "MAJOR_DISAGREEMENT"),
        "minor_disagreement_count": sum(1 for row in agreement_rows if row["agreement_class"] == "MINOR_DISAGREEMENT"),
        "unresolved_disagreement_count": sum(1 for row in agreement_rows if row["agreement_class"] == "UNRESOLVED"),
        "coverage_edge_text_jaccard": jaccard(p_edges, s_edges),
        "primary_coverage_edges": len(p_edges),
        "secondary_coverage_edges": len(s_edges),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "agreement_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in agreement_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = [
        "# Natural Coverage Annotation Agreement",
        "",
        f"- Primary: `{args.primary}`",
        f"- Secondary: `{args.secondary}`",
        f"- Common queries: {summary['common_queries']}",
        f"- Primary resolved: {summary['primary_resolved']}",
        f"- Secondary resolved: {summary['secondary_resolved']}",
        f"- Both resolved: {summary['both_resolved']}",
        f"- Exact required-text agreement: {summary['exact_required_text_agreement_rate']:.3f}",
        f"- Mean required-text Jaccard: {summary['mean_required_text_jaccard']:.3f}",
        f"- Coverage-edge text Jaccard: {summary['coverage_edge_text_jaccard']:.3f}",
        f"- Major disagreements: {summary['major_disagreement_count']}",
        f"- Minor disagreements: {summary['minor_disagreement_count']}",
        f"- Unresolved disagreements: {summary['unresolved_disagreement_count']}",
        "",
        "This is a model-model agreement audit. It does not certify semantic truth; it identifies which examples need manual adjudication.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
