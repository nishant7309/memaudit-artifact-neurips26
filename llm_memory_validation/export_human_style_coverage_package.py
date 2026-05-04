"""Export human-edited examples to the OracleMem coverage-package schema.

The human-style examples are stored as one JSON record per future query. This
script writes the same package files used by the natural Mem0/A-Mem runners:
experiences, evidence units, candidate memories, sparse coverage rows, and
queries. It does not create new annotations; it only normalizes the audited
example file into the shared evaluator format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def session_text(session: Mapping[str, Any]) -> str:
    messages = []
    for message in session.get("messages", []) or []:
        speaker = str(message.get("speaker", "speaker"))
        text = str(message.get("text", "")).strip()
        if text:
            messages.append(f"{speaker}: {text}")
    return "\n".join(messages)


def export_package(examples: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    experiences: list[dict[str, Any]] = []
    evidence_units: list[dict[str, Any]] = []
    candidate_memories: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    annotation_decisions: list[dict[str, Any]] = []

    for example_index, row in enumerate(examples):
        example_id = str(row["example_id"])
        for session_index, session in enumerate(row.get("sessions", []) or []):
            session_id = str(session.get("session_id", f"s{session_index}"))
            experiences.append(
                {
                    "experience_id": f"{example_id}::{session_id}",
                    "instance_id": example_id,
                    "time_index": session_index,
                    "text": session_text(session),
                    "timestamp": f"{example_index:04d}-{session_index:02d}",
                    "generator": "human_edited",
                }
            )

        required = {str(unit_id) for unit_id in row.get("required_unit_ids_for_query", []) or []}
        namespaced_required = [f"{example_id}::{unit_id}" for unit_id in sorted(required)]
        for unit in row.get("evidence_units", []) or []:
            unit_id = str(unit["unit_id"])
            namespaced = f"{example_id}::{unit_id}"
            evidence_units.append(
                {
                    "unit_id": namespaced,
                    "instance_id": example_id,
                    "canonical_text": str(unit.get("text", "")),
                    "kind": str(unit.get("state", "current")),
                    "unit_weight": 1.0 if unit_id in required else 0.0,
                    "source_session_ids": unit.get("source_session_ids", []),
                    "source_spans": [
                        {"text": quote}
                        for quote in unit.get("source_message_quotes", []) or []
                    ],
                    "generator": "human_edited",
                }
            )

        for candidate_index, candidate in enumerate(row.get("candidate_memories", []) or []):
            candidate_id = f"{example_id}::{candidate.get('candidate_id', f'c{candidate_index}')}"
            candidate_memories.append(
                {
                    "candidate_id": candidate_id,
                    "instance_id": example_id,
                    "experience_id": example_id,
                    "candidate_group": example_id,
                    "representation_type": str(candidate.get("representation_type", "unknown")),
                    "serialized": str(candidate.get("text", "")),
                    "cost": max(1, int(candidate.get("cost_tokens_estimate", 1) or 1)),
                    "time_index": example_index,
                    "generator": "human_edited",
                    "source_session_ids": candidate.get("source_session_ids", []),
                }
            )
            for unit_id, coverage in dict(candidate.get("coverage", {})).items():
                namespaced_unit = f"{example_id}::{unit_id}"
                coverage_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "unit_id": namespaced_unit,
                        "coverage": float(coverage),
                        "generator": "human_edited",
                    }
                )

        future_query = row.get("future_query", {}) or {}
        queries.append(
            {
                "query_id": example_id,
                "question": str(future_query.get("text", "")),
                "answer": str(future_query.get("answer", "")),
                "required_unit_ids": namespaced_required,
                "category": str(row.get("domain", "")),
                "split": "human_style_examples",
                "adjudication_status": "human_edited_schema_valid",
                "source_example_id": example_id,
            }
        )
        annotation_decisions.append(
            {
                "query_id": example_id,
                "status": "accepted",
                "adjudication_status": "human_edited_schema_valid",
                "source": "human_style_examples",
                "notes": str(row.get("annotation_notes", "")),
                "required_unit_ids": namespaced_required,
            }
        )

    write_jsonl(out_dir / "experiences.jsonl", experiences)
    write_jsonl(out_dir / "evidence_units.jsonl", evidence_units)
    write_jsonl(out_dir / "candidate_memories.jsonl", candidate_memories)
    write_jsonl(out_dir / "coverage_matrix.jsonl", coverage_rows)
    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "annotation_decisions.jsonl", annotation_decisions)
    manifest = {
        "annotation_decisions": len(annotation_decisions),
        "examples": len(examples),
        "experiences": len(experiences),
        "evidence_units": len(evidence_units),
        "candidate_memories": len(candidate_memories),
        "coverage_rows": len(coverage_rows),
        "source": "human_style_examples",
    }
    (out_dir / "candidate_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("llm_memory_validation/human_style_examples/examples_100.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("llm_memory_validation/human_style_examples/coverage_package"),
    )
    args = parser.parse_args()
    manifest = export_package(read_jsonl(args.examples_jsonl), args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
