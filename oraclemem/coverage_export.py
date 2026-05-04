"""Export finite OracleMem instances as inspectable coverage packages.

The exact-small runner keeps candidate-by-unit coverage in memory while it
solves.  These helpers serialize that hidden finite instance into the same
artifact shape expected by the coverage validation protocol, so reviewers can
inspect candidate memories and the sparse coverage matrix without rerunning a
solver or calling an external API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate import CandidateMemory, OracleMemInstance, generate_named_distribution


SYNTHETIC_ANNOTATOR_ID = "synthetic_generator"
PACKAGE_SCHEMA_VERSION = 1
REQUIRED_PACKAGE_FILES = (
    "experiences.jsonl",
    "evidence_units.jsonl",
    "queries.jsonl",
    "candidate_memories.jsonl",
    "coverage_matrix.jsonl",
    "annotation_decisions.jsonl",
    "candidate_generation_manifest.json",
)


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return token.strip("._") or "instance"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coverage_label(value: float) -> str:
    if value >= 1.0:
        return "full"
    if value >= 0.75:
        return "partial_strong"
    if value >= 0.5:
        return "partial_weak"
    return "hint_only"


def _unit_kind(unit_id: str, instance: OracleMemInstance) -> str:
    if unit_id in instance.current_units:
        return "current_truth"
    if unit_id in instance.invalidation_units:
        return "invalidation"
    if unit_id in instance.stale_units:
        return "stale_or_superseded"
    lowered = unit_id.lower()
    if lowered.startswith("current:") or ":current:" in lowered:
        return "current_truth"
    if lowered.startswith("invalid:") or ":invalidates:" in lowered:
        return "invalidation"
    if lowered.startswith("stale:"):
        return "stale_or_superseded"
    if lowered.startswith("context:") or ":context:" in lowered:
        return "context"
    if "abstain" in lowered:
        return "abstention"
    return "synthetic_evidence"


def _unit_state(unit_id: str, instance: OracleMemInstance) -> str:
    kind = _unit_kind(unit_id, instance)
    if kind == "invalidation":
        return "invalidates_prior"
    if kind == "stale_or_superseded":
        return "superseded"
    if kind == "abstention":
        return "missing_evidence"
    return "current"


def _source_span_id(candidate: CandidateMemory) -> str:
    return f"{candidate.experience_id}:synthetic_span"


def _experience_rows(instance: OracleMemInstance) -> list[dict[str, Any]]:
    by_experience: dict[str, list[CandidateMemory]] = {}
    for candidate in instance.candidates:
        by_experience.setdefault(candidate.experience_id, []).append(candidate)

    rows: list[dict[str, Any]] = []
    for experience_id, candidates in sorted(
        by_experience.items(),
        key=lambda item: (min(candidate.time_index for candidate in item[1]), item[0]),
    ):
        ordered = sorted(candidates, key=lambda candidate: (candidate.time_index, candidate.candidate_id))
        visible_units = sorted({unit_id for candidate in ordered for unit_id in candidate.coverage})
        preferred = next(
            (
                candidate
                for candidate in ordered
                if candidate.representation_type in {"raw_span", "raw"}
            ),
            ordered[0],
        )
        rows.append(
            {
                "experience_id": experience_id,
                "session_id": f"{instance.instance_id}:synthetic",
                "timestamp": min(candidate.time_index for candidate in ordered),
                "text": preferred.serialized,
                "source_span_ids": [_source_span_id(preferred)],
                "split": "synthetic",
                "visible_unit_ids": visible_units,
            }
        )
    return rows


def _unit_source_spans(unit_id: str, candidates: Sequence[CandidateMemory]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: (item.time_index, item.candidate_id)):
        if candidate.coverage.get(unit_id, 0.0) <= 0:
            continue
        span_id = _source_span_id(candidate)
        if span_id in seen:
            continue
        spans.append(
            {
                "span_id": span_id,
                "experience_id": candidate.experience_id,
                "timestamp": candidate.time_index,
                "text": candidate.serialized,
            }
        )
        seen.add(span_id)
    return spans


def _evidence_unit_rows(instance: OracleMemInstance) -> list[dict[str, Any]]:
    units = sorted(
        set(instance.unit_weights)
        | {unit_id for candidate in instance.candidates for unit_id in candidate.coverage}
    )
    rows: list[dict[str, Any]] = []
    for unit_id in units:
        spans = _unit_source_spans(unit_id, instance.candidates)
        rows.append(
            {
                "unit_id": unit_id,
                "kind": _unit_kind(unit_id, instance),
                "canonical_text": unit_id,
                "source_spans": spans,
                "timestamp": min((span["timestamp"] for span in spans), default=0),
                "state": _unit_state(unit_id, instance),
                "proposition_id": unit_id.rsplit(":", 1)[0] if ":" in unit_id else unit_id,
                "annotator_ids": [SYNTHETIC_ANNOTATOR_ID],
                "adjudication_status": "resolved",
                "unit_weight": float(instance.unit_weights.get(unit_id, 0.0)),
            }
        )
    return rows


def _query_rows(instance: OracleMemInstance, evidence_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, evidence in enumerate(evidence_rows):
        unit_id = str(evidence["unit_id"])
        source_spans = list(evidence.get("source_spans", []))
        rows.append(
            {
                "query_id": f"{instance.instance_id}:objective_unit:{index:04d}",
                "question": f"Synthetic objective probe for {unit_id}.",
                "answer": str(evidence["canonical_text"]),
                "category": str(evidence["kind"]),
                "required_unit_ids": [unit_id],
                "answer_session_ids": sorted(
                    {
                        str(span.get("experience_id", ""))
                        for span in source_spans
                        if span.get("experience_id")
                    }
                ),
                "split": "synthetic",
                "unit_weight": float(evidence.get("unit_weight", 0.0)),
            }
        )
    return rows


def _candidate_rows(
    instance: OracleMemInstance,
    generation_config_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(instance.candidates, key=lambda item: item.candidate_id):
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "experience_id": candidate.experience_id,
                "candidate_group": candidate.experience_id,
                "representation_type": candidate.representation_type,
                "text": candidate.serialized,
                "serialized": candidate.serialized,
                "cost_tokens": int(candidate.cost),
                "cost": int(candidate.cost),
                "generator_id": candidate.generator,
                "generation_config_hash": generation_config_hash,
                "time_index": int(candidate.time_index),
                "confidence": float(candidate.confidence),
                "estimated_value": candidate.estimated_value,
                "estimated_coverage": dict(sorted(candidate.estimated_coverage.items())),
                "estimator_model": candidate.estimator_model,
            }
        )
    return rows


def _coverage_rows(
    instance: OracleMemInstance,
    *,
    include_zero_coverage: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unit_ids = sorted(
        set(instance.unit_weights)
        | {unit_id for candidate in instance.candidates for unit_id in candidate.coverage}
    )
    for candidate in sorted(instance.candidates, key=lambda item: item.candidate_id):
        for unit_id in unit_ids:
            value = float(candidate.coverage.get(unit_id, 0.0))
            if value <= 0 and not include_zero_coverage:
                continue
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "experience_id": candidate.experience_id,
                    "candidate_group": candidate.experience_id,
                    "unit_id": unit_id,
                    "coverage": value,
                    "coverage_label": _coverage_label(value) if value > 0 else "none",
                    "rationale": (
                        "Synthetic OracleMem generator assigned this positive "
                        "candidate-by-unit coverage cell."
                    ),
                    "source_span_ids": [_source_span_id(candidate)],
                    "annotator_ids": [SYNTHETIC_ANNOTATOR_ID],
                    "adjudication_status": "resolved",
                }
            )
    return rows


def _annotation_decision_rows(instance: OracleMemInstance, coverage_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"{instance.instance_id}:synthetic_coverage:{index:06d}",
            "record_type": "coverage_cell",
            "decision": "accepted",
            "primary_annotator": SYNTHETIC_ANNOTATOR_ID,
            "verifier": SYNTHETIC_ANNOTATOR_ID,
            "adjudicator": SYNTHETIC_ANNOTATOR_ID,
            "notes": (
                "Automatically generated from the finite synthetic OracleMem "
                "instance; no human annotation was required."
            ),
            "candidate_id": row["candidate_id"],
            "unit_id": row["unit_id"],
        }
        for index, row in enumerate(coverage_rows)
    ]


def export_coverage_package(
    instance: OracleMemInstance,
    out_dir: str | Path,
    *,
    distribution: str | None = None,
    include_zero_coverage: bool = False,
) -> dict[str, str]:
    """Write a protocol-style coverage package for one finite instance.

    The package is intended for synthetic/exact-small instances where coverage
    is already part of the generated finite instance.  It is structural evidence
    that the matrix is inspectable; it does not certify a non-synthetic human
    annotation package.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generation_config = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "seed": instance.seed,
        "distribution": distribution or "unknown",
        "candidate_count": len(instance.candidates),
        "unit_count": len(instance.unit_weights),
        "generator": "oraclemem.synthetic",
    }
    generation_config_hash = _stable_json_hash(generation_config)

    experiences = _experience_rows(instance)
    evidence_units = _evidence_unit_rows(instance)
    queries = _query_rows(instance, evidence_units)
    candidate_memories = _candidate_rows(instance, generation_config_hash)
    coverage_matrix = _coverage_rows(
        instance,
        include_zero_coverage=include_zero_coverage,
    )
    annotation_decisions = _annotation_decision_rows(instance, coverage_matrix)

    paths = {
        "experiences": out_path / "experiences.jsonl",
        "evidence_units": out_path / "evidence_units.jsonl",
        "queries": out_path / "queries.jsonl",
        "candidate_memories": out_path / "candidate_memories.jsonl",
        "coverage_matrix": out_path / "coverage_matrix.jsonl",
        "annotation_decisions": out_path / "annotation_decisions.jsonl",
        "instance": out_path / "instance.json",
    }
    _write_jsonl(paths["experiences"], experiences)
    _write_jsonl(paths["evidence_units"], evidence_units)
    _write_jsonl(paths["queries"], queries)
    _write_jsonl(paths["candidate_memories"], candidate_memories)
    _write_jsonl(paths["coverage_matrix"], coverage_matrix)
    _write_jsonl(paths["annotation_decisions"], annotation_decisions)
    _write_json(
        paths["instance"],
        {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "instance_id": instance.instance_id,
            "seed": instance.seed,
            "distribution": distribution or "unknown",
            "unit_weights": dict(sorted(instance.unit_weights.items())),
            "current_units": list(instance.current_units),
            "invalidation_units": list(instance.invalidation_units),
            "stale_units": list(instance.stale_units),
            "candidate_count": len(instance.candidates),
            "coverage_cell_count": len(coverage_matrix),
            "positive_coverage_cell_count": sum(
                1 for row in coverage_matrix if float(row["coverage"]) > 0
            ),
        },
    )

    file_hashes = {
        path.name: _file_sha256(path)
        for path in sorted(paths.values(), key=lambda item: item.name)
    }
    manifest_path = out_path / "candidate_generation_manifest.json"
    _write_json(
        manifest_path,
        {
            **generation_config,
            "generation_config_hash": generation_config_hash,
            "synthetic_instance": True,
            "generators": [
                {
                    "generator_id": "oraclemem.synthetic",
                    "description": "Deterministic local OracleMem synthetic generator.",
                }
            ],
            "allowed_inputs": [
                "synthetic generator parameters",
                "candidate-visible synthetic experience text",
            ],
            "forbidden_inputs": [
                "held-out query text",
                "gold answers",
                "required_unit_ids beyond the generated synthetic objective",
                "oracle weights during deployable writer decisions",
                "method labels",
            ],
            "prompt_hashes": {},
            "cost_counter": "candidate.cost serialized as cost_tokens",
            "split_hashes": file_hashes,
            "counts": {
                "evidence_units": len(evidence_units),
                "queries": len(queries),
                "experiences": len(experiences),
                "candidate_memories": len(candidate_memories),
                "positive_coverage_rows": sum(
                    1 for row in coverage_matrix if float(row["coverage"]) > 0
                ),
                "coverage_rows": len(coverage_matrix),
            },
            "notes": (
                "Synthetic exact-small export. Coverage cells are generated by "
                "code and are inspectable in coverage_matrix.jsonl."
            ),
        },
    )
    file_hashes[manifest_path.name] = _file_sha256(manifest_path)
    readme_path = out_path / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# OracleMem Synthetic Coverage Package",
                "",
                f"Instance: `{instance.instance_id}`",
                f"Distribution: `{distribution or 'unknown'}`",
                f"Candidates: {len(instance.candidates)}",
                f"Positive coverage cells: {len(coverage_matrix)}",
                "",
                "`coverage_matrix.jsonl` is sparse: missing candidate/unit cells are zero.",
                "This package is inspectable synthetic coverage, not a human non-synthetic annotation certificate.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = {
        "manifest": str(manifest_path),
        "package_dir": str(out_path),
        "candidate_generation_manifest": str(manifest_path),
        "README": str(readme_path),
    }
    result.update({key: str(path) for key, path in paths.items()})
    return result


def export_coverage_packages(
    *,
    seeds: Sequence[int],
    distributions: Sequence[str],
    out_dir: str | Path,
    normal_count: int = 3,
    update_count: int = 2,
    max_packages: int | None = None,
) -> dict[str, Any]:
    """Generate and export coverage packages for a benchmark seed grid."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    packages: list[dict[str, Any]] = []
    for distribution in distributions:
        distribution_dir = root / _safe_path_token(distribution)
        for seed in seeds:
            if max_packages is not None and len(packages) >= max_packages:
                break
            instance = generate_named_distribution(
                distribution,
                seed,
                normal_count=normal_count,
                update_count=update_count,
            )
            package_dir = distribution_dir / f"seed_{seed}"
            paths = export_coverage_package(
                instance,
                package_dir,
                distribution=distribution,
            )
            packages.append(
                {
                    "distribution": distribution,
                    "seed": seed,
                    "instance_id": instance.instance_id,
                    "package_dir": paths["package_dir"],
                    "coverage_matrix": paths["coverage_matrix"],
                    "candidate_memories": paths["candidate_memories"],
                    "manifest": paths["candidate_generation_manifest"],
                }
            )
        if max_packages is not None and len(packages) >= max_packages:
            break

    manifest_path = root / "coverage_export_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "coverage_matrices_exportable": True,
            "package_count": len(packages),
            "packages": packages,
        },
    )
    return {
        "manifest": str(manifest_path),
        "package_count": len(packages),
        "packages": packages,
    }


__all__ = [
    "REQUIRED_PACKAGE_FILES",
    "export_coverage_package",
    "export_coverage_packages",
]
