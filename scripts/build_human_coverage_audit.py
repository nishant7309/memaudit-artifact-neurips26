"""Build and score a human coverage annotation audit for MemAudit.

The audit path creates annotation CSVs over all positive Gemini coverage cells
plus a stratified sample of zero cells. It computes Cohen's kappa and
package-ratio stability from filled human label files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import CandidateMemory, objective_value, selected_candidates, solve_exact
from llm_memory_validation.run_mem0_natural_baseline import load_package, package_instance, resolved_queries


DEFAULT_PACKAGE = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package")
DEFAULT_OUT = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit")
DEFAULT_RAW_RESULTS = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/raw_results.jsonl")


@dataclass(frozen=True)
class AuditCell:
    cell_id: str
    instance_id: str
    candidate_id: str
    unit_id: str
    gemini_label: int
    gemini_coverage: float
    sampling_stratum: str
    representation_type: str
    candidate_text: str
    evidence_text: str
    gemini_rationale: str


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prefix_of(item_id: str) -> str:
    return str(item_id).split("::", 1)[0]


def binary_label(value: float, threshold: float) -> int:
    return 1 if float(value) >= threshold else 0


def load_package_tables(package_dir: Path) -> dict[str, Any]:
    candidates = read_jsonl(package_dir / "candidate_memories.jsonl")
    units = read_jsonl(package_dir / "evidence_units.jsonl")
    coverage = read_jsonl(package_dir / "coverage_matrix.jsonl")
    queries = read_jsonl(package_dir / "queries.jsonl")
    return {
        "candidates": candidates,
        "units": units,
        "coverage": coverage,
        "queries": queries,
    }


def build_audit_cells(
    package_dir: Path,
    *,
    zero_sample_rate: float = 0.20,
    seed: int = 20260503,
    coverage_threshold: float = 0.5,
) -> list[AuditCell]:
    tables = load_package_tables(package_dir)
    rng = random.Random(seed)

    candidate_by_id = {str(row["candidate_id"]): row for row in tables["candidates"]}
    unit_by_id = {str(row["unit_id"]): row for row in tables["units"]}

    candidates_by_instance: dict[str, list[str]] = defaultdict(list)
    for candidate_id in candidate_by_id:
        candidates_by_instance[prefix_of(candidate_id)].append(candidate_id)

    units_by_instance: dict[str, list[str]] = defaultdict(list)
    for unit_id in unit_by_id:
        units_by_instance[prefix_of(unit_id)].append(unit_id)

    positive: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tables["coverage"]:
        candidate_id = str(row["candidate_id"])
        unit_id = str(row["unit_id"])
        value = float(row.get("coverage", row.get("fidelity", 0.0)) or 0.0)
        if value > 0:
            key = (candidate_id, unit_id)
            if value > float(positive.get(key, {}).get("coverage", -1.0)):
                positive[key] = dict(row)

    cells: list[AuditCell] = []

    def make_cell(candidate_id: str, unit_id: str, *, stratum: str, row: Mapping[str, Any] | None = None) -> AuditCell:
        candidate = candidate_by_id[candidate_id]
        unit = unit_by_id[unit_id]
        coverage = float((row or {}).get("coverage", (row or {}).get("fidelity", 0.0)) or 0.0)
        cell_id = f"{prefix_of(candidate_id)}__{safe_id(candidate_id)}__{safe_id(unit_id)}"
        return AuditCell(
            cell_id=cell_id,
            instance_id=prefix_of(candidate_id),
            candidate_id=candidate_id,
            unit_id=unit_id,
            gemini_label=binary_label(coverage, coverage_threshold),
            gemini_coverage=coverage,
            sampling_stratum=stratum,
            representation_type=str(candidate.get("representation_type", "")),
            candidate_text=str(candidate.get("serialized") or candidate.get("text") or ""),
            evidence_text=str(unit.get("canonical_text") or unit.get("text") or ""),
            gemini_rationale=str((row or {}).get("rationale", "")),
        )

    for candidate_id, unit_id in sorted(positive):
        cells.append(make_cell(candidate_id, unit_id, stratum="gemini_positive", row=positive[(candidate_id, unit_id)]))

    for instance_id in sorted(set(candidates_by_instance) & set(units_by_instance)):
        zero_pairs = [
            (candidate_id, unit_id)
            for candidate_id in candidates_by_instance[instance_id]
            for unit_id in units_by_instance[instance_id]
            if (candidate_id, unit_id) not in positive
        ]
        if not zero_pairs:
            continue
        rng.shuffle(zero_pairs)
        sample_n = max(1, int(round(len(zero_pairs) * zero_sample_rate)))
        for candidate_id, unit_id in sorted(zero_pairs[:sample_n]):
            cells.append(make_cell(candidate_id, unit_id, stratum="sampled_gemini_zero", row=None))

    cells.sort(key=lambda cell: (cell.instance_id, cell.sampling_stratum, cell.candidate_id, cell.unit_id))
    return cells


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))[:160]


def write_cells_csv(path: Path, cells: Sequence[AuditCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id",
        "instance_id",
        "candidate_id",
        "unit_id",
        "gemini_label",
        "gemini_coverage",
        "sampling_stratum",
        "representation_type",
        "candidate_text",
        "evidence_text",
        "gemini_rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow({name: getattr(cell, name) for name in fieldnames})


def write_empty_label_csv(path: Path, cells: Sequence[AuditCell], annotator_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_id", "annotator_id", "label", "notes"])
        writer.writeheader()
        for cell in cells:
            writer.writerow({"cell_id": cell.cell_id, "annotator_id": annotator_id, "label": "", "notes": ""})


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = str(row.get("label", "")).strip().lower()
            if raw in {"1", "yes", "y", "true"}:
                labels[str(row["cell_id"])] = 1
            elif raw in {"0", "no", "n", "false"}:
                labels[str(row["cell_id"])] = 0
    return labels


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b):
        raise ValueError("label vectors must have equal length")
    if not a:
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - expected) < 1e-12:
        return 1.0 if abs(observed - 1) < 1e-12 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_metrics(cells: Sequence[AuditCell], labels_a: Mapping[str, int], labels_b: Mapping[str, int]) -> dict[str, Any]:
    common = [cell for cell in cells if cell.cell_id in labels_a and cell.cell_id in labels_b]
    a = [int(labels_a[cell.cell_id]) for cell in common]
    b = [int(labels_b[cell.cell_id]) for cell in common]
    agreed = [cell for cell, x, y in zip(common, a, b) if x == y]
    consensus = [int(labels_a[cell.cell_id]) for cell in agreed]
    gemini = [int(cell.gemini_label) for cell in agreed]
    return {
        "doubly_labeled_cells": len(common),
        "human_human_kappa": cohen_kappa(a, b) if common else None,
        "human_human_raw_agreement": (sum(1 for x, y in zip(a, b) if x == y) / len(common)) if common else None,
        "human_agreed_cells_for_model_comparison": len(agreed),
        "human_disagreement_cells_dropped": len(common) - len(agreed),
        "human_vs_gemini_kappa_on_agreed_cells": cohen_kappa(consensus, gemini) if agreed else None,
        "human_vs_gemini_raw_agreement_on_agreed_cells": (
            sum(1 for x, y in zip(consensus, gemini) if x == y) / len(agreed)
        )
        if agreed
        else None,
    }


def coverage_overrides(
    cells: Sequence[AuditCell],
    labels_a: Mapping[str, int],
    labels_b: Mapping[str, int],
) -> dict[tuple[str, str], int]:
    overrides: dict[tuple[str, str], int] = {}
    for cell in cells:
        if cell.cell_id not in labels_a or cell.cell_id not in labels_b:
            continue
        if labels_a[cell.cell_id] != labels_b[cell.cell_id]:
            continue
        overrides[(cell.candidate_id, cell.unit_id)] = int(labels_a[cell.cell_id])
    return overrides


def humanized_package_instance(package_dir: Path, query: Mapping[str, Any], overrides: Mapping[tuple[str, str], int]):
    data = load_package(package_dir)
    base = package_instance(data, query)
    candidates: list[CandidateMemory] = []
    for candidate in base.candidates:
        coverage = dict(candidate.coverage)
        relevant_overrides = {
            unit_id: label
            for (candidate_id, unit_id), label in overrides.items()
            if candidate_id == candidate.candidate_id
        }
        for unit_id, label in relevant_overrides.items():
            if label:
                coverage[unit_id] = 1.0
            else:
                coverage.pop(unit_id, None)
        candidates.append(
            CandidateMemory(
                candidate_id=candidate.candidate_id,
                experience_id=candidate.experience_id,
                representation_type=candidate.representation_type,
                serialized=candidate.serialized,
                cost=candidate.cost,
                coverage=coverage,
                time_index=candidate.time_index,
                generator=candidate.generator,
                confidence=candidate.confidence,
                estimated_value=candidate.estimated_value,
                estimated_coverage=candidate.estimated_coverage,
                estimator_model=candidate.estimator_model,
            )
        )
    return type(base)(
        instance_id=base.instance_id,
        candidates=tuple(candidates),
        unit_weights=base.unit_weights,
        seed=base.seed,
        current_units=base.current_units,
        invalidation_units=base.invalidation_units,
        stale_units=base.stale_units,
    )


def ratio_stability(
    package_dir: Path,
    raw_results_path: Path,
    overrides: Mapping[tuple[str, str], int],
    *,
    methods: set[str] | None = None,
    solver: str = "exact_stdlib",
) -> dict[str, Any]:
    if not raw_results_path.exists():
        return {"status": "missing_raw_results", "raw_results_path": str(raw_results_path)}
    data = load_package(package_dir)
    query_by_id = {str(query["query_id"]): query for query in resolved_queries(data, None)}
    rows = read_jsonl(raw_results_path)
    by_query_budget: dict[tuple[str, int], Any] = {}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        if method == "opt":
            continue
        if methods and method not in methods:
            continue
        instance_id = str(row.get("instance_id", ""))
        if instance_id not in query_by_id:
            continue
        budget = int(row.get("budget", 0))
        key = (instance_id, budget)
        if key not in by_query_budget:
            by_query_budget[key] = humanized_package_instance(package_dir, query_by_id[instance_id], overrides)
        instance = by_query_budget[key]
        exact = solve_exact(instance, budget, solver=solver)
        selected_ids = [str(candidate_id) for candidate_id in row.get("selected_candidate_ids", [])]
        selected = selected_candidates(instance.candidates, selected_ids)
        human_objective = objective_value(selected, instance.unit_weights)
        human_ratio = human_objective / exact.objective_value if exact.objective_value > 0 else None
        original_ratio = row.get("ratio_to_opt")
        if original_ratio is None:
            original_ratio = row.get("ratio_to_package_candidate_opt")
        if original_ratio is None or human_ratio is None:
            continue
        out_rows.append(
            {
                "instance_id": instance_id,
                "budget": budget,
                "method": method,
                "original_ratio": original_ratio,
                "human_ratio": human_ratio,
                "abs_diff": abs(float(original_ratio) - float(human_ratio)),
            }
        )
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        grouped[(int(row["budget"]), str(row["method"]))].append(row)
    summary_rows = []
    for (budget, method), group in sorted(grouped.items()):
        diffs = [float(row["abs_diff"]) for row in group if row["abs_diff"] is not None]
        human = [float(row["human_ratio"]) for row in group if row["human_ratio"] is not None]
        original = [float(row["original_ratio"]) for row in group if row["original_ratio"] is not None]
        summary_rows.append(
            {
                "budget": budget,
                "method": method,
                "n": len(group),
                "mean_original_ratio": statistics.fmean(original) if original else None,
                "mean_human_ratio": statistics.fmean(human) if human else None,
                "mean_abs_diff": statistics.fmean(diffs) if diffs else None,
            }
        )
    rankings: dict[int, dict[str, Any]] = {}
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_budget[int(row["budget"])].append(row)
    for budget, group in sorted(by_budget.items()):
        original_rank = [
            row["method"]
            for row in sorted(group, key=lambda r: (r["mean_original_ratio"] is None, -(r["mean_original_ratio"] or -1)))
        ]
        human_rank = [
            row["method"]
            for row in sorted(group, key=lambda r: (r["mean_human_ratio"] is None, -(r["mean_human_ratio"] or -1)))
        ]
        rankings[budget] = {
            "original_rank": original_rank,
            "human_label_rank": human_rank,
            "ranking_preserved": original_rank == human_rank,
        }
    diffs_all = [float(row["abs_diff"]) for row in out_rows if row["abs_diff"] is not None]
    return {
        "status": "ok",
        "raw_results_path": str(raw_results_path),
        "row_count": len(out_rows),
        "mean_abs_diff": statistics.fmean(diffs_all) if diffs_all else None,
        "target_mean_abs_diff_le_0_05": (statistics.fmean(diffs_all) <= 0.05) if diffs_all else None,
        "summary_rows": summary_rows,
        "rankings": rankings,
    }


def write_stability_csv(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["budget", "method", "n", "mean_original_ratio", "mean_human_ratio", "mean_abs_diff"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-results", type=Path, default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--zero-sample-rate", type=float, default=0.20)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = build_audit_cells(
        args.package_dir,
        zero_sample_rate=args.zero_sample_rate,
        seed=args.seed,
        coverage_threshold=args.coverage_threshold,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_cells_csv(args.out_dir / "annotation_cells.csv", cells)
    human_a_path = args.out_dir / "human_a_labels.csv"
    human_b_path = args.out_dir / "human_b_labels.csv"
    if not human_a_path.exists():
        write_empty_label_csv(human_a_path, cells, "human_a")
    if not human_b_path.exists():
        write_empty_label_csv(human_b_path, cells, "human_b")

    counts = {
        "package_dir": str(args.package_dir),
        "audit_cells": len(cells),
        "gemini_positive_cells": sum(1 for cell in cells if cell.sampling_stratum == "gemini_positive"),
        "sampled_gemini_zero_cells": sum(1 for cell in cells if cell.sampling_stratum == "sampled_gemini_zero"),
        "instances": len({cell.instance_id for cell in cells}),
        "coverage_threshold": args.coverage_threshold,
        "zero_sample_rate": args.zero_sample_rate,
        "seed": args.seed,
    }
    write_json(args.out_dir / "audit_manifest.json", counts)

    readme = [
        "# Human Coverage Audit",
        "",
        "Fill `human_a_labels.csv` and `human_b_labels.csv` with binary labels.",
        "Do not edit `annotation_cells.csv`; it is the immutable sampling frame.",
        "",
        "Cells include all nonzero Gemini package coverage cells and a deterministic stratified sample of Gemini-zero cells.",
        "The same schema definition used for model annotation should be used by human annotators.",
        "",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    labels_a = load_labels(args.out_dir / "human_a_labels.csv")
    labels_b = load_labels(args.out_dir / "human_b_labels.csv")
    real_metrics = agreement_metrics(cells, labels_a, labels_b)
    real_stability = ratio_stability(
        args.package_dir,
        args.raw_results,
        coverage_overrides(cells, labels_a, labels_b),
        methods={"oracle_gvt", "estimated_gvt", "memgpt_tiered", "mem0_extract", "amem_graph"},
    )
    real_report = {
        "mode": "real_labels",
        "note": "Metrics are computed from human_a_labels.csv and human_b_labels.csv.",
        "agreement": real_metrics,
        "ratio_stability": real_stability,
    }
    write_json(args.out_dir / "human_audit_metrics.json", real_report)
    if real_stability.get("summary_rows"):
        write_stability_csv(args.out_dir / "ratio_stability_package_rows.csv", real_stability["summary_rows"])

    print(json.dumps({"out_dir": str(args.out_dir), **counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
