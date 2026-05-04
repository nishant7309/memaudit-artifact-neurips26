"""Cross-model coverage-label validation for the Natural-87 package.

This script compares the frozen Gemini adjudicated package, a second model
adjudicated package (Claude Sonnet 4.5 in the current run), and the released
human coverage audit. It reports:

* model-vs-model coverage agreement on the human-audit sampling frame;
* model-vs-human agreement on human-labeled cells;
* package-ratio stability and ranking preservation between model labels;
* bootstrap confidence intervals for Gemini and human-label package ratios;
* a small disagreement breakdown for the human audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import CandidateMemory, objective_value, selected_candidates, solve_exact
from llm_memory_validation.run_mem0_natural_baseline import load_package, package_instance, resolved_queries


DEFAULT_GEMINI_PACKAGE = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package")
DEFAULT_CLAUDE_PACKAGE = Path("llm_memory_validation/natural_adjudicated_100_claude_sonnet45/coverage_package")
DEFAULT_HUMAN_AUDIT = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit")
DEFAULT_CLAUDE_CELL_LABELS = Path("llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cell_coverage_audit/claude_cell_labels.csv")
DEFAULT_GEMINI_RAW = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/raw_results.jsonl")
DEFAULT_CLAUDE_RAW = Path("llm_memory_validation/natural_adjudicated_100_claude_sonnet45/raw_results.jsonl")
DEFAULT_OUT = Path("llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cross_model_check")
DEFAULT_METHODS = ("oracle_gvt", "estimated_gvt", "memgpt_tiered", "mem0_extract", "amem_graph")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in load_csv_rows(path):
        raw = str(row.get("label", "")).strip().lower()
        if raw in {"1", "yes", "y", "true"}:
            labels[str(row["cell_id"])] = 1
        elif raw in {"0", "no", "n", "false"}:
            labels[str(row["cell_id"])] = 0
    return labels


def load_coverage_labels(package_dir: Path, threshold: float) -> dict[tuple[str, str], int]:
    labels: dict[tuple[str, str], int] = {}
    for row in read_jsonl(package_dir / "coverage_matrix.jsonl"):
        candidate_id = str(row["candidate_id"])
        unit_id = str(row["unit_id"])
        value = float(row.get("coverage", row.get("fidelity", 0.0)) or 0.0)
        if value >= threshold:
            labels[(candidate_id, unit_id)] = 1
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
    expected = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else 0.0
    return (observed - expected) / (1.0 - expected)


def raw_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    if not a:
        return float("nan")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = sum((x - mean_a) ** 2 for x in a)
    den_b = sum((y - mean_b) ** 2 for y in b)
    if den_a <= 0 or den_b <= 0:
        return 1.0 if list(a) == list(b) else 0.0
    return num / ((den_a * den_b) ** 0.5)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(rankdata(a), rankdata(b))


def prefix_of(item_id: str) -> str:
    return str(item_id).split("::", 1)[0]


def consensus_overrides(cells: Sequence[Mapping[str, str]], labels_a: Mapping[str, int], labels_b: Mapping[str, int]) -> dict[tuple[str, str], int]:
    overrides: dict[tuple[str, str], int] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        if cell_id not in labels_a or cell_id not in labels_b:
            continue
        if labels_a[cell_id] != labels_b[cell_id]:
            continue
        overrides[(str(cell["candidate_id"]), str(cell["unit_id"]))] = int(labels_a[cell_id])
    return overrides


def humanized_package_instance(package_dir: Path, query: Mapping[str, Any], overrides: Mapping[tuple[str, str], int]):
    data = load_package(package_dir)
    base = package_instance(data, query)
    candidates: list[CandidateMemory] = []
    for candidate in base.candidates:
        coverage = dict(candidate.coverage)
        for (candidate_id, unit_id), label in overrides.items():
            if candidate_id != candidate.candidate_id:
                continue
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


def recompute_human_ratio_rows(
    package_dir: Path,
    raw_results: Path,
    overrides: Mapping[tuple[str, str], int],
    methods: set[str],
    solver: str,
) -> list[dict[str, Any]]:
    data = load_package(package_dir)
    query_by_id = {str(query["query_id"]): query for query in resolved_queries(data, None)}
    instance_cache: dict[tuple[str, int], Any] = {}
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(raw_results):
        method = str(row.get("method", ""))
        if method == "opt" or method not in methods:
            continue
        instance_id = str(row.get("instance_id", ""))
        if instance_id not in query_by_id:
            continue
        budget = int(row.get("budget", 0))
        key = (instance_id, budget)
        if key not in instance_cache:
            instance_cache[key] = humanized_package_instance(package_dir, query_by_id[instance_id], overrides)
        instance = instance_cache[key]
        exact = solve_exact(instance, budget, solver=solver)
        selected_ids = [str(candidate_id) for candidate_id in row.get("selected_candidate_ids", [])]
        selected = selected_candidates(instance.candidates, selected_ids)
        value = objective_value(selected, instance.unit_weights)
        if exact.objective_value <= 0:
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "budget": budget,
                "method": method,
                "label_source": "human_consensus",
                "ratio": value / exact.objective_value,
                "objective_value": value,
                "optimum_value": exact.objective_value,
            }
        )
    return rows


def model_ratio_rows(raw_results: Path, methods: set[str], label_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(raw_results):
        method = str(row.get("method", ""))
        if method == "opt" or method not in methods:
            continue
        ratio = row.get("ratio_to_opt")
        if ratio is None:
            continue
        rows.append(
            {
                "instance_id": str(row.get("instance_id", "")),
                "budget": int(row.get("budget", 0)),
                "method": method,
                "label_source": label_source,
                "ratio": float(ratio),
                "objective_value": row.get("objective_value"),
                "optimum_value": row.get("optimum_value"),
            }
        )
    return rows


def bootstrap_ci(values: Sequence[float], *, seed: int, n_bootstrap: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = [float(v) for v in values]
    means = []
    for _ in range(n_bootstrap):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(statistics.fmean(sample))
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return means[lo_idx], means[hi_idx]


def summarize_ratio_rows(rows: Sequence[Mapping[str, Any]], *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["label_source"]), int(row["budget"]), str(row["method"]))].append(float(row["ratio"]))
    out: list[dict[str, Any]] = []
    for (label_source, budget, method), values in sorted(grouped.items()):
        lo, hi = bootstrap_ci(values, seed=seed + budget + len(method), n_bootstrap=n_bootstrap)
        out.append(
            {
                "label_source": label_source,
                "budget": budget,
                "method": method,
                "n": len(values),
                "mean_ratio": statistics.fmean(values),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
            }
        )
    return out


def compare_ratio_rows(gemini_rows: Sequence[Mapping[str, Any]], claude_rows: Sequence[Mapping[str, Any]], methods: set[str]) -> dict[str, Any]:
    gemini = {
        (str(row["instance_id"]), int(row["budget"]), str(row["method"])): float(row["ratio"])
        for row in gemini_rows
        if str(row["method"]) in methods
    }
    claude = {
        (str(row["instance_id"]), int(row["budget"]), str(row["method"])): float(row["ratio"])
        for row in claude_rows
        if str(row["method"]) in methods
    }
    common_keys = sorted(set(gemini) & set(claude))
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key in common_keys:
        instance_id, budget, method = key
        by_budget[budget].append(
            {
                "instance_id": instance_id,
                "budget": budget,
                "method": method,
                "gemini_ratio": gemini[key],
                "claude_ratio": claude[key],
                "abs_diff": abs(gemini[key] - claude[key]),
            }
        )
    budget_rows = []
    ranking_rows = []
    for budget, rows in sorted(by_budget.items()):
        diffs = [float(row["abs_diff"]) for row in rows]
        budget_rows.append(
            {
                "budget": budget,
                "n_rows": len(rows),
                "mean_abs_ratio_diff": statistics.fmean(diffs) if diffs else None,
                "max_abs_ratio_diff": max(diffs) if diffs else None,
            }
        )
        methods_here = sorted({str(row["method"]) for row in rows})
        gemini_means = []
        claude_means = []
        for method in methods_here:
            g_vals = [float(row["gemini_ratio"]) for row in rows if row["method"] == method]
            c_vals = [float(row["claude_ratio"]) for row in rows if row["method"] == method]
            gemini_means.append(statistics.fmean(g_vals))
            claude_means.append(statistics.fmean(c_vals))
        gemini_rank = [m for _, m in sorted(zip(gemini_means, methods_here), reverse=True)]
        claude_rank = [m for _, m in sorted(zip(claude_means, methods_here), reverse=True)]
        ranking_rows.append(
            {
                "budget": budget,
                "methods": methods_here,
                "gemini_rank": gemini_rank,
                "claude_rank": claude_rank,
                "spearman_rho": spearman(gemini_means, claude_means),
                "ranking_preserved": gemini_rank == claude_rank,
            }
        )
    all_diffs = [abs(gemini[key] - claude[key]) for key in common_keys]
    return {
        "common_row_count": len(common_keys),
        "mean_abs_ratio_diff_all": statistics.fmean(all_diffs) if all_diffs else None,
        "budget_rows": budget_rows,
        "ranking_rows": ranking_rows,
        "common_rows": [row for rows in by_budget.values() for row in rows],
    }


def disagreement_breakdown(cells: Sequence[Mapping[str, str]], labels_a: Mapping[str, int], labels_b: Mapping[str, int], gemini_labels: Mapping[tuple[str, str], int]) -> dict[str, Any]:
    h_disagree = []
    hg_disagree = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        if cell_id in labels_a and cell_id in labels_b and labels_a[cell_id] != labels_b[cell_id]:
            h_disagree.append(cell)
        if cell_id in labels_a and cell_id in labels_b and labels_a[cell_id] == labels_b[cell_id]:
            human = labels_a[cell_id]
            gemini = int(gemini_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0))
            if human != gemini:
                hg_disagree.append(cell)

    def summarize(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "by_sampling_stratum": dict(Counter(str(row.get("sampling_stratum", "")) for row in rows).most_common()),
            "by_representation_type": dict(Counter(str(row.get("representation_type", "")) for row in rows).most_common(10)),
            "top_instances": dict(Counter(str(row.get("instance_id", "")) for row in rows).most_common(10)),
        }

    return {
        "human_human_disagreements": summarize(h_disagree),
        "human_gemini_disagreements_on_agreed_human_cells": summarize(hg_disagree),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini-package", type=Path, default=DEFAULT_GEMINI_PACKAGE)
    parser.add_argument("--claude-package", type=Path, default=DEFAULT_CLAUDE_PACKAGE)
    parser.add_argument("--human-audit-dir", type=Path, default=DEFAULT_HUMAN_AUDIT)
    parser.add_argument("--claude-cell-labels", type=Path, default=DEFAULT_CLAUDE_CELL_LABELS)
    parser.add_argument("--gemini-raw-results", type=Path, default=DEFAULT_GEMINI_RAW)
    parser.add_argument("--claude-raw-results", type=Path, default=DEFAULT_CLAUDE_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--solver", default="exact_stdlib")
    args = parser.parse_args(argv)

    methods = {method.strip() for method in args.methods.split(",") if method.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_csv_rows(args.human_audit_dir / "annotation_cells.csv")
    labels_a = load_labels(args.human_audit_dir / "human_a_labels.csv")
    labels_b = load_labels(args.human_audit_dir / "human_b_labels.csv")
    claude_cell_labels = load_labels(args.claude_cell_labels) if args.claude_cell_labels.exists() else {}
    gemini_labels = load_coverage_labels(args.gemini_package, args.coverage_threshold)
    claude_labels = load_coverage_labels(args.claude_package, args.coverage_threshold)

    audit_common = [
        cell
        for cell in cells
        if (str(cell["candidate_id"]), str(cell["unit_id"])) in gemini_labels
        or int(str(cell.get("gemini_label", "0"))) == 0
    ]
    gemini_vec = [int(gemini_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0)) for cell in audit_common]
    claude_vec = [int(claude_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0)) for cell in audit_common]

    human_common = [cell for cell in cells if str(cell["cell_id"]) in labels_a and str(cell["cell_id"]) in labels_b]
    human_a_vec = [labels_a[str(cell["cell_id"])] for cell in human_common]
    human_b_vec = [labels_b[str(cell["cell_id"])] for cell in human_common]
    claude_human_a_vec = [int(claude_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0)) for cell in human_common]

    agreed_human = [
        cell
        for cell in human_common
        if labels_a[str(cell["cell_id"])] == labels_b[str(cell["cell_id"])]
    ]
    consensus_vec = [labels_a[str(cell["cell_id"])] for cell in agreed_human]
    claude_consensus_vec = [int(claude_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0)) for cell in agreed_human]
    gemini_consensus_vec = [int(gemini_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0)) for cell in agreed_human]
    claude_cell_vec = [int(claude_cell_labels.get(str(cell["cell_id"]), 0)) for cell in cells if str(cell["cell_id"]) in claude_cell_labels]
    gemini_for_claude_cell_vec = [
        int(gemini_labels.get((str(cell["candidate_id"]), str(cell["unit_id"])), 0))
        for cell in cells
        if str(cell["cell_id"]) in claude_cell_labels
    ]
    claude_cell_human_common = [
        cell
        for cell in human_common
        if str(cell["cell_id"]) in claude_cell_labels
    ]
    claude_cell_vs_human_a = [int(claude_cell_labels[str(cell["cell_id"])]) for cell in claude_cell_human_common]
    human_a_for_claude_cell = [int(labels_a[str(cell["cell_id"])]) for cell in claude_cell_human_common]
    claude_cell_agreed_human = [
        cell
        for cell in agreed_human
        if str(cell["cell_id"]) in claude_cell_labels
    ]
    claude_cell_vs_consensus = [int(claude_cell_labels[str(cell["cell_id"])]) for cell in claude_cell_agreed_human]
    consensus_for_claude_cell = [int(labels_a[str(cell["cell_id"])]) for cell in claude_cell_agreed_human]

    human_overrides = consensus_overrides(cells, labels_a, labels_b)
    gemini_ratio_rows = model_ratio_rows(args.gemini_raw_results, methods, "gemini")
    claude_ratio_rows = model_ratio_rows(args.claude_raw_results, methods, "claude_sonnet45")
    human_ratio_rows = recompute_human_ratio_rows(
        args.gemini_package,
        args.gemini_raw_results,
        human_overrides,
        methods,
        args.solver,
    )
    ratio_comparison = compare_ratio_rows(gemini_ratio_rows, claude_ratio_rows, methods)
    ci_rows = summarize_ratio_rows(
        [*gemini_ratio_rows, *human_ratio_rows],
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    disagreement = disagreement_breakdown(cells, labels_a, labels_b, gemini_labels)

    report = {
        "coverage_threshold": args.coverage_threshold,
        "methods": sorted(methods),
        "model_agreement": {
            "audit_cells": len(audit_common),
            "claude_vs_gemini_kappa": cohen_kappa(claude_vec, gemini_vec),
            "claude_vs_gemini_raw_agreement": raw_agreement(claude_vec, gemini_vec),
            "claude_positive_rate": sum(claude_vec) / len(claude_vec) if claude_vec else None,
            "gemini_positive_rate": sum(gemini_vec) / len(gemini_vec) if gemini_vec else None,
        },
        "human_agreement": {
            "doubly_labeled_cells": len(human_common),
            "human_human_kappa": cohen_kappa(human_a_vec, human_b_vec),
            "human_human_raw_agreement": raw_agreement(human_a_vec, human_b_vec),
            "claude_vs_human_a_kappa": cohen_kappa(claude_human_a_vec, human_a_vec),
            "claude_vs_human_a_raw_agreement": raw_agreement(claude_human_a_vec, human_a_vec),
            "human_agreed_cells": len(agreed_human),
            "claude_vs_agreed_human_kappa": cohen_kappa(claude_consensus_vec, consensus_vec),
            "claude_vs_agreed_human_raw_agreement": raw_agreement(claude_consensus_vec, consensus_vec),
            "gemini_vs_agreed_human_kappa": cohen_kappa(gemini_consensus_vec, consensus_vec),
            "gemini_vs_agreed_human_raw_agreement": raw_agreement(gemini_consensus_vec, consensus_vec),
        },
        "blind_cell_model_agreement": {
            "claude_cell_labels_path": str(args.claude_cell_labels),
            "claude_cell_labeled_cells": len(claude_cell_vec),
            "claude_cell_vs_gemini_kappa": cohen_kappa(claude_cell_vec, gemini_for_claude_cell_vec),
            "claude_cell_vs_gemini_raw_agreement": raw_agreement(claude_cell_vec, gemini_for_claude_cell_vec),
            "claude_cell_vs_human_a_kappa": cohen_kappa(claude_cell_vs_human_a, human_a_for_claude_cell),
            "claude_cell_vs_human_a_raw_agreement": raw_agreement(claude_cell_vs_human_a, human_a_for_claude_cell),
            "claude_cell_vs_agreed_human_kappa": cohen_kappa(claude_cell_vs_consensus, consensus_for_claude_cell),
            "claude_cell_vs_agreed_human_raw_agreement": raw_agreement(claude_cell_vs_consensus, consensus_for_claude_cell),
            "claude_cell_positive_rate": (sum(claude_cell_vec) / len(claude_cell_vec)) if claude_cell_vec else None,
        },
        "ratio_comparison": {
            key: value
            for key, value in ratio_comparison.items()
            if key != "common_rows"
        },
        "disagreement_breakdown": disagreement,
        "bootstrap": {
            "n_bootstrap": args.bootstrap,
            "ratio_ci_rows_path": str(args.out_dir / "ratio_bootstrap_ci.csv"),
        },
    }

    write_json(args.out_dir / "cross_model_metrics.json", report)
    write_csv(
        args.out_dir / "ratio_bootstrap_ci.csv",
        ci_rows,
        ["label_source", "budget", "method", "n", "mean_ratio", "bootstrap95_low", "bootstrap95_high"],
    )
    write_csv(
        args.out_dir / "gemini_vs_claude_ratio_rows.csv",
        ratio_comparison["common_rows"],
        ["instance_id", "budget", "method", "gemini_ratio", "claude_ratio", "abs_diff"],
    )
    write_csv(
        args.out_dir / "gemini_vs_claude_ratio_summary.csv",
        ratio_comparison["budget_rows"],
        ["budget", "n_rows", "mean_abs_ratio_diff", "max_abs_ratio_diff"],
    )
    write_csv(
        args.out_dir / "ranking_stability.csv",
        ratio_comparison["ranking_rows"],
        ["budget", "methods", "gemini_rank", "claude_rank", "spearman_rho", "ranking_preserved"],
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
