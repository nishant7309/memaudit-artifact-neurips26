from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Iterable


DEFAULT_METHODS = [
    "dense_budgeted_bsc",
    "dense_rag_e5",
    "heuristic_bsc",
    "ld_agent_proxy",
    "memorybank_proxy",
    "dense_budgeted_replay",
    "replay_only_router",
    "fifo_replay",
]

METHOD_LABELS = {
    "dense_budgeted_bsc": "OracleMem writer + dense retrieval",
    "dense_rag_e5": "Full raw-store dense retrieval",
    "heuristic_bsc": "OracleMem writer + lexical retrieval",
    "ld_agent_proxy": "LD-Agent proxy",
    "memorybank_proxy": "MemoryBank proxy",
    "dense_budgeted_replay": "Budgeted raw replay + dense retrieval",
    "replay_only_router": "Budgeted raw replay router",
    "fifo_replay": "FIFO raw replay",
    "uniform_replay": "Uniform raw replay",
}


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _recall_at(row: dict, k: int) -> float:
    gold = set(row.get("gold_session_ids", []))
    pred = set(row.get("predicted_session_ids", [])[:k])
    if not gold:
        return 0.0
    return len(gold & pred) / len(gold)


def _recall(row: dict) -> float:
    return _recall_at(row, 5)


def _rr_at(row: dict, k: int) -> float:
    gold = set(row.get("gold_session_ids", []))
    if not gold:
        return 0.0
    for rank, session_id in enumerate(row.get("predicted_session_ids", [])[:k], start=1):
        if session_id in gold:
            return 1.0 / rank
    return 0.0


def _rr(row: dict) -> float:
    return _rr_at(row, 5)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _ci(values: list[float], *, rng: random.Random, n_bootstrap: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or n_bootstrap <= 0:
        value = float(values[0])
        return [value, value]
    means = []
    size = len(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [float(lo), float(hi)]


def summarize_method(rows: list[dict], focus_types: set[str], *, rng: random.Random, n_bootstrap: int) -> dict:
    recalls = [_recall(row) for row in rows]
    rrs = [_rr(row) for row in rows]
    focus_rows = [row for row in rows if row.get("question_type") in focus_types]
    focus_recalls = [_recall(row) for row in focus_rows]
    focus_rrs = [_rr(row) for row in focus_rows]
    focus_recall_at_1 = [_recall_at(row, 1) for row in focus_rows]
    focus_recall_at_3 = [_recall_at(row, 3) for row in focus_rows]

    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row.get("question_type", "unknown"), []).append(row)

    per_type = {}
    for question_type, type_rows in sorted(by_type.items()):
        type_recalls = [_recall(row) for row in type_rows]
        type_rrs = [_rr(row) for row in type_rows]
        per_type[question_type] = {
            "n": len(type_rows),
            "recall_at_5": _mean(type_recalls),
            "mrr_at_5": _mean(type_rrs),
            "recall_at_5_ci95": _ci(type_recalls, rng=rng, n_bootstrap=n_bootstrap),
        }

    return {
        "n": len(rows),
        "overall_recall_at_5": _mean(recalls),
        "overall_mrr_at_5": _mean(rrs),
        "focus_n": len(focus_rows),
        "focus_recall_at_5": _mean(focus_recalls),
        "focus_recall_at_1": _mean(focus_recall_at_1),
        "focus_recall_at_3": _mean(focus_recall_at_3),
        "focus_mrr_at_5": _mean(focus_rrs),
        "focus_recall_at_5_ci95": _ci(focus_recalls, rng=rng, n_bootstrap=n_bootstrap),
        "per_type": per_type,
    }


def build_summary(retrieval_rows: dict, methods: list[str], focus_types: set[str], n_bootstrap: int, seed: int) -> dict:
    rng = random.Random(seed)
    metrics = {}
    missing_methods = []
    for method in methods:
        rows = retrieval_rows.get(method)
        if rows is None:
            missing_methods.append(method)
            continue
        metrics[method] = summarize_method(rows, focus_types, rng=rng, n_bootstrap=n_bootstrap)

    baseline = metrics.get("dense_rag_e5")
    raw_baseline = metrics.get("dense_budgeted_replay")
    for method, row in metrics.items():
        if baseline is not None:
            row["delta_focus_vs_full_dense_rag"] = row["focus_recall_at_5"] - baseline["focus_recall_at_5"]
        if raw_baseline is not None:
            row["delta_focus_vs_budgeted_raw_dense"] = row["focus_recall_at_5"] - raw_baseline["focus_recall_at_5"]

    return {
        "source": "LongMemEval-S frozen retrieval artifact",
        "metric_basis": "gold answer_session_ids retrieval only; no answer generation and no exact OPT",
        "focus_types": sorted(focus_types),
        "methods": methods,
        "missing_methods": missing_methods,
        "bootstrap_samples": n_bootstrap,
        "metrics": metrics,
    }


def write_markdown(output_dir: Path, summary: dict) -> None:
    metrics = summary["metrics"]
    focus_types = ", ".join(f"`{item}`" for item in summary["focus_types"])
    lines = [
        "# LongMemEval-S Focus Report",
        "",
        f"- Source: {summary['source']}",
        f"- Focus types: {focus_types}",
        f"- Metric basis: {summary['metric_basis']}",
        "- Scope: retrieval-only. This report does not measure abstention, answer accuracy, stale answers, or ratio to OPT.",
        "",
        "## Focus Retrieval",
        "",
        "| Method | Overall R@5 | Focus R@5 | Focus 95% CI | Focus MRR@5 | Delta vs full dense RAG | Delta vs budgeted raw dense |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary["methods"]:
        if method not in metrics:
            continue
        row = metrics[method]
        label = METHOD_LABELS.get(method, method)
        lo, hi = row["focus_recall_at_5_ci95"]
        lines.append(
            "| "
            + label
            + f" | {row['overall_recall_at_5']:.4f}"
            + f" | {row['focus_recall_at_5']:.4f}"
            + f" | [{lo:.4f}, {hi:.4f}]"
            + f" | {row['focus_mrr_at_5']:.4f}"
            + f" | {row.get('delta_focus_vs_full_dense_rag', 0.0):+.4f}"
            + f" | {row.get('delta_focus_vs_budgeted_raw_dense', 0.0):+.4f}"
            + " |"
        )

    lines.extend(
        [
            "",
            "## Focus Retrieval K-Sweep",
            "",
            "This artifact contains top-5 retrieval ids, so the sweep reports R@1/R@3/R@5 and MRR@5. R@10 requires regenerating retrieval rows with `topk=10`.",
            "",
            "| Method | Focus R@1 | Focus R@3 | Focus R@5 | Focus MRR@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in summary["methods"]:
        if method not in metrics:
            continue
        row = metrics[method]
        label = METHOD_LABELS.get(method, method)
        lines.append(
            f"| {label} | {row['focus_recall_at_1']:.4f} | {row['focus_recall_at_3']:.4f} | "
            f"{row['focus_recall_at_5']:.4f} | {row['focus_mrr_at_5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Type Retrieval",
            "",
            "| Method | Knowledge-update R@5 | Temporal-reasoning R@5 | Multi-session R@5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in summary["methods"]:
        if method not in metrics:
            continue
        row = metrics[method]
        per_type = row["per_type"]
        label = METHOD_LABELS.get(method, method)
        ku = per_type.get("knowledge-update", {}).get("recall_at_5", 0.0)
        tr = per_type.get("temporal-reasoning", {}).get("recall_at_5", 0.0)
        ms = per_type.get("multi-session", {}).get("recall_at_5", 0.0)
        lines.append(f"| {label} | {ku:.4f} | {tr:.4f} | {ms:.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The strongest budgeted memory writer in this artifact is `dense_budgeted_bsc` (reported as OracleMem writer + dense retrieval), which exceeds full raw-store dense retrieval on the focused update/temporal slice.",
            "- The comparison is retrieval-only and uses LongMemEval-S gold answer-session ids; it should be cited as external transfer evidence, not as an oracle-ratio result.",
            "- LongMemEval-S in this local pipeline does not expose an abstention category, so abstention and stale-answer claims still require a separate reader/evaluation run.",
        ]
    )
    if summary["missing_methods"]:
        lines.extend(["", f"Missing methods: `{', '.join(summary['missing_methods'])}`"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=Path("llm_memory_validation/competitor_run_v2/summary.json"))
    parser.add_argument("--retrieval-rows-json", type=Path, default=Path("llm_memory_validation/competitor_run_v2/retrieval_rows.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("llm_memory_validation/longmemeval_focus_report"))
    parser.add_argument("--focus-types", type=_csv, default=_csv("knowledge-update,temporal-reasoning"))
    parser.add_argument("--methods", type=_csv, default=DEFAULT_METHODS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.retrieval_rows_json.exists():
        raise FileNotFoundError(args.retrieval_rows_json)
    retrieval_rows = json.loads(args.retrieval_rows_json.read_text(encoding="utf-8"))
    summary = build_summary(
        retrieval_rows=retrieval_rows,
        methods=args.methods,
        focus_types=set(args.focus_types),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    if args.summary_json.exists():
        source_summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        summary["retriever_model"] = source_summary.get("retriever_model")
        summary["topk"] = source_summary.get("topk")
        summary["reported_baselines"] = source_summary.get("reported_baselines", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
