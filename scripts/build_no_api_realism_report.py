"""Build a no-API realism report from existing OracleMem artifacts.

The report intentionally separates three evidence layers:

* exact-small synthetic oracle evidence, including deployable local baselines;
* cached LongMemEval-S retrieval/reader diagnostics, with no OPT denominator;
* the structural blocker for true non-synthetic OracleMem coverage evidence.

This script reads local JSON artifacts only. It never imports or calls API
reader code.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "amac_admission": "A-MAC-style admission",
    "memgpt_tiered": "MemGPT-style tiered",
    "mem0_extract": "Mem0-style extraction",
    "generic_candidate_opt": "Generic-candidate OPT",
    "no_tombstone_opt": "No-tombstone OPT",
    "summary_candidate_opt": "Summary-only candidate OPT",
    "oracle_gvt": "Oracle-GVT",
    "opt": "Exact OPT",
    "dense_budgeted_bsc": "OracleMem writer + dense retrieval",
    "dense_rag_e5": "Full raw-store dense retrieval",
    "dense_budgeted_replay": "Budgeted raw replay + dense retrieval",
    "fifo_replay": "FIFO raw replay",
}

DEPLOYABLE_METHODS = ("amac_admission", "memgpt_tiered", "mem0_extract")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _ci(row: dict[str, Any]) -> str:
    low = row.get("bootstrap95_ratio_to_opt_low")
    high = row.get("bootstrap95_ratio_to_opt_high")
    if low is None or high is None:
        return "n/a"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _method_rows(local_summary: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in local_summary.get("by_budget_method", []):
        key = (str(row["distribution"]), int(row["budget"]), str(row["method"]))
        rows[key] = row
    return rows


def _local_highlights(local_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _method_rows(local_summary)
    highlights: list[dict[str, Any]] = []
    for distribution in local_summary.get("distributions", []):
        for budget in local_summary.get("budgets", []):
            deployable_rows = [
                rows[(distribution, budget, method)]
                for method in DEPLOYABLE_METHODS
                if (distribution, budget, method) in rows
            ]
            if not deployable_rows:
                continue
            best = max(deployable_rows, key=lambda row: row.get("mean_ratio_to_opt", float("-inf")))
            highlights.append(
                {
                    "distribution": distribution,
                    "budget": budget,
                    "best_deployable_method": best["method"],
                    "best_deployable_ratio": best.get("mean_ratio_to_opt"),
                    "best_deployable_ci95": [
                        best.get("bootstrap95_ratio_to_opt_low"),
                        best.get("bootstrap95_ratio_to_opt_high"),
                    ],
                    "oracle_gvt_ratio": rows.get((distribution, budget, "oracle_gvt"), {}).get(
                        "mean_ratio_to_opt"
                    ),
                    "generic_candidate_opt_ratio": rows.get(
                        (distribution, budget, "generic_candidate_opt"), {}
                    ).get("mean_ratio_to_opt"),
                    "no_tombstone_opt_ratio": rows.get((distribution, budget, "no_tombstone_opt"), {}).get(
                        "mean_ratio_to_opt"
                    ),
                    "summary_candidate_opt_ratio": rows.get(
                        (distribution, budget, "summary_candidate_opt"), {}
                    ).get("mean_ratio_to_opt"),
                }
            )
    return highlights


def _reader_deltas(reader_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = reader_summary.get("metrics", {})
    return metrics.get("dense_budgeted_bsc", {}).get(
        "_paired_focus_deltas_vs_oraclemem_dense",
        metrics.get("_paired_focus_deltas_vs_oraclemem_dense", {}),
    )


def _extract_summary(
    local_summary: dict[str, Any],
    retrieval_summary: dict[str, Any],
    reader_summary: dict[str, Any],
    coverage_audit: dict[str, Any],
) -> dict[str, Any]:
    retrieval_metrics = retrieval_summary.get("metrics", {})
    reader_metrics = reader_summary.get("metrics", {})
    coverage_ready = coverage_audit.get("coverage_ready_artifacts", [])

    return {
        "schema_version": 1,
        "generated_date": _dt.date.today().isoformat(),
        "api_called": False,
        "local_exact_sweep": {
            "num_rows": local_summary.get("num_rows"),
            "distributions": local_summary.get("distributions", []),
            "budgets": local_summary.get("budgets", []),
            "methods": local_summary.get("methods", []),
            "highlights": _local_highlights(local_summary),
        },
        "external_cached_retrieval": {
            "source": retrieval_summary.get("source"),
            "metric_basis": retrieval_summary.get("metric_basis"),
            "topk": retrieval_summary.get("topk"),
            "focus_n": retrieval_metrics.get("dense_budgeted_bsc", {}).get("focus_n"),
            "oraclemem_focus_recall_at_5": retrieval_metrics.get("dense_budgeted_bsc", {}).get(
                "focus_recall_at_5"
            ),
            "full_raw_focus_recall_at_5": retrieval_metrics.get("dense_rag_e5", {}).get(
                "focus_recall_at_5"
            ),
            "oraclemem_minus_full_raw_focus_recall_at_5": retrieval_metrics.get(
                "dense_budgeted_bsc", {}
            ).get("delta_focus_vs_full_dense_rag"),
        },
        "external_cached_reader": {
            "reader": reader_summary.get("reader"),
            "reader_model": reader_summary.get("reader_model"),
            "scope": "cached artifact only; reader was not rerun",
            "focus_n": reader_metrics.get("dense_budgeted_bsc", {}).get("focus", {}).get("n"),
            "oraclemem_focus_f1": reader_metrics.get("dense_budgeted_bsc", {})
            .get("focus", {})
            .get("token_f1"),
            "full_raw_focus_f1": reader_metrics.get("dense_rag_e5", {})
            .get("focus", {})
            .get("token_f1"),
            "oraclemem_focus_evidence_use": reader_metrics.get("dense_budgeted_bsc", {})
            .get("focus", {})
            .get("evidence_use"),
            "full_raw_focus_evidence_use": reader_metrics.get("dense_rag_e5", {})
            .get("focus", {})
            .get("evidence_use"),
            "paired_deltas_vs_full_raw": _reader_deltas(reader_summary).get("dense_rag_e5", {}),
        },
        "non_synthetic_oracle_blocker": {
            "coverage_ready_artifacts": coverage_ready,
            "blocked": len(coverage_ready) == 0,
            "required_package": [
                "experiences.jsonl",
                "evidence_units.jsonl",
                "queries.jsonl with required_unit_ids",
                "candidate_memories.jsonl",
                "coverage_matrix.jsonl",
                "annotation_decisions.jsonl",
                "candidate_generation_manifest.json",
            ],
        },
    }


def _render_local_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Distribution | Budget | Best local deployable writer | Ratio to synthetic OPT | Oracle-GVT | Generic-candidate OPT | No-tombstone OPT |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["local_exact_sweep"]["highlights"]:
        best_label = METHOD_LABELS.get(row["best_deployable_method"], row["best_deployable_method"])
        lines.append(
            "| "
            f"`{row['distribution']}` | {row['budget']} | {best_label} | "
            f"{_fmt(row['best_deployable_ratio'])} {row['best_deployable_ci95'] and '[' + _fmt(row['best_deployable_ci95'][0]) + ', ' + _fmt(row['best_deployable_ci95'][1]) + ']'} | "
            f"{_fmt(row['oracle_gvt_ratio'])} | "
            f"{_fmt(row['generic_candidate_opt_ratio'])} | "
            f"{_fmt(row['no_tombstone_opt_ratio'])} |"
        )
    return lines


def _render_report(summary: dict[str, Any], inputs: dict[str, Path]) -> str:
    retrieval = summary["external_cached_retrieval"]
    reader = summary["external_cached_reader"]
    blocker = summary["non_synthetic_oracle_blocker"]
    paired = reader.get("paired_deltas_vs_full_raw", {})
    f1_delta = paired.get("token_f1", {})
    evidence_delta = paired.get("evidence_use", {})
    em_delta = paired.get("exact_match", {})

    lines = [
        "# No-API Realism Report",
        "",
        f"Date: {summary['generated_date']}",
        "",
        "Scope: local-only report for the reviewer concern that the empirical evidence is too synthetic. This report reads existing artifacts and the 50-seed no-API local sweep; it does not call OpenRouter, OpenAI, embedding services, or any API reader.",
        "",
        "## Verdict",
        "",
        "This strengthens realism in two limited ways: it adds deployable local writer heuristics that do not use oracle coverage to the exact-small benchmark, and it places cached LongMemEval-S retrieval/reader diagnostics beside the synthetic oracle evidence. It does not convert LongMemEval-S into an exact OracleMem benchmark, because the current external artifacts have session-level evidence only and no candidate-by-evidence coverage matrix.",
        "",
        "## Inputs",
        "",
        f"- Local exact-small realism sweep: `{inputs['local_summary'].relative_to(ROOT)}`",
        f"- Cached LongMemEval-S retrieval summary: `{inputs['retrieval_summary'].relative_to(ROOT)}`",
        f"- Cached frozen-context reader summary: `{inputs['reader_summary'].relative_to(ROOT)}`",
        f"- Coverage artifact audit: `{inputs['coverage_audit'].relative_to(ROOT)}`",
        "",
        "## Local No-API Realism Sweep",
        "",
        "Command used for the sweep:",
        "",
        "```powershell",
        "python run_oraclemem_mvp.py --n-seeds 50 --budgets 4,6 --distribution base,update_chain,scope_shift_v2 --methods opt,oracle_gvt,memgpt_tiered,mem0_extract,amac_admission,no_tombstone_opt,generic_candidate_opt,generic_candidate_gvt,summary_candidate_opt --out-dir oraclemem_runs\\no_api_realism_50 --enable-retrieval",
        "```",
        "",
        "These rows are still synthetic exact-small rows, because the hidden evidence units and coverage matrix come from the generator. The realism improvement is narrower: the deployable writer rows are local heuristics that do not read held-out query ids, oracle coverage vectors, oracle marginals, or API estimates.",
        "",
        *_render_local_table(summary),
        "",
        "Interpretation: the local deployable baselines are competitive on some validity/scope cases, especially `update_chain` and `scope_shift_v2`, while candidate-pool ablations still expose a large gap when validity-state records are unavailable. This is a stronger synthetic diagnostic, not natural-trace proof.",
        "",
        "## Cached External Diagnostics",
        "",
        f"LongMemEval-S retrieval remains external but retrieval-only: focus R@5 is {_fmt(retrieval.get('oraclemem_focus_recall_at_5'))} for OracleMem writer + dense retrieval versus {_fmt(retrieval.get('full_raw_focus_recall_at_5'))} for full raw-store dense retrieval, a delta of {_fmt(retrieval.get('oraclemem_minus_full_raw_focus_recall_at_5'))}. These are gold answer-session-id retrieval scores, not answer accuracy and not ratios to OPT.",
        "",
        f"The cached frozen-context reader artifact remains a historical API result, but this report did not rerun it. On the focus slice, OracleMem token F1 is {_fmt(reader.get('oraclemem_focus_f1'))} versus {_fmt(reader.get('full_raw_focus_f1'))} for full raw dense; evidence use is {_fmt(reader.get('oraclemem_focus_evidence_use'))} versus {_fmt(reader.get('full_raw_focus_evidence_use'))}. The paired F1 delta versus full raw is {_fmt(f1_delta.get('mean_delta'))} with CI [{_fmt((f1_delta.get('ci95') or [None, None])[0])}, {_fmt((f1_delta.get('ci95') or [None, None])[1])}], while the exact-match delta is {_fmt(em_delta.get('mean_delta'))} with CI [{_fmt((em_delta.get('ci95') or [None, None])[0])}, {_fmt((em_delta.get('ci95') or [None, None])[1])}].",
        "",
        f"Evidence-use delta versus full raw is {_fmt(evidence_delta.get('mean_delta'))} with CI [{_fmt((evidence_delta.get('ci95') or [None, None])[0])}, {_fmt((evidence_delta.get('ci95') or [None, None])[1])}]. This supports a transfer diagnostic under frozen contexts, not deployed memory-system superiority.",
        "",
        "## Blocker For True Non-Synthetic Oracle Evidence",
        "",
        "The exact remaining blocker is a complete non-synthetic OracleMem coverage package. Current audited artifacts are not coverage-ready, so no non-synthetic `ratio_to_opt` should be reported.",
        "",
        "Required package:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in blocker["required_package"])
    lines.extend(
        [
            "",
            "The package must pass the acceptance gate in `COVERAGE_VALIDATION_PROTOCOL.md`: 100% resolved evidence units, query `required_unit_ids`, candidate groups/costs/text, positive coverage rows with rationales/source spans, no future-source leakage, no forbidden generator inputs, and solver inputs derivable from the artifacts without hidden code defaults.",
            "",
            "## Claim Boundary",
            "",
            "- Synthetic exact-small results can report `ratio_to_opt` only because exact OPT is certified from generated hidden coverage.",
            "- Local deployable-writer rows improve realism inside that synthetic setting but do not prove performance on natural traces.",
            "- LongMemEval-S retrieval/reader rows are external diagnostics with session-level evidence only; they should never be described as exact OracleMem oracle ratios.",
            "- True non-synthetic evidence requires the coverage package above or a clearly named non-OPT reference/upper-bound denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-summary",
        default="oraclemem_runs/no_api_realism_50/summary.json",
        help="Summary JSON from the local no-API deployable-writer sweep.",
    )
    parser.add_argument(
        "--retrieval-summary",
        default="llm_memory_validation/longmemeval_focus_report_core4/summary.json",
        help="Cached LongMemEval-S retrieval summary JSON.",
    )
    parser.add_argument(
        "--reader-summary",
        default="llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json",
        help="Cached frozen-context reader summary JSON.",
    )
    parser.add_argument(
        "--coverage-audit",
        default="llm_memory_validation/coverage_artifact_audit/summary.json",
        help="Coverage artifact audit summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="oraclemem_runs/no_api_realism_50",
        help="Directory for REPORT.md and realism_summary.json.",
    )
    args = parser.parse_args()

    inputs = {
        "local_summary": _path(args.local_summary),
        "retrieval_summary": _path(args.retrieval_summary),
        "reader_summary": _path(args.reader_summary),
        "coverage_audit": _path(args.coverage_audit),
    }

    summary = _extract_summary(
        _load_json(inputs["local_summary"]),
        _load_json(inputs["retrieval_summary"]),
        _load_json(inputs["reader_summary"]),
        _load_json(inputs["coverage_audit"]),
    )
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "realism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(_render_report(summary, inputs), encoding="utf-8")
    print(f"wrote {output_dir / 'REPORT.md'}")
    print(f"wrote {output_dir / 'realism_summary.json'}")


if __name__ == "__main__":
    main()
