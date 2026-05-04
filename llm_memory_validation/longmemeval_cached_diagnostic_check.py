from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("llm_memory_validation/longmemeval_cached_diagnostic_check")
RETRIEVAL_SUMMARY = Path("llm_memory_validation/longmemeval_focus_report_core4/summary.json")
GPT55_READER_SUMMARY = Path("llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json")
GPT55_READER_OUTPUTS = Path("llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/reader_outputs.jsonl")
GPT55_NORMALIZED = Path("llm_memory_validation/scoring_audit_gpt55/normalized_scoring_v2.json")
GPT55_FAILURE_BUCKETS = Path(
    "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/failure_bucket_counts.json"
)
GEMINI_SUMMARY = Path("llm_memory_validation/longmemeval_reader_api_gemini31_flash_lite_focus_full_bsc_fifo/summary.json")
GPT54_MINI_SUMMARY = Path("llm_memory_validation/longmemeval_reader_api_gpt54mini_focus_full/summary.json")
PROMPT_DEV_SUMMARY = Path("llm_memory_validation/reader_prompt_dev_gpt55/prompt_comparison_summary.json")

ORACLE = "dense_budgeted_bsc"
FULL_RAW = "dense_rag_e5"
RAW_REPLAY = "dense_budgeted_replay"
FIFO = "fifo_replay"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def rate(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def ci(values: list[float]) -> str:
    return f"[{signed(values[0])}, {signed(values[1])}]"


def focus(summary: dict[str, Any], method: str) -> dict[str, Any]:
    return summary["metrics"][method]["focus"]


def paired(summary: dict[str, Any], baseline: str, metric: str) -> dict[str, Any]:
    return summary["metrics"]["_paired_focus_deltas_vs_oraclemem_dense"][baseline][metric]


def normalized_focus(normalized: dict[str, Any], method: str) -> dict[str, Any]:
    return normalized["method_summary"][method]["focus"]


def percent_less(smaller: float, larger: float) -> float:
    if larger == 0:
        return 0.0
    return 1.0 - smaller / larger


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def build_summary() -> dict[str, Any]:
    required_paths = [
        RETRIEVAL_SUMMARY,
        GPT55_READER_SUMMARY,
        GPT55_READER_OUTPUTS,
        GPT55_NORMALIZED,
        GPT55_FAILURE_BUCKETS,
        PROMPT_DEV_SUMMARY,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required cached artifacts: " + ", ".join(missing))

    retrieval = read_json(RETRIEVAL_SUMMARY)
    gpt55 = read_json(GPT55_READER_SUMMARY)
    normalized = read_json(GPT55_NORMALIZED)
    failures = read_json(GPT55_FAILURE_BUCKETS)
    prompt_dev = read_json(PROMPT_DEV_SUMMARY)
    gemini = optional_json(GEMINI_SUMMARY)
    gpt54 = optional_json(GPT54_MINI_SUMMARY)

    oracle_reader = focus(gpt55, ORACLE)
    full_reader = focus(gpt55, FULL_RAW)
    oracle_norm = normalized_focus(normalized, ORACLE)
    full_norm = normalized_focus(normalized, FULL_RAW)
    oracle_retrieval = retrieval["metrics"][ORACLE]
    full_retrieval = retrieval["metrics"][FULL_RAW]
    oracle_failures = failures["by_method"][ORACLE]
    full_failures = failures["by_method"][FULL_RAW]

    f1_delta = paired(gpt55, FULL_RAW, "token_f1")
    evidence_delta = paired(gpt55, FULL_RAW, "evidence_use")
    em_delta = paired(gpt55, FULL_RAW, "exact_match")

    prompt_candidates = prompt_dev["selection"]["candidates"]
    eligible_prompts = [row["prompt_mode"] for row in prompt_candidates if row.get("eligible")]

    summary: dict[str, Any] = {
        "scope": "cached-only LongMemEval-S diagnostic check; no model or API calls",
        "inputs": {
            "retrieval_summary": str(RETRIEVAL_SUMMARY),
            "gpt55_reader_summary": str(GPT55_READER_SUMMARY),
            "gpt55_reader_outputs": str(GPT55_READER_OUTPUTS),
            "gpt55_normalized_scoring": str(GPT55_NORMALIZED),
            "gpt55_failure_buckets": str(GPT55_FAILURE_BUCKETS),
            "gemini_summary": str(GEMINI_SUMMARY) if gemini else None,
            "gpt54_mini_summary": str(GPT54_MINI_SUMMARY) if gpt54 else None,
            "prompt_dev_summary": str(PROMPT_DEV_SUMMARY),
        },
        "row_counts": {
            "gpt55_reader_outputs_jsonl": count_jsonl(GPT55_READER_OUTPUTS),
            "focus_questions": int(oracle_reader["n"]),
            "reader_methods": len(gpt55["methods"]),
        },
        "retrieval_focus": {
            "oraclemem_r_at_5": oracle_retrieval["focus_recall_at_5"],
            "full_raw_r_at_5": full_retrieval["focus_recall_at_5"],
            "delta_vs_full_raw": oracle_retrieval["delta_focus_vs_full_dense_rag"],
            "basis": retrieval["metric_basis"],
        },
        "gpt55_focus": {
            "oraclemem_raw_em": oracle_reader["exact_match"],
            "full_raw_raw_em": full_reader["exact_match"],
            "raw_em_delta_vs_full_raw": em_delta["mean_delta"],
            "raw_em_delta_ci95": em_delta["ci95"],
            "oraclemem_normalized_em": oracle_norm["normalized_em"],
            "full_raw_normalized_em": full_norm["normalized_em"],
            "normalized_em_delta_vs_full_raw": oracle_norm["normalized_em"] - full_norm["normalized_em"],
            "oraclemem_f1": oracle_reader["token_f1"],
            "full_raw_f1": full_reader["token_f1"],
            "f1_delta_vs_full_raw": f1_delta["mean_delta"],
            "f1_delta_ci95": f1_delta["ci95"],
            "oraclemem_evidence_use": oracle_reader["evidence_use"],
            "full_raw_evidence_use": full_reader["evidence_use"],
            "evidence_use_delta_vs_full_raw": evidence_delta["mean_delta"],
            "evidence_use_delta_ci95": evidence_delta["ci95"],
            "oraclemem_insufficient_rate": oracle_reader["insufficient_evidence_rate"],
            "full_raw_insufficient_rate": full_reader["insufficient_evidence_rate"],
            "oraclemem_unsupported_rate": oracle_reader["unsupported_answer_rate"],
            "full_raw_unsupported_rate": full_reader["unsupported_answer_rate"],
            "oraclemem_avg_context_words": oracle_reader["avg_context_words"],
            "full_raw_avg_context_words": full_reader["avg_context_words"],
            "oraclemem_context_word_reduction_vs_full_raw": percent_less(
                oracle_reader["avg_context_words"], full_reader["avg_context_words"]
            ),
        },
        "conditional_failure": {
            "oraclemem_gold_retrieved_rate": oracle_failures["conditional_on_gold_retrieved"]["gold_retrieved_rate"],
            "full_raw_gold_retrieved_rate": full_failures["conditional_on_gold_retrieved"]["gold_retrieved_rate"],
            "oraclemem_true_miss_count": oracle_failures["true_miss_count"],
            "full_raw_true_miss_count": full_failures["true_miss_count"],
            "oraclemem_abstain_given_retrieved": oracle_failures["conditional_on_gold_retrieved"][
                "abstain_given_retrieved"
            ],
            "full_raw_abstain_given_retrieved": full_failures["conditional_on_gold_retrieved"][
                "abstain_given_retrieved"
            ],
            "oraclemem_high_f1_em0_candidates": oracle_failures["failure_bucket_counts"][
                "scoring_mismatch_possible"
            ],
            "oraclemem_used_gold_but_wrong": oracle_failures["failure_bucket_counts"]["used_gold_but_wrong"],
        },
        "prompt_dev": {
            "selected_prompt": prompt_dev["selection"]["selected_prompt"],
            "eligible_prompts": eligible_prompts,
            "interpretation": "No calibrated prompt met the predeclared safety criteria.",
        },
        "safe_claims": [
            "LongMemEval-S is a frozen-context diagnostic, not an exact-oracle benchmark and not main answer-accuracy evidence.",
            "On the focus slice, OracleMem improves retrieval R@5 over full raw-store dense retrieval under the cached top-5 protocol.",
            "With the cached GPT-5.5 reader, OracleMem improves token F1 and evidence use over full raw-store dense retrieval.",
            "OracleMem's exact-match gain over full raw-store dense retrieval is small and not statistically significant.",
            "Remaining LongMemEval-S failures include substantial reader over-abstention and answer-extraction errors after gold evidence is already in context.",
        ],
        "unsafe_claims": [
            "Do not claim significant exact-answer accuracy improvement over full raw-store dense retrieval.",
            "Do not call LongMemEval-S scores oracle ratios or evidence of exact memory optimality.",
            "Do not claim broad deployed memory-system superiority over full-store/native memory systems.",
            "Do not claim the prompt-calibration pass produced a safe calibrated-reader win.",
        ],
    }

    if gemini is not None:
        gemini_delta = paired(gemini, FIFO, "token_f1")
        gemini_evidence_delta = paired(gemini, FIFO, "evidence_use")
        summary["gemini_focus_diagnostic"] = {
            "methods": gemini["methods"],
            "oraclemem_em": focus(gemini, ORACLE)["exact_match"],
            "fifo_em": focus(gemini, FIFO)["exact_match"],
            "f1_delta_vs_fifo": gemini_delta["mean_delta"],
            "f1_delta_ci95": gemini_delta["ci95"],
            "evidence_use_delta_vs_fifo": gemini_evidence_delta["mean_delta"],
            "evidence_use_delta_ci95": gemini_evidence_delta["ci95"],
            "note": "Gemini diagnostic compares OracleMem only to FIFO; EM is zero for both.",
        }

    if gpt54 is not None:
        gpt54_em = paired(gpt54, FULL_RAW, "exact_match")
        gpt54_f1 = paired(gpt54, FULL_RAW, "token_f1")
        gpt54_evidence = paired(gpt54, FULL_RAW, "evidence_use")
        summary["gpt54_mini_focus_diagnostic"] = {
            "oraclemem_em": focus(gpt54, ORACLE)["exact_match"],
            "full_raw_em": focus(gpt54, FULL_RAW)["exact_match"],
            "em_delta_vs_full_raw": gpt54_em["mean_delta"],
            "em_delta_ci95": gpt54_em["ci95"],
            "f1_delta_vs_full_raw": gpt54_f1["mean_delta"],
            "f1_delta_ci95": gpt54_f1["ci95"],
            "evidence_use_delta_vs_full_raw": gpt54_evidence["mean_delta"],
            "evidence_use_delta_ci95": gpt54_evidence["ci95"],
            "note": "GPT-5.4-mini repeats the F1/evidence-use direction, but EM is still not significant versus full raw dense.",
        }

    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    gpt55 = summary["gpt55_focus"]
    retrieval = summary["retrieval_focus"]
    failures = summary["conditional_failure"]
    lines = [
        "# LongMemEval-S Cached Diagnostic Check",
        "",
        "- Scope: cached artifacts only; this script makes no model or API calls.",
        "- Verdict: LongMemEval-S should be reported as a diagnostic transfer and reader-bottleneck check, not as main answer-accuracy evidence.",
        f"- Cached rows checked: {summary['row_counts']['gpt55_reader_outputs_jsonl']} reader rows "
        f"({summary['row_counts']['focus_questions']} focus questions x {summary['row_counts']['reader_methods']} methods).",
        "",
        "## Safe Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in summary["safe_claims"])
    lines.extend(["", "## Do Not Claim", ""])
    lines.extend(f"- {claim}" for claim in summary["unsafe_claims"])
    lines.extend(
        [
            "",
            "## Cached Metrics Used",
            "",
            "| Check | OracleMem | Comparator | Delta / note |",
            "|---|---:|---:|---|",
            f"| Retrieval R@5 on focus slice | {rate(retrieval['oraclemem_r_at_5'])} | "
            f"{rate(retrieval['full_raw_r_at_5'])} full raw | "
            f"{signed(retrieval['delta_vs_full_raw'])}; retrieval-only, no answer accuracy |",
            f"| GPT-5.5 raw EM | {rate(gpt55['oraclemem_raw_em'])} | "
            f"{rate(gpt55['full_raw_raw_em'])} full raw | "
            f"{signed(gpt55['raw_em_delta_vs_full_raw'])}, 95% CI "
            f"{ci(gpt55['raw_em_delta_ci95'])}; not significant |",
            f"| GPT-5.5 normalized EM | {rate(gpt55['oraclemem_normalized_em'])} | "
            f"{rate(gpt55['full_raw_normalized_em'])} full raw | "
            f"{signed(gpt55['normalized_em_delta_vs_full_raw'])}; still low absolute accuracy |",
            f"| GPT-5.5 token F1 | {rate(gpt55['oraclemem_f1'])} | "
            f"{rate(gpt55['full_raw_f1'])} full raw | "
            f"{signed(gpt55['f1_delta_vs_full_raw'])}, 95% CI {ci(gpt55['f1_delta_ci95'])} |",
            f"| GPT-5.5 evidence use | {rate(gpt55['oraclemem_evidence_use'])} | "
            f"{rate(gpt55['full_raw_evidence_use'])} full raw | "
            f"{signed(gpt55['evidence_use_delta_vs_full_raw'])}, 95% CI "
            f"{ci(gpt55['evidence_use_delta_ci95'])} |",
            f"| Context words | {rate(gpt55['oraclemem_avg_context_words'])} | "
            f"{rate(gpt55['full_raw_avg_context_words'])} full raw | "
            f"{rate(gpt55['oraclemem_context_word_reduction_vs_full_raw'] * 100.0)}% fewer words |",
            f"| Gold evidence in top-5 | {rate(failures['oraclemem_gold_retrieved_rate'])} | "
            f"{rate(failures['full_raw_gold_retrieved_rate'])} full raw | "
            f"true misses: {failures['oraclemem_true_miss_count']} vs {failures['full_raw_true_miss_count']} |",
            f"| Abstain despite retrieved evidence | {rate(failures['oraclemem_abstain_given_retrieved'])} | "
            f"{rate(failures['full_raw_abstain_given_retrieved'])} full raw | reader-side bottleneck diagnostic |",
            f"| Prompt calibration | {summary['prompt_dev']['selected_prompt']} selected | "
            f"{len(summary['prompt_dev']['eligible_prompts'])} eligible prompts | no calibrated-reader win |",
        ]
    )

    if "gemini_focus_diagnostic" in summary:
        gemini = summary["gemini_focus_diagnostic"]
        lines.extend(
            [
                "",
                "## Optional Reader Robustness",
                "",
                f"- Gemini Flash-Lite diagnostic: OracleMem-vs-FIFO EM was "
                f"{rate(gemini['oraclemem_em'])} vs {rate(gemini['fifo_em'])}; token-F1 delta was "
                f"{signed(gemini['f1_delta_vs_fifo'])} with 95% CI {ci(gemini['f1_delta_ci95'])}; "
                "this supports evidence-use direction only.",
            ]
        )

    if "gpt54_mini_focus_diagnostic" in summary:
        gpt54 = summary["gpt54_mini_focus_diagnostic"]
        lines.append(
            f"- GPT-5.4-mini diagnostic: EM delta versus full raw was "
            f"{signed(gpt54['em_delta_vs_full_raw'])} with 95% CI {ci(gpt54['em_delta_ci95'])}; "
            "F1/evidence-use deltas were positive but this remains an appendix diagnostic."
        )

    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for label, source in summary["inputs"].items():
        if source:
            lines.append(f"- `{label}`: `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cached-only LongMemEval-S diagnostic report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "REPORT.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    write_report(report_path, summary)
    print(json.dumps({"wrote": [str(summary_path), str(report_path)], "api_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
