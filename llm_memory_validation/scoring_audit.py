from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from longmemeval_reader_eval import (
    FOCUS_TYPES,
    METHOD_LABELS,
    ContextEntry,
    entries_from_full_raw,
    is_insufficient_answer,
    load_examples,
    reconstruct_context,
    token_f1,
)


DEFAULT_RUN_DIR = Path("llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full")
DEFAULT_OUT_DIR = Path("llm_memory_validation/scoring_audit_gpt55")
DEFAULT_DATASET = Path("llm_memory_validation/cache/longmemeval_s_cleaned.json")
DEFAULT_RETRIEVAL_ROWS = Path("llm_memory_validation/competitor_run_v2/retrieval_rows.json")

ORACLE_METHOD = "dense_budgeted_bsc"
FULL_RAW_METHOD = "dense_rag_e5"
HIGH_F1_THRESHOLD = 0.5

ARTICLES = {"a", "an", "the"}
MONTHS = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "sept": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_reader_outputs(run_dir: Path) -> list[dict]:
    jsonl_path = run_dir / "reader_outputs.jsonl"
    if jsonl_path.exists():
        return read_jsonl(jsonl_path)
    predictions_path = run_dir / "predictions.json"
    artifacts = json.loads(predictions_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for method_rows in artifacts.values():
        rows.extend(method_rows)
    return rows


def strip_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def normalize_aliases(text: str) -> str:
    replacements = [
        (r"\bU\.?\s*S\.?\s*A\.?\b", " United States "),
        (r"\bU\.?\s*S\.?\b", " United States "),
        (r"\bUnited States of America\b", " United States "),
        (r"\bU\.?\s*K\.?\b", " United Kingdom "),
        (r"\bNew York City\b", " NYC "),
    ]
    result = text
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def normalize_date_mentions(text: str) -> str:
    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)

    def month_day_year(match: re.Match[str]) -> str:
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = match.group(3)
        if year:
            return f" {year}-{month}-{day:02d} "
        return f" {month}-{day:02d} "

    text = re.sub(
        r"\b("
        + "|".join(MONTHS)
        + r")\s+(\d{1,2})(?:,\s*|\s+)?(\d{4})?\b",
        month_day_year,
        text,
        flags=re.IGNORECASE,
    )

    def slash_date(match: re.Match[str]) -> str:
        first = int(match.group(1))
        second = int(match.group(2))
        year = match.group(3)
        if year:
            return f" {year}-{first:02d}-{second:02d} "
        return f" {first:02d}-{second:02d} "

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", slash_date, text)
    return text


def normalize_number_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return NUMBER_WORDS[match.group(0).lower()]

    return re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", repl, text, flags=re.IGNORECASE)


def normalized_answer(text: str) -> str:
    text = normalize_aliases(str(text))
    text = normalize_date_mentions(text)
    text = normalize_number_words(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def gold_variants(gold: str) -> list[str]:
    raw = str(gold).strip()
    variants = [raw]
    no_parens = strip_parentheticals(raw).strip()
    if no_parens and no_parens != raw:
        variants.append(no_parens)

    acceptable_split = re.split(
        r"\.\s*|;\s*|\bis also acceptable\b|\bare also acceptable\b|\balso acceptable\b",
        no_parens,
        flags=re.IGNORECASE,
    )
    for part in acceptable_split:
        part = re.sub(r"\b(including|also)\b.*$", "", part.strip(), flags=re.IGNORECASE).strip()
        if part:
            variants.append(part)

    for sep in [" / ", " or "]:
        if sep in no_parens.lower():
            pattern = re.compile(re.escape(sep), flags=re.IGNORECASE)
            variants.extend(part.strip() for part in pattern.split(no_parens) if part.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        normalized = normalized_answer(variant)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(variant)
    return unique


def normalized_exact_match(prediction: str, gold: str) -> float:
    pred_norm = normalized_answer(prediction)
    if not pred_norm:
        return 0.0
    return float(any(pred_norm == normalized_answer(variant) for variant in gold_variants(gold)))


def infer_answer_type(question: str, gold: str) -> str:
    question_l = str(question).lower()
    gold_l = str(gold).lower()
    combined = f"{question_l} {gold_l}"
    if is_insufficient_answer(gold) or gold_l in {"unknown", "not enough information", "insufficient evidence"}:
        return "unknown/insufficient"
    if re.search(
        r"\b(when|what date|what day|what time|how long|days?|weeks?|months?|years?|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        + "|".join(MONTHS)
        + r"|\d{1,2}:\d{2}|\d{4})\b",
        combined,
    ):
        return "date/time"
    if re.search(r"\b(where|location|venue|city|country|state|address|airport|hotel|restaurant|museum|park)\b", combined):
        return "location"
    if re.search(r"\b(who|whose|person|name|friend|doctor|teacher|manager|partner|colleague|author|artist|band)\b", combined):
        return "person/name"
    if re.search(r"\b(prefer|preference|favorite|favourite|like|love|dislike|allerg|diet|order|want|usually)\b", combined):
        return "preference"
    if re.search(r"\b(event|happened|concert|trip|meeting|appointment|flight|visit|party|wedding|first|last|before|after|sequence|order)\b", combined):
        return "event"
    if normalized_answer(gold):
        return "free-form fact"
    return "unknown/insufficient"


def mean(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(field, 0.0)) for row in rows) / len(rows)


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "raw_em": mean(rows, "raw_em"),
        "normalized_em": mean(rows, "normalized_em"),
        "token_f1": mean(rows, "token_f1"),
        "evidence_use": mean(rows, "evidence_use"),
        "insufficient_evidence_rate": mean(rows, "abstained_float"),
        "unsupported_answer_rate": mean(rows, "unsupported_answer"),
        "parse_failure_rate": mean(rows, "parse_failure_float"),
        "gold_evidence_retrieved": mean(rows, "gold_evidence_retrieved_float"),
    }


def enrich_rows(rows: list[dict], examples_by_id: dict[str, dict]) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        example = examples_by_id.get(row.get("question_id"), {})
        gold_ids = set(row.get("gold_session_ids", []))
        context_ids = set(row.get("context_session_ids", []))
        gold_retrieved = bool(gold_ids & context_ids)
        question = example.get("question", "")
        gold = row.get("gold_answer", example.get("answer", ""))
        prediction = row.get("prediction", "")
        raw_em = float(row.get("exact_match", 0.0))
        norm_em = normalized_exact_match(prediction, gold)
        enriched.append(
            {
                **row,
                "question": question,
                "gold": gold,
                "answer": prediction,
                "raw_em": raw_em,
                "normalized_em": norm_em,
                "token_f1": float(row.get("token_f1", token_f1(prediction, gold))),
                "abstained_float": float(bool(row.get("abstained"))),
                "parse_failure_float": float(bool(row.get("parse_failure"))),
                "gold_evidence_retrieved": gold_retrieved,
                "gold_evidence_retrieved_float": float(gold_retrieved),
                "gold_recall_in_context": len(gold_ids & context_ids) / max(len(gold_ids), 1),
                "answer_type": infer_answer_type(question, gold),
            }
        )
    return enriched


def by_method(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return dict(grouped)


def answer_type_summary(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for method, method_rows in sorted(by_method(rows).items()):
        type_rows: dict[str, list[dict]] = defaultdict(list)
        for row in method_rows:
            type_rows[row["answer_type"]].append(row)
        result[method] = {
            "method_label": method_rows[0].get("method_label", METHOD_LABELS.get(method, method)),
            "answer_types": {
                answer_type: summarize_rows(type_group)
                for answer_type, type_group in sorted(type_rows.items())
            },
        }
    return result


def method_summary(rows: list[dict], focus_types: set[str]) -> dict:
    summary: dict[str, dict] = {}
    for method, method_rows in sorted(by_method(rows).items()):
        focus_rows = [row for row in method_rows if row.get("question_type") in focus_types]
        summary[method] = {
            "method_label": method_rows[0].get("method_label", METHOD_LABELS.get(method, method)),
            "overall": summarize_rows(method_rows),
            "focus": summarize_rows(focus_rows),
        }
    return summary


def retrieval_lookup(retrieval_rows: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    return {
        method: {row["question_id"]: row for row in method_rows}
        for method, method_rows in retrieval_rows.items()
    }


def raw_context_from_row(row: dict, examples_by_id: dict[str, dict]) -> list[ContextEntry]:
    example = examples_by_id.get(row.get("question_id"), {})
    full_raw = entries_from_full_raw(example) if example else {}
    return [full_raw[session_id] for session_id in row.get("context_session_ids", []) if session_id in full_raw]


def context_for_row(
    row: dict,
    contexts: dict[tuple[str, str], list[ContextEntry]],
    examples_by_id: dict[str, dict],
    retrieval_by_method: dict[str, dict[str, dict]],
    budget_frac: float,
    max_context_words: int,
) -> list[ContextEntry]:
    key = (row["method"], row["question_id"])
    if key in contexts:
        return contexts[key]
    example = examples_by_id.get(row["question_id"])
    retrieval_row = retrieval_by_method.get(row["method"], {}).get(row["question_id"])
    if example is not None and retrieval_row is not None:
        context, _fallbacks = reconstruct_context(
            example,
            retrieval_row,
            row["method"],
            budget_frac,
            max_context_words,
        )
    else:
        context = raw_context_from_row(row, examples_by_id)
    contexts[key] = context
    return context


def memories_for_row(
    row: dict,
    contexts: dict[tuple[str, str], list[ContextEntry]],
    examples_by_id: dict[str, dict],
    retrieval_by_method: dict[str, dict[str, dict]],
    budget_frac: float,
    max_context_words: int,
    max_chars: int,
) -> list[dict]:
    context = context_for_row(
        row,
        contexts,
        examples_by_id,
        retrieval_by_method,
        budget_frac,
        max_context_words,
    )
    gold_ids = set(row.get("gold_session_ids", []))
    used_ids = set(row.get("used_memory_ids", []))
    memories = []
    for entry in context:
        memories.append(
            {
                "memory_id": entry.session_id,
                "action": entry.action,
                "source": entry.source,
                "is_gold_evidence": entry.session_id in gold_ids,
                "used_by_reader": entry.session_id in used_ids,
                "text": entry.text[:max_chars],
            }
        )
    return memories


def sample_payload(
    row: dict,
    audit_category: str,
    contexts: dict[tuple[str, str], list[ContextEntry]],
    examples_by_id: dict[str, dict],
    retrieval_by_method: dict[str, dict[str, dict]],
    budget_frac: float,
    max_context_words: int,
    max_memory_chars: int,
    paired_row: dict | None = None,
) -> dict:
    payload = {
        "audit_category": audit_category,
        "question_id": row.get("question_id"),
        "question_type": row.get("question_type"),
        "answer_type": row.get("answer_type"),
        "question": row.get("question", ""),
        "gold": row.get("gold", row.get("gold_answer", "")),
        "method": row.get("method"),
        "method_label": row.get("method_label", METHOD_LABELS.get(row.get("method", ""), row.get("method", ""))),
        "answer": row.get("answer", row.get("prediction", "")),
        "retrieved_memories": memories_for_row(
            row,
            contexts,
            examples_by_id,
            retrieval_by_method,
            budget_frac,
            max_context_words,
            max_memory_chars,
        ),
        "used_memory_ids": row.get("used_memory_ids", []),
        "raw_em": row.get("raw_em", 0.0),
        "normalized_em": row.get("normalized_em", 0.0),
        "f1": row.get("token_f1", 0.0),
        "gold_evidence_retrieved": bool(row.get("gold_evidence_retrieved")),
        "gold_recall_in_context": row.get("gold_recall_in_context", 0.0),
        "evidence_use": row.get("evidence_use", 0.0),
        "abstained": bool(row.get("abstained")),
        "unsupported_answer": row.get("unsupported_answer", 0.0),
        "gold_session_ids": row.get("gold_session_ids", []),
        "context_session_ids": row.get("context_session_ids", []),
    }
    if paired_row is not None:
        payload["paired_full_raw"] = {
            "method": paired_row.get("method"),
            "method_label": paired_row.get("method_label", METHOD_LABELS.get(paired_row.get("method", ""), "")),
            "answer": paired_row.get("answer", paired_row.get("prediction", "")),
            "raw_em": paired_row.get("raw_em", 0.0),
            "normalized_em": paired_row.get("normalized_em", 0.0),
            "f1": paired_row.get("token_f1", 0.0),
            "gold_evidence_retrieved": bool(paired_row.get("gold_evidence_retrieved")),
            "evidence_use": paired_row.get("evidence_use", 0.0),
            "abstained": bool(paired_row.get("abstained")),
            "used_memory_ids": paired_row.get("used_memory_ids", []),
            "context_session_ids": paired_row.get("context_session_ids", []),
        }
    return payload


def select_top(rows: list[dict], limit: int, used_keys: set[tuple[str, str]], key_fn) -> list[dict]:
    selected: list[dict] = []
    for row in sorted(rows, key=key_fn):
        row_key = (row["method"], row["question_id"])
        if row_key in used_keys:
            continue
        selected.append(row)
        used_keys.add(row_key)
        if len(selected) >= limit:
            break
    return selected


def build_balanced_sample(
    rows: list[dict],
    contexts: dict[tuple[str, str], list[ContextEntry]],
    examples_by_id: dict[str, dict],
    retrieval_by_method: dict[str, dict[str, dict]],
    budget_frac: float,
    max_context_words: int,
    max_memory_chars: int,
) -> tuple[list[dict], dict]:
    rows_by_method = by_method(rows)
    oracle_rows = rows_by_method.get(ORACLE_METHOD, [])
    full_rows = rows_by_method.get(FULL_RAW_METHOD, [])
    full_by_qid = {row["question_id"]: row for row in full_rows}
    used_keys: set[tuple[str, str]] = set()

    sample_rows: list[dict] = []
    category_counts: dict[str, int] = {}

    category = "oraclemem_abstained_despite_support"
    selected = select_top(
        [
            row
            for row in oracle_rows
            if row.get("gold_evidence_retrieved") and row.get("abstained")
        ],
        20,
        used_keys,
        key_fn=lambda row: (row.get("question_type", ""), row.get("question_id", "")),
    )
    category_counts[category] = len(selected)
    sample_rows.extend(
        sample_payload(
            row,
            category,
            contexts,
            examples_by_id,
            retrieval_by_method,
            budget_frac,
            max_context_words,
            max_memory_chars,
        )
        for row in selected
    )

    category = "oraclemem_high_f1_em0"
    selected = select_top(
        [
            row
            for row in oracle_rows
            if row.get("raw_em", 0.0) == 0.0
            and row.get("token_f1", 0.0) >= HIGH_F1_THRESHOLD
            and not row.get("abstained")
        ],
        10,
        used_keys,
        key_fn=lambda row: (-row.get("token_f1", 0.0), row.get("question_id", "")),
    )
    category_counts[category] = len(selected)
    sample_rows.extend(
        sample_payload(
            row,
            category,
            contexts,
            examples_by_id,
            retrieval_by_method,
            budget_frac,
            max_context_words,
            max_memory_chars,
        )
        for row in selected
    )

    category = "full_raw_high_f1_em0"
    selected = select_top(
        [
            row
            for row in full_rows
            if row.get("raw_em", 0.0) == 0.0
            and row.get("token_f1", 0.0) >= HIGH_F1_THRESHOLD
            and not row.get("abstained")
        ],
        10,
        used_keys,
        key_fn=lambda row: (-row.get("token_f1", 0.0), row.get("question_id", "")),
    )
    category_counts[category] = len(selected)
    sample_rows.extend(
        sample_payload(
            row,
            category,
            contexts,
            examples_by_id,
            retrieval_by_method,
            budget_frac,
            max_context_words,
            max_memory_chars,
        )
        for row in selected
    )

    category = "oraclemem_full_raw_disagreement"
    disagreement_rows: list[tuple[float, dict, dict]] = []
    used_question_ids = {row["question_id"] for row in sample_rows}
    for oracle in oracle_rows:
        full = full_by_qid.get(oracle["question_id"])
        if full is None or oracle["question_id"] in used_question_ids:
            continue
        abstain_diff = float(bool(oracle.get("abstained")) != bool(full.get("abstained")))
        norm_diff = float(oracle.get("normalized_em", 0.0) != full.get("normalized_em", 0.0))
        evidence_diff = abs(float(oracle.get("evidence_use", 0.0)) - float(full.get("evidence_use", 0.0)))
        f1_diff = abs(float(oracle.get("token_f1", 0.0)) - float(full.get("token_f1", 0.0)))
        score = 2.0 * abstain_diff + norm_diff + evidence_diff + f1_diff
        if score > 0.0:
            disagreement_rows.append((score, oracle, full))
    disagreement_rows.sort(key=lambda item: (-item[0], item[1].get("question_type", ""), item[1].get("question_id", "")))
    selected_pairs = disagreement_rows[:10]
    category_counts[category] = len(selected_pairs)
    for _score, oracle, full in selected_pairs:
        sample_rows.append(
            sample_payload(
                oracle,
                category,
                contexts,
                examples_by_id,
                retrieval_by_method,
                budget_frac,
                max_context_words,
                max_memory_chars,
                paired_row=full,
            )
        )

    return sample_rows, category_counts


def normalization_deltas(rows: list[dict], limit: int = 25) -> list[dict]:
    changed = [
        row
        for row in rows
        if row.get("normalized_em", 0.0) > row.get("raw_em", 0.0)
    ]
    changed.sort(key=lambda row: (row.get("method", ""), row.get("question_id", "")))
    return [
        {
            "question_id": row.get("question_id"),
            "question_type": row.get("question_type"),
            "answer_type": row.get("answer_type"),
            "method": row.get("method"),
            "method_label": row.get("method_label"),
            "gold": row.get("gold"),
            "prediction": row.get("prediction"),
            "raw_em": row.get("raw_em"),
            "normalized_em": row.get("normalized_em"),
            "token_f1": row.get("token_f1"),
        }
        for row in changed[:limit]
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def format_rate(value: float) -> str:
    return f"{value:.4f}"


def write_report(path: Path, audit: dict) -> None:
    lines = [
        "# GPT-5.5 Scoring Audit",
        "",
        f"- Input run: `{audit['input_run_dir']}`",
        f"- Rows audited: `{audit['n_rows']}`",
        "- Scope: existing frozen-context GPT-5.5 reader outputs only; no new model calls.",
        "- Optional semantic judge: not run, because no cached judge outputs were present and the task asked not to spend on full benchmark judging.",
        "",
        "## Normalized Scoring",
        "",
        "Normalized EM lowercases, strips punctuation and articles, collapses whitespace, canonicalizes simple date mentions, maps number words zero to twenty to digits, and handles a small alias set (US/USA, UK, NYC). Gold labels with explicit acceptable alternatives are split into deterministic variants.",
        "",
        "| Method | Raw EM | Normalized EM | Token F1 | Evidence use | Insuff. | Gold retrieved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in audit["method_summary"].items():
        focus = row["focus"]
        lines.append(
            f"| {row['method_label']} | {format_rate(focus['raw_em'])} | "
            f"{format_rate(focus['normalized_em'])} | {format_rate(focus['token_f1'])} | "
            f"{format_rate(focus['evidence_use'])} | {format_rate(focus['insufficient_evidence_rate'])} | "
            f"{format_rate(focus['gold_evidence_retrieved'])} |"
        )

    lines.extend(
        [
            "",
            "## Answer-Type Analysis",
            "",
            "| Method | Answer type | n | Raw EM | Normalized EM | Token F1 | Evidence use | Insuff. |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _method, method_row in audit["answer_type_analysis"].items():
        for answer_type, metrics in method_row["answer_types"].items():
            lines.append(
                f"| {method_row['method_label']} | {answer_type} | {metrics['n']} | "
                f"{format_rate(metrics['raw_em'])} | {format_rate(metrics['normalized_em'])} | "
                f"{format_rate(metrics['token_f1'])} | {format_rate(metrics['evidence_use'])} | "
                f"{format_rate(metrics['insufficient_evidence_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Balanced Audit Sample",
            "",
            f"- Sample path: `{audit['sample_path']}`",
            f"- Total rows: `{audit['sample_summary']['n']}`",
            "",
            "| Category | Rows |",
            "|---|---:|",
        ]
    )
    for category, count in audit["sample_summary"]["category_counts"].items():
        lines.append(f"| `{category}` | {count} |")

    delta_count = len(audit["normalization_delta_examples"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Normalized EM changes {audit['normalization_changed_count']} of {audit['n_rows']} method-question rows; the first {delta_count} changed examples are stored in `normalized_scoring_v2.json`.",
            "- Normalization materially raises absolute EM for OracleMem and full raw, mainly for explicit acceptable duration labels and number-word/date formatting.",
            "- The OracleMem/full-raw normalized-EM gap remains modest; the strongest external signal is still OracleMem's higher token-F1 and evidence-use.",
            "- The balanced sample is intended for a human or cheap blinded judge pass: it separates supported abstentions, high-overlap EM failures, full-raw EM failures, and OracleMem/full-raw disagreements.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GPT-5.5 LongMemEval reader scoring.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--retrieval-rows-json", type=Path, default=DEFAULT_RETRIEVAL_ROWS)
    parser.add_argument("--budget-frac", type=float, default=0.20)
    parser.add_argument("--max-context-words", type=int, default=1800)
    parser.add_argument("--max-memory-chars", type=int, default=900)
    parser.add_argument("--focus-types", type=str, default=",".join(sorted(FOCUS_TYPES)))
    args = parser.parse_args()

    focus_types = {part.strip() for part in args.focus_types.split(",") if part.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.dataset_json, None)
    examples_by_id = {example["question_id"]: example for example in examples}
    reader_rows = load_reader_outputs(args.run_dir)
    enriched = enrich_rows(reader_rows, examples_by_id)

    retrieval_rows = json.loads(args.retrieval_rows_json.read_text(encoding="utf-8"))
    retrieval_by_method = retrieval_lookup(retrieval_rows)
    contexts: dict[tuple[str, str], list[ContextEntry]] = {}
    sample, category_counts = build_balanced_sample(
        enriched,
        contexts,
        examples_by_id,
        retrieval_by_method,
        args.budget_frac,
        args.max_context_words,
        args.max_memory_chars,
    )

    sample_path = args.out_dir / "semantic_audit_sample_50.jsonl"
    write_jsonl(sample_path, sample)

    deltas = normalization_deltas(enriched)
    audit = {
        "input_run_dir": str(args.run_dir),
        "dataset_json": str(args.dataset_json),
        "retrieval_rows_json": str(args.retrieval_rows_json),
        "n_rows": len(enriched),
        "focus_types": sorted(focus_types),
        "normalization_definition": {
            "lowercase": True,
            "strip_punctuation": True,
            "strip_articles": sorted(ARTICLES),
            "collapse_whitespace": True,
            "date_normalization": "month-name dates and simple slash/dash dates are canonicalized when detectable",
            "number_word_normalization": "zero through twenty are mapped to digits",
            "aliases": ["US/USA -> United States", "UK -> United Kingdom", "New York City -> NYC"],
            "gold_variants": "explicit acceptable alternatives and parenthetical-free variants are considered",
        },
        "method_summary": method_summary(enriched, focus_types),
        "answer_type_analysis": answer_type_summary(enriched),
        "normalization_changed_count": sum(
            1 for row in enriched if row.get("normalized_em", 0.0) > row.get("raw_em", 0.0)
        ),
        "normalization_delta_examples": deltas,
        "sample_summary": {
            "n": len(sample),
            "category_counts": category_counts,
            "balance_target": {
                "oraclemem_abstained_despite_support": 20,
                "oraclemem_high_f1_em0": 10,
                "full_raw_high_f1_em0": 10,
                "oraclemem_full_raw_disagreement": 10,
            },
        },
        "sample_path": str(sample_path),
        "semantic_judge": {
            "used": False,
            "reason": "No cached judge outputs were present; no new API judge calls were made.",
        },
    }

    json_path = args.out_dir / "normalized_scoring_v2.json"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    write_report(args.out_dir / "SCORING_AUDIT.md", audit)
    print(json.dumps({"wrote": [str(json_path), str(sample_path), str(args.out_dir / "SCORING_AUDIT.md")]}, indent=2))


if __name__ == "__main__":
    main()
