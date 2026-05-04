from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import string
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DATA_URL = "https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json"
DEFAULT_METHODS = [
    "dense_budgeted_bsc",
    "dense_rag_e5",
    "dense_budgeted_replay",
    "fifo_replay",
]

PROMPT_MODES = (
    "answer_if_supported",
    "evidence_extraction_first",
    "extractive_answer",
)

METHOD_LABELS = {
    "dense_budgeted_bsc": "OracleMem writer + dense retrieval",
    "heuristic_bsc": "OracleMem writer + lexical retrieval",
    "dense_rag_e5": "Full raw-store dense retrieval",
    "dense_budgeted_replay": "Budgeted raw replay + dense retrieval",
    "replay_only_router": "Budgeted raw replay router",
    "fifo_replay": "FIFO raw replay",
    "uniform_replay": "Uniform raw replay",
    "memorybank_proxy": "MemoryBank proxy",
    "ld_agent_proxy": "LD-Agent proxy",
}

METHOD_ALIASES = {
    "oraclemem_dense": "dense_budgeted_bsc",
    "oracle_dense": "dense_budgeted_bsc",
    "full_raw_dense": "dense_rag_e5",
    "budgeted_raw_dense": "dense_budgeted_replay",
    "budgeted_raw_replay": "dense_budgeted_replay",
    "fifo_raw": "fifo_replay",
}

FOCUS_TYPES = {"knowledge-update", "temporal-reasoning"}

FIRST_PERSON_PATTERNS = [
    r"\bi am\b",
    r"\bi'm\b",
    r"\bi work\b",
    r"\bi live\b",
    r"\bi study\b",
    r"\bi like\b",
    r"\bi love\b",
    r"\bi prefer\b",
    r"\bmy favorite\b",
    r"\bmy name is\b",
    r"\bi usually\b",
    r"\bi always\b",
    r"\bi often\b",
    r"\bi hate\b",
    r"\bi enjoy\b",
    r"\bmy job\b",
    r"\bmy birthday\b",
    r"\bmy address\b",
    r"\bmy phone\b",
    r"\bi need\b",
    r"\bi have\b",
]
UPDATE_PATTERNS = [
    r"\bactually\b",
    r"\binstead\b",
    r"\bchange\b",
    r"\bchanged\b",
    r"\bupdate\b",
    r"\bupdated\b",
    r"\bfrom now on\b",
    r"\bgoing forward\b",
    r"\bnew\b",
    r"\bnot anymore\b",
]
TIME_PATTERNS = [
    r"\btoday\b",
    r"\btomorrow\b",
    r"\byesterday\b",
    r"\btonight\b",
    r"\bthis week\b",
    r"\bnext week\b",
    r"\bnext month\b",
    r"\bnext year\b",
    r"\bmonday\b",
    r"\btuesday\b",
    r"\bwednesday\b",
    r"\bthursday\b",
    r"\bfriday\b",
    r"\bsaturday\b",
    r"\bsunday\b",
    r"\bjan(?:uary)?\b",
    r"\bfeb(?:ruary)?\b",
    r"\bmar(?:ch)?\b",
    r"\bapr(?:il)?\b",
    r"\bmay\b",
    r"\bjun(?:e)?\b",
    r"\bjul(?:y)?\b",
    r"\baug(?:ust)?\b",
    r"\bsep(?:tember)?\b",
    r"\boct(?:ober)?\b",
    r"\bnov(?:ember)?\b",
    r"\bdec(?:ember)?\b",
]
FIRST_PERSON_RE = re.compile("|".join(FIRST_PERSON_PATTERNS), re.IGNORECASE)
UPDATE_RE = re.compile("|".join(UPDATE_PATTERNS), re.IGNORECASE)
TIME_RE = re.compile("|".join(TIME_PATTERNS), re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d{1,4}\b")
GENERIC_ASSISTANT_RE = re.compile(
    r"\b(certainty|confidence score|here are|i can help|let me know|feel free)\b",
    re.IGNORECASE,
)


@dataclass
class MemoryEntry:
    session_id: str
    session_index: int
    action: str
    text: str
    cost_words: int
    priority: float


@dataclass
class ContextEntry:
    session_id: str
    action: str
    text: str
    source: str


def csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def canonical_method_name(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def canonical_method_list(methods: Iterable[str]) -> list[str]:
    canonical: list[str] = []
    for method in methods:
        name = canonical_method_name(method)
        if name not in canonical:
            canonical.append(name)
    return canonical


def validate_prompt_modes(prompt_modes: Iterable[str]) -> list[str]:
    modes = [mode.strip() for mode in prompt_modes if mode.strip()]
    allowed = {"strict", *PROMPT_MODES}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"Unknown prompt mode(s): {', '.join(unknown)}")
    return modes


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def load_examples(dataset_json: Path | None, cache_json: Path | None) -> list[dict]:
    if dataset_json is not None:
        return json.loads(dataset_json.read_text(encoding="utf-8"))
    if cache_json is not None and cache_json.exists():
        return json.loads(cache_json.read_text(encoding="utf-8"))
    with urllib.request.urlopen(DATA_URL) as handle:
        examples = json.load(handle)
    if cache_json is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(examples), encoding="utf-8")
    return examples


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def split_question_rows(source: Path) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in read_jsonl(source):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            continue
        question_type = str(row.get("question_type", "")).strip()
        existing = seen.get(question_id)
        if existing is not None:
            if question_type and existing["question_type"] != question_type:
                raise ValueError(f"Conflicting question_type for {question_id}: {existing['question_type']} vs {question_type}")
            continue
        seen[question_id] = {
            "question_id": question_id,
            "question_type": question_type,
        }
    if not seen:
        raise ValueError(f"No question_id rows found in {source}")
    return sorted(seen.values(), key=lambda row: row["question_id"])


def stratified_dev_counts(by_type: dict[str, list[dict]], dev_size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in by_type.values())
    if dev_size <= 0 or dev_size >= total:
        raise ValueError(f"dev_size must be between 1 and {total - 1}; got {dev_size}")
    raw_targets = {
        question_type: dev_size * len(rows) / total for question_type, rows in by_type.items()
    }
    counts = {
        question_type: min(len(by_type[question_type]), int(raw_targets[question_type]))
        for question_type in by_type
    }
    remainder = dev_size - sum(counts.values())
    order = sorted(
        by_type,
        key=lambda question_type: (
            raw_targets[question_type] - int(raw_targets[question_type]),
            len(by_type[question_type]),
            question_type,
        ),
        reverse=True,
    )
    while remainder > 0:
        changed = False
        for question_type in order:
            if counts[question_type] < len(by_type[question_type]):
                counts[question_type] += 1
                remainder -= 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            raise ValueError("Could not allocate stratified dev split")
    return counts


def make_focus_dev_eval_split(source: Path, dev_size: int, out_dir: Path) -> dict:
    rows = split_question_rows(source)
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row)
    counts = stratified_dev_counts(by_type, dev_size)

    dev_ids: set[str] = set()
    for question_type, type_rows in sorted(by_type.items()):
        ordered = sorted(
            type_rows,
            key=lambda row: stable_hash(f"longmemeval-focus-dev-v1:{row['question_id']}"),
        )
        dev_ids.update(row["question_id"] for row in ordered[: counts[question_type]])

    dev_rows = sorted((row for row in rows if row["question_id"] in dev_ids), key=lambda row: row["question_id"])
    eval_rows = sorted((row for row in rows if row["question_id"] not in dev_ids), key=lambda row: row["question_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / f"focus_dev_{len(dev_rows)}.jsonl"
    eval_path = out_dir / f"focus_eval_{len(eval_rows)}.jsonl"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(eval_path, eval_rows)

    summary = {
        "source": str(source),
        "algorithm": "question_id SHA-256 hash within question_type strata",
        "dev_path": str(dev_path),
        "eval_path": str(eval_path),
        "total_questions": len(rows),
        "dev_size": len(dev_rows),
        "eval_size": len(eval_rows),
        "counts_by_type": {
            question_type: {
                "total": len(type_rows),
                "dev": sum(1 for row in dev_rows if row["question_type"] == question_type),
                "eval": sum(1 for row in eval_rows if row["question_type"] == question_type),
            }
            for question_type, type_rows in sorted(by_type.items())
        },
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_split_question_ids(split_path: Path) -> set[str]:
    rows = read_jsonl(split_path)
    ids = {str(row.get("question_id", "")).strip() for row in rows}
    ids.discard("")
    if not ids:
        raise ValueError(f"No question_id values found in split file {split_path}")
    return ids


def session_text(session: list[dict]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)


def count_words(text: str) -> int:
    return len(text.split())


def extract_fact_lines(session: list[dict]) -> list[str]:
    facts: list[str] = []
    for turn in session:
        if turn["role"] != "user":
            continue
        content = turn["content"].strip()
        if FIRST_PERSON_RE.search(content):
            facts.append(content)
    return facts[:6]


def tail_snippet(session: list[dict], turns: int = 4) -> str:
    return session_text(session[-turns:])


def session_features(session: list[dict], index: int, total: int) -> dict[str, float]:
    raw_text = session_text(session)
    user_turns = sum(1 for turn in session if turn["role"] == "user")
    assistant_turns = len(session) - user_turns
    fact_lines = extract_fact_lines(session)
    return {
        "words": count_words(raw_text),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "fact_hits": len(FIRST_PERSON_RE.findall(raw_text)),
        "update_hits": len(UPDATE_RE.findall(raw_text)),
        "time_hits": len(TIME_RE.findall(raw_text)),
        "number_hits": len(NUMBER_RE.findall(raw_text)),
        "fact_lines": len(fact_lines),
        "recent_rank": float(total - 1 - index),
        "recent_frac": float(total - index) / max(float(total), 1.0),
        "assistant_only": float(user_turns == 0),
        "generic_assistant": float(bool(GENERIC_ASSISTANT_RE.search(raw_text))),
    }


def classify_action(session: list[dict], index: int, total: int) -> str:
    features = session_features(session, index, total)
    raw_text = session_text(session).lower()
    if features["assistant_only"] and features["generic_assistant"]:
        return "discard"
    if features["fact_lines"] > 0 and (
        features["fact_hits"] > 0 or "favorite" in raw_text or "prefer" in raw_text
    ):
        return "consolidate"
    if features["recent_rank"] <= 4 or features["update_hits"] > 0:
        return "cache"
    if features["time_hits"] > 0 or features["number_hits"] >= 6:
        return "replay"
    if features["words"] < 80:
        return "discard"
    return "replay"


def make_entry(session: list[dict], session_id: str, session_index: int, action: str) -> MemoryEntry | None:
    raw_text = session_text(session)
    if action == "discard":
        return None
    if action == "replay":
        text = raw_text
        priority = 2.0
    elif action == "cache":
        text = tail_snippet(session, turns=4)
        priority = 3.0
    elif action == "consolidate":
        facts = extract_fact_lines(session)
        text = "\n".join(f"fact: {line}" for line in facts) if facts else tail_snippet(session, turns=2)
        priority = 4.0
    else:
        raise ValueError(f"Unknown action: {action}")
    return MemoryEntry(
        session_id=session_id,
        session_index=session_index,
        action=action,
        text=text,
        cost_words=count_words(text),
        priority=priority,
    )


def full_budget_words(example: dict) -> int:
    return sum(count_words(session_text(session)) for session in example["haystack_sessions"])


def take_under_budget(entries: Iterable[MemoryEntry], budget_words: int) -> list[MemoryEntry]:
    kept: list[MemoryEntry] = []
    used = 0
    for entry in entries:
        if used + entry.cost_words > budget_words:
            continue
        kept.append(entry)
        used += entry.cost_words
    return kept


def build_fifo_replay(example: dict, budget_frac: float) -> list[MemoryEntry]:
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    candidates = [
        MemoryEntry(
            session_id=session_id,
            session_index=index,
            action="replay",
            text=session_text(session),
            cost_words=count_words(session_text(session)),
            priority=1.0,
        )
        for index, (session_id, session) in enumerate(
            zip(example["haystack_session_ids"], example["haystack_sessions"])
        )
    ]
    return take_under_budget(reversed(candidates), budget_words)


def build_uniform_replay(example: dict, budget_frac: float) -> list[MemoryEntry]:
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    candidates = [
        MemoryEntry(
            session_id=session_id,
            session_index=index,
            action="replay",
            text=session_text(session),
            cost_words=count_words(session_text(session)),
            priority=1.0,
        )
        for index, (session_id, session) in enumerate(
            zip(example["haystack_session_ids"], example["haystack_sessions"])
        )
    ]
    approx_mean = max(1.0, statistics.mean(entry.cost_words for entry in candidates))
    target_count = max(1, int(budget_words / approx_mean))
    if target_count == 1:
        selected_indices = [len(candidates) - 1]
    else:
        step = (len(candidates) - 1) / max(target_count - 1, 1)
        selected_indices = [round(step * i) for i in range(target_count)]
    selected = [candidates[i] for i in selected_indices]
    leftovers = [entry for idx, entry in enumerate(candidates) if idx not in set(selected_indices)]
    return take_under_budget(selected + leftovers, budget_words)


def build_replay_only_router(example: dict, budget_frac: float) -> list[MemoryEntry]:
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    total = len(example["haystack_sessions"])
    candidates: list[tuple[float, MemoryEntry]] = []
    for index, (session_id, session) in enumerate(
        zip(example["haystack_session_ids"], example["haystack_sessions"])
    ):
        raw_text = session_text(session)
        features = session_features(session, index, total)
        score = (
            2.0 * features["fact_hits"]
            + 1.5 * features["update_hits"]
            + 1.0 * features["time_hits"]
            + 0.3 * features["number_hits"]
            + 1.2 * features["recent_frac"]
        )
        entry = MemoryEntry(
            session_id=session_id,
            session_index=index,
            action="replay",
            text=raw_text,
            cost_words=count_words(raw_text),
            priority=score,
        )
        candidates.append((score / max(entry.cost_words, 1), entry))
    ordered = [entry for _, entry in sorted(candidates, key=lambda item: item[0], reverse=True)]
    return take_under_budget(ordered, budget_words)


def build_bsc(example: dict, budget_frac: float) -> list[MemoryEntry]:
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    total = len(example["haystack_sessions"])
    candidates: list[tuple[float, float, int, MemoryEntry]] = []
    for index, (session_id, session) in enumerate(
        zip(example["haystack_session_ids"], example["haystack_sessions"])
    ):
        action = classify_action(session, index, total)
        entry = make_entry(session, session_id, index, action)
        if entry is None:
            continue
        density = entry.priority / max(entry.cost_words, 1)
        candidates.append((density, entry.priority, -index, entry))
    ordered = [entry for _, _, _, entry in sorted(candidates, reverse=True)]
    return take_under_budget(ordered, budget_words)


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(no_punct.split())


def normalize_answer_articles(text: str) -> str:
    tokens = normalize_answer(text).split()
    return " ".join(token for token in tokens if token not in {"a", "an", "the"})


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def article_stripped_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer_articles(prediction) == normalize_answer_articles(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_insufficient_answer(text: str) -> bool:
    compact = re.sub(r"[\W_]+", "", str(text).lower())
    return compact in {"insufficientevidence", "insufficientinfo", "notenoughinformation"}


def summarize_session_for_memorybank(session: list[dict]) -> str:
    facts = extract_fact_lines(session)
    if facts:
        return "\n".join(f"fact: {line}" for line in facts[:4])
    return tail_snippet(session, turns=3)


def summarize_session_for_ld_long(session: list[dict]) -> str:
    facts = extract_fact_lines(session)
    if facts:
        return "\n".join(f"persona: {line}" for line in facts[:3])
    return tail_snippet(session, turns=2)


def entries_from_full_raw(example: dict) -> dict[str, ContextEntry]:
    return {
        session_id: ContextEntry(
            session_id=session_id,
            action="raw",
            text=session_text(session),
            source="full_raw_store",
        )
        for session_id, session in zip(example["haystack_session_ids"], example["haystack_sessions"])
    }


def entries_from_memory_entries(entries: list[MemoryEntry], source: str) -> dict[str, ContextEntry]:
    return {
        entry.session_id: ContextEntry(
            session_id=entry.session_id,
            action=entry.action,
            text=entry.text,
            source=source,
        )
        for entry in entries
    }


def entries_from_memorybank(example: dict) -> dict[str, ContextEntry]:
    return {
        session_id: ContextEntry(
            session_id=session_id,
            action="fact_summary",
            text=summarize_session_for_memorybank(session),
            source="memorybank_proxy",
        )
        for session_id, session in zip(example["haystack_session_ids"], example["haystack_sessions"])
    }


def entries_from_ld_agent(example: dict) -> dict[str, ContextEntry]:
    total = len(example["haystack_sessions"])
    short_cutoff = max(total - 6, 0)
    entries = {}
    for index, (session_id, session) in enumerate(zip(example["haystack_session_ids"], example["haystack_sessions"])):
        if index >= short_cutoff:
            action = "short_term_raw"
            text = tail_snippet(session, turns=4)
        else:
            action = "long_term_summary"
            text = summarize_session_for_ld_long(session)
        entries[session_id] = ContextEntry(
            session_id=session_id,
            action=action,
            text=text,
            source="ld_agent_proxy",
        )
    return entries


def method_entry_lookup(example: dict, method: str, budget_frac: float) -> dict[str, ContextEntry]:
    if method == "dense_rag_e5":
        return entries_from_full_raw(example)
    if method == "memorybank_proxy":
        return entries_from_memorybank(example)
    if method == "ld_agent_proxy":
        return entries_from_ld_agent(example)
    if method == "fifo_replay":
        return entries_from_memory_entries(build_fifo_replay(example, budget_frac), "fifo_replay")
    if method == "uniform_replay":
        return entries_from_memory_entries(build_uniform_replay(example, budget_frac), "uniform_replay")
    if method in {"replay_only_router", "dense_budgeted_replay"}:
        return entries_from_memory_entries(build_replay_only_router(example, budget_frac), "budgeted_raw_replay")
    if method in {"heuristic_bsc", "dense_budgeted_bsc"}:
        return entries_from_memory_entries(build_bsc(example, budget_frac), "oraclemem_writer")
    raise KeyError(f"Unknown method: {method}")


def reconstruct_context(example: dict, retrieval_row: dict, method: str, budget_frac: float, max_context_words: int) -> tuple[list[ContextEntry], int]:
    lookup = method_entry_lookup(example, method, budget_frac)
    full_raw = entries_from_full_raw(example)
    context: list[ContextEntry] = []
    fallback_count = 0
    used_words = 0
    for session_id in retrieval_row.get("predicted_session_ids", []):
        entry = lookup.get(session_id)
        if entry is None:
            entry = full_raw.get(session_id)
            fallback_count += 1
        if entry is None:
            continue
        words = entry.text.split()
        clipped = " ".join(words[: min(len(words), 400)])
        block_words = count_words(clipped) + 8
        if context and used_words + block_words > max_context_words:
            break
        context.append(ContextEntry(session_id=entry.session_id, action=entry.action, text=clipped, source=entry.source))
        used_words += block_words
    return context, fallback_count


def context_prompt(question: str, context: list[ContextEntry], prompt_style: str = "strict") -> str:
    blocks = []
    for index, entry in enumerate(context, start=1):
        blocks.append(
            f"[{index}] memory_id={entry.session_id} action={entry.action} source={entry.source}\n{entry.text}"
        )
    memory = "\n\n".join(blocks) if blocks else "[no memory]"
    if prompt_style == "answer_if_supported":
        return (
            "You are answering a long-term memory question using only the provided memory context.\n\n"
            "Rules:\n"
            "1. If the context directly supports an answer, answer it.\n"
            "2. If the answer is supported but phrased differently from the question, still answer.\n"
            "3. If multiple memories conflict, prefer the most recent/current memory or a memory that explicitly supersedes an older one.\n"
            '4. Only output "INSUFFICIENT_EVIDENCE" if no provided memory supports an answer.\n'
            "5. Cite the memory ids used.\n\n"
            f"Question:\n{question}\n\n"
            f"Memory context:\n{memory}\n\n"
            "Return exactly this JSON and no extra text:\n"
            "{\n"
            '  "answer": "...",\n'
            '  "abstained": true,\n'
            '  "used_memory_ids": ["..."]\n'
            "}"
        )
    if prompt_style == "evidence_extraction_first":
        return (
            "You are answering a long-term memory question using only the provided memory context.\n\n"
            "Rules:\n"
            "1. First decide whether any provided memory directly or partially supports an answer.\n"
            "2. If at least one memory supports the answer, answer concisely.\n"
            '3. Use "INSUFFICIENT_EVIDENCE" only if no memory supports an answer.\n'
            "4. Do not require exact wording; paraphrased support is enough.\n"
            "5. Prefer the most recent/current memory when memories conflict.\n"
            "6. Cite the memory ids used.\n"
            "7. Do not reveal chain-of-thought or explanatory reasoning; return only the JSON object.\n\n"
            f"Question:\n{question}\n\n"
            f"Memory context:\n{memory}\n\n"
            "Return exactly this JSON and no extra text:\n"
            "{\n"
            '  "support_status": "SUPPORTED",\n'
            '  "answer": "...",\n'
            '  "abstained": false,\n'
            '  "used_memory_ids": ["..."]\n'
            "}\n"
            'Use support_status "SUPPORTED", "PARTIAL", or "UNSUPPORTED".'
        )
    if prompt_style == "extractive_answer":
        return (
            "You are answering a long-term memory question using only the provided memory context.\n\n"
            "Rules:\n"
            "1. If the memory contains a relevant value, name, date, event, or fact, extract it.\n"
            "2. A short answer span or concise paraphrase is preferred over a full sentence.\n"
            "3. Do not abstain merely because the answer is phrased differently from the question.\n"
            "4. Prefer current facts over historical facts when the question asks about the current state.\n"
            '5. Use "INSUFFICIENT_EVIDENCE" only if no provided memory contains a relevant answer.\n'
            "6. Cite the memory ids used.\n\n"
            f"Question:\n{question}\n\n"
            f"Memory context:\n{memory}\n\n"
            "Return exactly this JSON and no extra text:\n"
            "{\n"
            '  "answer": "...",\n'
            '  "abstained": false,\n'
            '  "used_memory_ids": ["..."]\n'
            "}"
        )
    if prompt_style != "strict":
        raise ValueError(f"Unknown prompt style: {prompt_style}")
    return (
        "You are answering a long-term memory question using only the provided memory context.\n"
        "Rules:\n"
        "1. Use only the memory context.\n"
        "2. If the context does not support the answer, output INSUFFICIENT_EVIDENCE.\n"
        "3. Prefer current facts over historical facts.\n"
        "4. If a memory says a prior fact was corrected, superseded, invalidated, or deleted, do not answer using the old fact as current truth.\n"
        "5. Cite the memory ids you used.\n\n"
        f"Question:\n{question}\n\n"
        f"Memory context:\n{memory}\n\n"
        "Return exactly this JSON and no extra text:\n"
        "{\n"
        '  "answer": "...",\n'
        '  "abstained": true,\n'
        '  "used_memory_ids": ["..."]\n'
        "}"
    )


def extractive_presence_reader(example: dict, context: list[ContextEntry]) -> dict:
    """A deterministic smoke-test reader, not a substitute for an LLM reader."""
    gold = str(example["answer"]).strip()
    normalized_gold = normalize_text(gold)
    used_ids = []
    if normalized_gold:
        for entry in context:
            if normalized_gold in normalize_text(entry.text):
                used_ids.append(entry.session_id)
    if used_ids:
        return {
            "answer": gold,
            "abstained": False,
            "used_memory_ids": used_ids,
            "parse_failure": False,
        }
    return {
        "answer": "INSUFFICIENT_EVIDENCE",
        "abstained": True,
        "used_memory_ids": [],
        "parse_failure": False,
    }


def parse_reader_json(text: str | None) -> dict:
    raw_text = "" if text is None else str(text)
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    candidate = match.group(0) if match else raw
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "answer": raw.splitlines()[0].strip() if raw else "",
            "abstained": False,
            "used_memory_ids": [],
            "support_status": None,
            "parse_failure": True,
            "raw_response": raw_text,
        }
    answer = str(parsed.get("answer", "")).strip()
    abstained = bool(parsed.get("abstained", is_insufficient_answer(answer)))
    used = parsed.get("used_memory_ids", [])
    if not isinstance(used, list):
        used = []
    support_status = parsed.get("support_status")
    if support_status is not None:
        support_status = str(support_status).strip().upper()
    return {
        "answer": answer,
        "abstained": abstained or is_insufficient_answer(answer),
        "used_memory_ids": [str(item) for item in used],
        "support_status": support_status,
        "parse_failure": False,
        "raw_response": raw_text,
    }


def normalize_used_memory_ids(raw_ids: Iterable[str], context: list[ContextEntry]) -> list[str]:
    normalized: list[str] = []
    context_ids = [entry.session_id for entry in context]
    context_id_set = set(context_ids)
    context_lower = {session_id.lower(): session_id for session_id in context_ids}
    for raw_id in raw_ids:
        value = str(raw_id).strip()
        cleaned = value.strip("[]# '\"")
        if cleaned.isdigit():
            index = int(cleaned) - 1
            if 0 <= index < len(context):
                normalized.append(context[index].session_id)
                continue
        if cleaned in context_id_set:
            normalized.append(cleaned)
            continue
        lowered = cleaned.lower()
        if lowered in context_lower:
            normalized.append(context_lower[lowered])
            continue

        # Some API readers cite shortened memory ids. Resolve only when the
        # abbreviation uniquely identifies one context id; otherwise keep the
        # raw value so unsupported/evidence-use metrics stay conservative.
        compact = re.sub(r"^(memory_id|memory|id)\s*[:=#-]?\s*", "", lowered).strip()
        if len(compact) >= 4:
            matches = [
                session_id
                for session_id in context_ids
                if session_id.lower().endswith(compact) or compact in session_id.lower()
            ]
            if len(matches) == 1:
                normalized.append(matches[0])
                continue
        normalized.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for memory_id in normalized:
        if memory_id not in seen:
            deduped.append(memory_id)
            seen.add(memory_id)
    return deduped


class OpenRouterReader:
    def __init__(
        self,
        api_key: str,
        model: str,
        cache_path: Path,
        *,
        max_tokens: int = 160,
        temperature: float = 0.0,
        request_sleep: float = 0.0,
        timeout: int = 90,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.cache_path = cache_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_sleep = request_sleep
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

    def __call__(self, prompt: str) -> dict:
        cache_settings = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "verbosity": self.verbosity,
        }
        prompt_hash = stable_hash(f"{json.dumps(cache_settings, sort_keys=True)}\n{prompt}")
        if prompt_hash in self.cache:
            cached = dict(self.cache[prompt_hash])
            cached["cache_hit"] = True
            cached["prompt_hash"] = prompt_hash
            return cached
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        if self.verbosity:
            payload["verbosity"] = self.verbosity
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/oraclemem",
                "X-Title": "OracleMem LongMemEval Reader",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {error.code}: {details}") from error
        content = body["choices"][0]["message"].get("content")
        parsed = parse_reader_json(content)
        parsed.update(
            {
                "cache_hit": False,
                "prompt_hash": prompt_hash,
                "model": self.model,
                "usage": body.get("usage", {}),
                "provider": body.get("provider"),
            }
        )
        self.cache[prompt_hash] = parsed
        self._write_cache()
        if self.request_sleep > 0:
            time.sleep(self.request_sleep)
        return parsed


def score_predictions(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "exact_match": 0.0,
            "token_f1": 0.0,
            "evidence_use": 0.0,
            "insufficient_evidence_rate": 0.0,
            "unsupported_answer_rate": 0.0,
            "parse_failure_rate": 0.0,
            "avg_context_words": 0.0,
            "avg_context_tokens_est": 0.0,
            "avg_fallback_contexts": 0.0,
            "cache_hit_rate": 0.0,
            "total_api_cost": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
        }
    prompt_tokens = [float(row.get("usage", {}).get("prompt_tokens", 0.0) or 0.0) for row in rows]
    completion_tokens = [float(row.get("usage", {}).get("completion_tokens", 0.0) or 0.0) for row in rows]
    costs = [float(row.get("usage", {}).get("cost", 0.0) or 0.0) for row in rows]
    return {
        "n": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "token_f1": sum(row["token_f1"] for row in rows) / len(rows),
        "evidence_use": sum(row["evidence_use"] for row in rows) / len(rows),
        "insufficient_evidence_rate": sum(row["abstained"] for row in rows) / len(rows),
        "unsupported_answer_rate": sum(row["unsupported_answer"] for row in rows) / len(rows),
        "parse_failure_rate": sum(row["parse_failure"] for row in rows) / len(rows),
        "avg_context_words": sum(row["context_words"] for row in rows) / len(rows),
        "avg_context_tokens_est": sum(row["context_tokens_est"] for row in rows) / len(rows),
        "avg_fallback_contexts": sum(row["fallback_contexts"] for row in rows) / len(rows),
        "cache_hit_rate": sum(row.get("cache_hit", False) for row in rows) / len(rows),
        "total_api_cost": sum(costs),
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
        "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens),
    }


def retrieval_stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "any_gold_retrieved": 0.0,
            "gold_recall": 0.0,
            "retrieved_count": 0,
        }
    any_hits = []
    recalls = []
    retrieved_count = 0
    for row in rows:
        gold = set(row.get("gold_session_ids", []))
        context = set(row.get("context_session_ids", []))
        hit_count = len(gold & context)
        any_hit = bool(hit_count)
        any_hits.append(float(any_hit))
        if any_hit:
            retrieved_count += 1
        recalls.append(hit_count / max(len(gold), 1))
    return {
        "n": len(rows),
        "any_gold_retrieved": sum(any_hits) / len(any_hits),
        "gold_recall": sum(recalls) / len(recalls),
        "retrieved_count": retrieved_count,
    }


def score_conditioned_on_retrieved(rows: list[dict]) -> dict:
    retrieved_rows = [
        row for row in rows if set(row.get("gold_session_ids", [])) & set(row.get("context_session_ids", []))
    ]
    result = score_predictions(retrieved_rows)
    result.update(retrieval_stats(rows))
    return result


def paired_bootstrap_delta(rows_a: list[dict], rows_b: list[dict], metric: str, *, n_bootstrap: int, seed: int) -> dict:
    by_a = {row["question_id"]: row for row in rows_a}
    by_b = {row["question_id"]: row for row in rows_b}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        return {"n": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    diffs = [float(by_a[item][metric]) - float(by_b[item][metric]) for item in ids]
    mean_delta = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    if len(diffs) == 1 or n_bootstrap <= 0:
        return {"n": len(diffs), "mean_delta": mean_delta, "ci95": [mean_delta, mean_delta]}
    means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "n": len(diffs),
        "mean_delta": mean_delta,
        "ci95": [
            means[int(0.025 * (len(means) - 1))],
            means[int(0.975 * (len(means) - 1))],
        ],
    }


def filter_examples(examples: list[dict], focus_types: set[str], *, focus_only: bool, per_type_limit: int, seed: int) -> list[dict]:
    pool = [example for example in examples if (not focus_only or example["question_type"] in focus_types)]
    if per_type_limit <= 0:
        return pool
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for example in pool:
        by_type.setdefault(example["question_type"], []).append(example)
    selected: list[dict] = []
    for question_type in sorted(by_type):
        rows = list(by_type[question_type])
        rng.shuffle(rows)
        selected.extend(rows[:per_type_limit])
    selected.sort(key=lambda item: item["question_id"])
    return selected


def evaluate(
    examples: list[dict],
    retrieval_rows: dict[str, list[dict]],
    methods: list[str],
    focus_types: set[str],
    budget_frac: float,
    max_context_words: int,
    save_prompts: bool,
    reader_name: str,
    openrouter_reader: OpenRouterReader | None,
    shuffle_jobs: bool,
    seed: int,
    bootstrap: int,
    prompt_style: str,
) -> tuple[dict, dict]:
    examples_by_id = {example["question_id"]: example for example in examples}
    allowed_ids = set(examples_by_id)
    method_rows_by_id: dict[str, dict[str, dict]] = {}
    for method in methods:
        method_rows = retrieval_rows.get(method)
        if method_rows is None:
            raise KeyError(f"Method not found in retrieval rows: {method}")
        method_rows_by_id[method] = {
            row["question_id"]: row for row in method_rows if row["question_id"] in allowed_ids
        }

    jobs = [
        (method, question_id)
        for method in methods
        for question_id in sorted(method_rows_by_id[method])
    ]
    if shuffle_jobs:
        random.Random(seed).shuffle(jobs)

    artifacts: dict[str, list[dict]] = {method: [] for method in methods}
    for method, question_id in jobs:
        example = examples_by_id[question_id]
        retrieval_row = method_rows_by_id[method][question_id]
        context, fallback_count = reconstruct_context(
            example=example,
            retrieval_row=retrieval_row,
            method=method,
            budget_frac=budget_frac,
            max_context_words=max_context_words,
        )
        prompt = context_prompt(example["question"], context, prompt_style=prompt_style)
        if reader_name == "extractive_presence_smoke":
            reader_output = extractive_presence_reader(example, context)
        elif reader_name == "openrouter":
            if openrouter_reader is None:
                raise ValueError("openrouter_reader is required for reader=openrouter")
            reader_output = openrouter_reader(prompt)
        else:
            raise ValueError(f"Unknown reader: {reader_name}")
        prediction = reader_output["answer"]
        gold = example["answer"]
        gold_ids = set(example.get("answer_session_ids", []))
        used_ids = set(normalize_used_memory_ids(reader_output.get("used_memory_ids", []), context))
        evidence_use = float(bool(used_ids & gold_ids))
        context_words = sum(count_words(entry.text) for entry in context)
        row = {
            "question_id": question_id,
            "question_type": example["question_type"],
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "gold_answer": gold,
            "prediction": prediction,
            "abstained": bool(reader_output["abstained"]),
            "used_memory_ids": sorted(used_ids),
            "gold_session_ids": sorted(gold_ids),
            "exact_match": exact_match(prediction, gold),
            "token_f1": token_f1(prediction, gold),
            "evidence_use": evidence_use,
            "unsupported_answer": float((not bool(reader_output["abstained"])) and evidence_use == 0.0),
            "parse_failure": bool(reader_output["parse_failure"]),
            "context_session_ids": [entry.session_id for entry in context],
            "context_words": context_words,
            "context_tokens_est": int(round(context_words * 1.33)),
            "fallback_contexts": fallback_count,
            "prompt_hash": stable_hash(prompt),
            "cache_hit": bool(reader_output.get("cache_hit", False)),
            "reader_model": reader_output.get("model"),
            "support_status": reader_output.get("support_status"),
            "usage": reader_output.get("usage", {}),
        }
        if save_prompts:
            row["prompt"] = prompt
        artifacts[method].append(row)

    summary: dict[str, dict] = {}
    for method in methods:
        predictions = sorted(artifacts[method], key=lambda row: row["question_id"])
        focus_rows = [row for row in predictions if row["question_type"] in focus_types]
        by_type = {}
        for question_type in sorted({row["question_type"] for row in predictions}):
            by_type[question_type] = score_predictions(
                [row for row in predictions if row["question_type"] == question_type]
            )
        summary[method] = {
            "method_label": METHOD_LABELS.get(method, method),
            "reader": reader_name,
            "scope": "API reader" if reader_name == "openrouter" else "deterministic smoke; not an LLM reader",
            "overall": score_predictions(predictions),
            "focus": score_predictions(focus_rows),
            "per_type": by_type,
        }
    if "dense_budgeted_bsc" in artifacts:
        oracle_focus = [row for row in artifacts["dense_budgeted_bsc"] if row["question_type"] in focus_types]
        deltas = {}
        for baseline in methods:
            if baseline == "dense_budgeted_bsc":
                continue
            baseline_focus = [row for row in artifacts[baseline] if row["question_type"] in focus_types]
            deltas[baseline] = {
                "baseline_label": METHOD_LABELS.get(baseline, baseline),
                "exact_match": paired_bootstrap_delta(oracle_focus, baseline_focus, "exact_match", n_bootstrap=bootstrap, seed=seed),
                "token_f1": paired_bootstrap_delta(oracle_focus, baseline_focus, "token_f1", n_bootstrap=bootstrap, seed=seed + 1),
                "evidence_use": paired_bootstrap_delta(oracle_focus, baseline_focus, "evidence_use", n_bootstrap=bootstrap, seed=seed + 2),
            }
        summary["_paired_focus_deltas_vs_oraclemem_dense"] = deltas
    return summary, artifacts


def load_reader_outputs(run_dir: Path) -> list[dict]:
    path = run_dir / "reader_outputs.jsonl"
    if not path.exists():
        predictions = run_dir / "predictions.json"
        if not predictions.exists():
            raise FileNotFoundError(f"Expected {path} or {predictions}")
        artifacts = json.loads(predictions.read_text(encoding="utf-8"))
        rows = []
        for method_rows in artifacts.values():
            rows.extend(method_rows)
        return rows
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def bucket_reader_errors(rows: list[dict]) -> dict[str, list[dict]]:
    buckets = {
        "retrieval_hit_but_abstained": [],
        "insufficient_despite_support": [],
        "evidence_used_but_wrong_answer": [],
        "high_f1_em_zero": [],
        "full_raw_retrieved_but_abstained": [],
        "oraclemem_missing_evidence": [],
        "unsupported_answer": [],
        "schema_conflict_answer_and_abstained": [],
        "abstain_with_gold_citation": [],
    }
    for row in rows:
        gold = set(row.get("gold_session_ids", []))
        context = set(row.get("context_session_ids", []))
        retrieved = bool(gold & context)
        answer_text = normalize_text(str(row.get("prediction", "")))
        answer_looks_substantive = bool(answer_text) and not is_insufficient_answer(row.get("prediction", ""))
        if retrieved and row.get("abstained"):
            buckets["retrieval_hit_but_abstained"].append(row)
            buckets["insufficient_despite_support"].append(row)
        if (
            row.get("evidence_use", 0.0) > 0.0
            and row.get("exact_match", 0.0) < 1.0
            and not row.get("abstained")
        ):
            buckets["evidence_used_but_wrong_answer"].append(row)
        if (
            row.get("exact_match", 0.0) == 0.0
            and row.get("token_f1", 0.0) >= 0.5
            and not row.get("abstained")
        ):
            buckets["high_f1_em_zero"].append(row)
        if row.get("method") == "dense_rag_e5" and retrieved and row.get("abstained"):
            buckets["full_raw_retrieved_but_abstained"].append(row)
        if row.get("method") == "dense_budgeted_bsc" and not retrieved:
            buckets["oraclemem_missing_evidence"].append(row)
        if row.get("unsupported_answer", 0.0) > 0.0:
            buckets["unsupported_answer"].append(row)
        if row.get("abstained") and answer_looks_substantive:
            buckets["schema_conflict_answer_and_abstained"].append(row)
        if row.get("abstained") and row.get("evidence_use", 0.0) > 0.0:
            buckets["abstain_with_gold_citation"].append(row)
    return buckets


def compact_error_row(row: dict, max_text: int = 160) -> dict:
    prediction = str(row.get("prediction", ""))
    gold = str(row.get("gold_answer", ""))
    return {
        "question_id": row.get("question_id"),
        "question_type": row.get("question_type"),
        "method": row.get("method"),
        "method_label": row.get("method_label"),
        "gold_answer": gold[:max_text],
        "prediction": prediction[:max_text],
        "abstained": row.get("abstained"),
        "exact_match": row.get("exact_match"),
        "token_f1": row.get("token_f1"),
        "evidence_use": row.get("evidence_use"),
        "gold_session_ids": row.get("gold_session_ids", []),
        "context_session_ids": row.get("context_session_ids", []),
        "used_memory_ids": row.get("used_memory_ids", []),
        "prompt_hash": row.get("prompt_hash"),
    }


def derive_audit_row(row: dict) -> dict:
    gold = set(row.get("gold_session_ids", []))
    context = set(row.get("context_session_ids", []))
    support_in_context = bool(gold & context)
    answer_looks_substantive = bool(normalize_answer(row.get("prediction", ""))) and not is_insufficient_answer(
        row.get("prediction", "")
    )
    return {
        **compact_error_row(row, max_text=240),
        "retrieved_at_5": support_in_context,
        "support_in_context": support_in_context,
        "gold_recall_in_context": len(gold & context) / max(len(gold), 1),
        "retrieval_hit_but_abstained": bool(support_in_context and row.get("abstained")),
        "insufficient_despite_support": bool(support_in_context and row.get("abstained")),
        "evidence_used_but_wrong_answer": bool(
            row.get("evidence_use", 0.0) > 0.0
            and row.get("exact_match", 0.0) < 1.0
            and not row.get("abstained")
        ),
        "high_f1_em_zero": bool(
            row.get("exact_match", 0.0) == 0.0 and row.get("token_f1", 0.0) >= 0.5 and not row.get("abstained")
        ),
        "oraclemem_missing_evidence": bool(row.get("method") == "dense_budgeted_bsc" and not support_in_context),
        "unsupported_answer": bool(row.get("unsupported_answer", 0.0) > 0.0),
        "abstain_answer_conflict": bool(row.get("abstained") and answer_looks_substantive),
        "abstain_with_gold_citation": bool(row.get("abstained") and row.get("evidence_use", 0.0) > 0.0),
        "article_stripped_exact_match": article_stripped_exact_match(row.get("prediction", ""), row.get("gold_answer", "")),
    }


def method_bucket_summary(rows: list[dict], bucket_names: list[str]) -> dict:
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = {}
    for method, method_rows in sorted(by_method.items()):
        method_summary = {
            "method_label": METHOD_LABELS.get(method, method),
            "n": len(method_rows),
            "buckets": {},
        }
        for bucket in bucket_names:
            count = sum(1 for row in method_rows if row.get(bucket))
            method_summary["buckets"][bucket] = {
                "count": count,
                "rate": count / max(len(method_rows), 1),
            }
        summary[method] = method_summary
    return summary


def normalized_scoring_summary(rows: list[dict], focus_types: set[str]) -> dict:
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = {}
    for method, method_rows in sorted(by_method.items()):
        focus_rows = [row for row in method_rows if row.get("question_type") in focus_types]
        for row in method_rows:
            row["article_stripped_exact_match"] = article_stripped_exact_match(
                row.get("prediction", ""), row.get("gold_answer", "")
            )
        summary[method] = {
            "method_label": METHOD_LABELS.get(method, method),
            "overall_article_stripped_em": sum(row["article_stripped_exact_match"] for row in method_rows)
            / max(len(method_rows), 1),
            "focus_article_stripped_em": sum(row["article_stripped_exact_match"] for row in focus_rows)
            / max(len(focus_rows), 1),
            "overall_script_em": sum(row.get("exact_match", 0.0) for row in method_rows) / max(len(method_rows), 1),
            "focus_script_em": sum(row.get("exact_match", 0.0) for row in focus_rows) / max(len(focus_rows), 1),
        }
    return {
        "definition": "article_stripped_em lowercases, strips punctuation/articles, and collapses whitespace.",
        "metrics": summary,
    }


def analyze_error_run(run_dir: Path, *, focus_types: set[str], top_n: int = 50) -> dict:
    rows = load_reader_outputs(run_dir)
    derived_rows = [derive_audit_row(row) for row in rows]
    rows_by_method: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_method.setdefault(row["method"], []).append(row)

    conditional = {}
    for method, method_rows in sorted(rows_by_method.items()):
        focus_rows = [row for row in method_rows if row.get("question_type") in focus_types]
        conditional[method] = {
            "method_label": METHOD_LABELS.get(method, method),
            "overall": score_conditioned_on_retrieved(method_rows),
            "focus": score_conditioned_on_retrieved(focus_rows),
        }

    buckets = bucket_reader_errors(rows)
    bucket_names = list(buckets)
    bucket_summary = {
        name: {
            "count": len(bucket_rows),
            "examples": [
                compact_error_row(row)
                for row in sorted(
                    bucket_rows,
                    key=lambda item: (
                        item.get("method", ""),
                        item.get("question_type", ""),
                        item.get("token_f1", 0.0),
                    ),
                    reverse=True,
                )[:top_n]
            ],
        }
        for name, bucket_rows in buckets.items()
    }
    audit = {
        "run_dir": str(run_dir),
        "n_rows": len(rows),
        "focus_types": sorted(focus_types),
        "conditional_reader_analysis": conditional,
        "error_buckets": bucket_summary,
        "per_method_error_buckets": method_bucket_summary(derived_rows, bucket_names),
        "normalized_scoring": normalized_scoring_summary(rows, focus_types),
        "notes": [
            "retrieved means at least one gold answer-session id appears in the frozen context ids.",
            "Evidence use means the reader cited at least one gold answer-session id.",
            "high_f1_em_zero is a heuristic proxy for semantically plausible but exact-match-zero cases; it is not an LLM judge.",
        ],
    }
    (run_dir / "error_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (run_dir / "error_audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (run_dir / "error_audit_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in derived_rows:
            handle.write(json.dumps(row) + "\n")
    with (run_dir / "failure_examples.jsonl").open("w", encoding="utf-8") as handle:
        for bucket, bucket_rows in buckets.items():
            for row in bucket_rows[:top_n]:
                handle.write(json.dumps({"bucket": bucket, **compact_error_row(row, max_text=240)}) + "\n")
    semantic_candidates = [
        row
        for row in derived_rows
        if row["high_f1_em_zero"] or (row["evidence_used_but_wrong_answer"] and row.get("token_f1", 0.0) >= 0.25)
    ]
    with (run_dir / "semantic_audit_sample_50.jsonl").open("w", encoding="utf-8") as handle:
        for row in semantic_candidates[:50]:
            handle.write(json.dumps(row) + "\n")
    (run_dir / "normalized_scoring.json").write_text(json.dumps(audit["normalized_scoring"], indent=2), encoding="utf-8")
    write_error_audit_report(run_dir, audit)
    return audit


def write_error_audit_report(run_dir: Path, audit: dict) -> None:
    lines = [
        "# Reader Error Audit",
        "",
        f"- Run directory: `{audit['run_dir']}`",
        f"- Rows audited: `{audit['n_rows']}`",
        "- Retrieved evidence is defined as at least one gold answer-session id appearing in the frozen context ids.",
        "",
        "## Conditional Reader Analysis",
        "",
        "| Method | Any gold retrieved | Gold recall | EM given retrieved | F1 given retrieved | Abstain given retrieved | Evidence use given retrieved | n retrieved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in audit["conditional_reader_analysis"].items():
        focus = row["focus"]
        lines.append(
            f"| {row['method_label']} | {focus['any_gold_retrieved']:.4f} | "
            f"{focus['gold_recall']:.4f} | {focus['exact_match']:.4f} | "
            f"{focus['token_f1']:.4f} | {focus['insufficient_evidence_rate']:.4f} | "
            f"{focus['evidence_use']:.4f} | {focus['retrieved_count']} |"
        )
    lines.extend(["", "## Error Buckets", "", "| Bucket | Count |", "|---|---:|"])
    for name, row in audit["error_buckets"].items():
        lines.append(f"| `{name}` | {row['count']} |")
    lines.extend(
        [
            "",
            "## Per-Method Error Rates",
            "",
            "| Method | Insufficient despite support | Evidence used but wrong | Unsupported answer | Abstain-answer conflict |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _method, row in audit["per_method_error_buckets"].items():
        buckets = row["buckets"]
        lines.append(
            f"| {row['method_label']} | "
            f"{buckets['insufficient_despite_support']['rate']:.4f} | "
            f"{buckets['evidence_used_but_wrong_answer']['rate']:.4f} | "
            f"{buckets['unsupported_answer']['rate']:.4f} | "
            f"{buckets['schema_conflict_answer_and_abstained']['rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary Scoring Check",
            "",
            "| Method | Script EM | Article-stripped EM |",
            "|---|---:|---:|",
        ]
    )
    for _method, row in audit["normalized_scoring"]["metrics"].items():
        lines.append(
            f"| {row['method_label']} | {row['focus_script_em']:.4f} | {row['focus_article_stripped_em']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `retrieval_hit_but_abstained` is the main over-conservative-reader bucket.",
            "- `high_f1_em_zero` is a heuristic exact-match harshness bucket; use a blinded judge before reporting it as semantic correctness.",
            "- `oraclemem_missing_evidence` is the write/retrieval failure bucket for the OracleMem dense method.",
            "",
            "Detailed examples are in `error_audit_summary.json`, `error_audit_rows.jsonl`, `failure_examples.jsonl`, and `semantic_audit_sample_50.jsonl`.",
        ]
    )
    (run_dir / "ERROR_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path, summary: dict, methods: list[str], reader_name: str, reader_model: str | None) -> None:
    is_api = reader_name == "openrouter"
    lines = [
        "# LongMemEval-S Frozen-Context Reader Evaluation",
        "",
        f"- Reader: `{reader_name}`" + (f" / `{reader_model}`." if reader_model else "."),
        "- Scope: API reader evaluation on frozen contexts." if is_api else "- Scope: deterministic reporting-path validation, not a replacement for an API or local LLM reader.",
        "- Contexts: reconstructed from frozen top-5 retrieval ids without re-retrieval.",
        "- Metrics: exact match and token F1 against LongMemEval-S answers; evidence-use checks whether cited memory ids overlap gold answer-session ids.",
        "",
        "## Focus Reader Results",
        "",
        "| Method | Overall EM | Focus EM | Focus F1 | Evidence use | Unsupported answer | Insufficient rate | Parse fail | Avg context words | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = summary[method]
        focus = row["focus"]
        overall = row["overall"]
        lines.append(
            f"| {row['method_label']} | {overall['exact_match']:.4f} | {focus['exact_match']:.4f} | "
            f"{focus['token_f1']:.4f} | {focus['evidence_use']:.4f} | "
            f"{focus['unsupported_answer_rate']:.4f} | {focus['insufficient_evidence_rate']:.4f} | "
            f"{focus['parse_failure_rate']:.4f} | {focus['avg_context_words']:.1f} | "
            f"${focus['total_api_cost']:.4f} |"
        )
    deltas = summary.get("_paired_focus_deltas_vs_oraclemem_dense", {})
    if deltas:
        lines.extend(
            [
                "",
                "## Paired Focus Deltas",
                "",
                "| Baseline | EM delta | EM 95% CI | F1 delta | F1 95% CI | Evidence-use delta | Evidence-use 95% CI |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for baseline, row in deltas.items():
            em = row["exact_match"]
            f1 = row["token_f1"]
            ev = row["evidence_use"]
            lo, hi = em["ci95"]
            f1_lo, f1_hi = f1["ci95"]
            ev_lo, ev_hi = ev["ci95"]
            lines.append(
                f"| OracleMem writer + dense minus {row['baseline_label']} | {em['mean_delta']:+.4f} | "
                f"[{lo:+.4f}, {hi:+.4f}] | {f1['mean_delta']:+.4f} | "
                f"[{f1_lo:+.4f}, {f1_hi:+.4f}] | {ev['mean_delta']:+.4f} | "
                f"[{ev_lo:+.4f}, {ev_hi:+.4f}] |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Method names are hidden from the reader prompt; the prompt contains only the question and memory context.",
            "- `INSUFFICIENT_EVIDENCE` is reported as an insufficient-evidence output rate, not as abstention accuracy.",
            "- Old-answer/stale-answer rates require identifiable superseded-answer labels and are not reported here.",
        ]
    )
    if not is_api:
        lines.append("- This deterministic smoke reader is pipeline validation only, not a submission-grade LLM reader result.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_outputs(
    output_dir: Path,
    output: dict,
    artifacts: dict,
    methods: list[str],
    reader_name: str,
    reader_model: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    outputs_path = output_dir / "reader_outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as handle:
        for method in methods:
            for row in artifacts[method]:
                handle.write(json.dumps(row) + "\n")
    write_report(
        output_dir,
        output["metrics"],
        methods,
        reader_name=reader_name,
        reader_model=reader_model,
    )


def prompt_comparison_metrics(artifacts: dict[str, list[dict]], methods: list[str]) -> dict:
    comparison: dict[str, dict] = {}
    for method in methods:
        rows = sorted(artifacts[method], key=lambda row: row["question_id"])
        overall = score_predictions(rows)
        supported = score_conditioned_on_retrieved(rows)
        comparison[method] = {
            "method_label": METHOD_LABELS.get(method, method),
            "n": overall["n"],
            "exact_match": overall["exact_match"],
            "token_f1": overall["token_f1"],
            "evidence_use": overall["evidence_use"],
            "insufficient_evidence_rate": overall["insufficient_evidence_rate"],
            "abstain_given_supported": supported["insufficient_evidence_rate"],
            "gold_retrieved": supported["any_gold_retrieved"],
            "retrieved_count": supported["retrieved_count"],
            "unsupported_answer_rate": overall["unsupported_answer_rate"],
            "parse_failure_rate": overall["parse_failure_rate"],
            "total_api_cost": overall["total_api_cost"],
        }
    return comparison


def choose_prompt_mode(comparison: dict[str, dict], methods: list[str]) -> dict:
    baseline_name = "answer_if_supported" if "answer_if_supported" in comparison else next(iter(comparison))
    baseline = comparison[baseline_name]
    fairness_methods = [method for method in ("dense_budgeted_bsc", "dense_rag_e5") if method in methods]
    if not fairness_methods:
        fairness_methods = methods

    candidates = []
    for prompt_mode, method_rows in comparison.items():
        parse_max = max(method_rows[method]["parse_failure_rate"] for method in methods)
        unsupported_increase = max(
            method_rows[method]["unsupported_answer_rate"] - baseline[method]["unsupported_answer_rate"]
            for method in methods
        )
        f1_stable = all(
            method_rows[method]["token_f1"] >= baseline[method]["token_f1"] - 0.01
            for method in fairness_methods
        )
        mean_abstain_supported = sum(
            method_rows[method]["abstain_given_supported"] for method in fairness_methods
        ) / len(fairness_methods)
        mean_f1 = sum(method_rows[method]["token_f1"] for method in fairness_methods) / len(fairness_methods)
        eligible = parse_max < 0.01 and unsupported_increase <= 0.05 and f1_stable
        candidates.append(
            {
                "prompt_mode": prompt_mode,
                "eligible": eligible,
                "parse_failure_max": parse_max,
                "unsupported_answer_max_increase_vs_baseline": unsupported_increase,
                "f1_stable_for_oraclemem_and_full_raw": f1_stable,
                "mean_abstain_given_supported_oraclemem_full_raw": mean_abstain_supported,
                "mean_f1_oraclemem_full_raw": mean_f1,
            }
        )
    eligible_candidates = [row for row in candidates if row["eligible"]]
    if not eligible_candidates:
        selected = baseline_name
    else:
        selected = sorted(
            eligible_candidates,
            key=lambda row: (
                row["mean_abstain_given_supported_oraclemem_full_raw"],
                -row["mean_f1_oraclemem_full_raw"],
                row["prompt_mode"],
            ),
        )[0]["prompt_mode"]
    return {
        "baseline_prompt": baseline_name,
        "selected_prompt": selected,
        "criteria": [
            "Minimize abstain_given_supported averaged over OracleMem dense and full raw dense, not OracleMem alone.",
            "Require parse failure below 1%.",
            "Require unsupported-answer rate not to increase by more than 5 absolute points versus answer_if_supported.",
            "Require OracleMem and full raw dense F1 to stay within 0.01 of baseline or improve.",
        ],
        "candidates": candidates,
    }


def write_prompt_dev_report(output_dir: Path, comparison: dict[str, dict], selection: dict, methods: list[str]) -> None:
    (output_dir / "prompt_comparison_summary.json").write_text(
        json.dumps(
            {
                "selection": selection,
                "metrics": comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Prompt Dev Report",
        "",
        "- Split: deterministic 50-question LongMemEval-S focus dev split.",
        "- Reader: GPT-5.5 through OpenRouter when run with `--reader openrouter --reader-model openai/gpt-5.5`.",
        "- Selection rule: choose by the predeclared criteria from the sprint review, prioritizing lower supported-case abstention without increasing unsupported answers or harming full raw dense.",
        f"- Selected prompt by script criteria: `{selection['selected_prompt']}`.",
        "",
        "## Prompt Comparison",
        "",
        "| Prompt | Method | EM | F1 | Evidence use | Insufficient | Abstain given supported | Unsupported | Parse fail | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt_mode in comparison:
        for method in methods:
            row = comparison[prompt_mode][method]
            lines.append(
                f"| `{prompt_mode}` | {row['method_label']} | "
                f"{row['exact_match']:.4f} | {row['token_f1']:.4f} | {row['evidence_use']:.4f} | "
                f"{row['insufficient_evidence_rate']:.4f} | {row['abstain_given_supported']:.4f} | "
                f"{row['unsupported_answer_rate']:.4f} | {row['parse_failure_rate']:.4f} | "
                f"${row['total_api_cost']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Selection Diagnostics",
            "",
            "| Prompt | Eligible | Max parse fail | Max unsupported increase | F1 stable for OracleMem/full raw | Mean abstain given supported | Mean F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selection["candidates"]:
        lines.append(
            f"| `{row['prompt_mode']}` | {str(row['eligible']).lower()} | "
            f"{row['parse_failure_max']:.4f} | {row['unsupported_answer_max_increase_vs_baseline']:.4f} | "
            f"{str(row['f1_stable_for_oraclemem_and_full_raw']).lower()} | "
            f"{row['mean_abstain_given_supported_oraclemem_full_raw']:.4f} | "
            f"{row['mean_f1_oraclemem_full_raw']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Per-prompt outputs are under `prompt_<mode>/` subdirectories.",
            "- Machine-readable comparison is in `prompt_comparison_summary.json`.",
        ]
    )
    (output_dir / "PROMPT_DEV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-errors", action="store_true")
    parser.add_argument("--make-split", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--cache-json", type=Path, default=Path("llm_memory_validation/cache/longmemeval_s_cleaned.json"))
    parser.add_argument("--retrieval-rows-json", type=Path, default=Path("llm_memory_validation/competitor_run_v2/retrieval_rows.json"))
    parser.add_argument("--output-dir", "--out", dest="output_dir", type=Path, default=Path("llm_memory_validation/longmemeval_reader_smoke"))
    parser.add_argument("--methods", type=csv_arg, default=DEFAULT_METHODS)
    parser.add_argument("--focus-types", type=csv_arg, default=sorted(FOCUS_TYPES))
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--focus-only", action="store_true")
    parser.add_argument("--per-type-limit", type=int, default=0)
    parser.add_argument("--budget-frac", type=float, default=0.20)
    parser.add_argument("--max-context-words", type=int, default=1800)
    parser.add_argument("--reader", "--provider", dest="reader", choices=["extractive_presence_smoke", "openrouter"], default="extractive_presence_smoke")
    parser.add_argument("--reader-model", "--model", dest="reader_model", type=str, default="openai/gpt-5.4-mini")
    parser.add_argument("--prompt-style", choices=["strict", *PROMPT_MODES], default=None)
    parser.add_argument("--prompt-mode", type=csv_arg, default=None)
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--api-cache", type=Path, default=None)
    parser.add_argument("--api-max-tokens", type=int, default=160)
    parser.add_argument("--api-timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--verbosity", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--shuffle-jobs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--save-prompts", action="store_true")
    args = parser.parse_args()

    focus_types = set(args.focus_types)
    if args.make_split:
        if args.source is None:
            raise SystemExit("--make-split requires --source")
        summary = make_focus_dev_eval_split(args.source, args.dev_size, args.output_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.analyze_errors:
        if args.run_dir is None:
            raise SystemExit("--analyze-errors requires --run-dir")
        audit = analyze_error_run(args.run_dir, focus_types=focus_types)
        print(json.dumps(audit, indent=2))
        return

    all_examples = load_examples(args.dataset_json, args.cache_json)
    if args.split is not None:
        split_ids = load_split_question_ids(args.split)
        examples = [example for example in all_examples if example["question_id"] in split_ids]
        found_ids = {example["question_id"] for example in examples}
        missing_ids = sorted(split_ids - found_ids)
        if missing_ids:
            raise ValueError(f"{len(missing_ids)} split question_id values were not found in the dataset, e.g. {missing_ids[:5]}")
        examples.sort(key=lambda example: example["question_id"])
    else:
        examples = filter_examples(
            all_examples,
            focus_types,
            focus_only=args.focus_only,
            per_type_limit=args.per_type_limit,
            seed=args.seed,
        )
    retrieval_rows = json.loads(args.retrieval_rows_json.read_text(encoding="utf-8"))
    methods = canonical_method_list(args.methods)
    prompt_modes = validate_prompt_modes(args.prompt_mode or [args.prompt_style or "strict"])
    openrouter_reader = None
    if args.reader == "openrouter":
        env = load_env_file(args.api_env)
        api_key = env.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(f"OPENROUTER_API_KEY not found in {args.api_env}")
        api_cache = args.api_cache or (args.output_dir / "openrouter_cache.json")
        openrouter_reader = OpenRouterReader(
            api_key=api_key,
            model=args.reader_model,
            cache_path=api_cache,
            max_tokens=args.api_max_tokens,
            temperature=args.temperature,
            request_sleep=args.request_sleep,
            timeout=args.api_timeout,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
        )
    prompt_comparison: dict[str, dict] = {}
    final_outputs: dict[str, dict] = {}
    for prompt_mode in prompt_modes:
        summary, artifacts = evaluate(
            examples=examples,
            retrieval_rows=retrieval_rows,
            methods=methods,
            focus_types=focus_types,
            budget_frac=args.budget_frac,
            max_context_words=args.max_context_words,
            save_prompts=args.save_prompts,
            reader_name=args.reader,
            openrouter_reader=openrouter_reader,
            shuffle_jobs=args.shuffle_jobs,
            seed=args.seed,
            bootstrap=args.bootstrap,
            prompt_style=prompt_mode,
        )
        output = {
            "dataset": str(args.dataset_json or args.cache_json),
            "retrieval_rows": str(args.retrieval_rows_json),
            "split": str(args.split) if args.split else None,
            "reader": args.reader,
            "reader_model": args.reader_model if args.reader == "openrouter" else None,
            "scope": "API reader" if args.reader == "openrouter" else "deterministic smoke; not an LLM reader",
            "focus_types": args.focus_types,
            "focus_only": args.focus_only,
            "per_type_limit": args.per_type_limit,
            "prompt_style": prompt_mode,
            "prompt_mode": prompt_mode,
            "temperature": args.temperature,
            "api_max_tokens": args.api_max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "verbosity": args.verbosity,
            "methods": methods,
            "requested_methods": args.methods,
            "metrics": summary,
        }
        run_output_dir = args.output_dir if len(prompt_modes) == 1 else args.output_dir / f"prompt_{prompt_mode}"
        write_evaluation_outputs(
            run_output_dir,
            output,
            artifacts,
            methods,
            reader_name=args.reader,
            reader_model=args.reader_model if args.reader == "openrouter" else None,
        )
        prompt_comparison[prompt_mode] = prompt_comparison_metrics(artifacts, methods)
        final_outputs[prompt_mode] = output

    if len(prompt_modes) > 1:
        selection = choose_prompt_mode(prompt_comparison, methods)
        write_prompt_dev_report(args.output_dir, prompt_comparison, selection, methods)
        final_outputs["_prompt_dev_selection"] = selection
        print(json.dumps(final_outputs, indent=2))
    else:
        print(json.dumps(next(iter(final_outputs.values())), indent=2))


if __name__ == "__main__":
    main()
