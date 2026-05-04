from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import string
import textwrap
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = "https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json"
QUESTION_TYPES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
]
METHOD_SPECS = {
    "fifo_replay": "Newest raw sessions until the shared budget fills.",
    "uniform_replay": "Evenly spaced raw sessions under the same budget.",
    "replay_only_router": "Heuristic segment scoring, but memory can only keep raw replay entries.",
    "bsc": "OracleMem-style budgeted writer with discard / replay / cache / consolidate.",
}
METHOD_LABELS = {
    "fifo_replay": "FIFO raw replay",
    "uniform_replay": "Uniform raw replay",
    "replay_only_router": "Budgeted raw replay router",
    "bsc": "OracleMem writer",
}

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


def load_dataset() -> list[dict]:
    with urllib.request.urlopen(DATA_URL) as handle:
        return json.load(handle)


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
    sub_session = session[-turns:]
    return session_text(sub_session)


def session_features(session: list[dict], index: int, total: int) -> dict[str, float]:
    raw_text = session_text(session)
    user_turns = sum(1 for turn in session if turn["role"] == "user")
    assistant_turns = len(session) - user_turns
    fact_lines = extract_fact_lines(session)
    features = {
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
    return features


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
    ordered = list(reversed(candidates))
    return take_under_budget(ordered, budget_words)


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


def take_under_budget(entries: Iterable[MemoryEntry], budget_words: int) -> list[MemoryEntry]:
    kept: list[MemoryEntry] = []
    used = 0
    for entry in entries:
        if used + entry.cost_words > budget_words:
            continue
        kept.append(entry)
        used += entry.cost_words
    return kept


def retrieve_entries(question: str, entries: list[MemoryEntry], topk: int) -> list[MemoryEntry]:
    if not entries:
        return []
    documents = [entry.text for entry in entries]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(documents + [question])
    similarities = cosine_similarity(matrix[:-1], matrix[-1]).reshape(-1)
    ranked: list[tuple[float, MemoryEntry]] = []
    for similarity, entry in zip(similarities, entries):
        recency_bonus = {"cache": 0.03, "consolidate": 0.02, "replay": 0.0}.get(entry.action, 0.0)
        ranked.append((float(similarity) + recency_bonus, entry))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ranked[:topk]]


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = no_punct.split()
    return " ".join(tokens)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


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


def generation_subset(examples: list[dict], per_type: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_type[example["question_type"]].append(index)
    selected: list[int] = []
    for question_type in QUESTION_TYPES:
        indices = list(by_type[question_type])
        rng.shuffle(indices)
        selected.extend(indices[:per_type])
    selected.sort()
    return selected


def prompt_from_entries(question: str, entries: list[MemoryEntry], prompt_word_budget: int) -> str:
    used = 0
    rendered_entries: list[str] = []
    for rank, entry in enumerate(entries, start=1):
        text_words = entry.text.split()
        max_words_for_item = min(len(text_words), 400)
        clipped = " ".join(text_words[:max_words_for_item])
        block = f"[{rank}] action={entry.action} session={entry.session_id}\n{clipped}"
        block_cost = count_words(block)
        if rendered_entries and used + block_cost > prompt_word_budget:
            break
        rendered_entries.append(block)
        used += block_cost
    memory_block = "\n\n".join(rendered_entries) if rendered_entries else "[no memory retained]"
    return textwrap.dedent(
        f"""
        You answer questions from a compressed long-term memory store.
        Use only the memory below.
        Give a short factual answer.
        If the memory is insufficient, answer with "unknown".

        Question:
        {question}

        Memory:
        {memory_block}

        Answer:
        """
    ).strip()


def evaluate_retrieval(examples: list[dict], budget_frac: float, topk: int) -> tuple[dict, dict]:
    builders = {
        "fifo_replay": build_fifo_replay,
        "uniform_replay": build_uniform_replay,
        "replay_only_router": build_replay_only_router,
        "bsc": build_bsc,
    }
    metrics_by_method: dict[str, dict] = {}
    artifacts: dict[str, list[dict]] = {}
    for method_name, builder in builders.items():
        recall_scores: list[float] = []
        reciprocal_ranks: list[float] = []
        action_counter: Counter[str] = Counter()
        actions_by_question_type: dict[str, Counter[str]] = defaultdict(Counter)
        decision_counter: Counter[str] = Counter()
        decision_by_question_type: dict[str, Counter[str]] = defaultdict(Counter)
        per_type_recall: dict[str, list[float]] = defaultdict(list)
        rows: list[dict] = []
        for example in examples:
            entries = builder(example, budget_frac)
            retrieved = retrieve_entries(example["question"], entries, topk=topk)
            gold_ids = set(example["answer_session_ids"])
            predicted_ids = [entry.session_id for entry in retrieved]
            hit_positions = [rank for rank, session_id in enumerate(predicted_ids, start=1) if session_id in gold_ids]
            recall_value = len(set(predicted_ids) & gold_ids) / max(len(gold_ids), 1)
            rr_value = 0.0 if not hit_positions else 1.0 / min(hit_positions)
            recall_scores.append(recall_value)
            reciprocal_ranks.append(rr_value)
            per_type_recall[example["question_type"]].append(recall_value)
            if method_name == "bsc":
                total = len(example["haystack_sessions"])
                for index, session in enumerate(example["haystack_sessions"]):
                    action = classify_action(session, index, total)
                    decision_counter[action] += 1
                    decision_by_question_type[example["question_type"]][action] += 1
            else:
                replay_decisions = len(example["haystack_sessions"])
                decision_counter["replay"] += replay_decisions
                decision_by_question_type[example["question_type"]]["replay"] += replay_decisions
            for entry in entries:
                action_counter[entry.action] += 1
                actions_by_question_type[example["question_type"]][entry.action] += 1
            rows.append(
                {
                    "question_id": example["question_id"],
                    "question_type": example["question_type"],
                    "gold_session_ids": example["answer_session_ids"],
                    "predicted_session_ids": predicted_ids,
                    "retrieved_entries": [
                        {
                            "session_id": entry.session_id,
                            "action": entry.action,
                            "cost_words": entry.cost_words,
                        }
                        for entry in retrieved
                    ],
                }
            )
        metrics_by_method[method_name] = {
            "recall_at_5": sum(recall_scores) / len(recall_scores),
            "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "avg_retained_entries": statistics.mean(
                len(builder(example, budget_frac)) for example in examples
            ),
            "avg_full_words": statistics.mean(full_budget_words(example) for example in examples),
            "avg_budget_words": statistics.mean(max(256, int(full_budget_words(example) * budget_frac)) for example in examples),
            "action_usage": dict(action_counter),
            "per_type_recall_at_5": {
                question_type: sum(values) / len(values) for question_type, values in per_type_recall.items()
            },
            "decision_usage": dict(decision_counter),
            "action_usage_by_question_type": {
                question_type: dict(counter) for question_type, counter in actions_by_question_type.items()
            },
            "decision_usage_by_question_type": {
                question_type: dict(counter) for question_type, counter in decision_by_question_type.items()
            },
        }
        artifacts[method_name] = rows
    return metrics_by_method, artifacts


def run_generation(
    examples: list[dict],
    retrieval_rows: dict[str, list[dict]],
    budget_frac: float,
    model_name: str,
    per_type_subset: int,
    seed: int,
    prompt_word_budget: int,
    max_new_tokens: int,
) -> tuple[dict, dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subset_indices = generation_subset(examples, per_type=per_type_subset, seed=seed)
    subset_lookup = {index: examples[index] for index in subset_indices}
    rows_by_method = {method: {row["question_id"]: row for row in rows} for method, rows in retrieval_rows.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    generation_metrics: dict[str, dict] = {}
    generation_artifacts: dict[str, list[dict]] = {}
    for method_name, row_lookup in rows_by_method.items():
        predictions: list[dict] = []
        em_scores: list[float] = []
        f1_scores: list[float] = []
        per_type_em: dict[str, list[float]] = defaultdict(list)
        per_type_f1: dict[str, list[float]] = defaultdict(list)
        for index in subset_indices:
            example = subset_lookup[index]
            question_id = example["question_id"]
            retrieval_row = row_lookup[question_id]
            entry_lookup = {}
            if method_name == "fifo_replay":
                entries = build_fifo_replay(example, budget_frac)
            elif method_name == "uniform_replay":
                entries = build_uniform_replay(example, budget_frac)
            elif method_name == "replay_only_router":
                entries = build_replay_only_router(example, budget_frac)
            else:
                entries = build_bsc(example, budget_frac)
            for entry in entries:
                entry_lookup[entry.session_id] = entry
            retrieved_entries = [entry_lookup[item["session_id"]] for item in retrieval_row["retrieved_entries"] if item["session_id"] in entry_lookup]
            prompt = prompt_from_entries(
                question=example["question"],
                entries=retrieved_entries,
                prompt_word_budget=prompt_word_budget,
            )
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion_tokens = generated[0][model_inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            prediction = prediction.split("\n")[0].strip()
            gold = example["answer"]
            em_value = exact_match(prediction, gold)
            f1_value = token_f1(prediction, gold)
            em_scores.append(em_value)
            f1_scores.append(f1_value)
            per_type_em[example["question_type"]].append(em_value)
            per_type_f1[example["question_type"]].append(f1_value)
            predictions.append(
                {
                    "question_id": question_id,
                    "question_type": example["question_type"],
                    "gold_answer": gold,
                    "prediction": prediction,
                    "exact_match": em_value,
                    "token_f1": f1_value,
                }
            )
        generation_metrics[method_name] = {
            "subset_size": len(subset_indices),
            "exact_match": sum(em_scores) / len(em_scores),
            "token_f1": sum(f1_scores) / len(f1_scores),
            "per_type_exact_match": {
                question_type: sum(values) / len(values) for question_type, values in per_type_em.items()
            },
            "per_type_token_f1": {
                question_type: sum(values) / len(values) for question_type, values in per_type_f1.items()
            },
            "model_name": model_name,
        }
        generation_artifacts[method_name] = predictions
    return generation_metrics, {"subset_indices": subset_indices, "predictions": generation_artifacts}


def plot_metrics(output_dir: Path, retrieval_metrics: dict, generation_metrics: dict | None) -> None:
    methods = list(METHOD_SPECS.keys())
    labels = [METHOD_LABELS.get(name, name).replace(" ", "\n") for name in methods]

    plt.figure(figsize=(8, 4.5))
    recall_values = [retrieval_metrics[name]["recall_at_5"] for name in methods]
    mrr_values = [retrieval_metrics[name]["mrr_at_5"] for name in methods]
    x = list(range(len(methods)))
    width = 0.38
    plt.bar([value - width / 2 for value in x], recall_values, width=width, label="Recall@5")
    plt.bar([value + width / 2 for value in x], mrr_values, width=width, label="MRR@5")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("LongMemEval-S Retrieval Under Equal Memory Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_metrics.png", dpi=200)
    plt.close()

    if generation_metrics is not None:
        plt.figure(figsize=(8, 4.5))
        em_values = [generation_metrics[name]["exact_match"] for name in methods]
        f1_values = [generation_metrics[name]["token_f1"] for name in methods]
        plt.bar([value - width / 2 for value in x], em_values, width=width, label="Exact Match")
        plt.bar([value + width / 2 for value in x], f1_values, width=width, label="Token F1")
        plt.xticks(x, labels)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Score")
        plt.title("Reader EM/F1 on Stratified Generation Subset")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "generation_metrics.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    actions = ["discard", "replay", "cache", "consolidate"]
    bottom = [0.0] * len(methods)
    for action in actions:
        values = []
        for method in methods:
            usage = retrieval_metrics[method]["decision_usage"]
            total = sum(usage.values()) or 1
            values.append(usage.get(action, 0) / total)
        plt.bar(labels, values, bottom=bottom, label=action)
        bottom = [current + value for current, value in zip(bottom, values)]
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction of Stored Items")
    plt.title("Memory Action Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "action_distribution.png", dpi=200)
    plt.close()


def write_report(
    output_dir: Path,
    budget_frac: float,
    retrieval_metrics: dict,
    generation_metrics: dict | None,
    generation_subset_size: int,
) -> None:
    best_retrieval = max(retrieval_metrics.items(), key=lambda item: item[1]["recall_at_5"])
    report_lines = [
        "# Fast LLM Memory Validation",
        "",
        f"- Dataset: `LongMemEval-S` (`{len(QUESTION_TYPES)}` question types, 500 examples)",
        f"- Shared memory budget: `{budget_frac:.0%}` of the original haystack words per example",
        "- Methods: FIFO raw replay, uniform raw replay, budgeted raw replay router, OracleMem writer",
        "- Retrieval metric: `Recall@5` and `MRR@5` against the gold `answer_session_ids`",
        f"- Reader metric: stratified subset with `{generation_subset_size}` examples per question type" if generation_metrics is not None else "- Reader metric: not run",
        "",
        "## Retrieval",
        "",
    ]
    for method_name, metrics in retrieval_metrics.items():
        label = METHOD_LABELS.get(method_name, method_name)
        report_lines.extend(
            [
                f"### {label}",
                f"- Artifact key: `{method_name}`",
                f"- Recall@5: `{metrics['recall_at_5']:.4f}`",
                f"- MRR@5: `{metrics['mrr_at_5']:.4f}`",
                f"- Avg retained entries: `{metrics['avg_retained_entries']:.2f}`",
                f"- Action usage: `{metrics['action_usage']}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Takeaway",
            "",
            f"- Best retrieval method: `{METHOD_LABELS.get(best_retrieval[0], best_retrieval[0])}` with Recall@5 `{best_retrieval[1]['recall_at_5']:.4f}` and MRR@5 `{best_retrieval[1]['mrr_at_5']:.4f}`.",
        ]
    )
    if generation_metrics is not None:
        best_generation = max(generation_metrics.items(), key=lambda item: item[1]["token_f1"])
        report_lines.extend(
            [
                f"- Best reader token F1: `{METHOD_LABELS.get(best_generation[0], best_generation[0])}` with Token F1 `{best_generation[1]['token_f1']:.4f}` and EM `{best_generation[1]['exact_match']:.4f}`.",
                "",
                "## Reader",
                "",
            ]
        )
        for method_name, metrics in generation_metrics.items():
            label = METHOD_LABELS.get(method_name, method_name)
            report_lines.extend(
                [
                    f"### {label}",
                    f"- Artifact key: `{method_name}`",
                    f"- Exact Match: `{metrics['exact_match']:.4f}`",
                    f"- Token F1: `{metrics['token_f1']:.4f}`",
                    f"- Model: `{metrics['model_name']}`",
                    "",
                ]
            )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-frac", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generation-per-type", type=int, default=20)
    parser.add_argument("--generation-seed", type=int, default=7)
    parser.add_argument("--prompt-word-budget", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--reader-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_dataset()

    retrieval_metrics, retrieval_rows = evaluate_retrieval(
        examples=examples,
        budget_frac=args.budget_frac,
        topk=args.topk,
    )
    generation_metrics = None
    generation_payload = None
    if args.run_generation:
        generation_metrics, generation_payload = run_generation(
            examples=examples,
            retrieval_rows=retrieval_rows,
            budget_frac=args.budget_frac,
            model_name=args.reader_model,
            per_type_subset=args.generation_per_type,
            seed=args.generation_seed,
            prompt_word_budget=args.prompt_word_budget,
            max_new_tokens=args.max_new_tokens,
        )

    summary = {
        "dataset_url": DATA_URL,
        "budget_frac": args.budget_frac,
        "topk": args.topk,
        "methods": METHOD_SPECS,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "retrieval_rows.json").write_text(json.dumps(retrieval_rows, indent=2), encoding="utf-8")
    if generation_payload is not None:
        (args.output_dir / "generation_payload.json").write_text(
            json.dumps(generation_payload, indent=2),
            encoding="utf-8",
        )

    plot_metrics(args.output_dir, retrieval_metrics=retrieval_metrics, generation_metrics=generation_metrics)
    write_report(
        output_dir=args.output_dir,
        budget_frac=args.budget_frac,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        generation_subset_size=args.generation_per_type,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
