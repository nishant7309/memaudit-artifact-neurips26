from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm_memory_validation.bsc_longmemeval import (
    QUESTION_TYPES,
    MemoryEntry,
    TIME_RE,
    UPDATE_RE,
    build_bsc,
    build_fifo_replay,
    build_replay_only_router,
    build_uniform_replay,
    classify_action,
    count_words,
    extract_fact_lines,
    full_budget_words,
    load_dataset,
    make_entry,
    normalize_answer,
    retrieve_entries,
    session_features,
    session_text,
    tail_snippet,
    token_f1,
)


ACTIONS = ["discard", "replay", "cache", "consolidate"]
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}
PREFERENCE_HINTS = ("prefer", "favorite", "like", "love", "enjoy")
METHOD_ORDER = [
    "fifo_replay",
    "uniform_replay",
    "replay_only_router",
    "heuristic_bsc",
    "oracle_bsc",
    "learned_bsc",
]


@dataclass
class ControllerBundle:
    pipeline: Pipeline
    seed: int
    train_accuracy: float
    val_accuracy: float
    train_macro_f1: float
    val_macro_f1: float


@dataclass
class OracleDecision:
    action: str
    best_utility: float
    utility_by_action: dict[str, float]


def keyword_overlap(lhs: str, rhs: str) -> float:
    lhs_tokens = set(normalize_answer(lhs).split())
    rhs_tokens = set(normalize_answer(rhs).split())
    if not lhs_tokens or not rhs_tokens:
        return 0.0
    return len(lhs_tokens & rhs_tokens) / len(lhs_tokens | rhs_tokens)


def question_features(question: str) -> dict[str, float]:
    normalized = normalize_answer(question)
    return {
        "question_words": len(normalized.split()),
        "question_time_hits": float(bool(TIME_RE.search(question))),
        "question_update_hits": float(bool(UPDATE_RE.search(question))),
        "question_pref_hits": float(any(token in normalized for token in PREFERENCE_HINTS)),
    }


def action_renderings(session: list[dict], session_id: str, index: int) -> dict[str, MemoryEntry | None]:
    return {
        action: make_entry(session, session_id, index, action) if action != "discard" else None
        for action in ACTIONS
    }


def oracle_action_for_session(example: dict, index: int, budget_frac: float) -> OracleDecision:
    session = example["haystack_sessions"][index]
    session_id = example["haystack_session_ids"][index]
    renderings = action_renderings(session, session_id, index)
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    gold_ids = set(example["answer_session_ids"])
    gold_answer = str(example["answer"])
    question = example["question"]
    question_type = example["question_type"]
    session_id_is_gold = float(session_id in gold_ids)
    question_is_temporal = float(question_type == "temporal-reasoning" or bool(TIME_RE.search(question)))
    question_is_update = float(question_type == "knowledge-update" or bool(UPDATE_RE.search(question)))
    question_is_preference = float(question_type in {"single-session-user", "single-session-preference"})
    multi_session_need = float(len(gold_ids) > 1 or question_type == "multi-session")
    utilities: dict[str, float] = {"discard": 0.0}

    for action in ("replay", "cache", "consolidate"):
        entry = renderings[action]
        assert entry is not None
        mem_cost = entry.cost_words / max(budget_words, 1)
        compute_cost = {"replay": 1.0, "cache": 0.35, "consolidate": 0.20}[action]
        answer_overlap = token_f1(entry.text, gold_answer)
        question_overlap = keyword_overlap(entry.text, question)
        temporal_detail = float(bool(TIME_RE.search(entry.text)))
        update_detail = float(bool(UPDATE_RE.search(entry.text)))
        preference_detail = float(any(token in normalize_answer(entry.text) for token in PREFERENCE_HINTS))
        utility = (
            2.8 * session_id_is_gold
            + 1.4 * answer_overlap
            + 0.8 * question_overlap
            + 0.55 * question_is_temporal * temporal_detail * float(action in {"replay", "cache"})
            + 0.45 * question_is_update * update_detail * float(action in {"cache", "consolidate"})
            + 0.40 * question_is_preference * preference_detail * float(action == "consolidate")
            + 0.30 * multi_session_need * float(action in {"replay", "cache"})
            - 0.65 * mem_cost
            - 0.18 * compute_cost
        )
        if action == "consolidate" and question_is_temporal and not temporal_detail:
            utility -= 0.25
        if action == "cache" and not (question_is_temporal or question_is_update):
            utility -= 0.05
        if action == "replay" and question_is_preference and answer_overlap < 0.1:
            utility -= 0.10
        utilities[action] = utility

    best_action, best_utility = max(utilities.items(), key=lambda item: item[1])
    if best_utility <= 0.0:
        best_action = "discard"
        best_utility = 0.0
    return OracleDecision(action=best_action, best_utility=best_utility, utility_by_action=utilities)


def feature_vector(example: dict, index: int, budget_frac: float) -> list[float]:
    session = example["haystack_sessions"][index]
    session_id = example["haystack_session_ids"][index]
    total = len(example["haystack_sessions"])
    feat = session_features(session, index, total)
    qfeat = question_features(example["question"])
    renderings = action_renderings(session, session_id, index)
    raw_text = session_text(session)
    cache_text = renderings["cache"].text if renderings["cache"] is not None else tail_snippet(session, turns=4)
    consolidate_text = (
        renderings["consolidate"].text
        if renderings["consolidate"] is not None
        else "\n".join(f"fact: {line}" for line in extract_fact_lines(session))
    )
    budget_words = max(256, int(full_budget_words(example) * budget_frac))

    vector = [
        math.log1p(feat["words"]),
        feat["user_turns"],
        feat["assistant_turns"],
        feat["fact_hits"],
        feat["update_hits"],
        feat["time_hits"],
        feat["number_hits"],
        feat["fact_lines"],
        feat["recent_frac"],
        feat["assistant_only"],
        feat["generic_assistant"],
        qfeat["question_words"],
        qfeat["question_time_hits"],
        qfeat["question_update_hits"],
        qfeat["question_pref_hits"],
        keyword_overlap(raw_text, example["question"]),
        keyword_overlap(cache_text, example["question"]),
        keyword_overlap(consolidate_text, example["question"]),
        count_words(raw_text) / budget_words,
        count_words(cache_text) / budget_words,
        count_words(consolidate_text) / budget_words,
        float(bool(TIME_RE.search(raw_text))),
        float(bool(UPDATE_RE.search(raw_text))),
    ]
    return vector


def build_oracle_bsc(example: dict, budget_frac: float) -> tuple[list[MemoryEntry], list[str], list[float]]:
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    candidates: list[tuple[float, float, int, MemoryEntry]] = []
    decisions: list[str] = []
    utilities: list[float] = []
    for index, session_id in enumerate(example["haystack_session_ids"]):
        decision = oracle_action_for_session(example, index, budget_frac)
        decisions.append(decision.action)
        utilities.append(decision.best_utility)
        if decision.action == "discard":
            continue
        entry = make_entry(example["haystack_sessions"][index], session_id, index, decision.action)
        assert entry is not None
        density = decision.best_utility / max(entry.cost_words, 1)
        candidates.append((density, decision.best_utility, -index, entry))
    kept = []
    used = 0
    for _, _, _, entry in sorted(candidates, reverse=True):
        if used + entry.cost_words > budget_words:
            continue
        kept.append(entry)
        used += entry.cost_words
    return kept, decisions, utilities


def build_dataset_rows(examples: list[dict], budget_frac: float) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    for example in examples:
        for index in range(len(example["haystack_sessions"])):
            decision = oracle_action_for_session(example, index, budget_frac)
            features.append(feature_vector(example, index, budget_frac))
            labels.append(ACTION_TO_ID[decision.action])
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def train_controller(
    train_examples: list[dict],
    val_examples: list[dict],
    budget_frac: float,
    seeds: list[int],
) -> tuple[ControllerBundle, list[dict]]:
    train_x, train_y = build_dataset_rows(train_examples, budget_frac)
    val_x, val_y = build_dataset_rows(val_examples, budget_frac)
    bundles: list[ControllerBundle] = []
    metrics: list[dict] = []
    for seed in seeds:
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        batch_size=256,
                        max_iter=200,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                    ),
                ),
            ]
        )
        pipeline.fit(train_x, train_y)
        train_pred = pipeline.predict(train_x)
        val_pred = pipeline.predict(val_x)
        bundle = ControllerBundle(
            pipeline=pipeline,
            seed=seed,
            train_accuracy=accuracy_score(train_y, train_pred),
            val_accuracy=accuracy_score(val_y, val_pred),
            train_macro_f1=f1_score(train_y, train_pred, average="macro"),
            val_macro_f1=f1_score(val_y, val_pred, average="macro"),
        )
        bundles.append(bundle)
        metrics.append(
            {
                "seed": seed,
                "train_accuracy": bundle.train_accuracy,
                "val_accuracy": bundle.val_accuracy,
                "train_macro_f1": bundle.train_macro_f1,
                "val_macro_f1": bundle.val_macro_f1,
            }
        )
    best = max(bundles, key=lambda item: (item.val_macro_f1, item.val_accuracy))
    return best, metrics


def build_learned_bsc(
    example: dict,
    budget_frac: float,
    controller: ControllerBundle,
) -> tuple[list[MemoryEntry], list[str], list[float]]:
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    candidates: list[tuple[float, float, int, MemoryEntry]] = []
    decisions: list[str] = []
    confidences: list[float] = []
    for index, session_id in enumerate(example["haystack_session_ids"]):
        features = np.asarray([feature_vector(example, index, budget_frac)], dtype=np.float32)
        probabilities = controller.pipeline.predict_proba(features)[0]
        action_id = int(np.argmax(probabilities))
        action = ACTIONS[action_id]
        confidence = float(probabilities[action_id])
        decisions.append(action)
        confidences.append(confidence)
        if action == "discard":
            continue
        entry = make_entry(example["haystack_sessions"][index], session_id, index, action)
        assert entry is not None
        density = confidence / max(entry.cost_words, 1)
        candidates.append((density, confidence, -index, entry))
    kept = []
    used = 0
    for _, _, _, entry in sorted(candidates, reverse=True):
        if used + entry.cost_words > budget_words:
            continue
        kept.append(entry)
        used += entry.cost_words
    return kept, decisions, confidences


def split_examples(
    examples: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    indices = list(range(len(examples)))
    labels = [example["question_type"] for example in examples]
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.40,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [labels[index] for index in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )
    return (
        [examples[index] for index in train_idx],
        [examples[index] for index in val_idx],
        [examples[index] for index in test_idx],
    )


def evaluate_methods(
    examples: list[dict],
    budget_frac: float,
    topk: int,
    controller: ControllerBundle,
) -> tuple[dict, dict]:
    metrics_by_method: dict[str, dict] = {}
    artifacts: dict[str, list[dict]] = {}

    def evaluate_builder(name: str, builder_fn):
        recall_scores: list[float] = []
        reciprocal_ranks: list[float] = []
        action_counter: Counter[str] = Counter()
        decision_counter: Counter[str] = Counter()
        per_type_recall: dict[str, list[float]] = defaultdict(list)
        retained_counts: list[int] = []
        rows: list[dict] = []
        for example in examples:
            result = builder_fn(example)
            if isinstance(result, tuple):
                entries, decisions, aux_values = result
            else:
                entries = result
                decisions = ["replay"] * len(example["haystack_sessions"])
                aux_values = []
            retrieved = retrieve_entries(example["question"], entries, topk=topk)
            retained_counts.append(len(entries))
            gold_ids = set(example["answer_session_ids"])
            predicted_ids = [entry.session_id for entry in retrieved]
            hit_positions = [rank for rank, sid in enumerate(predicted_ids, start=1) if sid in gold_ids]
            recall_value = len(set(predicted_ids) & gold_ids) / max(len(gold_ids), 1)
            rr_value = 0.0 if not hit_positions else 1.0 / min(hit_positions)
            recall_scores.append(recall_value)
            reciprocal_ranks.append(rr_value)
            per_type_recall[example["question_type"]].append(recall_value)
            decision_counter.update(decisions)
            action_counter.update(entry.action for entry in entries)
            row = {
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
            if aux_values:
                row["decision_scores"] = aux_values
            rows.append(row)
        metrics_by_method[name] = {
            "recall_at_5": sum(recall_scores) / len(recall_scores),
            "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "avg_retained_entries": statistics.mean(retained_counts),
            "action_usage": dict(action_counter),
            "decision_usage": dict(decision_counter),
            "per_type_recall_at_5": {
                question_type: sum(values) / len(values) for question_type, values in per_type_recall.items()
            },
        }
        artifacts[name] = rows

    evaluate_builder("fifo_replay", lambda example: build_fifo_replay(example, budget_frac))
    evaluate_builder("uniform_replay", lambda example: build_uniform_replay(example, budget_frac))
    evaluate_builder("replay_only_router", lambda example: build_replay_only_router(example, budget_frac))
    evaluate_builder(
        "heuristic_bsc",
        lambda example: (
            build_bsc(example, budget_frac),
            [classify_action(session, index, len(example["haystack_sessions"])) for index, session in enumerate(example["haystack_sessions"])],
            [],
        ),
    )
    evaluate_builder("oracle_bsc", lambda example: build_oracle_bsc(example, budget_frac))
    evaluate_builder("learned_bsc", lambda example: build_learned_bsc(example, budget_frac, controller))
    return metrics_by_method, artifacts


def controller_test_metrics(
    examples: list[dict],
    budget_frac: float,
    controller: ControllerBundle,
) -> dict:
    labels: list[int] = []
    predictions: list[int] = []
    for example in examples:
        for index in range(len(example["haystack_sessions"])):
            oracle = oracle_action_for_session(example, index, budget_frac)
            labels.append(ACTION_TO_ID[oracle.action])
            probs = controller.pipeline.predict_proba(
                np.asarray([feature_vector(example, index, budget_frac)], dtype=np.float32)
            )[0]
            predictions.append(int(np.argmax(probs)))
    return {
        "test_accuracy": accuracy_score(labels, predictions),
        "test_macro_f1": f1_score(labels, predictions, average="macro"),
        "label_distribution": dict(Counter(ACTIONS[label] for label in labels)),
        "prediction_distribution": dict(Counter(ACTIONS[pred] for pred in predictions)),
    }


def plot_results(output_dir: Path, metrics: dict) -> None:
    methods = METHOD_ORDER
    labels = [name.replace("_", "\n") for name in methods]
    x = np.arange(len(methods))
    width = 0.38
    plt.figure(figsize=(10, 4.8))
    recall = [metrics[name]["recall_at_5"] for name in methods]
    mrr = [metrics[name]["mrr_at_5"] for name in methods]
    plt.bar(x - width / 2, recall, width=width, label="Recall@5")
    plt.bar(x + width / 2, mrr, width=width, label="MRR@5")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Held-Out LongMemEval-S Retrieval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learned_controller_metrics.png", dpi=200)
    plt.close()


def write_report(
    output_dir: Path,
    split_seed: int,
    budget_frac: float,
    controller_metrics: list[dict],
    controller_test: dict,
    retrieval_metrics: dict,
) -> None:
    lines = [
        "# Learned Controller Validation",
        "",
        f"- Split seed: `{split_seed}`",
        f"- Budget fraction: `{budget_frac:.0%}`",
        "- Split: `60% train / 20% val / 20% test`, stratified by `question_type`",
        "- Controller: `MLPClassifier(128, 128)` over session and question-conditioned features",
        "- Oracle labels: hindsight action chosen by utility = answer/session usefulness minus memory and compute cost",
        "",
        "## Controller Training",
        "",
    ]
    for row in controller_metrics:
        lines.extend(
            [
                f"### Seed {row['seed']}",
                f"- Train accuracy: `{row['train_accuracy']:.4f}`",
                f"- Val accuracy: `{row['val_accuracy']:.4f}`",
                f"- Train macro-F1: `{row['train_macro_f1']:.4f}`",
                f"- Val macro-F1: `{row['val_macro_f1']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Controller Test",
            "",
            f"- Test accuracy: `{controller_test['test_accuracy']:.4f}`",
            f"- Test macro-F1: `{controller_test['test_macro_f1']:.4f}`",
            f"- Oracle label distribution: `{controller_test['label_distribution']}`",
            f"- Predicted label distribution: `{controller_test['prediction_distribution']}`",
            "",
            "## Retrieval On Held-Out Test Split",
            "",
        ]
    )
    for method in METHOD_ORDER:
        row = retrieval_metrics[method]
        lines.extend(
            [
                f"### {method}",
                f"- Recall@5: `{row['recall_at_5']:.4f}`",
                f"- MRR@5: `{row['mrr_at_5']:.4f}`",
                f"- Avg retained entries: `{row['avg_retained_entries']:.2f}`",
                f"- Decision usage: `{row['decision_usage']}`",
                "",
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-frac", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--controller-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_dataset()
    train_examples, val_examples, test_examples = split_examples(examples, seed=args.split_seed)

    best_controller, controller_metrics = train_controller(
        train_examples=train_examples,
        val_examples=val_examples,
        budget_frac=args.budget_frac,
        seeds=args.controller_seeds,
    )
    controller_test = controller_test_metrics(test_examples, args.budget_frac, best_controller)
    retrieval_metrics, retrieval_rows = evaluate_methods(
        examples=test_examples,
        budget_frac=args.budget_frac,
        topk=args.topk,
        controller=best_controller,
    )

    summary = {
        "budget_frac": args.budget_frac,
        "topk": args.topk,
        "split_seed": args.split_seed,
        "controller_seeds": args.controller_seeds,
        "split_sizes": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "controller_train_val": controller_metrics,
        "controller_test": controller_test,
        "retrieval": retrieval_metrics,
        "best_controller_seed": best_controller.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "retrieval_rows.json").write_text(json.dumps(retrieval_rows, indent=2), encoding="utf-8")
    plot_results(args.output_dir, retrieval_metrics)
    write_report(
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        budget_frac=args.budget_frac,
        controller_metrics=controller_metrics,
        controller_test=controller_test,
        retrieval_metrics=retrieval_metrics,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
