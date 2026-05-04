from __future__ import annotations

import argparse
import json
import math
import statistics
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_memory_validation.bsc_longmemeval import (
    MemoryEntry,
    build_bsc,
    build_replay_only_router,
    count_words,
    exact_match,
    full_budget_words,
    load_dataset,
    make_entry,
    session_features,
    token_f1,
)
from llm_memory_validation.paper_competitor_suite import DenseEmbedder, dense_items_from_entries, dense_rag_retrieve


ACTIONS = ["discard", "replay", "cache", "consolidate"]
ACTION_TO_ID = {action: idx for idx, action in enumerate(ACTIONS)}
POSITIVE_ACTIONS = ["replay", "cache", "consolidate"]
ACTION_COMPUTE_PENALTY = {"replay": 0.08, "cache": 0.03, "consolidate": 0.02}
METHOD_ORDER = [
    "dense_budgeted_replay",
    "heuristic_dense_bsc",
    "counterfactual_oracle_bsc",
    "counterfactual_learned_bsc",
    "dense_rag_e5",
]


@dataclass
class CounterfactualCandidate:
    session_id: str
    session_index: int
    action: str
    text: str
    cost_words: int
    similarity: float


@dataclass
class ExampleContext:
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    gold_session_ids: set[str]
    budget_words: int
    candidates_by_session: dict[int, dict[str, CounterfactualCandidate]]


@dataclass
class ControllerBundle:
    pipeline: Pipeline
    seed: int
    threshold: float
    train_mae: float
    val_mae: float
    train_macro_f1: float
    val_macro_f1: float
    train_accuracy: float
    val_accuracy: float


def split_examples(examples: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
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


def make_question_features(question: str) -> list[float]:
    normalized = question.lower()
    return [
        len(normalized.split()),
        float(any(token in normalized for token in ["today", "tomorrow", "yesterday", "week", "month", "year"])),
        float(any(token in normalized for token in ["change", "updated", "new", "now", "instead"])),
        float(any(token in normalized for token in ["prefer", "favorite", "like", "love", "enjoy"])),
    ]


def build_context(example: dict, budget_frac: float, embedder: DenseEmbedder) -> ExampleContext:
    question = example["question"]
    question_embedding = embedder.encode([question], prefix="query")[0]
    budget_words = max(256, int(full_budget_words(example) * budget_frac))
    candidates_by_session: dict[int, dict[str, CounterfactualCandidate]] = defaultdict(dict)

    all_texts: list[str] = []
    metadata: list[tuple[int, str, str, int]] = []
    for index, (session_id, session) in enumerate(zip(example["haystack_session_ids"], example["haystack_sessions"])):
        for action in ("replay", "cache", "consolidate"):
            entry = make_entry(session, session_id, index, action)
            assert entry is not None
            all_texts.append(entry.text)
            metadata.append((index, action, session_id, entry.cost_words))

    embeddings = embedder.encode(all_texts, prefix="passage")
    similarities = embeddings @ question_embedding
    for (index, action, session_id, cost_words), similarity, text in zip(metadata, similarities, all_texts):
        candidates_by_session[index][action] = CounterfactualCandidate(
            session_id=session_id,
            session_index=index,
            action=action,
            text=text,
            cost_words=cost_words,
            similarity=float(similarity),
        )
    return ExampleContext(
        question_id=example["question_id"],
        question_type=example["question_type"],
        question=question,
        gold_answer=str(example["answer"]),
        gold_session_ids=set(example["answer_session_ids"]),
        budget_words=budget_words,
        candidates_by_session=candidates_by_session,
    )


def objective_for_candidates(selected: list[CounterfactualCandidate], context: ExampleContext, topk: int) -> tuple[float, dict]:
    if not selected:
        return 0.0, {"recall": 0.0, "mrr": 0.0, "answer_support": 0.0}
    ranked = sorted(selected, key=lambda item: item.similarity, reverse=True)[:topk]
    predicted_ids = [item.session_id for item in ranked]
    hit_positions = [rank for rank, session_id in enumerate(predicted_ids, start=1) if session_id in context.gold_session_ids]
    recall = len(set(predicted_ids) & context.gold_session_ids) / max(len(context.gold_session_ids), 1)
    mrr = 0.0 if not hit_positions else 1.0 / min(hit_positions)
    combined_text = "\n".join(item.text for item in ranked)
    answer_support = token_f1(combined_text, context.gold_answer)
    score = 2.6 * recall + 1.1 * mrr + 1.0 * answer_support
    return score, {"recall": recall, "mrr": mrr, "answer_support": answer_support}


def candidate_gain(
    selected: list[CounterfactualCandidate],
    context: ExampleContext,
    candidate: CounterfactualCandidate,
    topk: int,
    used_words: int = 0,
) -> float:
    if used_words + candidate.cost_words > context.budget_words:
        return float("-inf")
    current_score, _ = objective_for_candidates(selected, context, topk)
    new_score, _ = objective_for_candidates(selected + [candidate], context, topk)
    mem_penalty = 0.25 * (candidate.cost_words / max(context.budget_words, 1))
    compute_penalty = ACTION_COMPUTE_PENALTY[candidate.action]
    return new_score - current_score - mem_penalty - compute_penalty


def counterfactual_oracle_select(context: ExampleContext, topk: int) -> tuple[list[CounterfactualCandidate], list[str], list[float]]:
    selected: list[CounterfactualCandidate] = []
    chosen_sessions: set[int] = set()
    decisions = ["discard"] * len(context.candidates_by_session)
    gains = [0.0] * len(context.candidates_by_session)
    used_words = 0

    while True:
        best_gain = 0.0
        best_candidate: CounterfactualCandidate | None = None
        best_session: int | None = None
        remaining = sorted(set(context.candidates_by_session.keys()) - chosen_sessions)
        for session_index in remaining:
            for action, candidate in context.candidates_by_session[session_index].items():
                gain = candidate_gain(selected, context, candidate, topk, used_words=used_words)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
                    best_session = session_index
        if best_candidate is None:
            break
        selected.append(best_candidate)
        chosen_sessions.add(best_session)
        decisions[best_session] = best_candidate.action
        gains[best_session] = best_gain
        used_words += best_candidate.cost_words
    return selected, decisions, gains


def action_utilities_for_session(context: ExampleContext, session_index: int, topk: int) -> np.ndarray:
    utilities = []
    for action in POSITIVE_ACTIONS:
        candidate = context.candidates_by_session[session_index][action]
        gain = candidate_gain([], context, candidate, topk)
        utilities.append(gain if math.isfinite(gain) else -1.0)
    return np.asarray(utilities, dtype=np.float32)


def feature_vector(example: dict, context: ExampleContext, session_index: int) -> list[float]:
    session = example["haystack_sessions"][session_index]
    total = len(example["haystack_sessions"])
    feat = session_features(session, session_index, total)
    qfeat = make_question_features(example["question"])
    replay_cand = context.candidates_by_session[session_index]["replay"]
    cache_cand = context.candidates_by_session[session_index]["cache"]
    consolidate_cand = context.candidates_by_session[session_index]["consolidate"]
    return [
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
        *qfeat,
        replay_cand.similarity,
        cache_cand.similarity,
        consolidate_cand.similarity,
        replay_cand.cost_words / context.budget_words,
        cache_cand.cost_words / context.budget_words,
        consolidate_cand.cost_words / context.budget_words,
    ]


def oversample_keep_rows(features: np.ndarray, utilities: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep_mask = np.max(utilities, axis=1) > 0.0
    keep_indices = np.where(keep_mask)[0]
    discard_indices = np.where(~keep_mask)[0]
    if len(keep_indices) == 0 or len(discard_indices) == 0:
        return features, utilities
    target = max(len(keep_indices), len(discard_indices))
    chosen_indices: list[int] = discard_indices.tolist()
    if len(discard_indices) < target:
        chosen_indices.extend(rng.choice(discard_indices, size=target - len(discard_indices), replace=True).tolist())
    chosen_indices.extend(keep_indices.tolist())
    if len(keep_indices) < target:
        chosen_indices.extend(rng.choice(keep_indices, size=target - len(keep_indices), replace=True).tolist())
    rng.shuffle(chosen_indices)
    return features[chosen_indices], utilities[chosen_indices]


def decisions_from_utilities(action_utilities: np.ndarray, threshold: float) -> np.ndarray:
    best_action_ids = np.argmax(action_utilities, axis=1)
    best_scores = np.max(action_utilities, axis=1)
    decisions = np.zeros(len(action_utilities), dtype=np.int64)
    keep_mask = best_scores > threshold
    decisions[keep_mask] = best_action_ids[keep_mask] + 1
    return decisions


def build_training_rows(
    examples: list[dict],
    contexts: dict[str, ExampleContext],
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    utility_targets: list[np.ndarray] = []
    oracle_labels: list[int] = []
    for example in examples:
        context = contexts[example["question_id"]]
        _, decisions, _ = counterfactual_oracle_select(context, topk)
        for session_index in range(len(example["haystack_sessions"])):
            features.append(feature_vector(example, context, session_index))
            utility_targets.append(action_utilities_for_session(context, session_index, topk))
            oracle_labels.append(ACTION_TO_ID[decisions[session_index]])
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(utility_targets, dtype=np.float32),
        np.asarray(oracle_labels, dtype=np.int64),
    )


def train_controller(
    train_examples: list[dict],
    val_examples: list[dict],
    contexts: dict[str, ExampleContext],
    topk: int,
    seeds: list[int],
) -> tuple[ControllerBundle, list[dict]]:
    train_x, train_y, train_oracle = build_training_rows(train_examples, contexts, topk)
    val_x, val_y, val_oracle = build_training_rows(val_examples, contexts, topk)
    bundles: list[ControllerBundle] = []
    metrics: list[dict] = []
    for seed in seeds:
        sampled_x, sampled_y = oversample_keep_rows(train_x, train_y, seed)
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        batch_size=256,
                        max_iter=250,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                    ),
                ),
            ]
        )
        pipeline.fit(sampled_x, sampled_y)
        train_pred_util = np.asarray(pipeline.predict(train_x), dtype=np.float32)
        val_pred_util = np.asarray(pipeline.predict(val_x), dtype=np.float32)
        candidate_thresholds = sorted(
            {
                -0.05,
                0.0,
                0.01,
                0.02,
                0.03,
                0.05,
                *np.quantile(np.max(val_pred_util, axis=1), [0.1, 0.25, 0.5, 0.75]).tolist(),
            }
        )
        best_threshold = 0.0
        best_val_macro_f1 = -1.0
        best_val_accuracy = -1.0
        for threshold in candidate_thresholds:
            val_pred = decisions_from_utilities(val_pred_util, float(threshold))
            val_macro_f1 = f1_score(val_oracle, val_pred, average="macro")
            val_accuracy = accuracy_score(val_oracle, val_pred)
            if (val_macro_f1, val_accuracy) > (best_val_macro_f1, best_val_accuracy):
                best_threshold = float(threshold)
                best_val_macro_f1 = val_macro_f1
                best_val_accuracy = val_accuracy
        train_pred = decisions_from_utilities(train_pred_util, best_threshold)
        val_pred = decisions_from_utilities(val_pred_util, best_threshold)
        bundle = ControllerBundle(
            pipeline=pipeline,
            seed=seed,
            threshold=best_threshold,
            train_mae=mean_absolute_error(train_y, train_pred_util),
            val_mae=mean_absolute_error(val_y, val_pred_util),
            train_macro_f1=f1_score(train_oracle, train_pred, average="macro"),
            val_macro_f1=f1_score(val_oracle, val_pred, average="macro"),
            train_accuracy=accuracy_score(train_oracle, train_pred),
            val_accuracy=accuracy_score(val_oracle, val_pred),
        )
        bundles.append(bundle)
        metrics.append(
            {
                "seed": seed,
                "threshold": bundle.threshold,
                "train_mae": bundle.train_mae,
                "val_mae": bundle.val_mae,
                "train_accuracy": bundle.train_accuracy,
                "val_accuracy": bundle.val_accuracy,
                "train_macro_f1": bundle.train_macro_f1,
                "val_macro_f1": bundle.val_macro_f1,
            }
        )
    best = max(bundles, key=lambda bundle: (bundle.val_macro_f1, bundle.val_accuracy))
    return best, metrics


def build_learned_selection(
    example: dict,
    context: ExampleContext,
    controller: ControllerBundle,
) -> tuple[list[CounterfactualCandidate], list[str], list[float]]:
    selected: list[CounterfactualCandidate] = []
    decisions = []
    confidences = []
    used_words = 0
    candidates = []
    for session_index in range(len(example["haystack_sessions"])):
        features = np.asarray([feature_vector(example, context, session_index)], dtype=np.float32)
        utilities = np.asarray(controller.pipeline.predict(features)[0], dtype=np.float32)
        positive_id = int(np.argmax(utilities))
        confidence = float(utilities[positive_id])
        action = POSITIVE_ACTIONS[positive_id]
        if confidence <= controller.threshold:
            action = "discard"
        decisions.append(action)
        confidences.append(confidence)
        if action == "discard":
            continue
        candidate = context.candidates_by_session[session_index][action]
        density = (confidence - controller.threshold) / max(candidate.cost_words, 1)
        candidates.append((density, confidence, -session_index, candidate))
    for _, _, _, candidate in sorted(candidates, reverse=True):
        if used_words + candidate.cost_words > context.budget_words:
            continue
        selected.append(candidate)
        used_words += candidate.cost_words
    return selected, decisions, confidences


def dense_predict_ids_from_candidates(context: ExampleContext, candidates: list[CounterfactualCandidate], topk: int) -> list[str]:
    ranked = sorted(candidates, key=lambda item: item.similarity, reverse=True)[:topk]
    return [item.session_id for item in ranked]


def prompt_from_dense_candidates(question: str, candidates: list[CounterfactualCandidate], topk: int, prompt_word_budget: int) -> str:
    ranked = sorted(candidates, key=lambda item: item.similarity, reverse=True)[:topk]
    blocks = []
    used = 0
    for rank, candidate in enumerate(ranked, start=1):
        words = candidate.text.split()
        clipped = " ".join(words[: min(len(words), 250)])
        block = f"[{rank}] action={candidate.action} session={candidate.session_id}\n{clipped}"
        block_cost = count_words(block)
        if blocks and used + block_cost > prompt_word_budget:
            break
        blocks.append(block)
        used += block_cost
    memory_text = "\n\n".join(blocks) if blocks else "[no memory]"
    return textwrap.dedent(
        f"""
        You answer a user question using retrieved long-term memory.
        Use only the memory below.
        Reply with a short direct answer and no explanation.
        If the answer is not supported, reply with "unknown".

        Question:
        {question}

        Memory:
        {memory_text}

        Answer:
        """
    ).strip()


def evaluate_retrieval(
    examples: list[dict],
    contexts: dict[str, ExampleContext],
    controller: ControllerBundle,
    dense_embedder: DenseEmbedder,
    topk: int,
) -> tuple[dict, dict, dict]:
    metrics: dict[str, dict] = {}
    rows_by_method: dict[str, list[dict]] = {}
    candidate_store: dict[str, dict[str, list[CounterfactualCandidate]]] = defaultdict(dict)

    def finalize(method: str, predicted_ids_by_example: list[list[str]], decision_usage: Counter[str] | None = None):
        recalls = []
        reciprocal_ranks = []
        per_type = defaultdict(list)
        rows = []
        for example, predicted_ids in zip(examples, predicted_ids_by_example):
            gold = set(example["answer_session_ids"])
            hits = [rank for rank, sid in enumerate(predicted_ids, start=1) if sid in gold]
            recall = len(set(predicted_ids) & gold) / max(len(gold), 1)
            rr = 0.0 if not hits else 1.0 / min(hits)
            recalls.append(recall)
            reciprocal_ranks.append(rr)
            per_type[example["question_type"]].append(recall)
            rows.append(
                {
                    "question_id": example["question_id"],
                    "question_type": example["question_type"],
                    "gold_session_ids": example["answer_session_ids"],
                    "predicted_session_ids": predicted_ids,
                }
            )
        metrics[method] = {
            "recall_at_5": float(sum(recalls) / len(recalls)),
            "mrr_at_5": float(sum(reciprocal_ranks) / len(reciprocal_ranks)),
            "per_type_recall_at_5": {
                question_type: float(sum(values) / len(values)) for question_type, values in per_type.items()
            },
        }
        if decision_usage is not None:
            metrics[method]["decision_usage"] = dict(decision_usage)
        rows_by_method[method] = rows

    replay_preds = []
    heuristic_preds = []
    oracle_preds = []
    learned_preds = []
    rag_preds = []
    oracle_usage = Counter()
    learned_usage = Counter()
    for example in examples:
        context = contexts[example["question_id"]]
        replay_entries = build_replay_only_router(example, 0.20)
        dense_replay = dense_items_from_entries(example, replay_entries, dense_embedder, topk)
        replay_preds.append([item.session_id for item in dense_replay])
        candidate_store[example["question_id"]]["dense_budgeted_replay"] = [
            context.candidates_by_session[entry.session_index]["replay"] for entry in replay_entries
        ]

        heuristic_entries = build_bsc(example, 0.20)
        dense_heuristic = dense_items_from_entries(example, heuristic_entries, dense_embedder, topk)
        heuristic_preds.append([item.session_id for item in dense_heuristic])
        heuristic_candidates = [context.candidates_by_session[entry.session_index][entry.action] for entry in heuristic_entries]
        candidate_store[example["question_id"]]["heuristic_dense_bsc"] = heuristic_candidates

        oracle_candidates, oracle_decisions, _ = counterfactual_oracle_select(context, topk)
        oracle_usage.update(oracle_decisions)
        oracle_preds.append(dense_predict_ids_from_candidates(context, oracle_candidates, topk))
        candidate_store[example["question_id"]]["counterfactual_oracle_bsc"] = oracle_candidates

        learned_candidates, learned_decisions, _ = build_learned_selection(example, context, controller)
        learned_usage.update(learned_decisions)
        learned_preds.append(dense_predict_ids_from_candidates(context, learned_candidates, topk))
        candidate_store[example["question_id"]]["counterfactual_learned_bsc"] = learned_candidates

        rag_items = dense_rag_retrieve(example, dense_embedder, topk)
        rag_preds.append([item.session_id for item in rag_items])
        candidate_store[example["question_id"]]["dense_rag_e5"] = [
            CounterfactualCandidate(
                session_id=item.session_id,
                session_index=-1,
                action="replay",
                text=item.text,
                cost_words=count_words(item.text),
                similarity=item.score,
            )
            for item in rag_items
        ]

    finalize("dense_budgeted_replay", replay_preds)
    finalize("heuristic_dense_bsc", heuristic_preds)
    finalize("counterfactual_oracle_bsc", oracle_preds, oracle_usage)
    finalize("counterfactual_learned_bsc", learned_preds, learned_usage)
    finalize("dense_rag_e5", rag_preds)
    return metrics, rows_by_method, candidate_store


def evaluate_controller_test(
    examples: list[dict],
    contexts: dict[str, ExampleContext],
    topk: int,
    controller: ControllerBundle,
) -> dict:
    labels = []
    preds = []
    for example in examples:
        context = contexts[example["question_id"]]
        _, decisions, _ = counterfactual_oracle_select(context, topk)
        for session_index in range(len(example["haystack_sessions"])):
            labels.append(ACTION_TO_ID[decisions[session_index]])
            features = np.asarray([feature_vector(example, context, session_index)], dtype=np.float32)
            utilities = np.asarray(controller.pipeline.predict(features)[0], dtype=np.float32)
            pred = int(decisions_from_utilities(utilities.reshape(1, -1), controller.threshold)[0])
            preds.append(pred)
    return {
        "test_accuracy": accuracy_score(labels, preds),
        "test_macro_f1": f1_score(labels, preds, average="macro"),
        "label_distribution": dict(Counter(ACTIONS[label] for label in labels)),
        "prediction_distribution": dict(Counter(ACTIONS[pred] for pred in preds)),
    }


def run_generation(
    examples: list[dict],
    candidate_store: dict[str, dict[str, list[CounterfactualCandidate]]],
    reader_model: str,
    methods: list[str],
    topk: int,
    prompt_word_budget: int,
    max_new_tokens: int,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(reader_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        reader_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    generation_metrics: dict[str, dict] = {}
    predictions_by_method: dict[str, list[dict]] = {}
    for method in methods:
        em_scores = []
        f1_scores = []
        per_type_em = defaultdict(list)
        per_type_f1 = defaultdict(list)
        predictions = []
        for example in examples:
            candidates = candidate_store[example["question_id"]][method]
            prompt = prompt_from_dense_candidates(
                question=example["question"],
                candidates=candidates,
                topk=topk,
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
            prediction = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip().split("\n")[0].strip()
            gold = str(example["answer"])
            em = exact_match(prediction, gold)
            f1 = token_f1(prediction, gold)
            em_scores.append(em)
            f1_scores.append(f1)
            per_type_em[example["question_type"]].append(em)
            per_type_f1[example["question_type"]].append(f1)
            predictions.append(
                {
                    "question_id": example["question_id"],
                    "question_type": example["question_type"],
                    "gold_answer": gold,
                    "prediction": prediction,
                    "exact_match": em,
                    "token_f1": f1,
                }
            )
        generation_metrics[method] = {
            "exact_match": float(sum(em_scores) / len(em_scores)),
            "token_f1": float(sum(f1_scores) / len(f1_scores)),
            "per_type_exact_match": {
                question_type: float(sum(values) / len(values)) for question_type, values in per_type_em.items()
            },
            "per_type_token_f1": {
                question_type: float(sum(values) / len(values)) for question_type, values in per_type_f1.items()
            },
            "model_name": reader_model,
        }
        predictions_by_method[method] = predictions
    return {"metrics": generation_metrics, "predictions": predictions_by_method}


def plot_metrics(output_dir: Path, retrieval_metrics: dict, generation_metrics: dict) -> None:
    methods = METHOD_ORDER
    labels = [name.replace("_", "\n") for name in methods]
    x = np.arange(len(methods))
    width = 0.38

    plt.figure(figsize=(11, 4.8))
    recall = [retrieval_metrics[method]["recall_at_5"] for method in methods]
    mrr = [retrieval_metrics[method]["mrr_at_5"] for method in methods]
    plt.bar(x - width / 2, recall, width=width, label="Recall@5")
    plt.bar(x + width / 2, mrr, width=width, label="MRR@5")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Counterfactual Dense Retrieval Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_metrics.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    em = [generation_metrics[method]["exact_match"] for method in methods]
    f1 = [generation_metrics[method]["token_f1"] for method in methods]
    plt.bar(x - width / 2, em, width=width, label="Exact Match")
    plt.bar(x + width / 2, f1, width=width, label="Token F1")
    plt.xticks(x, labels)
    plt.ylim(0.0, max(max(f1), max(em), 0.05) * 1.25)
    plt.ylabel("Score")
    plt.title("End-to-End Answer Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "generation_metrics.png", dpi=200)
    plt.close()


def write_report(
    output_dir: Path,
    split_sizes: dict,
    budget_frac: float,
    controller_train_val: list[dict],
    controller_test: dict,
    retrieval_metrics: dict,
    generation_metrics: dict,
) -> None:
    lines = [
        "# Counterfactual Dense BSC",
        "",
        f"- Split sizes: `{split_sizes}`",
        f"- Budget fraction: `{budget_frac:.0%}`",
        "- Oracle: greedy counterfactual selection using dense retrieval + answer-support objective",
        "- Controller: `MLPRegressor(128, 128)` trained on dense per-action counterfactual utilities",
        "- Inference: discard if all predicted action utilities are below the validation-selected threshold",
        "",
        "## Controller",
        "",
    ]
    for row in controller_train_val:
        lines.extend(
            [
                f"### Seed {row['seed']}",
                f"- Threshold: `{row['threshold']:.4f}`",
                f"- Train MAE: `{row['train_mae']:.4f}`",
                f"- Val MAE: `{row['val_mae']:.4f}`",
                f"- Train accuracy: `{row['train_accuracy']:.4f}`",
                f"- Val accuracy: `{row['val_accuracy']:.4f}`",
                f"- Train macro-F1: `{row['train_macro_f1']:.4f}`",
                f"- Val macro-F1: `{row['val_macro_f1']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            f"- Test accuracy: `{controller_test['test_accuracy']:.4f}`",
            f"- Test macro-F1: `{controller_test['test_macro_f1']:.4f}`",
            f"- Oracle label distribution: `{controller_test['label_distribution']}`",
            f"- Predicted label distribution: `{controller_test['prediction_distribution']}`",
            "",
            "## Retrieval",
            "",
        ]
    )
    for method in METHOD_ORDER:
        metrics = retrieval_metrics[method]
        lines.extend(
            [
                f"### {method}",
                f"- Recall@5: `{metrics['recall_at_5']:.4f}`",
                f"- MRR@5: `{metrics['mrr_at_5']:.4f}`",
                "",
            ]
        )
    lines.extend(["## Generation", ""])
    for method in METHOD_ORDER:
        metrics = generation_metrics[method]
        lines.extend(
            [
                f"### {method}",
                f"- Exact Match: `{metrics['exact_match']:.4f}`",
                f"- Token F1: `{metrics['token_f1']:.4f}`",
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
    parser.add_argument("--retriever-model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--reader-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt-word-budget", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_dataset()
    train_examples, val_examples, test_examples = split_examples(examples, seed=args.split_seed)

    embedder = DenseEmbedder(model_name=args.retriever_model)
    contexts = {example["question_id"]: build_context(example, args.budget_frac, embedder) for example in examples}

    best_controller, controller_train_val = train_controller(
        train_examples=train_examples,
        val_examples=val_examples,
        contexts=contexts,
        topk=args.topk,
        seeds=args.controller_seeds,
    )
    controller_test = evaluate_controller_test(
        examples=test_examples,
        contexts=contexts,
        topk=args.topk,
        controller=best_controller,
    )
    retrieval_metrics, retrieval_rows, candidate_store = evaluate_retrieval(
        examples=test_examples,
        contexts=contexts,
        controller=best_controller,
        dense_embedder=embedder,
        topk=args.topk,
    )

    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generation_payload = run_generation(
        examples=test_examples,
        candidate_store=candidate_store,
        reader_model=args.reader_model,
        methods=METHOD_ORDER,
        topk=args.topk,
        prompt_word_budget=args.prompt_word_budget,
        max_new_tokens=args.max_new_tokens,
    )
    generation_metrics = generation_payload["metrics"]

    summary = {
        "budget_frac": args.budget_frac,
        "topk": args.topk,
        "split_seed": args.split_seed,
        "controller_seeds": args.controller_seeds,
        "retriever_model": args.retriever_model,
        "reader_model": args.reader_model,
        "split_sizes": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "controller_train_val": controller_train_val,
        "controller_test": controller_test,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "best_controller_seed": best_controller.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "retrieval_rows.json").write_text(json.dumps(retrieval_rows, indent=2), encoding="utf-8")
    (args.output_dir / "generation_predictions.json").write_text(json.dumps(generation_payload["predictions"], indent=2), encoding="utf-8")
    plot_metrics(args.output_dir, retrieval_metrics, generation_metrics)
    write_report(
        output_dir=args.output_dir,
        split_sizes=summary["split_sizes"],
        budget_frac=args.budget_frac,
        controller_train_val=controller_train_val,
        controller_test=controller_test,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
