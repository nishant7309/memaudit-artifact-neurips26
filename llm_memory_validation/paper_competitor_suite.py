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
import torch
from transformers import AutoModel, AutoTokenizer

from llm_memory_validation.bsc_longmemeval import (
    build_bsc,
    build_fifo_replay,
    build_replay_only_router,
    build_uniform_replay,
    count_words,
    extract_fact_lines,
    load_dataset,
    normalize_answer,
    retrieve_entries,
    session_text,
    tail_snippet,
)


REPORTED_BASELINES = {
    "RAG_GTE_paper": 0.624,
    "RMM_GTE_paper": 0.698,
}

METHOD_ORDER = [
    "fifo_replay",
    "uniform_replay",
    "replay_only_router",
    "dense_budgeted_replay",
    "dense_rag_e5",
    "memorybank_proxy",
    "ld_agent_proxy",
    "heuristic_bsc",
    "dense_budgeted_bsc",
]

METHOD_DESCRIPTIONS = {
    "fifo_replay": "Newest raw sessions until storage fills.",
    "uniform_replay": "Evenly spaced raw sessions.",
    "replay_only_router": "Heuristic raw-session prioritization only.",
    "dense_budgeted_replay": "Same budgeted replay-only store, but retrieved with dense E5 embeddings.",
    "dense_rag_e5": "Full raw-store dense retrieval over all sessions using E5 embeddings.",
    "memorybank_proxy": "Fact summaries with forgetting-curve style recency weighting.",
    "ld_agent_proxy": "Short-term recent bank plus long-term persona/event summaries.",
    "heuristic_bsc": "OracleMem writer store retrieved with the lexical baseline retriever.",
    "dense_budgeted_bsc": "OracleMem writer store retrieved with the same fixed dense E5 top-k retriever.",
}

METHOD_LABELS = {
    "fifo_replay": "FIFO raw replay",
    "uniform_replay": "Uniform raw replay",
    "replay_only_router": "Budgeted raw replay router",
    "dense_budgeted_replay": "Budgeted raw replay + dense retrieval",
    "dense_rag_e5": "Full raw-store dense retrieval",
    "memorybank_proxy": "MemoryBank proxy",
    "ld_agent_proxy": "LD-Agent proxy",
    "heuristic_bsc": "OracleMem writer + lexical retrieval",
    "dense_budgeted_bsc": "OracleMem writer + dense retrieval",
}


@dataclass
class DenseItem:
    session_id: str
    text: str
    short_text: str
    score: float


class DenseEmbedder:
    def __init__(self, model_name: str = "intfloat/e5-base-v2", batch_size: int = 16, max_length: int = 256) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [f"{prefix}: {text}" for text in texts[start:start + self.batch_size]]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


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


def dense_rag_retrieve(example: dict, embedder: DenseEmbedder, topk: int) -> list[DenseItem]:
    session_texts = [session_text(session) for session in example["haystack_sessions"]]
    query_embedding = embedder.encode([example["question"]], prefix="query")[0]
    doc_embeddings = embedder.encode(session_texts, prefix="passage")
    similarities = doc_embeddings @ query_embedding
    ranked_indices = np.argsort(-similarities)[:topk]
    return [
        DenseItem(
            session_id=example["haystack_session_ids"][index],
            text=session_texts[index],
            short_text=tail_snippet(example["haystack_sessions"][index], turns=3),
            score=float(similarities[index]),
        )
        for index in ranked_indices
    ]


def dense_items_from_entries(example: dict, entries, embedder: DenseEmbedder, topk: int) -> list[DenseItem]:
    if not entries:
        return []
    texts = [entry.text for entry in entries]
    query_embedding = embedder.encode([example["question"]], prefix="query")[0]
    doc_embeddings = embedder.encode(texts, prefix="passage")
    similarities = doc_embeddings @ query_embedding
    ranked_indices = np.argsort(-similarities)[:topk]
    return [
        DenseItem(
            session_id=entries[index].session_id,
            text=entries[index].text,
            short_text=entries[index].text,
            score=float(similarities[index]),
        )
        for index in ranked_indices
    ]


def memorybank_retrieve(example: dict, embedder: DenseEmbedder, topk: int) -> list[DenseItem]:
    summaries = [summarize_session_for_memorybank(session) for session in example["haystack_sessions"]]
    query_embedding = embedder.encode([example["question"]], prefix="query")[0]
    memory_embeddings = embedder.encode(summaries, prefix="passage")
    total = len(summaries)
    scores = []
    for index, summary in enumerate(summaries):
        sim = float(memory_embeddings[index] @ query_embedding)
        age = total - 1 - index
        forgetting = math.exp(-0.045 * age)
        scores.append(sim + 0.25 * forgetting)
    ranked_indices = np.argsort(-np.asarray(scores))[:topk]
    return [
        DenseItem(
            session_id=example["haystack_session_ids"][index],
            text=summaries[index],
            short_text=summaries[index],
            score=float(scores[index]),
        )
        for index in ranked_indices
    ]


def ld_agent_retrieve(example: dict, embedder: DenseEmbedder, topk: int) -> list[DenseItem]:
    total = len(example["haystack_sessions"])
    short_cutoff = max(total - 6, 0)
    short_sessions = example["haystack_sessions"][short_cutoff:]
    short_ids = example["haystack_session_ids"][short_cutoff:]
    long_sessions = example["haystack_sessions"][:short_cutoff]
    long_ids = example["haystack_session_ids"][:short_cutoff]

    selected: list[DenseItem] = []
    query_embedding = embedder.encode([example["question"]], prefix="query")[0]

    if short_sessions:
        short_texts = [tail_snippet(session, turns=4) for session in short_sessions]
        short_embeddings = embedder.encode(short_texts, prefix="passage")
        scores = []
        for index, text in enumerate(short_texts):
            sim = float(short_embeddings[index] @ query_embedding)
            recency = 1.0 - (len(short_texts) - 1 - index) / max(len(short_texts), 1)
            scores.append(sim + 0.20 * recency)
        ranked_short = np.argsort(-np.asarray(scores))[: min(2, len(scores))]
        selected.extend(
            DenseItem(
                session_id=short_ids[index],
                text=short_texts[index],
                short_text=short_texts[index],
                score=float(scores[index]),
            )
            for index in ranked_short
        )

    if long_sessions:
        long_texts = [summarize_session_for_ld_long(session) for session in long_sessions]
        long_embeddings = embedder.encode(long_texts, prefix="passage")
        scores = []
        for index, text in enumerate(long_texts):
            sim = float(long_embeddings[index] @ query_embedding)
            persona_bonus = 0.08 if "persona:" in text else 0.0
            scores.append(sim + persona_bonus)
        ranked_long = np.argsort(-np.asarray(scores))[: max(topk - len(selected), 0)]
        selected.extend(
            DenseItem(
                session_id=long_ids[index],
                text=long_texts[index],
                short_text=long_texts[index],
                score=float(scores[index]),
            )
            for index in ranked_long
        )

    deduped: list[DenseItem] = []
    seen = set()
    for item in selected:
        if item.session_id in seen:
            continue
        deduped.append(item)
        seen.add(item.session_id)
        if len(deduped) >= topk:
            break
    return deduped


def evaluate_retrieval(examples: list[dict], embedder: DenseEmbedder, topk: int) -> tuple[dict, dict]:
    metrics_by_method: dict[str, dict] = {}
    rows_by_method: dict[str, list[dict]] = {}

    def score_predictions(method: str, predicted_ids_by_example: list[list[str]], action_usage: dict | None = None) -> None:
        recalls = []
        reciprocal_ranks = []
        per_type = defaultdict(list)
        rows = []
        for example, predicted_ids in zip(examples, predicted_ids_by_example):
            gold_ids = set(example["answer_session_ids"])
            hit_positions = [rank for rank, sid in enumerate(predicted_ids, start=1) if sid in gold_ids]
            recall = len(set(predicted_ids) & gold_ids) / max(len(gold_ids), 1)
            rr = 0.0 if not hit_positions else 1.0 / min(hit_positions)
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
        metrics_by_method[method] = {
            "recall_at_5": float(sum(recalls) / len(recalls)),
            "mrr_at_5": float(sum(reciprocal_ranks) / len(reciprocal_ranks)),
            "per_type_recall_at_5": {
                question_type: float(sum(values) / len(values)) for question_type, values in per_type.items()
            },
        }
        if action_usage is not None:
            metrics_by_method[method]["action_usage"] = action_usage
        rows_by_method[method] = rows

    score_predictions(
        "fifo_replay",
        [
            [entry.session_id for entry in retrieve_entries(example["question"], build_fifo_replay(example, 0.20), topk)]
            for example in examples
        ],
    )
    score_predictions(
        "uniform_replay",
        [
            [entry.session_id for entry in retrieve_entries(example["question"], build_uniform_replay(example, 0.20), topk)]
            for example in examples
        ],
    )
    score_predictions(
        "replay_only_router",
        [
            [entry.session_id for entry in retrieve_entries(example["question"], build_replay_only_router(example, 0.20), topk)]
            for example in examples
        ],
    )
    score_predictions(
        "dense_budgeted_replay",
        [
            [item.session_id for item in dense_items_from_entries(example, build_replay_only_router(example, 0.20), embedder, topk)]
            for example in examples
        ],
    )
    score_predictions(
        "heuristic_bsc",
        [
            [entry.session_id for entry in retrieve_entries(example["question"], build_bsc(example, 0.20), topk)]
            for example in examples
        ],
        action_usage=dict(
            Counter(
                action
                for example in examples
                for action in [entry.action for entry in build_bsc(example, 0.20)]
            )
        ),
    )
    score_predictions(
        "dense_rag_e5",
        [[item.session_id for item in dense_rag_retrieve(example, embedder, topk)] for example in examples],
    )
    score_predictions(
        "memorybank_proxy",
        [[item.session_id for item in memorybank_retrieve(example, embedder, topk)] for example in examples],
    )
    score_predictions(
        "ld_agent_proxy",
        [[item.session_id for item in ld_agent_retrieve(example, embedder, topk)] for example in examples],
    )
    score_predictions(
        "dense_budgeted_bsc",
        [
            [item.session_id for item in dense_items_from_entries(example, build_bsc(example, 0.20), embedder, topk)]
            for example in examples
        ],
    )
    return metrics_by_method, rows_by_method


def plot_results(output_dir: Path, metrics: dict) -> None:
    methods = METHOD_ORDER
    labels = [name.replace("_", "\n") for name in methods]
    x = np.arange(len(methods))
    width = 0.38
    plt.figure(figsize=(11, 5))
    recall = [metrics[name]["recall_at_5"] for name in methods]
    mrr = [metrics[name]["mrr_at_5"] for name in methods]
    plt.bar(x - width / 2, recall, width=width, label="Recall@5")
    plt.bar(x + width / 2, mrr, width=width, label="MRR@5")
    for label, value in REPORTED_BASELINES.items():
        plt.axhline(value, linestyle="--", linewidth=1.2, label=f"{label} ({value:.3f})")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("LongMemEval-S Competitor Suite")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "competitor_suite_metrics.png", dpi=200)
    plt.close()


def write_report(output_dir: Path, model_name: str, metrics: dict) -> None:
    lines = [
        "# Competitor Suite",
        "",
        "- Benchmark: `LongMemEval-S` full 500-example evaluation",
        "- Metric: `Recall@5` and `MRR@5` against gold `answer_session_ids`",
        f"- Dense retriever: `{model_name}`",
        "- Published paper references: `RAG_GTE_paper=0.624`, `RMM_GTE_paper=0.698` Recall@5",
        "",
    ]
    for method in METHOD_ORDER:
        row = metrics[method]
        label = METHOD_LABELS.get(method, method)
        lines.extend(
            [
                f"## {label}",
                f"- Artifact key: `{method}`",
                f"- Description: {METHOD_DESCRIPTIONS[method]}",
                f"- Recall@5: `{row['recall_at_5']:.4f}`",
                f"- MRR@5: `{row['mrr_at_5']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- The published RMM numbers are external paper references, not a local reproduction.",
            "- This suite is strongest as a retrieval comparison. It does not yet reproduce end-to-end answer accuracy with the same reader used in RMM.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--retriever-model", type=str, default="intfloat/e5-base-v2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_dataset()
    embedder = DenseEmbedder(model_name=args.retriever_model)
    metrics, rows = evaluate_retrieval(examples, embedder, topk=args.topk)
    summary = {
        "retriever_model": args.retriever_model,
        "topk": args.topk,
        "reported_baselines": REPORTED_BASELINES,
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "retrieval_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_results(args.output_dir, metrics)
    write_report(args.output_dir, args.retriever_model, metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
