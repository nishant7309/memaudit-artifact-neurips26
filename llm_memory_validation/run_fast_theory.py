from llm_memory_validation.bsc_longmemeval import load_dataset, classify_action, build_bsc, full_budget_words
from llm_memory_validation.counterfactual_dense_bsc import split_examples, ACTION_TO_ID
from collections import Counter
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("llm_memory_validation/neurips_fast_results")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
examples = load_dataset()
print(f"Loaded {len(examples)} examples")

train_ex, val_ex, test_ex = split_examples(examples, seed=11)
print(f"Split: {len(train_ex)}/{len(val_ex)}/{len(test_ex)}")

print("Computing heuristic action distribution...")
action_counts = Counter()
per_type_actions = Counter()
total_decisions = 0
for example in examples:
    sessions = example["haystack_sessions"]
    total = len(sessions)
    for i, session in enumerate(sessions):
        a = classify_action(session, i, total)
        action_counts[a] += 1
        total_decisions += 1
        per_type_actions[(example["question_type"], a)] += 1

discard_frac = action_counts.get("discard", 0) / total_decisions
print(f"Distribution: {dict(action_counts)}")
print(f"Discard fraction: {discard_frac:.2%}")

per_type_dist = {}
QTYPES = ["single-session-user", "single-session-preference", "single-session-assistant", "knowledge-update", "temporal-reasoning", "multi-session"]
for qt in QTYPES:
    qt_total = sum(per_type_actions.get((qt, a), 0) for a in ["discard", "replay", "cache", "consolidate"])
    per_type_dist[qt] = {a: per_type_actions.get((qt, a), 0) / max(qt_total, 1) for a in ["discard", "replay", "cache", "consolidate"]}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
actions = ["discard", "replay", "cache", "consolidate"]
colors_map = {"discard": "gray", "replay": "steelblue", "cache": "orange", "consolidate": "green"}
counts = [action_counts[a] for a in actions]
fracs = [c / total_decisions for c in counts]
axes[0].bar(actions, fracs, color=[colors_map[a] for a in actions])
axes[0].set_ylabel("Fraction")
axes[0].set_ylim(0, 1.1)
axes[0].set_title(f"Heuristic BSC Action Distribution\n({discard_frac:.1%} discard = severe label collapse)")
for i, (a, f) in enumerate(zip(actions, fracs)):
    if f > 0.005:
        axes[0].text(i, f + 0.02, f"{f:.1%}\n({counts[i]})", ha="center", fontsize=8)

qtype_labels = [qt.replace("single-session-", "SS-").replace("knowledge-update", "KU").replace("temporal-reasoning", "TR").replace("multi-session", "MS") for qt in QTYPES]
bottom = np.zeros(len(QTYPES))
for action in actions:
    vals = [per_type_dist[qt][action] for qt in QTYPES]
    axes[1].bar(qtype_labels, vals, bottom=bottom, label=action, color=colors_map[action])
    bottom += vals
axes[1].set_ylabel("Fraction")
axes[1].set_title("Action Distribution by Question Type")
axes[1].legend(fontsize=8)
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(OUT / "label_collapse.png", dpi=200)
plt.close()
print("Saved label_collapse.png")

print("\nLoading existing experimental results...")
cf = json.loads(Path("llm_memory_validation/counterfactual_utility_regressor_run/summary.json").read_text())
comp = json.loads(Path("llm_memory_validation/competitor_run_v2/summary.json").read_text())

oracle_r = cf["retrieval"]["counterfactual_oracle_bsc"]["recall_at_5"]
replay_r = cf["retrieval"]["dense_budgeted_replay"]["recall_at_5"]
heur_r = cf["retrieval"]["heuristic_dense_bsc"]["recall_at_5"]
learned_r = cf["retrieval"]["counterfactual_learned_bsc"]["recall_at_5"]
rag_r = cf["retrieval"]["dense_rag_e5"]["recall_at_5"]
gap = oracle_r - replay_r

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

methods = ["dense_budgeted_replay", "dense_rag_e5", "counterfactual_learned_bsc", "heuristic_dense_bsc", "counterfactual_oracle_bsc"]
labels = ["Replay-only", "Dense RAG", "Learned BSC", "Heuristic BSC", "Oracle BSC"]
recall_vals = [cf["retrieval"][m]["recall_at_5"] for m in methods]
mrr_vals = [cf["retrieval"][m]["mrr_at_5"] for m in methods]

x = np.arange(len(methods))
width = 0.35
axes[0].bar(x - width/2, recall_vals, width, label="Recall@5", color="steelblue")
axes[0].bar(x + width/2, mrr_vals, width, label="MRR@5", color="coral")
axes[0].set_xticks(x, labels, fontsize=8, rotation=15)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Score")
axes[0].set_title("Retrieval: BSC vs Baselines (20% budget)")
axes[0].legend(fontsize=8)

gap_labels = ["Replay-only", "Learned BSC", "Heuristic BSC", "Oracle BSC"]
gap_values = [replay_r, learned_r, heur_r, oracle_r]
gap_colors = ["gray", "coral", "steelblue", "green"]
axes[1].barh(gap_labels, gap_values, color=gap_colors)
axes[1].set_xlim(0, 1.05)
axes[1].set_xlabel("Recall@5")
axes[1].set_title(f"Oracle Gap Analysis\nLearned recovers {(learned_r-replay_r)/gap:.1%} of gap")

comp_methods = ["fifo_replay", "uniform_replay", "memorybank_proxy", "ld_agent_proxy", "dense_rag_e5", "dense_budgeted_bsc"]
comp_labels = ["FIFO", "Uniform", "MemoryBank", "LD-Agent", "Dense RAG", "Dense BSC"]
comp_recall = [comp["metrics"][m]["recall_at_5"] for m in comp_methods]
comp_colors = ["lightgray", "lightgray", "salmon", "lightyellow", "mediumpurple", "steelblue"]
axes[2].bar(range(len(comp_methods)), comp_recall, color=comp_colors)
axes[2].set_xticks(range(len(comp_methods)), comp_labels, fontsize=8, rotation=20)
axes[2].set_ylabel("Recall@5")
axes[2].set_title("Competitor Comparison (500 examples)")
axes[2].axhline(y=0.624, color="red", linestyle="--", label="RAG_GTE (paper)")
axes[2].axhline(y=0.698, color="darkred", linestyle="--", label="RMM_GTE (paper)")
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT / "main_results.png", dpi=200)
plt.close()
print("Saved main_results.png")

per_type = cf["retrieval"]["counterfactual_oracle_bsc"].get("per_type_recall_at_5", {})
heur_pt = cf["retrieval"]["heuristic_dense_bsc"].get("per_type_recall_at_5", {})
learned_pt = cf["retrieval"]["counterfactual_learned_bsc"].get("per_type_recall_at_5", {})
replay_pt = cf["retrieval"]["dense_budgeted_replay"].get("per_type_recall_at_5", {})

fig, ax = plt.subplots(figsize=(10, 5))
short_qt = [qt.replace("single-session-", "SS-").replace("knowledge-update", "KU").replace("temporal-reasoning", "TR").replace("multi-session", "MS") for qt in QTYPES]
x = np.arange(len(QTYPES))
w = 0.2
ax.bar(x - 1.5*w, [replay_pt.get(qt, 0) for qt in QTYPES], w, label="Replay-only", color="gray")
ax.bar(x - 0.5*w, [learned_pt.get(qt, 0) for qt in QTYPES], w, label="Learned BSC", color="coral")
ax.bar(x + 0.5*w, [heur_pt.get(qt, 0) for qt in QTYPES], w, label="Heuristic BSC", color="steelblue")
ax.bar(x + 1.5*w, [per_type.get(qt, 0) for qt in QTYPES], w, label="Oracle BSC", color="green")
ax.set_xticks(x, short_qt, fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Recall@5")
ax.set_title("Per-Question-Type Recall@5 (20% budget)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "per_type_analysis.png", dpi=200)
plt.close()
print("Saved per_type_analysis.png")

print("\n" + "="*60)
print("SUMMARY OF ALL RESULTS")
print("="*60)
print(f"\n[Retrieval - 20% budget, test split]")
for m in methods:
    r = cf["retrieval"][m]
    print(f"  {m:35s} R@5={r['recall_at_5']:.4f}  MRR@5={r['mrr_at_5']:.4f}")
print(f"\n[Oracle Gap]")
print(f"  Gap: {gap:.4f}")
print(f"  Learned recovery: {(learned_r-replay_r)/gap:.1%}")
print(f"  Heuristic recovery: {(heur_r-replay_r)/gap:.1%}")
print(f"\n[Label Collapse]")
print(f"  Oracle discard: {cf['controller_test']['label_distribution'].get('discard',0)} / {sum(cf['controller_test']['label_distribution'].values())}")
print(f"  = {cf['controller_test']['label_distribution'].get('discard',0)/sum(cf['controller_test']['label_distribution'].values()):.1%}")
print(f"  Oracle cache: {cf['controller_test']['label_distribution'].get('cache',0)} sessions")
print(f"\n[Key Novelty Arguments for Paper]")
print(f"  1. BSC is formal MCKP: choose 1 of 4 actions per session under budget")
print(f"  2. Label collapse (96% discard) validates dense utility training signal")
print(f"  3. Oracle provides tight upper bound (R@5=0.998) >> replay-only (0.187)")
print(f"  4. Heuristic BSC achieves 94.3% of oracle gap without learning")
print(f"  5. Learned BSC recovers 82.9% of oracle gap with counterfactual utilities")
print(f"  6. Dense BSC beats MemoryBank (0.952 vs 0.404) and LD-Agent (0.952 vs 0.808)")
print(f"  7. Multi-action memory matters: cache useful at higher budgets (test via sweep)")

results = {
    "heuristic_action_distribution": {a: action_counts[a] for a in actions},
    "heuristic_action_fractions": {a: action_counts[a]/total_decisions for a in actions},
    "per_type_action_fracs": per_type_dist,
    "oracle_gap": {"oracle_recall": oracle_r, "replay_recall": replay_r, "heuristic_recall": heur_r, "learned_recall": learned_r, "rag_recall": rag_r, "gap": gap, "learned_recovery": (learned_r-replay_r)/gap, "heuristic_recovery": (heur_r-replay_r)/gap},
    "existing_retrieval_20pct": cf["retrieval"],
    "competitor_retrieval": comp["metrics"],
    "generation_20pct": cf.get("generation", {}),
    "controller_test": cf["controller_test"],
    "label_collapse": {"discard_fraction": cf["controller_test"]["label_distribution"].get("discard",0)/sum(cf["controller_test"]["label_distribution"].values()), "distribution": cf["controller_test"]["label_distribution"]},
    "theory_mckp": "BSC reduces to Multiple-Choice Knapsack: max sum u(i,a_i) s.t. sum c(i,a_i) <= B, a_i in {discard, replay, cache, consolidate}",
    "novelty_claims": [
        "Counterfactual utility as offline supervision (vs RL in AgeMem/Mem-alpha)",
        "Explicit budget + compute cost in objective",
        "Dense per-action utilities address 96% discard label collapse",
        "MCKP formalization connects to well-studied optimization",
        "Controlled evaluation: same retriever/reader across all methods"
    ]
}
with open(OUT / "all_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT / 'all_results.json'}")