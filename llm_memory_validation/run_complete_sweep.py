"""Step-by-step NeurIPS experiments with progress bars.
Step 1: Build embeddings + contexts (one time cost)
Step 2: Evaluate all methods at each budget level
Step 3: Train learned controller at each budget
Step 4: Significance tests
Step 5: Save + plot
"""
from __future__ import annotations
import time, json, sys, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_memory_validation.bsc_longmemeval import (
    load_dataset, build_bsc, build_replay_only_router, build_fifo_replay,
    classify_action, full_budget_words, MemoryEntry,
)
from llm_memory_validation.counterfactual_dense_bsc import (
    POSITIVE_ACTIONS, ACTION_TO_ID, build_context, candidate_gain,
    action_utilities_for_session, feature_vector, decisions_from_utilities,
    oversample_keep_rows, counterfactual_oracle_select, split_examples,
    build_learned_selection, dense_predict_ids_from_candidates,
    ControllerBundle,
)
from llm_memory_validation.paper_competitor_suite import (
    DenseEmbedder, dense_items_from_entries, dense_rag_retrieve,
    memorybank_retrieve, ld_agent_retrieve,
)

OUT = Path("llm_memory_validation/neurips_full_results")
OUT.mkdir(parents=True, exist_ok=True)
TOPK = 5
BUDGET_FRACS = [0.10, 0.15, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2]

# -- STEP 1: Load dataset + embeddings --
print("\n" + "="*60)
print("STEP 1/5: Loading dataset + building E5 embeddings")
print("="*60)
examples = load_dataset()
print(f"  Dataset: {len(examples)} examples")
embedder = DenseEmbedder(model_name="intfloat/e5-base-v2")
train_ex, val_ex, test_ex = split_examples(examples, seed=11)
print(f"  Split: {len(train_ex)}/{len(val_ex)}/{len(test_ex)}")

# -- STEP 2: Build contexts for each budget --
print("\n" + "="*60)
print("STEP 2/5: Building contexts for each budget level")
print("="*60)
all_contexts = {}
for bf in tqdm(BUDGET_FRACS, desc="Building contexts"):
    t0 = time.time()
    all_contexts[bf] = {ex["question_id"]: build_context(ex, bf, embedder) for ex in examples}
    tqdm.write(f"  Budget {bf:.0%}: {time.time()-t0:.0f}s")

# -- STEP 3: Evaluate methods at each budget --
print("\n" + "="*60)
print("STEP 3/5: Evaluating all methods at each budget level")
print("="*60)

def eval_method(name, fn, test_list, ctx_map, topk=5):
    recalls, mrrs, per_type = [], [], defaultdict(list)
    for ex in tqdm(test_list, desc=f"  {name}", leave=False):
        ctx = ctx_map[ex["question_id"]]
        gold = set(ex["answer_session_ids"])
        ids = fn(ex, ctx)
        hits = [r for r, sid in enumerate(ids, 1) if sid in gold]
        recalls.append(len(set(ids) & gold) / max(len(gold), 1))
        mrrs.append(0.0 if not hits else 1.0 / min(hits))
        per_type[ex["question_type"]].append(recalls[-1])
    return {
        "recall_at_5": float(np.mean(recalls)),
        "mrr_at_5": float(np.mean(mrrs)),
        "per_type_recall_at_5": {qt: float(np.mean(v)) for qt, v in per_type.items()},
        "n": len(recalls),
        "_recalls": [float(r) for r in recalls],  # for significance tests
    }

sweep = {}

for bf in BUDGET_FRACS:
    print(f"\n  -- Budget {bf:.0%} --")
    ctx = all_contexts[bf]
    ret = {}

    # FIFO replay
    ret["fifo_replay"] = eval_method("FIFO",
        lambda ex, c: [item.session_id for item in dense_items_from_entries(ex, build_fifo_replay(ex, bf), embedder, TOPK)],
        test_ex, ctx)

    # Replay-only router
    ret["replay_only_router"] = eval_method("Replay-router",
        lambda ex, c: [item.session_id for item in dense_items_from_entries(ex, build_replay_only_router(ex, bf), embedder, TOPK)],
        test_ex, ctx)

    # Dense RAG
    ret["dense_rag_e5"] = eval_method("Dense-RAG",
        lambda ex, c: [item.session_id for item in dense_rag_retrieve(ex, embedder, TOPK)],
        test_ex, ctx)

    # MemoryBank proxy
    ret["memorybank"] = eval_method("MemBank",
        lambda ex, c: [item.session_id for item in memorybank_retrieve(ex, embedder, TOPK)],
        test_ex, ctx)

    # LD-Agent proxy
    ret["ld_agent"] = eval_method("LD-Agent",
        lambda ex, c: [item.session_id for item in ld_agent_retrieve(ex, embedder, TOPK)],
        test_ex, ctx)

    # Heuristic BSC
    ret["heuristic_bsc"] = eval_method("Heur-BSC",
        lambda ex, c: [item.session_id for item in dense_items_from_entries(ex, build_bsc(ex, bf), embedder, TOPK)],
        test_ex, ctx)

    # Oracle BSC
    def _oracle(ex, c):
        cands, _, _ = counterfactual_oracle_select(c, TOPK)
        return dense_predict_ids_from_candidates(c, cands, TOPK)
    ret["oracle_bsc"] = eval_method("Oracle-BSC", _oracle, test_ex, ctx)

    # No-cache ablation
    def _no_cache(ex, c):
        candidates = []
        for si in range(len(ex["haystack_sessions"])):
            best_a, best_u = "discard", -999.0
            for a in ["replay", "consolidate"]:
                cand = c.candidates_by_session.get(si, {}).get(a)
                if cand is None: continue
                g = candidate_gain([], c, cand, TOPK)
                if g > best_u: best_u, best_a = g, a
            if best_u > 0.01 and best_a != "discard":
                candidates.append(c.candidates_by_session[si][best_a])
        sorted_c = sorted(candidates, key=lambda x: (x.similarity - 0.25 * x.cost_words / max(c.budget_words, 1)), reverse=True)
        budget_c, used = [], 0
        for x in sorted_c:
            if used + x.cost_words <= c.budget_words: budget_c.append(x); used += x.cost_words
        return dense_predict_ids_from_candidates(c, budget_c, TOPK)
    ret["no_cache_oracle"] = eval_method("No-cache", _no_cache, test_ex, ctx)

    # No-consolidate ablation
    def _no_consol(ex, c):
        candidates = []
        for si in range(len(ex["haystack_sessions"])):
            best_a, best_u = "discard", -999.0
            for a in ["replay", "cache"]:
                cand = c.candidates_by_session.get(si, {}).get(a)
                if cand is None: continue
                g = candidate_gain([], c, cand, TOPK)
                if g > best_u: best_u, best_a = g, a
            if best_u > 0.01 and best_a != "discard":
                candidates.append(c.candidates_by_session[si][best_a])
        sorted_c = sorted(candidates, key=lambda x: (x.similarity - 0.25 * x.cost_words / max(c.budget_words, 1)), reverse=True)
        budget_c, used = [], 0
        for x in sorted_c:
            if used + x.cost_words <= c.budget_words: budget_c.append(x); used += x.cost_words
        return dense_predict_ids_from_candidates(c, budget_c, TOPK)
    ret["no_consolidate_oracle"] = eval_method("No-consol", _no_consol, test_ex, ctx)

    # Train learned controller at this budget
    print(f"  Training learned controller...")
    train_x, train_y, train_ora = [], [], []
    for ex in tqdm(train_ex, desc="  Train features", leave=False):
        c_ = ctx[ex["question_id"]]
        _, decs, _ = counterfactual_oracle_select(c_, TOPK)
        for si in range(len(ex["haystack_sessions"])):
            train_x.append(feature_vector(ex, c_, si))
            train_y.append(action_utilities_for_session(c_, si, TOPK))
            train_ora.append(ACTION_TO_ID[decs[si]])
    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)
    train_ora = np.array(train_ora, dtype=np.int64)

    val_x, val_y, val_ora = [], [], []
    for ex in tqdm(val_ex, desc="  Val features", leave=False):
        c_ = ctx[ex["question_id"]]
        _, decs, _ = counterfactual_oracle_select(c_, TOPK)
        for si in range(len(ex["haystack_sessions"])):
            val_x.append(feature_vector(ex, c_, si))
            val_y.append(action_utilities_for_session(c_, si, TOPK))
            val_ora.append(ACTION_TO_ID[decs[si]])
    val_x = np.array(val_x, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.float32)
    val_ora = np.array(val_ora, dtype=np.int64)

    best_pipe, best_thresh, best_f1 = None, 0.0, -1.0
    for seed in tqdm(SEEDS, desc="  Seeds", leave=False):
        sx, sy = oversample_keep_rows(train_x, train_y, seed)
        pipe = SKPipeline([
            ("s", StandardScaler()),
            ("m", MLPRegressor(hidden_layer_sizes=(128,128), activation="relu", solver="adam",
                             alpha=1e-4, learning_rate_init=1e-3, batch_size=256, max_iter=250,
                             random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)),
        ])
        pipe.fit(sx, sy)
        vp = pipe.predict(val_x)
        for th in [-0.05, 0.0, 0.01, 0.02, 0.03, 0.05]:
            vd = decisions_from_utilities(vp, float(th))
            f1 = f1_score(val_ora, vd, average="macro")
            acc = accuracy_score(val_ora, vd)
            if (f1, acc) > (best_f1, 0):
                best_pipe, best_thresh, best_f1 = pipe, float(th), f1

    controller = ControllerBundle(
        pipeline=best_pipe, seed=0, threshold=best_thresh,
        train_mae=0.0, val_mae=0.0, train_macro_f1=0.0,
        val_macro_f1=float(best_f1), train_accuracy=0.0, val_accuracy=0.0,
    )
    print(f"  Controller: threshold={best_thresh:.3f}, val_macro_f1={best_f1:.4f}")

    def _learned(ex, c):
        cands, _, _ = build_learned_selection(ex, c, controller)
        return dense_predict_ids_from_candidates(c, cands, TOPK)
    ret["learned_bsc"] = eval_method("Learned-BSC", _learned, test_ex, ctx)

    # Hybrid: heuristic action selection + utility-based discard
    def _hybrid(ex, c):
        heur_entries = build_bsc(ex, bf)
        filtered = []
        for entry in heur_entries:
            si_idx = None
            for si, sid in enumerate(ex["haystack_session_ids"]):
                if sid == entry.session_id: si_idx = si; break
            if si_idx is not None and si_idx < len(ex["haystack_sessions"]):
                feat = feature_vector(ex, c, si_idx).reshape(1, -1)
                pred_utils = best_pipe.predict(feat)[0]
                max_util = float(max(pred_utils))
                if max_util > best_thresh:
                    filtered.append(entry)
            else:
                filtered.append(entry)
        if not filtered:
            filtered = sorted(heur_entries, key=lambda e: getattr(e, 'priority', 0), reverse=True)[:max(1, len(heur_entries)//2)]
        items = dense_items_from_entries(ex, filtered, embedder, TOPK)
        return [item.session_id for item in items]
    ret["hybrid_bsc"] = eval_method("Hybrid-BSC", _hybrid, test_ex, ctx)

    # Print summary for this budget
    for name in ["fifo_replay", "replay_only_router", "dense_rag_e5", "memorybank", "ld_agent",
                 "heuristic_bsc", "learned_bsc", "hybrid_bsc", "no_cache_oracle", "no_consolidate_oracle", "oracle_bsc"]:
        if name in ret:
            r = ret[name]
            print(f"    {name:30s} R@5={r['recall_at_5']:.4f}  MRR@5={r['mrr_at_5']:.4f}")

    sweep[f"budget_{bf:.2f}"] = {"budget_frac": bf, "retrieval": ret}

# -- STEP 4: Significance tests --
print("\n" + "="*60)
print("STEP 4/5: Paired bootstrap significance tests (budget=20%)")
print("="*60)

ref_ret = sweep["budget_0.20"]["retrieval"]

# Heuristic vs RAG
h_recall = np.array(ref_ret["heuristic_bsc"]["_recalls"])
r_recall = np.array(ref_ret["dense_rag_e5"]["_recalls"])
diffs = h_recall - r_recall
obs_diff = float(np.mean(diffs))
rng = np.random.default_rng(42)
n = len(diffs)
boot = np.array([float(np.mean(diffs[rng.integers(0, n, size=n)])) for _ in range(10000)])
ci_lo = float(np.percentile(boot, 2.5))
ci_hi = float(np.percentile(boot, 97.5))
p = float(min(np.mean(boot <= 0) * 2, 1.0))
sig_heur_rag = {"diff": obs_diff, "ci_95": [ci_lo, ci_hi], "p": p, "sig": p < 0.05}
print(f"  Heuristic vs RAG: diff={obs_diff:+.4f}, CI=[{ci_lo:.4f},{ci_hi:.4f}], p={p:.6f}, sig={p<0.05}")

# Oracle vs replay
o_recall = np.array(ref_ret["oracle_bsc"]["_recalls"])
rp_recall = np.array(ref_ret["replay_only_router"]["_recalls"])
diffs2 = o_recall - rp_recall
obs2 = float(np.mean(diffs2))
boot2 = np.array([float(np.mean(diffs2[rng.integers(0, n, size=n)])) for _ in range(10000)])
p2 = float(min(np.mean(boot2 <= 0) * 2, 1.0))
sig_oracle_replay = {"diff": obs2, "ci_95": [float(np.percentile(boot2, 2.5)), float(np.percentile(boot2, 97.5))], "p": p2, "sig": p2 < 0.05}
print(f"  Oracle vs Replay: diff={obs2:+.4f}, CI=[{sig_oracle_replay['ci_95'][0]:.4f},{sig_oracle_replay['ci_95'][1]:.4f}], p={p2:.6f}, sig={p2<0.05}")

# Heuristic vs learned
l_recall = np.array(ref_ret["learned_bsc"]["_recalls"])
diffs3 = h_recall - l_recall
obs3 = float(np.mean(diffs3))
boot3 = np.array([float(np.mean(diffs3[rng.integers(0, n, size=n)])) for _ in range(10000)])
p3 = float(min(np.mean(boot3 <= 0) * 2, 1.0))
sig_heur_learned = {"diff": obs3, "ci_95": [float(np.percentile(boot3, 2.5)), float(np.percentile(boot3, 97.5))], "p": p3, "sig": p3 < 0.05}
print(f"  Heuristic vs Learned: diff={obs3:+.4f}, CI=[{sig_heur_learned['ci_95'][0]:.4f},{sig_heur_learned['ci_95'][1]:.4f}], p={p3:.6f}, sig={p3<0.05}")

# -- STEP 5: Save + plot --
print("\n" + "="*60)
print("STEP 5/5: Saving results and generating figures")
print("="*60)

# Strip per-example arrays for JSON (too large)
for bk in sweep:
    for mk in sweep[bk]["retrieval"]:
        if "_recalls" in sweep[bk]["retrieval"][mk]:
            del sweep[bk]["retrieval"][mk]["_recalls"]

results = {
    "budget_sweep": sweep,
    "significance": {
        "heuristic_vs_rag": sig_heur_rag,
        "oracle_vs_replay": sig_oracle_replay,
        "heuristic_vs_learned": sig_heur_learned,
    },
}

with open(OUT / "full_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Budget sweep figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
method_pairs = {
    "replay_only_router": ("Replay-only", "gray", "v"),
    "dense_rag_e5": ("Dense RAG", "mediumpurple", "D"),
    "memorybank": ("MemoryBank", "pink", "p"),
    "ld_agent": ("LD-Agent", "gold", "X"),
    "heuristic_bsc": ("Heuristic BSC", "steelblue", "o"),
    "learned_bsc": ("Learned BSC", "coral", "s"),
    "hybrid_bsc": ("Hybrid BSC", "darkred", "P"),
    "no_cache_oracle": ("No-cache oracle", "orange", "^"),
    "no_consolidate_oracle": ("No-consolidate oracle", "brown", "<"),
    "oracle_bsc": ("Oracle BSC", "green", "*"),
}

for ax_idx, (metric, ylabel) in enumerate([("recall_at_5", "Recall@5"), ("mrr_at_5", "MRR@5")]):
    ax = axes[ax_idx]
    for mk, (label, color, marker) in method_pairs.items():
        bvs, mvs = [], []
        for bk in sorted(sweep.keys()):
            if mk in sweep[bk]["retrieval"]:
                bvs.append(sweep[bk]["budget_frac"])
                mvs.append(sweep[bk]["retrieval"][mk][metric])
        if bvs:
            ax.plot(bvs, mvs, marker=marker, label=label, color=color, linewidth=1.5, markersize=6)
    ax.set_xlabel("Memory Budget (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Budget")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "budget_sweep.png", dpi=200)
plt.close()

# Ablation figure at 20%
if "budget_0.20" in sweep:
    ret20 = sweep["budget_0.20"]["retrieval"]
    ablation_methods = ["replay_only_router", "heuristic_bsc", "no_cache_oracle", "no_consolidate_oracle", "oracle_bsc"]
    ablation_labels = ["Replay-only", "Full BSC", "No-cache", "No-consolidate", "Oracle"]
    ablation_colors = ["gray", "steelblue", "orange", "brown", "green"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    r5 = [ret20[m]["recall_at_5"] for m in ablation_methods]
    m5 = [ret20[m]["mrr_at_5"] for m in ablation_methods]
    x = np.arange(len(ablation_methods))
    w = 0.35
    ax2.bar(x - w/2, r5, w, label="Recall@5", color="steelblue")
    ax2.bar(x + w/2, m5, w, label="MRR@5", color="coral")
    ax2.set_xticks(x, ablation_labels, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score")
    ax2.set_title("Ablation: Action Removal (20% budget)")
    ax2.legend()
    for i, (r, m) in enumerate(zip(r5, m5)):
        ax2.text(i - w/2, r + 0.02, f"{r:.3f}", ha="center", fontsize=7, color="steelblue")
        ax2.text(i + w/2, m + 0.02, f"{m:.3f}", ha="center", fontsize=7, color="coral")
    plt.tight_layout()
    plt.savefig(OUT / "ablations.png", dpi=200)
    plt.close()

print("\n" + "="*60)
print("COMPLETE RESULTS SUMMARY")
print("="*60)

for bk in sorted(sweep.keys()):
    bf = sweep[bk]["budget_frac"]
    r = sweep[bk]["retrieval"]
    print(f"\n  Budget {bf:.0%}:")
    for mk in ["fifo_replay", "replay_only_router", "dense_rag_e5", "memorybank", "ld_agent",
               "heuristic_bsc", "learned_bsc", "hybrid_bsc", "no_cache_oracle", "no_consolidate_oracle", "oracle_bsc"]:
        if mk in r:
            print(f"    {mk:35s} R@5={r[mk]['recall_at_5']:.4f}  MRR@5={r[mk]['mrr_at_5']:.4f}")

print(f"\n  Significance (paired bootstrap, 10000 resamples):")
print(f"    Heuristic vs RAG:       diff={sig_heur_rag['diff']:+.4f}, 95% CI=[{sig_heur_rag['ci_95'][0]:.4f},{sig_heur_rag['ci_95'][1]:.4f}], p={sig_heur_rag['p']:.6f}")
print(f"    Oracle vs Replay:       diff={sig_oracle_replay['diff']:+.4f}, 95% CI=[{sig_oracle_replay['ci_95'][0]:.4f},{sig_oracle_replay['ci_95'][1]:.4f}], p={sig_oracle_replay['p']:.6f}")
print(f"    Heuristic vs Learned:   diff={sig_heur_learned['diff']:+.4f}, 95% CI=[{sig_heur_learned['ci_95'][0]:.4f},{sig_heur_learned['ci_95'][1]:.4f}], p={sig_heur_learned['p']:.6f}")

print(f"\nAll results saved to {OUT}")
print(f"Figures: budget_sweep.png, ablations.png")