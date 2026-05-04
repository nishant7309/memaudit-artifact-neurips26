"""Budget sweep + ablations + significance + hybrid controller.
Run on local GPU. Handles the 4 most critical reviewer concerns:
1. Budget sweep at 5 budget levels
2. Ablations (no-cache, no-consolidate) at each level
3. Paired bootstrap significance tests
4. Hybrid heuristic+utility controller
"""
from __future__ import annotations
import time, json, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_memory_validation.bsc_longmemeval import (
    load_dataset, build_bsc, build_replay_only_router, build_fifo_replay,
    build_uniform_replay, classify_action, count_words, session_text, tail_snippet,
    extract_fact_lines, full_budget_words, MemoryEntry, QUESTION_TYPES,
)
from llm_memory_validation.counterfactual_dense_bsc import (
    POSITIVE_ACTIONS, ACTION_TO_ID, build_context, candidate_gain,
    action_utilities_for_session, feature_vector, decisions_from_utilities,
    oversample_keep_rows, counterfactual_oracle_select, split_examples,
    build_learned_selection, dense_predict_ids_from_candidates,
)
from llm_memory_validation.paper_competitor_suite import (
    DenseEmbedder, DenseItem, dense_rag_retrieve, dense_items_from_entries,
    memorybank_retrieve, ld_agent_retrieve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("llm_memory_validation/neurips_full_results")
OUT.mkdir(parents=True, exist_ok=True)

TOPK = 5
BUDGET_FRACS = [0.10, 0.15, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2]

print("[1/7] Loading dataset and building embeddings...")
t0 = time.time()
examples = load_dataset()
embedder = DenseEmbedder(model_name="intfloat/e5-base-v2")
train_ex, val_ex, test_ex = split_examples(examples, seed=11)
print(f"  Done in {time.time()-t0:.1f}s: {len(examples)} examples, {len(train_ex)}/{len(val_ex)}/{len(test_ex)} split")

print("[2/7] Building contexts for all budget levels...")
all_contexts = {}
for bf in BUDGET_FRACS:
    t1 = time.time()
    all_contexts[bf] = {ex["question_id"]: build_context(ex, bf, embedder) for ex in examples}
    print(f"  Budget {bf:.0%}: {time.time()-t1:.1f}s")

print("[3/7] Running budget sweep with all methods + ablations...")
sweep = {}
for bf in BUDGET_FRACS:
    print(f"\n  === Budget {bf:.0%} ===")
    t1 = time.time()
    contexts = all_contexts[bf]
    
    def eval_fn(name, fn, examples_list):
        recalls, mrrs, per_type = [], [], defaultdict(list)
        for ex in examples_list:
            ctx = contexts[ex["question_id"]]
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
        }
    
    budget_words_default = max(256, int(full_budget_words(examples[0]) * bf))
    
    # 1. FIFO replay
    def fifo_fn(ex, ctx):
        entries = build_fifo_replay(ex, bf)
        items = dense_items_from_entries(ex, entries, embedder, TOPK)
        return [item.session_id for item in items]
    
    # 2. Dense RAG
    def rag_fn(ex, ctx):
        items = dense_rag_retrieve(ex, embedder, TOPK)
        return [item.session_id for item in items]
    
    # 3. Replay-only router
    def replay_fn(ex, ctx):
        entries = build_replay_only_router(ex, bf)
        items = dense_items_from_entries(ex, entries, embedder, TOPK)
        return [item.session_id for item in items]
    
    # 4. Heuristic BSC
    def heur_fn(ex, ctx):
        entries = build_bsc(ex, bf)
        items = dense_items_from_entries(ex, entries, embedder, TOPK)
        return [item.session_id for item in items]
    
    # 5. Oracle BSC
    def oracle_fn(ex, ctx):
        cands, _, _ = counterfactual_oracle_select(ctx, TOPK)
        return dense_predict_ids_from_candidates(ctx, cands, TOPK)
    
    # 6. No-cache ablation (oracle: only replay + consolidate)
    def no_cache_fn(ex, ctx):
        candidates = []
        for si in range(len(ex["haystack_sessions"])):
            best_action, best_util = "discard", -999.0
            for a in ["replay", "consolidate"]:
                cand = ctx.candidates_by_session.get(si, {}).get(a)
                if cand is None: continue
                g = candidate_gain([], ctx, cand, TOPK)
                if g > best_util: best_util, best_action = g, a
            if best_util > 0.01 and best_action != "discard":
                candidates.append(ctx.candidates_by_session[si][best_action])
        sorted_c = sorted(candidates, key=lambda c: (c.similarity - 0.25 * c.cost_words / max(ctx.budget_words, 1)), reverse=True)
        budget_c, used = [], 0
        for c in sorted_c:
            if used + c.cost_words <= ctx.budget_words:
                budget_c.append(c); used += c.cost_words
        return dense_predict_ids_from_candidates(ctx, budget_c, TOPK)
    
    # 7. No-consolidate ablation (oracle: only replay + cache)
    def no_consolidate_fn(ex, ctx):
        candidates = []
        for si in range(len(ex["haystack_sessions"])):
            best_action, best_util = "discard", -999.0
            for a in ["replay", "cache"]:
                cand = ctx.candidates_by_session.get(si, {}).get(a)
                if cand is None: continue
                g = candidate_gain([], ctx, cand, TOPK)
                if g > best_util: best_util, best_action = g, a
            if best_util > 0.01 and best_action != "discard":
                candidates.append(ctx.candidates_by_session[si][best_action])
        sorted_c = sorted(candidates, key=lambda c: (c.similarity - 0.25 * c.cost_words / max(ctx.budget_words, 1)), reverse=True)
        budget_c, used = [], 0
        for c in sorted_c:
            if used + c.cost_words <= ctx.budget_words:
                budget_c.append(c); used += c.cost_words
        return dense_predict_ids_from_candidates(ctx, budget_c, TOPK)
    
    # 8. MemoryBank proxy
    def memorybank_fn(ex, ctx):
        items = memorybank_retrieve(ex, embedder, TOPK)
        return [item.session_id for item in items]
    
    # 9. LD-Agent proxy
    def ldagent_fn(ex, ctx):
        items = ld_agent_retrieve(ex, embedder, TOPK)
        return [item.session_id for item in items]
    
    methods = {
        "fifo_replay": fifo_fn,
        "dense_rag_e5": rag_fn,
        "replay_only_router": replay_fn,
        "heuristic_bsc": heur_fn,
        "oracle_bsc": oracle_fn,
        "no_cache_oracle": no_cache_fn,
        "no_consolidate_oracle": no_consolidate_fn,
        "memorybank": memorybank_fn,
        "ld_agent": ldagent_fn,
    }
    
    ret = {}
    for name, fn in methods.items():
        r = eval_fn(name, fn, test_ex)
        ret[name] = r
        print(f"    {name:30s} R@5={r['recall_at_5']:.4f}  MRR@5={r['mrr_at_5']:.4f}")
    
    # 10. Train learned controller at this budget
    print(f"  Training learned controller at {bf:.0%}...")
    train_x, train_y, train_ora = [], [], []
    for ex in train_ex:
        ctx_ = contexts[ex["question_id"]]
        _, decs, _ = counterfactual_oracle_select(ctx_, TOPK)
        for si in range(len(ex["haystack_sessions"])):
            train_x.append(feature_vector(ex, ctx_, si))
            train_y.append(action_utilities_for_session(ctx_, si, TOPK))
            train_ora.append(ACTION_TO_ID[decs[si]])
    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)
    train_ora = np.array(train_ora, dtype=np.int64)
    
    val_x, val_y, val_ora = [], [], []
    for ex in val_ex:
        ctx_ = contexts[ex["question_id"]]
        _, decs, _ = counterfactual_oracle_select(ctx_, TOPK)
        for si in range(len(ex["haystack_sessions"])):
            val_x.append(feature_vector(ex, ctx_, si))
            val_y.append(action_utilities_for_session(ctx_, si, TOPK))
            val_ora.append(ACTION_TO_ID[decs[si]])
    val_x = np.array(val_x, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.float32)
    val_ora = np.array(val_ora, dtype=np.int64)
    
    best_pipe, best_thresh, best_f1 = None, 0.0, -1.0
    for seed in SEEDS:
        sx, sy = oversample_keep_rows(train_x, train_y, seed)
        pipe = SKPipeline([
            ("s", StandardScaler()),
            ("m", MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
                             alpha=1e-4, learning_rate_init=1e-3, batch_size=256, max_iter=250,
                             random_state=seed, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=15)),
        ])
        pipe.fit(sx, sy)
        vp = pipe.predict(val_x)
        for th in [-0.05, 0.0, 0.01, 0.02, 0.03, 0.05]:
            vd = decisions_from_utilities(vp, float(th))
            f = f1_score(val_ora, vd, average="macro")
            a = accuracy_score(val_ora, vd)
            if (f, a) > (best_f1, 0):
                best_pipe, best_thresh, best_f1 = pipe, float(th), f
    
    controller = {"pipeline": best_pipe, "threshold": best_thresh}
    
    def learned_fn(ex, ctx):
        cands, _, _ = build_learned_selection(ex, ctx, controller)
        return dense_predict_ids_from_candidates(ctx, cands, TOPK)
    
    ret["learned_bsc"] = eval_fn("learned_bsc", learned_fn, test_ex)
    print(f"    {'learned_bsc':30s} R@5={ret['learned_bsc']['recall_at_5']:.4f}  MRR@5={ret['learned_bsc']['mrr_at_5']:.4f}")
    
    # 11. Hybrid: heuristic action selection + utility-based discard threshold
    def hybrid_fn(ex, ctx):
        heuristic_entries = build_bsc(ex, bf)
        filtered = []
        for entry in heuristic_entries:
            si_idx = None
            for si, sid in enumerate(ex["haystack_session_ids"]):
                if sid == entry.session_id:
                    si_idx = si
                    break
            if si_idx is not None:
                feat = feature_vector(ex, ctx, si_idx)
                pred_utils = best_pipe.predict(feat.reshape(1, -1))[0]
                max_util = float(max(pred_utils))
                if max_util > best_thresh:
                    filtered.append(entry)
            else:
                filtered.append(entry)  # keep if we can't find the session
        if not filtered:
            heuristic_entries.sort(key=lambda e: e.priority if hasattr(e, 'priority') and e.priority else 0, reverse=True)
            filtered = heuristic_entries[:max(1, int(len(heuristic_entries) * 0.5))]
        items = dense_items_from_entries(ex, filtered, embedder, TOPK)
        return [item.session_id for item in items]
    
    ret["hybrid_bsc"] = eval_fn("hybrid_bsc", hybrid_fn, test_ex)
    print(f"    {'hybrid_bsc':30s} R@5={ret['hybrid_bsc']['recall_at_5']:.4f}  MRR@5={ret['hybrid_bsc']['mrr_at_5']:.4f}")
    
    sweep[f"budget_{bf:.2f}"] = {"budget_frac": bf, "retrieval": ret}
    print(f"  Budget {bf:.0%} done in {time.time()-t1:.1f}s")

print("\n[4/7] Paired bootstrap significance tests (budget=0.20)...")
ref_idx = "budget_0.20"
if ref_idx in sweep:
    ref = sweep[ref_idx]["retrieval"]
    pairs = [
        ("oracle_bsc", "replay_only_router"),
        ("heuristic_bsc", "replay_only_router"),
        ("heuristic_bsc", "dense_rag_e5"),
        ("learned_bsc", "replay_only_router"),
        ("hybrid_bsc", "heuristic_bsc"),
        ("oracle_bsc", "heuristic_bsc"),
    ]
    sig_results = {}
    for ma, mb in pairs:
        if ma in ref and mb in ref:
            diff = ref[ma]["recall_at_5"] - ref[mb]["recall_at_5"]
            sig_results[f"{ma}_vs_{mb}"] = {
                "recall_diff": diff,
                "method_a": ref[ma]["recall_at_5"],
                "method_b": ref[mb]["recall_at_5"],
                "note": "Paired bootstrap CI requires per-example scores; aggregate diff reported here",
            }
            print(f"  {ma} vs {mb}: diff={diff:+.4f}")
else:
    sig_results = {}

print("\n[5/7] Computing heuristic action distribution by budget...")
action_dist = {}
for bf in BUDGET_FRACS:
    actions = Counter()
    for ex in examples:
        total = len(ex["haystack_sessions"])
        for i, session in enumerate(ex["haystack_sessions"]):
            a = classify_action(session, i, total)
            actions[a] += 1
    total_dec = sum(actions.values())
    action_dist[bf] = {a: actions[a] / total_dec for a in ["discard", "replay", "cache", "consolidate"]}
    action_dist[bf]["_total"] = total_dec
    action_dist[bf]["_counts"] = dict(actions)

print("\n[6/7] Per-example significance between heuristic and RAG at 20%...")
if ref_idx in sweep:
    heuristic_recalls = []
    rag_recalls = []
    for ex in test_ex:
        ctx = all_contexts[0.20][ex["question_id"]]
        gold = set(ex["answer_session_ids"])
        
        h_entries = build_bsc(ex, 0.20)
        h_items = dense_items_from_entries(ex, h_entries, embedder, TOPK)
        h_ids = [item.session_id for item in h_items]
        h_recall = len(set(h_ids) & gold) / max(len(gold), 1)
        heuristic_recalls.append(h_recall)
        
        r_items = dense_rag_retrieve(ex, embedder, TOPK)
        r_ids = [item.session_id for item in r_items]
        r_recall = len(set(r_ids) & gold) / max(len(gold), 1)
        rag_recalls.append(r_recall)
    
    heuristic_recalls = np.array(heuristic_recalls)
    rag_recalls = np.array(rag_recalls)
    diffs = heuristic_recalls - rag_recalls
    observed_diff = float(np.mean(diffs))
    
    rng = np.random.default_rng(42)
    n = len(diffs)
    bootstrap_diffs = np.array([float(np.mean(diffs[rng.integers(0, n, size=n)])) for _ in range(10000)])
    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))
    p_value = float(min(np.mean(bootstrap_diffs <= 0) * 2, 1.0))
    
    sig_results["heuristic_vs_rag_bootstrap"] = {
        "observed_diff": observed_diff,
        "ci_95": [ci_lower, ci_upper],
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
        "n_examples": n,
        "heuristic_mean": float(np.mean(heuristic_recalls)),
        "rag_mean": float(np.mean(rag_recalls)),
    }
    print(f"  Heuristic vs RAG: diff={observed_diff:+.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.6f}")
    print(f"  Significant at p<0.05: {p_value < 0.05}")

print("\n[7/7] Saving results and generating figures...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
method_labels = {
    "replay_only_router": "Replay-only",
    "dense_rag_e5": "Dense RAG",
    "heuristic_bsc": "Heuristic BSC",
    "oracle_bsc": "Oracle BSC",
    "learned_bsc": "Learned BSC",
    "hybrid_bsc": "Hybrid BSC",
    "no_cache_oracle": "No-cache",
    "no_consolidate_oracle": "No-consolidate",
    "memorybank": "MemoryBank",
    "ld_agent": "LD-Agent",
    "fifo_replay": "FIFO",
}
colors = {
    "replay_only_router": "gray", "dense_rag_e5": "purple", "heuristic_bsc": "steelblue",
    "oracle_bsc": "green", "learned_bsc": "coral", "hybrid_bsc": "darkred",
    "no_cache_oracle": "orange", "no_consolidate_oracle": "brown",
    "memorybank": "pink", "ld_agent": "gold", "fifo_replay": "lightgray",
}
markers = {
    "replay_only_router": "v", "dense_rag_e5": "D", "heuristic_bsc": "o",
    "oracle_bsc": "*", "learned_bsc": "s", "hybrid_bsc": "P",
    "no_cache_oracle": "^", "no_consolidate_oracle": "<",
}

for metric_key, metric_name, ax in [("recall_at_5", "Recall@5", axes[0]), ("mrr_at_5", "MRR@5", axes[1])]:
    for mk, label in method_labels.items():
        bvs, mvs = [], []
        for bk in sorted(sweep.keys()):
            if mk in sweep[bk]["retrieval"]:
                bvs.append(sweep[bk]["budget_frac"])
                mvs.append(sweep[bk]["retrieval"][mk][metric_key])
        if bvs:
            ax.plot(bvs, mvs, marker=markers.get(mk, "o"), label=label, color=colors.get(mk, "black"), linewidth=1.5)
    ax.set_xlabel("Memory Budget (%)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Memory Budget")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "budget_sweep.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
budgets = sorted(action_dist.keys())
for action, color in [("discard", "gray"), ("replay", "steelblue"), ("cache", "orange"), ("consolidate", "green")]:
    vals = [action_dist[bf][action] for bf in budgets]
    ax.plot(budgets, vals, marker="o", label=action, color=color)
ax.set_xlabel("Memory Budget (%)")
ax.set_ylabel("Fraction of sessions")
ax.set_title("Heuristic Action Distribution vs Budget")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "action_dist_vs_budget.png", dpi=200)
plt.close()

results = {
    "budget_sweep": {k: {kk: vv for kk, vv in v.items() if kk != "retrieval" or isinstance(vv, dict)} for k, v in sweep.items()},
    "action_distribution_by_budget": action_dist,
    "significance": sig_results,
}

with open(OUT / "full_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*70)
print("BUDGET SWEEP RESULTS")
print("="*70)
for bk in sorted(sweep.keys()):
    bf = sweep[bk]["budget_frac"]
    r = sweep[bk]["retrieval"]
    print(f"\n  Budget {bf:.0%}:")
    for mk in ["fifo_replay", "replay_only_router", "dense_rag_e5", "memorybank", "ld_agent", "heuristic_bsc", "learned_bsc", "hybrid_bsc", "no_cache_oracle", "no_consolidate_oracle", "oracle_bsc"]:
        if mk in r:
            print(f"    {mk:30s} R@5={r[mk]['recall_at_5']:.4f}  MRR={r[mk]['mrr_at_5']:.4f}")

print(f"\nResults saved to {OUT}")