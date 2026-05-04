"""Micro-experiments: each step < 2 min. Run individually."""
from __future__ import annotations
import json, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("llm_memory_validation/neurips_micro_results")
OUT.mkdir(parents=True, exist_ok=True)

# ── Quick significance from per-example competitor data ──
print("STEP R1: Significance tests from competitor per-example data")
comp_rows = json.loads(Path("llm_memory_validation/competitor_run_v2/retrieval_rows.json").read_text(encoding="utf-8"))
cf_summary = json.loads(Path("llm_memory_validation/counterfactual_utility_regressor_run/summary.json").read_text(encoding="utf-8"))
fast = json.loads(Path("llm_memory_validation/neurips_fast_results/all_results.json").read_text(encoding="utf-8"))

# Build per-example recall dictionaries
def per_example_recall(rows_dict):
    result = {}
    for method, rows in rows_dict.items():
        recalls = {}
        for row in rows:
            gold = set(row.get("gold_session_ids", row.get("answer_session_ids", [])))
            pred = row["predicted_session_ids"][:5]
            recalls[row["question_id"]] = len(set(pred) & gold) / max(len(gold), 1)
        result[method] = recalls
    return result

comp_recalls = per_example_recall(comp_rows)

# Find common question IDs across methods
all_qids = None
for method in comp_recalls:
    if all_qids is None:
        all_qids = set(comp_recalls[method].keys())
    else:
        all_qids &= set(comp_recalls[method].keys())
print(f"  Common question IDs across methods: {len(all_qids)}")

# Significance tests
rng = np.random.default_rng(42)
pairs = [
    ("heuristic_bsc", "dense_rag_e5", "Heuristic BSC vs Dense RAG"),
    ("dense_budgeted_bsc", "fifo_replay", "Dense BSC vs FIFO"),
    ("heuristic_bsc", "memorybank_proxy", "BSC vs MemoryBank"),
    ("heuristic_bsc", "ld_agent_proxy", "BSC vs LD-Agent"),
    ("dense_budgeted_bsc", "dense_rag_e5", "Dense BSC vs Dense RAG"),
]

sig_results = {}
for ma, mb, label in pairs:
    if ma in comp_recalls and mb in comp_recalls:
        common = all_qids
        ra = np.array([comp_recalls[ma].get(qid, 0) for qid in common])
        rb = np.array([comp_recalls[mb].get(qid, 0) for qid in common])
        diffs = ra - rb
        obs = float(np.mean(diffs))
        boot = np.array([float(np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))])) for _ in range(10000)])
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        p = float(min(np.mean(boot <= 0) * 2, 1.0))
        sig_results[label] = {"diff": obs, "ci_95": ci, "p": p, "sig_005": p < 0.05}
        print(f"  {label}: diff={obs:+.4f}, CI=[{ci[0]:.4f},{ci[1]:.4f}], p={p:.6f}, significant={'YES' if p<0.05 else 'no'}")

# ── Action distribution by question type ──
print("\nSTEP R2: Action distribution by budget")
from llm_memory_validation.bsc_longmemeval import load_dataset, classify_action
examples = load_dataset()
# Heuristic action distribution is budget-invariant (classify_action doesn't use budget)
actions = Counter()
for ex in examples:
    total = len(ex["haystack_sessions"])
    for i, session in enumerate(ex["haystack_sessions"]):
        a = classify_action(session, i, total)
        actions[a] += 1
tot = sum(actions.values())
action_frac = {a: actions[a]/tot for a in ["discard","replay","cache","consolidate"]}
print(f"  Heuristic actions: discard={actions['discard']/tot:.1%} replay={actions['replay']/tot:.1%} cache={actions['cache']/tot:.1%} consol={actions['consolidate']/tot:.1%}")
# Oracle uses different actions at different budgets — note this from counterfactual data
# At 20%: 96% discard, 3.9% consolidate, 0% replay, 0.02% cache

# ── Compute heuristic vs oracle action agreement ──
print("\nSTEP R3: Heuristic vs Oracle action agreement")
from llm_memory_validation.bsc_longmemeval import build_bsc
heuristic_actions = Counter()
for ex in examples:
    entries = build_bsc(ex, 0.20)
    for e in entries:
        heuristic_actions[e.action] += 1
total_h = sum(heuristic_actions.values())
print(f"  Heuristic: {dict(heuristic_actions)}")
print(f"  Fractions: { {a: heuristic_actions[a]/total_h for a in ['replay','cache','consolidate']} }")

oracle_actions = fast["label_collapse"]["distribution"]
total_o = sum(oracle_actions.values())
oracle_fractions = {a: oracle_actions.get(a, 0)/total_o for a in ["discard","replay","cache","consolidate"]}
print(f"  Oracle: {oracle_fractions}")

# ── Save everything ──
results = {
    "significance_competitor": sig_results,
    "action_distribution": {"heuristic_frac": action_frac, "heuristic_counts": dict(actions), "oracle_frac": oracle_fractions},
    "heuristic_action_counts": dict(heuristic_actions),
    "oracle_action_fractions": oracle_fractions,
}

# Add fast theory results too
theory = json.loads(Path("llm_memory_validation/neurips_fast_results/theory_robustness.json").read_text(encoding="utf-8"))
results["additivity"] = theory["additivity"]
results["diminishing_returns"] = {k: v for k, v in theory["diminishing_returns"].items() if k != "avg_by_position"}

(OUT / "significance_and_actions.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\nResults saved to {OUT / 'significance_and_actions.json'}")