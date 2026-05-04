"""Generate MemAudit paper figures from canonical artifacts.

The script is intentionally dependency-light: matplotlib plus the Python
standard library. It reads existing run summaries and writes vector PDF/SVG
figures to the figures/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

PALETTE = {
    "oracle": "#2563EB",
    "opt": "#0F172A",
    "full_raw": "#F97316",
    "density": "#F59E0B",
    "no_tombstone": "#E11D48",
    "fact": "#10B981",
    "summary": "#8B5CF6",
    "recency": "#64748B",
    "fifo": "#64748B",
    "light": "#FFF7ED",
    "mid": "#67E8F9",
    "dark": "#1E3A8A",
    "mem0": "#F97316",
    "letta": "#0F9F8F",
    "amem": "#64748B",
    "paper": "#FFF8F0",
    "grid": "#E8D9C7",
    "ink": "#111827",
    "muted": "#6B5F53",
    "success": "#22C55E",
}

HEATMAP = LinearSegmentedColormap.from_list(
    "memaudit_heat",
    ["#FFF8F0", "#FDE68A", "#67E8F9", "#2563EB", "#1E3A8A"],
)

METHOD_LABELS = {
    "oracle_gvt": "MemAudit-GVT",
    "density_only": "Density-only",
    "no_tombstone_opt": "No-tombstone OPT",
    "fact_only": "Fact-only",
    "summary_only": "Summary-only",
    "recency_raw": "Recency raw",
    "mem0_extract": "Mem0-extract",
    "estimated_gvt": "Estimated-GVT",
    "dense_budgeted_bsc": "MemAudit + dense",
    "dense_rag_e5": "Full raw dense",
    "dense_budgeted_replay": "Budgeted raw replay",
    "fifo_replay": "FIFO",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": PALETTE["paper"],
        "axes.edgecolor": "#D6C7B5",
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_json(path: str | Path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D6C7B5")
    ax.spines["bottom"].set_color("#D6C7B5")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(color="#B8A898", labelcolor=PALETTE["ink"])


def theme_legend(ax, **kwargs):
    legend = ax.legend(frameon=False, **kwargs)
    return legend


def add_box(ax, xy, w, h, text, color, fontsize=9):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p1, p2, color=None):
    color = color or PALETTE["ink"]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, lw=1.2, color=color))


def pipeline_schematic():
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.text(0.25, 3.73, "Frozen package construction", fontsize=9, fontweight="bold", color=PALETTE["muted"])
    ax.add_patch(Rectangle((0.18, 2.95), 10.0, 0.62, facecolor="#FFEFD8", edgecolor="#F0C98E", lw=0.8))
    ax.text(
        5.2,
        3.26,
        r"candidate memories + costs $c$ + evidence coverage $A$ + query weights $w$ + one-choice groups $G_i$",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["muted"],
    )

    add_box(ax, (0.35, 1.86), 1.32, 0.58, "Experience\nstream", PALETTE["recency"], fontsize=8.5)
    add_box(ax, (2.05, 1.86), 1.55, 0.58, "Candidate\nrepresentations", "#CC79A7", fontsize=8.5)
    add_box(ax, (4.02, 1.86), 1.34, 0.58, "Budgeted\nwriter", PALETTE["oracle"], fontsize=8.5)
    add_box(ax, (5.92, 1.86), 1.28, 0.58, "Written\nstore X", PALETTE["oracle"], fontsize=8.5)
    arrow(ax, (1.70, 2.15), (2.02, 2.15))
    arrow(ax, (3.63, 2.15), (3.99, 2.15))
    arrow(ax, (5.38, 2.15), (5.89, 2.15))

    ax.text(2.82, 1.58, "raw / fact / summary / tombstone / compound", ha="center", fontsize=7.6)
    ax.text(4.70, 1.58, "one representation\nper experience", ha="center", va="top", fontsize=7.3, color=PALETTE["muted"])

    add_box(ax, (7.80, 2.50), 1.62, 0.60, "MemAudit\nscorer", PALETTE["opt"], fontsize=8.5)
    add_box(ax, (7.80, 1.52), 1.62, 0.60, "Retriever +\nreader", "#009E73", fontsize=8.5)
    add_box(ax, (7.80, 0.56), 1.62, 0.60, "Downstream\nanswer metrics", "#009E73", fontsize=8.5)

    ax.add_patch(FancyArrowPatch((6.58, 2.44), (7.77, 2.82), arrowstyle="->", mutation_scale=12, lw=1.3, color=PALETTE["opt"]))
    ax.add_patch(FancyArrowPatch((6.58, 1.86), (7.77, 1.82), arrowstyle="->", mutation_scale=12, lw=1.3, color=PALETTE["letta"]))
    arrow(ax, (8.61, 1.50), (8.61, 1.18), PALETTE["letta"])
    ax.text(9.54, 2.82, r"$F(X)/\mathrm{OPT}$", ha="left", va="center", fontsize=9, color=PALETTE["opt"])
    ax.text(9.54, 2.58, "write-time ratio\nbefore retrieval", ha="left", va="top", fontsize=7.2, color=PALETTE["muted"])
    ax.plot([5.2, 8.62], [2.95, 2.95], color="#E2BE82", lw=0.8, ls=":")
    ax.text(
        4.45,
        0.82,
        "MemAudit scores write quality before\noptional retrieval and reader evaluation.",
        ha="center",
        fontsize=8.0,
        color=PALETTE["muted"],
    )
    save(fig, "pipeline_schematic")


def tombstone_timeline():
    fig, ax = plt.subplots(figsize=(8.7, 3.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.text(0.25, 3.15, "Experience timeline", fontsize=8, fontweight="bold", color=PALETTE["muted"])
    ax.plot([1, 9.2], [2.88, 2.88], color=PALETTE["ink"], lw=1.5)
    for x, label in [(2, "t1"), (6, "t2"), (8.7, "future query")]:
        ax.plot([x, x], [2.78, 2.98], color=PALETTE["ink"], lw=1)
        ax.text(x, 3.05, label, ha="center", fontsize=8)
    ax.text(2, 2.54, '"I prefer vegetarian\nmeals for travel."', ha="center", fontsize=7.7)
    ax.text(6, 2.54, '"Actually, I am\npescatarian now."', ha="center", fontsize=7.7)
    ax.text(8.7, 2.54, "What meals\nshould we book?", ha="center", fontsize=7.7)

    ax.text(0.25, 2.08, "Candidate writes", fontsize=8, fontweight="bold", color=PALETTE["muted"])
    add_box(ax, (0.65, 1.62), 1.75, 0.48, "Stale fact\ntravel = vegetarian", PALETTE["mem0"], fontsize=7.7)
    add_box(ax, (3.00, 1.62), 1.75, 0.48, "Current fact\ntravel = pescatarian", PALETTE["letta"], fontsize=7.7)
    add_box(ax, (5.25, 1.62), 1.90, 0.48, "Tombstone\nvegetarian invalid", PALETTE["no_tombstone"], fontsize=7.7)
    add_box(ax, (7.55, 1.62), 1.85, 0.48, "Compound update\nnew fact + invalidation", PALETTE["oracle"], fontsize=7.4)
    arrow(ax, (4.78, 1.86), (5.20, 1.86), PALETTE["ink"])
    arrow(ax, (7.18, 1.86), (7.50, 1.86), PALETTE["ink"])

    ax.text(0.25, 1.12, "Evidence units covered", fontsize=8, fontweight="bold", color=PALETTE["muted"])
    add_box(ax, (1.00, 0.54), 1.60, 0.42, "old preference\n(provenance)", PALETTE["full_raw"], fontsize=7)
    add_box(ax, (3.30, 0.54), 1.60, 0.42, "current truth", PALETTE["letta"], fontsize=7)
    add_box(ax, (5.45, 0.54), 1.65, 0.42, "invalidation", PALETTE["no_tombstone"], fontsize=7)
    add_box(ax, (7.70, 0.54), 1.55, 0.42, "temporal order", PALETTE["oracle"], fontsize=7)
    ax.text(
        5.0,
        0.13,
        "The package can reward current-state and invalidation evidence without penalizing raw old memories directly.",
        ha="center",
        fontsize=7.8,
        color=PALETTE["muted"],
    )
    save(fig, "tombstone_timeline")


def exact_budget_sweep():
    data = load_json("oraclemem_runs/exact_500/summary.json")
    rows = data["by_budget_method"]
    budgets = [2, 4, 8, 16]
    methods = ["oracle_gvt", "density_only", "no_tombstone_opt", "fact_only", "summary_only", "recency_raw"]
    colors = {
        "oracle_gvt": PALETTE["oracle"],
        "density_only": PALETTE["density"],
        "no_tombstone_opt": PALETTE["no_tombstone"],
        "fact_only": PALETTE["fact"],
        "summary_only": PALETTE["summary"],
        "recency_raw": PALETTE["recency"],
    }
    lookup = {(r["budget"], r["method"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for method in methods:
        ys = [lookup[(b, method)]["mean_ratio_to_opt"] for b in budgets]
        lows = [lookup[(b, method)]["bootstrap95_ratio_to_opt_low"] for b in budgets]
        highs = [lookup[(b, method)]["bootstrap95_ratio_to_opt_high"] for b in budgets]
        ax.plot(budgets, ys, marker="o", lw=2, label=METHOD_LABELS[method], color=colors[method])
        ax.fill_between(budgets, lows, highs, color=colors[method], alpha=0.12, linewidth=0)
    ax.axhline(1.0, color=PALETTE["opt"], lw=1, ls="--", label="Exact OPT")
    ax.set_xlabel("Storage budget B")
    ax.set_ylabel("Ratio to exact OPT")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(budgets)
    style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    save(fig, "exact_budget_sweep")


def stress_heatmap_and_gap():
    data = load_json("oraclemem_runs/stress_exact_500/summary.json")
    rows = data["by_distribution_budget_method"]
    dists = ["base", "update_chain", "temporal_interval"]
    methods = ["oracle_gvt", "density_only", "no_tombstone_opt"]
    lookup = {(r["distribution"], r["budget"], r["method"]): r for r in rows}
    budget = 6
    vals = [[lookup[(d, budget, m)]["mean_ratio_to_opt"] for d in dists] for m in methods]

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    im = ax.imshow(vals, cmap=HEATMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(dists)), ["Base", "Update\nchain", "Temporal\ninterval"])
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
    for i, row in enumerate(vals):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=PALETTE["ink"])
    ax.set_title("Validity-heavy stress suite (B=6)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Ratio to OPT")
    save(fig, "stress_heatmap")

    gaps = [1.0 - lookup[(d, budget, "no_tombstone_opt")]["mean_ratio_to_opt"] for d in dists]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(["Base", "Update\nchain", "Temporal\ninterval"], gaps, color=PALETTE["no_tombstone"], alpha=0.85)
    ax.set_ylabel("Full OPT - no-tombstone OPT")
    ax.set_ylim(0, max(gaps) * 1.25)
    style_axes(ax)
    for i, val in enumerate(gaps):
        ax.text(i, val + 0.015, f"{val:.3f}", ha="center", fontsize=9)
    save(fig, "validity_frontier_gap")


def longmemeval_retrieval():
    data = load_json("llm_memory_validation/longmemeval_focus_report_core4/summary.json")
    methods = ["dense_budgeted_bsc", "dense_rag_e5", "dense_budgeted_replay", "fifo_replay"]
    colors = {
        "dense_budgeted_bsc": PALETTE["oracle"],
        "dense_rag_e5": PALETTE["full_raw"],
        "dense_budgeted_replay": PALETTE["density"],
        "fifo_replay": PALETTE["fifo"],
    }
    ks = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for method in methods:
        metrics = data["metrics"][method]
        ys = [metrics[f"focus_recall_at_{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", lw=2, label=METHOD_LABELS[method], color=colors[method])
        offsets = {
            "dense_budgeted_bsc": 0.000,
            "dense_rag_e5": -0.005,
            "dense_budgeted_replay": -0.025,
            "fifo_replay": 0.020,
        }
        ax.text(5.08, ys[-1] + offsets[method], METHOD_LABELS[method], va="center", fontsize=7.5, color=colors[method])
    ax.set_xlabel("k")
    ax.set_ylabel("Focus recall@k")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.9, 5.9)
    ax.set_xticks(ks)
    style_axes(ax)
    save(fig, "longmemeval_retrieval_rk")


def gpt55_reader_bars():
    data = load_json("llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json")
    methods = ["dense_budgeted_bsc", "dense_rag_e5", "dense_budgeted_replay", "fifo_replay"]
    labels = ["MemAudit\n+dense", "Full raw\ndense", "Budgeted\nraw", "FIFO"]
    metrics = [
        ("token_f1", "Token F1", True),
        ("evidence_use", "Evidence use", True),
        ("insufficient_evidence_rate", "Insufficient", False),
    ]
    colors = [PALETTE["oracle"], PALETTE["full_raw"], PALETTE["density"], PALETTE["fifo"]]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), sharey=True)
    for ax, (key, title, _) in zip(axes, metrics):
        vals = [data["metrics"][m]["focus"][key] for m in methods]
        ax.bar(range(len(methods)), vals, color=colors, alpha=0.9)
        ax.set_title(title, fontsize=10)
        for i, val in enumerate(vals):
            ax.text(i, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(methods)), labels, rotation=0, ha="center", fontsize=7)
        ax.set_ylim(0, 1.0)
        style_axes(ax)
    axes[0].set_ylabel("Rate")
    save(fig, "gpt55_reader_bars")


def longmemeval_diagnostics():
    """Combined appendix figure for LongMemEval retrieval and frozen-reader behavior."""

    retrieval = load_json("llm_memory_validation/longmemeval_focus_report_core4/summary.json")
    reader = load_json("llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json")
    methods = ["dense_budgeted_bsc", "dense_rag_e5", "dense_budgeted_replay", "fifo_replay"]
    labels = ["MemAudit+dense", "Full raw dense", "Budgeted raw", "FIFO"]
    colors = {
        "dense_budgeted_bsc": PALETTE["oracle"],
        "dense_rag_e5": PALETTE["full_raw"],
        "dense_budgeted_replay": PALETTE["density"],
        "fifo_replay": PALETTE["fifo"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    ks = [1, 3, 5]
    for method in methods:
        ys = [retrieval["metrics"][method][f"focus_recall_at_{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", lw=2, color=colors[method])
        offsets = {
            "dense_budgeted_bsc": 0.000,
            "dense_rag_e5": -0.005,
            "dense_budgeted_replay": -0.025,
            "fifo_replay": 0.020,
        }
        ax.text(5.10, ys[-1] + offsets[method], METHOD_LABELS[method], va="center", fontsize=7.1, color=colors[method])
    ax.set_title("Retrieval evidence recall")
    ax.set_xlabel("Retrieved memories k")
    ax.set_ylabel("Focus recall@k")
    ax.set_xlim(0.9, 6.2)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ks)
    style_axes(ax)

    ax = axes[1]
    metric_defs = [
        ("token_f1", "Token F1", PALETTE["oracle"]),
        ("evidence_use", "Evidence use", PALETTE["letta"]),
        ("insufficient_evidence_rate", "Insufficient", PALETTE["no_tombstone"]),
    ]
    x = list(range(len(labels)))
    width = 0.24
    for offset, (metric, label, color) in zip([-width, 0.0, width], metric_defs):
        vals = [reader["metrics"][m]["focus"][metric] for m in methods]
        ax.bar([i + offset for i in x], vals, width, label=label, color=color, alpha=0.92)
    ax.set_title("Frozen-reader diagnostic")
    ax.set_ylabel("Rate")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    save(fig, "longmemeval_diagnostics")


def _mean_ratio_by_method(rows, method, budget, key):
    for row in rows:
        if row.get("method") == method and row.get("budget") == budget:
            return row[key]
    raise KeyError(f"Missing method={method} budget={budget} key={key}")


def system_diagnostic():
    """Visualize exported-system scores from the natural adjudicated subset."""

    natural = load_json("llm_memory_validation/natural_adjudicated_100_gemini_flash/summary.json")
    mem0 = load_json("llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash/summary.json")
    letta = load_json(
        "llm_memory_validation/natural_adjudicated_100_gemini_flash/"
        "actual_letta_openrouter_gemini_passage_87/summary.json"
    )
    amem = load_json(
        "llm_memory_validation/natural_adjudicated_100_gemini_flash/"
        "actual_amem_gemini_flash_87/summary.json"
    )

    budgets = [30, 60, 100]
    series = [
        (
            "MemAudit-GVT\n(package)",
            [_mean_ratio_by_method(natural["by_budget_method"], "oracle_gvt", b, "mean_ratio_to_opt") for b in budgets],
            PALETTE["oracle"],
            "o",
            "-",
        ),
        (
            "Estimated-GVT\n(package)",
            [_mean_ratio_by_method(natural["by_budget_method"], "estimated_gvt", b, "mean_ratio_to_opt") for b in budgets],
            PALETTE["summary"],
            "P",
            "-",
        ),
        (
            "Letta salience\n(union)",
            [
                _mean_ratio_by_method(
                    letta["summary_rows"], "actual_letta_combined_salience_pruned", b, "mean_ratio_to_union_opt"
                )
                for b in budgets
            ],
            PALETTE["letta"],
            "s",
            "-",
        ),
        (
            "Letta upper\n(union)",
            [
                _mean_ratio_by_method(
                    letta["summary_rows"], "actual_letta_combined_oracle_pruned_upper", b, "mean_ratio_to_union_opt"
                )
                for b in budgets
            ],
            PALETTE["letta"],
            "s",
            "--",
        ),
        (
            "Mem0 salience\n(union)",
            [
                _mean_ratio_by_method(
                    mem0["summary_rows"], "actual_mem0_salience_pruned", b, "mean_ratio_to_union_opt"
                )
                for b in budgets
            ],
            PALETTE["mem0"],
            "^",
            "-",
        ),
        (
            "Mem0 upper\n(union)",
            [
                _mean_ratio_by_method(
                    mem0["summary_rows"], "actual_mem0_oracle_pruned_upper", b, "mean_ratio_to_union_opt"
                )
                for b in budgets
            ],
            PALETTE["mem0"],
            "^",
            "--",
        ),
        (
            "A-Mem metadata\n(union)",
            [
                _mean_ratio_by_method(
                    amem["by_method_budget"], "actual_amem_metadata_recency_pruned", b, "mean_ratio_to_union_opt"
                )
                for b in budgets
            ],
            PALETTE["amem"],
            "D",
            "-",
        ),
        (
            "A-Mem full store\n(union)",
            [
                _mean_ratio_by_method(
                    amem["by_method_budget"], "actual_amem_full_native_retrieval_pruned", b, "mean_ratio_to_union_opt"
                )
                for b in budgets
            ],
            PALETTE["amem"],
            "x",
            ":",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.35), gridspec_kw={"width_ratios": [1.62, 1.0]})
    ax = axes[0]
    for label, ys, color, marker, linestyle in series:
        ax.plot(
            budgets,
            ys,
            marker=marker,
            lw=2.0,
            ms=4.5,
            color=color,
            linestyle=linestyle,
            label=label,
            alpha=0.95,
        )
    endpoint_offsets = {
        "MemAudit-GVT\n(package)": 0.000,
        "Mem0 upper\n(union)": 0.045,
        "Estimated-GVT\n(package)": -0.045,
        "Letta upper\n(union)": 0.022,
        "Letta salience\n(union)": -0.020,
        "Mem0 salience\n(union)": 0.000,
        "A-Mem metadata\n(union)": 0.018,
        "A-Mem full store\n(union)": 0.026,
    }
    endpoint_labels = {
        "MemAudit-GVT\n(package)": "GVT",
        "Estimated-GVT\n(package)": "Est. GVT",
        "Letta salience\n(union)": "Letta sel.",
        "Letta upper\n(union)": "Letta upper",
        "Mem0 salience\n(union)": "Mem0 sel.",
        "Mem0 upper\n(union)": "Mem0 upper",
        "A-Mem metadata\n(union)": "A-Mem meta",
        "A-Mem full store\n(union)": "A-Mem full",
    }
    for label, ys, color, _, linestyle in series:
        label_text = endpoint_labels[label]
        ax.text(
            104.0,
            max(0.0, min(1.03, ys[-1] + endpoint_offsets.get(label, 0.0))),
            label_text,
            color=color,
            fontsize=7,
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.82),
            clip_on=False,
        )
    ax.set_title("Exported stores under the same budget")
    ax.set_xlabel("Storage budget B")
    ax.set_ylabel("Ratio to exact package/union OPT")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(28, 126)
    ax.set_xticks(budgets)
    style_axes(ax)

    ax = axes[1]
    selected = [
        _mean_ratio_by_method(mem0["summary_rows"], "actual_mem0_salience_pruned", 100, "mean_ratio_to_union_opt"),
        _mean_ratio_by_method(letta["summary_rows"], "actual_letta_combined_salience_pruned", 100, "mean_ratio_to_union_opt"),
        _mean_ratio_by_method(amem["by_method_budget"], "actual_amem_metadata_recency_pruned", 100, "mean_ratio_to_union_opt"),
        _mean_ratio_by_method(amem["by_method_budget"], "actual_amem_full_native_retrieval_pruned", 100, "mean_ratio_to_union_opt"),
    ]
    upper = [
        _mean_ratio_by_method(mem0["summary_rows"], "actual_mem0_oracle_pruned_upper", 100, "mean_ratio_to_union_opt"),
        _mean_ratio_by_method(letta["summary_rows"], "actual_letta_combined_oracle_pruned_upper", 100, "mean_ratio_to_union_opt"),
        _mean_ratio_by_method(amem["by_method_budget"], "actual_amem_metadata_oracle_pruned_upper", 100, "mean_ratio_to_union_opt"),
        _mean_ratio_by_method(amem["by_method_budget"], "actual_amem_full_oracle_pruned_upper", 100, "mean_ratio_to_union_opt"),
    ]
    labels = ["Mem0", "Letta", "A-Mem\nmetadata", "A-Mem\nfull"]
    x = list(range(len(labels)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], selected, width, label="Selected/pruned", color=PALETTE["oracle"], alpha=0.92)
    ax.bar([i + width / 2 for i in x], upper, width, label="Upper bound", color=PALETTE["density"], alpha=0.92)
    for i, (sel, up) in enumerate(zip(selected, upper)):
        ax.text(i - width / 2, sel + 0.025, f"{sel:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width / 2, up + 0.025, f"{up:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("B=100: extraction vs selection")
    ax.set_ylabel("Union ratio")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.subplots_adjust(bottom=0.24, wspace=0.40)
    save(fig, "system_diagnostic")


def conditional_failure_audit():
    data = load_json("llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/failure_bucket_counts.json")
    methods = ["dense_budgeted_bsc", "dense_rag_e5", "dense_budgeted_replay", "fifo_replay"]
    method_labels = ["MemAudit + dense", "Full raw dense", "Budgeted raw", "FIFO"]
    buckets = [
        (["missing_gold_evidence"], "Missing retrieved\nevidence", PALETTE["full_raw"]),
        (["abstained_despite_gold"], "Abstained despite\nsupport", PALETTE["density"]),
        (["used_gold_but_wrong", "unsupported_answer"], "Reader/error or\nunsupported", PALETTE["no_tombstone"]),
        (["scoring_mismatch_possible", "parse_failure"], "Scoring/parse\nuncertain", PALETTE["mid"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    lefts = [0.0] * len(methods)
    for keys, label, color in buckets:
        vals = []
        for m in methods:
            row = data["by_method"][m]
            vals.append(sum(row["outcome_counts"].get(k, 0) for k in keys) / row["n"])
        ax.barh(range(len(methods)), vals, left=lefts, label=label, color=color, alpha=0.92)
        lefts = [b + v for b, v in zip(lefts, vals)]
    residual = [max(0.0, 1.0 - v) for v in lefts]
    ax.barh(range(len(methods)), residual, left=lefts, label="No flagged\nfailure", color=PALETTE["success"], alpha=0.55)
    for i, (fail, ok) in enumerate(zip(lefts, residual)):
        if ok <= 0:
            continue
        x = max(0.025, min(0.99, fail + ok / 2.0))
        ax.text(
            x,
            i,
            f"{ok:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.78),
        )
    ax.set_yticks(range(len(methods)), method_labels)
    ax.set_xlabel("Share of LongMemEval-S focus questions")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.25)
    save(fig, "conditional_failure_audit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print inputs/outputs without generating figures.")
    args = parser.parse_args()
    inputs = [
        "oraclemem_runs/exact_500/summary.json",
        "oraclemem_runs/stress_exact_500/summary.json",
        "llm_memory_validation/longmemeval_focus_report_core4/summary.json",
        "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json",
        "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/failure_bucket_counts.json",
        "llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash/summary.json",
        "llm_memory_validation/natural_adjudicated_100_gemini_flash/summary.json",
        "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/summary.json",
        "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/summary.json",
    ]
    outputs = [
        "pipeline_schematic",
        "tombstone_timeline",
        "exact_budget_sweep",
        "stress_heatmap",
        "validity_frontier_gap",
        "longmemeval_retrieval_rk",
        "gpt55_reader_bars",
        "longmemeval_diagnostics",
        "system_diagnostic",
        "conditional_failure_audit",
    ]
    if args.dry_run:
        print("Inputs:")
        for item in inputs:
            print(f"  {item}")
        print("Outputs:")
        for item in outputs:
            print(f"  figures/{item}.pdf")
            print(f"  figures/{item}.svg")
        return

    pipeline_schematic()
    tombstone_timeline()
    exact_budget_sweep()
    stress_heatmap_and_gap()
    longmemeval_retrieval()
    gpt55_reader_bars()
    longmemeval_diagnostics()
    system_diagnostic()
    conditional_failure_audit()
    print(f"Wrote {len(outputs) * 2} vector figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
