"""Evaluate package-candidate memory writers under exact OracleMem denominators.

This is the no-new-API path for denominator-matched writer comparisons on an
existing coverage package. It loads a finite OracleMem package, evaluates local
writer adapters such as Letta/MemGPT-style tiering and A-Mem-style graph memory,
and reports exact ratios to the package OPT for each query.

The adapters operate only on visible candidate metadata. They do not call the
published systems and should be reported as faithful/local adapters, not as
full production-system executions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oraclemem.evaluate import evaluate_instance, write_benchmark_outputs
from oraclemem.writer_baselines import WRITER_BASELINE_DESCRIPTIONS

from llm_memory_validation.evaluate_human_style_examples import parse_tokens
from llm_memory_validation.run_mem0_natural_baseline import (
    load_package,
    package_instance,
    resolved_queries,
    write_json,
)


DEFAULT_METHODS = (
    "opt",
    "oracle_gvt",
    "memgpt_tiered",
    "amem_graph",
    "mem0_extract",
    "amac_admission",
    "estimated_gvt",
    "density_only",
    "summary_only",
    "fact_only",
    "recency_raw",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package"),
        help="Existing OracleMem coverage package directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters"),
        help="Output directory.",
    )
    parser.add_argument(
        "--budgets",
        default="30,60,100",
        help="Comma or space separated integer budgets.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma or space separated method ids.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solver", default="exact_stdlib")
    return parser


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _by_budget_method(summary: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    rows: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in summary.get("by_budget_method", []):
        rows[(int(row["budget"]), str(row["method"]))] = row
    return rows


def write_report(
    out_dir: Path,
    *,
    package_dir: Path,
    query_count: int,
    methods: Sequence[str],
    budgets: Sequence[int],
    summary: Mapping[str, Any],
) -> None:
    by_key = _by_budget_method(summary)
    lines = [
        "# Coverage-Package Writer Adapter Report",
        "",
        f"- Package: `{package_dir}`",
        f"- Queries evaluated: {query_count}",
        f"- Budgets: `{','.join(str(budget) for budget in budgets)}`",
        "- Denominator: exact package OPT over the finite coverage package.",
        "- API calls: none.",
        "",
        "## Claim Boundary",
        "",
        "- These rows evaluate visible-metadata writer adapters under the same package denominator.",
        "- `memgpt_tiered` is a Letta/MemGPT-style archival/recency adapter, not a Letta server run.",
        "- `amem_graph` is an A-Mem-style graph/evolving-memory adapter, not the published A-Mem pipeline.",
        "- Local reference repos present in this workspace: `external_repos/letta` and `external_repos/AgenticMemory`.",
        "",
        "## Adapter Provenance",
        "",
    ]
    for method in methods:
        description = WRITER_BASELINE_DESCRIPTIONS.get(method)
        if not description:
            continue
        lines.append(f"- `{method}`: {_sentence(description.get('proxy_for', 'local adapter'))}")
        lines.append(f"  Decision features: {_sentence(description.get('decision_features', 'visible metadata'))}")
        lines.append(f"  Limitation: {_sentence(description.get('limitation', 'local adapter only'))}")
    lines.extend(["", "## Mean Ratio To Exact Package OPT", ""])
    header = "| Method | " + " | ".join(f"B={budget}" for budget in budgets) + " |"
    sep = "| --- | " + " | ".join("---" for _ in budgets) + " |"
    lines.extend([header, sep])
    for method in methods:
        cells = []
        for budget in budgets:
            row = by_key.get((budget, method))
            if row is None:
                cells.append("--")
                continue
            value = row.get("mean_ratio_to_opt")
            cells.append("--" if value is None else f"{float(value):.3f}")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else f"{text}."


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budgets = tuple(int(token) for token in parse_tokens(args.budgets))
    methods = parse_tokens(args.methods)

    data = load_package(args.package_dir)
    queries = resolved_queries(data, args.limit)
    results = []
    for query in queries:
        instance = package_instance(data, query)
        results.extend(
            evaluate_instance(
                instance,
                budgets,
                methods=methods,
                solver=args.solver,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_benchmark_outputs(results, args.out_dir)
    summary = json.loads((args.out_dir / "summary.json").read_text(encoding="utf-8"))
    write_report(
        args.out_dir,
        package_dir=args.package_dir,
        query_count=len(queries),
        methods=methods,
        budgets=budgets,
        summary=summary,
    )
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "package_dir": str(args.package_dir),
            "out_dir": str(args.out_dir),
            "query_count": len(queries),
            "budgets": list(budgets),
            "methods": list(methods),
            "denominator": "exact_package_opt",
            "api_calls": 0,
            **paths,
        },
    )
    print(json.dumps({"queries": len(queries), **paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
