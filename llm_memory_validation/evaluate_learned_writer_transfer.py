"""Train a non-oracle utility writer and evaluate it on natural packages.

This is the deployable-writer diagnostic for OracleMem. Training may use oracle
coverage labels on train packages, but test-time selection uses only visible
candidate metadata through ``EstimatedUtilityModel.predict``. The reported
ratios are still scored against exact finite-package optima.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oraclemem.evaluate import (
    LEARNED_ESTIMATOR_PROFILE,
    LOCAL_LEARNED_ESTIMATOR_MODEL,
    OracleMemInstance,
    aggregate_results,
    evaluate_instance,
    generate_named_distribution,
    objective_value,
    train_feature_utility_estimator,
)

from llm_memory_validation.evaluate_human_style_examples import (
    build_instance as build_human_instance,
    evaluate_human_package,
    load_examples,
    parse_tokens,
)
from llm_memory_validation.run_mem0_natural_baseline import (
    load_package,
    package_instance,
    resolved_queries,
    write_json,
)


DEFAULT_METHODS = (
    "opt",
    "oracle_gvt",
    "estimated_gvt",
    "estimated_utility",
    "memgpt_tiered",
    "amem_graph",
    "amac_admission",
    "mem0_extract",
    "density_only",
    "greedy",
    "fact_only",
    "summary_only",
    "recency_raw",
    "no_tombstone_opt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a visible-feature OracleMem utility estimator on synthetic "
            "and model-annotated natural packages, then test on a human-edited "
            "finite package with exact OPT scoring."
        )
    )
    parser.add_argument(
        "--human-examples-jsonl",
        default="llm_memory_validation/human_style_examples/examples_100.jsonl",
        help="Human-edited JSONL package used for held-out evaluation.",
    )
    parser.add_argument(
        "--train-natural-package-dir",
        action="append",
        default=["llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_package"],
        help=(
            "Natural coverage package directory to use for train labels. "
            "Can be supplied multiple times. Defaults to Natural-200."
        ),
    )
    parser.add_argument(
        "--train-natural-limit",
        type=int,
        default=None,
        help="Optional per-package cap on natural train queries.",
    )
    parser.add_argument(
        "--natural-train-weight",
        type=int,
        default=1,
        help=(
            "Integer replication weight for allowed natural train instances. "
            "This changes estimator fitting only; manifests report weighted and "
            "unweighted counts."
        ),
    )
    parser.add_argument(
        "--tune-natural-train-weight",
        action="store_true",
        help=(
            "Choose natural-train weight and ridge from train-only validation "
            "labels before fitting the final estimator."
        ),
    )
    parser.add_argument(
        "--candidate-natural-train-weights",
        default="1,2,3,5,8,10,15,20,30,50",
        help="Comma or space separated natural weights for train-only tuning.",
    )
    parser.add_argument(
        "--candidate-ridges",
        default="0.05,0.25,1.0,2.0",
        help="Comma or space separated ridge values for train-only tuning.",
    )
    parser.add_argument(
        "--validation-natural-stride",
        type=int,
        default=5,
        help="Use every Nth allowed natural train instance as train-only validation.",
    )
    parser.add_argument(
        "--validation-synthetic-fraction",
        type=float,
        default=0.20,
        help="Fraction of synthetic train seeds reserved for train-only validation.",
    )
    parser.add_argument(
        "--validation-synthetic-budgets",
        default="4,6",
        help="Synthetic validation budgets used only for hyperparameter selection.",
    )
    parser.add_argument(
        "--validation-natural-budgets",
        default="30,60,100",
        help="Natural validation budgets used only for hyperparameter selection.",
    )
    parser.add_argument(
        "--n-synthetic-train-seeds",
        type=int,
        default=200,
        help="Use synthetic train seeds 0..N-1. Set 0 to disable synthetic train data.",
    )
    parser.add_argument(
        "--synthetic-distributions",
        default="base,update_chain,temporal_interval,scope_shift_v2,density_trap_v2",
        help="Comma or space separated synthetic train distributions.",
    )
    parser.add_argument(
        "--normal-count",
        type=int,
        default=3,
        help="Synthetic normal fact count.",
    )
    parser.add_argument(
        "--update-count",
        type=int,
        default=2,
        help="Synthetic update/tombstone pair count.",
    )
    parser.add_argument(
        "--budgets",
        default="150,300,600,1000",
        help="Comma or space separated held-out test budgets.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma or space separated evaluation methods.",
    )
    parser.add_argument(
        "--eval-coverage-package-dir",
        action="append",
        default=["llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package"],
        help=(
            "Held-out coverage package directory to evaluate with exact package OPT. "
            "Can be supplied multiple times. Defaults to the adjudicated natural package."
        ),
    )
    parser.add_argument(
        "--skip-coverage-eval",
        action="store_true",
        help="Evaluate only the human-style examples package.",
    )
    parser.add_argument(
        "--eval-coverage-limit",
        type=int,
        default=None,
        help="Optional per-held-out coverage-package query cap.",
    )
    parser.add_argument(
        "--eval-coverage-budgets",
        default="30,60,100",
        help="Comma or space separated held-out coverage-package budgets.",
    )
    parser.add_argument(
        "--eval-coverage-methods",
        default=(
            "opt,oracle_gvt,estimated_gvt,estimated_utility,memgpt_tiered,"
            "amem_graph,amac_admission,mem0_extract,density_only,summary_only,"
            "fact_only,recency_raw"
        ),
        help="Comma or space separated methods for held-out coverage-package evaluation.",
    )
    parser.add_argument(
        "--allow-natural-train-overlap",
        action="store_true",
        help=(
            "Do not exclude held-out coverage-package query ids from natural train "
            "packages. The default is safer and excludes overlaps."
        ),
    )
    parser.add_argument(
        "--estimator-ridge",
        type=float,
        default=0.25,
        help="Ridge penalty for the visible-feature linear estimator.",
    )
    parser.add_argument(
        "--estimated-noise-scale",
        type=float,
        default=0.0,
        help="Optional deterministic noise scale applied to learned predictions.",
    )
    parser.add_argument(
        "--estimated-noise-seed",
        type=int,
        default=0,
        help="Seed for deterministic learned-estimator prediction noise.",
    )
    parser.add_argument(
        "--out-dir",
        default="llm_memory_validation/learned_writer_deployable_noapi",
        help="Output directory.",
    )
    return parser


def synthetic_train_instances(
    *,
    n_seeds: int,
    distributions: Sequence[str],
    normal_count: int,
    update_count: int,
) -> list[OracleMemInstance]:
    if n_seeds <= 0:
        return []
    return [
        generate_named_distribution(
            distribution,
            seed,
            normal_count=normal_count,
            update_count=update_count,
        )
        for distribution in distributions
        for seed in range(n_seeds)
    ]


def natural_train_instances(
    package_dirs: Sequence[str],
    *,
    limit: int | None,
    exclude_query_ids: set[str] | None = None,
) -> tuple[list[OracleMemInstance], list[dict[str, Any]]]:
    instances: list[OracleMemInstance] = []
    manifest_rows: list[dict[str, Any]] = []
    exclude_query_ids = set(exclude_query_ids or ())
    for package_dir_text in package_dirs:
        package_dir = Path(package_dir_text)
        data = load_package(package_dir)
        all_queries = resolved_queries(data, limit)
        excluded = [
            query
            for query in all_queries
            if str(query.get("query_id", "")) in exclude_query_ids
        ]
        queries = [
            query
            for query in all_queries
            if str(query.get("query_id", "")) not in exclude_query_ids
        ]
        before = len(instances)
        for query in queries:
            instance = package_instance(data, query)
            if instance.candidates and any(weight > 0 for weight in instance.unit_weights.values()):
                instances.append(instance)
        manifest_rows.append(
            {
                "package_dir": str(package_dir),
                "resolved_queries_before_exclusion": len(all_queries),
                "excluded_query_ids": sorted(str(query["query_id"]) for query in excluded),
                "excluded_query_count": len(excluded),
                "resolved_queries": len(queries),
                "usable_instances": len(instances) - before,
            }
        )
    return instances, manifest_rows


def coverage_eval_query_ids(package_dirs: Sequence[str], *, limit: int | None) -> dict[str, list[str]]:
    query_ids: dict[str, list[str]] = {}
    for package_dir_text in package_dirs:
        package_dir = Path(package_dir_text)
        data = load_package(package_dir)
        query_ids[str(package_dir)] = [
            str(query.get("query_id", ""))
            for query in resolved_queries(data, limit)
        ]
    return query_ids


def weighted_train_instances(
    synthetic_instances: Sequence[OracleMemInstance],
    natural_instances: Sequence[OracleMemInstance],
    *,
    natural_weight: int,
) -> list[OracleMemInstance]:
    weight = max(0, int(natural_weight))
    return [*synthetic_instances, *(list(natural_instances) * weight)]


def estimator_coefficients(model: Any, limit: int = 25) -> list[dict[str, float | str]]:
    rows = [
        {"feature": name, "weight": float(weight), "abs_weight": abs(float(weight))}
        for name, weight in zip(model.feature_names, model.weights)
    ]
    rows.sort(key=lambda row: (-float(row["abs_weight"]), str(row["feature"])))
    return rows[:limit]


def write_transfer_report(
    out_dir: Path,
    *,
    train_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# Learned Writer Transfer Report",
        "",
        "This run trains a local visible-feature utility estimator on train-only oracle labels and evaluates held-out memory-writing decisions against exact finite-package OPT.",
        "",
        "## Train Data",
        "",
        f"- Synthetic train instances: {train_manifest['synthetic_train_instances']}",
        f"- Natural train instances: {train_manifest['natural_train_instances']}",
        f"- Total train instances: {train_manifest['total_train_instances']}",
        f"- Train candidates: {train_manifest['train_candidate_count']}",
        f"- Ridge: {train_manifest['estimator_ridge']}",
        f"- Test package: `{train_manifest['human_examples_jsonl']}`",
        "",
        "## Claim Boundary",
        "",
        "- Oracle coverage is used to create train labels only.",
        "- Held-out estimated-writer decisions use visible candidate metadata only.",
        "- The human-edited test package is schema-valid and exact-scored, but it is not an inter-annotator agreement study.",
        "",
        "## Held-Out Package Ratios",
        "",
    ]
    methods = sorted(summary.get("methods", []))
    by_budget = {}
    for row in summary.get("by_budget_method", []):
        by_budget.setdefault(int(row["budget"]), {})[str(row["method"])] = row
    for budget in sorted(by_budget):
        lines.append(f"### Budget {budget}")
        for method in methods:
            row = by_budget[budget].get(method)
            if row is None:
                continue
            lines.append(
                "- `{method}`: ratio_to_opt={ratio:.3f}, objective={objective:.3f}, cost={cost:.1f}".format(
                    method=method,
                    ratio=float(row.get("mean_ratio_to_opt", 0.0)),
                    objective=float(row.get("mean_objective", 0.0)),
                    cost=float(row.get("mean_selected_cost", 0.0)),
                )
            )
        lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_distributions = parse_tokens(args.synthetic_distributions)
    synthetic_instances = synthetic_train_instances(
        n_seeds=args.n_synthetic_train_seeds,
        distributions=synthetic_distributions,
        normal_count=args.normal_count,
        update_count=args.update_count,
    )
    natural_instances, natural_manifest = natural_train_instances(
        args.train_natural_package_dir,
        limit=args.train_natural_limit,
    )
    train_instances = [*synthetic_instances, *natural_instances]
    if not train_instances:
        raise ValueError("at least one synthetic or natural train instance is required")

    estimator = train_feature_utility_estimator(
        train_instances,
        train_distributions=(
            *(f"synthetic:{name}" for name in synthetic_distributions),
            *(f"natural:{Path(path).name}" for path in args.train_natural_package_dir),
        ),
        train_seeds=tuple(range(max(0, args.n_synthetic_train_seeds))),
        estimator_model=LOCAL_LEARNED_ESTIMATOR_MODEL,
        estimator_profile=LEARNED_ESTIMATOR_PROFILE,
        ridge=args.estimator_ridge,
        noise_scale=args.estimated_noise_scale,
        noise_seed=args.estimated_noise_seed,
    )

    human_examples_path = Path(args.human_examples_jsonl)
    human_rows = load_examples(human_examples_path)
    human_instance = build_human_instance(human_rows)
    budgets = tuple(int(token) for token in parse_tokens(args.budgets))
    methods = parse_tokens(args.methods)
    results = evaluate_human_package(
        human_instance,
        budgets,
        methods,
        estimator_model=estimator.estimator_model,
        estimator_profile=estimator.estimator_profile,
        estimator_state=estimator,
    )
    paths = write_benchmark_outputs(results, out_dir)
    write_human_report(out_dir, human_examples_path, human_rows, results)

    train_manifest = {
        "human_examples_jsonl": str(human_examples_path),
        "synthetic_train_distributions": list(synthetic_distributions),
        "synthetic_train_seeds": list(range(max(0, args.n_synthetic_train_seeds))),
        "synthetic_train_instances": len(synthetic_instances),
        "natural_train_packages": natural_manifest,
        "natural_train_instances": len(natural_instances),
        "total_train_instances": len(train_instances),
        "train_candidate_count": sum(len(instance.candidates) for instance in train_instances),
        "estimator_model": estimator.estimator_model,
        "estimator_profile": estimator.estimator_profile,
        "estimator_ridge": args.estimator_ridge,
        "estimated_noise_scale": args.estimated_noise_scale,
        "estimated_noise_seed": args.estimated_noise_seed,
        "top_coefficients": estimator_coefficients(estimator),
        "decision_features": "visible candidate metadata only at held-out test time",
        "oracle_coverage_used_for_training": True,
        "oracle_coverage_used_for_test_decision": False,
        **paths,
    }
    write_json(out_dir / "train_manifest.json", train_manifest)

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    write_transfer_report(out_dir, train_manifest=train_manifest, summary=summary)
    print(json.dumps(train_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
