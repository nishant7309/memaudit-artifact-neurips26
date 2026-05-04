"""Run the OracleMem exact-small synthetic MVP benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oraclemem.coverage_export import export_coverage_packages
from oraclemem.evaluate import (
    DEFAULT_METHODS,
    DEFAULT_ESTIMATOR_MODEL,
    DEFAULT_ESTIMATOR_PROFILE,
    ESTIMATED_METHODS,
    ESTIMATOR_PROFILES,
    LEARNED_ESTIMATOR_PROFILE,
    LOCAL_LEARNED_ESTIMATOR_MODEL,
    SUPPORTED_METHODS,
    generate_named_distribution,
    parse_int_list,
    parse_token_list,
    run_synthetic_benchmark,
    run_synthetic_train_dev_benchmark,
    write_benchmark_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OracleMem's pure-stdlib exact-small synthetic benchmark across "
            "seeds and storage budgets."
        )
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma or space separated integer seeds. Default: 0,1,2,3,4.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Use seeds 0..N-1. Overrides --seeds.",
    )
    parser.add_argument(
        "--budgets",
        default="4,6,9,12",
        help=(
            "Comma or space separated budgets. Integer tokens are absolute budgets; "
            "decimals in (0,1] are converted to token budgets as a fraction of the "
            "average generated candidate cost. Default: 4,6,9,12."
        ),
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma or space separated methods. Default: MVP + local writer baselines.",
    )
    parser.add_argument(
        "--estimated-model",
        default=DEFAULT_ESTIMATOR_MODEL,
        help=(
            "Estimator model label recorded for estimated_* methods. "
            f"Default: {DEFAULT_ESTIMATOR_MODEL}."
        ),
    )
    parser.add_argument(
        "--estimated-profile",
        choices=ESTIMATOR_PROFILES,
        default=DEFAULT_ESTIMATOR_PROFILE,
        help=(
            "Estimated-policy profile. The default is a deterministic local "
            "Gemini Flash-Lite-style utility prior; no API call is made."
        ),
    )
    parser.add_argument(
        "--train-dev-estimator",
        action="store_true",
        help=(
            "Train a local synthetic feature utility estimator on train seeds "
            "and evaluate estimated_* methods only on held-out dev seeds."
        ),
    )
    parser.add_argument(
        "--train-seeds",
        default=None,
        help="Comma or space separated train seeds for --train-dev-estimator.",
    )
    parser.add_argument(
        "--dev-seeds",
        default=None,
        help="Comma or space separated dev/evaluation seeds for --train-dev-estimator.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="Fraction of --seeds/--n-seeds used for training when explicit split seeds are omitted.",
    )
    parser.add_argument(
        "--estimator-ridge",
        type=float,
        default=1.0,
        help="Ridge penalty for the local train/dev feature utility estimator.",
    )
    parser.add_argument(
        "--estimated-noise-scale",
        type=float,
        default=0.0,
        help="Optional deterministic prediction-noise scale for the train/dev feature estimator.",
    )
    parser.add_argument(
        "--estimated-noise-seed",
        type=int,
        default=0,
        help="Seed for deterministic train/dev estimator prediction noise.",
    )
    parser.add_argument(
        "--distribution",
        "--distributions",
        dest="distributions",
        default="base",
        help="Comma or space separated exact-small distributions. Default: base.",
    )
    parser.add_argument(
        "--out-dir",
        "--out",
        dest="out_dir",
        default="oraclemem_mvp_runs",
        help="Output directory for raw JSONL, summary JSON, and summary Markdown.",
    )
    parser.add_argument(
        "--raw-jsonl",
        default="raw_results.jsonl",
        help="Raw result JSONL filename within --out-dir.",
    )
    parser.add_argument(
        "--summary-json",
        default="summary.json",
        help="Summary JSON filename within --out-dir.",
    )
    parser.add_argument(
        "--summary-md",
        default="summary.md",
        help="Summary Markdown filename within --out-dir.",
    )
    parser.add_argument(
        "--normal-count",
        type=int,
        default=3,
        help="Normal fact experiences per synthetic instance. Keep small for exact runs.",
    )
    parser.add_argument(
        "--update-count",
        type=int,
        default=2,
        help="Update/tombstone stress pairs per synthetic instance. Keep small for exact runs.",
    )
    parser.add_argument(
        "--solver",
        choices=("exact_stdlib", "milp"),
        default="exact_stdlib",
        help="Exact solver backend. MILP requires optional dependency `pulp`.",
    )
    parser.add_argument(
        "--verify-against",
        choices=("exact_stdlib", "milp"),
        default=None,
        help="Optional exact-solver cross-check. Raises if objective values differ.",
    )
    parser.add_argument(
        "--enable-retrieval",
        action="store_true",
        help="Attach deterministic write/retrieval decomposition metrics to raw JSONL rows.",
    )
    parser.add_argument(
        "--retrieval",
        default="fixed,oracle",
        help="Comma or space separated retrieval modes for --enable-retrieval. Supported: fixed, oracle.",
    )
    parser.add_argument(
        "--reader",
        default="local_evidence",
        help="Reader label for future API/local readers. Current implementation is local evidence-only.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress completion summary on stdout.",
    )
    parser.add_argument(
        "--export-coverage-matrices",
        "--export-coverage-package",
        dest="export_coverage_matrices",
        action="store_true",
        help=(
            "Export protocol-style synthetic coverage packages for generated "
            "instances. Each package includes candidate_memories.jsonl and "
            "sparse coverage_matrix.jsonl."
        ),
    )
    parser.add_argument(
        "--coverage-export-dir",
        default=None,
        help=(
            "Directory for --export-coverage-matrices. Default: "
            "<out-dir>/coverage_instances."
        ),
    )
    parser.add_argument(
        "--coverage-package-limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of generated instances to export. By "
            "default every distribution/seed package is exported."
        ),
    )
    return parser


def _parse_methods(value: str) -> tuple[str, ...]:
    return tuple(value.replace(",", " ").split())


def _resolve_budgets(
    value: str,
    seeds: list[int],
    *,
    distributions: tuple[str, ...],
    normal_count: int,
    update_count: int,
) -> tuple[list[int], str]:
    tokens = parse_token_list(value)
    budgets: list[int] = []
    fraction_tokens: list[float] = []
    for token in tokens:
        parsed = float(token)
        if 0.0 < parsed <= 1.0 and ("." in token or "e" in token.lower()):
            fraction_tokens.append(parsed)
        else:
            budgets.append(int(parsed))

    if not fraction_tokens:
        return budgets, "absolute"

    probe_costs = []
    for distribution in distributions:
        for seed in seeds:
            instance = generate_named_distribution(
                distribution,
                seed,
                normal_count=normal_count,
                update_count=update_count,
            )
            probe_costs.append(sum(candidate.cost for candidate in instance.candidates))
    base_cost = sum(probe_costs) / max(len(probe_costs), 1)
    budgets.extend(max(1, int(round(fraction * base_cost))) for fraction in fraction_tokens)
    return sorted(set(budgets)), f"fraction_of_avg_candidate_cost:{base_cost:.2f}"


def _dedupe_ints(values: list[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _resolve_train_dev_seeds(
    args: argparse.Namespace,
    seeds: list[int],
    parser: argparse.ArgumentParser,
) -> tuple[list[int], list[int]]:
    explicit_train = parse_int_list(args.train_seeds) if args.train_seeds else None
    explicit_dev = parse_int_list(args.dev_seeds) if args.dev_seeds else None
    if explicit_train is not None and explicit_dev is not None:
        return _dedupe_ints(explicit_train), _dedupe_ints(explicit_dev)
    if explicit_train is not None:
        train = _dedupe_ints(explicit_train)
        dev = [seed for seed in _dedupe_ints(seeds) if seed not in set(train)]
        if not dev:
            parser.error("--train-seeds was provided but no held-out dev seeds remain")
        return train, dev
    if explicit_dev is not None:
        dev = _dedupe_ints(explicit_dev)
        train = [seed for seed in _dedupe_ints(seeds) if seed not in set(dev)]
        if not train:
            parser.error("--dev-seeds was provided but no train seeds remain")
        return train, dev

    split_source = _dedupe_ints(seeds)
    if len(split_source) < 2:
        parser.error("--train-dev-estimator requires at least two total seeds or explicit train/dev seeds")
    if not 0.0 < float(args.train_fraction) < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    split_index = int(round(len(split_source) * float(args.train_fraction)))
    split_index = max(1, min(len(split_source) - 1, split_index))
    return split_source[:split_index], split_source[split_index:]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    seeds = list(range(args.n_seeds)) if args.n_seeds is not None else parse_int_list(args.seeds)
    distributions = _parse_methods(args.distributions)
    budgets, budget_basis = _resolve_budgets(
        args.budgets,
        seeds,
        distributions=distributions,
        normal_count=args.normal_count,
        update_count=args.update_count,
    )
    methods = _parse_methods(args.methods)
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    retrieval_modes = tuple(args.retrieval.replace(",", " ").split()) if args.enable_retrieval else ()
    if args.reader != "local_evidence" and args.enable_retrieval:
        print(
            "warning: --reader is logged as a label only; current retrieval decomposition "
            "uses a deterministic local evidence-only reader.",
            file=sys.stderr,
        )

    use_train_dev_estimator = (
        args.train_dev_estimator or args.estimated_profile == LEARNED_ESTIMATOR_PROFILE
    )
    train_seeds: list[int] = []
    dev_seeds: list[int] = []
    estimator_model = args.estimated_model
    if use_train_dev_estimator:
        train_seeds, dev_seeds = _resolve_train_dev_seeds(args, seeds, parser)
        if args.estimated_profile not in (DEFAULT_ESTIMATOR_PROFILE, LEARNED_ESTIMATOR_PROFILE):
            print(
                "warning: --train-dev-estimator uses "
                f"{LEARNED_ESTIMATOR_PROFILE}; overriding --estimated-profile.",
                file=sys.stderr,
            )
        if estimator_model == DEFAULT_ESTIMATOR_MODEL:
            estimator_model = LOCAL_LEARNED_ESTIMATOR_MODEL
        results = run_synthetic_train_dev_benchmark(
            train_seeds,
            dev_seeds,
            budgets,
            methods=methods,
            distributions=distributions,
            normal_count=args.normal_count,
            update_count=args.update_count,
            solver=args.solver,
            verify_against=args.verify_against,
            retrieval_modes=retrieval_modes,
            estimator_model=estimator_model,
            estimator_ridge=args.estimator_ridge,
            estimator_noise_scale=args.estimated_noise_scale,
            estimator_noise_seed=args.estimated_noise_seed,
        )
    else:
        results = run_synthetic_benchmark(
            seeds,
            budgets,
            methods=methods,
            distributions=distributions,
            normal_count=args.normal_count,
            update_count=args.update_count,
            solver=args.solver,
            verify_against=args.verify_against,
            retrieval_modes=retrieval_modes,
            estimator_model=estimator_model,
            estimator_profile=args.estimated_profile,
        )
    paths = write_benchmark_outputs(
        results,
        args.out_dir,
        raw_jsonl_name=args.raw_jsonl,
        summary_json_name=args.summary_json,
        summary_md_name=args.summary_md,
    )
    coverage_export = None
    if args.export_coverage_matrices:
        coverage_export_dir = (
            Path(args.coverage_export_dir)
            if args.coverage_export_dir is not None
            else Path(args.out_dir) / "coverage_instances"
        )
        export_seeds = dev_seeds if use_train_dev_estimator else seeds
        coverage_export = export_coverage_packages(
            seeds=export_seeds,
            distributions=distributions,
            out_dir=coverage_export_dir,
            normal_count=args.normal_count,
            update_count=args.update_count,
            max_packages=args.coverage_package_limit,
        )

    if not args.quiet:
        evaluated_seed_count = len(dev_seeds) if use_train_dev_estimator else len(seeds)
        print(
            "OracleMem MVP complete: "
            f"{len(distributions)} distributions x {evaluated_seed_count} eval seeds x "
            f"{len(budgets)} budgets x {len(methods)} methods"
        )
        print(f"distributions: {', '.join(distributions)}")
        print(f"budget_basis: {budget_basis}")
        if any(method in ESTIMATED_METHODS for method in methods):
            active_profile = (
                LEARNED_ESTIMATOR_PROFILE if use_train_dev_estimator else args.estimated_profile
            )
            print(
                "estimated_policy: "
                f"model={estimator_model}; "
                f"profile={active_profile}; api_called=false"
            )
            if use_train_dev_estimator:
                print(
                    "train_dev_estimator: "
                    f"train_seeds={len(train_seeds)}; dev_seeds={len(dev_seeds)}; "
                    f"ridge={args.estimator_ridge}; noise_scale={args.estimated_noise_scale}"
                )
        if retrieval_modes:
            print(f"retrieval_modes: {', '.join(retrieval_modes)}; reader: {args.reader}")
        print(f"raw_jsonl: {paths['raw_jsonl']}")
        print(f"summary_json: {paths['summary_json']}")
        print(f"summary_md: {paths['summary_md']}")
        if coverage_export is not None:
            print(f"coverage_export_manifest: {coverage_export['manifest']}")
            print(f"coverage_packages: {coverage_export['package_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
