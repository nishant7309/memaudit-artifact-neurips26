"""Stdlib tests for the OracleMem exact-small MVP."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oraclemem import evaluate


class OracleMemEvaluationTests(unittest.TestCase):
    def test_exact_solver_matches_bruteforce(self) -> None:
        instance = evaluate.generate_synthetic_instance(11, normal_count=2, update_count=1)
        for budget in (3, 5, 8):
            exact = evaluate.exact_solve(instance, budget)
            brute = evaluate.brute_force_solve(instance, budget)
            self.assertAlmostEqual(exact.objective_value, brute.objective_value, places=9)
            self.assertTrue(
                evaluate.is_feasible(instance.candidates, exact.selected_candidate_ids, budget)
            )

    def test_budget_and_group_feasibility_for_all_methods(self) -> None:
        instance = evaluate.generate_synthetic_instance(3, normal_count=2, update_count=1)
        rows = evaluate.evaluate_instance(instance, budgets=(3, 5, 7))
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertTrue(row.budget_feasible, row.to_json())
            self.assertTrue(row.group_feasible, row.to_json())
            self.assertLessEqual(row.selected_cost, row.budget)

    def test_ratio_labels_are_exact_and_not_reference_denominators(self) -> None:
        instance = evaluate.generate_synthetic_instance(5, normal_count=2, update_count=1)
        rows = evaluate.evaluate_instance(
            instance, budgets=(4,), methods=("opt", "oracle_gvt", "greedy")
        )
        for row in rows:
            self.assertEqual(row.denominator_label, "exact_opt")
            self.assertEqual(row.upper_bound_source, "exact_opt")
            self.assertIsNotNone(row.ratio_to_opt)
            self.assertIsNotNone(row.ratio_to_upper_bound)
            self.assertLessEqual(row.ratio_to_opt or 0.0, 1.0 + 1e-9)
            if row.method == "opt":
                self.assertAlmostEqual(row.ratio_to_opt or 0.0, 1.0)

        summary = evaluate.aggregate_results(rows)
        labels = summary["label_definitions"]
        self.assertIn("exact optimum is certified", labels["ratio_to_opt"])
        self.assertIn("never labeled as OPT", labels["ratio_to_reference"])

    def test_tombstone_benefit_on_update_stream(self) -> None:
        instance = evaluate.make_update_stress_instance()
        rows = evaluate.evaluate_instance(
            instance,
            budgets=(3,),
            methods=("oracle_gvt", "no_tombstone_gvt", "no_tombstone_opt", "fact_only"),
        )
        by_method = {row.method: row for row in rows}
        aware = by_method["oracle_gvt"]
        no_tombstone = by_method["no_tombstone_gvt"]
        no_tombstone_opt = by_method["no_tombstone_opt"]
        fact_only = by_method["fact_only"]

        self.assertGreater(aware.objective_value, no_tombstone.objective_value)
        self.assertGreater(aware.objective_value, no_tombstone_opt.objective_value)
        self.assertGreater(aware.objective_value, fact_only.objective_value)
        self.assertEqual(aware.update_metrics["invalidation_units_covered"], 1.0)
        self.assertEqual(no_tombstone.update_metrics["invalidation_units_covered"], 0.0)
        self.assertEqual(no_tombstone_opt.update_metrics["invalidation_units_covered"], 0.0)
        self.assertGreater(aware.update_metrics["selected_tombstone_like"], 0.0)

    def test_estimated_methods_use_estimates_not_oracle_marginals(self) -> None:
        instance = evaluate.OracleMemInstance(
            "estimated_policy_fixture",
            (
                evaluate.CandidateMemory(
                    "oracle_best",
                    "exp0",
                    "atomic_fact",
                    "FACT hidden gold evidence",
                    1,
                    {"gold": 1.0},
                    estimated_value=0.1,
                    estimator_model="fixture",
                ),
                evaluate.CandidateMemory(
                    "estimated_best",
                    "exp0",
                    "summary",
                    "SUMMARY visibly important but actually weak",
                    1,
                    {"gold": 0.0},
                    estimated_value=5.0,
                    estimator_model="fixture",
                ),
            ),
            {"gold": 1.0},
            current_units=("gold",),
        )
        rows = evaluate.evaluate_instance(
            instance,
            budgets=(1,),
            methods=("oracle_gvt", "estimated_gvt", "estimated_utility"),
            estimator_model=evaluate.DEFAULT_ESTIMATOR_MODEL,
            estimator_profile="external",
        )
        by_method = {row.method: row for row in rows}

        self.assertEqual(by_method["oracle_gvt"].selected_candidate_ids, ("oracle_best",))
        self.assertEqual(by_method["estimated_gvt"].selected_candidate_ids, ("estimated_best",))
        self.assertEqual(
            by_method["estimated_utility"].selected_candidate_ids,
            ("estimated_best",),
        )
        self.assertLess(by_method["estimated_gvt"].objective_value, by_method["oracle_gvt"].objective_value)
        self.assertEqual(
            by_method["estimated_gvt"].policy_metadata["estimator_model"],
            evaluate.DEFAULT_ESTIMATOR_MODEL,
        )
        self.assertFalse(by_method["estimated_gvt"].policy_metadata["api_called"])

    def test_estimated_methods_run_on_stress_distribution(self) -> None:
        rows = evaluate.run_synthetic_benchmark(
            seeds=(0, 1),
            budgets=(4,),
            distributions=("scope_shift_v2",),
            methods=("estimated_gvt", "estimated_utility"),
            estimator_model=evaluate.DEFAULT_ESTIMATOR_MODEL,
        )
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue(row.budget_feasible, row.to_json())
            self.assertTrue(row.group_feasible, row.to_json())
            self.assertEqual(
                row.policy_metadata["estimator_model"],
                evaluate.DEFAULT_ESTIMATOR_MODEL,
            )
            self.assertEqual(row.policy_metadata["estimator_profile"], "gemini_flash_lite_v1")

    def test_noisy_estimator_profile_is_local_and_supported(self) -> None:
        rows = evaluate.run_synthetic_benchmark(
            seeds=(0,),
            budgets=(4,),
            distributions=("scope_shift_v2",),
            methods=("estimated_gvt", "estimated_utility"),
            estimator_model=evaluate.DEFAULT_ESTIMATOR_MODEL,
            estimator_profile=evaluate.NOISY_ESTIMATOR_PROFILE,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row.budget_feasible, row.to_json())
            self.assertTrue(row.group_feasible, row.to_json())
            self.assertEqual(row.policy_metadata["estimator_profile"], evaluate.NOISY_ESTIMATOR_PROFILE)
            self.assertFalse(row.policy_metadata["api_called"])

    def test_train_dev_estimator_uses_visible_features_not_dev_oracle(self) -> None:
        train_instance = evaluate.OracleMemInstance(
            "learned_train_fixture",
            (
                evaluate.CandidateMemory(
                    "train_update_good",
                    "train_update",
                    "compound_update",
                    "UPDATE corrected current preference with explicit invalidation",
                    1,
                    {"train_gold": 1.0},
                ),
                evaluate.CandidateMemory(
                    "train_fact_weak",
                    "train_fact",
                    "atomic_fact",
                    "FACT weak standalone note",
                    1,
                    {"train_weak": 0.05},
                ),
            ),
            {"train_gold": 4.0, "train_weak": 1.0},
        )
        estimator = evaluate.train_feature_utility_estimator(
            (train_instance,),
            train_distributions=("fixture",),
            train_seeds=(0,),
            ridge=0.01,
        )
        dev_instance = evaluate.OracleMemInstance(
            "learned_dev_fixture",
            (
                evaluate.CandidateMemory(
                    "oracle_best",
                    "dev_exp",
                    "atomic_fact",
                    "FACT hidden gold evidence",
                    1,
                    {"gold": 1.0},
                ),
                evaluate.CandidateMemory(
                    "learned_visible",
                    "dev_exp",
                    "compound_update",
                    "UPDATE corrected current preference with explicit invalidation",
                    1,
                    {},
                ),
            ),
            {"gold": 1.0},
            current_units=("gold",),
        )
        rows = evaluate.evaluate_instance(
            dev_instance,
            budgets=(1,),
            methods=("oracle_gvt", "estimated_gvt", "estimated_utility"),
            estimator_model=estimator.estimator_model,
            estimator_profile=evaluate.LEARNED_ESTIMATOR_PROFILE,
            estimator_state=estimator,
        )
        by_method = {row.method: row for row in rows}

        self.assertEqual(by_method["oracle_gvt"].selected_candidate_ids, ("oracle_best",))
        self.assertEqual(by_method["estimated_gvt"].selected_candidate_ids, ("learned_visible",))
        self.assertEqual(by_method["estimated_utility"].selected_candidate_ids, ("learned_visible",))
        self.assertLess(by_method["estimated_gvt"].objective_value, by_method["oracle_gvt"].objective_value)
        metadata = by_method["estimated_gvt"].policy_metadata
        self.assertEqual(metadata["estimator_profile"], evaluate.LEARNED_ESTIMATOR_PROFILE)
        self.assertFalse(metadata["api_called"])
        self.assertTrue(metadata["trained_estimator"])
        self.assertTrue(metadata["oracle_coverage_used_for_training"])
        self.assertFalse(metadata["oracle_coverage_used_for_dev_decision"])

    def test_synthetic_train_dev_benchmark_evaluates_only_dev_seeds(self) -> None:
        rows = evaluate.run_synthetic_train_dev_benchmark(
            train_seeds=(0, 1),
            dev_seeds=(2, 3),
            budgets=(4,),
            distributions=("base",),
            methods=("estimated_gvt", "estimated_utility"),
            normal_count=1,
            update_count=1,
            estimator_ridge=0.1,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual({row.seed for row in rows}, {2, 3})
        for row in rows:
            self.assertTrue(row.budget_feasible, row.to_json())
            self.assertTrue(row.group_feasible, row.to_json())
            self.assertEqual(row.policy_metadata["estimator_profile"], evaluate.LEARNED_ESTIMATOR_PROFILE)
            self.assertEqual(row.policy_metadata["train_seeds"], [0, 1])
            self.assertFalse(row.policy_metadata["oracle_coverage_used_for_dev_decision"])

    def test_human_package_evaluator_accepts_trained_estimator(self) -> None:
        from llm_memory_validation.evaluate_human_style_examples import evaluate_human_package

        train_instance = evaluate.OracleMemInstance(
            "human_learned_train_fixture",
            (
                evaluate.CandidateMemory(
                    "train_visible_good",
                    "train_exp",
                    "compound_update",
                    "UPDATE corrected current preference with explicit invalidation",
                    1,
                    {"train_gold": 1.0},
                ),
                evaluate.CandidateMemory(
                    "train_weak",
                    "train_weak_exp",
                    "summary",
                    "short low-signal summary",
                    1,
                    {"train_weak": 0.01},
                ),
            ),
            {"train_gold": 3.0, "train_weak": 1.0},
        )
        estimator = evaluate.train_feature_utility_estimator(
            (train_instance,),
            train_distributions=("fixture",),
            train_seeds=(0,),
            ridge=0.01,
        )
        heldout_instance = evaluate.OracleMemInstance(
            "human_learned_heldout_fixture",
            (
                evaluate.CandidateMemory(
                    "oracle_best",
                    "heldout_exp",
                    "atomic_fact",
                    "FACT hidden gold evidence",
                    1,
                    {"gold": 1.0},
                ),
                evaluate.CandidateMemory(
                    "learned_visible",
                    "heldout_exp",
                    "compound_update",
                    "UPDATE corrected current preference with explicit invalidation",
                    1,
                    {},
                ),
            ),
            {"gold": 1.0},
            current_units=("gold",),
        )
        rows = evaluate_human_package(
            heldout_instance,
            budgets=(1,),
            methods=("opt", "oracle_gvt", "estimated_gvt", "estimated_utility"),
            estimator_model=estimator.estimator_model,
            estimator_profile=evaluate.LEARNED_ESTIMATOR_PROFILE,
            estimator_state=estimator,
        )
        by_method = {row.method: row for row in rows}

        self.assertEqual(by_method["opt"].selected_candidate_ids, ("oracle_best",))
        self.assertEqual(by_method["oracle_gvt"].selected_candidate_ids, ("oracle_best",))
        self.assertEqual(by_method["estimated_gvt"].selected_candidate_ids, ("learned_visible",))
        self.assertEqual(by_method["estimated_utility"].selected_candidate_ids, ("learned_visible",))
        metadata = by_method["estimated_gvt"].policy_metadata
        self.assertTrue(metadata["trained_estimator"])
        self.assertTrue(metadata["oracle_coverage_used_for_training"])
        self.assertFalse(metadata["oracle_coverage_used_for_dev_decision"])

    def test_method_comparisons_are_nondegenerate(self) -> None:
        instance = evaluate.generate_synthetic_instance(7, normal_count=3, update_count=2)
        rows = evaluate.evaluate_instance(instance, budgets=(4,), methods=evaluate.DEFAULT_METHODS)
        values = {row.method: round(row.objective_value, 8) for row in rows}
        self.assertGreaterEqual(len(set(values.values())), 3, values)
        opt_value = values["opt"]
        for method, value in values.items():
            self.assertLessEqual(value, opt_value + 1e-8, method)
        self.assertLess(values["recency_raw"], opt_value)
        self.assertIn("reservoir_raw", values)

    def test_deployable_writer_baselines_are_feasible(self) -> None:
        instance = evaluate.generate_synthetic_instance(9, normal_count=3, update_count=2)
        rows = evaluate.evaluate_instance(
            instance,
            budgets=(5,),
            methods=("memgpt_tiered", "mem0_extract", "amem_graph", "amac_admission"),
        )
        self.assertEqual(
            {row.method for row in rows},
            {"memgpt_tiered", "mem0_extract", "amem_graph", "amac_admission"},
        )
        for row in rows:
            self.assertTrue(row.budget_feasible, row.to_json())
            self.assertTrue(row.group_feasible, row.to_json())
            self.assertLessEqual(row.selected_cost, row.budget)
            self.assertEqual(
                row.policy_metadata.get("policy_family"),
                "deployable_writer_baseline",
            )
            self.assertFalse(row.policy_metadata.get("external_service_dependencies"))
            self.assertFalse(row.policy_metadata.get("oracle_coverage_used_for_decision"))
            self.assertIn("proxy_for", row.policy_metadata)
            self.assertRegex(
                row.policy_metadata.get("limitation", ""),
                r"(Local proxy only|Faithful local adapter only)",
            )

    def test_candidate_quality_ablations_use_filtered_candidate_pools(self) -> None:
        instance = evaluate.generate_synthetic_instance(10, normal_count=3, update_count=2)
        rows = evaluate.evaluate_instance(
            instance,
            budgets=(6,),
            methods=("opt", "generic_candidate_opt", "generic_candidate_gvt", "summary_candidate_opt"),
        )
        by_method = {row.method: row for row in rows}
        by_id = evaluate.candidates_by_id(instance.candidates)

        opt_value = by_method["opt"].objective_value
        for method in ("generic_candidate_opt", "generic_candidate_gvt", "summary_candidate_opt"):
            self.assertLessEqual(by_method[method].objective_value, opt_value + 1e-8)
            self.assertEqual(
                by_method[method].policy_metadata.get("policy_family"),
                "candidate_quality_ablation",
            )

        for candidate_id in by_method["generic_candidate_opt"].selected_candidate_ids:
            self.assertIn(
                by_id[candidate_id].representation_type,
                evaluate.GENERIC_CANDIDATE_TYPES,
            )
        for candidate_id in by_method["summary_candidate_opt"].selected_candidate_ids:
            self.assertEqual(by_id[candidate_id].representation_type, "summary")

    def test_distribution_field_and_summary_grouping(self) -> None:
        rows = evaluate.run_synthetic_benchmark(
            seeds=(0, 1),
            budgets=(4,),
            distributions=("base",),
            methods=("opt", "oracle_gvt", "reservoir_raw"),
            normal_count=1,
            update_count=1,
        )
        self.assertTrue(all(row.distribution == "base" for row in rows))
        summary = evaluate.aggregate_results(rows)
        self.assertEqual(summary["distributions"], ["base"])
        self.assertIn("by_distribution_budget_method", summary)
        self.assertTrue(
            all("distribution" in row for row in summary["by_distribution_budget_method"])
        )

    def test_outputs_include_jsonl_json_and_markdown(self) -> None:
        rows = evaluate.run_synthetic_benchmark(
            seeds=(0, 1),
            budgets=(3, 5),
            methods=("opt", "oracle_gvt", "no_tombstone_gvt", "no_tombstone_opt"),
            normal_count=1,
            update_count=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = evaluate.write_benchmark_outputs(rows, tmp)
            raw_path = Path(paths["raw_jsonl"])
            summary_json_path = Path(paths["summary_json"])
            summary_md_path = Path(paths["summary_md"])
            self.assertTrue(raw_path.exists())
            self.assertTrue(summary_json_path.exists())
            self.assertTrue(summary_md_path.exists())

            raw_rows = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(raw_rows), len(rows))
            summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
            self.assertIn("by_budget_method", summary)
            self.assertIn("Ratio Labels", summary_md_path.read_text(encoding="utf-8"))

    def test_coverage_package_export_contains_solver_inputs(self) -> None:
        instance = evaluate.generate_synthetic_instance(12, normal_count=2, update_count=1)
        expected_positive_rows = sum(
            1
            for candidate in instance.candidates
            for value in candidate.coverage.values()
            if value > 0
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = evaluate.write_coverage_package(instance, tmp)
            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)

            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["instance_id"], instance.instance_id)
            self.assertEqual(manifest["counts"]["candidate_memories"], len(instance.candidates))
            self.assertEqual(manifest["counts"]["positive_coverage_rows"], expected_positive_rows)

            coverage_rows = [
                json.loads(line)
                for line in Path(paths["coverage_matrix"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(coverage_rows), expected_positive_rows)
            self.assertTrue(all(0.0 < row["coverage"] <= 1.0 for row in coverage_rows))

    def test_exported_coverage_package_passes_structural_audit(self) -> None:
        from scripts.audit_coverage_artifacts import audit_artifact

        instance = evaluate.generate_synthetic_instance(13, normal_count=2, update_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = evaluate.write_coverage_package(instance, tmp)
            audit = audit_artifact(
                "synthetic_package",
                Path(paths["package_dir"]),
                "Synthetic OracleMem coverage package.",
                sample_rows=1000,
            )

            self.assertEqual(audit.format, "coverage_package_dir")
            self.assertEqual(
                audit.statuses["oracle_denominator"],
                "machine-checkable coverage package",
            )
            self.assertEqual(
                audit.statuses["coverage_matrix"],
                "candidate-unit coverage present",
            )
            self.assertTrue(all(audit.package_files.values()), audit.package_files)
            coverage_rows = [
                json.loads(line)
                for line in Path(paths["coverage_matrix"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all("candidate_id" in row and "unit_id" in row for row in coverage_rows))

            query_rows = [
                json.loads(line)
                for line in Path(paths["queries"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(query_rows), 0)
            self.assertTrue(all(row["required_unit_ids"] for row in query_rows))


if __name__ == "__main__":
    unittest.main()
