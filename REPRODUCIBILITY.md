# Reproducibility

This document records the current reproducibility path for the active root
manuscript, `main.tex`. The repository is intentionally split into deterministic
non-API experiments, cached external LongMemEval artifacts, and API reader runs.

## Environment

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Optional dependencies are separated by task:

```bash
python -m pip install -r requirements-api.txt
python -m pip install -r requirements-milp.txt
```

The exact-small MemAudit benchmark and unit tests use only the Python standard
library plus `pytest` for tests. LongMemEval retrieval regeneration uses local
ML dependencies and downloads the LongMemEval-S dataset and dense retriever
model. API reader runs use OpenRouter and require an API key.

## LaTeX Build

On this machine, `latexmk`, `pdflatex`, and `tectonic` were not available on
PATH during the 2026-04-28 local check. The attempted local build is recorded in
`latex_compile_attempt.txt`. A generated `latex_compile.log` also exists
locally, but `*.log` is ignored by the repository.

If a TeX distribution is installed locally, run one of:

```bash
make paper
make paper-pdflatex
make paper-tectonic
```

Because local compilation was unavailable here, `.github/workflows/latex.yml`
builds `main.tex` with GitHub Actions on push and pull request.

## Unit Tests

```bash
python -m pytest test_oraclemem.py
```

Current verification on 2026-05-01: both `python -m unittest test_oraclemem.py`
and `python -m pytest test_oraclemem.py` ran 17 tests and passed.

## Quickcheck

Use this before any expensive API or GPU work:

```bash
python -m unittest test_oraclemem.py
python run_oraclemem_mvp.py --n-seeds 3 --budgets 4 --distribution base --methods opt,oracle_gvt,density_only --out-dir oraclemem_runs/quickcheck
```

Expected outputs:

- `oraclemem_runs/quickcheck/raw_results.jsonl`
- `oraclemem_runs/quickcheck/summary.json`
- `oraclemem_runs/quickcheck/summary.md`

## Exact-Small Benchmark

Used by the exact-small budget-sweep figure in `main.tex`.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 500 \
  --budgets 0.01,0.02,0.05,0.10,0.20 \
  --distribution base \
  --methods opt,oracle_gvt,density_only,recency_raw,summary_only,fact_only,no_tombstone_gvt,no_tombstone_opt \
  --out oraclemem_runs/exact_500
```

Expected outputs:

- `oraclemem_runs/exact_500/raw_results.jsonl`
- `oraclemem_runs/exact_500/summary.json`
- `oraclemem_runs/exact_500/summary.md`

The reported `ratio_to_opt` field is valid only for these exact-small runs where
the denominator is an exact certified optimum.

## Stress Suite

Used by the validity-heavy stress figure in `main.tex`. The manuscript reports
the validity-heavy subset `base`, `update_chain`, and `temporal_interval` from
the larger stress artifact.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 500 \
  --budgets 0.02,0.05,0.10,0.20 \
  --distribution base,update_chain,temporal_interval,density_trap,scope_shift,summary_tradeoff,redundancy_heavy,abstention_hard \
  --methods opt,oracle_gvt,density_only,greedy,recency_raw,reservoir_raw,summary_only,fact_only,no_tombstone_gvt,no_tombstone_opt \
  --out oraclemem_runs/stress_exact_500
```

Expected outputs:

- `oraclemem_runs/stress_exact_500/raw_results.jsonl`
- `oraclemem_runs/stress_exact_500/summary.json`
- `oraclemem_runs/stress_exact_500/summary.md`

## Representative Non-Oracle Writers

Used by the text diagnostic on Estimated-GVT, A-MAC-like admission, and
Mem0-style extraction proxies. These methods use visible candidate features,
not hidden coverage labels.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 500 \
  --budgets 4,6 \
  --distribution base,update_chain,temporal_interval \
  --methods opt,oracle_gvt,estimated_gvt,amac_admission,mem0_extract,density_only,recency_raw,summary_only,fact_only,no_tombstone_opt \
  --out-dir oraclemem_runs/representative_writers_500
```

Expected outputs:

- `oraclemem_runs/representative_writers_500/raw_results.jsonl`
- `oraclemem_runs/representative_writers_500/summary.json`
- `oraclemem_runs/representative_writers_500/summary.md`

## No-API Proxy Writer Baselines

Diagnostic only; not a main-paper result after the 9-page compression pass. This
local diagnostic addresses the real-system-comparison concern without calling
OpenRouter, OpenAI, embedding services, or API reader code. It runs deterministic proxies for
MemGPT-style tiering, Mem0-style extraction, A-Mem-style graph/evolving memory,
and A-MAC-style admission under the same MemAudit candidate protocol and exact
OPT denominator.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 50 \
  --distribution base,update_chain,scope_shift_v2,density_trap_v2,temporal_interval \
  --budgets 4,6 \
  --methods opt,oracle_gvt,memgpt_tiered,mem0_extract,amem_graph,amac_admission,generic_candidate_opt,no_tombstone_opt \
  --out-dir oraclemem_runs/proxy_writer_baselines_50 \
  --enable-retrieval \
  --retrieval fixed,oracle
```

Expected outputs:

- `oraclemem_runs/proxy_writer_baselines_50/raw_results.jsonl`
- `oraclemem_runs/proxy_writer_baselines_50/summary.json`
- `oraclemem_runs/proxy_writer_baselines_50/summary.md`
- `oraclemem_runs/proxy_writer_baselines_50/REPORT.md`

The report is explicit that these local ratios are synthetic exact-small ratios
for proxy writers. A real-system comparison still requires running the actual
systems with budget-matched memory generation, storage accounting, retrieval
configuration, and evaluation traces.

## Gemini Natural Coverage Pilot

Superseded by the Natural-200 and adjudicated-subset results in `main.tex`.
This run builds a smaller LongMemEval-S support-slice MemAudit coverage package
using Gemini through OpenRouter. It requires `api.env` with
`OPENROUTER_API_KEY`. Candidate generation receives only support sessions plus
distractors; query/gold-answer fields are used only in the separate labeling
step.

```bash
python llm_memory_validation/gemini_natural_oraclemem.py \
  --limit 50 \
  --distractors-per-example 2 \
  --budgets 30,60,100 \
  --out-dir llm_memory_validation/gemini_natural_oraclemem_50 \
  --request-sleep 0.02

python scripts/audit_coverage_artifacts.py \
  --no-defaults \
  --artifact gemini_natural_50=llm_memory_validation/gemini_natural_oraclemem_50/coverage_package \
  --output-dir llm_memory_validation/gemini_natural_oraclemem_50/coverage_audit
```

Expected outputs:

- `llm_memory_validation/gemini_natural_oraclemem_50/REPORT.md`
- `llm_memory_validation/gemini_natural_oraclemem_50/coverage_resolved_summary.json`
- `llm_memory_validation/gemini_natural_oraclemem_50/coverage_package/`
- `llm_memory_validation/gemini_natural_oraclemem_50/coverage_audit/REPORT.md`

The first uncached 50-example run used 248 API calls, 502,698 total tokens, and
about `$0.286` in OpenRouter-reported cost. Cached reruns use zero additional
API calls. This run is a pilot: 30/50 examples are coverage-resolved and the
labels are single-model annotations rather than human adjudications.

## Natural-200 And Model-Adjudicated Subsets

Used by the natural package reliability table and the model-adjudicated subset
table in `main.tex`.

Primary Natural-200 package:

```bash
python llm_memory_validation/gemini_natural_oraclemem.py \
  --limit 200 \
  --distractors-per-example 0 \
  --max-session-words 1800 \
  --budgets 30,60,100 \
  --out-dir llm_memory_validation/oraclemem_natural_200_gemini_v2 \
  --request-sleep 0.02

python scripts/audit_coverage_artifacts.py \
  --no-defaults \
  --artifact natural_200_gemini_v2=llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_package \
  --output-dir llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_audit
```

Gemini Flash adjudicated subset:

```bash
python llm_memory_validation/adjudicate_natural_package.py \
  --primary-package-dir llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_package \
  --out-dir llm_memory_validation/natural_adjudicated_100_gemini_flash \
  --model google/gemini-2.5-flash \
  --limit 100 \
  --budgets 30,60,100 \
  --secondary-agreement-rows llm_memory_validation/natural50_annotation_agreement_gemini31_vs_gemini25/agreement_rows.jsonl \
  --mem0-raw-results llm_memory_validation/mem0_natural200_actual/raw_results.jsonl \
  --request-sleep 0.02
```

The released human audit for the 87-example adjudicated subset is stored in:

```text
llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit
```

It contains `annotation_cells.csv`, independent `human_a_labels.csv` and
`human_b_labels.csv`, `human_audit_metrics.json`, and
`ratio_stability_package_rows.csv`. The paper-facing values are
human-human `kappa=0.831`, human-consensus-vs-Gemini `kappa=0.863`, and
mean absolute ratio shift `0.035` after re-scoring package-writer rows with
human labels.

The Claude Sonnet 4.5 cross-model stress audit can be regenerated with:

```bash
python llm_memory_validation/adjudicate_natural_package.py \
  --primary-package-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package \
  --out-dir llm_memory_validation/natural_adjudicated_100_claude_sonnet45 \
  --model anthropic/claude-sonnet-4.5 \
  --limit 87 \
  --budgets 30,60,100 \
  --methods oracle_gvt,estimated_gvt,memgpt_tiered,mem0_extract,amem_graph \
  --request-sleep 0.05 \
  --skip-existing

python scripts/run_claude_cell_coverage_audit.py \
  --package-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package \
  --human-audit-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit \
  --out-dir llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cell_coverage_audit \
  --model anthropic/claude-sonnet-4.5 \
  --skip-existing

python scripts/cross_model_coverage_check.py \
  --gemini-package llm_memory_validation/natural_adjudicated_100_gemini_flash \
  --claude-package llm_memory_validation/natural_adjudicated_100_claude_sonnet45 \
  --human-audit-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit \
  --claude-cell-labels llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cell_coverage_audit/claude_cell_labels.csv \
  --out-dir llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cross_model_check
```

Current Sonnet stress-test result: package-writer rankings are preserved at
`B=30,60,100` with Spearman `rho=1.0`, but Sonnet binary coverage labels do not
meet the target agreement thresholds (`kappa=0.649` vs Gemini and `0.588` vs
agreed humans at package level; `0.531` vs Gemini and `0.484` vs agreed humans
for the blind cell-level pass). The paper treats this as a model-sensitivity
boundary, not as a replacement annotation source.

Gemini 3.1 Flash-Lite spot-check:

```bash
python llm_memory_validation/adjudicate_natural_package.py \
  --primary-package-dir llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_package \
  --out-dir llm_memory_validation/natural_spotcheck_30_gemini31_flash_lite \
  --model google/gemini-3.1-flash-lite-preview \
  --limit 30 \
  --budgets 30,60,100 \
  --methods opt,oracle_gvt,estimated_gvt,amac_admission,summary_only,fact_only,recency_raw \
  --secondary-agreement-rows llm_memory_validation/natural50_annotation_agreement_gemini31_vs_gemini25/agreement_rows.jsonl \
  --mem0-raw-results llm_memory_validation/mem0_natural200_actual/raw_results.jsonl \
  --request-sleep 0.02 \
  --skip-existing

python scripts/audit_coverage_artifacts.py \
  --no-defaults \
  --artifact natural_spotcheck_30_gemini31_flash_lite=llm_memory_validation/natural_spotcheck_30_gemini31_flash_lite/coverage_package \
  --output-dir llm_memory_validation/natural_spotcheck_30_gemini31_flash_lite/coverage_audit
```

The Flash-Lite spot-check attempted 30 examples, exported 29
accepted/corrected examples, rejected 1, used 201,301 total tokens, and cost
`$0.0639` through OpenRouter. It is model adjudication, not human validation.

## Human-Edited Natural Seed Package

This package is a fictional 100-example natural-memory seed set that was
manually edited/audited after generation. It is used as an artifact-validity
check for manual annotation plus exact finite-package scoring. It is not an
inter-annotator agreement study.

Validate the canonical JSONL:

```bash
python scripts/validate_human_style_examples.py llm_memory_validation/human_style_examples/examples_100.jsonl
```

Evaluate the finite package with an exact dynamic-programming denominator:

```bash
python llm_memory_validation/evaluate_human_style_examples.py \
  --examples-jsonl llm_memory_validation/human_style_examples/examples_100.jsonl \
  --out-dir llm_memory_validation/human_style_examples/eval_package_100 \
  --budgets 150,300,600,1000 \
  --methods opt,oracle_gvt,estimated_gvt,memgpt_tiered,amem_graph,amac_admission,mem0_extract,density_only,greedy,fact_only,summary_only,recency_raw,no_tombstone_opt
```

Expected outputs:

- `llm_memory_validation/human_style_examples/eval_package_100/raw_results.jsonl`
- `llm_memory_validation/human_style_examples/eval_package_100/summary.json`
- `llm_memory_validation/human_style_examples/eval_package_100/summary.md`
- `llm_memory_validation/human_style_examples/eval_package_100/REPORT.md`

Current verification on 2026-05-01: validation passed with 100 records and no
structural errors. The evaluator reports the denominator as
`exact_human_audited_package_dp`.

Export the same examples to the shared coverage-package schema:

```bash
python llm_memory_validation/export_human_style_coverage_package.py \
  --examples-jsonl llm_memory_validation/human_style_examples/examples_100.jsonl \
  --out-dir llm_memory_validation/human_style_examples/coverage_package

python scripts/audit_coverage_artifacts.py \
  --no-defaults \
  --artifact human_style_coverage=llm_memory_validation/human_style_examples/coverage_package \
  --output-dir llm_memory_validation/human_style_examples/coverage_package_audit
```

Run actual public A-Mem on the exported human-edited package:

```bash
python llm_memory_validation/run_actual_amem_natural_baseline.py \
  --package-dir llm_memory_validation/human_style_examples/coverage_package \
  --out-dir llm_memory_validation/human_style_examples/actual_amem_gemini_flash_100 \
  --limit 100 \
  --budgets 150,300,600,1000,5000 \
  --amem-model google/gemini-2.5-flash \
  --coverage-model google/gemini-2.5-flash \
  --request-sleep 0.02 \
  --amem-max-tokens 3000
```

Current actual A-Mem human-edited run: 85 query-resolved examples, 456 cached API
prompts, 269,742 tokens, estimated OpenRouter cost `$0.233`. Full A-Mem notes
reach union-OPT ratio `0.971` at all reported budgets; metadata-only reaches
`0.247`. This result is strong but should be interpreted with the package caveat:
the sessions are short enough that full notes fit the 150+ word budgets.

## Learned Writer Transfer Diagnostic

This local run trains a visible-feature utility estimator on train-only oracle
labels from synthetic instances plus the Natural-200 model-annotated package,
then evaluates held-out decisions on the human-edited seed package. Hidden
coverage is used for train labels only; held-out selection sees visible
candidate metadata only.

```bash
python llm_memory_validation/evaluate_learned_writer_transfer.py \
  --out-dir llm_memory_validation/human_style_examples/learned_writer_transfer \
  --budgets 150,300,600,1000 \
  --methods opt,oracle_gvt,estimated_gvt,estimated_utility,memgpt_tiered,amem_graph,amac_admission,mem0_extract,density_only,greedy,fact_only,summary_only,recency_raw,no_tombstone_opt
```

Expected outputs:

- `llm_memory_validation/human_style_examples/learned_writer_transfer/raw_results.jsonl`
- `llm_memory_validation/human_style_examples/learned_writer_transfer/summary.json`
- `llm_memory_validation/human_style_examples/learned_writer_transfer/summary.md`
- `llm_memory_validation/human_style_examples/learned_writer_transfer/REPORT.md`
- `llm_memory_validation/human_style_examples/learned_writer_transfer/train_manifest.json`

Current run: 1,000 synthetic train instances plus 200 natural train instances
with 22,106 train candidates. Estimated-GVT reaches held-out exact package-OPT
ratios `0.933/0.926/0.854/0.792` at budgets `150/300/600/1000`. This is a
deployable-writer diagnostic, not an inter-annotator natural benchmark.

Training-source ablations:

```bash
python llm_memory_validation/evaluate_learned_writer_transfer.py \
  --out-dir llm_memory_validation/human_style_examples/learned_writer_transfer_synth_only \
  --train-natural-limit 0 \
  --budgets 150,300,600,1000 \
  --methods opt,oracle_gvt,estimated_gvt,estimated_utility,amac_admission,mem0_extract,density_only,greedy,fact_only,summary_only,recency_raw,no_tombstone_opt

python llm_memory_validation/evaluate_learned_writer_transfer.py \
  --out-dir llm_memory_validation/human_style_examples/learned_writer_transfer_natural_only \
  --n-synthetic-train-seeds 0 \
  --budgets 150,300,600,1000 \
  --methods opt,oracle_gvt,estimated_gvt,estimated_utility,amac_admission,mem0_extract,density_only,greedy,no_tombstone_opt
```

Current ablations: synthetic-only Estimated-GVT reaches
`0.667/0.778/0.792/0.833`; Natural-200-only reaches
`0.000/0.074/0.375/0.486`. The combined run is therefore the paper-facing
learned-writer result because it is strongest at tight and medium budgets.

## Natural Writer Adapter Diagnostic

This local run scores Letta/MemGPT-style archival/recency and A-Mem-style graph
adapters on the adjudicated natural package under the same exact package OPT
denominator. It does not call an API and does not run Letta or A-Mem itself.

```bash
python llm_memory_validation/evaluate_coverage_package_writers.py \
  --package-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package \
  --out-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters \
  --budgets 30,60,100 \
  --methods opt,oracle_gvt,memgpt_tiered,amem_graph,mem0_extract,amac_admission,estimated_gvt,density_only,summary_only,fact_only,recency_raw
```

Expected outputs:

- `llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/raw_results.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/summary.json`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/summary.md`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/REPORT.md`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters/run_manifest.json`

Current run: 87 accepted/corrected adjudicated examples, zero API calls.
Letta/MemGPT-style reaches `0.638/0.433/0.431`, A-Mem-style reaches
`0.481/0.374/0.377`, and density-only reaches `0.991/0.955/0.962` at budgets
`30/60/100`. The density result is a warning that this copied-candidate natural
denominator is unusually density-friendly.

## Human-Edited Writer Adapter Diagnostic

This local run scores the same Letta/MemGPT-style, A-Mem-style, Mem0-style, and
A-MAC-style adapters on the exported human-edited coverage package. It is a
zero-API denominator-matched check. It does not run the Letta service or MemGPT
controller; the checked-out Letta repository requires a service/API/model
configuration for a true production run.

```bash
python llm_memory_validation/evaluate_coverage_package_writers.py \
  --package-dir llm_memory_validation/human_style_examples/coverage_package \
  --out-dir llm_memory_validation/human_style_examples/writer_adapters \
  --budgets 150,300,600,1000 \
  --methods opt,oracle_gvt,memgpt_tiered,amem_graph,mem0_extract,amac_admission,estimated_gvt,density_only,summary_only,fact_only,recency_raw
```

Expected outputs:

- `llm_memory_validation/human_style_examples/writer_adapters/raw_results.jsonl`
- `llm_memory_validation/human_style_examples/writer_adapters/summary.json`
- `llm_memory_validation/human_style_examples/writer_adapters/summary.md`
- `llm_memory_validation/human_style_examples/writer_adapters/REPORT.md`
- `llm_memory_validation/human_style_examples/writer_adapters/run_manifest.json`

Current run: 85 query-resolved examples, zero API calls. Letta/MemGPT-style
reaches `0.847`, A-Mem-style reaches `0.876`, Mem0-style reaches `0.753`, and
A-MAC-style reaches `0.835` across budgets `150/300/600/1000`. Density-only is
`1.000` on this per-query exported package, so this row is a MemGPT-style
adapter reproducibility check rather than the strongest algorithmic separation.

## Faithful MemGPT/Letta Union Baseline

This no-API runner is the current MemGPT/Letta-strengthened baseline on the
adjudicated natural package. It checks the local `external_repos/letta` checkout
metadata, records that the actual Letta import path is not available without the
full service dependency stack, then simulates the relevant core/archival/recall
memory tiers over exported package candidates. Writing and retrieval use visible
metadata only; hidden coverage is used only for scoring, except in the
analysis-only upper-pruned bound row.

```bash
python llm_memory_validation/run_faithful_memgpt_letta_baseline.py \
  --package-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package \
  --out-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union \
  --budgets 30,60,100 \
  --limit 87
```

Expected outputs:

- `llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union/raw_results.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union/summary.json`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union/REPORT.md`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union/written_stores.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union/run_manifest.json`

Current run: 87/87 examples, zero API calls. Archival-search pruning reaches
`0.746/0.739/0.866` ratio to union OPT at budgets `30/60/100`; recency pruning
reaches `0.642/0.700/0.877`; the analysis-only upper-pruned bound reaches
`0.829/0.907/0.939`.

## Actual Letta OpenRouter Passage Run

This runs the checked-out Letta server (`external_repos/letta`, version
`0.16.7`) with Postgres/pgvector, OpenRouter Gemini, and authenticated
OpenRouter passage embeddings. Apply
`llm_memory_validation/patches/letta_openrouter_embedding_auth.patch` to the
Letta checkout before starting the server; without it, OpenRouter passage
search uses the wrong API key path.

```powershell
.\.venv_letta_prod\Scripts\python.exe llm_memory_validation\run_actual_letta_openrouter_baseline.py `
  --package-dir llm_memory_validation\natural_adjudicated_100_gemini_flash\coverage_package `
  --out-dir llm_memory_validation\natural_adjudicated_100_gemini_flash\actual_letta_openrouter_gemini_passage_87 `
  --limit 87 `
  --budgets 30,60,100 `
  --include-salience-pruned `
  --include-oracle-pruned-upper `
  --max-steps 12 `
  --message-retries 2 `
  --request-sleep 0.02
```

Expected outputs:

- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/raw_results.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/summary.json`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/REPORT.md`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/written_stores.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/coverage_scoring_calls.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87/salience_scoring_calls.jsonl`

Current run: 87/87 examples, zero failed instances. Letta writes archival
passages for 85 examples and core-memory atoms for 30 examples. The combined
core+archival store reaches union-OPT ratios `0.652/0.696/0.734` with salience
pruning, `0.219/0.260/0.342` with recency pruning, and `0.723/0.763/0.765` for
the analysis-only upper-pruned bound at budgets `30/60/100`.

## Actual A-Mem Gemini-Flash Pilot

This runs the checked-out public `external_repos/AgenticMemory` implementation,
using Gemini Flash through OpenRouter for A-Mem metadata/evolution calls and for
post-hoc coverage scoring. It reports a finite union denominator over package
candidates plus A-Mem-written memories. The full-memory rows score A-Mem's actual
stored notes; the metadata rows are a compact diagnostic serialization of
A-Mem-generated context/keywords/tags/links.

```bash
python llm_memory_validation/run_actual_amem_natural_baseline.py \
  --package-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package \
  --out-dir llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87 \
  --limit 87 \
  --budgets 30,60,100,5000 \
  --amem-model google/gemini-2.5-flash \
  --coverage-model google/gemini-2.5-flash \
  --request-sleep 0.02 \
  --amem-max-tokens 3000
```

Expected outputs:

- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/REPORT.md`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/summary.json`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/raw_results.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/written_stores.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/coverage_scoring_calls.jsonl`
- `llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87/run_manifest.json`

Current 87-example run: raw full A-Mem notes have mean serialized cost `4446`
words and therefore score `0.000/0.000/0.000` at budgets `30/60/100`; at the
diagnostic budget `5000`, the raw full-store oracle upper reaches `0.845`.
The compact metadata diagnostic has mean cost `66` words and reaches
`0.204/0.158/0.180` with oracle pruning at budgets `30/60/100`. The run used
524 cached API prompts, 2,433,021 tokens, and an estimated OpenRouter cost of
`$1.576`.

## Actual Mem0 Smoke

This verifies executable integration with the public Mem0 codebase. It is not a
benchmark and should not be reported as a budget-matched Mem0 comparison.

Prerequisites from this environment:

```bash
python -m pip install qdrant-client==1.12.2 rank-bm25==0.2.2 litellm==1.83.7
python -m pip install -e external_repos/mem0
python -m pip install "huggingface-hub>=0.34,<1.0"
```

Run:

```bash
python llm_memory_validation/mem0_actual_smoke.py \
  --api-env api.env \
  --out-dir llm_memory_validation/mem0_actual_smoke
```

Expected outputs:

- `llm_memory_validation/mem0_actual_smoke/search_result.json`
- `llm_memory_validation/actual_system_repo_audit/REPORT.md`

## LongMemEval-S Retrieval Transfer

Diagnostic only after the 9-page compression pass. This report is
retrieval-only: no answer generation, no abstention scoring, and no exact OPT
denominator.

To regenerate the focus report from the cached retrieval rows:

```bash
python llm_memory_validation/longmemeval_focus_report.py \
  --summary-json llm_memory_validation/competitor_run_v2/summary.json \
  --retrieval-rows-json llm_memory_validation/competitor_run_v2/retrieval_rows.json \
  --output-dir llm_memory_validation/longmemeval_focus_report_core4 \
  --methods dense_budgeted_bsc,dense_rag_e5,dense_budgeted_replay,fifo_replay
```

Expected outputs:

- `llm_memory_validation/longmemeval_focus_report_core4/summary.json`
- `llm_memory_validation/longmemeval_focus_report_core4/REPORT.md`

The current paper-facing label map is:

- `dense_budgeted_bsc`: MemAudit writer + dense retrieval
- `dense_rag_e5`: Full raw-store dense retrieval
- `dense_budgeted_replay`: Budgeted raw replay + dense retrieval
- `fifo_replay`: FIFO raw replay

To regenerate the upstream dense retrieval rows, use:

```bash
python llm_memory_validation/paper_competitor_suite.py \
  --output-dir llm_memory_validation/competitor_run_v2 \
  --topk 5 \
  --retriever-model intfloat/e5-base-v2
```

This upstream regeneration downloads external data/model artifacts and may vary
with model or dataset revisions unless those are pinned outside this repository.

## GPT-5.5 Frozen-Context Reader

Appendix diagnostic only after the 9-page compression pass. The current artifact
uses frozen top-5 retrieval contexts, `openai/gpt-5.5` through OpenRouter, and
the `answer_if_supported` prompt.

Set up `api.env` locally. Do not commit it.

```text
OPENROUTER_API_KEY=...
```

Then run:

```bash
python llm_memory_validation/longmemeval_reader_eval.py \
  --dataset-json llm_memory_validation/cache/longmemeval_s_cleaned.json \
  --retrieval-rows-json llm_memory_validation/competitor_run_v2/retrieval_rows.json \
  --output-dir llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full \
  --methods dense_budgeted_bsc,dense_rag_e5,dense_budgeted_replay,fifo_replay \
  --focus-only \
  --focus-types knowledge-update,temporal-reasoning \
  --reader openrouter \
  --reader-model openai/gpt-5.5 \
  --prompt-style answer_if_supported \
  --api-env api.env \
  --api-cache llm_memory_validation/openrouter_cache_gpt55_answer_supported_focus_full.json
```

Expected outputs:

- `llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json`
- `llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/REPORT.md`
- `llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/reader_outputs.jsonl`
- `llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/predictions.json`

The committed/cacheable outputs should be treated as the reproducible artifact
for the paper. Re-running the API may change costs, latency, or model behavior.

## Reader Audit

Appendix diagnostic only after the 9-page compression pass.

```bash
python llm_memory_validation/longmemeval_reader_eval.py \
  --analyze-errors \
  --run-dir llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full
```

Expected outputs in the same run directory:

- `ERROR_AUDIT.md`
- `error_audit_summary.json`
- `error_audit_rows.jsonl`
- `failure_examples.jsonl`
- `semantic_audit_sample_50.jsonl`
- `normalized_scoring.json`
- `llm_memory_validation/scoring_audit_gpt55/normalized_scoring_v2.json`

## Deterministic Decomposition

Diagnostic only after the 9-page compression pass. This is a local evidence-only
reader path and does not use an API.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 300 \
  --budgets 0.05,0.10,0.20 \
  --distribution base,update_chain,temporal_interval \
  --methods opt,oracle_gvt,density_only,greedy,recency_raw,reservoir_raw,summary_only,fact_only,no_tombstone_gvt \
  --enable-retrieval \
  --retrieval fixed,oracle \
  --reader evidence_only \
  --out oraclemem_runs/decomp_det_300
```

Expected outputs:

- `oraclemem_runs/decomp_det_300/raw_results.jsonl`
- `oraclemem_runs/decomp_det_300/summary.json`
- `oraclemem_runs/decomp_det_300/summary.md`

## MILP Verification

Referenced in the exact-small solver audit text. This optional run requires
`pulp` from `requirements-milp.txt`.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 100 \
  --budgets 0.02,0.05,0.10,0.20 \
  --distribution base,update_chain,temporal_interval \
  --methods opt \
  --solver milp \
  --verify-against exact_stdlib \
  --out oraclemem_runs/milp_verify_100_agent4
```

Expected outputs:

- `oraclemem_runs/milp_verify_100_agent4/raw_results.jsonl`
- `oraclemem_runs/milp_verify_100_agent4/summary.json`
- `oraclemem_runs/milp_verify_100_agent4/summary.md`
- `oraclemem_runs/milp_verify_100_agent4/REPORT.md`

## Gemini Flash-Lite Diagnostic

This API run is a robustness diagnostic, not a theorem-facing result. It uses
OpenRouter model `google/gemini-3.1-flash-lite-preview` and requires `api.env`.

```bash
python llm_memory_validation/longmemeval_reader_eval.py \
  --reader openrouter \
  --reader-model google/gemini-3.1-flash-lite-preview \
  --prompt-style answer_if_supported \
  --focus-only \
  --methods dense_budgeted_bsc,fifo_replay \
  --api-env api.env \
  --api-cache llm_memory_validation/openrouter_cache_gemini31_flash_lite_focus_full_bsc_fifo.json \
  --output-dir llm_memory_validation/longmemeval_reader_api_gemini31_flash_lite_focus_full_bsc_fifo \
  --api-max-tokens 320 \
  --api-timeout 120 \
  --temperature 0 \
  --request-sleep 0.02 \
  --bootstrap 1000 \
  --save-prompts
```

## Noisy Estimated-Policy Diagnostic

This run does not call an API. It records Gemini Flash-Lite as provenance for a
local noisy estimated-utility profile and is useful as a synthetic stress
diagnostic for non-oracle writer evaluation.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 500 \
  --distribution scope_shift_v2,density_trap_v2 \
  --budgets 4,6 \
  --methods opt,oracle_gvt,estimated_gvt,estimated_utility,mem0_extract,amac_admission,no_tombstone_gvt,no_tombstone_opt \
  --estimated-model google/gemini-3.1-flash-lite-preview \
  --estimated-profile noisy_gemini_flash_lite_v1 \
  --enable-retrieval \
  --retrieval fixed,oracle \
  --export-coverage-matrices \
  --coverage-package-limit 4 \
  --out-dir oraclemem_runs/estimated_policy_noisy_noapi_1000
```

To audit an exported coverage package:

```bash
python scripts/audit_coverage_artifacts.py \
  --no-defaults \
  --artifact exported_oraclemem_package=oraclemem_runs/estimated_policy_noisy_noapi_1000/coverage_instances/scope_shift_v2/seed_0 \
  --output-dir oraclemem_runs/estimated_policy_noisy_noapi_1000/coverage_audit
```

## Train/Dev Estimated-Writer Diagnostic

This local run trains a ridge utility estimator on synthetic train seeds and
evaluates `estimated_*` methods only on held-out dev seeds. It does not call an
API and is diagnostic rather than final deployed-writer evidence.

```bash
python run_oraclemem_mvp.py \
  --n-seeds 60 \
  --train-dev-estimator \
  --train-fraction 0.5 \
  --distribution base,update_chain,temporal_interval,density_trap,scope_shift,summary_tradeoff,redundancy_heavy,abstention_hard,scope_shift_v2,density_trap_v2 \
  --budgets 4,6 \
  --methods opt,oracle_gvt,estimated_gvt,estimated_utility,mem0_extract,amac_admission,no_tombstone_gvt,no_tombstone_opt \
  --out-dir oraclemem_runs/estimated_policy_train_dev_local_60
```

## Known Non-Reproducible Or External Pieces

- Local LaTeX compilation depends on a TeX distribution; this machine did not
  have `latexmk`, `pdflatex`, or `tectonic` on PATH.
- GPT-5.5 reader outputs require OpenRouter access, model availability, and API
  spending. Use the cached reader outputs for paper auditability.
- Gemini natural coverage and actual Mem0 smoke outputs require OpenRouter
  access if regenerated from scratch; use cached artifacts for audit where
  possible.
- LongMemEval-S retrieval regeneration downloads the dataset and
  `intfloat/e5-base-v2`; exact rows can drift if upstream artifacts change.
- API costs in `summary.json` are historical and should not be treated as a
  stable price quote.
