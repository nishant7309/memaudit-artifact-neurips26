# MemAudit Evaluation Card

## Intended Use

MemAudit evaluates long-term LLM memory writers under an explicit storage
budget and a finite set of candidate memories. It is intended for measuring
write-time memory quality, comparing budgeted representation choices, auditing
validity-state/tombstone behavior, and diagnosing whether external memory stores
fail through extraction quality or budget-aware selection.

## Not Intended Use

MemAudit ratios are not end-to-end assistant quality guarantees. They are not
global optima over all possible memories, all possible natural-language
compressions, or all possible retrieval policies. LongMemEval reader/retrieval
numbers are downstream diagnostics, not exact oracle ratios.

## Denominators

- Package denominator: `OPT_P(B)`, the exact optimum for a finite MemAudit
  package.
- Union denominator: `OPT_{P^+(Y)}(B)`, the exact optimum after adding an
  external written store `Y` to the package candidate set.
- Upper-pruned bound: the best budget-feasible subset of an external store,
  used only as an extraction-versus-selection diagnostic.
- Retrieval/reader metrics: accuracy, recall, F1, abstention, stale-answer rate,
  and token cost; no exact OPT denominator is claimed.

## Main Package Artifacts

- Synthetic exact-small: `oraclemem_runs/exact_500`.
- Validity-heavy stress: `oraclemem_runs/stress_exact_500`.
- Representative non-oracle writers: `oraclemem_runs/representative_writers_500`.
- Natural support-sliced package: `llm_memory_validation/oraclemem_natural_200_gemini_v2`.
- Natural adjudicated subset: `llm_memory_validation/natural_adjudicated_100_gemini_flash`.
- Natural Flash-Lite spot-check: `llm_memory_validation/natural_spotcheck_30_gemini31_flash_lite`.
- Human-edited natural seed package: `llm_memory_validation/human_style_examples`.
- Learned writer transfer diagnostic: `llm_memory_validation/human_style_examples/learned_writer_transfer`.
- Natural writer adapters: `llm_memory_validation/natural_adjudicated_100_gemini_flash/writer_adapters`.
- Mem0 adjudicated rescore: `llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash`.

## Annotation Status

Exact synthetic coverage matrices are generated from the simulator and are
machine-checkable. Natural coverage packages are model-generated and
model-adjudicated, then audited with explicit reliability checks. The
87-example adjudicated subset includes a released human coverage audit:
two project annotators independently labeled all model-positive coverage cells
plus a stratified 20% sample of model-zero cells, yielding 1,975 doubly labeled
cells. Human-human agreement is substantial (`kappa=0.831`), Gemini agrees with
the human-consensus subset (`kappa=0.863`), and recomputing package-writer ratios
with human labels changes ratios by 0.035 on average while preserving reported
rankings. A Claude Sonnet 4.5 stress audit preserves package-writer rankings but
does not meet the binary-label agreement target, so it is reported as
model-sensitivity evidence rather than as a replacement annotator. The
`human_style_examples` package has been human-edited/audited and is structurally
validated, but it remains a fictional seed package rather than an
inter-annotator agreement benchmark.

## External Memory Systems

External stores such as Mem0 are evaluated with union-denominator diagnostics.
This prevents the invalid claim that an external writer should be measured
against a denominator that excludes its own candidate memories. The upper-pruned
upper is not a deployable method; it asks how much value is present in the
written store if budget selection were solved post hoc.

System-style local adapters such as Letta/MemGPT-style tiering and A-Mem-style
graph writing are evaluated as visible-metadata policies over package
candidates. They are denominator-matched baselines, not full published-system
executions. The checked-out Letta repository was inspected, but a true
Letta/MemGPT run requires a service/API/model configuration; the reported
MemGPT-style rows therefore remain local adapter rows.

The actual A-Mem run executes the checked-out public `AgenticMemory` code path
on the 87-example adjudicated package with Gemini Flash. It is intentionally labeled separately from the local
adapter rows: raw A-Mem notes are scored as full external memories, and a compact
metadata view is reported only as a diagnostic derived from A-Mem's generated
context/keywords/tags/links.

The human-edited examples are also exported to the same coverage-package schema
and used for an actual A-Mem run. That result is stronger than a purely
model-adjudicated package, but it remains a sanity check rather than an
inter-annotator benchmark because the examples are fictional and short.
The exported human package also has zero-API system-adapter rows: the
Letta/MemGPT-style adapter reaches 0.847 ratio to exact package OPT, while
density-only is 1.000, so this row is treated as a protocol check rather than a
separation result.

The adjudicated natural package includes a stronger faithful MemGPT/Letta union
baseline. It simulates core/archival/recall memory tiers over package-derived
written memories and scores against a package-plus-written-store union
denominator. It is still not a Letta server/API run, but it is closer to the
MemGPT memory architecture than the simple adapter.

## API Use

API calls are used for natural package construction, adjudication, external
store rescoring, and reader diagnostics. Exact synthetic labels and exact
synthetic optima are deterministic and do not depend on API calls. API costs and
cache files are recorded in the corresponding run directories.

## Learned Writer Status

The learned writer transfer diagnostic trains a local visible-feature estimator
from oracle labels on train packages, then evaluates held-out selections without
access to hidden coverage labels or query requirements. It is a deployable-writer
diagnostic, not a proof that learned writing is solved across natural traces.
The source ablations show that the current paper-facing estimator depends on
combining synthetic stress labels with Natural-200 labels; neither source alone
is sufficient on the human-edited package.

## Quickcheck

```powershell
python -m unittest test_oraclemem.py
python run_oraclemem_mvp.py --n-seeds 3 --budgets 4 --distribution base --methods opt,oracle_gvt,density_only --out-dir oraclemem_runs/quickcheck
```

## Release Checks

- Verify no API keys or private credentials are included.
- Verify paper-facing labels match `artifact_manifest.md`.
- Verify human-validation claims refer to the 87-example adjudicated natural
  subset and its released `human_coverage_audit` files.
- Verify greedy, retrieval, and reader diagnostics are not labeled as exact OPT.
