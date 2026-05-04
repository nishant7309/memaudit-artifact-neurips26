---
license: mit
pretty_name: MemAudit Code Artifact
tags:
- llm-memory
- benchmark
- evaluation
- reproducibility
---

# MemAudit

MemAudit is an exact-oracle evaluation protocol for budgeted long-term LLM
memory writing. The core question is finite and package-conditional:

> Given a fixed storage budget and a finite semantic evidence package, how close
> is a written memory store to the best package-feasible store?

This repository contains the manuscript source, exact-oracle implementation,
writer baselines, scoring utilities, figure scripts, and reproducibility
commands. The larger cached datasets and run artifacts are released separately
through the dataset artifact referenced in the paper.

MemAudit is not a runtime memory product. It is an evaluation layer for
memory writers: it scores finite candidate packages, budgeted representation
choices, and external written stores against explicit denominators.

## Quickcheck

Run the deterministic tests:

```powershell
python -m unittest test_oraclemem.py
```

Run a tiny exact-oracle smoke benchmark:

```powershell
python run_oraclemem_mvp.py --n-seeds 3 --budgets 4 --distribution base --methods opt,oracle_gvt,density_only --out-dir oraclemem_runs/quickcheck
```

Expected smoke outputs:

- `oraclemem_runs/quickcheck/raw_results.jsonl`
- `oraclemem_runs/quickcheck/summary.json`
- `oraclemem_runs/quickcheck/summary.md`

## Main Code Components

- `main.tex`: active manuscript.
- `references.bib`: bibliography.
- `figures/`: paper figure assets generated from cached experiment summaries.
- `oraclemem/`: package schema, objective, exact solvers, and writer baselines.
- `llm_memory_validation/`: natural-package construction, exported-system scoring, and reader/retrieval diagnostics.
- `scripts/`: figure generation, artifact packaging, coverage audits, and cross-model checks.
- `run_oraclemem_mvp.py`: exact package benchmark runner.
- `test_oraclemem.py`: deterministic unit and solver checks.

See `artifact_manifest.md` for table-to-artifact mapping and full rerun
commands. Some commands require the separate dataset artifact. See
`REPRODUCIBILITY.md` for setup, exact-oracle runs, API runs, and known local
build limitations.

## Denominator Types

- Package ratio: exact ratio to `OPT_P(B)` for a finite MemAudit candidate package.
- Union ratio: exact ratio to `OPT_{P^+(Y)}(B)` after adding an external written store to the candidate package.
- Upper-pruned bound: best budget-feasible subset of an external store, used only to separate extraction quality from budget-aware selection.
- Retrieval/reader metrics: downstream diagnostics, not MemAudit optimum ratios.

## Caveats

The strongest exact claims are finite-package claims. LongMemEval-derived
natural coverage packages are model-adjudicated; the separate
`human_style_examples` package is human-edited/audited but does not include an
inter-annotator agreement file. LongMemEval reader/retrieval results
are downstream diagnostics and do not have exact OPT denominators. Mem0 and
A-Mem rescoring use union-denominator and upper-pruned-bound diagnostics rather
than claiming deployable optimal pruning policies.

Do not commit API keys. `api.env` is local-only and should stay ignored.
