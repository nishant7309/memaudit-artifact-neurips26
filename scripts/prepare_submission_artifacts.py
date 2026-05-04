"""Build clean anonymous MemAudit submission artifacts.

This script creates two local folders:

* ``memaudit_submission_package/code``: executable code, paper sources, tests,
  docs, and figures.
* ``memaudit_submission_package/dataset``: curated data/results needed for the
  E&D dataset URL and Croissant metadata.

It intentionally fails if the output directory already exists so we do not
accidentally delete or overwrite local work.
"""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "memaudit_submission_package"
CODE = OUT / "code"
DATASET = OUT / "dataset"


CODE_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "artifact_manifest.md",
    "EVALUATION_CARD.md",
    "requirements.txt",
    "requirements-api.txt",
    "requirements-milp.txt",
    "run_oraclemem_mvp.py",
    "test_oraclemem.py",
    "main.tex",
    "checklist.tex",
    "references.bib",
    "neurips_2026.sty",
    "main.pdf",
]

CODE_DIRS = [
    "oraclemem",
    "figures",
]

CODE_GLOBS = [
    "scripts/*.py",
    "llm_memory_validation/*.py",
    "llm_memory_validation/patches/*",
]

DATASET_PATHS = [
    "oraclemem_runs/exact_500",
    "oraclemem_runs/stress_exact_500",
    "oraclemem_runs/milp_verify_100_agent4",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/REPORT.md",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/summary.json",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/summary.md",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/raw_results.jsonl",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/resolved_examples.jsonl",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_resolved_summary.json",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_resolution_report.md",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/run_manifest.json",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_package",
    "llm_memory_validation/oraclemem_natural_200_gemini_v2/coverage_audit",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/REPORT.md",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/summary.json",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/summary.md",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/raw_results.jsonl",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/adjudication_summary.json",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/adjudication_decisions.jsonl",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/selected_queries.jsonl",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_package",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/coverage_audit",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/sensitivity",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/REPORT.md",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/summary.json",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/summary.md",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/raw_results.jsonl",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/adjudication_summary.json",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/adjudication_decisions.jsonl",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/selected_queries.jsonl",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/coverage_package",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cross_model_check",
    "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cell_coverage_audit",
    "llm_memory_validation/natural_spotcheck_30_gemini31_flash_lite",
    "llm_memory_validation/human_style_examples/README.md",
    "llm_memory_validation/human_style_examples/examples_100.jsonl",
    "llm_memory_validation/human_style_examples/eval_package_100",
    "llm_memory_validation/human_style_examples/coverage_package",
    "llm_memory_validation/human_style_examples/coverage_package_audit",
    "llm_memory_validation/human_style_examples/writer_adapters",
    "llm_memory_validation/human_style_examples/learned_writer_transfer",
    "llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/faithful_memgpt_letta_union",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87",
    "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_amem_gemini_flash_87",
    "llm_memory_validation/longmemeval_focus_report_core4",
    "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/REPORT.md",
    "llm_memory_validation/longmemeval_reader_api_gpt55_answer_supported_focus_full/summary.json",
]

EXCLUDE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "openrouter_cache_gemini_natural_oraclemem.json",
    "openrouter_cache_adjudication.json",
    "openrouter_cache_claude_cell_coverage.json",
    "api.env",
}


def copy_path(src_rel: str, dst_root: Path) -> None:
    src = ROOT / src_rel
    if not src.exists():
        return
    dst = dst_root / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore)
    else:
        shutil.copy2(src, dst)


def ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if (
            name in EXCLUDE_NAMES
            or name.startswith("openrouter_cache")
            or name.endswith(("_cache.json", ".pyc", ".pyo", ".log"))
        ):
            ignored.add(name)
    return ignored


def write_readmes() -> None:
    (OUT / "README.md").write_text(
        "# MemAudit Anonymous Submission Package\n\n"
        "This folder is split into `code/` and `dataset/`.\n\n"
        "- `code/` contains the executable benchmark/evaluation artifact, paper sources, tests, and figure scripts.\n"
        "- `dataset/` contains the curated package/result artifacts referenced by the paper and Croissant metadata.\n\n"
        "Do not upload local API keys or uncurated review notes. The package was generated by "
        "`python scripts/prepare_submission_artifacts.py`.\n",
        encoding="utf-8",
    )
    (CODE / "LICENSE").write_text(
        "MIT License\n\n"
        "Copyright (c) 2026 Anonymous\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n",
        encoding="utf-8",
    )
    code_readme = (CODE / "README.md").read_text(encoding="utf-8")
    if code_readme.startswith("---"):
        code_readme = code_readme.split("---", 2)[2].lstrip()
    (CODE / "README.md").write_text(
        "---\n"
        "license: mit\n"
        "pretty_name: MemAudit Code Artifact\n"
        "tags:\n"
        "- llm-memory\n"
        "- benchmark\n"
        "- evaluation\n"
        "- reproducibility\n"
        "---\n\n"
        + code_readme,
        encoding="utf-8",
    )
    (DATASET / "README.md").write_text(
        "---\n"
        "license: cc-by-4.0\n"
        "pretty_name: MemAudit Benchmark Artifacts\n"
        "task_categories:\n"
        "- question-answering\n"
        "- text-retrieval\n"
        "tags:\n"
        "- llm-memory\n"
        "- benchmark\n"
        "- evaluation\n"
        "- croissant\n"
        "---\n\n"
        "# MemAudit Dataset Artifact\n\n"
        "This dataset contains finite memory-writing packages, exact/verified solver outputs, "
        "model-adjudicated natural support-slice packages, human-edited fictional seed packages, "
        "and exported-system diagnostic results for Mem0, Letta, and A-Mem. It is intended to "
        "support the NeurIPS 2026 E&D submission `MemAudit: An Exact-Oracle Evaluation Protocol "
        "for Budgeted Long-Term LLM Memory Writing`.\n\n"
        "Historical directory names may contain `oraclemem`; paper-facing terminology is `MemAudit`.\n",
        encoding="utf-8",
    )
    (DATASET / "LICENSE").write_text(
        "MemAudit Dataset Artifact License\n\n"
        "SPDX-License-Identifier: CC-BY-4.0\n\n"
        "The dataset artifacts in this folder are released under the Creative Commons\n"
        "Attribution 4.0 International License: https://creativecommons.org/licenses/by/4.0/\n\n"
        "This license applies to the curated MemAudit package metadata, synthetic\n"
        "examples, generated benchmark packages, and released diagnostic outputs. Source\n"
        "benchmark excerpts and third-party system names remain subject to their original\n"
        "licenses and terms where applicable.\n",
        encoding="utf-8",
    )


def write_dataset_manifest() -> str:
    """Write a compact machine-readable manifest used by Croissant metadata."""
    records = [
        {
            "artifact_path": "oraclemem_runs/exact_500",
            "artifact_type": "controlled_exact_packages",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "oraclemem_runs/validity_500",
            "artifact_type": "validity_stress_packages",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/natural_adjudicated_100_gemini_flash",
            "artifact_type": "natural_support_slices",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/natural_adjudicated_100_gemini_flash/sensitivity",
            "artifact_type": "sensitivity_audit",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit",
            "artifact_type": "human_coverage_audit",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cross_model_check",
            "artifact_type": "cross_model_stress_audit",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cell_coverage_audit",
            "artifact_type": "cross_model_cell_audit",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/mem0_rescore_adjudicated100_gemini_flash",
            "artifact_type": "exported_system_diagnostics",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "llm_memory_validation/natural_adjudicated_100_gemini_flash/actual_letta_openrouter_gemini_passage_87",
            "artifact_type": "exported_system_diagnostics",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
        {
            "artifact_path": "README_REPO.md",
            "artifact_type": "documentation",
            "budget": None,
            "objective_value": None,
            "ratio": None,
        },
    ]
    path = DATASET / "artifact_records.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_croissant_placeholder() -> None:
    # URL fields should be replaced after the dataset is hosted under an
    # anonymized reviewer-accessible URL.
    manifest_sha256 = write_dataset_manifest()
    metadata = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "equivalentProperty": "cr:equivalentProperty",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "regex": "cr:regex",
            "samplingRate": "cr:samplingRate",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "sc:Dataset",
        "name": "MemAudit benchmark artifacts",
        "description": (
            "Finite package artifacts for exact-oracle evaluation of budgeted long-term LLM "
            "memory writing, including exact-small packages, validity stress packages, "
            "model-adjudicated natural support slices, human-edited fictional seed examples, "
            "and exported-system diagnostic results."
        ),
        "url": "TO_BE_FILLED_WITH_ANONYMIZED_DATASET_URL",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "version": "0.1.0",
        "datePublished": "2026-05-03",
        "citeAs": "Anonymous. MemAudit: An Exact-Oracle Evaluation Protocol for Budgeted Long-Term LLM Memory Writing. NeurIPS 2026 submission artifact.",
        "creator": [{"@type": "Organization", "name": "Anonymous"}],
        "keywords": [
            "LLM memory",
            "long-term memory",
            "benchmark",
            "exact oracle",
            "budgeted memory writing",
        ],
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "rai:dataLimitations": (
            "Scores are package-conditional and do not measure unconstrained end-to-end assistant "
            "quality. Natural support-slice labels are model-adjudicated with audit hooks; the "
            "human-edited seed package is fictional and schema-validating rather than a human-subject dataset."
        ),
        "rai:dataBiases": (
            "Synthetic generators emphasize memory-writing, update, and validity-state phenomena. "
            "Support-sliced natural packages reflect the source benchmark and model-adjudication pipeline."
        ),
        "rai:personalSensitiveInformation": (
            "The synthetic and human-edited seed examples are fictional. Natural support-slice artifacts "
            "derive from benchmark text and should be used for evaluation research, not for profiling individuals."
        ),
        "rai:dataUseCases": (
            "Intended for evaluating write-time memory policies under finite candidate sets, costs, "
            "coverage matrices, and exact or certified budgeted optima. Not intended as a deployed "
            "personal-memory product or as a standalone measure of assistant quality."
        ),
        "rai:dataSocialImpact": (
            "The artifact can improve transparency of long-term memory systems by separating writing, "
            "retrieval, and reader failures. Potential misuse includes optimizing systems to retain "
            "sensitive personal information; validity, deletion, and abstention units are included as "
            "mitigation-oriented diagnostics."
        ),
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": [
            "Synthetic MemAudit generators",
            "LongMemEval-style support slices",
            "Public memory-system exports and diagnostics",
        ],
        "prov:wasGeneratedBy": (
            "Generated by deterministic package generators and documented API/model-adjudication scripts. "
            "See README.md, REPRODUCIBILITY.md, artifact_manifest.md, and run manifests."
        ),
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "artifact_records_jsonl",
                "name": "artifact_records_jsonl",
                "description": "Compact manifest of the main MemAudit dataset artifact groups.",
                "contentUrl": "TO_BE_FILLED_WITH_ANONYMIZED_DATASET_URL/resolve/main/artifact_records.jsonl",
                "encodingFormat": "application/jsonlines",
                "sha256": manifest_sha256,
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "package_results",
                "name": "Package and result artifacts",
                "description": "JSONL/JSON/Markdown records for exact package scoring and exported-system diagnostics.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "artifact_path",
                        "name": "artifact_path",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "artifact_records_jsonl"}, "extract": {"jsonPath": "$.artifact_path"}},
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "artifact_type",
                        "name": "artifact_type",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "artifact_records_jsonl"}, "extract": {"jsonPath": "$.artifact_type"}},
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "budget",
                        "name": "budget",
                        "dataType": "sc:Float",
                        "source": {"fileObject": {"@id": "artifact_records_jsonl"}, "extract": {"jsonPath": "$.budget"}},
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "objective_value",
                        "name": "objective_value",
                        "dataType": "sc:Float",
                        "source": {"fileObject": {"@id": "artifact_records_jsonl"}, "extract": {"jsonPath": "$.objective_value"}},
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "ratio",
                        "name": "ratio",
                        "dataType": "sc:Float",
                        "source": {"fileObject": {"@id": "artifact_records_jsonl"}, "extract": {"jsonPath": "$.ratio"}},
                    },
                ],
            }
        ],
    }
    (DATASET / "croissant_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT}")
    CODE.mkdir(parents=True)
    DATASET.mkdir(parents=True)

    for rel in CODE_FILES:
        copy_path(rel, CODE)
    for rel in CODE_DIRS:
        copy_path(rel, CODE)
    for pattern in CODE_GLOBS:
        for src in ROOT.glob(pattern):
            if src.is_file():
                copy_path(str(src.relative_to(ROOT)), CODE)

    for rel in DATASET_PATHS:
        copy_path(rel, DATASET)

    write_readmes()
    shutil.copy2(ROOT / "README.md", DATASET / "README_REPO.md")
    shutil.copy2(ROOT / "REPRODUCIBILITY.md", DATASET / "REPRODUCIBILITY.md")
    shutil.copy2(ROOT / "artifact_manifest.md", DATASET / "artifact_manifest.md")
    shutil.copy2(ROOT / "EVALUATION_CARD.md", DATASET / "EVALUATION_CARD.md")
    write_croissant_placeholder()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
