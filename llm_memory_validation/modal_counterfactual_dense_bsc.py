from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/project"
REMOTE_RESULTS = "/results"
REMOTE_HF_CACHE = "/root/.cache/huggingface"
IGNORE = [
    ".git",
    ".git-archives",
    "__pycache__",
    "dreamerv3/.venv",
    "dreamerv3/pilot_logs",
    "dreamerv3/smoke_logs",
    "dreamerv3/cw_modal_runs",
    "dreamerv3/paper_runs_smoke",
    "results*",
    "seq_results",
    "llm_memory_validation/modal_run",
    "llm_memory_validation/learned_run",
    "llm_memory_validation/competitor_run_v2",
    "llm_memory_validation/counterfactual_run",
]

app = modal.App("llm-memory-counterfactual-bsc")
results_volume = modal.Volume.from_name(
    "llm-memory-counterfactual-bsc-results", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    "llm-memory-counterfactual-bsc-hf-cache", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "accelerate>=1.6.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.9.0",
        "sentencepiece>=0.2.0",
        "safetensors>=0.4.5",
        "huggingface_hub[hf_transfer]>=0.30.2",
        "numpy>=2.0.0",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(ROOT, REMOTE_ROOT, copy=True, ignore=IGNORE)
)


def _stream_subprocess(command: list[str], cwd: str, env: dict[str, str], logfile: str) -> None:
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=12,
    memory=65536,
    timeout=60 * 60 * 6,
    volumes={
        REMOTE_RESULTS: results_volume,
        REMOTE_HF_CACHE: hf_cache_volume,
    },
)
def run_validation(
    budget_frac: float = 0.20,
    split_seed: int = 11,
    run_suffix: str = "utility_regressor",
    reader_model: str = "Qwen/Qwen2.5-3B-Instruct",
    retriever_model: str = "intfloat/e5-base-v2",
    prompt_word_budget: int = 1400,
    max_new_tokens: int = 40,
    controller_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = REMOTE_HF_CACHE
    env["HF_HUB_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["TRANSFORMERS_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")

    safe_suffix = "".join(char if char.isalnum() or char in "-_" else "_" for char in run_suffix)
    run_name = (
        f"counterfactual_{safe_suffix}_budget_{str(budget_frac).replace('.', 'p')}"
        f"_seed_{split_seed}"
    )
    output_dir = f"{REMOTE_RESULTS}/{run_name}"
    logfile = f"{output_dir}/stdout.log"
    command = [
        "python",
        "llm_memory_validation/counterfactual_dense_bsc.py",
        "--output-dir",
        output_dir,
        "--budget-frac",
        str(budget_frac),
        "--split-seed",
        str(split_seed),
        "--topk",
        "5",
        "--retriever-model",
        retriever_model,
        "--reader-model",
        reader_model,
        "--prompt-word-budget",
        str(prompt_word_budget),
        "--max-new-tokens",
        str(max_new_tokens),
        "--controller-seeds",
        *[str(seed) for seed in controller_seeds],
    ]

    _stream_subprocess(command, cwd=REMOTE_ROOT, env=env, logfile=logfile)
    results_volume.commit()
    hf_cache_volume.commit()

    summary_path = Path(output_dir) / "summary.json"
    report_path = Path(output_dir) / "REPORT.md"
    payload = {
        "run_name": run_name,
        "output_dir": output_dir,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "report_md": report_path.read_text(encoding="utf-8"),
        "stdout_log": logfile,
    }
    return payload


@app.local_entrypoint()
def main(
    budget_frac: float = 0.20,
    split_seed: int = 11,
    run_suffix: str = "utility_regressor",
    reader_model: str = "Qwen/Qwen2.5-3B-Instruct",
    retriever_model: str = "intfloat/e5-base-v2",
    prompt_word_budget: int = 1400,
    max_new_tokens: int = 40,
    controller_seeds: str = "0,1,2",
    background: bool = False,
) -> None:
    seeds = tuple(int(seed) for seed in controller_seeds.split(",") if seed)
    kwargs = {
        "budget_frac": budget_frac,
        "split_seed": split_seed,
        "run_suffix": run_suffix,
        "reader_model": reader_model,
        "retriever_model": retriever_model,
        "prompt_word_budget": prompt_word_budget,
        "max_new_tokens": max_new_tokens,
        "controller_seeds": seeds,
    }
    if background:
        call = run_validation.spawn(**kwargs)
        payload = {"function_call_id": call.object_id, "kwargs": kwargs}
    else:
        payload = run_validation.remote(**kwargs)
    print(json.dumps(payload, indent=2))
