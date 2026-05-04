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
    "llm_memory_validation/counterfactual_utility_regressor_run",
    "llm_memory_validation/counterfactual_staged_run",
]

app = modal.App("neurips-bsc-experiments")

results_volume = modal.Volume.from_name("neurips-bsc-results", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("llm-memory-counterfactual-bsc-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "accelerate>=1.6.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.14.0",
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
            "MPLBACKEND": "Agg",
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
    timeout=60 * 60 * 8,
    volumes={
        REMOTE_RESULTS: results_volume,
        REMOTE_HF_CACHE: hf_cache_volume,
    },
)
def run_full_neurips_suite(
    budget_frac: float = 0.20,
    split_seed: int = 11,
    controller_seeds: tuple[int, ...] = (0, 1, 2),
    retriever_model: str = "intfloat/e5-base-v2",
    budget_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40),
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = REMOTE_HF_CACHE
    env["HF_HUB_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["TRANSFORMERS_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["MPLBACKEND"] = "Agg"

    output_dir = f"{REMOTE_RESULTS}/neurips_full_suite"
    logfile = f"{output_dir}/stdout.log"

    command = [
        "python", "llm_memory_validation/neurips_experiments.py",
        "--output-dir", output_dir,
        "--budget-frac", str(budget_frac),
        "--split-seed", str(split_seed),
        "--topk", "5",
        "--retriever-model", retriever_model,
        "--controller-seeds", *[str(s) for s in controller_seeds],
        "--budget-fractions", *[str(f) for f in budget_fractions],
    ]

    _stream_subprocess(command, cwd=REMOTE_ROOT, env=env, logfile=logfile)
    results_volume.commit()

    results_path = Path(output_dir) / "neurips_results.json"
    report_path = Path(output_dir) / "NEURIPS_REPORT.md"
    payload = {
        "output_dir": output_dir,
        "results_exist": results_path.exists(),
        "report_exist": report_path.exists(),
    }
    if results_path.exists():
        payload["results"] = json.loads(results_path.read_text(encoding="utf-8"))
    if report_path.exists():
        payload["report"] = report_path.read_text(encoding="utf-8")
    return payload


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 4,
    volumes={
        REMOTE_RESULTS: results_volume,
        REMOTE_HF_CACHE: hf_cache_volume,
    },
)
def run_theory_only(
    split_seed: int = 11,
    retriever_model: str = "intfloat/e5-base-v2",
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = REMOTE_HF_CACHE
    env["HF_HUB_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["TRANSFORMERS_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["MPLBACKEND"] = "Agg"

    output_dir = f"{REMOTE_RESULTS}/neurips_theory"
    logfile = f"{output_dir}/stdout.log"

    command = [
        "python", "llm_memory_validation/neurips_experiments.py",
        "--output-dir", output_dir,
        "--split-seed", str(split_seed),
        "--retriever-model", retriever_model,
        "--skip-budget-sweep",
        "--skip-stat-tests",
        "--skip-retriever-swap",
        "--skip-adversarial",
    ]

    _stream_subprocess(command, cwd=REMOTE_ROOT, env=env, logfile=logfile)
    results_volume.commit()

    results_path = Path(output_dir) / "neurips_results.json"
    payload = {"output_dir": output_dir}
    if results_path.exists():
        payload["results"] = json.loads(results_path.read_text(encoding="utf-8"))
    return payload


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
def run_budget_sweep_only(
    budget_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40),
    split_seed: int = 11,
    controller_seeds: tuple[int, ...] = (0, 1, 2),
    retriever_model: str = "intfloat/e5-base-v2",
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = REMOTE_HF_CACHE
    env["HF_HUB_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["TRANSFORMERS_CACHE"] = str(Path(REMOTE_HF_CACHE) / "hub")
    env["MPLBACKEND"] = "Agg"

    output_dir = f"{REMOTE_RESULTS}/neurips_budget_sweep"
    logfile = f"{output_dir}/stdout.log"

    command = [
        "python", "llm_memory_validation/neurips_experiments.py",
        "--output-dir", output_dir,
        "--split-seed", str(split_seed),
        "--retriever-model", retriever_model,
        "--controller-seeds", *[str(s) for s in controller_seeds],
        "--budget-fractions", *[str(f) for f in budget_fractions],
        "--skip-theory",
        "--skip-stat-tests",
        "--skip-retriever-swap",
        "--skip-adversarial",
    ]

    _stream_subprocess(command, cwd=REMOTE_ROOT, env=env, logfile=logfile)
    results_volume.commit()

    results_path = Path(output_dir) / "neurips_results.json"
    payload = {"output_dir": output_dir}
    if results_path.exists():
        payload["results"] = json.loads(results_path.read_text(encoding="utf-8"))
    return payload


@app.local_entrypoint()
def main(
    phase: str = "full",
    budget_frac: float = 0.20,
    split_seed: int = 11,
    retriever_model: str = "intfloat/e5-base-v2",
    background: bool = False,
):
    if phase == "theory":
        fn = run_theory_only
        kwargs = {"split_seed": split_seed, "retriever_model": retriever_model}
    elif phase == "sweep":
        fn = run_budget_sweep_only
        kwargs = {
            "budget_fractions": (0.10, 0.15, 0.20, 0.30, 0.40),
            "split_seed": split_seed,
            "controller_seeds": (0, 1, 2),
            "retriever_model": retriever_model,
        }
    else:
        fn = run_full_neurips_suite
        kwargs = {
            "budget_frac": budget_frac,
            "split_seed": split_seed,
            "controller_seeds": (0, 1, 2),
            "retriever_model": retriever_model,
            "budget_fractions": (0.10, 0.15, 0.20, 0.30, 0.40),
        }

    if background:
        call = fn.spawn(**kwargs)
        print(f"Spawned background job: {call.object_id}")
        print(json.dumps({"function_call_id": call.object_id, "kwargs": kwargs}, indent=2))
    else:
        payload = fn.remote(**kwargs)
        print(json.dumps(payload, indent=2, default=str))