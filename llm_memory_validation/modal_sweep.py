from __future__ import annotations
import json, os, subprocess
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/project"

IGNORE = [
    ".git", ".git-archives", "__pycache__",
    "dreamerv3/.venv", "dreamerv3/pilot_logs",
    "dreamerv3/smoke_logs", "dreamerv3/cw_modal_runs",
    "dreamerv3/paper_runs_smoke", "results*",
    "seq_results", "llm_memory_validation/modal_run",
    "llm_memory_validation/learned_run",
    "llm_memory_validation/competitor_run_v2",
    "llm_memory_validation/counterfactual_run",
    "llm_memory_validation/counterfactual_utility_regressor_run",
    "llm_memory_validation/counterfactual_staged_run",
    "llm_memory_validation/neurips_fast_results",
    "llm_memory_validation/neurips_micro_results",
    "llm_memory_validation/neurips_full_results",
]

app = modal.App("bsc-budget-sweep")

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
        "tqdm",
        "datasets",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MPLBACKEND": "Agg",
    })
    .add_local_dir(ROOT, REMOTE_ROOT, copy=True, ignore=IGNORE)
)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=12,
    memory=65536,
    timeout=60 * 60 * 4,
    volumes={
        "/results": results_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_sweep() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["HF_HUB_CACHE"] = "/root/.cache/huggingface/hub"
    env["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    output_dir = "/results/neurips_full_results"
    os.makedirs(output_dir, exist_ok=True)

    script = os.path.join(REMOTE_ROOT, "llm_memory_validation", "run_complete_sweep.py")
    result = subprocess.run(
        ["python", script],
        cwd=REMOTE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )

    results_volume.commit()

    results_path = Path(output_dir) / "full_results.json"
    payload = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
        "stderr_tail": result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr,
        "results_exist": results_path.exists(),
    }
    if results_path.exists():
        payload["results"] = json.loads(results_path.read_text(encoding="utf-8"))
        for fig in ["budget_sweep.png", "ablations.png"]:
            fp = Path(output_dir) / fig
            payload[f"{fig}_exists"] = fp.exists()
    return payload


@app.local_entrypoint()
def main():
    payload = run_sweep.remote()
    print(json.dumps(payload, indent=2, default=str))