"""Run a minimal actual Mem0 smoke test with Gemini via OpenRouter.

This is intentionally a smoke test, not a benchmark. It verifies that the
public Mem0 codebase can execute in this environment with:

* OpenRouter/Gemini as the LLM backend;
* local HuggingFace embeddings;
* local Qdrant storage.

The script writes JSON outputs under ``llm_memory_validation/mem0_actual_smoke``
and never prints API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_config(out_dir: Path, model: str) -> dict[str, Any]:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "temperature": 0.0,
                "max_tokens": 700,
                "openrouter_base_url": "https://openrouter.ai/api/v1",
                "site_url": "https://localhost/oraclemem",
                "app_name": "OracleMem Mem0 Baseline Smoke",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "oraclemem_mem0_smoke",
                "path": str(out_dir / "qdrant"),
                "embedding_model_dims": 384,
            },
        },
        "history_db_path": str(out_dir / "history.db"),
        "version": "v1.1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--out-dir", type=Path, default=Path("llm_memory_validation/mem0_actual_smoke"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reuse-store", action="store_true")
    args = parser.parse_args()

    load_env_file(args.api_env)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment or api.env")

    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    if args.out_dir.exists() and not args.reuse_store:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from mem0 import Memory

    config = build_config(args.out_dir, args.model)
    status: dict[str, Any] = {"ok": False, "stage": "init", "model": args.model}
    try:
        memory = Memory.from_config(config)
        if not args.reuse_store:
            status["stage"] = "add"
            add_result = memory.add(
                [
                    {"role": "user", "content": "I moved to Seattle last month. I prefer vegetarian restaurants."},
                    {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": "Actually, I now live in Portland, but I still prefer vegetarian food."},
                ],
                user_id="oraclemem_smoke",
            )
        else:
            add_result = {"skipped": True}

        filters = {"user_id": "oraclemem_smoke"}
        status["stage"] = "get_all"
        all_result = memory.get_all(filters=filters, top_k=20)
        status["stage"] = "search"
        search_result = memory.search(
            query="Where do I live now and what food do I prefer?",
            filters=filters,
            top_k=5,
        )
        status.update(
            {
                "ok": True,
                "stage": "done",
                "add_result": add_result,
                "all_result": all_result,
                "search_result": search_result,
            }
        )
    except Exception as exc:
        status.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

    (args.out_dir / "search_result.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: status[k] for k in status if k not in {"add_result", "all_result", "search_result"}}, indent=2))
    if not status["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
