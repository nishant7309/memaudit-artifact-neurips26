"""Run a Claude Sonnet cell-level coverage audit through OpenRouter.

The input is the frozen human coverage audit sampling frame. The model sees the
candidate memory text and the evidence-unit text, but not Gemini's label or
rationale. It returns binary labels for the same cells that human annotators
labeled, enabling a second-model-vs-human check on exactly the human audit task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_AUDIT_DIR = Path("llm_memory_validation/natural_adjudicated_100_gemini_flash/human_coverage_audit")
DEFAULT_OUT_DIR = Path("llm_memory_validation/natural_adjudicated_100_claude_sonnet45/cell_coverage_audit")
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def extract_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(stripped[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class OpenRouterJsonClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cache_path: Path,
        max_tokens: int,
        temperature: float,
        timeout: float,
        request_sleep: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.cache_path = cache_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.request_sleep = request_sleep
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")

    def __call__(self, prompt: str, *, purpose: str) -> dict[str, Any]:
        settings = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "purpose": purpose,
        }
        prompt_hash = stable_hash(json.dumps(settings, sort_keys=True) + "\n" + prompt)
        if prompt_hash in self.cache:
            cached = dict(self.cache[prompt_hash])
            cached["cache_hit"] = True
            cached["prompt_hash"] = prompt_hash
            return cached

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/memaudit",
                "X-Title": "MemAudit Claude Cell Coverage Audit",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {error.code}: {details}") from error

        content = body["choices"][0]["message"].get("content")
        result = {
            "cache_hit": False,
            "prompt_hash": prompt_hash,
            "raw_content": content,
            "parsed": extract_json_object(content),
            "usage": body.get("usage", {}),
            "model": self.model,
        }
        self.cache[prompt_hash] = {key: value for key, value in result.items() if key != "prompt_hash"}
        self._write_cache()
        if self.request_sleep:
            time.sleep(self.request_sleep)
        return result


def chunked(rows: Sequence[Mapping[str, str]], size: int) -> list[list[Mapping[str, str]]]:
    return [list(rows[i : i + size]) for i in range(0, len(rows), size)]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def make_prompt(rows: Sequence[Mapping[str, str]]) -> str:
    payload = {
        "task": "coverage_cell_binary_annotation",
        "label_definition": (
            "Return label=1 iff the candidate memory would let a future memory reader recover "
            "the evidence unit. Exact wording is not required, but the memory must entail the "
            "substance of the evidence unit. Return label=0 for unsupported, contradicted, too "
            "vague, or merely topically related memories."
        ),
        "partial_coverage_rule": (
            "If the memory preserves a useful but incomplete part of the evidence unit, label 1 "
            "only when the preserved content would still help answer a future query requiring "
            "that evidence. Otherwise label 0."
        ),
        "cells": [
            {
                "cell_id": row["cell_id"],
                "representation_type": row.get("representation_type", ""),
                "candidate_memory": truncate_words(row.get("candidate_text", ""), 180),
                "evidence_unit": truncate_words(row.get("evidence_text", ""), 80),
            }
            for row in rows
        ],
    }
    return (
        "You are independently annotating MemAudit memory-coverage cells.\n"
        "You must not infer from any hidden labels; use only the candidate_memory and evidence_unit shown.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "labels": [\n'
        '    {"cell_id": "...", "label": 0, "rationale": "short reason"}\n'
        "  ]\n"
        "}\n\n"
        f"ANNOTATION_BATCH:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def parse_labels(parsed: Mapping[str, Any], allowed_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in parsed.get("labels", []) or []:
        if not isinstance(item, Mapping):
            continue
        cell_id = str(item.get("cell_id", ""))
        if cell_id not in allowed_ids:
            continue
        raw_label = item.get("label", 0)
        if isinstance(raw_label, str):
            label = 1 if raw_label.strip().lower() in {"1", "yes", "y", "true", "cover", "covered"} else 0
        else:
            label = 1 if int(raw_label or 0) else 0
        rows.append(
            {
                "cell_id": cell_id,
                "annotator_id": "claude_sonnet_4_5",
                "label": label,
                "notes": "openrouter_anthropic_claude_sonnet_4_5",
                "rationale": str(item.get("rationale", "")),
            }
        )
    return rows


def fallback_labels_for_missing(
    client: OpenRouterJsonClient,
    batch: Sequence[Mapping[str, str]],
    missing_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows_by_id = {str(row["cell_id"]): row for row in batch}
    labels: list[dict[str, Any]] = []
    for cell_id in missing_ids:
        row = rows_by_id[cell_id]
        response = client(
            make_prompt([row]),
            purpose="claude_cell_coverage_audit_single_retry",
        )
        parsed_labels = parse_labels(response.get("parsed", {}), {cell_id})
        if len(parsed_labels) != 1:
            raise RuntimeError(f"single-cell retry failed for {cell_id}")
        labels.extend(parsed_labels)
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-sleep", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    for key, value in load_env_file(args.api_env).items():
        os.environ.setdefault(key, value)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required in api.env or environment")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_csv(args.audit_dir / "annotation_cells.csv")
    client = OpenRouterJsonClient(
        api_key=api_key,
        model=args.model,
        cache_path=args.out_dir / "openrouter_cache_claude_cell_coverage.json",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        request_sleep=args.request_sleep,
    )

    all_labels: list[dict[str, Any]] = []
    batches = chunked(cells, args.batch_size)
    for index, batch in enumerate(batches, start=1):
        marker = args.out_dir / "per_batch" / f"batch_{index:04d}.json"
        if args.skip_existing and marker.exists():
            cached = json.loads(marker.read_text(encoding="utf-8"))
            all_labels.extend(cached["labels"])
            continue
        response = client(make_prompt(batch), purpose="claude_cell_coverage_audit")
        labels = parse_labels(response.get("parsed", {}), {str(row["cell_id"]) for row in batch})
        if len(labels) != len(batch):
            missing = sorted({str(row["cell_id"]) for row in batch} - {row["cell_id"] for row in labels})
            labels.extend(fallback_labels_for_missing(client, batch, missing))
        if len(labels) != len(batch):
            missing = sorted({str(row["cell_id"]) for row in batch} - {row["cell_id"] for row in labels})
            raise RuntimeError(f"batch {index} returned {len(labels)}/{len(batch)} labels after retry; missing {missing[:5]}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "batch_index": index,
                    "model": args.model,
                    "prompt_hash": response.get("prompt_hash"),
                    "cache_hit": response.get("cache_hit"),
                    "usage": response.get("usage", {}),
                    "labels": labels,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        all_labels.extend(labels)
        print(f"batch {index}/{len(batches)} labels={len(labels)} cache_hit={response.get('cache_hit')}")

    write_csv(
        args.out_dir / "claude_cell_labels.csv",
        all_labels,
        ["cell_id", "annotator_id", "label", "notes", "rationale"],
    )
    usage_rows = []
    for path in sorted((args.out_dir / "per_batch").glob("batch_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        usage = row.get("usage", {}) or {}
        usage_rows.append(
            {
                "batch": path.stem,
                "prompt_hash": row.get("prompt_hash"),
                "cache_hit": row.get("cache_hit"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    write_csv(
        args.out_dir / "usage.csv",
        usage_rows,
        ["batch", "prompt_hash", "cache_hit", "prompt_tokens", "completion_tokens", "total_tokens"],
    )
    print(json.dumps({"out_dir": str(args.out_dir), "cells": len(cells), "labels": len(all_labels), "batches": len(batches)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
