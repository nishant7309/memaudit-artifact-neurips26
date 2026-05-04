"""Build a Gemini-annotated natural OracleMem pilot from LongMemEval-S.

This script is intentionally separate from the synthetic OracleMem runner.  It
uses Gemini through OpenRouter to create an auditable natural-trace coverage
package:

* candidate memories are generated from conversation sessions only;
* query/gold answers are used only in a separate annotation step that maps
  extracted evidence units to the evaluation question;
* exact OPT is solved over the resulting finite candidate set;
* local published-system-inspired writer policies are scored under the same
  candidate set and budget.

The default run is a small pilot.  Scale `--limit` only after checking cache hit
rate, cost, and package quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import string
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oraclemem.evaluate import (
    CandidateMemory,
    OracleMemInstance,
    SelectionResult,
    evaluate_instance,
    write_benchmark_outputs,
)


FOCUS_TYPES = {"knowledge-update", "temporal-reasoning"}
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_METHODS = (
    "opt",
    "oracle_gvt",
    "memgpt_tiered",
    "mem0_extract",
    "amem_graph",
    "amac_admission",
    "recency_raw",
    "summary_only",
    "fact_only",
)


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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_token(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._") or "item"


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def truncate_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " ..."


def extract_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OpenRouterJsonClient:
    """Small cached OpenRouter JSON client for Gemini annotation."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cache_path: Path,
        max_tokens: int = 1400,
        temperature: float = 0.0,
        timeout: int = 120,
        request_sleep: float = 0.02,
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
                "HTTP-Referer": "https://localhost/oraclemem",
                "X-Title": "OracleMem Natural Coverage Pilot",
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
        parsed = extract_json_object(content)
        result = {
            "cache_hit": False,
            "prompt_hash": prompt_hash,
            "purpose": purpose,
            "model": self.model,
            "parsed": parsed,
            "raw_content": content,
            "usage": body.get("usage", {}),
            "provider": body.get("provider"),
        }
        self.cache[prompt_hash] = result
        self._write_cache()
        if self.request_sleep > 0:
            time.sleep(self.request_sleep)
        return result


@dataclass(frozen=True)
class GeneratedSession:
    session_id: str
    date: str
    source_kind: str
    text: str
    response: dict[str, Any]
    prompt_hash: str
    cache_hit: bool
    usage: Mapping[str, Any]


def session_text(turns: Sequence[Mapping[str, Any]], *, max_words: int) -> str:
    lines: list[str] = []
    for turn in turns:
        role = str(turn.get("role", "unknown")).strip() or "unknown"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return truncate_words("\n".join(lines), max_words)


def session_prompt(session_id: str, date: str, text: str) -> str:
    return f"""You are constructing a write-time memory benchmark from one conversation session.

Do not use any hidden question or answer. Use only the session text below.

Extract up to 4 source-backed evidence units that could matter for future long-term memory questions. Then generate alternative candidate memory representations for this same session:
- one Mem0-style atomic fact candidate, if useful;
- one A-Mem-style graph/linked note candidate, if useful;
- one MemGPT-style compact summary candidate, if useful;
- one tombstone/update candidate only if the session explicitly corrects, supersedes, invalidates, or updates prior information.

Every candidate must list which evidence unit ids it supports. Use only ids you created. Do not invent facts unsupported by the session.

Return exactly JSON:
{{
  "evidence_units": [
    {{
      "unit_id": "u1",
      "kind": "current_fact|temporal_fact|preference|update|abstention|other",
      "canonical_text": "...",
      "source_quote": "short exact quote from session",
      "importance": 0.5
    }}
  ],
  "candidates": [
    {{
      "candidate_id": "c1",
      "representation_type": "atomic_fact|graph_edge|summary|tombstone|compound_update",
      "generator": "gemini_mem0|gemini_amem|gemini_memgpt|gemini_validity",
      "text": "...",
      "covers_unit_ids": ["u1"],
      "confidence": 0.8
    }}
  ]
}}

Session id: {session_id}
Session date: {date}
Session text:
{text}
"""


def query_prompt(question: str, answer: str, units: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "unit_id": row["unit_id"],
            "canonical_text": row["canonical_text"],
            "source_quote": row.get("source_quote", ""),
            "session_id": row.get("session_id", ""),
        }
        for row in units
    ]
    return f"""You are annotating a long-term memory evaluation question.

Select the minimal evidence unit ids needed to answer the question. Use the gold answer only for annotation.
A set of units is sufficient if a careful reader can derive the answer from those units by simple reasoning:
- For temporal questions, include the event/date units needed to compare order or compute a duration.
- For "which happened first/earlier" questions, include units for both compared events when available.
- For update/current-truth questions, include the current-truth unit and any invalidating or superseded unit needed to avoid a stale answer.
- Individual units do not need to literally contain the final answer if their combination supports it.
Return an empty list only when the provided units cannot support the answer even with simple temporal, arithmetic, or update reasoning. Do not create new unit ids.

Return exactly JSON:
{{
  "required_unit_ids": ["..."],
  "rationale": "..."
}}

Question: {question}
Gold answer: {answer}
Evidence units:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def derived_required_units_prompt(
    question: str,
    answer: str,
    sessions: Sequence[GeneratedSession],
    existing_units: Sequence[Mapping[str, Any]],
) -> str:
    session_payload = [
        {
            "session_id": session.session_id,
            "date": session.date,
            "source_kind": session.source_kind,
            "text": truncate_words(session.text, 900),
        }
        for session in sessions
    ]
    unit_payload = [
        {
            "unit_id": row.get("unit_id", ""),
            "canonical_text": row.get("canonical_text", ""),
            "source_quote": row.get("source_quote", ""),
            "session_id": row.get("session_id", ""),
        }
        for row in existing_units
    ]
    payload = {
        "question": question,
        "gold_answer": answer,
        "sessions": session_payload,
        "existing_units": unit_payload,
    }
    return f"""You are adding missing hidden evidence labels for an OracleMem benchmark package.

The memory candidates have already been generated from sessions only. Do not propose or edit memory candidates.
Your task is only to create benchmark evidence units when the existing units are too coarse or omitted the answer-critical fact.

Create the minimal source-backed evidence units needed to answer the question. Use the gold answer only for annotation.
Each unit must be supported by a quote from one of the listed sessions. For temporal questions, create event/date units that allow a reader to compare order or compute the duration; the unit does not have to state the final derived answer.
Return an empty list only if the sessions themselves do not support the answer.

Return exactly JSON:
{{
  "required_evidence_units": [
    {{
      "session_id": "...",
      "canonical_text": "...",
      "source_quote": "...",
      "kind": "temporal_fact|current_fact|update|preference|other",
      "importance": 1.0
    }}
  ],
  "rationale": "..."
}}

PACKAGE:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def clean_float(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return min(1.0, max(0.0, numeric))


def candidate_cost(representation_type: str, text: str) -> int:
    words = max(1, word_count(text))
    if representation_type == "raw_span":
        return max(12, words)
    if representation_type in {"atomic_fact", "tombstone"}:
        return max(4, min(20, words))
    if representation_type == "graph_edge":
        return max(8, min(35, words))
    if representation_type in {"summary", "compound_update"}:
        return max(10, min(45, words))
    return max(6, min(45, words))


def build_instance(
    example: Mapping[str, Any],
    generated_sessions: Sequence[GeneratedSession],
    query_annotation: Mapping[str, Any],
) -> tuple[OracleMemInstance, dict[str, Any]]:
    question_id = str(example["question_id"])
    candidates: list[CandidateMemory] = []
    unit_rows: list[dict[str, Any]] = []
    unit_weights: dict[str, float] = {}
    current_units: list[str] = []
    invalidation_units: list[str] = []
    stale_units: list[str] = []

    for session_index, generated in enumerate(generated_sessions):
        parsed = generated.response
        local_unit_map: dict[str, str] = {}
        for unit_index, unit in enumerate(parsed.get("evidence_units", []) or []):
            local_id = str(unit.get("unit_id", f"u{unit_index + 1}")).strip()
            global_id = f"{safe_token(question_id)}::{safe_token(generated.session_id)}::{safe_token(local_id)}"
            kind = str(unit.get("kind", "other")).strip() or "other"
            canonical = str(unit.get("canonical_text", "")).strip()
            quote = str(unit.get("source_quote", "")).strip()
            if not canonical:
                continue
            local_unit_map[local_id] = global_id
            importance = clean_float(unit.get("importance"), default=0.5)
            unit_rows.append(
                {
                    "unit_id": global_id,
                    "local_unit_id": local_id,
                    "session_id": generated.session_id,
                    "kind": kind,
                    "canonical_text": canonical,
                    "source_quote": quote,
                    "importance": importance,
                    "source_kind": generated.source_kind,
                    "timestamp": session_index,
                }
            )
            unit_weights.setdefault(global_id, 0.0)
            if kind in {"update", "current_fact", "temporal_fact", "preference"}:
                current_units.append(global_id)
            if kind == "update":
                invalidation_units.append(global_id)

        if local_unit_map:
            raw_coverage = {unit_id: 1.0 for unit_id in local_unit_map.values()}
            raw_text = truncate_words(generated.text, 220)
            candidates.append(
                CandidateMemory(
                    candidate_id=f"{safe_token(question_id)}::{safe_token(generated.session_id)}::raw",
                    experience_id=f"{safe_token(question_id)}::{safe_token(generated.session_id)}",
                    representation_type="raw_span",
                    serialized=raw_text,
                    cost=candidate_cost("raw_span", raw_text),
                    coverage=raw_coverage,
                    time_index=session_index,
                    generator="longmemeval_raw",
                    confidence=1.0,
                )
            )

        for candidate_index, raw_candidate in enumerate(parsed.get("candidates", []) or []):
            text = str(raw_candidate.get("text", "")).strip()
            if not text:
                continue
            representation_type = str(raw_candidate.get("representation_type", "summary")).strip() or "summary"
            if representation_type not in {
                "atomic_fact",
                "graph_edge",
                "summary",
                "tombstone",
                "compound_update",
            }:
                representation_type = "summary"
            coverage: dict[str, float] = {}
            for local_id in raw_candidate.get("covers_unit_ids", []) or []:
                global_id = local_unit_map.get(str(local_id).strip())
                if global_id:
                    coverage[global_id] = 1.0
            if not coverage:
                continue
            generator = str(raw_candidate.get("generator", "gemini_writer")).strip() or "gemini_writer"
            candidates.append(
                CandidateMemory(
                    candidate_id=(
                        f"{safe_token(question_id)}::{safe_token(generated.session_id)}::"
                        f"{safe_token(generator)}_{candidate_index}"
                    ),
                    experience_id=f"{safe_token(question_id)}::{safe_token(generated.session_id)}",
                    representation_type=representation_type,
                    serialized=text,
                    cost=candidate_cost(representation_type, text),
                    coverage=coverage,
                    time_index=session_index,
                    generator=generator,
                    confidence=clean_float(raw_candidate.get("confidence"), default=0.75),
                )
            )

    available_units = {row["unit_id"] for row in unit_rows}
    required_unit_ids = [
        str(unit_id)
        for unit_id in query_annotation.get("required_unit_ids", [])
        if str(unit_id) in available_units
    ]
    for unit_id in required_unit_ids:
        unit_weights[unit_id] = 1.0

    instance = OracleMemInstance(
        instance_id=f"longmemeval_gemini_{safe_token(question_id)}",
        seed=None,
        candidates=tuple(candidates),
        unit_weights=unit_weights,
        current_units=tuple(current_units),
        invalidation_units=tuple(invalidation_units),
        stale_units=tuple(stale_units),
    )
    metadata = {
        "question_id": question_id,
        "question_type": example.get("question_type"),
        "question": example.get("question"),
        "answer": example.get("answer"),
        "answer_session_ids": list(example.get("answer_session_ids", []) or []),
        "required_unit_ids": required_unit_ids,
        "query_annotation": dict(query_annotation),
        "unit_rows": unit_rows,
        "selected_sessions": [asdict(generated) for generated in generated_sessions],
    }
    return instance, metadata


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_label(value: float) -> str:
    if value >= 1.0:
        return "full"
    if value >= 0.75:
        return "partial_strong"
    if value >= 0.5:
        return "partial_weak"
    return "hint_only"


def export_natural_package(
    *,
    out_dir: Path,
    instances: Sequence[OracleMemInstance],
    metadata_by_instance: Mapping[str, Mapping[str, Any]],
    model: str,
    cache_path: Path,
    prompt_hashes: Mapping[str, Sequence[str]],
    total_usage: Mapping[str, float],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    experience_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []

    for instance in instances:
        metadata = dict(metadata_by_instance[instance.instance_id])
        session_meta = {
            row["session_id"]: row for row in metadata.get("selected_sessions", [])
        }
        for session_id, session in sorted(session_meta.items()):
            experience_id = f"{safe_token(metadata['question_id'])}::{safe_token(session_id)}"
            experience_rows.append(
                {
                    "experience_id": experience_id,
                    "session_id": session_id,
                    "timestamp": session.get("date", ""),
                    "text": session.get("text", ""),
                    "split": "longmemeval_s_support_slice",
                    "source_kind": session.get("source_kind", ""),
                    "source_span_ids": [f"{experience_id}:full_session"],
                }
            )
        for unit in metadata.get("unit_rows", []):
            evidence_rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "kind": unit["kind"],
                    "canonical_text": unit["canonical_text"],
                    "source_spans": [
                        {
                            "span_id": f"{safe_token(metadata['question_id'])}::{safe_token(unit['session_id'])}:full_session",
                            "session_id": unit["session_id"],
                            "text": unit.get("source_quote") or unit["canonical_text"],
                        }
                    ],
                    "timestamp": unit.get("timestamp", 0),
                    "state": "current",
                    "proposition_id": unit["unit_id"],
                    "annotator_ids": [model],
                    "adjudication_status": "model_annotated",
                    "unit_weight": float(instance.unit_weights.get(unit["unit_id"], 0.0)),
                    "source_kind": unit.get("source_kind", ""),
                }
            )
        query_rows.append(
            {
                "query_id": metadata["question_id"],
                "question": metadata["question"],
                "answer": metadata["answer"],
                "category": metadata["question_type"],
                "required_unit_ids": metadata.get("required_unit_ids", []),
                "answer_session_ids": metadata.get("answer_session_ids", []),
                "split": "longmemeval_s_support_slice",
                "annotation_rationale": metadata.get("query_annotation", {}).get("rationale", ""),
            }
        )
        for candidate in instance.candidates:
            candidate_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "experience_id": candidate.experience_id,
                    "candidate_group": candidate.experience_id,
                    "representation_type": candidate.representation_type,
                    "text": candidate.serialized,
                    "serialized": candidate.serialized,
                    "cost_tokens": candidate.cost,
                    "cost": candidate.cost,
                    "generator_id": candidate.generator,
                    "confidence": candidate.confidence,
                    "time_index": candidate.time_index,
                }
            )
            for unit_id, value in sorted(candidate.coverage.items()):
                coverage_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "experience_id": candidate.experience_id,
                        "candidate_group": candidate.experience_id,
                        "unit_id": unit_id,
                        "coverage": float(value),
                        "coverage_label": coverage_label(float(value)),
                        "rationale": "Gemini-generated candidate declares support for this extracted source-backed evidence unit; raw spans cover all units extracted from their source session.",
                        "source_span_ids": [f"{candidate.experience_id}:full_session"],
                        "annotator_ids": [model],
                        "adjudication_status": "model_annotated",
                    }
                )

    for index, row in enumerate(coverage_rows):
        annotation_rows.append(
            {
                "record_id": f"gemini_natural_coverage:{index:06d}",
                "record_type": "coverage_cell",
                "decision": "accepted_model_annotation",
                "primary_annotator": model,
                "verifier": model,
                "adjudicator": "not_human_adjudicated",
                "candidate_id": row["candidate_id"],
                "unit_id": row["unit_id"],
                "notes": "Single-model annotation; not a human-adjudicated final benchmark label.",
            }
        )

    paths = {
        "experiences": out_dir / "experiences.jsonl",
        "evidence_units": out_dir / "evidence_units.jsonl",
        "queries": out_dir / "queries.jsonl",
        "candidate_memories": out_dir / "candidate_memories.jsonl",
        "coverage_matrix": out_dir / "coverage_matrix.jsonl",
        "annotation_decisions": out_dir / "annotation_decisions.jsonl",
    }
    write_jsonl(paths["experiences"], experience_rows)
    write_jsonl(paths["evidence_units"], evidence_rows)
    write_jsonl(paths["queries"], query_rows)
    write_jsonl(paths["candidate_memories"], candidate_rows)
    write_jsonl(paths["coverage_matrix"], coverage_rows)
    write_jsonl(paths["annotation_decisions"], annotation_rows)

    file_hashes = {path.name: file_sha256(path) for path in paths.values()}
    manifest = {
        "schema_version": 1,
        "synthetic_instance": False,
        "dataset": "LongMemEval-S",
        "split": "support-slice pilot",
        "generator_model": model,
        "api_provider": "OpenRouter",
        "api_cache": str(cache_path),
        "prompt_hashes": {key: list(values) for key, values in prompt_hashes.items()},
        "allowed_inputs": [
            "conversation session text for candidate generation",
            "question and gold answer for separate required-unit annotation",
        ],
        "forbidden_inputs_for_candidate_generation": [
            "held-out question text",
            "gold answer",
            "required_unit_ids",
            "solver outputs",
        ],
        "limitations": [
            "support-slice package includes selected answer-support sessions and optional sampled distractors; it is not a full-haystack write-time benchmark",
            "coverage is single-model annotated and not human adjudicated",
            "published-system rows are local policy mappings over Gemini-generated candidate types unless an external system adapter is explicitly reported",
        ],
        "counts": {
            "instances": len(instances),
            "experiences": len(experience_rows),
            "evidence_units": len(evidence_rows),
            "queries": len(query_rows),
            "candidate_memories": len(candidate_rows),
            "positive_coverage_rows": len(coverage_rows),
        },
        "usage": dict(total_usage),
        "file_hashes": file_hashes,
    }
    manifest_path = out_dir / "candidate_generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Gemini Natural OracleMem Coverage Package",
                "",
                "This is a LongMemEval-S support-slice pilot, not a finalized human-adjudicated benchmark.",
                "Candidate generation used only conversation sessions. Query/gold answer was used only to annotate required evidence units.",
                "",
                f"Instances: {len(instances)}",
                f"Evidence units: {len(evidence_rows)}",
                f"Candidate memories: {len(candidate_rows)}",
                f"Positive coverage rows: {len(coverage_rows)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "package_dir": str(out_dir),
        "candidate_generation_manifest": str(manifest_path),
        "README": str(readme_path),
        **{key: str(value) for key, value in paths.items()},
    }


def choose_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    focus_only: bool,
    limit: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    filtered = [
        example
        for example in examples
        if (not focus_only or example.get("question_type") in FOCUS_TYPES)
    ]
    rng = random.Random(seed)
    filtered = sorted(filtered, key=lambda row: str(row.get("question_id", "")))
    rng.shuffle(filtered)
    return filtered[:limit]


def choose_session_indices(example: Mapping[str, Any], *, distractors: int, rng: random.Random) -> list[int]:
    session_ids = list(example.get("haystack_session_ids", []) or [])
    answer_ids = set(example.get("answer_session_ids", []) or [])
    answer_indices = [index for index, sid in enumerate(session_ids) if sid in answer_ids]
    distractor_indices = [index for index, sid in enumerate(session_ids) if sid not in answer_ids]
    rng.shuffle(distractor_indices)
    selected = sorted(set(answer_indices + distractor_indices[:distractors]))
    if not selected and session_ids:
        selected = [len(session_ids) - 1]
    return selected


def usage_totals(api_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals = defaultdict(float)
    for row in api_rows:
        usage = row.get("usage", {}) or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            try:
                totals[key] += float(usage.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        totals["api_calls"] += 0.0 if row.get("cache_hit") else 1.0
        totals["cache_hits"] += 1.0 if row.get("cache_hit") else 0.0
    return dict(totals)


def render_report(
    *,
    summary: Mapping[str, Any],
    resolved_summary: Sequence[Mapping[str, Any]],
    resolved_count: int,
    unresolved_count: int,
    package_paths: Mapping[str, Any],
    audit_summary: Mapping[str, Any] | None,
    usage: Mapping[str, float],
    source_repos: Mapping[str, str],
) -> str:
    lines = [
        "# Gemini Natural OracleMem Pilot",
        "",
        "This run uses Gemini through OpenRouter to build a LongMemEval-S support-slice coverage package.",
        "It is stronger than synthetic-only evidence, but it is not yet a full non-synthetic benchmark because labels are single-model annotated and the haystack is sliced to selected support/distractor sessions.",
        "",
        "## Source Repos Inspected",
        "",
    ]
    for name, path in sorted(source_repos.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## API Usage",
            "",
            f"- New API calls: {int(usage.get('api_calls', 0.0))}",
            f"- Cache hits: {int(usage.get('cache_hits', 0.0))}",
            f"- Total tokens: {usage.get('total_tokens', 0.0):.0f}",
            f"- Estimated cost from OpenRouter usage: ${usage.get('cost', 0.0):.4f}",
            f"- Coverage-resolved instances: {resolved_count}",
            f"- Unresolved instances with zero required units: {unresolved_count}",
            "",
            "## Coverage Package",
            "",
        ]
    )
    if int(usage.get("api_calls", 0.0)) == 0 and int(usage.get("cache_hits", 0.0)) > 0 and usage.get("total_tokens", 0.0) > 0:
        lines[-2:-2] = [
            "Note: this report was regenerated from cache. The cached rerun made zero additional API calls while preserving historical token/cost metadata from the original uncached calls.",
            "",
        ]
    for key, value in sorted(package_paths.items()):
        lines.append(f"- `{key}`: `{value}`")
    if audit_summary:
        ready = audit_summary.get("coverage_ready_artifacts", [])
        lines.extend(
            [
                "",
                "## Structural Audit",
                "",
                f"- Coverage-ready artifacts according to structural audit: {ready}",
            ]
        )
    lines.extend(["", "## Aggregate Results", ""])
    for row in summary.get("by_budget_method", []):
        lines.append(
            "- budget {budget}, `{method}`: ratio_to_opt={ratio:.3f}, objective={obj:.3f}, cost={cost:.1f}, feasible={feasible}".format(
                budget=row.get("budget"),
                method=row.get("method"),
                ratio=row.get("mean_ratio_to_opt", 0.0),
                obj=row.get("mean_objective", 0.0),
                cost=row.get("mean_selected_cost", 0.0),
                feasible=row.get("all_budget_feasible") and row.get("all_group_feasible"),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage-Resolved Subset",
            "",
            "These rows exclude examples whose required evidence units could not be resolved from the generated coverage package. This is the safer number for paper discussion.",
            "",
        ]
    )
    for row in resolved_summary:
        lines.append(
            "- budget {budget}, `{method}`: n={n}, ratio_to_opt={ratio:.3f}, objective={obj:.3f}, cost={cost:.1f}".format(
                budget=row["budget"],
                method=row["method"],
                n=row["n"],
                ratio=row["mean_ratio_to_opt"],
                obj=row["mean_objective"],
                cost=row["mean_selected_cost"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Candidate generation is query-independent at the session level.",
            "- Required-unit annotation uses the question and gold answer; this is benchmark labeling, not a writer input.",
            "- The MemGPT/Mem0/A-Mem/A-MAC rows use local policy mappings over Gemini-generated candidate types. They are not full published-system executions unless a future adapter records that explicitly.",
            "- This pilot is suitable as a NeurIPS rebuttal/progress artifact, not as the final main empirical table without scaling and adjudication.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_resolved_subset(
    results: Sequence[SelectionResult],
    metadata_by_instance: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[SelectionResult]] = defaultdict(list)
    for row in results:
        metadata = metadata_by_instance.get(row.instance_id, {})
        if not metadata.get("required_unit_ids"):
            continue
        grouped[(int(row.budget), str(row.method))].append(row)
    summary: list[dict[str, Any]] = []
    for (budget, method), rows in sorted(grouped.items()):
        ratios = [float(row.ratio_to_opt) for row in rows if row.ratio_to_opt is not None]
        summary.append(
            {
                "budget": budget,
                "method": method,
                "n": len(rows),
                "mean_ratio_to_opt": statistics.mean(ratios) if ratios else 0.0,
                "mean_objective": statistics.mean(float(row.objective_value) for row in rows),
                "mean_selected_cost": statistics.mean(float(row.selected_cost) for row in rows),
                "all_budget_feasible": all(row.budget_feasible for row in rows),
                "all_group_feasible": all(row.group_feasible for row in rows),
            }
        )
    return summary


def resolution_rows(metadata_by_instance: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return resolved/unresolved example rows for downstream natural-package runs."""

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for instance_id, metadata in sorted(metadata_by_instance.items()):
        row = {
            "instance_id": instance_id,
            "question_id": metadata.get("question_id"),
            "question_type": metadata.get("question_type"),
            "question": metadata.get("question"),
            "answer": metadata.get("answer"),
            "answer_session_ids": metadata.get("answer_session_ids", []),
            "required_unit_ids": metadata.get("required_unit_ids", []),
            "selected_session_ids": [
                session.get("session_id")
                for session in metadata.get("selected_sessions", [])
            ],
            "n_units": len(metadata.get("unit_rows", [])),
            "n_required_units": len(metadata.get("required_unit_ids", [])),
        }
        if row["required_unit_ids"]:
            resolved.append(row)
        else:
            row["unresolved_reason"] = "no_required_units_resolved_from_generated_evidence"
            unresolved.append(row)
    return resolved, unresolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=Path("llm_memory_validation/cache/longmemeval_s_cleaned.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("llm_memory_validation/gemini_natural_oraclemem_pilot"))
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--api-cache", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distractors-per-example", type=int, default=2)
    parser.add_argument("--max-session-words", type=int, default=850)
    parser.add_argument("--budgets", default="30,60")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--focus-only", action="store_true", default=True)
    parser.add_argument("--no-focus-only", action="store_false", dest="focus_only")
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--request-sleep", type=float, default=0.02)
    args = parser.parse_args()

    env = load_env_file(args.api_env)
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not found in {args.api_env}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api_cache = args.api_cache or (args.out_dir / "openrouter_cache_gemini_natural_oraclemem.json")
    client = OpenRouterJsonClient(
        api_key=api_key,
        model=args.model,
        cache_path=api_cache,
        max_tokens=args.max_tokens,
        request_sleep=args.request_sleep,
    )

    examples = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    selected_examples = choose_examples(
        examples,
        focus_only=args.focus_only,
        limit=args.limit,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    instances: list[OracleMemInstance] = []
    metadata_by_instance: dict[str, dict[str, Any]] = {}
    api_rows: list[dict[str, Any]] = []
    prompt_hashes: dict[str, list[str]] = defaultdict(list)

    for example_index, example in enumerate(selected_examples):
        session_ids = list(example.get("haystack_session_ids", []) or [])
        session_dates = list(example.get("haystack_dates", []) or [])
        sessions = list(example.get("haystack_sessions", []) or [])
        answer_ids = set(example.get("answer_session_ids", []) or [])
        generated_sessions: list[GeneratedSession] = []
        for session_index in choose_session_indices(
            example,
            distractors=args.distractors_per_example,
            rng=rng,
        ):
            if session_index >= len(sessions):
                continue
            sid = str(session_ids[session_index]) if session_index < len(session_ids) else f"session_{session_index}"
            date = str(session_dates[session_index]) if session_index < len(session_dates) else ""
            text = session_text(sessions[session_index], max_words=args.max_session_words)
            source_kind = "answer_support" if sid in answer_ids else "distractor"
            response = client(
                session_prompt(sid, date, text),
                purpose="session_candidate_generation",
            )
            api_rows.append(response)
            prompt_hashes["session_candidate_generation"].append(str(response["prompt_hash"]))
            generated_sessions.append(
                GeneratedSession(
                    session_id=sid,
                    date=date,
                    source_kind=source_kind,
                    text=text,
                    response=dict(response.get("parsed", {})),
                    prompt_hash=str(response["prompt_hash"]),
                    cache_hit=bool(response.get("cache_hit")),
                    usage=dict(response.get("usage", {}) or {}),
                )
            )

        all_unit_rows: list[dict[str, Any]] = []
        for generated in generated_sessions:
            for unit in generated.response.get("evidence_units", []) or []:
                local_id = str(unit.get("unit_id", "")).strip()
                if not local_id:
                    continue
                global_id = f"{safe_token(example['question_id'])}::{safe_token(generated.session_id)}::{safe_token(local_id)}"
                all_unit_rows.append(
                    {
                        "unit_id": global_id,
                        "canonical_text": str(unit.get("canonical_text", "")).strip(),
                        "source_quote": str(unit.get("source_quote", "")).strip(),
                        "session_id": generated.session_id,
                    }
                )
        query_response = client(
            query_prompt(
                str(example.get("question", "")),
                str(example.get("answer", "")),
                all_unit_rows,
            ),
            purpose="query_required_unit_annotation",
        )
        api_rows.append(query_response)
        prompt_hashes["query_required_unit_annotation"].append(str(query_response["prompt_hash"]))
        query_annotation = dict(query_response.get("parsed", {}))
        available_unit_ids = {str(row["unit_id"]) for row in all_unit_rows}
        resolved_required_ids = [
            str(unit_id)
            for unit_id in query_annotation.get("required_unit_ids", []) or []
            if str(unit_id) in available_unit_ids
        ]
        if not resolved_required_ids and generated_sessions:
            derived_response = client(
                derived_required_units_prompt(
                    str(example.get("question", "")),
                    str(example.get("answer", "")),
                    generated_sessions,
                    all_unit_rows,
                ),
                purpose="query_derived_required_unit_annotation",
            )
            api_rows.append(derived_response)
            prompt_hashes["query_derived_required_unit_annotation"].append(str(derived_response["prompt_hash"]))
            derived = derived_response.get("parsed", {}) if isinstance(derived_response, Mapping) else {}
            session_by_id = {session.session_id: session for session in generated_sessions}
            derived_required_ids: list[str] = []
            local_counts: dict[str, int] = defaultdict(int)
            for session in generated_sessions:
                local_counts[session.session_id] = len(session.response.get("evidence_units", []) or [])
            for unit in derived.get("required_evidence_units", []) or []:
                if not isinstance(unit, Mapping):
                    continue
                session_id = str(unit.get("session_id", "")).strip()
                if session_id not in session_by_id:
                    continue
                canonical = str(unit.get("canonical_text", "")).strip()
                quote = str(unit.get("source_quote", "")).strip()
                if not canonical:
                    continue
                local_counts[session_id] += 1
                local_id = f"dq{local_counts[session_id]}"
                session = session_by_id[session_id]
                session.response.setdefault("evidence_units", []).append(
                    {
                        "unit_id": local_id,
                        "canonical_text": canonical,
                        "source_quote": quote,
                        "kind": str(unit.get("kind", "temporal_fact")).strip() or "temporal_fact",
                        "importance": clean_float(unit.get("importance"), default=1.0),
                    }
                )
                global_id = f"{safe_token(example['question_id'])}::{safe_token(session_id)}::{safe_token(local_id)}"
                derived_required_ids.append(global_id)
            if derived_required_ids:
                query_annotation = {
                    "required_unit_ids": derived_required_ids,
                    "rationale": (
                        "Derived evidence-unit fallback: "
                        + str(derived.get("rationale", query_annotation.get("rationale", "")))
                    ),
                    "derived_required_unit_annotation": True,
                    "initial_query_annotation": dict(query_response.get("parsed", {})),
                }
        instance, metadata = build_instance(
            example,
            generated_sessions,
            query_annotation,
        )
        if not instance.candidates:
            continue
        instances.append(instance)
        metadata_by_instance[instance.instance_id] = metadata
        print(
            f"[{example_index + 1}/{len(selected_examples)}] {example.get('question_id')} "
            f"candidates={len(instance.candidates)} required={len(metadata['required_unit_ids'])}"
        )

    budgets = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    results: list[SelectionResult] = []
    for instance in instances:
        results.extend(
            evaluate_instance(
                instance,
                budgets,
                methods=methods,
                retrieval_modes=("fixed", "oracle"),
            )
        )

    paths = write_benchmark_outputs(results, args.out_dir)
    usage = usage_totals(api_rows)
    package_paths = export_natural_package(
        out_dir=args.out_dir / "coverage_package",
        instances=instances,
        metadata_by_instance=metadata_by_instance,
        model=args.model,
        cache_path=api_cache,
        prompt_hashes=prompt_hashes,
        total_usage=usage,
    )
    api_rows_path = args.out_dir / "api_calls.jsonl"
    write_jsonl(api_rows_path, api_rows)
    metadata_path = args.out_dir / "instance_metadata.json"
    metadata_path.write_text(json.dumps(metadata_by_instance, indent=2, sort_keys=True), encoding="utf-8")

    audit_summary = None
    audit_path = args.out_dir / "coverage_audit" / "summary.json"
    if audit_path.exists():
        audit_summary = json.loads(audit_path.read_text(encoding="utf-8"))

    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    resolved_count = sum(1 for metadata in metadata_by_instance.values() if metadata.get("required_unit_ids"))
    unresolved_count = len(metadata_by_instance) - resolved_count
    resolved_summary = aggregate_resolved_subset(results, metadata_by_instance)
    resolved_summary_path = args.out_dir / "coverage_resolved_summary.json"
    resolved_rows, unresolved_rows = resolution_rows(metadata_by_instance)
    resolved_rows_path = args.out_dir / "resolved_examples.jsonl"
    unresolved_rows_path = args.out_dir / "unresolved_examples.jsonl"
    write_jsonl(resolved_rows_path, resolved_rows)
    write_jsonl(unresolved_rows_path, unresolved_rows)
    resolution_report_path = args.out_dir / "coverage_resolution_report.md"
    resolution_rate = (len(resolved_rows) / len(metadata_by_instance)) if metadata_by_instance else 0.0
    resolution_report_path.write_text(
        "\n".join(
            [
                "# Coverage Resolution Report",
                "",
                f"- Attempted/constructed instances: {len(metadata_by_instance)}",
                f"- Coverage-resolved instances: {len(resolved_rows)}",
                f"- Unresolved instances: {len(unresolved_rows)}",
                f"- Coverage-resolved rate: {resolution_rate:.3f}",
                "",
                "An instance is coverage-resolved when the query annotation maps at least one required evidence unit to evidence units generated from the selected support/distractor sessions or the source-backed derived-unit annotation pass.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_summary_path.write_text(
        json.dumps(
            {
                "coverage_resolved_instances": resolved_count,
                "unresolved_instances": unresolved_count,
                "by_budget_method": resolved_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = render_report(
        summary=summary,
        resolved_summary=resolved_summary,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        package_paths=package_paths,
        audit_summary=audit_summary,
        usage=usage,
        source_repos={
            "Mem0": "external_repos/mem0",
            "A-Mem": "external_repos/AgenticMemory",
            "Letta/MemGPT": "external_repos/letta",
        },
    )
    report_path = args.out_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    run_manifest = {
        "schema_version": 1,
        "model": args.model,
        "limit": args.limit,
        "focus_only": args.focus_only,
        "distractors_per_example": args.distractors_per_example,
        "instances": len(instances),
        "budgets": budgets,
        "methods": methods,
        "paths": {
            **paths,
            "package": package_paths,
            "api_calls": str(api_rows_path),
            "metadata": str(metadata_path),
            "coverage_resolved_summary": str(resolved_summary_path),
            "resolved_examples": str(resolved_rows_path),
            "unresolved_examples": str(unresolved_rows_path),
            "coverage_resolution_report": str(resolution_report_path),
            "report": str(report_path),
        },
        "usage": usage,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
