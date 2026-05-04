"""Run a production Letta/MemGPT writer on a natural OracleMem package.

This runner uses a live Letta REST server backed by Postgres/pgvector and an
OpenRouter-served Gemini model.  It is intentionally separate from the
``faithful_memgpt_letta`` proxy runner: the memories scored here are written by
actual Letta agents through the Letta API.

The output format matches the existing external-writer rescoring convention:
written memories are mapped to OracleMem evidence units by a cached Gemini judge,
then scored under an exact finite union denominator consisting of package
candidates plus the Letta-written memories.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from letta_client import Letta

from oraclemem.evaluate import CandidateMemory, OracleMemInstance, objective_value, solve_exact

from llm_memory_validation.gemini_natural_oraclemem import OpenRouterJsonClient, load_env_file, word_count
from llm_memory_validation.run_mem0_natural_baseline import (
    PackageData,
    load_package,
    ordered_experiences,
    package_instance,
    read_jsonl,
    resolved_queries,
    score_mem0_coverage,
    select_oracle_density_pruned,
    select_recency_pruned,
    write_json,
    write_jsonl,
)
from llm_memory_validation.score_mem0_written_stores import (
    attach_salience,
    score_salience,
    select_salience_pruned,
)


SEED_HUMAN_MEMORY = (
    "The human is the user in the conversation transcripts. Store durable current facts, "
    "updates, preferences, deadlines, invalidations, and facts needed for future questions."
)
SEED_PERSONA_MEMORY = (
    "You are a production Letta/MemGPT memory writer. Maintain compact core memory and "
    "use archival memory for durable details. Prefer concise atomic memories over long transcripts."
)

FILTER_PHRASES = {
    "the human is the user",
    "store durable current facts",
    "production letta/memgpt memory writer",
    "maintain compact core memory",
    "prefer concise atomic memories",
    "read these conversation transcripts",
    "update your durable memory",
    "do not answer",
    "do not copy whole transcripts",
    "only perform memory maintenance",
    "provide concise atomic memories",
    "write concise atomic memories",
}


def mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def stdev(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return statistics.stdev(clean)


def truncate_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", str(text))
    if len(words) <= limit:
        return str(text)
    return " ".join(words[:limit]) + " ..."


def compact_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): compact_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def compact_without_embeddings(value: Any) -> Any:
    """Compact Letta objects while dropping large vector payloads from logs."""

    value = compact_json(value)
    if isinstance(value, Mapping):
        return {
            str(key): compact_without_embeddings(item)
            for key, item in value.items()
            if str(key) != "embedding"
        }
    if isinstance(value, list):
        return [compact_without_embeddings(item) for item in value]
    return value


def split_memory_atoms(text: str) -> list[str]:
    """Extract compact memory-like atoms from Letta core-memory block text."""

    normalized = str(text).replace("\r", "\n")
    chunks: list[str] = []
    for line in normalized.splitlines():
        line = line.strip(" -*\t")
        if not line:
            continue
        # Letta often appends multiple atomic facts into one core-memory line.
        parts = re.split(r"(?<=[.!?])\s+(?=(?:User|The user|They|Their|He|She|Current|Stale|Superseded)\b)", line)
        chunks.extend(part.strip(" -*\t") for part in parts if part.strip(" -*\t"))

    cleaned: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        lowered = chunk.lower()
        if any(phrase in lowered for phrase in FILTER_PHRASES):
            continue
        if lowered in seen:
            continue
        words = re.findall(r"\S+", chunk)
        if len(words) < 4:
            continue
        if len(words) > 90:
            # Extremely long chunks are usually raw transcript fragments, not
            # Letta's compact written memories.
            chunk = " ".join(words[:90]) + " ..."
        seen.add(lowered)
        cleaned.append(chunk)
    return cleaned


def build_writer_prompt(
    *,
    instance_id: str,
    experiences: Sequence[Mapping[str, Any]],
    max_words_per_experience: int,
) -> str:
    rows = []
    for index, row in enumerate(experiences, start=1):
        rows.append(
            {
                "experience_index": index,
                "experience_id": row.get("experience_id"),
                "timestamp": row.get("timestamp"),
                "text": truncate_words(str(row.get("text", "")), max_words_per_experience),
            }
        )
    return (
        "Read these conversation transcripts and update your durable memory.\n"
        "Use Letta core memory only for the shortest user/profile summary. "
        "For each durable fact, preference, commitment, date, quantity, update, invalidation, or tombstone, "
        "call archival_memory_insert with one concise atomic memory. "
        "Do not answer any downstream question. Only perform memory maintenance.\n"
        "Do not copy whole transcripts. Do not insert duplicate archival memories. "
        "After the required memory writes, stop.\n\n"
        f"INSTANCE_ID: {instance_id}\n"
        f"TRANSCRIPTS:\n{json.dumps(rows, indent=2, sort_keys=True)}"
    )


def archival_tool_ids(client: Letta) -> list[str]:
    """Return Letta built-in archival tool ids when available."""

    wanted = {"archival_memory_insert", "archival_memory_search"}
    ids: list[str] = []
    try:
        tools = list(client.tools.list().items)
    except Exception:
        return ids
    for tool in tools:
        record = compact_json(tool)
        name = str((record.get("json_schema") or {}).get("name") or record.get("name") or "")
        if name in wanted and record.get("id"):
            ids.append(str(record["id"]))
    return ids


def create_agent(
    *,
    client: Letta,
    name: str,
    model: str,
    embedding: str,
    context_window_limit: int,
    tool_ids: Sequence[str] = (),
) -> Any:
    kwargs: dict[str, Any] = {}
    if tool_ids:
        kwargs["tool_ids"] = list(tool_ids)
    return client.agents.create(
        name=name,
        model=model,
        embedding=embedding,
        memory_blocks=[
            {"label": "human", "value": SEED_HUMAN_MEMORY},
            {"label": "persona", "value": SEED_PERSONA_MEMORY},
        ],
        include_base_tools=True,
        context_window_limit=context_window_limit,
        **kwargs,
    )


def extract_core_memories(client: Letta, agent_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = list(client.agents.blocks.list(agent_id=agent_id).items)
    raw_blocks: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    memory_index = 0
    for block in blocks:
        block_record = compact_json(block)
        raw_blocks.append(block_record)
        label = str(getattr(block, "label", "") or block_record.get("label", ""))
        value = str(getattr(block, "value", "") or block_record.get("value", ""))
        for atom in split_memory_atoms(value):
            memories.append(
                {
                    "memory_index": memory_index,
                    "memory_id": f"letta_core::{agent_id}::{memory_index}",
                    "text": atom,
                    "created_at": str(block_record.get("created_at", "")),
                    "updated_at": str(block_record.get("updated_at", "")),
                    "raw": {"agent_id": agent_id, "block_label": label, "block_id": block_record.get("id")},
                }
            )
            memory_index += 1
    memories.sort(key=lambda row: (row["memory_index"], row["memory_id"]))
    return memories, raw_blocks


def extract_archival_memories(
    client: Letta,
    agent_id: str,
    *,
    passage_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract actual Letta archival passages attached to an agent."""

    try:
        passages = list(client.agents.passages.list(agent_id, limit=passage_limit))
    except Exception:
        return [], []
    raw_passages = [compact_without_embeddings(passage) for passage in passages]
    memories: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, passage in enumerate(passages):
        record = compact_without_embeddings(passage)
        text = str(
            getattr(passage, "text", "")
            or getattr(passage, "content", "")
            or (record.get("text") if isinstance(record, Mapping) else "")
            or (record.get("content") if isinstance(record, Mapping) else "")
            or ""
        ).strip()
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        passage_id = str(getattr(passage, "id", "") or (record.get("id") if isinstance(record, Mapping) else "") or index)
        memories.append(
            {
                "memory_index": index,
                "memory_id": f"letta_archival::{agent_id}::{passage_id}",
                "text": text,
                "created_at": str(
                    getattr(passage, "created_at", "")
                    or (record.get("created_at") if isinstance(record, Mapping) else "")
                ),
                "updated_at": str(
                    getattr(passage, "updated_at", "")
                    or (record.get("updated_at") if isinstance(record, Mapping) else "")
                ),
                "raw": {"agent_id": agent_id, "passage": record},
            }
        )
    memories.sort(key=lambda row: (row["memory_index"], row["memory_id"]))
    return memories, raw_passages


def rename_candidates(
    candidates: Sequence[CandidateMemory],
    *,
    generator: str,
    representation_type: str,
    token: str,
) -> list[CandidateMemory]:
    renamed: list[CandidateMemory] = []
    for index, candidate in enumerate(candidates):
        instance_id = candidate.candidate_id.split("::", 1)[0]
        renamed.append(
            CandidateMemory(
                candidate_id=f"{instance_id}::{token}::{index:04d}",
                experience_id=f"{instance_id}::{token}::{index:04d}",
                representation_type=representation_type,
                serialized=candidate.serialized,
                cost=candidate.cost,
                coverage=candidate.coverage,
                time_index=candidate.time_index,
                generator=generator,
                confidence=candidate.confidence,
                estimated_value=candidate.estimated_value,
                estimator_model=candidate.estimator_model,
            )
        )
    return renamed


def union_instance(package: OracleMemInstance, external_candidates: Sequence[CandidateMemory]) -> OracleMemInstance:
    return OracleMemInstance(
        instance_id=f"{package.instance_id}::package_plus_letta",
        candidates=tuple(package.candidates) + tuple(external_candidates),
        unit_weights=package.unit_weights,
        seed=package.seed,
        current_units=package.current_units,
        invalidation_units=package.invalidation_units,
        stale_units=package.stale_units,
    )


def result_row(
    *,
    instance_id: str,
    budget: int,
    method: str,
    selected: Sequence[CandidateMemory],
    package: OracleMemInstance,
    package_denominator: float,
    union_denominator: float,
    runtime_sec: float,
    written_count: int,
    written_cost: int,
) -> dict[str, Any]:
    value = objective_value(selected, package.unit_weights)
    return {
        "instance_id": instance_id,
        "budget": budget,
        "method": method,
        "objective_value": value,
        "package_candidate_exact_opt": package_denominator,
        "package_plus_letta_exact_opt": union_denominator,
        "ratio_to_package_candidate_opt": value / package_denominator if package_denominator > 0 else None,
        "ratio_to_union_opt": value / union_denominator if union_denominator > 0 else None,
        "selected_cost": sum(candidate.cost for candidate in selected),
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "selected_memory_texts": [candidate.serialized for candidate in selected],
        "written_memory_count": written_count,
        "written_store_cost": written_cost,
        "denominator_label": "package_plus_letta_exact_opt",
        "runtime_sec": runtime_sec,
    }


def run_letta_instance(
    *,
    client: Letta,
    data: PackageData,
    query: Mapping[str, Any],
    model: str,
    embedding: str,
    context_window_limit: int,
    max_words_per_experience: int,
    max_steps: int,
    archival_tool_ids_: Sequence[str],
    passage_limit: int,
    keep_agents: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    instance_id = str(query["query_id"])
    started = time.perf_counter()
    experiences = ordered_experiences(data, instance_id)
    agent_name = f"oraclemem_letta_{instance_id}_{int(time.time() * 1000)}"
    agent = create_agent(
        client=client,
        name=agent_name,
        model=model,
        embedding=embedding,
        context_window_limit=context_window_limit,
        tool_ids=archival_tool_ids_,
    )
    agent_id = str(agent.id)
    raw_record: dict[str, Any] = {
        "instance_id": instance_id,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "model": model,
        "embedding": embedding,
        "experience_ids": [row.get("experience_id") for row in experiences],
        "message_response": None,
        "raw_blocks": [],
        "raw_passages": [],
        "archival_tool_ids": list(archival_tool_ids_),
        "delete_error": None,
        "runtime_sec": None,
    }
    try:
        prompt = build_writer_prompt(
            instance_id=instance_id,
            experiences=experiences,
            max_words_per_experience=max_words_per_experience,
        )
        response = client.agents.messages.create(agent_id=agent_id, input=prompt, max_steps=max_steps)
        raw_record["message_response"] = compact_json(response)
        core_memories, raw_blocks = extract_core_memories(client, agent_id)
        archival_memories, raw_passages = extract_archival_memories(
            client,
            agent_id,
            passage_limit=passage_limit,
        )
        raw_record["raw_blocks"] = raw_blocks
        raw_record["raw_passages"] = raw_passages
        combined_memories = list(core_memories) + list(archival_memories)
        row = {
            "instance_id": instance_id,
            "question": query.get("question"),
            "answer": query.get("answer"),
            "agent_id": agent_id,
            "model": model,
            "embedding": embedding,
            "core_memories": core_memories,
            "archival_memories": archival_memories,
            "memories": combined_memories,
            "core_memory_count": len(core_memories),
            "archival_memory_count": len(archival_memories),
            "memory_count": len(combined_memories),
            "store_dir": "letta_postgres_core_and_archival_memory",
            "runtime_sec": time.perf_counter() - started,
        }
        raw_record["runtime_sec"] = row["runtime_sec"]
        return row, raw_record
    finally:
        if not keep_agents:
            try:
                client.agents.delete(agent_id)
            except Exception as exc:
                raw_record["delete_error"] = {"type": type(exc).__name__, "message": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-env", type=Path, default=Path("api.env"))
    parser.add_argument("--letta-url", default="http://127.0.0.1:8283")
    parser.add_argument("--letta-model", default="openrouter/google/gemini-2.5-flash-lite")
    parser.add_argument("--letta-embedding", default="openrouter/text-embedding-3-small")
    parser.add_argument("--coverage-model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--salience-model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--budgets", default="30,60,100")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-words-per-experience", type=int, default=900)
    parser.add_argument("--context-window-limit", type=int, default=16000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--message-retries", type=int, default=2)
    parser.add_argument("--passage-limit", type=int, default=200)
    parser.add_argument("--request-sleep", type=float, default=0.02)
    parser.add_argument("--include-salience-pruned", action="store_true")
    parser.add_argument("--include-oracle-pruned-upper", action="store_true")
    parser.add_argument("--disable-archival-tools", action="store_true")
    parser.add_argument("--keep-agents", action="store_true")
    args = parser.parse_args()

    env_values = load_env_file(args.api_env)
    for key, value in env_values.items():
        # The Letta server already has its own key; these are for coverage and
        # salience judging through OpenRouter.
        import os

        os.environ.setdefault(key, value)
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment or api.env")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_package(args.package_dir)
    queries = resolved_queries(data, args.limit)
    budgets = [int(float(item.strip())) for item in args.budgets.split(",") if item.strip()]
    letta_client = Letta(base_url=args.letta_url, timeout=900)
    archival_ids = [] if args.disable_archival_tools else archival_tool_ids(letta_client)
    coverage_client = OpenRouterJsonClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.coverage_model,
        cache_path=args.out_dir / "coverage_scoring_cache.json",
        max_tokens=1800,
        request_sleep=args.request_sleep,
    )
    salience_client = OpenRouterJsonClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.salience_model,
        cache_path=args.out_dir / "salience_scoring_cache.json",
        max_tokens=4000,
        request_sleep=args.request_sleep,
    )

    written_rows: list[dict[str, Any]] = []
    raw_letta_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    salience_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for query in queries:
        instance_id = str(query["query_id"])
        store = None
        raw_letta = None
        last_exc: Exception | None = None
        for attempt in range(max(0, args.message_retries) + 1):
            try:
                store, raw_letta = run_letta_instance(
                    client=letta_client,
                    data=data,
                    query=query,
                    model=args.letta_model,
                    embedding=args.letta_embedding,
                    context_window_limit=args.context_window_limit,
                    max_words_per_experience=args.max_words_per_experience,
                    max_steps=args.max_steps,
                    archival_tool_ids_=archival_ids,
                    passage_limit=args.passage_limit,
                    keep_agents=args.keep_agents,
                )
                if raw_letta is not None:
                    raw_letta["attempt"] = attempt + 1
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max(0, args.message_retries):
                    time.sleep(2.0 * (attempt + 1))
                    continue
        if store is None or raw_letta is None:
            exc = last_exc or RuntimeError("Letta run failed without exception details")
            skipped_rows.append(
                {
                    "instance_id": instance_id,
                    "reason": "letta_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "attempts": max(0, args.message_retries) + 1,
                }
            )
            continue
        written_rows.append(store)
        raw_letta_rows.append(raw_letta)
        package = package_instance(data, query)
        memory_scopes = [
            (
                "core",
                "actual_letta_core",
                "letta_core",
                "letta_core_memory",
                list(store.get("core_memories", []) or []),
            ),
            (
                "archival",
                "actual_letta_archival",
                "letta_archival",
                "letta_archival_passage",
                list(store.get("archival_memories", []) or []),
            ),
            (
                "combined",
                "actual_letta_combined",
                "letta_combined",
                "letta_combined_memory",
                list(store.get("memories", []) or []),
            ),
        ]
        for scope_name, generator, token, representation_type, memories in memory_scopes:
            if not memories:
                skipped_rows.append({"instance_id": instance_id, "reason": f"no_letta_{scope_name}_memories"})
                continue
            try:
                external_candidates, scoring_record = score_mem0_coverage(
                    client=coverage_client,
                    data=data,
                    query=query,
                    memories=memories,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "instance_id": instance_id,
                        "reason": f"{scope_name}_coverage_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            external_candidates = rename_candidates(
                external_candidates,
                generator=generator,
                representation_type=representation_type,
                token=token,
            )
            scoring_record["system"] = generator
            scoring_record["memory_scope"] = scope_name
            scoring_rows.append(scoring_record)

            salience_candidates = external_candidates
            if args.include_salience_pruned:
                try:
                    salience_by_memory = score_salience(
                        client=salience_client,
                        query=query,
                        memories=memories,
                    )
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "instance_id": instance_id,
                            "reason": f"{scope_name}_salience_exception",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    salience_by_memory = {}
                salience_rows.append(
                    {"instance_id": instance_id, "memory_scope": scope_name, "scores": salience_by_memory}
                )
                salience_candidates = rename_candidates(
                    attach_salience(external_candidates, memories, salience_by_memory),
                    generator=f"{generator}_salience",
                    representation_type=representation_type,
                    token=token,
                )

            for budget in budgets:
                package_exact = solve_exact(package, budget, solver="exact_stdlib")
                union_exact = solve_exact(union_instance(package, external_candidates), budget, solver="exact_stdlib")
                package_denominator = package_exact.objective_value
                union_denominator = union_exact.objective_value
                written_cost = sum(candidate.cost for candidate in external_candidates)
                result_rows.append(
                    result_row(
                        instance_id=instance_id,
                        budget=budget,
                        method=f"{generator}_recency_pruned",
                        selected=select_recency_pruned(external_candidates, budget),
                        package=package,
                        package_denominator=package_denominator,
                        union_denominator=union_denominator,
                        runtime_sec=float(store.get("runtime_sec", 0.0) or 0.0),
                        written_count=len(external_candidates),
                        written_cost=written_cost,
                    )
                )
                if args.include_salience_pruned:
                    result_rows.append(
                        result_row(
                            instance_id=instance_id,
                            budget=budget,
                            method=f"{generator}_salience_pruned",
                            selected=select_salience_pruned(salience_candidates, budget),
                            package=package,
                            package_denominator=package_denominator,
                            union_denominator=union_denominator,
                            runtime_sec=float(store.get("runtime_sec", 0.0) or 0.0),
                            written_count=len(external_candidates),
                            written_cost=written_cost,
                        )
                    )
                if args.include_oracle_pruned_upper:
                    result_rows.append(
                        result_row(
                            instance_id=instance_id,
                            budget=budget,
                            method=f"{generator}_oracle_pruned_upper",
                            selected=select_oracle_density_pruned(external_candidates, budget, package.unit_weights),
                            package=package,
                            package_denominator=package_denominator,
                            union_denominator=union_denominator,
                            runtime_sec=float(store.get("runtime_sec", 0.0) or 0.0),
                            written_count=len(external_candidates),
                            written_cost=written_cost,
                        )
                    )

    write_jsonl(args.out_dir / "written_stores.jsonl", written_rows)
    write_jsonl(args.out_dir / "letta_raw_responses.jsonl", raw_letta_rows)
    write_jsonl(args.out_dir / "coverage_scoring_calls.jsonl", scoring_rows)
    write_jsonl(args.out_dir / "salience_scoring_calls.jsonl", salience_rows)
    write_jsonl(args.out_dir / "raw_results.jsonl", result_rows)
    write_jsonl(args.out_dir / "skipped_instances.jsonl", skipped_rows)

    by_method_budget: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        by_method_budget[(str(row["method"]), int(row["budget"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (method, budget), rows in sorted(by_method_budget.items()):
        union_ratios = [row["ratio_to_union_opt"] for row in rows if row.get("ratio_to_union_opt") is not None]
        package_ratios = [
            row["ratio_to_package_candidate_opt"]
            for row in rows
            if row.get("ratio_to_package_candidate_opt") is not None
        ]
        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(rows),
                "ratio_defined_n": len(union_ratios),
                "mean_ratio_to_union_opt": mean(union_ratios),
                "std_ratio_to_union_opt": stdev(union_ratios),
                "mean_ratio_to_package_candidate_opt": mean(package_ratios),
                "std_ratio_to_package_candidate_opt": stdev(package_ratios),
                "mean_objective_value": mean([float(row["objective_value"]) for row in rows]),
                "mean_package_candidate_exact_opt": mean([float(row["package_candidate_exact_opt"]) for row in rows]),
                "mean_package_plus_letta_exact_opt": mean([float(row["package_plus_letta_exact_opt"]) for row in rows]),
                "mean_written_memory_count": mean([float(row["written_memory_count"]) for row in rows]),
                "mean_written_store_cost": mean([float(row["written_store_cost"]) for row in rows]),
                "mean_runtime_sec": mean([float(row["runtime_sec"]) for row in rows]),
            }
        )

    failed_instance_ids = {
        str(row.get("instance_id"))
        for row in skipped_rows
        if "exception" in str(row.get("reason", ""))
    }
    empty_scope_records = [
        row for row in skipped_rows if str(row.get("reason", "")).startswith("no_letta_")
    ]
    summary = {
        "package_dir": str(args.package_dir),
        "letta_url": args.letta_url,
        "letta_model": args.letta_model,
        "letta_embedding": args.letta_embedding,
        "archival_tool_ids": archival_ids,
        "coverage_model": args.coverage_model,
        "salience_model": args.salience_model if args.include_salience_pruned else None,
        "attempted_instances": len(queries),
        "completed_instances": len({row["instance_id"] for row in result_rows}),
        "written_store_instances": len(written_rows),
        "skipped_instances": len(failed_instance_ids),
        "skipped_records": len(skipped_rows),
        "empty_scope_records": len(empty_scope_records),
        "budgets": budgets,
        "denominator_label": "package_plus_letta_exact_opt",
        "summary_rows": summary_rows,
        "notes": [
            "This is a true Letta REST/API run, not the faithful proxy runner.",
            "Letta archival_memory_insert/search tools are attached when available and scored separately from core memory.",
            "Letta core-memory blocks are split into compact written-memory atoms before scoring; archival rows use actual agent passages.",
            "Use openrouter/text-embedding-3-small or another authenticated embedding handle for Letta passage search.",
        ],
    }
    write_json(args.out_dir / "summary.json", summary)

    lines = [
        "# Actual Letta OpenRouter-Gemini Baseline",
        "",
        f"- Package: `{args.package_dir}`",
        f"- Letta server: `{args.letta_url}`",
        f"- Letta model: `{args.letta_model}`",
        f"- Letta embedding: `{args.letta_embedding}`",
        f"- Coverage judge: `{args.coverage_model}`",
        f"- Salience judge: `{args.salience_model if args.include_salience_pruned else 'not used'}`",
        f"- Attempted instances: {len(queries)}",
        f"- Completed scored instances: {summary['completed_instances']}",
        f"- Written-store instances: {summary['written_store_instances']}",
        f"- Skipped records: {len(skipped_rows)}",
        "- Primary denominator: exact finite optimum over package candidates plus Letta-written memories (`package_plus_letta_exact_opt`).",
        "",
        "| Method | Budget | N | Mean ratio to union OPT | Mean ratio to package-candidate OPT | Mean written memories | Mean store cost | Mean runtime sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {budget} | {n} | {union_ratio:.3f} | {package_ratio:.3f} | {count:.2f} | {cost:.1f} | {runtime:.1f} |".format(
                method=row["method"],
                budget=row["budget"],
                n=row["n"],
                union_ratio=row["mean_ratio_to_union_opt"] if row["mean_ratio_to_union_opt"] is not None else float("nan"),
                package_ratio=(
                    row["mean_ratio_to_package_candidate_opt"]
                    if row["mean_ratio_to_package_candidate_opt"] is not None
                    else float("nan")
                ),
                count=row["mean_written_memory_count"] if row["mean_written_memory_count"] is not None else float("nan"),
                cost=row["mean_written_store_cost"] if row["mean_written_store_cost"] is not None else float("nan"),
                runtime=row["mean_runtime_sec"] if row["mean_runtime_sec"] is not None else float("nan"),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This run demonstrates a production Letta/OpenRouter-Gemini memory writer under the OracleMem denominator. "
            "It is not an oracle writer: Letta does not see coverage labels or downstream required evidence units at write time. "
            "The oracle-pruned row, when enabled, is an analysis-only upper bound over Letta-written memories.",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
