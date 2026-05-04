"""Smoke-test Letta passage search with an authenticated embedding endpoint.

This checks the production path that previously failed when Letta routed
embedding calls through an unauthenticated OpenAI client. It requires a running
Letta server and inserts a single known passage, then verifies semantic search
returns that passage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from letta_client import Letta


DEFAULT_TEXT = (
    "A durable memory: Riley moved the launch deadline to Friday and cancelled "
    "the old Monday deadline."
)
DEFAULT_QUERY = "What happened to Riley launch deadline?"


def compact(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "embedding":
                continue
            cleaned[key] = compact(item)
        return cleaned
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--letta-url", default="http://127.0.0.1:8283")
    parser.add_argument("--embedding", default="openrouter/text-embedding-3-small")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--agent-scoped", action="store_true")
    args = parser.parse_args()

    client = Letta(base_url=args.letta_url, timeout=180)
    if args.agent_scoped:
        agent = client.agents.create(
            name=f"oraclemem_agent_passage_smoke_{int(time.time())}",
            model="openrouter/google/gemini-2.5-flash-lite",
            embedding=args.embedding,
            include_default_source=True,
            include_base_tools=True,
            memory_blocks=[
                {"label": "human", "value": "Smoke-test user."},
                {"label": "persona", "value": "Smoke-test memory agent."},
            ],
            context_window_limit=8000,
        )
        created = client.agents.passages.create(str(agent.id), text=args.text, tags=["oraclemem_smoke"])
        results = client.agents.passages.search(str(agent.id), query=args.query, top_k=3)
        record = {
            "status": "ok",
            "scope": "agent",
            "embedding": args.embedding,
            "agent_id": str(agent.id),
            "created_passage": compact(created),
            "query": args.query,
            "raw_response": compact(results),
        }
        serialized = json.dumps(record)
        record["found_inserted_text"] = args.text in serialized
        try:
            client.agents.delete(str(agent.id))
        except Exception as exc:  # pragma: no cover - cleanup best effort.
            record["delete_error"] = {"type": type(exc).__name__, "message": str(exc)}
    else:
        archive = client.archives.create(name=f"oraclemem_passage_smoke_{int(time.time())}", embedding=args.embedding)
        passage = client.archives.passages.create(
            str(archive.id),
            text=args.text,
            metadata={"smoke": True},
            tags=["oraclemem_smoke"],
        )
        results = client.passages.search(archive_id=str(archive.id), query=args.query, limit=3)
        result_items = compact(results)
        record = {
            "status": "ok",
            "scope": "archive",
            "embedding": args.embedding,
            "archive_id": str(archive.id),
            "inserted_passage_id": str(passage.id),
            "query": args.query,
            "result_count": len(result_items) if isinstance(result_items, list) else None,
            "found_inserted_text": args.text in json.dumps(result_items),
            "results": result_items,
        }

    if not record.get("found_inserted_text"):
        raise RuntimeError("Passage search did not retrieve the inserted smoke passage")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
