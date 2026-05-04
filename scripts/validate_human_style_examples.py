"""Validate human-style OracleMem natural-example JSONL files.

Each input row is expected to be a self-contained example with sessions,
messages, evidence units, query-required evidence IDs, and candidate memories
with per-unit coverage scores. This script performs structural validation only;
it does not judge annotation quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TOP_LEVEL_FIELDS: dict[str, tuple[str, ...]] = {
    "id": ("id", "example_id", "instance_id"),
    "domain": ("domain",),
    "sessions": ("sessions",),
    "future_query": ("future_query", "query", "question"),
    "required_unit_ids_for_query": ("required_unit_ids_for_query",),
    "evidence_units": ("evidence_units",),
    "candidate_memories": ("candidate_memories",),
}

FUTURE_QUERY_FIELDS: dict[str, tuple[str, ...]] = {
    "text": ("text", "query", "question"),
    "answer": ("answer", "gold_answer"),
    "should_abstain": ("should_abstain",),
}

SESSION_FIELDS: dict[str, tuple[str, ...]] = {
    "session_id": ("session_id", "id"),
    "messages": ("messages",),
}

MESSAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "role": ("role", "speaker"),
    "content": ("content", "text"),
}

EVIDENCE_UNIT_FIELDS: dict[str, tuple[str, ...]] = {
    "unit_id": ("unit_id", "id", "evidence_unit_id"),
    "text": ("canonical_text", "text"),
}

CANDIDATE_FIELDS: dict[str, tuple[str, ...]] = {
    "candidate_id": ("candidate_id", "memory_id", "id"),
    "text": ("text", "serialized", "content"),
    "coverage": ("coverage", "coverage_by_unit", "unit_coverage", "coverage_scores"),
}


class Validator:
    def __init__(self, max_errors: int) -> None:
        self.max_errors = max_errors
        self.errors: list[str] = []
        self.omitted_errors = 0
        self.seen_example_ids: dict[str, str] = {}
        self.domain_counts: Counter[str] = Counter()
        self.domain_sessions: Counter[str] = Counter()
        self.domain_messages: Counter[str] = Counter()
        self.domain_units: Counter[str] = Counter()
        self.domain_candidates: Counter[str] = Counter()
        self.rows_seen = 0

    def add_error(self, location: str, message: str) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(f"{location}: {message}")
        else:
            self.omitted_errors += 1

    @property
    def truncated_errors(self) -> bool:
        return self.omitted_errors > 0


def first_present(record: dict[str, Any], aliases: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in aliases:
        if key in record:
            return key, record[key]
    return None, None


def require_fields(
    validator: Validator,
    location: str,
    record: Any,
    fields: dict[str, tuple[str, ...]],
) -> bool:
    if not isinstance(record, dict):
        validator.add_error(location, "expected object")
        return False

    ok = True
    for canonical_name, aliases in fields.items():
        key, _ = first_present(record, aliases)
        if key is None:
            alias_text = "/".join(aliases)
            validator.add_error(location, f"missing required field {canonical_name} ({alias_text})")
            ok = False
    return ok


def as_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def validate_stringish(validator: Validator, location: str, field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        validator.add_error(location, f"{field_name} must be a non-empty string")


def validate_id_list(
    validator: Validator,
    location: str,
    field_name: str,
    value: Any,
) -> list[str]:
    if not isinstance(value, list):
        validator.add_error(location, f"{field_name} must be a list")
        return []

    ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_id = as_id(item)
        item_location = f"{location}.{field_name}[{index}]"
        if item_id is None:
            validator.add_error(item_location, "must be a non-empty string or integer ID")
            continue
        if item_id in seen:
            validator.add_error(item_location, f"duplicate ID {item_id!r}")
        seen.add(item_id)
        ids.append(item_id)
    return ids


def validate_sessions(validator: Validator, location: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, list):
        validator.add_error(location, "sessions must be a list")
        return 0, 0

    session_ids: set[str] = set()
    message_count = 0
    for session_index, session in enumerate(value):
        session_location = f"{location}.sessions[{session_index}]"
        if not require_fields(validator, session_location, session, SESSION_FIELDS):
            continue

        _, raw_session_id = first_present(session, SESSION_FIELDS["session_id"])
        session_id = as_id(raw_session_id)
        if session_id is None:
            validator.add_error(session_location, "session_id must be a non-empty string or integer ID")
        elif session_id in session_ids:
            validator.add_error(session_location, f"duplicate session_id {session_id!r}")
        else:
            session_ids.add(session_id)

        _, messages = first_present(session, SESSION_FIELDS["messages"])
        if not isinstance(messages, list):
            validator.add_error(f"{session_location}.messages", "messages must be a list")
            continue

        message_count += len(messages)
        for message_index, message in enumerate(messages):
            message_location = f"{session_location}.messages[{message_index}]"
            if not require_fields(validator, message_location, message, MESSAGE_FIELDS):
                continue
            _, role = first_present(message, MESSAGE_FIELDS["role"])
            _, content = first_present(message, MESSAGE_FIELDS["content"])
            validate_stringish(validator, message_location, "role", role)
            validate_stringish(validator, message_location, "content", content)

    return len(value), message_count


def validate_evidence_units(validator: Validator, location: str, value: Any) -> tuple[set[str], int]:
    if not isinstance(value, list):
        validator.add_error(location, "evidence_units must be a list")
        return set(), 0

    unit_ids: set[str] = set()
    for unit_index, unit in enumerate(value):
        unit_location = f"{location}.evidence_units[{unit_index}]"
        if not require_fields(validator, unit_location, unit, EVIDENCE_UNIT_FIELDS):
            continue

        _, raw_unit_id = first_present(unit, EVIDENCE_UNIT_FIELDS["unit_id"])
        unit_id = as_id(raw_unit_id)
        if unit_id is None:
            validator.add_error(unit_location, "unit_id must be a non-empty string or integer ID")
        elif unit_id in unit_ids:
            validator.add_error(unit_location, f"duplicate unit_id {unit_id!r}")
        else:
            unit_ids.add(unit_id)

        _, text = first_present(unit, EVIDENCE_UNIT_FIELDS["text"])
        validate_stringish(validator, unit_location, "text", text)

    return unit_ids, len(value)


def validate_candidate_memories(
    validator: Validator,
    location: str,
    value: Any,
    evidence_unit_ids: set[str],
) -> int:
    if not isinstance(value, list):
        validator.add_error(location, "candidate_memories must be a list")
        return 0

    candidate_ids: set[str] = set()
    for candidate_index, candidate in enumerate(value):
        candidate_location = f"{location}.candidate_memories[{candidate_index}]"
        if not require_fields(validator, candidate_location, candidate, CANDIDATE_FIELDS):
            continue

        _, raw_candidate_id = first_present(candidate, CANDIDATE_FIELDS["candidate_id"])
        candidate_id = as_id(raw_candidate_id)
        if candidate_id is None:
            validator.add_error(candidate_location, "candidate_id must be a non-empty string or integer ID")
        elif candidate_id in candidate_ids:
            validator.add_error(candidate_location, f"duplicate candidate_id {candidate_id!r}")
        else:
            candidate_ids.add(candidate_id)

        _, text = first_present(candidate, CANDIDATE_FIELDS["text"])
        validate_stringish(validator, candidate_location, "text", text)

        coverage_key, coverage = first_present(candidate, CANDIDATE_FIELDS["coverage"])
        if not isinstance(coverage, dict):
            validator.add_error(candidate_location, "coverage must be an object mapping unit IDs to scores")
            continue

        for unit_id, score in coverage.items():
            coverage_location = f"{candidate_location}.{coverage_key}[{unit_id!r}]"
            if unit_id not in evidence_unit_ids:
                validator.add_error(coverage_location, "references unknown evidence unit")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                validator.add_error(coverage_location, "coverage score must be numeric")
                continue
            if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
                validator.add_error(coverage_location, "coverage score must be in [0, 1]")

    return len(value)


def validate_example(validator: Validator, path: Path, line_number: int, row: Any) -> None:
    validator.rows_seen += 1
    location = f"{path}:{line_number}"

    if not require_fields(validator, location, row, TOP_LEVEL_FIELDS):
        if not isinstance(row, dict):
            return

    assert isinstance(row, dict)

    _, raw_example_id = first_present(row, TOP_LEVEL_FIELDS["id"])
    example_id = as_id(raw_example_id)
    if example_id is None:
        validator.add_error(location, "id must be a non-empty string or integer ID")
        example_key = f"<missing-id:{path}:{line_number}>"
    else:
        example_key = example_id
        if example_id in validator.seen_example_ids:
            first_location = validator.seen_example_ids[example_id]
            validator.add_error(location, f"duplicate example id {example_id!r}; first seen at {first_location}")
        else:
            validator.seen_example_ids[example_id] = location

    _, domain_value = first_present(row, TOP_LEVEL_FIELDS["domain"])
    domain = domain_value if isinstance(domain_value, str) and domain_value else "<missing>"
    if domain == "<missing>":
        validator.add_error(location, "domain must be a non-empty string")

    _, sessions = first_present(row, TOP_LEVEL_FIELDS["sessions"])
    _, evidence_units = first_present(row, TOP_LEVEL_FIELDS["evidence_units"])
    _, required_units = first_present(row, TOP_LEVEL_FIELDS["required_unit_ids_for_query"])
    _, candidate_memories = first_present(row, TOP_LEVEL_FIELDS["candidate_memories"])

    session_count, message_count = validate_sessions(validator, location, sessions)
    evidence_unit_ids, unit_count = validate_evidence_units(validator, location, evidence_units)
    required_unit_ids = validate_id_list(
        validator,
        location,
        "required_unit_ids_for_query",
        required_units,
    )

    for unit_id in required_unit_ids:
        if unit_id not in evidence_unit_ids:
            validator.add_error(
                f"{location}.required_unit_ids_for_query",
                f"required unit {unit_id!r} is not present in evidence_units",
            )

    candidate_count = validate_candidate_memories(validator, location, candidate_memories, evidence_unit_ids)

    validator.domain_counts[domain] += 1
    validator.domain_sessions[domain] += session_count
    validator.domain_messages[domain] += message_count
    validator.domain_units[domain] += unit_count
    validator.domain_candidates[domain] += candidate_count

    _, future_query = first_present(row, TOP_LEVEL_FIELDS["future_query"])
    if isinstance(future_query, str):
        validate_stringish(validator, location, "future_query", future_query)
    elif isinstance(future_query, dict):
        if require_fields(validator, f"{location}.future_query", future_query, FUTURE_QUERY_FIELDS):
            _, query_text = first_present(future_query, FUTURE_QUERY_FIELDS["text"])
            _, answer = first_present(future_query, FUTURE_QUERY_FIELDS["answer"])
            _, should_abstain = first_present(
                future_query, FUTURE_QUERY_FIELDS["should_abstain"]
            )
            validate_stringish(validator, f"{location}.future_query", "text", query_text)
            validate_stringish(validator, f"{location}.future_query", "answer", answer)
            if not isinstance(should_abstain, bool):
                validator.add_error(
                    f"{location}.future_query",
                    "should_abstain must be a boolean",
                )
    else:
        validator.add_error(location, "future_query must be a non-empty string or object")

    # Keep linters honest and make debugging partial records easier.
    _ = example_key


def validate_path(validator: Validator, path: Path) -> None:
    if not path.exists():
        validator.add_error(str(path), "file does not exist")
        return
    if not path.is_file():
        validator.add_error(str(path), "path is not a file")
        return

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                validator.add_error(f"{path}:{line_number}", "blank lines are not valid JSONL records")
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                validator.add_error(f"{path}:{line_number}", f"invalid JSON: {exc.msg}")
                continue
            validate_example(validator, path, line_number, row)


def print_summary(validator: Validator, paths: list[Path]) -> None:
    print(f"Validated {validator.rows_seen} records from {len(paths)} file(s).")
    print("Domain summary:")
    if not validator.domain_counts:
        print("  <none>")
    else:
        for domain in sorted(validator.domain_counts):
            print(
                "  "
                f"{domain}: "
                f"records={validator.domain_counts[domain]}, "
                f"sessions={validator.domain_sessions[domain]}, "
                f"messages={validator.domain_messages[domain]}, "
                f"evidence_units={validator.domain_units[domain]}, "
                f"candidate_memories={validator.domain_candidates[domain]}"
            )

    if validator.errors:
        print(f"Errors ({len(validator.errors)} shown):")
        for error in validator.errors:
            print(f"  - {error}")
        if validator.truncated_errors:
            print(f"  - {validator.omitted_errors} additional error(s) omitted after --max-errors={validator.max_errors}")
    else:
        print("Errors: none")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate human-style OracleMem natural-example JSONL files.",
    )
    parser.add_argument("jsonl_paths", nargs="+", type=Path, help="One or more JSONL files to validate.")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Maximum number of validation errors to print before truncating output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    max_errors = args.max_errors if args.max_errors > 0 else 1
    validator = Validator(max_errors=max_errors)

    for path in args.jsonl_paths:
        validate_path(validator, path)

    print_summary(validator, args.jsonl_paths)
    return 1 if validator.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
