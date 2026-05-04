"""Deterministic OracleMem-Small instance generator."""

from __future__ import annotations

import random

from .schema import CandidateMemory, EvidenceUnit, Experience, Instance, Query


PREFERENCES = [
    ("coffee", "prefers pour-over coffee"),
    ("seat", "prefers aisle seats"),
    ("food", "is vegetarian"),
    ("music", "likes ambient music while working"),
    ("hotel", "prefers quiet hotels"),
    ("editor", "uses VS Code"),
    ("language", "prefers Python examples"),
    ("meeting", "prefers morning meetings"),
]

UPDATED_VALUES = {
    "coffee": "prefers green tea",
    "seat": "prefers window seats",
    "food": "is no longer vegetarian",
    "music": "prefers silence while working",
    "hotel": "prefers hotels near transit",
    "editor": "uses Neovim",
    "language": "prefers Rust examples",
    "meeting": "prefers afternoon meetings",
}

TOOL_FACTS = [
    ("indexer", "build_index skips tombstoned memories before embedding"),
    ("calendar", "schedule_sync retries failed calendar writes once"),
    ("tests", "test_memory_budget checks one representation per experience"),
    ("retriever", "search filters deleted memories before reranking"),
]


def _cost(text: str) -> int:
  return max(1, len(text.split()))


def _candidate(
    experience_id: str,
    representation: str,
    text: str,
    coverage: dict[str, float],
    generator: str = "oracle",
) -> CandidateMemory:
  return CandidateMemory(
      candidate_id=f"{experience_id}:{representation}",
      experience_id=experience_id,
      representation=representation,
      text=text,
      cost=_cost(text),
      coverage={unit: float(value) for unit, value in coverage.items() if value > 0},
      generator=generator,
  )


def generate_instance(
    seed: int = 0,
    num_base_facts: int = 6,
    num_updates: int = 3,
    num_tool_facts: int = 2,
) -> Instance:
  """Generate one exact OracleMem-Small synthetic instance.

  The visible experience stream contains no query ids. Held-out queries are
  generated only after units and candidates exist, which keeps write-time
  policies from seeing future evidence requirements.
  """
  rng = random.Random(seed)
  facts = rng.sample(PREFERENCES, k=min(num_base_facts, len(PREFERENCES)))
  update_keys = [key for key, _ in facts if key in UPDATED_VALUES]
  update_keys = rng.sample(update_keys, k=min(num_updates, len(update_keys)))
  tool_facts = rng.sample(TOOL_FACTS, k=min(num_tool_facts, len(TOOL_FACTS)))

  units: list[EvidenceUnit] = []
  experiences: list[Experience] = []
  candidates: list[CandidateMemory] = []
  queries: list[Query] = []
  current_unit_for_key: dict[str, str] = {}
  invalidation_for_key: dict[str, str] = {}
  timestamp = 0

  for key, text in facts:
    unit_id = f"unit:{key}:initial"
    prop_id = f"prop:{key}"
    units.append(EvidenceUnit(unit_id, "preference", text, prop_id, timestamp))
    exp_id = f"exp:{timestamp:04d}"
    visible = (unit_id,)
    exp_text = f"User says: My {key} preference is that I {text}."
    experiences.append(Experience(exp_id, f"session:{timestamp // 4}", timestamp, exp_text, visible))
    candidates.extend([
        _candidate(exp_id, "raw", exp_text, {unit_id: 1.0}),
        _candidate(exp_id, "fact", f"FACT {prop_id}: user {text}.", {unit_id: 1.0}),
        _candidate(exp_id, "summary", f"SUMMARY: user {text}.", {unit_id: 0.6}),
    ])
    current_unit_for_key[key] = unit_id
    timestamp += 1

  for key in update_keys:
    old_unit = current_unit_for_key[key]
    new_text = UPDATED_VALUES[key]
    current_id = f"unit:{key}:current"
    invalid_id = f"unit:{key}:invalidates_initial"
    prop_id = f"prop:{key}"
    units.append(EvidenceUnit(current_id, "current_preference", new_text, prop_id, timestamp))
    units.append(EvidenceUnit(
        invalid_id,
        "invalidation",
        f"{prop_id} initial value is no longer current",
        prop_id,
        timestamp,
        state="superseded",
        metadata={"invalidates": old_unit, "replacement": current_id},
    ))
    exp_id = f"exp:{timestamp:04d}"
    visible = (current_id, invalid_id)
    exp_text = f"User correction: Actually, for {key}, I changed my mind and now {new_text}."
    experiences.append(Experience(exp_id, f"session:{timestamp // 4}", timestamp, exp_text, visible))
    candidates.extend([
        _candidate(exp_id, "raw", exp_text, {current_id: 1.0, invalid_id: 1.0}),
        _candidate(exp_id, "fact", f"FACT {prop_id}: user now {new_text}.", {current_id: 1.0}),
        _candidate(exp_id, "tombstone", f"TOMBSTONE {prop_id}: initial value invalid after t={timestamp}.", {invalid_id: 1.0}),
        _candidate(
            exp_id,
            "compound_update",
            f"UPDATE {prop_id}: user now {new_text}; tombstone previous value after t={timestamp}.",
            {current_id: 1.0, invalid_id: 1.0},
        ),
        _candidate(exp_id, "summary", f"SUMMARY: {key} changed; current state is {new_text}.", {current_id: 0.7, invalid_id: 0.5}),
    ])
    current_unit_for_key[key] = current_id
    invalidation_for_key[key] = invalid_id
    timestamp += 1

  for key, text in tool_facts:
    unit_id = f"unit:tool:{key}"
    prop_id = f"prop:tool:{key}"
    units.append(EvidenceUnit(unit_id, "tool_outcome", text, prop_id, timestamp))
    exp_id = f"exp:{timestamp:04d}"
    exp_text = f"Tool trace: after debugging {key}, the result is that {text}."
    experiences.append(Experience(exp_id, f"session:{timestamp // 4}", timestamp, exp_text, (unit_id,)))
    candidates.extend([
        _candidate(exp_id, "raw", exp_text, {unit_id: 1.0}),
        _candidate(exp_id, "graph_edge", f"EDGE tool:{key} outcome '{text}'.", {unit_id: 1.0}),
        _candidate(exp_id, "skill", f"SKILL: when handling {key}, remember that {text}.", {unit_id: 0.9}),
        _candidate(exp_id, "summary", f"SUMMARY: {key} outcome: {text}.", {unit_id: 0.6}),
    ])
    timestamp += 1

  query_id = 0
  for key, unit_id in current_unit_for_key.items():
    unit = next(item for item in units if item.unit_id == unit_id)
    required = [unit_id]
    if key in invalidation_for_key:
      required.append(invalidation_for_key[key])
      category = "current_truth_update"
      text = f"What is the user's current {key} preference?"
    else:
      category = "single_hop_fact"
      text = f"What did the user say about {key}?"
    queries.append(Query(
        f"query:{query_id:04d}",
        text,
        category,
        tuple(required),
        unit.text,
    ))
    query_id += 1

  for key in invalidation_for_key:
    queries.append(Query(
        f"query:{query_id:04d}",
        f"Is the user's initial {key} preference still current?",
        "deletion_or_supersession",
        (invalidation_for_key[key],),
        "No, it was superseded.",
    ))
    query_id += 1

  for key, _ in tool_facts:
    unit_id = f"unit:tool:{key}"
    unit = next(item for item in units if item.unit_id == unit_id)
    queries.append(Query(
        f"query:{query_id:04d}",
        f"What should the agent remember about {key}?",
        "tool_outcome",
        (unit_id,),
        unit.text,
    ))
    query_id += 1

  queries.append(Query(
      f"query:{query_id:04d}",
      "What is the user's favorite vacation island?",
      "abstention",
      tuple(),
      "Insufficient evidence.",
  ))

  return Instance(
      instance_id=f"oraclemem-small-{seed}",
      seed=seed,
      units=tuple(units),
      experiences=tuple(experiences),
      candidates=tuple(candidates),
      queries=tuple(queries),
      metadata={
          "num_base_facts": num_base_facts,
          "num_updates": num_updates,
          "num_tool_facts": num_tool_facts,
      },
  )
