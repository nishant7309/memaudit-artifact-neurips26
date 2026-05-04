"""Core data structures for OracleMem.

The benchmark treats memory writing as selection over virtual
experience-representation items. These dataclasses intentionally stay small
and JSON-friendly so exact synthetic instances can be inspected and serialized
without framework dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceUnit:
  unit_id: str
  kind: str
  text: str
  proposition_id: str
  timestamp: int
  state: str = "current"
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class Experience:
  experience_id: str
  session_id: str
  timestamp: int
  text: str
  visible_unit_ids: tuple[str, ...]
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class CandidateMemory:
  candidate_id: str
  experience_id: str
  representation: str
  text: str
  cost: int
  coverage: dict[str, float]
  generator: str = "oracle"
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class Query:
  query_id: str
  text: str
  category: str
  required_unit_ids: tuple[str, ...]
  answer: str
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class Instance:
  instance_id: str
  seed: int
  units: tuple[EvidenceUnit, ...]
  experiences: tuple[Experience, ...]
  candidates: tuple[CandidateMemory, ...]
  queries: tuple[Query, ...]
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class SolverResult:
  method: str
  budget: int
  selected_ids: tuple[str, ...]
  utility: float
  cost: int
  ratio: float | None = None
  ratio_basis: str = "none"
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)
