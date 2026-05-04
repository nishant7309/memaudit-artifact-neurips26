"""OracleMem exact-small evaluation utilities.

This module is intentionally pure Python stdlib.  It implements the MVP
evaluation path described in the OracleMem notes:

* sparse semantic coverage over candidate memory representations;
* one storage budget plus one-representation-per-experience feasibility;
* exact finite-instance optimization for small synthetic benchmarks;
* baseline writers and denominator labels that distinguish exact optima,
  certified upper bounds, and references.

Expected external API shape
---------------------------
Future generator/solver modules can interoperate by providing candidate-like
objects or dictionaries with these fields:

``candidate_id``, ``experience_id``, ``representation_type``, ``serialized``,
``cost`` (integer or ``{"total": int}``), and ``coverage`` (mapping from unit
id to fidelity, or a list of ``{"unit_id": ..., "fidelity": ...}`` records).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import itertools
import json
import math
import random
import time

from .writer_baselines import (
    WRITER_BASELINE_DESCRIPTIONS,
    WRITER_BASELINE_METHODS,
    select_writer_baseline,
)


DEFAULT_METHODS: Tuple[str, ...] = (
    "opt",
    "oracle_gvt",
    "greedy",
    "density_only",
    "recency_raw",
    "reservoir_raw",
    "fact_only",
    "summary_only",
    "no_tombstone_gvt",
    "no_tombstone_greedy",
    "no_tombstone_opt",
    "memgpt_tiered",
    "mem0_extract",
    "amem_graph",
    "amac_admission",
    "generic_candidate_opt",
    "generic_candidate_gvt",
    "summary_candidate_opt",
)

ESTIMATED_METHODS: Tuple[str, ...] = (
    "estimated_gvt",
    "estimated_utility",
)

SUPPORTED_METHODS: Tuple[str, ...] = tuple(dict.fromkeys((*DEFAULT_METHODS, *ESTIMATED_METHODS)))

DEFAULT_ESTIMATOR_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_ESTIMATOR_PROFILE = "gemini_flash_lite_v1"
NOISY_ESTIMATOR_PROFILE = "noisy_gemini_flash_lite_v1"
LEARNED_ESTIMATOR_PROFILE = "synthetic_train_dev_v1"
LOCAL_LEARNED_ESTIMATOR_MODEL = "local-linear-synthetic-utility-v1"
ESTIMATOR_PROFILES: Tuple[str, ...] = (
    DEFAULT_ESTIMATOR_PROFILE,
    NOISY_ESTIMATOR_PROFILE,
    LEARNED_ESTIMATOR_PROFILE,
    "external",
)

TOMBSTONE_TYPES = {"tombstone", "compound_update"}
GENERIC_CANDIDATE_TYPES = {"raw", "raw_span", "atomic_fact", "fact", "summary"}
CANDIDATE_QUALITY_EXACT_METHODS: Mapping[str, Tuple[str, ...]] = {
    "generic_candidate_opt": tuple(sorted(GENERIC_CANDIDATE_TYPES)),
    "summary_candidate_opt": ("summary",),
}


@dataclass(frozen=True)
class CandidateMemory:
    """A virtual memory representation candidate for one experience."""

    candidate_id: str
    experience_id: str
    representation_type: str
    serialized: str
    cost: int
    coverage: Mapping[str, float]
    time_index: int = 0
    generator: str = "oracle"
    confidence: float = 1.0
    estimated_value: Optional[float] = None
    estimated_coverage: Mapping[str, float] = field(default_factory=dict)
    estimator_model: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if not self.experience_id:
            raise ValueError("experience_id must be nonempty")
        if self.cost < 0:
            raise ValueError(f"{self.candidate_id} has negative cost")
        clean_coverage: Dict[str, float] = {}
        for unit_id, fidelity in dict(self.coverage).items():
            value = float(fidelity)
            if value < 0:
                raise ValueError(f"{self.candidate_id} has negative coverage")
            if value > 0:
                clean_coverage[str(unit_id)] = value
        object.__setattr__(self, "coverage", clean_coverage)
        clean_estimated_coverage: Dict[str, float] = {}
        for unit_id, fidelity in dict(self.estimated_coverage or {}).items():
            value = float(fidelity)
            if value < 0:
                raise ValueError(f"{self.candidate_id} has negative estimated coverage")
            if value > 0:
                clean_estimated_coverage[str(unit_id)] = value
        object.__setattr__(self, "estimated_coverage", clean_estimated_coverage)
        if self.estimated_value is not None:
            estimated_value = float(self.estimated_value)
            if estimated_value < 0:
                raise ValueError(f"{self.candidate_id} has negative estimated value")
            object.__setattr__(self, "estimated_value", estimated_value)

    def to_json(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "experience_id": self.experience_id,
            "representation_type": self.representation_type,
            "serialized": self.serialized,
            "cost": self.cost,
            "coverage": dict(sorted(self.coverage.items())),
            "time_index": self.time_index,
            "generator": self.generator,
            "confidence": self.confidence,
            "estimated_value": self.estimated_value,
            "estimated_coverage": dict(sorted(self.estimated_coverage.items())),
            "estimator_model": self.estimator_model,
        }


@dataclass(frozen=True)
class OracleMemInstance:
    """A finite OracleMem evaluation instance."""

    instance_id: str
    candidates: Sequence[CandidateMemory]
    unit_weights: Mapping[str, float]
    seed: Optional[int] = None
    current_units: Sequence[str] = field(default_factory=tuple)
    invalidation_units: Sequence[str] = field(default_factory=tuple)
    stale_units: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{self.instance_id} has duplicate candidate ids")
        clean_weights = {str(k): float(v) for k, v in dict(self.unit_weights).items()}
        for unit_id, weight in clean_weights.items():
            if weight < 0:
                raise ValueError(f"{self.instance_id} has negative weight for {unit_id}")
        object.__setattr__(self, "unit_weights", clean_weights)
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "current_units", tuple(self.current_units))
        object.__setattr__(self, "invalidation_units", tuple(self.invalidation_units))
        object.__setattr__(self, "stale_units", tuple(self.stale_units))


@dataclass(frozen=True)
class SelectionResult:
    """One method's selected memory store for one instance and budget."""

    instance_id: str
    seed: Optional[int]
    distribution: str
    budget: int
    method: str
    selected_candidate_ids: Sequence[str]
    selected_cost: int
    objective_value: float
    denominator_label: str
    ratio_to_opt: Optional[float]
    ratio_to_upper_bound: Optional[float]
    ratio_to_reference: Optional[float]
    optimum_value: Optional[float]
    upper_bound: Optional[float]
    upper_bound_source: Optional[str]
    reference_value: Optional[float]
    runtime_sec: float
    budget_feasible: bool
    group_feasible: bool
    representation_mix: Mapping[str, int]
    update_metrics: Mapping[str, float]
    retrieval_metrics: Mapping[str, Any] = field(default_factory=dict)
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "distribution": self.distribution,
            "budget": self.budget,
            "method": self.method,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_cost": self.selected_cost,
            "objective_value": self.objective_value,
            "denominator_label": self.denominator_label,
            "ratio_to_opt": self.ratio_to_opt,
            "ratio_to_upper_bound": self.ratio_to_upper_bound,
            "ratio_to_reference": self.ratio_to_reference,
            "optimum_value": self.optimum_value,
            "upper_bound": self.upper_bound,
            "upper_bound_source": self.upper_bound_source,
            "reference_value": self.reference_value,
            "runtime_sec": self.runtime_sec,
            "budget_feasible": self.budget_feasible,
            "group_feasible": self.group_feasible,
            "representation_mix": dict(sorted(self.representation_mix.items())),
            "update_metrics": dict(sorted(self.update_metrics.items())),
            "retrieval_metrics": self.retrieval_metrics,
            "policy_metadata": dict(self.policy_metadata),
        }


@dataclass(frozen=True)
class EstimatedUtilityModel:
    """Local train/dev utility estimator for non-oracle writer diagnostics.

    The model is fit from oracle singleton utilities on train instances. At
    dev-time it predicts utility only from visible candidate metadata: text,
    representation type, cost, confidence, and stream position.
    """

    estimator_model: str
    estimator_profile: str
    feature_names: Tuple[str, ...]
    weights: Tuple[float, ...]
    ridge: float
    noise_scale: float
    noise_seed: int
    train_distributions: Tuple[str, ...]
    train_seeds: Tuple[int, ...]
    train_instance_count: int
    train_candidate_count: int
    train_target_mean: float

    def predict(self, candidate: CandidateMemory, universe: Sequence[CandidateMemory]) -> float:
        features = learned_candidate_features(candidate, universe)
        value = sum(
            self.weights[index] * features.get(name, 0.0)
            for index, name in enumerate(self.feature_names)
        )
        if self.noise_scale > 0:
            value *= 1.0 + self.noise_scale * _stable_unit_noise(
                f"{self.noise_seed}:{self.estimator_profile}:{candidate.candidate_id}"
            )
        return max(0.0, value)

    def metadata(self) -> Dict[str, Any]:
        return {
            "trained_estimator": True,
            "estimator_model": self.estimator_model,
            "estimator_profile": self.estimator_profile,
            "feature_count": len(self.feature_names),
            "ridge": self.ridge,
            "noise_scale": self.noise_scale,
            "noise_seed": self.noise_seed,
            "train_distributions": list(self.train_distributions),
            "train_seeds": list(self.train_seeds),
            "train_instance_count": self.train_instance_count,
            "train_candidate_count": self.train_candidate_count,
            "train_target_mean": self.train_target_mean,
            "training_target": "oracle_singleton_utility_on_train_instances",
            "oracle_coverage_used_for_training": True,
            "oracle_coverage_used_for_dev_decision": False,
            "dev_selection_features": "visible_candidate_metadata_only",
        }


def parse_int_list(value: str) -> List[int]:
    """Parse comma/space separated integers for CLI arguments."""

    parts = value.replace(",", " ").split()
    if not parts:
        raise ValueError("expected at least one integer")
    return [int(part) for part in parts]


def parse_token_list(value: str) -> List[str]:
    """Parse comma/space separated CLI tokens while preserving decimals."""

    parts = value.replace(",", " ").split()
    if not parts:
        raise ValueError("expected at least one value")
    return parts


def _coerce_coverage(raw_coverage: Any) -> Dict[str, float]:
    if isinstance(raw_coverage, Mapping):
        return {str(unit): float(value) for unit, value in raw_coverage.items()}
    coverage: Dict[str, float] = {}
    for row in raw_coverage or ():
        if isinstance(row, Mapping):
            unit_id = row.get("unit_id", row.get("id"))
            fidelity = row.get("fidelity", row.get("coverage", 1.0))
        else:
            unit_id = getattr(row, "unit_id")
            fidelity = getattr(row, "fidelity", 1.0)
        coverage[str(unit_id)] = float(fidelity)
    return coverage


def coerce_candidate(obj: Any) -> CandidateMemory:
    """Convert a future API candidate object or dict into CandidateMemory."""

    if isinstance(obj, CandidateMemory):
        return obj
    if isinstance(obj, Mapping):
        get = obj.get
    else:
        get = lambda key, default=None: getattr(obj, key, default)

    cost_obj = get("cost", get("storage_tokens", 0))
    if isinstance(cost_obj, Mapping):
        cost = int(cost_obj.get("total", cost_obj.get("storage_tokens", 0)))
    else:
        cost = int(cost_obj)

    coverage = _coerce_coverage(get("coverage", {}))
    estimated_value_raw = get(
        "estimated_value",
        get("estimated_utility", get("utility_estimate", None)),
    )
    estimated_value = (
        None if estimated_value_raw is None else float(estimated_value_raw)
    )

    return CandidateMemory(
        candidate_id=str(get("candidate_id")),
        experience_id=str(get("experience_id")),
        representation_type=str(get("representation_type", get("type", "unknown"))),
        serialized=str(get("serialized", "")),
        cost=cost,
        coverage=coverage,
        time_index=int(get("time_index", get("turn_index", 0))),
        generator=str(get("generator", "oracle")),
        confidence=float(get("confidence", 1.0)),
        estimated_value=estimated_value,
        estimated_coverage=_coerce_coverage(get("estimated_coverage", {})),
        estimator_model=str(get("estimator_model", get("estimated_model", ""))),
    )


def load_candidates_jsonl(path: str | Path) -> List[CandidateMemory]:
    """Load candidate memories from JSONL using the expected candidate API."""

    candidates: List[CandidateMemory] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                candidates.append(coerce_candidate(json.loads(stripped)))
            except Exception as exc:  # pragma: no cover - defensive context
                raise ValueError(f"bad candidate JSONL at line {line_number}") from exc
    return candidates


def make_instance_from_candidates(
    candidates: Sequence[Any],
    *,
    instance_id: str = "external",
    unit_weights: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
) -> OracleMemInstance:
    """Create an evaluation instance from candidate-like objects."""

    coerced = [coerce_candidate(candidate) for candidate in candidates]
    weights = dict(unit_weights) if unit_weights is not None else infer_unit_weights(coerced)
    return OracleMemInstance(instance_id, coerced, weights, seed=seed)


def infer_unit_weights(candidates: Sequence[CandidateMemory]) -> Dict[str, float]:
    units = sorted({unit for candidate in candidates for unit in candidate.coverage})
    return {unit: 1.0 for unit in units}


def candidates_by_id(candidates: Sequence[CandidateMemory]) -> Dict[str, CandidateMemory]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def ordered_groups(candidates: Sequence[CandidateMemory]) -> List[List[CandidateMemory]]:
    groups: Dict[str, List[CandidateMemory]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.experience_id, []).append(candidate)
    return [
        sorted(groups[key], key=lambda candidate: (candidate.cost, candidate.candidate_id))
        for key in sorted(
            groups,
            key=lambda group_id: (
                min(candidate.time_index for candidate in groups[group_id]),
                group_id,
            ),
        )
    ]


def selected_candidates(
    candidates: Sequence[CandidateMemory], selected_ids: Sequence[str]
) -> List[CandidateMemory]:
    by_id = candidates_by_id(candidates)
    return [by_id[candidate_id] for candidate_id in selected_ids]


def coverage_totals(selected: Sequence[CandidateMemory]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for candidate in selected:
        for unit_id, fidelity in candidate.coverage.items():
            totals[unit_id] = totals.get(unit_id, 0.0) + fidelity
    return totals


def saturated(value: float, saturation: str = "cap1") -> float:
    if saturation == "cap1":
        return min(1.0, value)
    if saturation == "log1p":
        return math.log1p(value)
    if saturation == "linear":
        return value
    raise ValueError(f"unknown saturation: {saturation}")


def value_from_totals(
    totals: Mapping[str, float],
    unit_weights: Mapping[str, float],
    *,
    saturation: str = "cap1",
) -> float:
    return sum(
        float(weight) * saturated(float(totals.get(unit_id, 0.0)), saturation)
        for unit_id, weight in unit_weights.items()
    )


def objective_value(
    selected: Sequence[CandidateMemory],
    unit_weights: Mapping[str, float],
    *,
    saturation: str = "cap1",
) -> float:
    return value_from_totals(coverage_totals(selected), unit_weights, saturation=saturation)


def marginal_value(
    candidate: CandidateMemory,
    current_totals: Mapping[str, float],
    unit_weights: Mapping[str, float],
    *,
    saturation: str = "cap1",
) -> float:
    before = value_from_totals(current_totals, unit_weights, saturation=saturation)
    after_totals = dict(current_totals)
    for unit_id, fidelity in candidate.coverage.items():
        after_totals[unit_id] = after_totals.get(unit_id, 0.0) + fidelity
    after = value_from_totals(after_totals, unit_weights, saturation=saturation)
    return after - before


def total_cost(selected: Sequence[CandidateMemory]) -> int:
    return sum(candidate.cost for candidate in selected)


def feasibility_report(
    candidates: Sequence[CandidateMemory], selected_ids: Sequence[str], budget: int
) -> Dict[str, Any]:
    by_id = candidates_by_id(candidates)
    selected = [by_id[candidate_id] for candidate_id in selected_ids]
    groups = [candidate.experience_id for candidate in selected]
    selected_cost = total_cost(selected)
    duplicate_group_count = len(groups) - len(set(groups))
    return {
        "selected_cost": selected_cost,
        "budget_feasible": selected_cost <= budget,
        "group_feasible": duplicate_group_count == 0,
        "duplicate_group_count": duplicate_group_count,
    }


def is_feasible(
    candidates: Sequence[CandidateMemory], selected_ids: Sequence[str], budget: int
) -> bool:
    report = feasibility_report(candidates, selected_ids, budget)
    return bool(report["budget_feasible"] and report["group_feasible"])


def representation_mix(selected: Sequence[CandidateMemory]) -> Dict[str, int]:
    mix: Dict[str, int] = {}
    for candidate in selected:
        mix[candidate.representation_type] = mix.get(candidate.representation_type, 0) + 1
    return mix


def update_metrics(instance: OracleMemInstance, selected: Sequence[CandidateMemory]) -> Dict[str, float]:
    totals = coverage_totals(selected)

    def covered_mass(units: Sequence[str]) -> float:
        return sum(min(1.0, totals.get(unit_id, 0.0)) for unit_id in units)

    return {
        "current_units_total": float(len(instance.current_units)),
        "current_units_covered": covered_mass(instance.current_units),
        "invalidation_units_total": float(len(instance.invalidation_units)),
        "invalidation_units_covered": covered_mass(instance.invalidation_units),
        "stale_units_total": float(len(instance.stale_units)),
        "stale_positive_units_covered": covered_mass(instance.stale_units),
        "selected_tombstone_like": float(
            sum(1 for candidate in selected if candidate.representation_type in TOMBSTONE_TYPES)
        ),
    }


def filter_instance(
    instance: OracleMemInstance,
    *,
    allow_types: Optional[Iterable[str]] = None,
    disallow_types: Optional[Iterable[str]] = None,
    suffix: str = "filtered",
) -> OracleMemInstance:
    """Return an instance with a subset of candidates but the same objective units."""

    allowed = set(allow_types) if allow_types is not None else None
    disallowed = set(disallow_types) if disallow_types is not None else set()
    candidates = [
        candidate
        for candidate in instance.candidates
        if (allowed is None or candidate.representation_type in allowed)
        and candidate.representation_type not in disallowed
    ]
    return OracleMemInstance(
        f"{instance.instance_id}_{suffix}",
        candidates,
        instance.unit_weights,
        seed=instance.seed,
        current_units=instance.current_units,
        invalidation_units=instance.invalidation_units,
        stale_units=instance.stale_units,
    )


def candidate_quality_filter(
    candidates: Sequence[CandidateMemory],
    allowed_types: Iterable[str],
) -> List[CandidateMemory]:
    """Keep a deployable candidate-generator profile for quality ablations."""

    allowed = set(allowed_types)
    return [
        candidate
        for candidate in candidates
        if candidate.representation_type in allowed
    ]


def retrieval_decomposition(
    instance: OracleMemInstance,
    selected: Sequence[CandidateMemory],
    *,
    modes: Sequence[str] = ("oracle",),
    fixed_top_k: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Deterministic write/retrieval failure decomposition for synthetic instances.

    The local reader is evidence-only: if a required semantic unit is retrieved, it
    is considered answerable. This intentionally isolates write and retrieval
    failures before any API-based reader is introduced.
    """

    required_units = tuple(dict.fromkeys(tuple(instance.current_units) + tuple(instance.invalidation_units)))
    stale_units = tuple(instance.stale_units)
    full_totals = coverage_totals(selected)

    def covered_mass(units: Sequence[str], totals: Mapping[str, float]) -> float:
        return sum(min(1.0, totals.get(unit_id, 0.0)) for unit_id in units)

    full_required = covered_mass(required_units, full_totals)
    result: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        if mode == "oracle":
            retrieved = list(selected)
        elif mode == "fixed":
            retrieved = sorted(
                selected,
                key=lambda candidate: (-candidate.time_index, candidate.cost, candidate.candidate_id),
            )[:fixed_top_k]
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        retrieved_totals = coverage_totals(retrieved)
        retrieved_required = covered_mass(required_units, retrieved_totals)
        stale_retrieved = covered_mass(stale_units, retrieved_totals)
        result[mode] = {
            "required_units_total": float(len(required_units)),
            "write_units_covered": float(full_required),
            "retrieved_units_covered": float(retrieved_required),
            "write_failure_units": float(max(0.0, len(required_units) - full_required)),
            "retrieval_failure_units": float(max(0.0, full_required - retrieved_required)),
            "reader_failure_units": 0.0,
            "stale_units_total": float(len(stale_units)),
            "stale_units_retrieved": float(stale_retrieved),
            "retrieved_candidate_count": float(len(retrieved)),
        }
    return result


def _suffix_coverage_upper(
    groups: Sequence[Sequence[CandidateMemory]], unit_weights: Mapping[str, float]
) -> List[Dict[str, float]]:
    units = list(unit_weights)
    suffix: List[Dict[str, float]] = [{unit_id: 0.0 for unit_id in units} for _ in range(len(groups) + 1)]
    for index in range(len(groups) - 1, -1, -1):
        upper = dict(suffix[index + 1])
        for unit_id in units:
            best_group_coverage = max(
                (candidate.coverage.get(unit_id, 0.0) for candidate in groups[index]),
                default=0.0,
            )
            if best_group_coverage:
                upper[unit_id] = upper.get(unit_id, 0.0) + best_group_coverage
        suffix[index] = upper
    return suffix


def exact_solve(
    instance_or_candidates: OracleMemInstance | Sequence[CandidateMemory],
    budget: int,
    *,
    unit_weights: Optional[Mapping[str, float]] = None,
    saturation: str = "cap1",
) -> SelectionResult:
    """Exact branch-and-bound solver for small finite OracleMem instances."""

    instance = _as_instance(instance_or_candidates, unit_weights=unit_weights)
    start = time.perf_counter()
    groups = ordered_groups(instance.candidates)
    suffix = _suffix_coverage_upper(groups, instance.unit_weights)
    best_value = 0.0
    best_ids: Tuple[str, ...] = ()
    best_cost = 0

    def optimistic_value(index: int, totals: Mapping[str, float]) -> float:
        optimistic_totals = dict(totals)
        for unit_id, addend in suffix[index].items():
            optimistic_totals[unit_id] = optimistic_totals.get(unit_id, 0.0) + addend
        return value_from_totals(optimistic_totals, instance.unit_weights, saturation=saturation)

    def recurse(
        index: int,
        used_cost: int,
        selected_ids: Tuple[str, ...],
        totals: Mapping[str, float],
    ) -> None:
        nonlocal best_value, best_ids, best_cost
        if optimistic_value(index, totals) + 1e-12 < best_value:
            return
        if index == len(groups):
            value = value_from_totals(totals, instance.unit_weights, saturation=saturation)
            if (
                value > best_value + 1e-12
                or (abs(value - best_value) <= 1e-12 and used_cost < best_cost)
            ):
                best_value = value
                best_ids = selected_ids
                best_cost = used_cost
            return

        recurse(index + 1, used_cost, selected_ids, totals)
        for candidate in sorted(
            groups[index],
            key=lambda item: (
                -objective_value([item], instance.unit_weights, saturation=saturation),
                item.cost,
                item.candidate_id,
            ),
        ):
            if used_cost + candidate.cost > budget:
                continue
            next_totals = dict(totals)
            for unit_id, fidelity in candidate.coverage.items():
                next_totals[unit_id] = next_totals.get(unit_id, 0.0) + fidelity
            recurse(
                index + 1,
                used_cost + candidate.cost,
                selected_ids + (candidate.candidate_id,),
                next_totals,
            )

    recurse(0, 0, (), {})
    selected = selected_candidates(instance.candidates, best_ids)
    runtime_sec = time.perf_counter() - start
    return _make_result(
        instance,
        budget,
        "opt",
        best_ids,
        selected,
        objective_value(selected, instance.unit_weights, saturation=saturation),
        optimum_value=best_value,
        upper_bound=best_value,
        upper_bound_source="exact_opt",
        reference_value=None,
        runtime_sec=runtime_sec,
        denominator_label="exact_opt",
    )


def brute_force_solve(
    instance_or_candidates: OracleMemInstance | Sequence[CandidateMemory],
    budget: int,
    *,
    unit_weights: Optional[Mapping[str, float]] = None,
    saturation: str = "cap1",
) -> SelectionResult:
    """Exhaustive product enumeration, used as an independent correctness check."""

    instance = _as_instance(instance_or_candidates, unit_weights=unit_weights)
    start = time.perf_counter()
    groups = ordered_groups(instance.candidates)
    best_value = 0.0
    best_ids: Tuple[str, ...] = ()
    best_cost = 0
    options = [[None] + list(group) for group in groups]
    for assignment in itertools.product(*options):
        selected = [candidate for candidate in assignment if candidate is not None]
        used_cost = total_cost(selected)
        if used_cost > budget:
            continue
        value = objective_value(selected, instance.unit_weights, saturation=saturation)
        selected_ids = tuple(candidate.candidate_id for candidate in selected)
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12 and used_cost < best_cost
        ):
            best_value = value
            best_ids = selected_ids
            best_cost = used_cost
    runtime_sec = time.perf_counter() - start
    selected = selected_candidates(instance.candidates, best_ids)
    return _make_result(
        instance,
        budget,
        "brute_force",
        best_ids,
        selected,
        best_value,
        optimum_value=best_value,
        upper_bound=best_value,
        upper_bound_source="brute_force_exact",
        reference_value=None,
        runtime_sec=runtime_sec,
        denominator_label="exact_opt",
    )


def solve_exact(
    instance: OracleMemInstance,
    budget: int,
    *,
    solver: str = "exact_stdlib",
    saturation: str = "cap1",
) -> SelectionResult:
    """Dispatch exact solving to the default branch-and-bound or optional MILP backend."""

    if solver == "exact_stdlib":
        return exact_solve(instance, budget, saturation=saturation)
    if solver == "milp":
        from .solvers_milp import milp_solve

        return milp_solve(instance, budget, saturation=saturation)
    raise ValueError(f"unknown exact solver: {solver}")


def greedy_select(
    candidates: Sequence[CandidateMemory],
    budget: int,
    unit_weights: Mapping[str, float],
    *,
    allow_types: Optional[Iterable[str]] = None,
    disallow_types: Optional[Iterable[str]] = None,
    saturation: str = "cap1",
) -> Tuple[str, ...]:
    allowed = set(allow_types) if allow_types is not None else None
    disallowed = set(disallow_types) if disallow_types is not None else set()
    selected_ids: List[str] = []
    used_groups: set[str] = set()
    used_cost = 0
    totals: Dict[str, float] = {}

    while True:
        best_key: Optional[Tuple[float, float, int, str]] = None
        best_candidate: Optional[CandidateMemory] = None
        for candidate in candidates:
            if candidate.experience_id in used_groups:
                continue
            if allowed is not None and candidate.representation_type not in allowed:
                continue
            if candidate.representation_type in disallowed:
                continue
            if used_cost + candidate.cost > budget:
                continue
            marginal = marginal_value(
                candidate, totals, unit_weights, saturation=saturation
            )
            if marginal <= 1e-12:
                continue
            density = marginal / max(candidate.cost, 1e-12)
            key = (density, marginal, -candidate.cost, candidate.candidate_id)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate
        if best_candidate is None:
            break
        selected_ids.append(best_candidate.candidate_id)
        used_groups.add(best_candidate.experience_id)
        used_cost += best_candidate.cost
        for unit_id, fidelity in best_candidate.coverage.items():
            totals[unit_id] = totals.get(unit_id, 0.0) + fidelity
    return tuple(selected_ids)


def density_only_select(
    candidates: Sequence[CandidateMemory],
    budget: int,
    unit_weights: Mapping[str, float],
    *,
    saturation: str = "cap1",
) -> Tuple[str, ...]:
    selected_ids: List[str] = []
    used_cost = 0
    totals: Dict[str, float] = {}
    for group in ordered_groups(candidates):
        best_key: Optional[Tuple[float, float, int, str]] = None
        best_candidate: Optional[CandidateMemory] = None
        for candidate in group:
            if used_cost + candidate.cost > budget:
                continue
            marginal = marginal_value(candidate, totals, unit_weights, saturation=saturation)
            if marginal <= 1e-12:
                continue
            density = marginal / max(candidate.cost, 1e-12)
            key = (density, marginal, -candidate.cost, candidate.candidate_id)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate
        if best_candidate is not None:
            selected_ids.append(best_candidate.candidate_id)
            used_cost += best_candidate.cost
            for unit_id, fidelity in best_candidate.coverage.items():
                totals[unit_id] = totals.get(unit_id, 0.0) + fidelity
    return tuple(selected_ids)


def oracle_gvt_select(
    candidates: Sequence[CandidateMemory],
    budget: int,
    unit_weights: Mapping[str, float],
    *,
    saturation: str = "cap1",
) -> Tuple[str, ...]:
    """Grouped value-threshold over a finite threshold grid.

    This is the oracle-utility MVP baseline: density gates admissibility, then
    raw marginal value chooses among candidates in the same arriving group.
    """

    singleton_densities = {
        objective_value([candidate], unit_weights, saturation=saturation)
        / max(candidate.cost, 1e-12)
        for candidate in candidates
        if candidate.cost <= budget
    }
    thresholds = sorted(singleton_densities | {0.0}, reverse=True)
    best_ids: Tuple[str, ...] = ()
    best_value = -1.0
    best_cost = 0

    for threshold in thresholds:
        selected_ids: List[str] = []
        used_cost = 0
        totals: Dict[str, float] = {}
        for group in ordered_groups(candidates):
            admissible: List[Tuple[float, int, str, CandidateMemory]] = []
            for candidate in group:
                if used_cost + candidate.cost > budget:
                    continue
                marginal = marginal_value(
                    candidate, totals, unit_weights, saturation=saturation
                )
                if marginal <= 1e-12:
                    continue
                density = marginal / max(candidate.cost, 1e-12)
                if density + 1e-12 >= threshold:
                    admissible.append((marginal, -candidate.cost, candidate.candidate_id, candidate))
            if not admissible:
                continue
            _, _, _, chosen = max(admissible)
            selected_ids.append(chosen.candidate_id)
            used_cost += chosen.cost
            for unit_id, fidelity in chosen.coverage.items():
                totals[unit_id] = totals.get(unit_id, 0.0) + fidelity

        selected = selected_candidates(candidates, selected_ids)
        value = objective_value(selected, unit_weights, saturation=saturation)
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12 and total_cost(selected) < best_cost
        ):
            best_value = value
            best_cost = total_cost(selected)
            best_ids = tuple(selected_ids)
    return best_ids


def recency_raw_select(candidates: Sequence[CandidateMemory], budget: int) -> Tuple[str, ...]:
    selected_ids: List[str] = []
    used_groups: set[str] = set()
    used_cost = 0
    raw_candidates = [
        candidate
        for candidate in candidates
        if candidate.representation_type in {"raw", "raw_span"}
    ]
    for candidate in sorted(raw_candidates, key=lambda item: (-item.time_index, item.cost, item.candidate_id)):
        if candidate.experience_id in used_groups:
            continue
        if used_cost + candidate.cost > budget:
            continue
        selected_ids.append(candidate.candidate_id)
        used_groups.add(candidate.experience_id)
        used_cost += candidate.cost
    return tuple(selected_ids)


def reservoir_raw_select(candidates: Sequence[CandidateMemory], budget: int) -> Tuple[str, ...]:
    """Deterministic reservoir-style raw baseline using a stable pseudo-random order."""

    selected_ids: List[str] = []
    used_groups: set[str] = set()
    used_cost = 0
    raw_candidates = [
        candidate
        for candidate in candidates
        if candidate.representation_type in {"raw", "raw_span"}
    ]
    for candidate in sorted(raw_candidates, key=lambda item: (_stable_hash(item.candidate_id), item.cost)):
        if candidate.experience_id in used_groups:
            continue
        if used_cost + candidate.cost > budget:
            continue
        selected_ids.append(candidate.candidate_id)
        used_groups.add(candidate.experience_id)
        used_cost += candidate.cost
    return tuple(selected_ids)


def type_only_select(
    candidates: Sequence[CandidateMemory],
    budget: int,
    unit_weights: Mapping[str, float],
    representation_type: str,
    *,
    saturation: str = "cap1",
) -> Tuple[str, ...]:
    return greedy_select(
        candidates,
        budget,
        unit_weights,
        allow_types={representation_type},
        saturation=saturation,
    )


_ESTIMATED_TYPE_PRIORS: Dict[str, float] = {
    "compound_update": 3.2,
    "compound_evidence": 3.0,
    "tombstone": 2.6,
    "abstention": 2.4,
    "uncertainty": 2.3,
    "interval_fact": 2.0,
    "raw_span": 2.0,
    "raw": 2.0,
    "summary": 1.7,
    "graph_edge": 1.6,
    "skill": 1.6,
    "atomic_fact": 1.55,
    "fact": 1.55,
}

_ESTIMATED_CUE_BONUSES: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("tombstone", "invalid", "invalidated", "superseded", "no longer"), 1.10),
    (("update", "correction", "corrected", "changed", "current"), 0.70),
    (("complete", "full", "explicit", "scoped"), 0.45),
    (("abstain", "insufficient evidence", "conflict", "ambiguous", "uncertainty"), 0.65),
    (("source detail", "provenance"), 0.20),
    (("hint", "partial", "generic", "unsupported", "overconfident", "without the full"), -0.65),
)

_ESTIMATED_STOPWORDS = {
    "about",
    "after",
    "before",
    "between",
    "current",
    "detail",
    "evidence",
    "fact",
    "from",
    "that",
    "the",
    "this",
    "user",
    "with",
}


_LEARNED_BASE_FEATURES: Tuple[str, ...] = (
    "bias",
    "cost",
    "inv_cost",
    "sqrt_cost",
    "confidence",
    "time_norm",
    "signature_count",
    "cue_hit_count",
)
_LEARNED_TYPES: Tuple[str, ...] = tuple(
    sorted(set(_ESTIMATED_TYPE_PRIORS) | GENERIC_CANDIDATE_TYPES | TOMBSTONE_TYPES)
)
_LEARNED_CUE_NAMES: Tuple[str, ...] = tuple(
    f"cue:{index}" for index in range(len(_ESTIMATED_CUE_BONUSES))
)
DEFAULT_LEARNED_FEATURE_NAMES: Tuple[str, ...] = (
    *_LEARNED_BASE_FEATURES,
    *(f"type:{representation}" for representation in _LEARNED_TYPES),
    "type:other",
    *_LEARNED_CUE_NAMES,
)


def _validate_estimator_profile(estimator_profile: str) -> None:
    if estimator_profile not in ESTIMATOR_PROFILES:
        available = ", ".join(ESTIMATOR_PROFILES)
        raise ValueError(f"unknown estimator profile {estimator_profile!r}; available: {available}")


def _estimated_text(candidate: CandidateMemory) -> str:
    return f"{candidate.representation_type} {candidate.serialized}".lower()


def _estimated_signature_tokens(candidate: CandidateMemory) -> set[str]:
    text = "".join(char.lower() if char.isalnum() else " " for char in _estimated_text(candidate))
    return {
        token
        for token in text.split()
        if len(token) >= 4 and token not in _ESTIMATED_STOPWORDS
    }


def _candidate_time_norm(
    candidate: CandidateMemory,
    universe: Optional[Sequence[CandidateMemory]] = None,
) -> float:
    candidates = universe or (candidate,)
    times = [int(item.time_index) for item in candidates]
    if not times:
        return 0.5
    low = min(times)
    high = max(times)
    if high == low:
        return 0.5
    return (int(candidate.time_index) - low) / (high - low)


def _cue_hits(candidate: CandidateMemory) -> List[bool]:
    text = _estimated_text(candidate)
    return [any(cue in text for cue in cues) for cues, _ in _ESTIMATED_CUE_BONUSES]


def _stable_unit_noise(value: str) -> float:
    return 2.0 * (_stable_hash(value) / 1_000_000_007.0) - 1.0


def learned_candidate_features(
    candidate: CandidateMemory,
    universe: Optional[Sequence[CandidateMemory]] = None,
) -> Dict[str, float]:
    """Visible-only candidate features for train/dev estimated utility."""

    cost = max(float(candidate.cost), 1.0)
    representation = candidate.representation_type.strip().lower()
    signature_count = min(20, len(_estimated_signature_tokens(candidate))) / 20.0
    cue_hits = _cue_hits(candidate)
    features: Dict[str, float] = {
        "bias": 1.0,
        "cost": min(cost, 20.0) / 20.0,
        "inv_cost": 1.0 / cost,
        "sqrt_cost": math.sqrt(cost) / math.sqrt(20.0),
        "confidence": max(0.0, min(1.25, float(candidate.confidence))) / 1.25,
        "time_norm": _candidate_time_norm(candidate, universe),
        "signature_count": signature_count,
        "cue_hit_count": sum(1.0 for hit in cue_hits if hit) / max(len(cue_hits), 1),
        f"type:{representation}" if representation in _LEARNED_TYPES else "type:other": 1.0,
    }
    for index, hit in enumerate(cue_hits):
        features[f"cue:{index}"] = 1.0 if hit else 0.0
    return features


def _ridge_fit(
    feature_rows: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    feature_names: Sequence[str],
    *,
    ridge: float,
) -> Tuple[float, ...]:
    if len(feature_rows) != len(targets):
        raise ValueError("feature row and target counts differ")
    if not feature_rows:
        return tuple(0.0 for _ in feature_names)

    n_features = len(feature_names)
    index_by_name = {name: index for index, name in enumerate(feature_names)}
    normal_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    normal_rhs = [0.0 for _ in range(n_features)]
    for features, target in zip(feature_rows, targets):
        dense = [
            (index_by_name[name], float(value))
            for name, value in features.items()
            if name in index_by_name and abs(float(value)) > 1e-12
        ]
        for row_index, row_value in dense:
            normal_rhs[row_index] += row_value * float(target)
            row = normal_matrix[row_index]
            for col_index, col_value in dense:
                row[col_index] += row_value * col_value

    ridge_value = max(0.0, float(ridge))
    for index, name in enumerate(feature_names):
        normal_matrix[index][index] += ridge_value * (0.01 if name == "bias" else 1.0)
    return _solve_linear_system(normal_matrix, normal_rhs)


def _solve_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Tuple[float, ...]:
    n = len(rhs)
    augmented = [list(row) + [float(rhs[index])] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) <= 1e-12:
            continue
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for item in range(col, n + 1):
            augmented[col][item] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) <= 1e-12:
                continue
            for item in range(col, n + 1):
                augmented[row][item] -= factor * augmented[col][item]
    return tuple(
        augmented[index][n] if any(abs(value) > 1e-12 for value in augmented[index][:n]) else 0.0
        for index in range(n)
    )


def train_feature_utility_estimator(
    instances: Sequence[OracleMemInstance],
    *,
    train_distributions: Sequence[str] = ("unknown",),
    train_seeds: Sequence[int] = (),
    estimator_model: str = LOCAL_LEARNED_ESTIMATOR_MODEL,
    estimator_profile: str = LEARNED_ESTIMATOR_PROFILE,
    ridge: float = 1.0,
    noise_scale: float = 0.0,
    noise_seed: int = 0,
    saturation: str = "cap1",
) -> EstimatedUtilityModel:
    """Fit a local visible-feature estimator from train-only oracle utility."""

    _validate_estimator_profile(estimator_profile)
    if estimator_profile != LEARNED_ESTIMATOR_PROFILE:
        raise ValueError(
            f"train_feature_utility_estimator requires {LEARNED_ESTIMATOR_PROFILE!r}"
        )
    feature_rows: List[Mapping[str, float]] = []
    targets: List[float] = []
    for instance in instances:
        for candidate in instance.candidates:
            feature_rows.append(learned_candidate_features(candidate, instance.candidates))
            targets.append(objective_value([candidate], instance.unit_weights, saturation=saturation))

    weights = _ridge_fit(
        feature_rows,
        targets,
        DEFAULT_LEARNED_FEATURE_NAMES,
        ridge=ridge,
    )
    target_mean = sum(targets) / len(targets) if targets else 0.0
    return EstimatedUtilityModel(
        estimator_model=estimator_model,
        estimator_profile=estimator_profile,
        feature_names=tuple(DEFAULT_LEARNED_FEATURE_NAMES),
        weights=weights,
        ridge=float(ridge),
        noise_scale=float(noise_scale),
        noise_seed=int(noise_seed),
        train_distributions=tuple(train_distributions),
        train_seeds=tuple(int(seed) for seed in train_seeds),
        train_instance_count=len(instances),
        train_candidate_count=len(targets),
        train_target_mean=target_mean,
    )


def train_synthetic_feature_estimator(
    train_seeds: Sequence[int],
    *,
    distributions: Sequence[str] = ("base",),
    normal_count: int = 3,
    update_count: int = 2,
    estimator_model: str = LOCAL_LEARNED_ESTIMATOR_MODEL,
    ridge: float = 1.0,
    noise_scale: float = 0.0,
    noise_seed: int = 0,
    saturation: str = "cap1",
) -> EstimatedUtilityModel:
    train_instances = [
        generate_named_distribution(
            distribution,
            seed,
            normal_count=normal_count,
            update_count=update_count,
        )
        for distribution in distributions
        for seed in train_seeds
    ]
    return train_feature_utility_estimator(
        train_instances,
        train_distributions=distributions,
        train_seeds=train_seeds,
        estimator_model=estimator_model,
        estimator_profile=LEARNED_ESTIMATOR_PROFILE,
        ridge=ridge,
        noise_scale=noise_scale,
        noise_seed=noise_seed,
        saturation=saturation,
    )


def estimated_singleton_value(
    candidate: CandidateMemory,
    *,
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
    candidate_universe: Optional[Sequence[CandidateMemory]] = None,
) -> float:
    """Estimated write-time utility from external scores or visible candidate cues.

    The default profile is deterministic and local: it records the Gemini
    Flash-Lite estimator label for experiment provenance, but does not call any
    external API. If callers provide ``estimated_value`` or ``estimated_coverage``
    on candidates, those estimates take precedence over the heuristic fallback.
    """

    _validate_estimator_profile(estimator_profile)
    if candidate.estimated_value is not None:
        return float(candidate.estimated_value)
    if candidate.estimated_coverage:
        return sum(min(1.0, float(value)) for value in candidate.estimated_coverage.values())
    if estimator_profile == LEARNED_ESTIMATOR_PROFILE:
        if estimator_state is None:
            raise ValueError(
                f"{LEARNED_ESTIMATOR_PROFILE} requires a trained EstimatedUtilityModel"
            )
        return estimator_state.predict(candidate, candidate_universe or (candidate,))
    if estimator_profile == "external":
        return 0.0

    representation = candidate.representation_type.strip().lower()
    text = _estimated_text(candidate)
    value = _ESTIMATED_TYPE_PRIORS.get(representation, 1.0)
    for cues, bonus in _ESTIMATED_CUE_BONUSES:
        if any(cue in text for cue in cues):
            value += bonus
    value += min(0.55, 0.035 * len(_estimated_signature_tokens(candidate)))
    confidence = max(0.0, min(1.25, float(candidate.confidence)))
    value = max(0.0, value * (0.55 + 0.45 * confidence))
    if estimator_profile == NOISY_ESTIMATOR_PROFILE:
        # Deterministic candidate-local perturbation: this models an imperfect
        # learned scorer without calling an API or leaking oracle coverage.
        # The hash includes the model label so provenance changes alter the
        # synthetic estimator deterministically.
        noise_key = f"{estimator_model}|{candidate.candidate_id}|{candidate.serialized}"
        centered = (_stable_hash(noise_key) / 1_000_000_007.0) - 0.5
        value *= max(0.05, 1.0 + 0.40 * centered)
    return value


def estimated_marginal_value(
    candidate: CandidateMemory,
    selected_signature_tokens: set[str],
    *,
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
    candidate_universe: Optional[Sequence[CandidateMemory]] = None,
) -> float:
    value = estimated_singleton_value(
        candidate,
        estimator_model=estimator_model,
        estimator_profile=estimator_profile,
        estimator_state=estimator_state,
        candidate_universe=candidate_universe,
    )
    if value <= 0:
        return 0.0
    tokens = _estimated_signature_tokens(candidate)
    if not tokens:
        return value
    novelty = len(tokens - selected_signature_tokens) / max(len(tokens), 1)
    # The learned profile is trained to predict singleton utility from visible
    # metadata.  Keep only a mild duplicate penalty here; otherwise repeated
    # natural terms such as "preference" or "update" suppress unrelated
    # examples in larger held-out packages.
    novelty_floor = 0.85 if estimator_profile == LEARNED_ESTIMATOR_PROFILE else 0.35
    return value * (novelty_floor + (1.0 - novelty_floor) * novelty)


def estimated_utility_select(
    candidates: Sequence[CandidateMemory],
    budget: int,
    *,
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> Tuple[str, ...]:
    """Offline greedy writer using estimated marginal utility density only."""

    selected_ids: List[str] = []
    selected_tokens: set[str] = set()
    used_groups: set[str] = set()
    used_cost = 0
    while True:
        best_key: Optional[Tuple[float, float, int, str]] = None
        best_candidate: Optional[CandidateMemory] = None
        for candidate in candidates:
            if candidate.experience_id in used_groups:
                continue
            if used_cost + candidate.cost > budget:
                continue
            marginal = estimated_marginal_value(
                candidate,
                selected_tokens,
                estimator_model=estimator_model,
                estimator_profile=estimator_profile,
                estimator_state=estimator_state,
                candidate_universe=candidates,
            )
            if marginal <= 1e-12:
                continue
            density = marginal / max(candidate.cost, 1e-12)
            key = (density, marginal, -candidate.cost, candidate.candidate_id)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate
        if best_candidate is None:
            break
        selected_ids.append(best_candidate.candidate_id)
        selected_tokens.update(_estimated_signature_tokens(best_candidate))
        used_groups.add(best_candidate.experience_id)
        used_cost += best_candidate.cost
    return tuple(selected_ids)


def estimated_gvt_select(
    candidates: Sequence[CandidateMemory],
    budget: int,
    *,
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> Tuple[str, ...]:
    """Grouped value-threshold using estimated utility, scored later by oracle F."""

    singleton_densities = {
        estimated_singleton_value(
            candidate,
            estimator_model=estimator_model,
            estimator_profile=estimator_profile,
            estimator_state=estimator_state,
            candidate_universe=candidates,
        )
        / max(candidate.cost, 1e-12)
        for candidate in candidates
        if candidate.cost <= budget
    }
    thresholds = sorted({density for density in singleton_densities if density > 0} | {0.0}, reverse=True)
    best_ids: Tuple[str, ...] = ()
    best_estimated_value = -1.0
    best_cost = 0

    for threshold in thresholds:
        selected_ids: List[str] = []
        selected_tokens: set[str] = set()
        used_cost = 0
        estimated_total = 0.0
        for group in ordered_groups(candidates):
            admissible: List[Tuple[float, int, str, CandidateMemory]] = []
            for candidate in group:
                if used_cost + candidate.cost > budget:
                    continue
                marginal = estimated_marginal_value(
                    candidate,
                    selected_tokens,
                    estimator_model=estimator_model,
                    estimator_profile=estimator_profile,
                    estimator_state=estimator_state,
                    candidate_universe=candidates,
                )
                if marginal <= 1e-12:
                    continue
                density = marginal / max(candidate.cost, 1e-12)
                if density + 1e-12 >= threshold:
                    admissible.append((marginal, -candidate.cost, candidate.candidate_id, candidate))
            if not admissible:
                continue
            marginal, _, _, chosen = max(admissible)
            selected_ids.append(chosen.candidate_id)
            selected_tokens.update(_estimated_signature_tokens(chosen))
            used_cost += chosen.cost
            estimated_total += marginal
        if estimated_total > best_estimated_value + 1e-12 or (
            abs(estimated_total - best_estimated_value) <= 1e-12 and used_cost < best_cost
        ):
            best_estimated_value = estimated_total
            best_cost = used_cost
            best_ids = tuple(selected_ids)
    return best_ids


def _stable_hash(value: str) -> int:
    total = 0
    for byte in value.encode("utf-8"):
        total = (total * 131 + byte) % 1_000_000_007
    return total


def select_method(
    method: str,
    candidates: Sequence[CandidateMemory],
    budget: int,
    unit_weights: Mapping[str, float],
    *,
    exact_ids: Optional[Sequence[str]] = None,
    saturation: str = "cap1",
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> Tuple[str, ...]:
    if method in {"opt", "exact_opt"}:
        if exact_ids is None:
            raise ValueError("exact_ids must be supplied for opt method")
        return tuple(exact_ids)
    if method in WRITER_BASELINE_METHODS:
        return select_writer_baseline(method, candidates, budget)
    if method == "oracle_gvt":
        return oracle_gvt_select(candidates, budget, unit_weights, saturation=saturation)
    if method == "greedy":
        return greedy_select(candidates, budget, unit_weights, saturation=saturation)
    if method == "density_only":
        return density_only_select(candidates, budget, unit_weights, saturation=saturation)
    if method == "recency_raw":
        return recency_raw_select(candidates, budget)
    if method == "reservoir_raw":
        return reservoir_raw_select(candidates, budget)
    if method == "estimated_gvt":
        return estimated_gvt_select(
            candidates,
            budget,
            estimator_model=estimator_model,
            estimator_profile=estimator_profile,
            estimator_state=estimator_state,
        )
    if method == "estimated_utility":
        return estimated_utility_select(
            candidates,
            budget,
            estimator_model=estimator_model,
            estimator_profile=estimator_profile,
            estimator_state=estimator_state,
        )
    if method == "fact_only":
        return type_only_select(
            candidates, budget, unit_weights, "atomic_fact", saturation=saturation
        )
    if method == "summary_only":
        return type_only_select(candidates, budget, unit_weights, "summary", saturation=saturation)
    if method == "no_tombstone_gvt":
        filtered = [
            candidate
            for candidate in candidates
            if candidate.representation_type not in TOMBSTONE_TYPES
        ]
        return oracle_gvt_select(filtered, budget, unit_weights, saturation=saturation)
    if method == "no_tombstone_greedy":
        return greedy_select(
            candidates,
            budget,
            unit_weights,
            disallow_types=TOMBSTONE_TYPES,
            saturation=saturation,
        )
    if method == "generic_candidate_gvt":
        filtered = candidate_quality_filter(candidates, GENERIC_CANDIDATE_TYPES)
        return oracle_gvt_select(filtered, budget, unit_weights, saturation=saturation)
    if method in CANDIDATE_QUALITY_EXACT_METHODS:
        filtered = candidate_quality_filter(
            candidates, CANDIDATE_QUALITY_EXACT_METHODS[method]
        )
        exact = exact_solve(filtered, budget, unit_weights=unit_weights, saturation=saturation)
        return tuple(exact.selected_candidate_ids)
    raise ValueError(f"unknown method: {method}")


def policy_metadata_for_method(
    method: str,
    *,
    estimator_model: str,
    estimator_profile: str,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> Dict[str, Any]:
    if method in ESTIMATED_METHODS:
        metadata = {
            "policy_family": "estimated_utility",
            "estimator_model": estimator_model,
            "estimator_profile": estimator_profile,
            "api_called": False,
            "oracle_coverage_used_for_decision": False,
        }
        if estimator_state is not None:
            metadata.update(estimator_state.metadata())
        if estimator_profile == NOISY_ESTIMATOR_PROFILE:
            metadata["noise_profile"] = "deterministic_candidate_hash"
        return metadata
    if method in WRITER_BASELINE_METHODS:
        return {
            "policy_family": "deployable_writer_baseline",
            "external_service_dependencies": False,
            "oracle_coverage_used_for_decision": False,
            **dict(WRITER_BASELINE_DESCRIPTIONS.get(method, {})),
        }
    if method == "generic_candidate_gvt":
        return {
            "policy_family": "candidate_quality_ablation",
            "candidate_pool": "generic_raw_fact_summary",
            "selector": "oracle_gvt",
        }
    if method in CANDIDATE_QUALITY_EXACT_METHODS:
        return {
            "policy_family": "candidate_quality_ablation",
            "candidate_pool": ",".join(CANDIDATE_QUALITY_EXACT_METHODS[method]),
            "selector": "exact_opt_on_filtered_pool",
        }
    if method.startswith("no_tombstone"):
        return {
            "policy_family": "validity_ablation",
            "candidate_pool": "tombstone_and_compound_update_removed",
        }
    return {}


def evaluate_instance(
    instance: OracleMemInstance,
    budgets: Sequence[int],
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    saturation: str = "cap1",
    solver: str = "exact_stdlib",
    verify_against: Optional[str] = None,
    retrieval_modes: Sequence[str] = (),
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> List[SelectionResult]:
    """Evaluate methods across budgets with exact OPT denominator labels."""

    _validate_estimator_profile(estimator_profile)
    results: List[SelectionResult] = []
    for budget in budgets:
        exact = solve_exact(instance, budget, solver=solver, saturation=saturation)
        if verify_against is not None and verify_against != solver:
            verifier = solve_exact(instance, budget, solver=verify_against, saturation=saturation)
            if abs(verifier.objective_value - exact.objective_value) > 1e-8:
                raise AssertionError(
                    "exact solver verification failed for "
                    f"{instance.instance_id} budget={budget}: "
                    f"{solver}={exact.objective_value}, "
                    f"{verify_against}={verifier.objective_value}"
                )

        no_tombstone_exact: Optional[SelectionResult] = None
        if "no_tombstone_opt" in methods:
            no_tombstone_instance = filter_instance(
                instance, disallow_types=TOMBSTONE_TYPES, suffix="no_tombstone"
            )
            no_tombstone_exact = solve_exact(
                no_tombstone_instance, budget, solver=solver, saturation=saturation
            )
            if verify_against is not None and verify_against != solver:
                verifier = solve_exact(
                    no_tombstone_instance,
                    budget,
                    solver=verify_against,
                    saturation=saturation,
                )
                if abs(verifier.objective_value - no_tombstone_exact.objective_value) > 1e-8:
                    raise AssertionError(
                        "no-tombstone exact solver verification failed for "
                        f"{instance.instance_id} budget={budget}: "
                        f"{solver}={no_tombstone_exact.objective_value}, "
                        f"{verify_against}={verifier.objective_value}"
                    )

        candidate_quality_exact: Dict[str, SelectionResult] = {}
        for method, allowed_types in CANDIDATE_QUALITY_EXACT_METHODS.items():
            if method not in methods:
                continue
            filtered_instance = filter_instance(
                instance,
                allow_types=allowed_types,
                suffix=method,
            )
            candidate_quality_exact[method] = solve_exact(
                filtered_instance, budget, solver=solver, saturation=saturation
            )
            if verify_against is not None and verify_against != solver:
                verifier = solve_exact(
                    filtered_instance,
                    budget,
                    solver=verify_against,
                    saturation=saturation,
                )
                if abs(verifier.objective_value - candidate_quality_exact[method].objective_value) > 1e-8:
                    raise AssertionError(
                        "candidate-quality exact solver verification failed for "
                        f"{instance.instance_id} method={method} budget={budget}: "
                        f"{solver}={candidate_quality_exact[method].objective_value}, "
                        f"{verify_against}={verifier.objective_value}"
                    )

        reference_ids = greedy_select(
            instance.candidates, budget, instance.unit_weights, saturation=saturation
        )
        reference_value = objective_value(
            selected_candidates(instance.candidates, reference_ids),
            instance.unit_weights,
            saturation=saturation,
        )
        for method in methods:
            start = time.perf_counter()
            if method == "no_tombstone_opt":
                if no_tombstone_exact is None:
                    raise RuntimeError("no_tombstone_exact was not computed")
                selected_ids = tuple(no_tombstone_exact.selected_candidate_ids)
                runtime_sec = no_tombstone_exact.runtime_sec
            elif method in candidate_quality_exact:
                selected_ids = tuple(candidate_quality_exact[method].selected_candidate_ids)
                runtime_sec = candidate_quality_exact[method].runtime_sec
            else:
                selected_ids = select_method(
                    method,
                    instance.candidates,
                    budget,
                    instance.unit_weights,
                    exact_ids=exact.selected_candidate_ids,
                    saturation=saturation,
                    estimator_model=estimator_model,
                    estimator_profile=estimator_profile,
                    estimator_state=estimator_state,
                )
                runtime_sec = (
                    exact.runtime_sec
                    if method in {"opt", "exact_opt"}
                    else time.perf_counter() - start
                )
            selected = selected_candidates(instance.candidates, selected_ids)
            value = objective_value(selected, instance.unit_weights, saturation=saturation)
            results.append(
                _make_result(
                    instance,
                    budget,
                    method,
                    tuple(selected_ids),
                    selected,
                    value,
                    optimum_value=exact.objective_value,
                    upper_bound=exact.objective_value,
                    upper_bound_source="exact_opt",
                    reference_value=reference_value,
                    runtime_sec=runtime_sec,
                    denominator_label="exact_opt",
                    retrieval_modes=retrieval_modes,
                    policy_metadata=policy_metadata_for_method(
                        method,
                        estimator_model=estimator_model,
                        estimator_profile=estimator_profile,
                        estimator_state=estimator_state,
                    ),
                )
            )
    return results


def run_synthetic_benchmark(
    seeds: Sequence[int],
    budgets: Sequence[int],
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    distributions: Sequence[str] = ("base",),
    normal_count: int = 3,
    update_count: int = 2,
    saturation: str = "cap1",
    solver: str = "exact_stdlib",
    verify_against: Optional[str] = None,
    retrieval_modes: Sequence[str] = (),
    estimator_model: str = DEFAULT_ESTIMATOR_MODEL,
    estimator_profile: str = DEFAULT_ESTIMATOR_PROFILE,
    estimator_state: Optional[EstimatedUtilityModel] = None,
) -> List[SelectionResult]:
    rows: List[SelectionResult] = []
    for distribution in distributions:
        for seed in seeds:
            instance = generate_named_distribution(
                distribution,
                seed,
                normal_count=normal_count,
                update_count=update_count,
            )
            rows.extend(
                evaluate_instance(
                    instance,
                    budgets,
                    methods=methods,
                    saturation=saturation,
                    solver=solver,
                    verify_against=verify_against,
                    retrieval_modes=retrieval_modes,
                    estimator_model=estimator_model,
                    estimator_profile=estimator_profile,
                    estimator_state=estimator_state,
                )
            )
    return rows


def run_synthetic_train_dev_benchmark(
    train_seeds: Sequence[int],
    dev_seeds: Sequence[int],
    budgets: Sequence[int],
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    distributions: Sequence[str] = ("base",),
    normal_count: int = 3,
    update_count: int = 2,
    saturation: str = "cap1",
    solver: str = "exact_stdlib",
    verify_against: Optional[str] = None,
    retrieval_modes: Sequence[str] = (),
    estimator_model: str = LOCAL_LEARNED_ESTIMATOR_MODEL,
    estimator_ridge: float = 1.0,
    estimator_noise_scale: float = 0.0,
    estimator_noise_seed: int = 0,
) -> List[SelectionResult]:
    """Train a visible-feature estimator on train seeds and evaluate dev seeds."""

    if not train_seeds:
        raise ValueError("train_seeds must be nonempty")
    if not dev_seeds:
        raise ValueError("dev_seeds must be nonempty")
    estimator_state = train_synthetic_feature_estimator(
        train_seeds,
        distributions=distributions,
        normal_count=normal_count,
        update_count=update_count,
        estimator_model=estimator_model,
        ridge=estimator_ridge,
        noise_scale=estimator_noise_scale,
        noise_seed=estimator_noise_seed,
        saturation=saturation,
    )
    return run_synthetic_benchmark(
        dev_seeds,
        budgets,
        methods=methods,
        distributions=distributions,
        normal_count=normal_count,
        update_count=update_count,
        saturation=saturation,
        solver=solver,
        verify_against=verify_against,
        retrieval_modes=retrieval_modes,
        estimator_model=estimator_state.estimator_model,
        estimator_profile=estimator_state.estimator_profile,
        estimator_state=estimator_state,
    )


def generate_named_distribution(
    distribution: str,
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Generate a named exact-small distribution, defaulting to the base MVP."""

    if distribution == "base":
        return generate_synthetic_instance(
            seed, normal_count=normal_count, update_count=update_count
        )
    try:
        from .distributions import generate_distribution
        from .distributions import DISTRIBUTIONS
    except ImportError as exc:
        raise ValueError(
            f"unknown distribution {distribution!r}; optional distribution module is unavailable"
        ) from exc
    if distribution in DISTRIBUTIONS:
        return generate_distribution(
            distribution,
            seed,
            normal_count=normal_count,
            update_count=update_count,
        )
    try:
        from .distributions_v2 import generate_distribution_v2
    except ImportError as exc:
        raise ValueError(f"unknown distribution {distribution!r}") from exc
    return generate_distribution_v2(
        distribution,
        seed,
        normal_count=normal_count,
        update_count=update_count,
    )


def generate_synthetic_instance(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Deterministic exact-small benchmark with facts, updates, and tombstones."""

    rng = random.Random(seed)
    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    current_units: List[str] = []
    invalidation_units: List[str] = []
    stale_units: List[str] = []
    time_index = 0

    def add_candidate(
        exp: str,
        rep: str,
        cost: int,
        coverage: Mapping[str, float],
        text: str,
    ) -> None:
        nonlocal time_index
        candidates.append(
            CandidateMemory(
                candidate_id=f"{exp}_{rep}",
                experience_id=exp,
                representation_type=rep,
                serialized=text,
                cost=cost,
                coverage=coverage,
                time_index=time_index,
            )
        )

    for index in range(normal_count):
        unit = f"fact:{seed}:{index}"
        context = f"context:{seed}:{index // 2}"
        unit_weights.setdefault(unit, rng.choice([1.0, 1.2, 1.4]))
        unit_weights.setdefault(context, 0.35)
        exp = f"s{seed}_fact_{index}"
        add_candidate(exp, "raw_span", 5 + rng.randint(0, 1), {unit: 1.0, context: 0.4}, f"raw {unit}")
        add_candidate(exp, "atomic_fact", 2, {unit: 1.0}, f"FACT {unit}")
        add_candidate(exp, "summary", 3, {unit: 0.7, context: 0.5}, f"summary {unit}")
        time_index += 1

    for index in range(update_count):
        stale = f"stale:pref:{seed}:{index}:old"
        current = f"current:pref:{seed}:{index}:new"
        invalid = f"invalid:pref:{seed}:{index}:old_after_update"
        stale_units.append(stale)
        current_units.append(current)
        invalidation_units.append(invalid)
        unit_weights[stale] = 0.15
        unit_weights[current] = 2.0 + 0.25 * rng.randint(0, 2)
        unit_weights[invalid] = 2.0 + 0.25 * rng.randint(0, 2)

        old_exp = f"s{seed}_pref_{index}_old"
        add_candidate(old_exp, "raw_span", 4, {stale: 1.0}, f"raw old preference {index}")
        add_candidate(old_exp, "atomic_fact", 2, {stale: 1.0}, f"FACT old preference {index}")
        add_candidate(old_exp, "summary", 3, {stale: 0.75}, f"summary old preference {index}")
        time_index += 1

        update_exp = f"s{seed}_pref_{index}_update"
        add_candidate(
            update_exp,
            "raw_span",
            6,
            {current: 1.0, invalid: 0.45, stale: 0.2},
            f"raw correction {index}",
        )
        add_candidate(
            update_exp,
            "atomic_fact",
            2,
            {current: 1.0},
            f"FACT current preference {index}",
        )
        add_candidate(
            update_exp,
            "tombstone",
            1,
            {invalid: 1.0},
            f"TOMBSTONE old preference {index}",
        )
        add_candidate(
            update_exp,
            "compound_update",
            3,
            {current: 1.0, invalid: 1.0},
            f"UPDATE old to current preference {index}",
        )
        add_candidate(
            update_exp,
            "summary",
            4,
            {current: 0.75, invalid: 0.75, stale: 0.1},
            f"summary correction {index}",
        )
        time_index += 1

    return OracleMemInstance(
        instance_id=f"synthetic_seed_{seed}",
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=current_units,
        invalidation_units=invalidation_units,
        stale_units=stale_units,
    )


def make_update_stress_instance() -> OracleMemInstance:
    """Small fixture where tombstone-aware representation choice strictly helps."""

    candidates = [
        CandidateMemory(
            "old_atomic",
            "old_pref",
            "atomic_fact",
            "FACT old vegetarian preference",
            2,
            {"stale:meal:vegetarian": 1.0},
            time_index=0,
        ),
        CandidateMemory(
            "old_raw",
            "old_pref",
            "raw_span",
            "User used to prefer vegetarian meals.",
            3,
            {"stale:meal:vegetarian": 1.0},
            time_index=0,
        ),
        CandidateMemory(
            "new_atomic",
            "meal_update",
            "atomic_fact",
            "FACT current pescatarian preference",
            2,
            {"current:meal:pescatarian": 1.0},
            time_index=1,
        ),
        CandidateMemory(
            "new_tombstone",
            "meal_update",
            "tombstone",
            "TOMBSTONE vegetarian no longer current",
            1,
            {"invalid:meal:vegetarian_after_update": 1.0},
            time_index=1,
        ),
        CandidateMemory(
            "new_compound",
            "meal_update",
            "compound_update",
            "UPDATE vegetarian -> pescatarian",
            3,
            {
                "current:meal:pescatarian": 1.0,
                "invalid:meal:vegetarian_after_update": 1.0,
            },
            time_index=1,
        ),
    ]
    return OracleMemInstance(
        "update_stress",
        candidates,
        {
            "current:meal:pescatarian": 2.0,
            "invalid:meal:vegetarian_after_update": 2.0,
            "stale:meal:vegetarian": 0.1,
        },
        current_units=("current:meal:pescatarian",),
        invalidation_units=("invalid:meal:vegetarian_after_update",),
        stale_units=("stale:meal:vegetarian",),
    )


def aggregate_results(results: Sequence[SelectionResult | Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [_row_to_dict(row) for row in results]
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("distribution", _distribution_from_instance_id(str(row["instance_id"])))),
             int(row["budget"]),
             str(row["method"])),
            [],
        ).append(row)

    aggregates: List[Dict[str, Any]] = []
    for (distribution, budget, method), group_rows in sorted(grouped.items()):
        ratio_values = [
            row["ratio_to_opt"] for row in group_rows if row["ratio_to_opt"] is not None
        ]
        ratio_ci_low, ratio_ci_high = _bootstrap_mean_ci(ratio_values)
        aggregates.append(
            {
                "distribution": distribution,
                "budget": budget,
                "method": method,
                "n": len(group_rows),
                "mean_objective": _mean(row["objective_value"] for row in group_rows),
                "mean_ratio_to_opt": _mean(
                    ratio_values
                ),
                "bootstrap95_ratio_to_opt_low": ratio_ci_low,
                "bootstrap95_ratio_to_opt_high": ratio_ci_high,
                "mean_ratio_to_upper_bound": _mean(
                    row["ratio_to_upper_bound"]
                    for row in group_rows
                    if row["ratio_to_upper_bound"] is not None
                ),
                "mean_ratio_to_reference": _mean(
                    row["ratio_to_reference"]
                    for row in group_rows
                    if row["ratio_to_reference"] is not None
                ),
                "mean_selected_cost": _mean(row["selected_cost"] for row in group_rows),
                "mean_invalidation_covered": _mean(
                    row["update_metrics"].get("invalidation_units_covered", 0.0)
                    for row in group_rows
                ),
                "retrieval_summary": _aggregate_retrieval_metrics(group_rows),
                "all_budget_feasible": all(row["budget_feasible"] for row in group_rows),
                "all_group_feasible": all(row["group_feasible"] for row in group_rows),
            }
        )

    best_by_budget: List[Dict[str, Any]] = []
    for distribution in sorted({row["distribution"] for row in aggregates}):
        for budget in sorted({row["budget"] for row in aggregates if row["distribution"] == distribution}):
            candidates = [
                row
                for row in aggregates
                if row["distribution"] == distribution and row["budget"] == budget
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: (row["mean_ratio_to_opt"], row["mean_objective"]))
            best_by_budget.append(
                {
                    "distribution": distribution,
                    "budget": budget,
                    "best_method_by_mean_ratio_to_opt": best["method"],
                    "mean_ratio_to_opt": best["mean_ratio_to_opt"],
                }
            )

    return {
        "schema_version": 1,
        "label_definitions": {
            "ratio_to_opt": "F(method_store) / F(exact_opt_store); emitted only when exact optimum is certified.",
            "ratio_to_upper_bound": "F(method_store) / certified_upper_bound; exact-small uses exact OPT as the upper bound.",
            "ratio_to_reference": "F(method_store) / F(greedy_reference_store); never labeled as OPT.",
            "denominator_label": "Source of the primary oracle denominator for ratio_to_opt.",
            "policy_metadata": "Rows record estimated-policy, local proxy writer, validity-ablation, or candidate-quality-ablation provenance; train/dev estimated rows mark train-time oracle labels separately from dev-time visible-feature decisions and use no external services.",
            "retrieval_summary": "Aggregated deterministic retrieval/write decomposition, emitted when --enable-retrieval is used.",
        },
        "distributions": sorted({row["distribution"] for row in aggregates}),
        "budgets": sorted({row["budget"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "writer_baseline_descriptions": {
            method: dict(description)
            for method, description in WRITER_BASELINE_DESCRIPTIONS.items()
            if method in {row["method"] for row in rows}
        },
        "num_rows": len(rows),
        "by_distribution_budget_method": aggregates,
        "by_budget_method": aggregates,
        "best_by_budget": best_by_budget,
    }


def _distribution_from_instance_id(instance_id: str) -> str:
    if instance_id.startswith("synthetic_seed_"):
        return "base"
    if "_seed_" in instance_id:
        return instance_id.split("_seed_", 1)[0]
    prefix, sep, suffix = instance_id.rpartition("_s")
    if sep and suffix.isdigit():
        return prefix
    return "unknown"


def _aggregate_retrieval_metrics(group_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    modes = sorted(
        {
            mode
            for row in group_rows
            for mode in dict(row.get("retrieval_metrics", {})).keys()
        }
    )
    summary: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        mode_rows = [
            dict(row.get("retrieval_metrics", {})).get(mode, {})
            for row in group_rows
            if mode in dict(row.get("retrieval_metrics", {}))
        ]
        if not mode_rows:
            continue
        summary[mode] = {
            "mean_required_units_total": _mean(row.get("required_units_total") for row in mode_rows),
            "mean_write_units_covered": _mean(row.get("write_units_covered") for row in mode_rows),
            "mean_retrieved_units_covered": _mean(row.get("retrieved_units_covered") for row in mode_rows),
            "mean_write_failure_units": _mean(row.get("write_failure_units") for row in mode_rows),
            "mean_retrieval_failure_units": _mean(row.get("retrieval_failure_units") for row in mode_rows),
            "mean_reader_failure_units": _mean(row.get("reader_failure_units") for row in mode_rows),
            "mean_stale_units_retrieved": _mean(row.get("stale_units_retrieved") for row in mode_rows),
            "mean_retrieved_candidate_count": _mean(row.get("retrieved_candidate_count") for row in mode_rows),
        }
    return summary


def _legacy_aggregate_results(results: Sequence[SelectionResult | Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [_row_to_dict(row) for row in results]
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["budget"]), str(row["method"])), []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for (budget, method), group_rows in sorted(grouped.items()):
        ratio_values = [
            row["ratio_to_opt"] for row in group_rows if row["ratio_to_opt"] is not None
        ]
        ratio_ci_low, ratio_ci_high = _bootstrap_mean_ci(ratio_values)
        aggregates.append(
            {
                "budget": budget,
                "method": method,
                "n": len(group_rows),
                "mean_objective": _mean(row["objective_value"] for row in group_rows),
                "mean_ratio_to_opt": _mean(
                    ratio_values
                ),
                "bootstrap95_ratio_to_opt_low": ratio_ci_low,
                "bootstrap95_ratio_to_opt_high": ratio_ci_high,
                "mean_ratio_to_upper_bound": _mean(
                    row["ratio_to_upper_bound"]
                    for row in group_rows
                    if row["ratio_to_upper_bound"] is not None
                ),
                "mean_ratio_to_reference": _mean(
                    row["ratio_to_reference"]
                    for row in group_rows
                    if row["ratio_to_reference"] is not None
                ),
                "mean_selected_cost": _mean(row["selected_cost"] for row in group_rows),
                "mean_invalidation_covered": _mean(
                    row["update_metrics"].get("invalidation_units_covered", 0.0)
                    for row in group_rows
                ),
                "all_budget_feasible": all(row["budget_feasible"] for row in group_rows),
                "all_group_feasible": all(row["group_feasible"] for row in group_rows),
            }
        )

    best_by_budget: List[Dict[str, Any]] = []
    for budget in sorted({row["budget"] for row in rows}):
        candidates = [row for row in aggregates if row["budget"] == budget]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: (row["mean_ratio_to_opt"], row["mean_objective"]))
        best_by_budget.append(
            {
                "budget": budget,
                "best_method_by_mean_ratio_to_opt": best["method"],
                "mean_ratio_to_opt": best["mean_ratio_to_opt"],
            }
        )

    return {
        "schema_version": 1,
        "label_definitions": {
            "ratio_to_opt": "F(method_store) / F(exact_opt_store); emitted only when exact optimum is certified.",
            "ratio_to_upper_bound": "F(method_store) / certified_upper_bound; exact-small uses exact OPT as the upper bound.",
            "ratio_to_reference": "F(method_store) / F(greedy_reference_store); never labeled as OPT.",
            "denominator_label": "Source of the primary oracle denominator for ratio_to_opt.",
            "policy_metadata": "Rows record estimated-policy, local proxy writer, validity-ablation, or candidate-quality-ablation provenance; train/dev estimated rows mark train-time oracle labels separately from dev-time visible-feature decisions and use no external services.",
        },
        "budgets": sorted({row["budget"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "writer_baseline_descriptions": {
            method: dict(description)
            for method, description in WRITER_BASELINE_DESCRIPTIONS.items()
            if method in {row["method"] for row in rows}
        },
        "num_rows": len(rows),
        "by_budget_method": aggregates,
        "best_by_budget": best_by_budget,
    }


def render_markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# OracleMem MVP Summary",
        "",
        "Exact-small synthetic benchmark with sparse semantic coverage, one budget, and one representation per experience.",
        "",
        "## Ratio Labels",
        "",
    ]
    for label, definition in summary["label_definitions"].items():
        lines.append(f"- `{label}`: {definition}")
    baseline_descriptions = summary.get("writer_baseline_descriptions", {})
    if baseline_descriptions:
        lines.extend(["", "## Local Proxy Writer Baselines", ""])
        for method, description in sorted(dict(baseline_descriptions).items()):
            proxy_for = dict(description).get("proxy_for", "unspecified system family")
            limitation = dict(description).get("limitation", "local proxy only")
            lines.append(f"- `{method}`: proxy for {proxy_for}. {limitation}")
    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Distribution | Budget | Method | N | Mean Objective | Mean Ratio to OPT | Mean Cost | Mean Invalidation Covered | Feasible |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["by_budget_method"]:
        feasible = "yes" if row["all_budget_feasible"] and row["all_group_feasible"] else "no"
        ratio_with_ci = "{ratio:.4f} [{lo:.4f}, {hi:.4f}]".format(
            ratio=row["mean_ratio_to_opt"],
            lo=row["bootstrap95_ratio_to_opt_low"],
            hi=row["bootstrap95_ratio_to_opt_high"],
        )
        lines.append(
            "| `{distribution}` | {budget} | `{method}` | {n} | {obj:.4f} | {ratio} | {cost:.2f} | {invalid:.2f} | {feasible} |".format(
                distribution=row.get("distribution", "base"),
                budget=row["budget"],
                method=row["method"],
                n=row["n"],
                obj=row["mean_objective"],
                ratio=ratio_with_ci,
                cost=row["mean_selected_cost"],
                invalid=row["mean_invalidation_covered"],
                feasible=feasible,
            )
        )
    lines.extend(["", "## Best Method by Budget", ""])
    for row in summary["best_by_budget"]:
        lines.append(
            "- `{distribution}`, budget {budget}: `{method}` with mean `ratio_to_opt={ratio:.4f}`.".format(
                distribution=row.get("distribution", "base"),
                budget=row["budget"],
                method=row["best_method_by_mean_ratio_to_opt"],
                ratio=row["mean_ratio_to_opt"],
            )
        )
    lines.append("")
    retrieval_rows = [
        row
        for row in summary["by_budget_method"]
        if row.get("retrieval_summary")
    ]
    if retrieval_rows:
        lines.extend(
            [
                "## Deterministic Decomposition",
                "",
                "| Distribution | Budget | Method | Mode | Write Fail | Retrieval Fail | Reader Fail | Stale Retrieved | Retrieved Units |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in retrieval_rows:
            for mode, metrics in sorted(row["retrieval_summary"].items()):
                lines.append(
                    "| `{distribution}` | {budget} | `{method}` | `{mode}` | {write_fail:.2f} | {retr_fail:.2f} | {reader_fail:.2f} | {stale:.2f} | {retrieved:.2f} |".format(
                        distribution=row.get("distribution", "base"),
                        budget=row["budget"],
                        method=row["method"],
                        mode=mode,
                        write_fail=metrics.get("mean_write_failure_units", 0.0),
                        retr_fail=metrics.get("mean_retrieval_failure_units", 0.0),
                        reader_fail=metrics.get("mean_reader_failure_units", 0.0),
                        stale=metrics.get("mean_stale_units_retrieved", 0.0),
                        retrieved=metrics.get("mean_retrieved_units_covered", 0.0),
                    )
                )
        lines.append("")
    return "\n".join(lines)


def write_benchmark_outputs(
    results: Sequence[SelectionResult],
    out_dir: str | Path,
    *,
    raw_jsonl_name: str = "raw_results.jsonl",
    summary_json_name: str = "summary.json",
    summary_md_name: str = "summary.md",
) -> Dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_path = out_path / raw_jsonl_name
    summary_json_path = out_path / summary_json_name
    summary_md_path = out_path / summary_md_name

    with raw_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row.to_json(), sort_keys=True) + "\n")

    summary = aggregate_results(results)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown_summary(summary))

    return {
        "raw_jsonl": str(raw_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
    }


def write_coverage_package(
    instance: OracleMemInstance,
    out_dir: str | Path,
    *,
    include_zero_coverage: bool = False,
) -> Dict[str, str]:
    """Export a machine-checkable OracleMem coverage package for one instance.

    Result JSONL summaries are intentionally compact and do not expose the
    hidden finite-instance denominator. This package is the inspectable form
    needed by reviewers or benchmark adopters: evidence units, query
    requirements, candidate memories, candidate-unit coverage rows, and a
    manifest tying them to the objective.
    """

    from .coverage_export import export_coverage_package

    return export_coverage_package(
        instance,
        out_dir,
        include_zero_coverage=include_zero_coverage,
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _infer_unit_kind(unit_id: str, instance: OracleMemInstance) -> str:
    if unit_id in set(instance.current_units):
        return "current_fact"
    if unit_id in set(instance.invalidation_units):
        return "invalidation"
    if unit_id in set(instance.stale_units):
        return "stale_fact"
    if unit_id.startswith("context:"):
        return "context"
    if unit_id.startswith("fact:"):
        return "fact"
    if "interval" in unit_id:
        return "temporal_interval"
    return unit_id.split(":", 1)[0] if ":" in unit_id else "evidence"


def _infer_unit_state(unit_id: str, instance: OracleMemInstance) -> str:
    if unit_id in set(instance.current_units):
        return "current"
    if unit_id in set(instance.invalidation_units):
        return "invalidates_stale"
    if unit_id in set(instance.stale_units):
        return "stale"
    return "unknown"


def _as_instance(
    instance_or_candidates: OracleMemInstance | Sequence[CandidateMemory],
    *,
    unit_weights: Optional[Mapping[str, float]] = None,
) -> OracleMemInstance:
    if isinstance(instance_or_candidates, OracleMemInstance):
        return instance_or_candidates
    candidates = [coerce_candidate(candidate) for candidate in instance_or_candidates]
    return make_instance_from_candidates(
        candidates, unit_weights=unit_weights, instance_id="ad_hoc"
    )


def _safe_ratio(numerator: float, denominator: Optional[float]) -> Optional[float]:
    if denominator is None:
        return None
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(numerator) <= 1e-12 else None
    return numerator / denominator


def _make_result(
    instance: OracleMemInstance,
    budget: int,
    method: str,
    selected_ids: Sequence[str],
    selected: Sequence[CandidateMemory],
    value: float,
    *,
    optimum_value: Optional[float],
    upper_bound: Optional[float],
    upper_bound_source: Optional[str],
    reference_value: Optional[float],
    runtime_sec: float,
    denominator_label: str,
    retrieval_modes: Sequence[str] = (),
    policy_metadata: Optional[Mapping[str, Any]] = None,
) -> SelectionResult:
    feasibility = feasibility_report(instance.candidates, selected_ids, budget)
    return SelectionResult(
        instance_id=instance.instance_id,
        seed=instance.seed,
        distribution=_distribution_from_instance_id(instance.instance_id),
        budget=budget,
        method=method,
        selected_candidate_ids=tuple(selected_ids),
        selected_cost=int(feasibility["selected_cost"]),
        objective_value=value,
        denominator_label=denominator_label,
        ratio_to_opt=_safe_ratio(value, optimum_value),
        ratio_to_upper_bound=_safe_ratio(value, upper_bound),
        ratio_to_reference=_safe_ratio(value, reference_value),
        optimum_value=optimum_value,
        upper_bound=upper_bound,
        upper_bound_source=upper_bound_source,
        reference_value=reference_value,
        runtime_sec=runtime_sec,
        budget_feasible=bool(feasibility["budget_feasible"]),
        group_feasible=bool(feasibility["group_feasible"]),
        representation_mix=representation_mix(selected),
        update_metrics=update_metrics(instance, selected),
        retrieval_metrics=(
            retrieval_decomposition(instance, selected, modes=retrieval_modes)
            if retrieval_modes
            else {}
        ),
        policy_metadata=dict(policy_metadata or {}),
    )


def _row_to_dict(row: SelectionResult | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(row, SelectionResult):
        return row.to_json()
    return dict(row)


def _mean(values: Iterable[Optional[float]]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _bootstrap_mean_ci(
    values: Iterable[Optional[float]],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0, 0.0
    if len(clean) == 1:
        return clean[0], clean[0]
    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(samples):
        means.append(sum(clean[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_index = max(0, int(0.025 * samples) - 1)
    high_index = min(samples - 1, int(0.975 * samples))
    return means[low_index], means[high_index]
