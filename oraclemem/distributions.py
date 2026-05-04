"""Harder deterministic exact-small OracleMem distributions.

Each generator returns an ``OracleMemInstance`` built from the evaluation-layer
dataclasses.  The cases are intentionally small and hand-shaped so exact
branch-and-bound remains practical while still exposing different failure modes
for heuristic memory writers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping
import random

from .evaluate import CandidateMemory, OracleMemInstance, generate_synthetic_instance


DistributionGenerator = Callable[..., OracleMemInstance]

MAX_NORMAL_COUNT = 6
MAX_UPDATE_COUNT = 4


def _bounded_count(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))


def _rng(seed: int, salt: int) -> random.Random:
    return random.Random((int(seed) + 1) * 1_000_003 + salt)


def _candidate(
    prefix: str,
    exp: str,
    variant: str,
    representation_type: str,
    cost: int,
    coverage: Mapping[str, float],
    time_index: int,
    serialized: str,
    *,
    confidence: float = 1.0,
) -> CandidateMemory:
    return CandidateMemory(
        candidate_id=f"{prefix}:{exp}:{variant}",
        experience_id=f"{prefix}:{exp}",
        representation_type=representation_type,
        serialized=serialized,
        cost=cost,
        coverage=coverage,
        time_index=time_index,
        generator="oraclemem.distributions",
        confidence=confidence,
    )


def base(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Wrapper around the existing synthetic OracleMem generator."""

    return generate_synthetic_instance(
        seed,
        normal_count=normal_count,
        update_count=update_count,
    )


def density_trap(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Cheap overlapping memories lure density-based writers away from uniques."""

    rng = _rng(seed, 11)
    size = _bounded_count(normal_count + update_count, minimum=4, maximum=6)
    prefix = f"density_trap_s{seed}"
    common = f"{prefix}:shared_lure"
    unit_weights: Dict[str, float] = {common: 2.2}
    candidates: List[CandidateMemory] = []

    for index in range(size):
        unit = f"{prefix}:specific:{index}"
        context = f"{prefix}:context:{index // 2}"
        unit_weights[unit] = 1.45 + 0.15 * rng.randrange(3)
        unit_weights.setdefault(context, 0.3)
        exp = f"memory_{index}"
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "cheap_lure",
                    "atomic_fact",
                    1,
                    {common: 0.55, unit: 0.05},
                    index,
                    f"cheap salient overlap for item {index}",
                    confidence=0.82,
                ),
                _candidate(
                    prefix,
                    exp,
                    "specific_fact",
                    "atomic_fact",
                    2,
                    {unit: 1.0},
                    index,
                    f"precise specific fact {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "context_summary",
                    "summary",
                    3,
                    {unit: 0.7, common: 0.25, context: 0.35},
                    index,
                    f"summary with context for item {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "raw_detail",
                    "raw_span",
                    4,
                    {unit: 1.0, context: 0.6, common: 0.15},
                    index,
                    f"raw detailed evidence for item {index}",
                ),
            ]
        )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
    )


def update_chain(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Sequential corrections reward compact current+invalidation memories."""

    chain_len = _bounded_count(update_count + 1, minimum=2, maximum=4)
    prefix = f"update_chain_s{seed}"
    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    current_units: List[str] = []
    invalidation_units: List[str] = []
    stale_units: List[str] = []

    original = f"{prefix}:version:0"
    stale_units.append(original)
    unit_weights[original] = 0.1
    candidates.extend(
        [
            _candidate(
                prefix,
                "version_0",
                "raw_old",
                "raw_span",
                4,
                {original: 1.0},
                0,
                "raw original preference before any correction",
            ),
            _candidate(
                prefix,
                "version_0",
                "fact_old",
                "atomic_fact",
                2,
                {original: 1.0},
                0,
                "FACT original preference before any correction",
            ),
            _candidate(
                prefix,
                "version_0",
                "summary_old",
                "summary",
                3,
                {original: 0.7},
                0,
                "summary of original preference",
            ),
        ]
    )

    for step in range(1, chain_len + 1):
        previous = f"{prefix}:version:{step - 1}"
        current = f"{prefix}:version:{step}"
        invalid = f"{prefix}:invalidates:version:{step - 1}"
        stale_units.append(previous)
        invalidation_units.append(invalid)
        unit_weights[previous] = 0.1
        unit_weights[current] = 2.3 if step == chain_len else 0.25
        unit_weights[invalid] = 1.85
        exp = f"update_{step}"
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "raw_correction",
                    "raw_span",
                    5,
                    {current: 1.0, invalid: 0.35, previous: 0.15},
                    step,
                    f"raw correction from version {step - 1} to {step}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "current_only",
                    "atomic_fact",
                    2,
                    {current: 1.0},
                    step,
                    f"FACT current version {step}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "invalidate_only",
                    "tombstone",
                    1,
                    {invalid: 1.0},
                    step,
                    f"TOMBSTONE version {step - 1}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "compound",
                    "compound_update",
                    3,
                    {current: 0.95, invalid: 1.0},
                    step,
                    f"UPDATE version {step - 1} to {step}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "summary_update",
                    "summary",
                    3,
                    {current: 0.65, invalid: 0.65},
                    step,
                    f"summary correction to version {step}",
                ),
            ]
        )

    current_units.append(f"{prefix}:version:{chain_len}")
    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=current_units,
        invalidation_units=invalidation_units,
        stale_units=tuple(dict.fromkeys(stale_units)),
    )


def scope_shift(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Scoped preferences punish generic memories that blur contexts."""

    rng = _rng(seed, 23)
    count = _bounded_count(normal_count + 1, minimum=3, maximum=5)
    prefix = f"scope_shift_s{seed}"
    scopes = ("home", "work", "travel", "medical", "finance")
    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    scoped_units: List[str] = []

    for index, scope in enumerate(scopes[:count]):
        unit = f"{prefix}:pref:{scope}"
        context = f"{prefix}:context:{scope}"
        generic = f"{prefix}:pref:generic"
        scoped_units.append(unit)
        unit_weights[unit] = 1.6 + 0.2 * rng.randrange(3)
        unit_weights[context] = 0.55
        unit_weights[generic] = 0.35
        exp = f"scope_{scope}"
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "scoped_fact",
                    "atomic_fact",
                    2,
                    {unit: 1.0},
                    index,
                    f"FACT preference only in {scope} scope",
                ),
                _candidate(
                    prefix,
                    exp,
                    "generic_summary",
                    "summary",
                    2,
                    {generic: 0.8, unit: 0.35},
                    index,
                    f"generic summary that blurs {scope} scope",
                    confidence=0.7,
                ),
                _candidate(
                    prefix,
                    exp,
                    "scoped_summary",
                    "summary",
                    3,
                    {unit: 0.75, context: 0.8},
                    index,
                    f"summary preserving {scope} context",
                ),
                _candidate(
                    prefix,
                    exp,
                    "raw_scope",
                    "raw_span",
                    5,
                    {unit: 1.0, context: 1.0, generic: 0.2},
                    index,
                    f"raw evidence with explicit {scope} scope",
                ),
            ]
        )

    review_coverage = {unit: 0.42 for unit in scoped_units}
    review_coverage.update({f"{prefix}:context:{scope}": 0.35 for scope in scopes[:count]})
    candidates.append(
        _candidate(
            prefix,
            "cross_scope_review",
            "broad_review",
            "summary",
            4,
            review_coverage,
            count,
            "cross-scope review with partial details for every scope",
        )
    )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=scoped_units,
    )


def summary_tradeoff(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Partial summaries compete with full but costly raw or atomic memories."""

    rng = _rng(seed, 37)
    count = _bounded_count(normal_count + update_count, minimum=3, maximum=5)
    prefix = f"summary_tradeoff_s{seed}"
    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}

    for index in range(count):
        primary = f"{prefix}:primary:{index}"
        secondary = f"{prefix}:secondary:{index}"
        context = f"{prefix}:context:{index}"
        unit_weights[primary] = 1.35 + 0.15 * rng.randrange(3)
        unit_weights[secondary] = 1.0 + 0.1 * rng.randrange(2)
        unit_weights[context] = 0.35
        exp = f"episode_{index}"
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "primary_fact",
                    "atomic_fact",
                    2,
                    {primary: 1.0},
                    index,
                    f"FACT primary detail {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "secondary_fact",
                    "atomic_fact",
                    2,
                    {secondary: 1.0},
                    index,
                    f"FACT secondary detail {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "compressed_summary",
                    "summary",
                    2,
                    {primary: 0.45, secondary: 0.45, context: 0.25},
                    index,
                    f"compressed lossy summary {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "balanced_summary",
                    "summary",
                    3,
                    {primary: 0.72, secondary: 0.72, context: 0.45},
                    index,
                    f"balanced summary {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "raw_episode",
                    "raw_span",
                    5,
                    {primary: 1.0, secondary: 1.0, context: 0.85},
                    index,
                    f"raw episode evidence {index}",
                ),
            ]
        )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
    )


def redundancy_heavy(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Many candidates cover the same core unit, so saturation rewards diversity."""

    rng = _rng(seed, 41)
    count = _bounded_count(normal_count + update_count, minimum=4, maximum=6)
    prefix = f"redundancy_heavy_s{seed}"
    core = f"{prefix}:core"
    unit_weights: Dict[str, float] = {core: 2.4}
    candidates: List[CandidateMemory] = []

    for index in range(count):
        unique = f"{prefix}:tail:{index}"
        local = f"{prefix}:local:{index // 2}"
        unit_weights[unique] = 1.0 + 0.15 * rng.randrange(3)
        unit_weights.setdefault(local, 0.3)
        exp = f"redundant_{index}"
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "core_fact",
                    "atomic_fact",
                    2,
                    {core: 0.85},
                    index,
                    f"FACT repeated core claim {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "tail_fact",
                    "atomic_fact",
                    2,
                    {unique: 1.0},
                    index,
                    f"FACT unique tail {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "core_tail_summary",
                    "summary",
                    3,
                    {core: 0.6, unique: 0.8, local: 0.25},
                    index,
                    f"summary of core plus unique tail {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "raw_redundant",
                    "raw_span",
                    4,
                    {core: 1.0, unique: 1.0, local: 0.4},
                    index,
                    f"raw evidence repeating core and tail {index}",
                ),
            ]
        )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
    )


def temporal_interval(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Interval endpoints and closure updates require temporally precise memories."""

    interval_count = _bounded_count(normal_count + update_count, minimum=3, maximum=5)
    closure_count = _bounded_count(update_count, minimum=1, maximum=3)
    prefix = f"temporal_interval_s{seed}"
    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    current_units: List[str] = []
    invalidation_units: List[str] = []
    stale_units: List[str] = []

    for index in range(interval_count):
        label = f"{prefix}:interval:{index}:label"
        start = f"{prefix}:interval:{index}:start"
        end = f"{prefix}:interval:{index}:end"
        duration = f"{prefix}:interval:{index}:duration"
        current_units.extend([label, start, end])
        unit_weights[label] = 0.9
        unit_weights[start] = 0.75
        unit_weights[end] = 0.85
        unit_weights[duration] = 0.55
        exp = f"interval_{index}"
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "label_fact",
                    "atomic_fact",
                    2,
                    {label: 1.0},
                    index,
                    f"FACT interval {index} happened",
                ),
                _candidate(
                    prefix,
                    exp,
                    "endpoint_fact",
                    "interval_fact",
                    3,
                    {start: 1.0, end: 1.0, duration: 0.7},
                    index,
                    f"FACT interval {index} start and end",
                ),
                _candidate(
                    prefix,
                    exp,
                    "coarse_summary",
                    "summary",
                    2,
                    {label: 0.85, start: 0.35, end: 0.35},
                    index,
                    f"coarse temporal summary {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "raw_interval",
                    "raw_span",
                    5,
                    {label: 1.0, start: 1.0, end: 1.0, duration: 1.0},
                    index,
                    f"raw interval evidence {index}",
                ),
            ]
        )

    for index in range(closure_count):
        old_open = f"{prefix}:open_interval:{index}:old"
        closed = f"{prefix}:closed_interval:{index}:new_end"
        invalid = f"{prefix}:invalid_open_interval:{index}"
        stale_units.append(old_open)
        current_units.append(closed)
        invalidation_units.append(invalid)
        unit_weights[old_open] = 0.1
        unit_weights[closed] = 1.4
        unit_weights[invalid] = 1.6
        exp = f"closure_{index}"
        time_index = interval_count + index
        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "raw_closure",
                    "raw_span",
                    5,
                    {closed: 1.0, invalid: 0.45, old_open: 0.15},
                    time_index,
                    f"raw closure for interval {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "closed_fact",
                    "atomic_fact",
                    2,
                    {closed: 1.0},
                    time_index,
                    f"FACT interval {index} closed",
                ),
                _candidate(
                    prefix,
                    exp,
                    "invalidate_open",
                    "tombstone",
                    1,
                    {invalid: 1.0},
                    time_index,
                    f"TOMBSTONE open interval {index}",
                ),
                _candidate(
                    prefix,
                    exp,
                    "compound_closure",
                    "compound_update",
                    3,
                    {closed: 1.0, invalid: 1.0},
                    time_index,
                    f"UPDATE open interval {index} to closed",
                ),
                _candidate(
                    prefix,
                    exp,
                    "closure_summary",
                    "summary",
                    3,
                    {closed: 0.7, invalid: 0.7},
                    time_index,
                    f"summary closure for interval {index}",
                ),
            ]
        )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=tuple(dict.fromkeys(current_units)),
        invalidation_units=invalidation_units,
        stale_units=stale_units,
    )


def abstention_hard(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Ambiguous experiences reward retaining uncertainty over false precision."""

    rng = _rng(seed, 53)
    count = _bounded_count(normal_count + update_count, minimum=3, maximum=5)
    prefix = f"abstention_hard_s{seed}"
    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    current_units: List[str] = []
    stale_units: List[str] = []

    for index in range(count):
        exp = f"question_{index}"
        if index % 2 == 0:
            abstain = f"{prefix}:abstain:{index}"
            conflict = f"{prefix}:conflict:{index}"
            unsupported = f"{prefix}:unsupported_answer:{index}"
            current_units.extend([abstain, conflict])
            stale_units.append(unsupported)
            unit_weights[abstain] = 2.0 + 0.15 * rng.randrange(3)
            unit_weights[conflict] = 0.9
            unit_weights[unsupported] = 0.05
            candidates.extend(
                [
                    _candidate(
                        prefix,
                        exp,
                        "overconfident_fact",
                        "atomic_fact",
                        2,
                        {unsupported: 1.0},
                        index,
                        f"unsupported answer for ambiguous question {index}",
                        confidence=0.42,
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "abstain_marker",
                        "abstention",
                        1,
                        {abstain: 1.0},
                        index,
                        f"ABSTAIN insufficient evidence for question {index}",
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "conflict_marker",
                        "uncertainty",
                        2,
                        {abstain: 0.8, conflict: 1.0},
                        index,
                        f"conflicting evidence marker for question {index}",
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "raw_conflict",
                        "raw_span",
                        4,
                        {abstain: 0.9, conflict: 1.0, unsupported: 0.1},
                        index,
                        f"raw conflicting evidence for question {index}",
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "ambiguous_summary",
                        "summary",
                        3,
                        {abstain: 0.65, conflict: 0.7},
                        index,
                        f"summary noting ambiguity for question {index}",
                    ),
                ]
            )
        else:
            answer = f"{prefix}:answer:{index}"
            evidence = f"{prefix}:evidence:{index}"
            abstain = f"{prefix}:unneeded_abstain:{index}"
            current_units.append(answer)
            unit_weights[answer] = 1.75 + 0.15 * rng.randrange(2)
            unit_weights[evidence] = 0.55
            unit_weights[abstain] = 0.2
            candidates.extend(
                [
                    _candidate(
                        prefix,
                        exp,
                        "answer_fact",
                        "atomic_fact",
                        2,
                        {answer: 1.0},
                        index,
                        f"FACT supported answer {index}",
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "unneeded_abstain",
                        "abstention",
                        1,
                        {abstain: 1.0},
                        index,
                        f"unnecessary abstention for answerable question {index}",
                        confidence=0.5,
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "evidence_summary",
                        "summary",
                        3,
                        {answer: 0.75, evidence: 0.8},
                        index,
                        f"summary with support for answer {index}",
                    ),
                    _candidate(
                        prefix,
                        exp,
                        "raw_answer",
                        "raw_span",
                        4,
                        {answer: 1.0, evidence: 1.0},
                        index,
                        f"raw support for answer {index}",
                    ),
                ]
            )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=current_units,
        stale_units=stale_units,
    )


DISTRIBUTIONS: Dict[str, DistributionGenerator] = {
    "base": base,
    "density_trap": density_trap,
    "update_chain": update_chain,
    "scope_shift": scope_shift,
    "summary_tradeoff": summary_tradeoff,
    "redundancy_heavy": redundancy_heavy,
    "temporal_interval": temporal_interval,
    "abstention_hard": abstention_hard,
}


def generate_distribution(
    name: str,
    seed: int,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Generate a named deterministic exact-small distribution instance."""

    normalized = name.strip().lower()
    try:
        generator = DISTRIBUTIONS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(DISTRIBUTIONS))
        raise ValueError(f"unknown distribution {name!r}; available: {available}") from exc
    return generator(seed, normal_count=normal_count, update_count=update_count)


__all__ = [
    "DISTRIBUTIONS",
    "abstention_hard",
    "base",
    "density_trap",
    "generate_distribution",
    "redundancy_heavy",
    "scope_shift",
    "summary_tradeoff",
    "temporal_interval",
    "update_chain",
]
