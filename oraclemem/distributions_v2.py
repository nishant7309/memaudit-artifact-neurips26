"""Second-generation exact-small OracleMem stress distributions.

These generators are hand-shaped review fixtures.  They keep the instances
small enough for exact search while making the heuristic failure semantic:
dense but incomplete notes compete with fuller representations, and scoped
corrections require update-aware candidates rather than stale broad memories.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping
import random

from .evaluate import CandidateMemory, OracleMemInstance


DistributionGeneratorV2 = Callable[..., OracleMemInstance]

MIN_QUERY_COUNT = 2
MAX_QUERY_COUNT = 4
MIN_UNITS_PER_QUERY = 2
MAX_UNITS_PER_QUERY = 4


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
        generator="oraclemem.distributions_v2",
        confidence=confidence,
    )


def density_trap_v2(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Dense hint memories lose to complete future-query evidence bundles.

    Each experience is a future query with 2-4 semantic units.  The cheap hint
    is intentionally high-density because it records a query anchor, but it
    omits the constraints/outcome needed to answer the query.  Complete and
    compound representations are larger, lower-density, and higher raw value
    because they cover all evidence units.
    """

    rng = _rng(seed, 101)
    query_count = _bounded_count(normal_count + 1, minimum=MIN_QUERY_COUNT, maximum=MAX_QUERY_COUNT)
    units_per_query = _bounded_count(
        update_count + 1,
        minimum=MIN_UNITS_PER_QUERY,
        maximum=MAX_UNITS_PER_QUERY,
    )
    prefix = f"density_trap_v2_s{seed}"
    roles = ("intent", "constraint", "outcome", "exception")
    topics = ["meal", "hotel", "flight", "calendar", "budget", "client"]
    rng.shuffle(topics)

    candidates: List[CandidateMemory] = []
    unit_weights: Dict[str, float] = {}
    current_units: List[str] = []

    for query_index in range(query_count):
        topic = topics[query_index % len(topics)]
        exp = f"future_query_{query_index}"
        required_units = [
            f"{prefix}:q{query_index}:{role}"
            for role in roles[:units_per_query]
        ]
        bridge_unit = f"{prefix}:q{query_index}:compound_bridge"
        provenance_unit = f"{prefix}:q{query_index}:source_detail"

        for role_index, unit in enumerate(required_units):
            unit_weights[unit] = 1.05 - 0.05 * min(role_index, 3)
        unit_weights[bridge_unit] = 0.65
        unit_weights[provenance_unit] = 0.45
        current_units.extend(required_units)

        hint_coverage: Dict[str, float] = {required_units[0]: 1.0}
        if len(required_units) > 1:
            hint_coverage[required_units[1]] = 0.12

        complete_coverage = {unit: 1.0 for unit in required_units}
        compound_coverage = dict(complete_coverage)
        compound_coverage[bridge_unit] = 1.0
        raw_coverage = dict(compound_coverage)
        raw_coverage[provenance_unit] = 1.0

        candidates.extend(
            [
                _candidate(
                    prefix,
                    exp,
                    "cheap_hint",
                    "atomic_fact",
                    1,
                    hint_coverage,
                    query_index,
                    (
                        f"HINT {topic}: salient keyword for future query "
                        f"{query_index}, without the full constraint/outcome."
                    ),
                    confidence=0.74,
                ),
                _candidate(
                    prefix,
                    exp,
                    "complete_summary",
                    "summary",
                    max(3, units_per_query),
                    complete_coverage,
                    query_index,
                    (
                        f"COMPLETE {topic}: intent, constraints, and outcome "
                        f"needed by future query {query_index}."
                    ),
                ),
                _candidate(
                    prefix,
                    exp,
                    "compound_case",
                    "compound_evidence",
                    max(4, units_per_query + 1),
                    compound_coverage,
                    query_index,
                    (
                        f"COMPOUND {topic}: complete evidence plus the link "
                        "between the separate facts."
                    ),
                ),
                _candidate(
                    prefix,
                    exp,
                    "raw_complete",
                    "raw_span",
                    max(5, units_per_query + 2),
                    raw_coverage,
                    query_index,
                    (
                        f"RAW {topic}: full exchange preserving complete "
                        "evidence and source detail."
                    ),
                ),
            ]
        )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=current_units,
    )


def scope_shift_v2(
    seed: int,
    *,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Broad-vs-narrow scope conflicts with a current scoped correction.

    The fixed core models five related memories: a general preference,
    travel-only preference, conference-only exception, historical preference,
    and current scoped correction.  The exact no-tombstone optimum drops because
    the only complete correction is tombstone-like, while density-only is lured
    by broad or partial high-density memories and then keeps only the
    invalidation half of the correction.
    """

    rng = _rng(seed, 211)
    prefix = f"scope_shift_v2_s{seed}"
    subjects = ("lodging", "meals", "seating", "transport")
    subject = subjects[rng.randrange(len(subjects))]
    candidates: List[CandidateMemory] = []

    general = f"{prefix}:{subject}:pref:general"
    travel = f"{prefix}:{subject}:pref:travel_only"
    conference = f"{prefix}:{subject}:pref:conference_exception"
    stale = f"{prefix}:{subject}:stale:historical"
    current = f"{prefix}:{subject}:current:conference_correction"
    invalid = f"{prefix}:{subject}:invalid:historical_after_correction"
    travel_scope = f"{prefix}:{subject}:scope:travel"
    conference_scope = f"{prefix}:{subject}:scope:conference"
    historical_scope = f"{prefix}:{subject}:scope:historical"

    unit_weights: Dict[str, float] = {
        general: 1.10,
        travel: 1.20,
        conference: 1.50,
        stale: 0.20,
        current: 2.00,
        invalid: 3.00,
        travel_scope: 0.60,
        conference_scope: 0.80,
        historical_scope: 0.30,
    }

    candidates.extend(
        [
            _candidate(
                prefix,
                "general_preference",
                "broad_fact",
                "atomic_fact",
                1,
                {general: 1.0},
                0,
                f"GENERAL {subject}: default preference outside special scopes.",
            ),
            _candidate(
                prefix,
                "general_preference",
                "scoped_general_summary",
                "summary",
                3,
                {general: 1.0, travel_scope: 0.35, conference_scope: 0.35},
                0,
                f"GENERAL {subject}: default preference with scope boundaries.",
            ),
            _candidate(
                prefix,
                "travel_only_preference",
                "travel_hint",
                "atomic_fact",
                1,
                {travel: 0.88},
                1,
                f"TRAVEL HINT {subject}: says there is a travel-specific preference.",
                confidence=0.76,
            ),
            _candidate(
                prefix,
                "travel_only_preference",
                "travel_scoped_fact",
                "summary",
                2,
                {travel: 1.0, travel_scope: 1.0},
                1,
                f"TRAVEL ONLY {subject}: narrow preference and explicit travel scope.",
            ),
            _candidate(
                prefix,
                "conference_exception",
                "conference_hint",
                "atomic_fact",
                1,
                {conference: 0.85},
                2,
                f"CONFERENCE HINT {subject}: exception exists but scope is incomplete.",
                confidence=0.72,
            ),
            _candidate(
                prefix,
                "conference_exception",
                "conference_scoped_exception",
                "summary",
                3,
                {conference: 1.0, conference_scope: 1.0},
                2,
                f"CONFERENCE ONLY {subject}: exception with explicit conference scope.",
            ),
            _candidate(
                prefix,
                "historical_preference",
                "historical_broad_summary",
                "summary",
                1,
                {general: 0.55, stale: 0.50, historical_scope: 0.25},
                3,
                f"HISTORICAL {subject}: old broad preference, not marked obsolete.",
                confidence=0.68,
            ),
            _candidate(
                prefix,
                "historical_preference",
                "historical_raw",
                "raw_span",
                3,
                {stale: 1.0, historical_scope: 1.0},
                3,
                f"RAW HISTORICAL {subject}: older preference before later correction.",
            ),
            _candidate(
                prefix,
                "current_scoped_correction",
                "current_fact_only",
                "atomic_fact",
                2,
                {current: 1.0},
                4,
                f"CURRENT {subject}: corrected conference-scoped preference.",
            ),
            _candidate(
                prefix,
                "current_scoped_correction",
                "invalidate_historical",
                "tombstone",
                1,
                {invalid: 1.0},
                4,
                f"TOMBSTONE {subject}: historical preference no longer applies.",
            ),
            _candidate(
                prefix,
                "current_scoped_correction",
                "compound_scoped_update",
                "compound_update",
                3,
                {current: 1.0, invalid: 1.0, conference_scope: 1.0},
                4,
                (
                    f"UPDATE {subject}: historical preference is invalidated "
                    "and replaced only in conference scope."
                ),
            ),
            _candidate(
                prefix,
                "current_scoped_correction",
                "ambiguous_current_summary",
                "summary",
                3,
                {current: 0.72, conference_scope: 0.70},
                4,
                (
                    f"SUMMARY {subject}: current correction but does not carry "
                    "the invalidation evidence."
                ),
            ),
        ]
    )

    # Keep the signature meaningful without changing the exact-small character:
    # extra requested updates add low-weight scoped context, not new conflicts.
    extra_context_count = max(0, _bounded_count(normal_count + update_count, minimum=3, maximum=5) - 4)
    for offset in range(extra_context_count):
        unit = f"{prefix}:{subject}:context:routine_{offset}"
        unit_weights[unit] = 0.25
        candidates.append(
            _candidate(
                prefix,
                f"routine_context_{offset}",
                "context_note",
                "summary",
                2,
                {unit: 1.0, general: 0.20},
                5 + offset,
                f"ROUTINE CONTEXT {subject}: ancillary scope detail {offset}.",
            )
        )

    return OracleMemInstance(
        instance_id=prefix,
        candidates=candidates,
        unit_weights=unit_weights,
        seed=seed,
        current_units=(travel, conference, current),
        invalidation_units=(invalid,),
        stale_units=(stale,),
    )


DISTRIBUTIONS_V2: Dict[str, DistributionGeneratorV2] = {
    "density_trap_v2": density_trap_v2,
    "scope_shift_v2": scope_shift_v2,
}


def generate_distribution_v2(
    name: str,
    seed: int,
    normal_count: int = 3,
    update_count: int = 2,
) -> OracleMemInstance:
    """Generate a named deterministic v2 exact-small distribution instance."""

    normalized = name.strip().lower()
    try:
        generator = DISTRIBUTIONS_V2[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(DISTRIBUTIONS_V2))
        raise ValueError(f"unknown v2 distribution {name!r}; available: {available}") from exc
    return generator(seed, normal_count=normal_count, update_count=update_count)


__all__ = [
    "DISTRIBUTIONS_V2",
    "density_trap_v2",
    "generate_distribution_v2",
    "scope_shift_v2",
]
