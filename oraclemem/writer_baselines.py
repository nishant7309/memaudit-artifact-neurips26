"""Deployable local memory-writer baselines for OracleMem.

These policies intentionally do not inspect oracle coverage vectors or unit
weights.  They use only representation type, serialized text, confidence,
recency, novelty, and budget accounting so they can stand in for lightweight
Letta/MemGPT/Mem0/A-Mem/A-MAC-style writer adapters without external services.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WRITER_BASELINE_METHODS: Tuple[str, ...] = (
    "memgpt_tiered",
    "mem0_extract",
    "amem_graph",
    "amac_admission",
)

WRITER_BASELINE_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = {
    "memgpt_tiered": {
        "proxy_for": "Letta/MemGPT-style archival/recency tiered memory",
        "decision_features": "representation type, serialized text, confidence, recency, novelty, and budget",
        "limitation": (
            "Faithful local adapter only: it does not run a Letta server, "
            "MemGPT's controller, paging loop, tool calls, summarizer, or retriever."
        ),
    },
    "mem0_extract": {
        "proxy_for": "Mem0-style extraction and consolidation",
        "decision_features": "compact fact/update candidates, duplicate penalties, confidence, novelty, and budget",
        "limitation": (
            "Local proxy only: it does not run Mem0's extraction model, vector store, "
            "graph store, or update pipeline."
        ),
    },
    "amem_graph": {
        "proxy_for": "A-Mem-style adaptive graph/evolving memory",
        "decision_features": "graph/summary/update type priors, text anchors, link overlap, novelty, recency, and budget",
        "limitation": (
            "Faithful local adapter only: it does not run A-Mem's learned memory "
            "evolution, LLM-generated relation expansion, or retrieval-time graph traversal."
        ),
    },
    "amac_admission": {
        "proxy_for": "A-MAC-style memory admission",
        "decision_features": "estimated salience, confidence, novelty, recency, type prior, online admission, and eviction",
        "limitation": (
            "Local proxy only: it does not run the published A-MAC policy, learned "
            "admission model, or task-specific reward estimator."
        ),
    },
}

_TOKEN_RE = re.compile(r"[a-z0-9_:-]+")
_STOPWORDS = {
    "a",
    "about",
    "after",
    "and",
    "for",
    "in",
    "is",
    "of",
    "old",
    "or",
    "that",
    "the",
    "to",
    "user",
    "with",
}

_SALIENT_TERMS = {
    "abstain",
    "actually",
    "changed",
    "closed",
    "conflict",
    "correction",
    "current",
    "evidence",
    "exception",
    "fact",
    "invalid",
    "invalidate",
    "invalidated",
    "no",
    "not",
    "now",
    "outcome",
    "prefer",
    "preference",
    "scope",
    "superseded",
    "tombstone",
    "update",
}

_MEMGPT_TYPE_PRIOR: Mapping[str, float] = {
    "raw": 1.15,
    "raw_span": 1.15,
    "summary": 1.05,
    "atomic_fact": 1.00,
    "fact": 1.00,
    "compound_update": 1.15,
    "tombstone": 1.05,
    "compound_evidence": 1.10,
    "interval_fact": 1.00,
    "graph_edge": 0.95,
    "skill": 0.95,
    "abstention": 1.00,
    "uncertainty": 1.00,
}

_MEM0_TYPE_PRIOR: Mapping[str, float] = {
    "atomic_fact": 1.28,
    "fact": 1.28,
    "graph_edge": 1.18,
    "skill": 1.16,
    "tombstone": 1.18,
    "compound_update": 1.14,
    "abstention": 1.12,
    "uncertainty": 1.12,
    "interval_fact": 1.10,
    "summary": 1.00,
    "compound_evidence": 1.00,
    "raw": 0.72,
    "raw_span": 0.72,
}

_AMAC_TYPE_PRIOR: Mapping[str, float] = {
    "atomic_fact": 1.10,
    "fact": 1.10,
    "summary": 1.06,
    "compound_update": 1.22,
    "tombstone": 1.20,
    "compound_evidence": 1.14,
    "interval_fact": 1.12,
    "graph_edge": 1.08,
    "skill": 1.08,
    "abstention": 1.12,
    "uncertainty": 1.12,
    "raw": 0.92,
    "raw_span": 0.92,
}

_AMEM_TYPE_PRIOR: Mapping[str, float] = {
    "graph_edge": 1.34,
    "compound_update": 1.30,
    "compound_evidence": 1.26,
    "summary": 1.22,
    "interval_fact": 1.18,
    "skill": 1.16,
    "tombstone": 1.14,
    "atomic_fact": 1.08,
    "fact": 1.08,
    "abstention": 1.05,
    "uncertainty": 1.05,
    "raw": 0.76,
    "raw_span": 0.76,
}

_COMPACT_TYPES = {
    "abstention",
    "atomic_fact",
    "fact",
    "graph_edge",
    "interval_fact",
    "skill",
    "summary",
    "tombstone",
    "uncertainty",
}

_GRAPH_ANCHOR_TERMS = {
    "bridge",
    "conference",
    "constraint",
    "current",
    "duration",
    "edge",
    "end",
    "exception",
    "historical",
    "invalid",
    "invalidated",
    "link",
    "outcome",
    "scope",
    "scoped",
    "source",
    "start",
    "tool",
    "travel",
    "update",
}


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: Any
    score: float
    density: float
    novelty: float


def select_writer_baseline(
    method: str,
    candidates: Sequence[Any],
    budget: int,
) -> Tuple[str, ...]:
    """Dispatch a local deployable writer baseline."""

    if method == "memgpt_tiered":
        return memgpt_tiered_select(candidates, budget)
    if method == "mem0_extract":
        return mem0_extract_select(candidates, budget)
    if method == "amem_graph":
        return amem_graph_select(candidates, budget)
    if method == "amac_admission":
        return amac_admission_select(candidates, budget)
    raise ValueError(f"unknown writer baseline: {method}")


def memgpt_tiered_select(candidates: Sequence[Any], budget: int) -> Tuple[str, ...]:
    """Letta/MemGPT-inspired tiered writer with local compaction.

    The writer initially keeps one useful representation per experience, gives
    recent raw spans a main-context bonus, then compacts raw/verbose memories to
    cheaper fact or summary records before evicting low-priority memories.
    """

    groups = _ordered_groups(candidates)
    selected: List[Any] = []
    selected_by_exp: Dict[str, Any] = {}
    selected_tokens: set[str] = set()

    for group in groups:
        best = _best_in_group(
            group,
            candidates,
            selected_tokens,
            scorer=_memgpt_score,
            density_power=0.35,
        )
        if best is None or best.score <= 0.35:
            continue
        chosen = best.candidate
        selected.append(chosen)
        selected_by_exp[str(chosen.experience_id)] = chosen
        selected_tokens.update(_candidate_tokens(chosen))

    score_cache = {
        str(candidate.candidate_id): _memgpt_score(candidate, _recency_score(candidate, candidates), 1.0)
        for candidate in candidates
    }
    selected = _compact_or_evict(
        selected,
        groups,
        budget,
        score_cache,
        preferred_replacements=_COMPACT_TYPES,
    )
    return tuple(str(candidate.candidate_id) for candidate in _chronological(selected))


def mem0_extract_select(candidates: Sequence[Any], budget: int) -> Tuple[str, ...]:
    """Mem0-inspired extraction/consolidation writer.

    The writer favors concise fact, graph, skill, summary, and validity-state
    records, penalizes near-duplicate memories, and then packs the highest
    estimated-value records under the same storage budget.
    """

    groups = _ordered_groups(candidates)
    proposed: List[_ScoredCandidate] = []
    accepted_tokens: set[str] = set()
    seen_signatures: List[set[str]] = []

    for group in groups:
        best = _best_in_group(
            group,
            candidates,
            accepted_tokens,
            scorer=_mem0_score,
            density_power=0.70,
            duplicate_sets=seen_signatures,
        )
        if best is None or best.score <= 0.35:
            continue
        proposed.append(best)
        accepted_tokens.update(_candidate_tokens(best.candidate))
        seen_signatures.append(_signature_tokens(best.candidate))

    selected: List[Any] = []
    used_groups: set[str] = set()
    used_cost = 0
    for scored in sorted(
        proposed,
        key=lambda item: (item.density, item.score, _recency_score(item.candidate, candidates)),
        reverse=True,
    ):
        candidate = scored.candidate
        exp_id = str(candidate.experience_id)
        if exp_id in used_groups or used_cost + int(candidate.cost) > budget:
            continue
        selected.append(candidate)
        used_groups.add(exp_id)
        used_cost += int(candidate.cost)

    return tuple(str(candidate.candidate_id) for candidate in _chronological(selected))


def amem_graph_select(candidates: Sequence[Any], budget: int) -> Tuple[str, ...]:
    """A-Mem-inspired adaptive graph/evolving-memory writer.

    The writer favors compact memories that expose scope, relation, temporal,
    and update anchors, gives a small bonus to candidates that link to anchors
    already retained, and admits or evicts records under the same budget.  It
    is a no-API proxy over pre-generated OracleMem candidates, not A-Mem code.
    """

    groups = _ordered_groups(candidates)
    selected: List[Any] = []
    priorities: Dict[str, float] = {}
    selected_tokens: set[str] = set()
    anchor_counts: Dict[str, int] = {}
    used_cost = 0

    def rebuild_state() -> None:
        selected_tokens.clear()
        anchor_counts.clear()
        for item in selected:
            selected_tokens.update(_candidate_tokens(item))
            for token in _anchor_tokens(item):
                anchor_counts[token] = anchor_counts.get(token, 0) + 1

    for group in groups:
        scored: List[_ScoredCandidate] = []
        for candidate in group:
            novelty = _novelty(candidate, selected_tokens)
            recency = _recency_score(candidate, candidates)
            link_overlap = _anchor_overlap(candidate, anchor_counts)
            score = _amem_score(candidate, recency, novelty, link_overlap)
            cost = max(float(candidate.cost), 1.0)
            density = score / (cost ** 0.60)
            scored.append(_ScoredCandidate(candidate, score, density, novelty))
        if not scored:
            continue

        best = max(
            scored,
            key=lambda item: (
                item.density,
                item.score,
                -int(item.candidate.cost),
                str(item.candidate.candidate_id),
            ),
        )
        if best.score <= 0.38:
            continue

        candidate = best.candidate
        candidate_cost = int(candidate.cost)
        if candidate_cost > budget:
            continue
        priority = best.score / max(float(candidate_cost) ** 0.55, 1.0)

        if used_cost + candidate_cost <= budget:
            selected.append(candidate)
            priorities[str(candidate.candidate_id)] = priority
            used_cost += candidate_cost
            rebuild_state()
            continue

        working = list(selected)
        working_cost = used_cost
        removed: List[Any] = []
        while working and working_cost + candidate_cost > budget:
            worst = min(
                working,
                key=lambda item: (
                    priorities.get(str(item.candidate_id), 0.0),
                    _anchor_overlap(item, anchor_counts),
                    _recency_score(item, candidates),
                    -int(item.cost),
                ),
            )
            worst_priority = priorities.get(str(worst.candidate_id), 0.0)
            if worst_priority >= priority:
                break
            working.remove(worst)
            removed.append(worst)
            working_cost -= int(worst.cost)

        if working_cost + candidate_cost <= budget and (
            not removed or priority > max(priorities.get(str(item.candidate_id), 0.0) for item in removed)
        ):
            selected = working + [candidate]
            used_cost = working_cost + candidate_cost
            for item in removed:
                priorities.pop(str(item.candidate_id), None)
            priorities[str(candidate.candidate_id)] = priority
            rebuild_state()

    return tuple(str(candidate.candidate_id) for candidate in _chronological(selected))


def amac_admission_select(candidates: Sequence[Any], budget: int) -> Tuple[str, ...]:
    """A-MAC-inspired online admission with eviction.

    Each arriving experience proposes one candidate scored by estimated utility,
    confidence, novelty, recency, and content type.  The writer admits it when
    it fits or when it can evict lower-priority memories to recover budget.
    """

    groups = _ordered_groups(candidates)
    selected: List[Any] = []
    priorities: Dict[str, float] = {}
    selected_tokens: set[str] = set()
    used_cost = 0

    for group in groups:
        best = _best_in_group(
            group,
            candidates,
            selected_tokens,
            scorer=_amac_score,
            density_power=0.55,
        )
        if best is None or best.score <= 0.40:
            continue

        candidate = best.candidate
        candidate_cost = int(candidate.cost)
        if candidate_cost > budget:
            continue
        priority = best.score / max(math.sqrt(candidate_cost), 1.0)

        if used_cost + candidate_cost <= budget:
            selected.append(candidate)
            priorities[str(candidate.candidate_id)] = priority
            used_cost += candidate_cost
            selected_tokens.update(_candidate_tokens(candidate))
            continue

        working = list(selected)
        working_cost = used_cost
        removed: List[Any] = []
        while working and working_cost + candidate_cost > budget:
            worst = min(
                working,
                key=lambda item: (
                    priorities.get(str(item.candidate_id), 0.0),
                    _recency_score(item, candidates),
                    -int(item.cost),
                ),
            )
            worst_priority = priorities.get(str(worst.candidate_id), 0.0)
            if worst_priority >= priority:
                break
            working.remove(worst)
            removed.append(worst)
            working_cost -= int(worst.cost)

        if working_cost + candidate_cost <= budget and (
            not removed or priority > max(priorities.get(str(item.candidate_id), 0.0) for item in removed)
        ):
            selected = working + [candidate]
            used_cost = working_cost + candidate_cost
            for item in removed:
                priorities.pop(str(item.candidate_id), None)
            priorities[str(candidate.candidate_id)] = priority
            selected_tokens = set()
            for item in selected:
                selected_tokens.update(_candidate_tokens(item))

    return tuple(str(candidate.candidate_id) for candidate in _chronological(selected))


def _best_in_group(
    group: Sequence[Any],
    universe: Sequence[Any],
    selected_tokens: set[str],
    *,
    scorer: Callable[[Any, float, float], float],
    density_power: float,
    duplicate_sets: Optional[Sequence[set[str]]] = None,
) -> Optional[_ScoredCandidate]:
    scored: List[_ScoredCandidate] = []
    for candidate in group:
        novelty = _novelty(candidate, selected_tokens)
        duplicate_penalty = _duplicate_penalty(candidate, duplicate_sets or ())
        recency = _recency_score(candidate, universe)
        score = scorer(candidate, recency, novelty) * duplicate_penalty
        cost = max(float(candidate.cost), 1.0)
        density = score / (cost ** density_power)
        scored.append(_ScoredCandidate(candidate, score, density, novelty))
    if not scored:
        return None
    return max(
        scored,
        key=lambda item: (
            item.density,
            item.score,
            -int(item.candidate.cost),
            str(item.candidate.candidate_id),
        ),
    )


def _memgpt_score(candidate: Any, recency: float, novelty: float) -> float:
    rep = _representation_type(candidate)
    type_prior = _MEMGPT_TYPE_PRIOR.get(rep, 0.90)
    recent_raw_bonus = 0.35 if rep in {"raw", "raw_span"} and recency >= 0.65 else 0.0
    compact_bonus = 0.18 if rep in _COMPACT_TYPES and recency < 0.75 else 0.0
    return (
        type_prior
        + recent_raw_bonus
        + compact_bonus
        + 0.55 * _salience(candidate)
        + 0.25 * novelty
        + 0.30 * recency
        + 0.20 * _confidence(candidate)
    )


def _mem0_score(candidate: Any, recency: float, novelty: float) -> float:
    rep = _representation_type(candidate)
    type_prior = _MEM0_TYPE_PRIOR.get(rep, 0.85)
    extraction_bonus = 0.20 if rep in {"atomic_fact", "fact", "graph_edge", "skill"} else 0.0
    validity_bonus = 0.18 if rep in {"tombstone", "compound_update", "abstention", "uncertainty"} else 0.0
    return (
        type_prior
        + extraction_bonus
        + validity_bonus
        + 0.60 * _salience(candidate)
        + 0.35 * novelty
        + 0.15 * recency
        + 0.30 * _confidence(candidate)
    )


def _amem_score(candidate: Any, recency: float, novelty: float, link_overlap: float) -> float:
    rep = _representation_type(candidate)
    type_prior = _AMEM_TYPE_PRIOR.get(rep, 0.90)
    graph_bonus = 0.28 if rep in {"graph_edge", "summary", "compound_update", "compound_evidence"} else 0.0
    temporal_bonus = 0.16 if rep in {"interval_fact", "tombstone"} else 0.0
    return (
        type_prior
        + graph_bonus
        + temporal_bonus
        + 0.48 * _graph_affinity(candidate)
        + 0.42 * link_overlap
        + 0.34 * novelty
        + 0.22 * recency
        + 0.26 * _confidence(candidate)
        + 0.24 * _salience(candidate)
    )


def _amac_score(candidate: Any, recency: float, novelty: float) -> float:
    rep = _representation_type(candidate)
    type_prior = _AMAC_TYPE_PRIOR.get(rep, 0.95)
    salience = _salience(candidate)
    confidence = _confidence(candidate)
    return (
        type_prior
        + 0.62 * salience
        + 0.42 * confidence
        + 0.38 * novelty
        + 0.24 * recency
    )


def _compact_or_evict(
    selected: Sequence[Any],
    groups: Sequence[Sequence[Any]],
    budget: int,
    score_cache: Mapping[str, float],
    *,
    preferred_replacements: Iterable[str],
) -> List[Any]:
    selected = list(selected)
    by_exp = {str(candidate.experience_id): tuple(group) for group in groups for candidate in group[:1]}
    replacement_types = set(preferred_replacements)

    while _cost(selected) > budget and selected:
        best_replacement: Optional[Tuple[float, int, str, Any, Any]] = None
        for current in selected:
            current_score = score_cache.get(str(current.candidate_id), 0.0)
            for alternative in by_exp.get(str(current.experience_id), ()):
                if str(alternative.candidate_id) == str(current.candidate_id):
                    continue
                if int(alternative.cost) >= int(current.cost):
                    continue
                if _representation_type(alternative) not in replacement_types:
                    continue
                saved = int(current.cost) - int(alternative.cost)
                loss = current_score - score_cache.get(str(alternative.candidate_id), 0.0)
                key = (loss / max(saved, 1), -saved, str(alternative.candidate_id))
                if best_replacement is None or key < best_replacement[:3]:
                    best_replacement = (key[0], key[1], key[2], current, alternative)

        if best_replacement is not None:
            _, _, _, current, alternative = best_replacement
            selected = [alternative if item is current else item for item in selected]
            continue

        selected.remove(
            min(
                selected,
                key=lambda item: (
                    score_cache.get(str(item.candidate_id), 0.0) / max(float(item.cost), 1.0),
                    score_cache.get(str(item.candidate_id), 0.0),
                    _recency_score(item, selected),
                ),
            )
        )

    return selected


def _ordered_groups(candidates: Sequence[Any]) -> List[List[Any]]:
    groups: Dict[str, List[Any]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate.experience_id), []).append(candidate)
    return [
        sorted(groups[key], key=lambda item: (int(item.cost), str(item.candidate_id)))
        for key in sorted(
            groups,
            key=lambda group_id: (
                min(int(candidate.time_index) for candidate in groups[group_id]),
                group_id,
            ),
        )
    ]


def _chronological(candidates: Sequence[Any]) -> List[Any]:
    return sorted(
        candidates,
        key=lambda item: (int(item.time_index), str(item.experience_id), str(item.candidate_id)),
    )


def _cost(candidates: Sequence[Any]) -> int:
    return sum(int(candidate.cost) for candidate in candidates)


def _candidate_tokens(candidate: Any) -> set[str]:
    text = f"{_representation_type(candidate)} {_serialized(candidate)}"
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _signature_tokens(candidate: Any) -> set[str]:
    tokens = _candidate_tokens(candidate)
    return {token for token in tokens if ":" in token or token in _SALIENT_TERMS}


def _anchor_tokens(candidate: Any) -> set[str]:
    tokens = _candidate_tokens(candidate)
    return {
        token
        for token in tokens
        if ":" in token
        or token in _GRAPH_ANCHOR_TERMS
        or token.startswith(("invalid", "scope", "update"))
    }


def _anchor_overlap(candidate: Any, anchor_counts: Mapping[str, int]) -> float:
    anchors = _anchor_tokens(candidate)
    if not anchors or not anchor_counts:
        return 0.0
    seen = sum(1 for token in anchors if token in anchor_counts)
    weighted = sum(min(anchor_counts.get(token, 0), 3) for token in anchors)
    return min(1.0, 0.55 * seen / len(anchors) + 0.45 * weighted / (3 * len(anchors)))


def _novelty(candidate: Any, selected_tokens: set[str]) -> float:
    tokens = _candidate_tokens(candidate)
    if not tokens or not selected_tokens:
        return 1.0
    return len(tokens - selected_tokens) / max(len(tokens), 1)


def _duplicate_penalty(candidate: Any, duplicate_sets: Sequence[set[str]]) -> float:
    signature = _signature_tokens(candidate)
    if not signature:
        return 1.0
    max_overlap = 0.0
    for existing in duplicate_sets:
        if not existing:
            continue
        overlap = len(signature & existing) / max(len(signature | existing), 1)
        max_overlap = max(max_overlap, overlap)
    if max_overlap >= 0.80:
        return 0.45
    if max_overlap >= 0.55:
        return 0.70
    return 1.0


def _recency_score(candidate: Any, candidates: Sequence[Any]) -> float:
    times = [int(item.time_index) for item in candidates]
    if not times:
        return 0.0
    low = min(times)
    high = max(times)
    if high == low:
        return 0.5
    return (int(candidate.time_index) - low) / (high - low)


def _salience(candidate: Any) -> float:
    tokens = _candidate_tokens(candidate)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in _SALIENT_TERMS or token.startswith("invalid"))
    rep = _representation_type(candidate)
    type_hit = 1 if rep in {"tombstone", "compound_update", "abstention", "uncertainty"} else 0
    return min(1.0, (hits + type_hit) / 5.0)


def _graph_affinity(candidate: Any) -> float:
    tokens = _candidate_tokens(candidate)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in _GRAPH_ANCHOR_TERMS)
    colon_hits = min(2, sum(1 for token in tokens if ":" in token))
    rep = _representation_type(candidate)
    type_hit = 1 if rep in {"graph_edge", "compound_update", "compound_evidence", "interval_fact"} else 0
    return min(1.0, (hits + colon_hits + type_hit) / 6.0)


def _confidence(candidate: Any) -> float:
    value = float(getattr(candidate, "confidence", 1.0))
    return max(0.0, min(1.0, value))


def _representation_type(candidate: Any) -> str:
    return str(getattr(candidate, "representation_type", getattr(candidate, "representation", "")))


def _serialized(candidate: Any) -> str:
    return str(getattr(candidate, "serialized", getattr(candidate, "text", "")))
