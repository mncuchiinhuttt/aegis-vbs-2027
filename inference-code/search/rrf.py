"""Pure reciprocal-rank fusion for already-ranked result lists."""

import math
from decimal import Decimal
from numbers import Real


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def merge_rrf(*ranked_lists, k=60, labels=None, weights=None) -> list:
    """Fuse ranked hit lists with optional nonnegative per-list RRF weights.

    Ranks are one-based and ties retain first-seen candidate order. A zero
    weight disables that list entirely, including candidate and label
    contributions.
    """
    k_value = _finite_real(k, "k")
    if k_value <= 0:
        raise ValueError("k must be greater than zero")

    if labels is not None:
        try:
            label_count = len(labels)
        except TypeError as exc:
            raise ValueError("labels must have the same length as ranked_lists") from exc
        if label_count != len(ranked_lists):
            raise ValueError("labels must have the same length as ranked_lists")

    if weights is None:
        normalized_weights = (1.0,) * len(ranked_lists)
    else:
        try:
            raw_weights = tuple(weights)
        except TypeError as exc:
            raise ValueError("weights must have the same length as ranked_lists") from exc
        if len(raw_weights) != len(ranked_lists):
            raise ValueError("weights must have the same length as ranked_lists")
        normalized_weights = tuple(
            _finite_real(weight, f"weights[{index}]")
            for index, weight in enumerate(raw_weights)
        )
        if any(weight < 0 for weight in normalized_weights):
            raise ValueError("weights must be nonnegative")

    rrf_scores = {}
    payload_map = {}
    matched_via = {}

    for list_idx, (hits, weight) in enumerate(zip(ranked_lists, normalized_weights)):
        if weight == 0:
            continue
        for rank, hit in enumerate(hits):
            doc_id = hit["id"]
            payload_map[doc_id] = hit["payload"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight / (k_value + rank + 1)
            if labels is not None:
                matched_via.setdefault(doc_id, []).append(labels[list_idx])

    sorted_ids = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    merged_results = []
    for doc_id, rrf_score in sorted_ids:
        result = {"id": doc_id, "rrf_score": rrf_score, "payload": payload_map[doc_id]}
        if labels is not None:
            result["matched_via"] = matched_via[doc_id]
        merged_results.append(result)
    return merged_results
