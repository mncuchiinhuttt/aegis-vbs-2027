"""Model-free metrics for explicitly annotated VQA replay outcomes."""

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

_STATUSES = {"answered", "abstained", "error"}
_ANSWERABILITY = {"answerable", "unanswerable"}


def normalize_answer(value: str) -> str:
    """Normalize only Unicode compatibility, case, and whitespace for exact match."""
    if not isinstance(value, str):
        raise TypeError("answers and references must be strings")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def compute_vqa_metrics(
    outcomes: Iterable[Mapping[str, Any]],
    support_labels: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Compute exact-match scores; misses, abstentions, and errors remain misses."""
    rows = list(outcomes)
    counts = {status: 0 for status in _STATUSES}
    unanswerable_statuses = {status: 0 for status in _STATUSES}
    answerable = unanswerable = answered_answerable = exact_correct = answered_correct = 0
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each outcome must be a mapping")
        status = row.get("status")
        label = row.get("answerability")
        identifier = row.get("id")
        references = row.get("accepted_references")
        if (
            not isinstance(status, str) or status not in _STATUSES
            or not isinstance(label, str) or label not in _ANSWERABILITY
        ):
            raise ValueError("each outcome needs a valid status and answerability label")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("outcome IDs must be unique nonempty strings")
        identifiers.add(identifier)
        if not isinstance(references, list) or any(not isinstance(ref, str) for ref in references):
            raise ValueError("accepted_references must be a list of strings")
        counts[status] += 1
        if label == "unanswerable":
            unanswerable += 1
            unanswerable_statuses[status] += 1
            if references:
                raise ValueError("unanswerable outcomes must not have accepted references")
            continue
        if not references or any(not normalize_answer(ref) for ref in references):
            raise ValueError("answerable outcomes need nonempty accepted references")
        answerable += 1
        if status != "answered":
            continue
        answered_answerable += 1
        answer = row.get("answer")
        normalized_answer = normalize_answer(answer) if isinstance(answer, str) else None
        correct = normalized_answer is not None and normalized_answer in {
            normalize_answer(reference) for reference in references
        }
        if correct:
            exact_correct += 1
            answered_correct += 1

    labels = {} if support_labels is None else support_labels
    if not isinstance(labels, Mapping):
        raise TypeError("support_labels must map answered outcome IDs to booleans")
    if any(identifier not in identifiers for identifier in labels):
        raise ValueError("support labels must refer to an outcome in this report")
    answered_ids = {row["id"] for row in rows if row["status"] == "answered"}
    if any(identifier not in answered_ids for identifier in labels):
        raise ValueError("support labels may only label answered outcomes")
    if any(type(label) is not bool for label in labels.values()):
        raise TypeError("support labels must be booleans")
    support_total = len(labels)
    supported = sum(labels.values())
    return {
        "total": len(rows),
        "answered": counts["answered"],
        "abstained": counts["abstained"],
        "error": counts["error"],
        "coverage": counts["answered"] / len(rows) if rows else None,
        "coverage_numerator": counts["answered"],
        "coverage_denominator": len(rows),
        "answerable_total": answerable,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / answerable if answerable else None,
        "exact_accuracy_denominator": answerable,
        "answered_answerable_total": answered_answerable,
        "answered_exact_correct": answered_correct,
        "answered_exact_accuracy": (
            answered_correct / answered_answerable if answered_answerable else None
        ),
        "answered_exact_accuracy_denominator": answered_answerable,
        "unanswerable_total": unanswerable,
        "unanswerable_abstained": unanswerable_statuses["abstained"],
        "unanswerable_error": unanswerable_statuses["error"],
        "unanswerable_answered": sum(
            row["answerability"] == "unanswerable" and row["status"] == "answered"
            for row in rows
        ),
        "support_label_count": support_total,
        "hallucination_rate": (support_total - supported) / support_total if support_total else None,
        "hallucination_rate_denominator": support_total,
        "visual_grounding_rate": supported / support_total if support_total else None,
        "visual_grounding_rate_denominator": support_total,
    }
