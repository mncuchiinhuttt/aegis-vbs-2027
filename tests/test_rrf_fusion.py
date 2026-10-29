"""Pure behavior tests for weighted reciprocal-rank fusion."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.rrf import merge_rrf


def _hit(doc_id, source):
    return {"id": doc_id, "payload": {"source_file": source}}


def test_default_weights_match_explicit_uniform_weights():
    dense = [_hit("a", "dense-a"), _hit("b", "dense-b")]
    lexical = [_hit("b", "lexical-b"), _hit("c", "lexical-c")]

    assert merge_rrf(dense, lexical) == merge_rrf(dense, lexical, weights=[1, 1])


def test_weights_change_ranking_without_changing_rank_fusion_inputs():
    dense = [_hit("a", "video-a"), _hit("b", "video-b")]
    lexical = [_hit("b", "video-b"), _hit("a", "video-a")]

    assert [hit["id"] for hit in merge_rrf(dense, lexical)] == ["a", "b"]
    assert [hit["id"] for hit in merge_rrf(dense, lexical, weights=[1, 3])] == ["b", "a"]


def test_zero_weight_omits_candidates_labels_and_payload_overwrite():
    enabled_payload = {"source_file": "enabled-source"}
    disabled_payload = {"source_file": "disabled-source"}
    result = merge_rrf(
        [_hit("shared", "enabled-source")],
        [
            {"id": "zero-only", "payload": {"source_file": "zero-only-source"}},
            {"id": "shared", "payload": disabled_payload},
        ],
        labels=["dense", "disabled"],
        weights=[1, 0],
    )

    assert [hit["id"] for hit in result] == ["shared"]
    assert result[0]["payload"]["source_file"] == "enabled-source"
    assert result[0]["matched_via"] == ["dense"]
    assert result[0]["rrf_score"] == pytest.approx(1 / 61)


def test_equal_scores_keep_first_seen_order_and_last_enabled_payload():
    first_payload = {"source_file": "first-source"}
    last_payload = {"source_file": "last-source"}
    result = merge_rrf(
        [{"id": "a", "payload": first_payload}, _hit("b", "b-first")],
        [{"id": "a", "payload": last_payload}, _hit("b", "b-last")],
        labels=["query", "hyde"],
    )

    assert [hit["id"] for hit in result] == ["a", "b"]
    assert result[0]["payload"] is last_payload
    assert result[0]["matched_via"] == ["query", "hyde"]


@pytest.mark.parametrize("weights", [[], [True], [-1], [float("nan")], [float("inf")], ["1"]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        merge_rrf([_hit("a", "video")], weights=weights)


@pytest.mark.parametrize("k", [0, -1, True, float("nan"), float("inf"), "60"])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError):
        merge_rrf([_hit("a", "video")], k=k)
