"""Bounded VQA concurrency, deterministic ordering and error-state contracts."""

import os
import sys
import threading
import time

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.reranker import Reranker  # noqa: E402
import search.reranker as reranker_module  # noqa: E402
from vqa_fixture_helpers import FakeVLM, _candidate  # noqa: E402


class _ParallelFixtureVLM:
    supports_parallel_generate = True

    def __init__(self):
        self._lock = threading.Lock()
        self._first_pair = threading.Barrier(2)
        self._active = 0
        self.max_active = 0
        self.calls = []

    def generate(self, image, prompt):
        color = image.getpixel((0, 0))
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append(color)
        try:
            if color in ((255, 0, 0), (0, 0, 255)):
                self._first_pair.wait(timeout=3)
            if color == (255, 0, 0):
                time.sleep(0.03)
                return '{"found": true, "answer": "red object", "confidence": 0.5}'
            if color == (0, 0, 255):
                raise RuntimeError("private API credential must not be exposed")
            return '{"found": true, "answer": "green object", "confidence": 0.5}'
        finally:
            with self._lock:
                self._active -= 1


class _SerialFixtureVLM:
    supports_parallel_generate = False

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate(self, image, prompt):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.01)
            return '{"found": true, "answer": "object", "confidence": 0.5}'
        finally:
            with self._lock:
                self._active -= 1


def test_vqa_parallel_results_keep_identity_status_and_stable_ties(tmp_path):
    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    green_path = tmp_path / "green.png"
    Image.new("RGB", (20, 20), color="red").save(red_path)
    Image.new("RGB", (20, 20), color="blue").save(blue_path)
    Image.new("RGB", (20, 20), color="green").save(green_path)

    candidates = [
        _candidate("red", "red.mp4", keyframe_path=str(red_path)),
        _candidate("missing", "missing.mp4", keyframe_path=str(tmp_path / "missing.png")),
        _candidate("error", "blue.mp4", keyframe_path=str(blue_path)),
        _candidate("green", "green.mp4", keyframe_path=str(green_path)),
    ]

    class Detector:
        def detect(self, image, sub_queries):
            return [{"conf": 0.8, "bbox": [-5, 1, 15, 18]}]

    vlm = _ParallelFixtureVLM()
    reranker = Reranker(vlm, Detector(), vqa_max_workers=2)
    result = reranker.rerank_type2_vqa(
        "What is visible?", ["object"], candidates, str(tmp_path), verify=False
    )

    assert reranker.last_vqa_workers == 2
    assert vlm.max_active == 2
    assert len(vlm.calls) == 3
    assert [hit["id"] for hit in result] == ["red", "green", "missing", "error"]

    by_id = {hit["id"]: hit for hit in result}
    assert by_id["red"]["vqa_answer"] == "red object"
    assert by_id["red"]["vqa_candidate_id"] == "red"
    assert by_id["red"]["vqa_evidence_path"] == str(red_path)
    assert by_id["red"]["vqa_status"] == "answered"
    assert by_id["red"]["vqa_crop_bbox"] == [0, 1, 15, 18]
    assert by_id["red"]["vqa_raw_response"] == (
        '{"found": true, "answer": "red object", "confidence": 0.5}'
    )

    assert by_id["missing"]["vqa_status"] == "abstained"
    assert by_id["missing"]["vqa_raw_response"] is None
    assert by_id["missing"]["vqa_evidence_reason"] == "frame_unavailable"
    assert by_id["error"]["vqa_status"] == "error"
    assert by_id["error"]["vqa_answer_valid"] is False
    assert by_id["error"]["vqa_raw_response"] is None
    assert by_id["error"]["vqa_evidence_reason"] == "vlm_error"
    assert "private API credential" not in str(by_id["error"])


def test_local_vlm_without_parallel_capability_stays_serial(tmp_path):
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        Image.new("RGB", (12, 12), color="white").save(path)
    vlm = _SerialFixtureVLM()
    reranker = Reranker(vlm, vqa_max_workers=8)
    candidates = [
        _candidate(str(index), f"{index}.mp4", keyframe_path=str(path))
        for index, path in enumerate(paths)
    ]

    result = reranker.rerank_type2_vqa(
        "What is visible?", [], candidates, str(tmp_path), verify=False
    )

    assert len(result) == 2
    assert reranker.last_vqa_workers == 1
    assert vlm.max_active == 1


def test_vqa_worker_cap_is_positive_and_empty_input_uses_zero_workers(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        Reranker(FakeVLM([]), vqa_max_workers=0)

    reranker = Reranker(FakeVLM([]), vqa_max_workers=2)
    assert reranker.rerank_type2_vqa(
        "What is visible?", [], [], str(tmp_path), verify=False
    ) == []
    assert reranker.last_vqa_workers == 0


def test_verification_error_fails_vqa_answer_closed_but_keeps_raw_response(tmp_path):
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), color="white").save(frame_path)
    raw_response = '{"found": true, "answer": "person", "confidence": 0.9}'

    class VerificationFailureReranker(Reranker):
        def generate_verification_questions(self, query):
            return ["Is a person visible?"]

        def verify_candidate(self, image, context_text, questions):
            raise RuntimeError("private API credential must not be exposed")

    reranker = VerificationFailureReranker(FakeVLM([raw_response]))
    candidate = _candidate("one", "frame.mp4", keyframe_path=str(frame_path))
    result = reranker.rerank_type2_vqa(
        "Is a person visible?", [], [candidate], str(tmp_path), verify=True
    )[0]

    assert result["vqa_status"] == "error"
    assert result["vqa_answer_valid"] is False
    assert result["vqa_answer"] == "UNKNOWN"
    assert result["vqa_score"] == 0.0
    assert result["vqa_raw_response"] == raw_response
    assert result["verification_ratio"] == 0.0
    assert result["final_score"] == pytest.approx(
        reranker_module.TYPE2_RRF_WEIGHT * result["rrf_score"]
    )
    assert "private API credential" not in str(result)
