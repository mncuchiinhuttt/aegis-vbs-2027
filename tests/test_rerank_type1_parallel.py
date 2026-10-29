"""Type-1 rerank concurrency contract: same ranking as serial, bounded workers."""

import os
import re
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "inference-code"))

from search.reranker import Reranker  # noqa: E402

SCORES = {"c0": "0.35", "c1": "0.9", "c2": "0.1", "c3": "0.7", "c4": "no idea", "c5": "0.55", "c6": "0.2", "c7": "0.8"}


class ScoreVLM:
    """Replies with a score keyed by the candidate caption found in the prompt."""

    def __init__(self, parallel: bool):
        self.supports_parallel_generate = parallel
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate(self, image, prompt):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.03)
        with self._lock:
            self._active -= 1
        return SCORES[re.search(r"Caption: (c\d)\.", prompt).group(1)]


def _candidates():
    return [{"id": name, "payload": {"caption": name, "source_file": f"{name}.mp4"}} for name in SCORES]


def _rank(vlm, workers=4):
    ranked = Reranker(vlm, vqa_max_workers=workers).rerank_type1("a query", _candidates())
    return [(hit["id"], hit["rerank_score"], hit["rerank_score_valid"]) for hit in ranked]


def test_parallel_rerank_matches_serial_and_respects_worker_bound():
    serial_vlm, parallel_vlm = ScoreVLM(parallel=False), ScoreVLM(parallel=True)

    serial, parallel = _rank(serial_vlm), _rank(parallel_vlm)

    assert parallel == serial
    assert [item[0] for item in parallel][:3] == ["c1", "c7", "c3"]
    assert serial_vlm.max_active == 1
    assert 1 < parallel_vlm.max_active <= 4


def test_unparsable_score_is_zero_and_marked_invalid_in_parallel_path():
    ranked = dict((item[0], item[1:]) for item in _rank(ScoreVLM(parallel=True)))

    assert ranked["c4"] == (0.0, False)
    assert ranked["c1"] == (0.9, True)
