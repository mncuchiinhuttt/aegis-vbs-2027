"""VQA evidence resolution must fall back to video decoding for non-image media."""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "inference-code"))

import search.reranker as reranker_module  # noqa: E402


class FakeCapture:
    def __init__(self, path):
        self.positions = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.positions.append((prop, value))

    def read(self):
        return True, np.zeros((6, 8, 3), dtype=np.uint8)

    def get(self, prop):
        return 1250.0

    def release(self):
        pass


@pytest.mark.parametrize("error", [ModuleNotFoundError("pi_heif"), OSError("cannot identify")])
def test_video_evidence_is_decoded_when_image_open_fails(tmp_path, monkeypatch, error):
    (tmp_path / "clip.mp4").write_bytes(b"not an image")

    def failing_open(path, *args, **kwargs):
        raise error

    monkeypatch.setattr(reranker_module.Image, "open", failing_open)
    monkeypatch.setattr(reranker_module.cv2, "VideoCapture", FakeCapture)

    evidence = reranker_module.resolve_candidate_evidence(
        str(tmp_path), {"source_file": "clip.mp4", "frame_idx": 25, "timestamp": 1.0}
    )

    assert evidence is not None
    assert evidence["frame_idx"] == 25
    assert evidence["image"].size == (8, 6)
    assert evidence["path"] == str(tmp_path / "clip.mp4")
