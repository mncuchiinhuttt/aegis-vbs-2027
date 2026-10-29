"""Public HTTP VQA response provenance tests using only loopback HTTP."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from evaluation import replay_vqa
from evaluation.replay_schema import load_manifest


def _public_result(status, candidate_id, media_name, frame_idx, timestamp, reason, raw=None):
    return {
        "rank": 1, "id": candidate_id, "score": 0.05, "vqa_score": 0.0, "rrf_score": 0.01,
        "payload": {"source_file": f"{candidate_id}.mp4", "frame_idx": frame_idx, "timestamp": timestamp},
        "answer": "N/A" if status != "answered" else "AB-12",
        "vqa_answer": "UNKNOWN" if status != "answered" else "AB-12",
        "vqa_answer_valid": status == "answered", "vqa_status": status, "vqa_raw_response": raw,
        "vqa_crop_bbox": [5, 6, 70, 80] if media_name else None,
        "vqa_evidence_available": media_name is not None, "vqa_evidence_reason": reason,
        "answer_candidate_id": candidate_id if status == "answered" else None,
        "answer_video_id": f"{candidate_id}.mp4" if status == "answered" else None,
        "answer_frame_idx": frame_idx if status == "answered" else None,
        "evidence_media_name": media_name, "evidence_frame_idx": frame_idx,
        "evidence_timestamp": timestamp,
    }


def _manifest_file(path):
    path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_provenance": "test fixture declaration",
        "query_provenance": "test fixture declaration",
        "queries": [
            {"id": "q1", "question": "Which plate?", "answerability": "answerable", "accepted_references": ["AB-12"]},
            {"id": "q2", "question": "refusal", "answerability": "answerable", "accepted_references": ["an answer"]},
            {"id": "q3", "question": "model-error", "answerability": "answerable", "accepted_references": ["an answer"]},
            {"id": "q4", "question": "request-error", "answerability": "answerable", "accepted_references": ["an answer"]},
        ],
    }), encoding="utf-8")
    return load_manifest(path)


def test_replay_preserves_public_refusal_and_error_evidence_provenance(tmp_path):
    public_results = {
        "Which plate?": _public_result(
            "answered", "candidate-answered", "frames/answered.png", 100, 3.3,
            "grounded", '{"found":true,"answer":"AB-12"}',
        ),
        "refusal": _public_result(
            "abstained", "candidate-refusal", "frames/refusal.png", 200, 6.7,
            "answer_not_found", '{"found":false,"answer":"UNKNOWN"}',
        ),
        "model-error": _public_result(
            "error", "candidate-error", "frames/error.png", 300, 10.0, "vlm_error",
        ),
    }

    class ReplayHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append(payload)
            if payload["query"] == "request-error":
                self.send_response(503)
                body = b""
            else:
                results = [public_results[payload["query"]]]
                if payload["query"] == "Which plate?":
                    results.append({"rank": 2, "id": "lower-ranked", "vqa_answer": "not the answer"})
                body = json.dumps({
                    "type": 2, "runtime_metadata": {"worker_limit_used": 1}, "results": results,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    manifest = _manifest_file(tmp_path / "annotations.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
    server.requests = []
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        report = replay_vqa.run_replay(manifest, f"http://127.0.0.1:{server.server_port}", 10)
    finally:
        server.shutdown()
        worker.join()
        server.server_close()

    assert len(server.requests) == 4
    assert all(item["type"] == 2 and item["reset_session"] is True for item in server.requests)
    answered, refusal, model_error, request_error = report["queries"]
    assert answered["answer"] == "AB-12"
    assert answered["candidate"]["id"] == "candidate-answered"
    assert answered["candidate"]["score"] == 0.05
    assert answered["evidence"]["media_name"] == "frames/answered.png"
    assert answered["runtime_metadata"] == {"worker_limit_used": 1}
    assert refusal["status"] == "abstained"
    assert refusal["vqa_raw_response"] == '{"found":false,"answer":"UNKNOWN"}'
    assert refusal["answer_identity"] == {"candidate_id": None, "video_id": None, "frame_idx": None}
    assert refusal["candidate"]["id"] == "candidate-refusal"
    assert refusal["evidence"] == {
        "available": True, "reason": "answer_not_found", "media_name": "frames/refusal.png",
        "frame_idx": 200, "timestamp": 6.7, "crop_bbox": [5, 6, 70, 80],
    }
    assert model_error["status"] == "error"
    assert model_error["candidate"]["id"] == "candidate-error"
    assert model_error["answer_identity"] == {"candidate_id": None, "video_id": None, "frame_idx": None}
    assert model_error["evidence"]["media_name"] == "frames/error.png"
    assert model_error["evidence"]["frame_idx"] == 300
    assert model_error["evidence"]["timestamp"] == 10.0
    assert model_error["evidence"]["reason"] == "vlm_error"
    assert model_error["runtime_metadata"] == {"worker_limit_used": 1}
    assert request_error["status"] == "error" and request_error["evidence"] is None
    assert report["summary"]["error"] == 2
    assert report["summary"]["abstained"] == 1
    assert report["summary"]["exact_accuracy_denominator"] == 4


def test_replay_ignores_private_path_and_does_not_infer_missing_evidence():
    query = {"id": "q1", "question": "question", "answerability": "answerable", "accepted_references": ["answer"]}
    response = {"type": 2, "results": [{
        "id": "candidate", "payload": {"source_file": "video.mp4", "frame_idx": 777, "timestamp": 99.0},
        "vqa_status": "abstained", "vqa_answer": "UNKNOWN",
        "vqa_evidence_available": False, "vqa_evidence_reason": "frame_unavailable",
        "vqa_evidence_path": "/private/datasets/video/frame.png",
        "evidence_media_name": None, "evidence_frame_idx": None, "evidence_timestamp": None,
        "answer_candidate_id": None, "answer_video_id": None, "answer_frame_idx": None,
    }]}
    outcome = replay_vqa._outcome(query, response)
    serialized = json.dumps(outcome)
    assert outcome["evidence"]["media_name"] is None
    assert outcome["evidence"]["frame_idx"] is None
    assert outcome["evidence"]["timestamp"] is None
    assert outcome["candidate"]["video_id"] == "video.mp4"
    assert "frame_idx" not in outcome["candidate"]
    assert "/private/datasets" not in serialized
    assert "vqa_evidence_path" not in serialized
