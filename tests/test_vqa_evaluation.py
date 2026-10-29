"""CPU-safe tests for explicit VQA annotation, replay, and exact metrics."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from evaluation import replay_vqa
from evaluation.replay_schema import load_manifest, validate_base_url
from evaluation.vqa_metrics import compute_vqa_metrics, normalize_answer


def _outcome(identifier, status="answered", answerability="answerable", answer="Plate AB-12", references=None):
    accepted = ["plate ab-12"] if references is None else references
    return {
        "id": identifier, "status": status, "answerability": answerability,
        "answer": answer, "accepted_references": accepted,
    }


def _manifest(queries=None):
    return {
        "schema_version": 1,
        "dataset_provenance": "declared source, not independently verified",
        "query_provenance": "declared annotation process, not independently verified",
        "queries": queries if queries is not None else [{
            "id": "q-1", "question": "Which plate is visible?",
            "answerability": "answerable", "accepted_references": ["AB-12"],
        }],
    }


def test_normalization_is_unicode_casefold_and_whitespace_only():
    assert normalize_answer("ＡＢＣ\tStraße\u00a0Road") == "abc strasse road"


def test_exact_match_does_not_credit_a_substring():
    metrics = compute_vqa_metrics([_outcome("q1", answer="plate ab-12 extra")])
    assert metrics["exact_correct"] == 0
    assert metrics["exact_accuracy"] == 0.0


def test_misses_abstentions_errors_and_unanswerable_are_denominated_explicitly():
    rows = [
        _outcome("correct"),
        _outcome("abstain", status="abstained"),
        _outcome("error", status="error"),
        _outcome("unanswerable", answerability="unanswerable", answer="UNKNOWN", references=[]),
        _outcome("refusal", status="abstained", answerability="unanswerable", answer=None, references=[]),
    ]
    metrics = compute_vqa_metrics(rows)
    assert (metrics["total"], metrics["answered"], metrics["abstained"], metrics["error"]) == (5, 2, 2, 1)
    assert metrics["coverage"] == 0.4
    assert metrics["exact_correct"] == 1
    assert metrics["exact_accuracy_denominator"] == 3
    assert metrics["exact_accuracy"] == pytest.approx(1 / 3)
    assert metrics["answered_answerable_total"] == 1
    assert metrics["answered_exact_accuracy_denominator"] == 1
    assert metrics["answered_exact_accuracy"] == 1.0
    assert metrics["unanswerable_total"] == 2
    assert metrics["unanswerable_answered"] == 1
    assert metrics["unanswerable_abstained"] == 1
    assert metrics["hallucination_rate"] is None
    assert metrics["visual_grounding_rate"] is None


def test_empty_denominators_are_none_and_manual_support_labels_are_explicit():
    assert compute_vqa_metrics([])["exact_accuracy"] is None
    assert compute_vqa_metrics([])["coverage"] is None
    assert compute_vqa_metrics([_outcome("q1", status="abstained")])["answered_exact_accuracy"] is None
    metrics = compute_vqa_metrics([_outcome("q1")], support_labels={"q1": False})
    assert metrics["hallucination_rate"] == 1.0
    assert metrics["visual_grounding_rate"] == 0.0
    assert metrics["support_label_count"] == 1


def test_manifest_rejects_duplicate_ids_and_missing_answerable_references(tmp_path):
    value = _manifest([
        {"id": "same", "question": "Q1", "answerability": "answerable", "accepted_references": ["A"]},
        {"id": "same", "question": "Q2", "answerability": "answerable", "accepted_references": ["B"]},
    ])
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_manifest(path)
    value["queries"] = [{"id": "q", "question": "Q", "answerability": "answerable", "accepted_references": []}]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="iff it is answerable"):
        load_manifest(path)


def test_validate_only_does_not_contact_a_local_backend(tmp_path, capsys):
    class ProbeHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.request_count += 1
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    manifest_path = tmp_path / "annotations.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
    server.request_count = 0
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    output_path = tmp_path / "report.json"
    try:
        assert replay_vqa.main([
            "--manifest", str(manifest_path), "--base-url", f"http://127.0.0.1:{server.server_port}",
            "--output", str(output_path), "--validate-only",
        ]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "valid"
        assert server.request_count == 0
        assert not output_path.exists()
    finally:
        server.shutdown()
        worker.join()
        server.server_close()



def test_replay_rejects_embedded_base_url_credentials_or_query():
    with pytest.raises(ValueError):
        validate_base_url("https://user:secret@example.test")
    with pytest.raises(ValueError):
        validate_base_url("https://example.test?token=secret")
