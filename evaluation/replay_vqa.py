"""Model-free HTTP replay of annotated Type 2 VQA questions."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from .replay_schema import load_environment, load_manifest, validate_base_url
    from .vqa_metrics import compute_vqa_metrics
except ImportError:  # Direct execution: python evaluation/replay_vqa.py
    from replay_schema import load_environment, load_manifest, validate_base_url
    from vqa_metrics import compute_vqa_metrics

MAX_RESPONSE_BYTES = 10 * 1024 * 1024

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST",
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from None
    except URLError as exc:
        raise RuntimeError(f"backend request failed ({type(exc.reason).__name__})") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("backend response exceeded the 10 MiB limit")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("backend returned invalid JSON") from None
    if not isinstance(body, dict):
        raise RuntimeError("backend JSON response must be an object")
    return body


def _outcome(query: dict, response: dict) -> dict:
    row = {
        "id": query["id"], "question": query["question"],
        "answerability": query["answerability"],
        "accepted_references": query["accepted_references"],
        "status": "error", "vqa_status": None, "answer": None,
        "vqa_answer_valid": None, "vqa_raw_response": None,
        "candidate": None, "answer_identity": None, "evidence": None,
        "runtime_metadata": response.get("runtime_metadata"), "error": None,
    }
    if response.get("type") != 2 or not isinstance(response.get("results"), list):
        row["error"] = "invalid Type 2 API response shape"
        return row
    results = response["results"]
    if not results:
        row.update(status="abstained", abstention_reason="no_candidate_result")
        return row
    result = results[0]
    if not isinstance(result, dict):
        row["error"] = "first API result is not an object"
        return row

    status = result.get("vqa_status")
    answer = result.get("vqa_answer", result.get("answer"))
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    row.update(
        vqa_status=status,
        answer=answer,
        vqa_answer_valid=result.get("vqa_answer_valid"),
        vqa_raw_response=result.get("vqa_raw_response"),
        candidate={
            "id": result.get("id"),
            "rank": result.get("rank"),
            "video_id": payload.get("source_file"),
            "score": result.get("score"),
            "vqa_score": result.get("vqa_score"),
            "rrf_score": result.get("rrf_score"),
        },
        answer_identity={
            "candidate_id": result.get("answer_candidate_id"),
            "video_id": result.get("answer_video_id"),
            "frame_idx": result.get("answer_frame_idx"),
        },
        evidence={
            "available": result.get("vqa_evidence_available"),
            "reason": result.get("vqa_evidence_reason"),
            "media_name": result.get("evidence_media_name"),
            "frame_idx": result.get("evidence_frame_idx"),
            "timestamp": result.get("evidence_timestamp"),
            "crop_bbox": result.get("vqa_crop_bbox"),
        },
    )
    if status == "error":
        row["error"] = "backend reported vqa_status=error"
    elif status == "abstained":
        row.update(status="abstained", abstention_reason=result.get("vqa_evidence_reason"))
    elif status == "answered":
        if not isinstance(answer, str) or not answer.strip():
            row.update(status="abstained", abstention_reason="missing_answer")
        else:
            row["status"] = "answered"
    elif status is None:
        row.update(status="abstained", abstention_reason="missing_vqa_status")
    else:
        row["error"] = "backend returned an invalid vqa_status"
    return row


def run_replay(
    manifest: dict, base_url: str, timeout: float,
    environment_capture: dict | None = None,
) -> dict:
    outcomes = []
    endpoint = f"{base_url}/api/search"
    for query in manifest["queries"]:
        started = time.monotonic()
        try:
            response = _post_json(endpoint, {"type": 2, "query": query["question"], "reset_session": True}, timeout)
            row = _outcome(query, response)
        except Exception as exc:
            row = {
                "id": query["id"], "question": query["question"],
                "answerability": query["answerability"],
                "accepted_references": query["accepted_references"],
                "status": "error", "vqa_status": None, "answer": None,
                "vqa_answer_valid": None, "vqa_raw_response": None,
                "candidate": None, "answer_identity": None, "evidence": None, "runtime_metadata": None,
                "error": str(exc) or type(exc).__name__,
            }
        row["latency_seconds"] = time.monotonic() - started
        outcomes.append(row)
    return {
        "schema_version": 1,
        "evaluation_kind": "production_http_vqa_replay",
        "manifest": {
            "path": manifest["manifest_path"], "sha256": manifest["manifest_sha256"],
            "dataset_provenance_declaration": manifest["dataset_provenance"],
            "query_provenance_declaration": manifest["query_provenance"],
            "provenance_status": "declared_not_independently_verified",
        },
        "backend": {"base_url": base_url, "endpoint": f"{base_url}/api/search"},
        "server_environment_capture": environment_capture,
        "summary": compute_vqa_metrics(outcomes),
        "queries": outcomes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to the explicit annotated VQA manifest")
    parser.add_argument("--base-url", required=True, help="Backend base URL (HTTPS; HTTP only on loopback)")
    parser.add_argument("--output", required=True, help="User-selected JSON report path")
    parser.add_argument("--timeout", type=float, default=120.0, help="One-request timeout in seconds (default: 120)")
    parser.add_argument("--environment", help="Optional server capture_environment.py JSON to pair with runtime metadata")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without contacting the backend")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        environment_capture = load_environment(args.environment) if args.environment else None
        base_url = validate_base_url(args.base_url)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        output_path = Path(args.output).expanduser().resolve()
        input_paths = {Path(manifest["manifest_path"])}
        if environment_capture:
            input_paths.add(Path(environment_capture["path"]))
        if output_path in input_paths:
            raise ValueError("output path must not overwrite the manifest or environment capture")
        if output_path.exists() and output_path.is_dir():
            raise ValueError("output path must be a file path")
        if not args.validate_only:
            output_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "query_count": len(manifest["queries"]),
            "manifest_sha256": manifest["manifest_sha256"],
            "provenance_status": "declared_not_independently_verified",
        }, ensure_ascii=False))
        return 0
    report = run_replay(
        manifest, base_url, args.timeout, environment_capture=environment_capture,
    )
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved replay report to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
