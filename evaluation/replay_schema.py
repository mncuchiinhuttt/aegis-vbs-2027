"""Strict, model-free validation for VQA replay inputs and environment captures."""

import hashlib
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit


def validate_base_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("base URL must be an absolute HTTP(S) URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("base URL has an invalid port") from exc
    try:
        loopback = parts.hostname.lower() == "localhost" or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        loopback = False
    if parts.scheme == "http" and not loopback:
        raise ValueError("HTTP is allowed only for localhost/loopback; use HTTPS otherwise")
    return value.rstrip("/")


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest must be valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "dataset_provenance", "query_provenance", "queries"
    }:
        raise ValueError("manifest must contain exactly schema_version, dataset_provenance, query_provenance, queries")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    for field in ("dataset_provenance", "query_provenance"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ValueError(f"{field} must be a nonempty free-form declaration")
    queries = data["queries"]
    if not isinstance(queries, list):
        raise ValueError("queries must be a list")
    identifiers = set()
    for index, query in enumerate(queries):
        prefix = f"queries[{index}]"
        if not isinstance(query, dict) or set(query) != {
            "id", "question", "answerability", "accepted_references"
        }:
            raise ValueError(f"{prefix} must contain exactly id, question, answerability, accepted_references")
        identifier, question = query["id"], query["question"]
        if (
            not isinstance(identifier, str) or not identifier.strip()
            or identifier != identifier.strip() or identifier in identifiers
        ):
            raise ValueError(f"{prefix}.id must be a unique nonempty string without outer whitespace")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{prefix}.question must be a nonempty string")
        identifiers.add(identifier)
        answerability, references = query["answerability"], query["accepted_references"]
        if not isinstance(answerability, str) or answerability not in {"answerable", "unanswerable"}:
            raise ValueError(f"{prefix}.answerability must be answerable or unanswerable")
        if not isinstance(references, list) or any(not isinstance(ref, str) or not ref.strip() for ref in references):
            raise ValueError(f"{prefix}.accepted_references must be a list of nonempty strings")
        if (answerability == "answerable") != bool(references):
            raise ValueError(f"{prefix} needs references iff it is answerable")
    data["manifest_path"] = str(manifest_path)
    data["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def load_environment(path: str | Path) -> dict:
    environment_path = Path(path).expanduser().resolve()
    raw = environment_path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"environment capture must be valid UTF-8 JSON: {exc}") from exc
    required = {
        "schema_version", "capture_kind", "captured_at_utc", "python", "platform",
        "installed_packages_status", "installed_packages", "source_file_hashes", "gpu",
    }
    if (
        not isinstance(data, dict) or set(data) != required
        or type(data.get("schema_version")) is not int or data["schema_version"] != 1
        or data.get("capture_kind") != "server_environment"
        or not isinstance(data.get("python"), dict) or not isinstance(data.get("platform"), dict)
        or not isinstance(data.get("installed_packages"), list)
        or not isinstance(data.get("source_file_hashes"), dict) or not isinstance(data.get("gpu"), dict)
    ):
        raise ValueError("environment file does not match capture_environment.py schema version 1")
    return {"path": str(environment_path), "sha256": hashlib.sha256(raw).hexdigest(), "record": data}
