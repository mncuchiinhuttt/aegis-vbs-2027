"""Capture observed server-host, package, source, and NVIDIA runtime metadata."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    "inference-code/config.py",
    "inference-code/search/hybrid_search.py",
    "inference-code/search/reranker.py",
    "models/base_vlm.py",
    "models/openai_vlm.py",
    "webapp/backend/main.py",
    "evaluation/run_eval.py",
    "evaluation/replay_vqa.py",
    "evaluation/replay_schema.py",
    "evaluation/vqa_metrics.py",
    "evaluation/capture_environment.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_runtime() -> dict:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi_not_found", "driver_version": None, "devices": []}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "unavailable", "reason": type(exc).__name__,
            "driver_version": None, "devices": [],
        }
    if result.returncode != 0:
        return {
            "status": "unavailable", "reason": f"nvidia-smi_exit_{result.returncode}",
            "driver_version": None, "devices": [],
        }
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 3:
            continue
        name, driver, memory = (value.strip() for value in row)
        try:
            memory_mib = int(memory)
        except ValueError:
            memory_mib = None
        devices.append({"name": name, "driver_version": driver or None, "memory_total_mib": memory_mib})
    if not devices:
        return {"status": "unavailable", "reason": "no_nvidia_gpu_reported", "driver_version": None, "devices": []}
    return {"status": "available", "reason": None, "driver_version": devices[0]["driver_version"], "devices": devices}


def _source_hashes(root: Path, sources: list[str]) -> dict:
    hashes = {}
    for relative in sources:
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source path escapes repository root: {relative}") from exc
        if source.is_file():
            hashes[relative] = {"status": "available", "sha256": _sha256(source)}
        else:
            hashes[relative] = {"status": "unavailable", "sha256": None, "reason": "source_file_not_found"}
    return hashes


def capture_environment(root: Path = DEFAULT_ROOT, extra_sources: list[str] | None = None) -> dict:
    root = root.resolve()
    sources = list(DEFAULT_SOURCES) + list(extra_sources or [])
    if len(set(sources)) != len(sources):
        raise ValueError("source paths must be unique")
    try:
        packages = sorted(
            ({"name": dist.metadata.get("Name") or "", "version": dist.version}
             for dist in importlib.metadata.distributions()),
            key=lambda item: (item["name"].casefold(), item["version"]),
        )
        package_status = "available"
    except (OSError, importlib.metadata.PackageNotFoundError) as exc:
        packages = []
        package_status = f"unavailable:{type(exc).__name__}"
    return {
        "schema_version": 1,
        "capture_kind": "server_environment",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(),
        },
        "installed_packages_status": package_status,
        "installed_packages": packages,
        "source_file_hashes": _source_hashes(root, sources),
        "gpu": _gpu_runtime(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="User-selected output JSON path")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Repository root (default: this checkout)")
    parser.add_argument("--source", action="append", default=[], help="Additional source path relative to --root")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error("--root must name an existing directory")
    try:
        report = capture_environment(root, args.source)
        output = Path(args.output).expanduser().resolve()
        if output in {(root / relative).resolve() for relative in report["source_file_hashes"]}:
            raise ValueError("output path must not overwrite a hashed source file")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved observed environment capture to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
